#!/usr/bin/env python3
"""calibrate_count_max.py -- measure the ACTUAL accept_count distribution of the SMC
cold two-stage init across many independent prior draws, so count_max can be set from
data instead of a guess.

Why this exists: README.md/config_schema.py only had two anchor points -- a single
baseline warm-up convergence (2667 steps, job 64144) and a qualitative "typical ~5k"
claim -- neither is a real percentile over the actual prior. This script draws
``--n-draws`` samples from the SAME prior the production run uses (same seed
derivation as run_smc.py's calibrate()), runs the batched full-width cold two-stage
solve via ``pipeline.batch_eval_cold_l_diag`` (the diagnostic evaluator added
2026-07-07 for the count_max loud-failure check), and reports the empirical
accept_count distribution -- so you can pick a count_max that actually covers a
chosen fraction of the prior instead of guessing.

IMPORTANT: this probes with ``--count-max-probe`` (default 20000), NOT the config's
own (possibly much lower) count_max -- otherwise every slow draw would just get
truncated at the production cap and you'd never see how far past it they needed.
Draws that still hit the PROBE cap are reported as right-censored (>= probe cap) --
if too many are censored, rerun with a higher --count-max-probe.

Usage (mirrors run_smc.py's preset/override mechanism exactly)
----------------------------------------------------------------
    SMC_RETRIEVAL_PRESET=gpu \\
    SMC_RETRIEVAL_OVERRIDES_FILE=overrides/poc_fast.json \\
        python -m retrieval_framework.calibrate_count_max runs/w39b_smc_retrieval \\
            --n-draws 200 --count-max-probe 20000

Runs on the GH200 (real chemistry+RT build); not a local/CPU-friendly script.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from retrieval_framework.run_smc import make_config   # the exact preset/override logic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=".",
                     help="retrieval case directory containing case.py (default: cwd)")
    ap.add_argument("--n-draws", type=int, default=200,
                     help="prior draws to sample (chemistry is full-width batched, so "
                          "this is nearly free up to GPU width -- more draws = a "
                          "tighter percentile estimate, not much more wall time)")
    ap.add_argument("--count-max-probe", type=int, default=20000,
                     help="count_max used ONLY for this measurement (kept generous so "
                          "the real tail is visible instead of truncated at whatever "
                          "the production config's count_max is)")
    ap.add_argument("--seed-offset", type=int, default=0,
                     help="added to cfg.seed before splitting, so repeated calibration "
                          "runs sample different prior corners")
    ap.add_argument("--resolution", type=float, default=100.0,
                     help="native spectral resolution R for the calibration (default 100, "
                          "matching the data). accept_count is a property of the CHEMISTRY "
                          "convergence and is band/resolution-INDEPENDENT, so the RT is run "
                          "at low resolution purely to cheapen the opacity build + RT eval "
                          "and slash memory -- it does not change the measured step counts. "
                          "Pass <=0 to keep the preset's own nu_pts.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("calibrate_count_max")

    cfg, _preset = make_config(Path(args.run_dir))
    preset_count_max = cfg.count_max   # the cap the production run would use (None -> library default)
    cfg = replace(cfg, count_max=int(args.count_max_probe))

    # Force the calibration to a cheap native resolution (default R=100). accept_count
    # depends only on the chemistry (nz, molecules, priors), NOT on nu_pts, so running
    # the RT at R=100 instead of the production R~10000 measures the same step counts for
    # a fraction of the build time, RT cost, and memory. exojax's premodit grid is
    # log-uniform, so R = (nu_pts-1)/ln(nu_max/nu_min); invert for the target R.
    if args.resolution > 0:
        span = math.log(float(cfg.nu_max) / float(cfg.nu_min))
        nu_pts_R = max(4, int(round(args.resolution * span)) + 1)
        if nu_pts_R < int(cfg.nu_pts):
            log.info(f"calibration resolution: R~{args.resolution:.0f} -> nu_pts "
                     f"{cfg.nu_pts} -> {nu_pts_R} (accept_count is nu_pts-independent; "
                     "this only cheapens the RT/build + memory, not the chemistry).")
            cfg = replace(cfg, nu_pts=nu_pts_R)
        else:
            log.info(f"calibration resolution R~{args.resolution:.0f} would need "
                     f"nu_pts={nu_pts_R} >= preset {cfg.nu_pts}; keeping preset nu_pts.")

    # loud dump of the RESOLVED calibration config (shows the R=100 nu_pts + probe cap)
    from retrieval_framework import config_schema as C
    log.info(C.describe_config(cfg, f"{_preset}+CALIBRATE"))
    log.info(f"calibration: nz={cfg.nz} nu_pts={cfg.nu_pts} art_nlayer={cfg.art_nlayer} "
             f"count_max(probe)={cfg.count_max} n_draws={args.n_draws}")

    from retrieval_framework import pipeline as P
    import jax

    t0 = time.perf_counter()
    pipe = P.build_pipeline(cfg)
    log.info(f"Built pipeline in {time.perf_counter() - t0:.1f}s | n_dim={pipe.n_dim}")

    # Observations are required before any jitted likelihood call (L is a byproduct
    # here, not the point, but batch_eval_cold_l_diag computes it regardless).
    if cfg.generate_synthetic_data:
        P.generate_observations(pipe, seed=int(cfg.seed))
    else:
        P.load_real_into_pipe(pipe)

    key = jax.random.PRNGKey(int(cfg.seed) + int(args.seed_offset))
    key, sub = jax.random.split(key)
    U = pipe.sample_prior_u(sub, int(args.n_draws))
    Y0, refs0 = P._blank_state(pipe, int(args.n_draws))

    log.info("Running batched cold two-stage init at the probe count_max "
             "(single lockstep while_loop bounded by the SLOWEST draw -- this can "
             "legitimately take a while if the probe cap is high and a corner is hard)...")
    t0 = time.perf_counter()
    fn = jax.jit(pipe.batch_eval_cold_l_diag)
    L, Y, refs, worst_accept = fn(U, Y0, refs0)
    jax.block_until_ready(L)
    dt = time.perf_counter() - t0
    log.info(f"done in {dt:.1f}s ({dt / max(1, int(args.n_draws)):.3f}s/draw amortized; "
             "NOT per-draw cost -- wall time is set by the single slowest draw)")

    wa = np.asarray(jax.device_get(worst_accept), np.int64)
    censored = wa >= int(args.count_max_probe)
    n_censored = int(censored.sum())
    if n_censored:
        log.warning(f"{n_censored}/{len(wa)} draw(s) still hadn't converged at the probe "
                     f"cap ({args.count_max_probe}) -- their true step count is unknown "
                     "(right-censored); rerun with a higher --count-max-probe for a clean "
                     "read of the tail. Percentiles below TREAT them as exactly the probe "
                     "cap, which UNDERSTATES the true value at high percentiles.")

    qs = [50, 75, 90, 95, 99, 100]
    pct = {q: int(np.percentile(wa, q)) for q in qs}
    log.info("=== accept_count percentiles (cold two-stage init, this prior/config) ===")
    for q in qs:
        log.info(f"  p{q:<3d} = {pct[q]}")
    log.info(f"  mean = {float(wa.mean()):.0f}  max = {int(wa.max())}  "
             f"min = {int(wa.min())}  n_censored = {n_censored}/{len(wa)}")

    for target in (0.75, 0.90, 0.95):
        need = int(np.percentile(wa, target * 100))
        if need >= int(args.count_max_probe):
            log.info(f"  p{target * 100:.0f} is censored at the probe cap -- no count_max "
                     "recommendation at this coverage from this sample; rerun with a "
                     "higher --count-max-probe")
        else:
            log.info(f"  count_max={need:>6d} would cover ~{target:.0%} of this sample "
                     f"(round up for margin; this sample has n={len(wa)} draws, so treat "
                     "single-draw percentile estimates as noisy)")

    # What the production cold init would do at candidate caps. _init_state now REJECTS
    # non-converged draws and OVERSAMPLES: it draws ceil(N*init_oversample) and keeps N
    # healthy survivors, raising ONLY when the reject fraction leaves < N survivors, i.e.
    # reject frac > 1 - 1/init_oversample. init_max_nonconverged_frac is a WARNING
    # threshold. A draw counts as non-converged at cap c when accept_count >= c (same
    # convention as the censoring check above and _init_state's exhausted test).
    warn = float(cfg.init_max_nonconverged_frac)
    over = float(cfg.init_oversample)
    fail_frac = 1.0 - 1.0 / over    # reject frac above which oversampling can't fill N
    log.info(f"=== production cold-init gate (reject+oversample: init_oversample={over:g} "
             f"tolerates reject frac up to {fail_frac:.0%}, warns above {warn:.0%}; "
             f"this preset's count_max={preset_count_max}) ===")
    cands = sorted({int(c) for c in (preset_count_max, 5000, 10000,
                                     int(args.count_max_probe)) if c})
    for cand in cands:
        if cand > int(args.count_max_probe):
            log.info(f"  at count_max={cand:>6d}: unknown (above the probe cap)")
            continue
        frac = float(np.mean(wa >= cand))
        if frac > fail_frac:
            verdict = f"RAISE -- oversample x{over:g} cannot fill N (need reject <= {fail_frac:.0%})"
        elif frac > warn:
            verdict = "reject+cull OK, but WARN (prior hits many corners)"
        else:
            verdict = "reject+cull OK"
        tag = "   <- this preset" if preset_count_max and cand == int(preset_count_max) else ""
        log.info(f"  at count_max={cand:>6d}: {frac:>5.1%} non-converged -> {verdict}{tag}")

    # Map the slow draws to prior corners: without the parameter values a censored
    # draw is unactionable (can't tell "tighten the prior" from "raise count_max").
    Theta = np.asarray(jax.device_get(jax.vmap(pipe.theta_from_u)(U)), np.float64)
    names = list(pipe.names)
    if n_censored:
        log.info("=== parameters of the censored draws (the hard prior corners) ===")
        for i in np.flatnonzero(censored):
            pstr = ", ".join(f"{n}={Theta[i, j]:+.4g}" for j, n in enumerate(names))
            log.info(f"  draw {int(i):>3d}: {pstr}")

    out = {
        "n_draws": int(args.n_draws), "count_max_probe": int(args.count_max_probe),
        "seed_offset": int(args.seed_offset),
        "preset_count_max": None if preset_count_max is None else int(preset_count_max),
        "init_max_nonconverged_frac": warn, "init_oversample": over,
        "n_censored": n_censored, "accept_count": wa.tolist(), "percentiles": pct,
        "param_names": names, "theta": Theta.tolist(),
    }
    suffix = "" if int(args.seed_offset) == 0 else f"_seed{int(args.seed_offset)}"
    out_path = cfg.out_dir / f"count_max_calibration{suffix}.json"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
