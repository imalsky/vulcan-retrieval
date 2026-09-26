#!/usr/bin/env python3
"""calibrate_count_max.py -- measure the ACTUAL accept_count distribution of the SMC
cold two-stage init across many independent prior draws, so count_max can be set from
data instead of a guess.

Why this exists: a single baseline warm-up convergence (2667 steps) and
a qualitative "typical ~5k" claim are not a percentile over the actual prior. This
script draws ``--n-draws`` samples from the SAME prior the production run uses (same
seed derivation as run_smc.py's calibrate()), runs the batched full-width cold
two-stage solve via ``pipeline.batch_eval_cold_l_diag`` (the diagnostic evaluator
for the count_max loud-failure check), and reports the empirical
accept_count distribution -- so you can pick a count_max that actually covers a
chosen fraction of the prior instead of guessing.

IMPORTANT: this probes with ``--count-max-probe`` (default 20000), NOT the config's
own (possibly much lower) count_max -- otherwise every slow draw would just get
truncated at the production cap and you'd never see how far past it they needed.
Draws that still hit the PROBE cap are reported as right-censored (>= probe cap) --
if too many are censored, rerun with a higher --count-max-probe.

``--fixed-steps K`` instead turns this into a step-cost BENCHMARK: count_min =
count_max = K pins every lane at exactly K accepted steps per stage (two stages, no
lane certifies), so the reported ms/step times the batched cold chemistry step at the
production shape with convergence taken out of the measurement. ``--grad`` benchmarks
the production cold gradient evaluator ``batch_eval_cold_vg`` instead of the primal
``batch_eval_cold_l_diag``.

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
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from retrieval_framework.run_smc import (   # the exact preset/override logic
    _cuda_profiler, make_config, set_observations)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=".",
                     help="retrieval case directory containing case.py (default: cwd)")
    ap.add_argument("--n-draws", type=int, default=None,
                     help="prior draws to sample (chemistry is full-width batched, so "
                          "this is nearly free up to GPU width -- more draws = a "
                          "tighter percentile estimate, not much more wall time); "
                          "default 200, or the preset's smc_num_particles under "
                          "--fixed-steps (the production width)")
    ap.add_argument("--count-max-probe", type=int, default=20000,
                     help="count_max used ONLY for this measurement (kept generous so "
                          "the real tail is visible instead of truncated at whatever "
                          "the production config's count_max is)")
    ap.add_argument("--seed-offset", type=int, default=0,
                     help="added to cfg.seed before splitting, so repeated calibration "
                          "runs sample different prior corners")
    ap.add_argument("--fixed-steps", type=int, default=0,
                     help="run every lane for exactly this many accepted steps per "
                          "stage, count_min = count_max, no lane certifies; a step-cost "
                          "benchmark, not a calibration")
    ap.add_argument("--lanes", type=int, default=None,
                     help="lanes the cold chemistry batch runs on (cfg.cold_lanes); "
                          "0 = every draw in one lockstep batch, k > 0 = k lanes "
                          "refilled from the draw queue; k > 0 is refused with "
                          "--fixed-steps, which runs lockstep (0) unless told "
                          "otherwise. Default: the config's value")
    ap.add_argument("--grad", action="store_true",
                     help="benchmark the production cold GRADIENT evaluator "
                          "batch_eval_cold_vg instead of the primal "
                          "batch_eval_cold_l_diag; requires --fixed-steps")
    args = ap.parse_args()
    if args.grad and int(args.fixed_steps) <= 0:
        ap.error("--grad requires --fixed-steps")

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("calibrate_count_max")

    cfg, _preset = make_config(Path(args.run_dir))
    preset_count_max = cfg.count_max   # the cap the production run would use (None -> library default)
    K = int(args.fixed_steps)
    if args.n_draws is None:
        args.n_draws = int(cfg.smc_num_particles) if K > 0 else 200
    # count_min = count_max = K: the runner may only certify above count_min, so every
    # lane runs exactly K accepted steps per stage and none certifies -- convergence is
    # out of the timing. warm_count_max comes along only because validate_config
    # refuses warm_count_max > count_max; the bench runs cold evaluators, which are
    # never warm-capped.
    cfg = replace(cfg, count_min=K, count_max=K, warm_count_max=K) if K > 0 else replace(
        cfg, count_max=int(args.count_max_probe))
    # The lane-count arm of the bench: how much of the wall time is the slowest
    # draw holding a full-width lockstep batch open.
    if args.lanes is not None:
        cfg = replace(cfg, cold_lanes=int(args.lanes))
    elif K > 0:
        # The fixed-step bench times one accepted step of a lockstep batch; the
        # config's lane queue (on by default) would refill every capped lane and
        # time queue throughput instead, so the bench runs lockstep unless
        # --lanes asks for something else (refused just below).
        cfg = replace(cfg, cold_lanes=0)
    if K > 0 and int(cfg.cold_lanes) > 0:
        raise SystemExit(
            f"--fixed-steps cannot run with cold_lanes={int(cfg.cold_lanes)}: a "
            "capped lane is is_done and gets refilled, so the bench would measure "
            "queue throughput, not the cost of one accepted step. Bench the step "
            "cost with --lanes 0.")

    # accept_count depends only on the chemistry (nz, molecules, priors), not on
    # the RT; the correlated-k band grid is fixed by the tables, so the RT runs at
    # the production band as-is.
    # loud dump of the RESOLVED calibration config (shows the probe cap)
    from retrieval_framework import config_schema as C
    log.info(C.describe_config(cfg, f"{_preset}+CALIBRATE"))
    log.info(f"calibration: nz={cfg.nz} art_nlayer={cfg.art_nlayer} "
             f"count_max(probe)={cfg.count_max} n_draws={args.n_draws} "
             f"cold_lanes={cfg.cold_lanes} (0 = one lockstep batch)")

    from retrieval_framework import pipeline as P
    import jax

    def _setup(cfg_):
        t0 = time.perf_counter()
        pipe = P.build_pipeline(cfg_)
        log.info(f"Built pipeline in {time.perf_counter() - t0:.1f}s | n_dim={pipe.n_dim}")
        # Observations are required before any jitted likelihood call (L is a
        # byproduct here, not the point, but batch_eval_cold_l_diag computes it
        # regardless).
        set_observations(cfg_, pipe, P)
        key = jax.random.PRNGKey(int(cfg_.seed) + int(args.seed_offset))
        key, sub = jax.random.split(key)
        U = pipe.sample_prior_u(sub, int(args.n_draws))
        Y0, refs0 = P._blank_state(pipe, int(args.n_draws))
        return pipe, U, Y0, refs0

    def _timed(cfg_, capture):
        """Compile pass, then one timed pass (inside the nsys capture range if
        `capture`). Returns (t_first, t_steady, out, U)."""
        pipe, U, Y0, refs0 = _setup(cfg_)
        fn = jax.jit(pipe.batch_eval_cold_vg if args.grad else pipe.batch_eval_cold_l_diag)
        t0 = time.perf_counter()
        out = fn(U, Y0, refs0)
        jax.block_until_ready(out[0])
        t_first = time.perf_counter() - t0
        if capture:
            _cuda_profiler(True)     # no-op unless NSYS_CAPTURE_API=1
        t0 = time.perf_counter()
        out = fn(U, Y0, refs0)
        jax.block_until_ready(out[0])
        t_steady = time.perf_counter() - t0
        if capture:
            _cuda_profiler(False)
        return t_first, t_steady, out, U

    if K > 0:
        # Baseline at K=1 (the floor validate_config allows): the same seeds, RT and
        # likelihood with one accepted step per stage. Its wall is everything that
        # is NOT the solver loop (the equilibrium seed and the RT above all), so the
        # loop's cost is the difference between the two timed passes.
        _t1_first, t1_steady, _o1, _u1 = _timed(
            replace(cfg, count_min=1, count_max=1, warm_count_max=1), capture=False)
        t_first, t_steady, out, U = _timed(cfg, capture=True)
        # The runner's exit test is accept_count > count_max (VULCAN-JAX
        # outer_loop._real_terminate), so a capped lane stops after exactly K+1
        # accepted steps.
        # A cold solve runs one such loop per stage, two stages.
        n_acc = K + 1
        n_stages = 2
        n_step = n_stages * n_acc
        n_extra = n_stages * (K - 1)   # accepted steps this bench adds over the K=1 baseline
        # Timing first: it is the point of the job and must land in the log even if
        # the output handling below trips.
        log.info(f"fixed-step bench ({'GRADIENT batch_eval_cold_vg' if args.grad else 'primal batch_eval_cold_l_diag'}): "
                 f"lanes={int(args.n_draws)} K={K} steps_per_lane={n_step} "
                 f"({n_stages} stage(s) x K+1)")
        log.info(f"  t_first  = {t_first:.3f} s (compile + run)")
        log.info(f"  t_steady = {t_steady:.3f} s  (K={K}, inside the nsys capture range)")
        log.info(f"  t_steady = {t1_steady:.3f} s  (K=1 baseline: seeds + RT + likelihood + "
                 f"{n_stages * 2} accepted steps, {n_stages} stage(s) x 2)")
        loop = t_steady - t1_steady
        log.info(f"  loop     = {loop:.3f} s for the extra {n_stages}(K-1)={n_extra} accepted steps "
                 f"of the slowest lane; {1000.0 * loop / max(1, n_extra):.3f} ms per accepted "
                 "step. Per loop ITERATION (accepted + rejected) divide `loop` by the "
                 "factorisation count in the capture: the LU path runs one getrf_panel "
                 "per layer, so nsys instances / nz, while the FFI block-Thomas kernel "
                 "traverses every layer AND every lane in ONE launch, so its "
                 "bt_factor instances ARE the iterations.")

        Lb = np.asarray(jax.device_get(out[0]), np.float64)
        Yb = np.asarray(jax.device_get(out[2] if args.grad else out[1]), np.float64)
        log.info(f"  n_finite(L) = {int(np.sum(np.isfinite(Lb)))}/{Lb.size}  "
                 f"max|Y| = {np.nanmax(np.abs(Yb)):.6g}")
        save = {"U": np.asarray(jax.device_get(U), np.float64), "Y": Yb, "L": Lb,
                "t_steady": t_steady, "t1_steady": t1_steady, "K": K,
                # the lane knobs in effect (a fixed-step bench is refused with
                # lanes, so these record the lockstep batch)
                "cold_lanes": int(cfg.cold_lanes),
                "cold_refill_chunk": int(cfg.cold_refill_chunk)}
        if args.grad:
            Gb = np.asarray(jax.device_get(out[1]), np.float64)
            save["G"] = Gb
            log.info(f"  n_bad_grad = {int(jax.device_get(out[4]))}  "
                     f"finite(G) = {float(np.mean(np.isfinite(Gb))):.3f}")
        else:
            cd = out[3]
            wa = np.asarray(jax.device_get(cd.accept_count), np.int64)
            save["accept_count"] = wa
            vals, cnts = np.unique(wa, return_counts=True)
            log.info("  accept_count histogram (value: lanes) = "
                     + ", ".join(f"{int(v)}: {int(c)}" for v, c in zip(vals, cnts))
                     + f"; conv_normal lanes = {int(np.sum(np.asarray(jax.device_get(cd.conv_normal))))}")
            if int(wa.min()) != n_acc or int(wa.max()) != n_acc:
                log.warning(f"accept_count is not exactly K+1={n_acc} on every lane -- "
                            "lanes below it stopped early (reason 2 runtime / 5 non-finite); "
                            "the slowest lane still sets the wall")
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        bench_path = cfg.out_dir / f"bench_fixed_steps{'_grad' if args.grad else ''}.npz"
        np.savez(bench_path, **save)
        log.info(f"wrote {bench_path}")
        return

    pipe, U, Y0, refs0 = _setup(cfg)
    log.info("Running batched cold two-stage init at the probe count_max "
             "(on the lane queue with cold_lanes > 0, else one lockstep while_loop "
             "bounded by the SLOWEST draw -- this can legitimately take a while if "
             "the probe cap is high and a corner is hard)...")
    t0 = time.perf_counter()
    fn = jax.jit(pipe.batch_eval_cold_l_diag)
    L, _Y, _refs, cd = fn(U, Y0, refs0)
    jax.block_until_ready(L)
    dt = time.perf_counter() - t0
    log.info(f"done in {dt:.1f}s ({dt / max(1, int(args.n_draws)):.3f}s/draw amortized; "
             "NOT per-draw cost -- the wall is total work / lanes on the queue, "
             "the slowest draw in one lockstep batch)")

    wa = np.asarray(jax.device_get(cd.accept_count), np.int64)
    Lnp = np.asarray(jax.device_get(L), np.float64)

    # free per-draw convergence-quality read from the same solve: longdy
    # percentiles + the stall-certified count (the class the SMC gates reject)
    longdy = np.asarray(jax.device_get(cd.longdy), np.float64)
    conv_ok = np.asarray(jax.device_get(cd.conv_normal), bool)
    longdy_pct = np.percentile(longdy[np.isfinite(longdy)], [50, 90, 99]) if np.any(
        np.isfinite(longdy)) else [float("nan")] * 3
    log.info(f"exit longdy percentiles p50/p90/p99 = {longdy_pct[0]:.3g}/{longdy_pct[1]:.3g}/"
             f"{longdy_pct[2]:.3g}; stall-certified (not canonically certified) draws: "
             f"{int(np.sum(~conv_ok))}/{len(conv_ok)}")

    # Exit element-budget drift (C23) per draw, with the exit model time t beside
    # it ON PURPOSE: the molecular-diffusion boundary rows leak the column at a
    # fixed rate (VULCAN-JAX notes P9, ~8e-18 /s for S), so a drift that grows
    # linearly with t and only reaches the tolerance near t ~ 1e15 s is that known
    # term, not geometry. Vetoes at long t with a constant drift/t = the boundary
    # rows; a short-t, element-specific drift = geometry or a real leak.
    drift = np.asarray(jax.device_get(cd.budget_drift_max), np.float64)
    atom = np.asarray(jax.device_get(cd.budget_drift_atom), np.int64)
    t_exit = np.asarray(jax.device_get(cd.t), np.float64)
    atom_names = pipe.fwd.chem.atom_order
    log.info(f"exit element-budget drift |X/H - 1| p50/p90/max = "
             f"{np.nanpercentile(drift, 50):.3g}/{np.nanpercentile(drift, 90):.3g}/"
             f"{np.nanmax(drift):.3g}")
    for i in np.flatnonzero(~conv_ok):
        log.info(f"  draw {i}: uncertified, accept_count={int(wa[i])}, "
                 f"budget drift {drift[i]:.3g} on {atom_names[atom[i]]}, "
                 f"drift/t {drift[i] / max(t_exit[i], 1.0):.3g} /s, "
                 f"t {t_exit[i]:.3g} s, longdy {longdy[i]:.3g}")

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
    # reject frac > 1 - 1/init_oversample. init_max_nonconverged_frac is a GATE:
    # _init_state raises above it. A draw counts as non-converged at cap c when
    # accept_count >= c (same
    # convention as the censoring check above and _init_state's exhausted test).
    warn = float(cfg.init_max_nonconverged_frac)   # the gate, not a warning
    over = float(cfg.init_oversample)
    fail_frac = 1.0 - 1.0 / over    # reject frac above which oversampling can't fill N
    log.info(f"=== production cold-init gate (reject+oversample: init_oversample={over:g} "
             f"tolerates reject frac up to {fail_frac:.0%}, RAISES above {warn:.0%}; "
             f"this preset's count_max={preset_count_max}) ===")
    # _init_state rejects THREE classes (pipeline.py: exhausted | stalled | nonfinite),
    # so a verdict built on the cap alone understates the attrition. A NaN equilibrium
    # seed is the sharpest case: the runner exits at reason 5 with zero accepted steps,
    # which reads as the cheapest possible success on accept_count and as a -1e30
    # likelihood here.
    nonfinite = ~np.isfinite(Lnp) | (Lnp <= P.REJECT_BELOW)
    ex_probe = wa >= int(args.count_max_probe)
    log.info(f"  rejection classes at the probe cap: {int(ex_probe.sum())} exhausted, "
             f"{int((~conv_ok & ~ex_probe & ~nonfinite).sum())} stall-certified, "
             f"{int(nonfinite.sum())} non-finite / <= -1e29 likelihood, of {len(wa)} draws")
    cands = sorted({int(c) for c in (preset_count_max, 5000, 10000,
                                     int(args.count_max_probe)) if c})
    for cand in cands:
        if cand > int(args.count_max_probe):
            log.info(f"  at count_max={cand:>6d}: unknown (above the probe cap)")
            continue
        exhausted = wa >= cand
        stalled = ~conv_ok & ~exhausted & ~nonfinite
        frac = float(np.mean(nonfinite | exhausted | stalled))
        if frac > fail_frac:
            verdict = f"RAISE -- oversample x{over:g} cannot fill N (need reject <= {fail_frac:.0%})"
        elif frac > warn:
            verdict = (f"RAISE -- exceeds the declared init_max_nonconverged_frac "
                       f"{warn:.0%} (pipeline._init_state raises)")
        else:
            verdict = "reject+cull OK"
        tag = "   <- this preset" if preset_count_max and cand == int(preset_count_max) else ""
        log.info(f"  at count_max={cand:>6d}: {frac:>5.1%} rejected "
                 f"(exhausted+stall-certified+non-finite) -> {verdict}{tag}")

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
        # the cold-batch route this calibration ran on (0 = one lockstep batch)
        "cold_lanes": int(cfg.cold_lanes),
        "cold_refill_chunk": int(cfg.cold_refill_chunk),
        "n_censored": n_censored, "accept_count": wa.tolist(), "percentiles": pct,
        "param_names": names, "theta": Theta.tolist(),
        # per-draw likelihood and certificate: the attrition justification
        # compares the censored draws' L against the certified bulk
        "log_l": Lnp.tolist(),
        "conv_normal": conv_ok.tolist(), "longdy": longdy.tolist(),
        "budget_drift_max": drift.tolist(), "t_exit_s": t_exit.tolist(),
    }
    # The lane count goes in the name: a multi-arm bench (cal_all / cal_l48 /
    # cal_l16) runs several lane counts into ONE out_dir and would otherwise
    # overwrite its own results.
    suffix = "" if int(args.seed_offset) == 0 else f"_seed{int(args.seed_offset)}"
    if int(cfg.cold_lanes) > 0:
        suffix += f"_lanes{int(cfg.cold_lanes)}"
    out_path = cfg.out_dir / f"count_max_calibration{suffix}.json"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    log.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
