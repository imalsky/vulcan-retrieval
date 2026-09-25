"""run_nautilus.py -- nested sampling (nautilus) on the retrieval's likelihood.

Same case, preset, overrides and observations as run_smc; a different sampler.
Every likelihood call is a COLD column through ``pipe.batch_eval_cold_l_diag``
(the certificate gate, on the lane queue at ``cfg.cold_lanes``) with the SMC
init's rejection rule: a non-finite forward, a T-P draw outside the window, a
count_max-exhausted solve or a stall-certified exit gets log L = -inf. The
evidence is therefore the ZERO-FILLED box evidence, the quantity the SMC run
reports as ``logZ_box`` -- not its ``logZ`` (pipeline.evidence_report).

nautilus samples the unit cube; z -> u = logit(z) is the retrieval's own
u-space (pipeline.make_uspace), so theta = theta_from_u(u) follows the box
prior exactly and the posterior is stored in theta.

Output directory: ``<out_dir>_nautilus`` (default runs/<case>/data/gpu_nautilus):
nautilus_checkpoint.hdf5 (rewritten every iteration), nautilus_tally.json
(rejection counts, summed over jobs), and when the run finishes
nautilus_posterior.npz + nautilus_summary.json. ``SMC_RESUME=1`` (the PBS's
RESUME=1) continues the checkpoint; without it an existing checkpoint raises.
The run stops at ``cfg.walltime_seconds`` (the preset's governor) and exits 0
unfinished; resubmit with RESUME=1.

    python -m retrieval_framework.run_nautilus <run_dir>
Env: NAUTILUS_N_LIVE (default 500), NAUTILUS_N_EFF (default 10000),
NAUTILUS_N_BATCH (default 2 x cold_lanes: each lane solves ~2 columns per batch;
the SMC init already runs 2.5 x lanes columns in one call).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework.run_smc import make_config, set_observations, write_config_json

log = logging.getLogger("retrieval")


def make_loglike(pipe, n_batch: int, tally: dict):
    """Batched log-likelihood of unit-cube points (n, n_dim) -> (n,), -inf on
    every rejection class. Calls the compiled evaluator at one fixed width
    ``n_batch`` (a short last chunk is padded with copies of its last row) and
    adds this call's counts to ``tally``."""
    import jax
    import jax.numpy as jnp

    from retrieval_framework import pipeline as P

    ev = jax.jit(pipe.batch_eval_cold_l_diag)
    tp_ok = jax.jit(jax.vmap(lambda u: pipe.tp_valid(pipe.theta_from_u(u))))
    Y0, refs0 = P._blank_state(pipe, n_batch)
    count_max = int(pipe.fwd.chem.count_max)

    def loglike(Z):
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        U = np.log(Z) - np.log1p(-Z)
        out = np.empty(len(Z))
        for i in range(0, len(Z), n_batch):
            Uc = U[i:i + n_batch]
            m = len(Uc)
            if m < n_batch:
                Uc = np.concatenate([Uc, np.repeat(Uc[-1:], n_batch - m, axis=0)])
            Uj = jnp.asarray(Uc, pipe.dtype)
            L, _, _, cd = ev(Uj, Y0, refs0)
            L = np.asarray(jax.device_get(L), np.float64)[:m]
            acc = np.asarray(jax.device_get(cd.accept_count), np.int64)[:m]
            conv = np.asarray(jax.device_get(cd.conv_normal), bool)[:m]
            tp_out = ~np.asarray(jax.device_get(tp_ok(Uj)), bool)[:m]
            nonfinite = ~np.isfinite(L) | (L <= P.REJECT_BELOW)
            exhausted = ~tp_out & (acc >= count_max)
            stalled = ~tp_out & ~conv & ~exhausted & ~nonfinite
            dead = tp_out | nonfinite | exhausted | stalled
            out[i:i + m] = np.where(dead, -np.inf, L)
            for k, v in (("n_eval", m), ("tp_out", tp_out.sum()),
                         ("exhausted", exhausted.sum()), ("stalled", stalled.sum()),
                         ("nonfinite", (nonfinite & ~tp_out & ~exhausted).sum())):
                tally[k] = tally.get(k, 0) + int(v)
        return out

    return loglike


def main() -> None:
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=".",
                    help="retrieval case directory containing case.py (default: cwd)")
    args = ap.parse_args()

    cfg, preset = make_config(Path(args.run_dir))
    cfg = replace(cfg, out_dir=cfg.out_dir.parent / f"{cfg.out_dir.name}_nautilus")
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log", mode="a")],
        force=True)
    n_live = int(os.environ.get("NAUTILUS_N_LIVE", "500"))
    n_eff = int(os.environ.get("NAUTILUS_N_EFF", "10000"))
    lanes = int(cfg.cold_lanes)
    n_batch = int(os.environ.get("NAUTILUS_N_BATCH", str(2 * lanes if lanes > 0 else 132)))
    log.info(f"run_dir={Path(args.run_dir).resolve()} preset={preset} out_dir={out}")
    log.info(f"nautilus: n_live={n_live} n_eff={n_eff} n_batch={n_batch} "
             f"(cold_lanes={lanes}) seed={cfg.seed}")
    log.info(C.describe_config(cfg, preset))

    import jax
    import nautilus

    from retrieval_framework import certificate as _cert
    from retrieval_framework import pipeline as P
    log.info(f"jax backend={jax.default_backend()} devices={jax.devices()} "
             f"nautilus {nautilus.__version__}")
    hw = C.hardware_profile()
    log.info(f"hardware: {hw}")
    for w in C.hardware_warnings(hw):
        log.warning(w)

    t0 = time.perf_counter()
    pipe = P.build_pipeline(cfg)
    log.info(f"Built pipeline in {time.perf_counter() - t0:.1f}s | n_dim={pipe.n_dim}: {pipe.names}")
    obs_path = out / "observations.npz"
    obs_save = set_observations(cfg, pipe, P, obs_path)

    # resume identity before anything in the run directory is written
    ckpt = out / "nautilus_checkpoint.hdf5"
    digest_path = out / "target_digest.txt"
    tally_path = out / "nautilus_tally.json"
    want = str(getattr(pipe, "target_digest", "") or "")
    resume = os.environ.get("SMC_RESUME", "").strip().lower() in ("1", "true", "yes")
    if resume:
        if not ckpt.exists():
            raise FileNotFoundError(f"SMC_RESUME=1 but no checkpoint at {ckpt}; unset RESUME "
                                    "to start a fresh run.")
        have = digest_path.read_text().strip() if digest_path.exists() else ""
        if have != want:
            raise RuntimeError(f"nautilus resume refused: {ckpt.name} belongs to a different "
                               f"target (digest {have[:16] or '<absent>'} vs this run's "
                               f"{want[:16]}). Restore the original target or use a fresh "
                               "output directory.")
    elif ckpt.exists():
        raise FileExistsError(f"{ckpt} exists: set RESUME=1 to continue it, or point "
                              "SMC_RETRIEVAL_OUT_DIR at a fresh directory.")
    write_config_json(cfg, pipe, preset)
    if obs_save is not None:
        P.save_npz(obs_path, **obs_save)
    (out / _cert.MANIFEST_FILE).write_text(json.dumps(
        _cert.target_manifest(cfg, pipe), indent=2, sort_keys=True, default=str) + "\n")
    digest_path.write_text(want + "\n")

    tally = json.loads(tally_path.read_text()) if (resume and tally_path.exists()) else {}
    like = make_loglike(pipe, n_batch, tally)

    def loglike_saved(Z):
        L = like(Z)
        tally_path.write_text(json.dumps(tally, indent=2) + "\n")
        return L

    sampler = nautilus.Sampler(lambda z: z, loglike_saved, n_dim=pipe.n_dim, n_live=n_live,
                               n_batch=n_batch, vectorized=True, pass_dict=False,
                               seed=int(cfg.seed), filepath=str(ckpt), resume=resume)
    budget = float(cfg.walltime_seconds)
    timeout = max(budget - (time.time() - t_start), 60.0) if budget > 0 else float("inf")
    log.info(f"nautilus: starting at n_like={sampler.n_like}; timeout {timeout / 3600:.2f} h")
    t0 = time.perf_counter()
    done = sampler.run(n_eff=n_eff, discard_exploration=True, timeout=timeout, verbose=True)
    log.info(f"nautilus: {'finished' if done else 'stopped at the wall budget'} after "
             f"{(time.perf_counter() - t0) / 3600:.2f} h | n_like={sampler.n_like} "
             f"| log_z={sampler.log_z:.3f} | n_eff={sampler.n_eff:.0f} | tally={tally}")
    if not done:
        log.warning("nautilus did not finish: resubmit with RESUME=1 (the checkpoint is "
                    f"{ckpt}).")
        return

    z, log_w, log_l = sampler.posterior()
    z_eq, _, _ = sampler.posterior(equal_weight=True)
    to_theta = jax.jit(jax.vmap(pipe.theta_from_u))
    theta = np.asarray(to_theta(np.log(z) - np.log1p(-z)), np.float64)
    theta_eq = np.asarray(to_theta(np.log(z_eq) - np.log1p(-z_eq)), np.float64)
    P.save_npz(out / "nautilus_posterior.npz",
               param_names=np.asarray(pipe.names, dtype="<U64"),
               param_labels=np.asarray(pipe.labels, dtype="<U64"),
               target_digest=np.asarray(want),
               theta=theta, log_w=log_w, log_l=log_l, samples=theta_eq,
               logZ_box=np.asarray(sampler.log_z), n_like=np.asarray(sampler.n_like))
    summary = {
        "logZ_box": float(sampler.log_z), "n_like": int(sampler.n_like),
        "n_eff": float(sampler.n_eff), "n_live": n_live, "n_batch": n_batch,
        "rejections": tally, "preset": preset, "seed": int(cfg.seed),
        "evidence_note": "zero-filled box evidence (log L = -inf on every rejection); "
                         "compare with an SMC run's logZ_box",
        "medians": {n: float(np.median(theta_eq[:, i])) for i, n in enumerate(pipe.names)},
    }
    (out / "nautilus_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info(f"wrote {out / 'nautilus_posterior.npz'} and nautilus_summary.json")


if __name__ == "__main__":
    main()
