#!/usr/bin/env python3
"""validate_warm.py -- measure the warm-continuation likelihood bias directly.

The SMC mutation kernel warm-continues each particle's chemistry from its carried
column, so the accepted likelihood is history-dependent at the convergence
tolerance and the MALA kernel is only approximately invariant. This tool measures
that bias where it matters: it re-solves a checkpointed particle cloud COLD (the
published solve-from-baseline two-stage map, the same map that anchored the run's
init and carries no history at all) and compares against the warm-carried
log-likelihoods stored in the checkpoint. |dlogL| is the chi^2-weighted spectrum
difference, i.e. exactly the quantity that enters MH acceptance and tempering
weights -- if it is small relative to ~0.1 log-units, the warm kernel is
posterior-exact for all practical purposes.

Run it on the GPU node against a finished run (the per-stage checkpoint IS the
final cloud), with the SAME preset/overrides the run used:

    SMC_RETRIEVAL_PRESET=gpu python -m retrieval_framework.validate_warm runs/w39b_smc_retrieval

Reads the run's own observations.npz (never regenerates) and smc_checkpoint.npz;
writes validate_warm.npz next to them, logs a verdict, and exits nonzero on FAIL
so a PBS wrapper notices. A particle that does not cold-converge within count_max
(possible at posterior edges) is reported separately, never folded into the bias
statistics. Cost: one cold init-phase-1-equivalent pass (~minutes on the GH200).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):                      # direct-file execution support
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# PASS gate on max|logL_cold - logL_warm| over the cloud. 0.1 log-units is far
# inside a 1-sigma contour shift for a ~10-D posterior (whose logL spans ~n_dim/2
# across the cloud); the convergence tolerance (yconv_cri=0.01) predicts ~1e-2.
DLOGL_MAX_PASS = 0.1
# WARN if more than this fraction of the cloud fails to cold-converge: the
# posterior would be sitting against the count_max convergence cliff.
COLD_NONCONV_WARN_FRAC = 0.10

logger = logging.getLogger("retrieval")


def compare(L_warm, L_cold, worst_accept, count_max: int) -> dict:
    """Pure-numpy warm-vs-cold comparison (unit-tested). Excludes cold-nonconverged
    particles (count_max-exhausted or -1e30 forward) and any dead warm entries from
    the bias statistics; returns them as separate counts."""
    L_warm = np.asarray(L_warm, np.float64)
    L_cold = np.asarray(L_cold, np.float64)
    wa = np.asarray(worst_accept, np.int64)
    dead_warm = ~np.isfinite(L_warm) | (L_warm <= -1.0e29)
    cold_nonconv = ~np.isfinite(L_cold) | (L_cold <= -1.0e29) | (wa >= int(count_max))
    ok = ~dead_warm & ~cold_nonconv
    d = np.where(ok, L_cold - L_warm, np.nan)
    dd = d[ok]
    stats = dict(
        n=int(L_warm.size), n_ok=int(ok.sum()),
        n_dead_warm=int(dead_warm.sum()), n_cold_nonconverged=int(cold_nonconv.sum()),
        dlogl=d,
        abs_max=float(np.max(np.abs(dd))) if dd.size else float("nan"),
        abs_median=float(np.median(np.abs(dd))) if dd.size else float("nan"),
        abs_p95=float(np.percentile(np.abs(dd), 95.0)) if dd.size else float("nan"),
        logl_spread=float(L_warm[ok].max() - L_warm[ok].min()) if dd.size else float("nan"),
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=".",
                    help="retrieval case directory containing case.py (default: cwd)")
    args = ap.parse_args()

    from retrieval_framework.run_smc import make_config
    cfg, preset = make_config(Path(args.run_dir))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    out = cfg.out_dir
    ck_path, obs_path = out / "smc_checkpoint.npz", out / "observations.npz"
    for p in (ck_path, obs_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- validate_warm runs against a "
                                    "finished run's output dir (same preset/overrides)")
    ck = np.load(ck_path)
    needed = ("u_particles", "y_state", "chem_refs", "loglik", "betas")
    if not all(k in ck.files for k in needed):
        raise KeyError(f"{ck_path} predates the carried chemistry state "
                       f"(needs {needed}); nothing to validate")
    final_beta = float(ck["betas"][-1])
    logger.info(f"validating checkpoint at beta={final_beta:.4f} "
                f"({ck['u_particles'].shape[0]} particles, preset={preset})")

    from retrieval_framework import pipeline as P
    import jax
    import jax.numpy as jnp

    pipe = P.build_pipeline(cfg)
    if ck["u_particles"].shape[1] != pipe.n_dim:
        raise ValueError(f"checkpoint n_dim {ck['u_particles'].shape[1]} != pipeline "
                         f"{pipe.n_dim}: wrong preset/overrides for this run dir")
    d = np.load(obs_path)
    pipe.set_observations(d["depth"], d["sigma"])   # the run's own obs, never regenerated

    U = jnp.asarray(ck["u_particles"], pipe.dtype)
    N = int(U.shape[0])
    Y0, refs0 = P._blank_state(pipe, N)             # cold map: no history enters
    t0 = time.perf_counter()
    L_cold, _, _, worst = jax.jit(pipe.batch_eval_cold_l_diag)(U, Y0, refs0)
    jax.block_until_ready(L_cold)
    logger.info(f"cold re-solve of the cloud done in {time.perf_counter() - t0:.1f}s")

    count_max = int(pipe.fwd.chem.count_max)
    s = compare(ck["loglik"], np.asarray(jax.device_get(L_cold)),
                np.asarray(jax.device_get(worst)), count_max)

    P.save_npz(out / "validate_warm.npz",
               loglik_warm=np.asarray(ck["loglik"], np.float64),
               loglik_cold=np.asarray(jax.device_get(L_cold), np.float64),
               worst_accept=np.asarray(jax.device_get(worst), np.int64),
               dlogl=s["dlogl"], final_beta=np.asarray(final_beta),
               count_max=np.asarray(count_max, np.int64))

    logger.info(f"warm-vs-cold on {s['n_ok']}/{s['n']} particles "
                f"(cold-nonconverged: {s['n_cold_nonconverged']}, dead warm: {s['n_dead_warm']}) | "
                f"|dlogL| median={s['abs_median']:.3e} p95={s['abs_p95']:.3e} "
                f"max={s['abs_max']:.3e} | cloud logL spread={s['logl_spread']:.2f}")
    if s["n_cold_nonconverged"] > COLD_NONCONV_WARN_FRAC * s["n"]:
        logger.warning(f"{s['n_cold_nonconverged']}/{s['n']} particles did not "
                       "cold-converge within count_max -- the cloud sits near the "
                       "convergence cliff; the excluded particles are unvalidated")
    ok = s["n_ok"] > 0 and s["abs_max"] < DLOGL_MAX_PASS
    logger.info(f"VERDICT: {'PASS' if ok else 'FAIL'} "
                f"(max|dlogL| {s['abs_max']:.3e} vs gate {DLOGL_MAX_PASS}) "
                + ("-- warm-continuation bias is negligible at this cloud" if ok else
                   "-- warm likelihoods are history-dependent beyond the gate; tighten "
                   "yconv_cri or rerun with smc_chem_mode='cold' before publishing"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
