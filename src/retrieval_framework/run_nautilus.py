"""run_nautilus.py -- nested sampling (nautilus) on the retrieval's likelihood.

Same case, preset, overrides and observations as run_smc; a different sampler.
Every likelihood call goes through the pipeline's batched chemistry + RT on the
lane queue at ``cfg.cold_lanes`` with the SMC init's rejection rule: a
non-finite forward, a T-P draw outside the window, a count_max-exhausted solve
or a stall-certified exit gets log L = -inf. The evidence is therefore a
ZERO-FILLED box evidence, the kind of quantity an SMC run reports as
``logZ_box`` -- not its ``logZ`` (pipeline.evidence_report) -- but not the
identical one. nautilus gates each draw on the PRIMAL solve's certificate
only; SMC gates on the certificate of the jvp'd program, and its init also
culls primal survivors that fail that phase-2 re-certification (SMC's
``logZ_box`` carries the cull as f_c2). There is no phase-2 cull here, so the
two box evidences agree only when the phase-2 flip rate is negligible: report
SMC's f_c2 next to any comparison.

Warm starts (default; NAUTILUS_WARM=0 turns them off): every certified column
becomes an anchor, keyed by its chemistry + T-P coordinates in the unit cube,
and each later solve continues from its nearest anchor (the pipeline's warm
map at the cold count_max) instead of the cold two-stage solve; batches before
the first certified column run cold. The certified state is start-dependent at
the convergence tolerance (<= 5 ppm on 15 of 16 CPU-screen targets, one 48 ppm,
against ~69 ppm noise; vulcan-retrieval notes 2.14), and so is WHICH draws are
rejected (a warm continuation certifies or fails on its own: 16/16 certified
warm against 14/16 cold in that screen). With warm starts the likelihood and
the rejection set depend on the evaluation order, so the posterior and
evidence are approximate and the evidence is NOT an SMC run's logZ_box (the
maintainer's choice, for ~3-5x fewer steps); use NAUTILUS_WARM=0 for an
evidence claim. NAUTILUS_WARM=0 is the cold primal map on the lane queue (a
draw's column depends on its batch at the convergence scale); its evidence
compares with SMC's only as stated above.

Anchors are written inside the likelihood call, before nautilus checkpoints
the batch: a job killed in between resumes by redrawing that batch, which then
starts from its own columns (and the tally counts it twice). The set is never
pruned: ~44 KB per certified column, on disk and in memory.

nautilus samples the unit cube; z -> u = logit(z) is the retrieval's own
u-space (pipeline.make_uspace), so theta = theta_from_u(u) follows the box
prior exactly and the posterior is stored in theta.

Output directory: ``<out_dir>_nautilus`` (default runs/<case>/data/gpu_nautilus):
nautilus_checkpoint.hdf5 (rewritten every iteration), anchors/ (one npz of
certified columns per batch, reloaded on resume), nautilus_tally.json
(rejection and warm-start counts, summed over jobs), and when the run finishes
nautilus_posterior.npz + nautilus_summary.json. ``SMC_RESUME=1`` (the PBS's
RESUME=1) continues the checkpoint; without it an existing checkpoint raises.
The run stops at ``cfg.walltime_seconds`` (the preset's governor) and exits 0
unfinished; resubmit with RESUME=1.

    python -m retrieval_framework.run_nautilus <run_dir>
Env: NAUTILUS_N_LIVE (default N_LIVE), NAUTILUS_N_EFF (default N_EFF),
NAUTILUS_N_BATCH (default 2 x cold_lanes: each lane solves ~2 columns per batch;
the SMC init already runs 2.5 x lanes columns in one call; with cold_lanes = 0,
one lockstep batch of config_schema.device_lane_count() columns), NAUTILUS_WARM
(default 1).
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
from retrieval_framework.run_smc import (
    log_hardware, make_config, set_observations, setup_logging, write_run_identity)

log = logging.getLogger("retrieval")

N_LIVE = 500        # live points
N_EFF = 10_000      # effective posterior samples at which nautilus stops


class Anchors:
    """Certified columns so far: unit-cube chemistry + T-P coordinates ``z``,
    the column ``Y`` and its reference (lnZ, c_o). One npz per added batch in
    ``folder``; an existing folder is reloaded (resume)."""

    def __init__(self, folder: Path, n_key: int):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.z = np.empty((0, n_key))
        self.refs = np.empty((0, 2))
        self.parts, self.part, self.row = [], np.empty(0, int), np.empty(0, int)
        for f in sorted(folder.glob("anchors_*.npz")):
            with np.load(f) as d:
                self._append(d["z"], d["Y"], d["refs"])

    def __len__(self):
        return len(self.z)

    def _append(self, z, Y, refs):
        self.part = np.concatenate([self.part, np.full(len(z), len(self.parts))])
        self.row = np.concatenate([self.row, np.arange(len(z))])
        self.parts.append(np.asarray(Y))
        self.z = np.concatenate([self.z, z])
        self.refs = np.concatenate([self.refs, refs])

    def add(self, z, Y, refs):
        if len(z):
            f = self.folder / f"anchors_{len(self.parts):05d}.npz"
            tmp = f.with_name(f".{f.name}.part")
            with open(tmp, "wb") as fh:
                np.savez(fh, z=z, Y=Y, refs=refs)
            os.replace(tmp, f)
            self._append(z, Y, refs)

    def nearest(self, zq):
        """(index, distance) of each query's nearest anchor (Euclidean, unit cube)."""
        idx = np.empty(len(zq), int)
        dist = np.empty(len(zq))
        for j in range(0, len(zq), 64):
            d2 = ((zq[j:j + 64, None, :] - self.z[None]) ** 2).sum(-1)
            idx[j:j + 64] = d2.argmin(1)
            dist[j:j + 64] = np.sqrt(d2.min(1))
        return idx, dist

    def columns(self, idx):
        return (np.stack([self.parts[p][r] for p, r in zip(self.part[idx], self.row[idx])]),
                self.refs[idx])


def make_loglike(pipe, n_batch: int, tally: dict, anchors: Anchors | None = None):
    """Batched log-likelihood of unit-cube points (n, n_dim) -> (n,), -inf on
    every rejection class. Calls a compiled evaluator at one fixed width
    ``n_batch`` (a short last chunk is padded with copies of its last row) and
    adds this call's counts to ``tally``. With ``anchors``, a chunk starts
    from the nearest certified column (warm, cold count_max) once one exists,
    and every certified column it returns becomes an anchor."""
    import jax
    import jax.numpy as jnp

    from retrieval_framework import pipeline as P

    ev_cold = jax.jit(pipe.batch_eval_cold_l_diag)
    ev_warm = (jax.jit(pipe._make_batch_eval("warm", False, mutation_cap=False))
               if anchors is not None else None)
    tp_ok = jax.jit(jax.vmap(lambda u: pipe.tp_valid(pipe.theta_from_u(u))))
    Y0, refs0 = P._blank_state(pipe, n_batch)
    count_max = int(pipe.fwd.chem.count_max)
    k = int(pipe.n_chem_tp)

    def loglike(Z):
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        out = np.empty(len(Z))
        for i in range(0, len(Z), n_batch):
            Zc = Z[i:i + n_batch]
            m = len(Zc)
            if m < n_batch:
                Zc = np.concatenate([Zc, np.repeat(Zc[-1:], n_batch - m, axis=0)])
            Uj = jnp.asarray(np.log(Zc) - np.log1p(-Zc), pipe.dtype)
            warm = anchors is not None and len(anchors) > 0
            if warm:
                idx, dist = anchors.nearest(Zc[:, :k])
                Yw, rw = anchors.columns(idx)
                L, Ynew, rnew, st = ev_warm(Uj, jnp.asarray(Yw, pipe.dtype),
                                            jnp.asarray(rw, pipe.dtype))
                acc, conv = st.acc, st.conv_ok
            else:
                L, Ynew, rnew, cd = ev_cold(Uj, Y0, refs0)
                acc, conv = cd.accept_count, cd.conv_normal
            L = np.asarray(jax.device_get(L), np.float64)[:m]
            acc = np.asarray(jax.device_get(acc), np.int64)[:m]
            conv = np.asarray(jax.device_get(conv), bool)[:m]
            tp_out = ~np.asarray(jax.device_get(tp_ok(Uj)), bool)[:m]
            # solver state first: the warm evaluator already floors L on a
            # stalled or exhausted exit, which is not a non-finite forward
            exhausted = ~tp_out & (acc >= count_max)
            stalled = ~tp_out & ~exhausted & ~conv
            nonfinite = (~tp_out & ~exhausted & conv
                         & (~np.isfinite(L) | (L <= P.REJECT_BELOW)))
            dead = tp_out | exhausted | stalled | nonfinite
            out[i:i + m] = np.where(dead, -np.inf, L)
            mode = "warm" if warm else "cold"
            for key, v in (("n_eval", m), ("tp_out", tp_out.sum()),
                           ("exhausted", exhausted.sum()), ("stalled", stalled.sum()),
                           ("nonfinite", nonfinite.sum()),
                           (f"n_{mode}", m), (f"{mode}_certified", (~dead).sum()),
                           (f"{mode}_steps_certified", acc[~dead].sum())):
                tally[key] = tally.get(key, 0) + int(v)
            if warm:
                tally["warm_anchor_dist_sum"] = (tally.get("warm_anchor_dist_sum", 0.0)
                                                 + float(dist[:m].sum()))
            if anchors is not None:
                alive = ~dead
                anchors.add(Zc[:m][alive][:, :k],
                            np.asarray(jax.device_get(Ynew), np.float64)[:m][alive],
                            np.asarray(jax.device_get(rnew), np.float64)[:m][alive])
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
    setup_logging(out / "run.log")
    n_live = int(os.environ.get("NAUTILUS_N_LIVE", str(N_LIVE)))
    n_eff = int(os.environ.get("NAUTILUS_N_EFF", str(N_EFF)))
    lanes = int(cfg.cold_lanes)
    n_batch = int(os.environ.get(
        "NAUTILUS_N_BATCH", str(2 * lanes if lanes > 0 else C.device_lane_count())))
    warm = os.environ.get("NAUTILUS_WARM", "1").strip() != "0"
    log.info(f"run_dir={Path(args.run_dir).resolve()} preset={preset} out_dir={out}")
    log.info(f"nautilus: n_live={n_live} n_eff={n_eff} n_batch={n_batch} "
             f"(cold_lanes={lanes}) seed={cfg.seed} "
             f"starts={'anchored warm (growing)' if warm else 'cold'}")
    log.info(C.describe_config(cfg, preset))

    import jax
    import nautilus

    from retrieval_framework import pipeline as P
    log.info(f"jax backend={jax.default_backend()} devices={jax.devices()} "
             f"nautilus {nautilus.__version__}")
    log_hardware()

    t0 = time.perf_counter()
    pipe = P.build_pipeline(cfg)
    log.info(f"Built pipeline in {time.perf_counter() - t0:.1f}s | n_dim={pipe.n_dim}: {pipe.names}")
    obs_save = set_observations(cfg, pipe, P)

    # resume identity before anything in the run directory is written; the
    # start policy, the batch width (the lane queue's composition) and n_live
    # (nautilus does not restore it) are part of the target, so a resume
    # cannot switch them
    ckpt = out / "nautilus_checkpoint.hdf5"
    digest_path = out / "target_digest.txt"
    tally_path = out / "nautilus_tally.json"
    want = (f"{pipe.target_digest}|{'warm' if warm else 'cold'}"
            f"|n_batch={n_batch}|n_live={n_live}")
    resume = os.environ.get("SMC_RESUME", "").strip().lower() in ("1", "true", "yes")
    if resume:
        if not ckpt.exists():
            raise FileNotFoundError(f"SMC_RESUME=1 but no checkpoint at {ckpt}; unset RESUME "
                                    "to start a fresh run.")
        have = digest_path.read_text().strip() if digest_path.exists() else ""
        if have != want:
            raise RuntimeError(f"nautilus resume refused: {ckpt.name} belongs to a different "
                               f"target ({have or '<absent>'} vs this run's {want}). Restore "
                               "the original target and settings or use a fresh output "
                               "directory.")
    elif ckpt.exists() or any((out / "anchors").glob("anchors_*.npz")):
        raise FileExistsError(f"{out} holds a previous nautilus run (checkpoint or anchors/): "
                              "set RESUME=1 to continue it, or point SMC_RETRIEVAL_OUT_DIR "
                              "at a fresh directory.")
    write_run_identity(cfg, pipe, preset, obs_save)
    digest_path.write_text(want + "\n")

    tally = json.loads(tally_path.read_text()) if (resume and tally_path.exists()) else {}
    anchors = Anchors(out / "anchors", int(pipe.n_chem_tp)) if warm else None
    if anchors is not None:
        log.info(f"nautilus: {len(anchors)} anchor column(s) loaded")
    like = make_loglike(pipe, n_batch, tally, anchors)

    def loglike_saved(Z):
        L = like(Z)
        tmp = tally_path.with_name(f".{tally_path.name}.part")
        tmp.write_text(json.dumps(tally, indent=2) + "\n")
        os.replace(tmp, tally_path)
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
               target_digest=np.asarray(want),
               theta=theta, log_w=log_w, log_l=log_l, samples=theta_eq,
               logZ_box=np.asarray(sampler.log_z), n_like=np.asarray(sampler.n_like))
    summary = {
        "logZ_box": float(sampler.log_z), "n_like": int(sampler.n_like),
        "n_eff": float(sampler.n_eff), "n_live": n_live, "n_batch": n_batch,
        "starts": "anchored warm (growing)" if warm else "cold",
        "rejections_and_starts": tally, "preset": preset, "seed": int(cfg.seed),
        "evidence_note": "zero-filled box evidence (log L = -inf on every rejection)"
                         + ("; APPROXIMATE: warm starts move the likelihood and which draws "
                            "are rejected, so this is not an SMC logZ_box (NAUTILUS_WARM=0 "
                            "for an evidence claim)" if warm
                            else "; compares with an SMC run's logZ_box only "
                                 "when that run's f_c2 (phase-2 cull) is near 1"),
        "medians": {n: float(np.median(theta_eq[:, i])) for i, n in enumerate(pipe.names)},
    }
    (out / "nautilus_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info(f"wrote {out / 'nautilus_posterior.npz'} and nautilus_summary.json")


if __name__ == "__main__":
    main()
