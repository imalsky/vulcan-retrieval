"""run_nautilus.py -- nested sampling (nautilus) on the retrieval's likelihood.

The case, preset, overrides and observations come from run_setup. Every
likelihood call goes through the pipeline's batched chemistry + RT on the lane
queue (``lanes``, below): a non-finite forward, a T-P draw outside the window,
a count_max-exhausted solve or a stall-certified exit gets log L = -inf. The
evidence is therefore a ZERO-FILLED box evidence, ``logZ_box``: the integral of
prior x likelihood over the box with every rejected region at zero, so it
depends on the solver settings (count_max, rtol) through what is rejected.

Warm starts (``sampler.warm_starts``, on by default): every certified column
becomes an anchor keyed by its unit-cube chemistry + T-P coordinates, and each
later solve continues from its nearest anchor (the warm map at the cold
count_max); batches before the first certified column run cold. The certified
state and which draws are rejected then depend on the evaluation order at the
convergence tolerance, so the posterior and evidence are approximate. An
evidence claim uses ``warm_starts: false``, the cold primal map on the lane queue (a
draw's column depends on its batch at the convergence scale).

Anchors are written inside the likelihood call, before nautilus checkpoints
the batch: a job killed in between resumes by redrawing that batch, which then
starts from its own columns (and the tally counts it twice). The set is never
pruned: ~44 KB per certified column, on disk and in memory.

nautilus samples the unit cube; z -> u = logit(z) is the retrieval's own
u-space (pipeline.make_uspace), so theta = theta_from_u(u) follows the box
prior exactly and the posterior is stored in theta.

Output directory: output/<case>/<name>_nautilus for the spec runs/<case>/<name>.yaml
(run_setup.load_spec):
nautilus_checkpoint.hdf5 (rewritten every iteration), anchors/ (one npz of
certified columns per batch, reloaded on resume), rejected/ (one npz per chunk
of the rejected draws: unit-cube point ``z``, class ``cls`` 0 T-P out, 1
exhausted, 2 stalled, 3 non-finite, accepted steps ``acc`` and the chunk's
start mode ``warm``; the conditioning map's input), with integrated nuisances
nuisance/ (one npz per chunk of the certified rows: ``z``, the ``log_l`` returned
to nautilus and the group ``stats`` their posterior is drawn from;
nuisance_posterior), nautilus_tally.json
(rejection and warm-start counts, summed over jobs), and when the run finishes
nautilus_posterior.npz, nautilus_summary.json and nautilus_certificate.json
(nautilus_report; its CLI also draws the figures). ``--resume`` continues the
checkpoint; without it an existing checkpoint raises. The run stops at
``cfg.walltime_seconds`` (the spec's governor) and exits 0 unfinished; resubmit
with ``--resume`` (on NAS: tools/submit.py --resume).

    python -m retrieval_framework.run_nautilus runs/<case>/<name>.yaml [--resume] [--dry-run]
The sampler settings are the spec's ``sampler`` block (n_live, with n_update at
half of it; n_eff; n_batch, default BATCH_PER_LANE x the lane count, so a lane
whose column certifies refills from the same batch; warm_starts); the engine's
environment is its ``runtime`` block, exported before the first jax import.
VULCAN_PROFILE_BATCH (BatchProfiler, off unless set) comes from the job script.
Each evaluator is a plain jax.jit program, compiled at its first call.

Sampler settings (the constants below, built by make_sampler): STOCK
nautilus-sampler 1.0.6 from PyPI with its default networks (N_NETWORKS ReLU MLPs),
n_update = n_live / 2, exploration until f_live = F_LIVE, bounds enlarged by
ENLARGE_PER_DIM, and the exploration phase discarded (DISCARD_EXPLORATION, the
unbiased estimator; keeping it biased log Z by about -0.2 on surrogates of the W39b
posterior). Each sampling-phase batch goes to one shell, as in stock nautilus; with
the exploration discarded every shell starts the sampling phase empty and gets one
whole batch first. validate_env refuses the maintainer's former fork (SiLU networks,
spread_batch), which carried the same 1.0.6 version string.

Every likelihood chunk appends one line to ``batches.jsonl``: its wall time,
start mode and the accepted step count of each column.

The first chunk of a fresh run (cold, box-prior draws) is its startup check
(startup_check): the run stops there, before any anchor or checkpoint is written,
if no draw was kept or if a decade of log10 Kzz at the case's reference pressure
holds STARTUP_MIN_DRAWS or more in-window draws and none certified, so a prior
region the solver cannot certify is refused in minutes instead of shaping the
posterior for a day.

Padding and out-of-window draws cost no solve: a chunk shorter than n_batch
is filled with copies of its last row, and the evaluator masks those and every
T-P-invalid row (``active``): their lanes take no chemistry step and no RT. The
RT runs only on the rows whose log L can be kept (T-P in window, certified).

Lanes: ``cfg.cold_lanes`` (one SM per lane in the gpu preset), times
FFI_BLOCKS_PER_SM under the ffi solver, whose one-buffer kernels keep that many
lanes resident per SM (396 lanes on a GH200).
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import json
import logging
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework.run_setup import (
    check_run_dir, export_runtime, load_spec, log_hardware, set_observations, setup_logging,
    write_run_identity)

log = logging.getLogger("retrieval")

F_LIVE = 0.03       # exploration ends when the live set holds this evidence share
ENLARGE_PER_DIM = 1.3
N_NETWORKS = 4      # nautilus's default ReLU networks (hidden layers 100-50-20)
DISCARD_EXPLORATION = True
BATCH_PER_LANE = 3  # points per lane per batch: certified lanes refill in-batch
# ffi one-buffer blocks resident per SM at ni 89 on compute capability 9.0:
# ~63 KB of shared memory each in the SM's 228 KB (VULCAN-JAX
# block_thomas_cuda.cu, bt_choose picks that variant at this width).
FFI_BLOCKS_PER_SM = 3
# In-window cold draws a log10 Kzz decade needs before none certifying refuses the
# run: 0 of 30 bounds that decade's certified fraction below 10% (95%).
STARTUP_MIN_DRAWS = 30


def startup_check(log10_kzz, in_window, kept, counts: dict) -> list[str]:
    """The first chunk's verdict: one line per decade of ``log10_kzz`` (at the
    reference pressure) with its in-window and kept (certified, finite log L) draws.
    Raises when nothing was kept (``counts``: the chunk's rejection classes, for the
    message) or when a decade with STARTUP_MIN_DRAWS or more in-window draws kept
    none: the posterior there would be cut by the solver, not by the data."""
    dec = np.floor(np.asarray(log10_kzz, np.float64)[in_window]).astype(int)
    ok = np.asarray(kept, bool)[in_window]
    lines, cut = [], []
    for d in np.unique(dec):
        n, k = int((dec == d).sum()), int(ok[dec == d].sum())
        thin = " (too few to judge)" if n < STARTUP_MIN_DRAWS else ""
        lines.append(f"startup: log10 Kzz [{d}, {d + 1}): {n} in window, {k} kept{thin}")
        if n >= STARTUP_MIN_DRAWS and k == 0:
            cut.append(f"[{d}, {d + 1})")
    if not ok.any():
        raise RuntimeError(
            "startup check: no draw of the first chunk was kept "
            f"({', '.join(f'{v} {c}' for c, v in counts.items())}); the run cannot start. "
            + " | ".join(lines))
    if cut:
        raise RuntimeError(
            f"startup check: no in-window cold draw certified in log10 Kzz {', '.join(cut)} "
            f"(>= {STARTUP_MIN_DRAWS} draws each): the posterior there would be cut by the "
            "solver, not the data. Narrow prior_lnKzz to exclude it, or measure the "
            "count_max that certifies it before raising the cap. " + " | ".join(lines))
    return lines


class Anchors:
    """Certified columns so far: unit-cube chemistry + T-P coordinates ``z``,
    the column ``Y`` and its reference composition (lnZ, c_o, s_o). One npz per
    added batch in ``folder``; an existing folder is reloaded (resume; a file
    with two-wide references from before the sulfur knob reads as s_o = 0)."""

    N_REF = 3

    def __init__(self, folder: Path, n_key: int):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.z = np.empty((0, n_key))
        self.refs = np.empty((0, self.N_REF))
        self.parts, self.part, self.row = [], np.empty(0, int), np.empty(0, int)
        for f in sorted(folder.glob("anchors_*.npz")):
            with np.load(f) as d:
                self._append(d["z"], d["Y"], d["refs"])

    def __len__(self):
        return len(self.z)

    def _append(self, z, Y, refs):
        refs = np.asarray(refs, np.float64)
        if refs.shape[1] < self.N_REF:
            refs = np.pad(refs, ((0, 0), (0, self.N_REF - refs.shape[1])))
        self.part = np.concatenate([self.part, np.full(len(z), len(self.parts))])
        self.row = np.concatenate([self.row, np.arange(len(z))])
        self.parts.append(np.asarray(Y))
        self.z = np.concatenate([self.z, z])
        self.refs = np.concatenate([self.refs, refs])

    def add(self, z, Y, refs):
        """Append certified columns as one new file."""
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


class BatchProfiler:
    """Env-gated profiling hook; inert (no library loaded, no call made) unless
    VULCAN_PROFILE_BATCH is set.

    VULCAN_PROFILE_BATCH=k: cudaProfilerStart just before likelihood batch k of
    THIS process (the k-th evaluator call, i.e. the k-th line it appends to
    batches.jsonl; batch 1 compiles the cold evaluator and the first warm batch
    compiles the warm one, so k = 3 is the first steady warm batch of a fresh
    run) and cudaProfilerStop when batch k+1 starts (or at exit): the window
    holds batch k and the nautilus update after it. Meant for nsys / nsys-jax
    --capture-range=cudaProfilerApi --capture-range-end=stop, so the run goes on
    unprofiled after the window."""

    def __init__(self):
        self.k = int(os.environ.get("VULCAN_PROFILE_BATCH", "0") or 0)
        self.n = 0
        self.active = False
        if self.k <= 0:
            return
        self._rt = self._cudart()
        atexit.register(self.stop)
        log.info(f"profile: capture batch {self.k} (cudaProfilerStart/Stop)")

    @staticmethod
    def _cudart():
        paths = []
        try:
            import nvidia.cuda_runtime as m  # the CUDA runtime of the jax wheels
            paths = [os.path.join(p, "lib", "libcudart.so.12") for p in m.__path__]
        except ImportError:
            pass
        for p in [*paths, "libcudart.so.12", "libcudart.so"]:
            try:
                return ctypes.CDLL(p)
            except OSError:
                continue
        raise OSError("VULCAN_PROFILE_BATCH: no libcudart to call cudaProfilerStart from")

    def stop(self):
        if self.active:
            rc = int(self._rt.cudaProfilerStop())
            self.active = False
            log.info(f"profile: cudaProfilerStop after batch {self.k} (rc {rc})")

    def batch_begin(self):
        """Called before every evaluator call; device work of the previous batch
        is complete here (its results were fetched to the host)."""
        self.n += 1
        if self.k and self.n == self.k + 1:
            self.stop()
        if self.k and self.n == self.k:
            rc = int(self._rt.cudaProfilerStart())
            self.active = True
            log.info(f"profile: cudaProfilerStart before batch {self.k} (rc {rc})")


def make_loglike(pipe, n_batch: int, tally: dict, anchors: Anchors | None = None,
                 batch_log: Path | None = None, prof: BatchProfiler | None = None,
                 rejects: Path | None = None, nuisance: Path | None = None,
                 kzz_ref: float | None = None):
    """Batched log-likelihood of unit-cube points (n, n_dim) -> (n,), -inf on
    every rejection class. Calls a jitted evaluator at one fixed width
    ``n_batch`` (a short last chunk is padded with masked copies of its last
    row; those and the T-P-invalid rows take no chemistry step and no RT) and
    adds this call's counts to ``tally``. With ``anchors``, a chunk starts
    from the nearest certified column (warm, cold count_max) once one exists,
    and every certified column it returns becomes an anchor. With
    ``batch_log``, each chunk appends its wall time, mode and per-column
    accepted step counts there. With ``rejects`` (a directory), each chunk
    writes its rejected rows as one npz (``z`` unit-cube points, ``cls`` the
    class, ``acc`` accepted steps, ``warm`` the start mode), so the rejected
    fraction can be mapped over the posterior afterwards. With ``nuisance`` (a
    directory; integrated nuisances), each chunk writes its certified rows' ``z``,
    returned ``log_l`` and group ``stats`` before nautilus sees them. With ``kzz_ref``
    (the case's base Kzz at the reference pressure, cm^2/s), the first chunk is the
    run's startup check (startup_check), run before that chunk writes anything."""
    import jax
    import jax.numpy as jnp

    from retrieval_framework import pipeline as P

    tp_ok = jax.jit(jax.vmap(lambda u: pipe.tp_valid(pipe.theta_from_u(u))))
    Y0, refs0 = P._blank_state(pipe, n_batch)
    count_max = int(pipe.fwd.chem.count_max)
    k = int(pipe.n_chem_tp)
    n_rej = n_nui = None
    if rejects is not None:
        rejects.mkdir(parents=True, exist_ok=True)
        n_rej = [len(list(rejects.glob("rejected_*.npz")))]
    if nuisance is not None:
        nuisance.mkdir(parents=True, exist_ok=True)
        n_nui = [len(list(nuisance.glob("nuisance_*.npz")))]
    # every evaluator masks padded rows (active) and RTs only the rows it can
    # keep; each compiles at its first call
    fns = {"cold": jax.jit(pipe._make_batch_eval("cold"))}
    if anchors is not None:
        fns["warm"] = jax.jit(pipe._make_batch_eval("warm"))
    pending = [kzz_ref is not None]                  # the startup check, once
    if pending[0]:
        theta_of = jax.jit(jax.vmap(pipe.theta_from_u))
        i_kzz = list(pipe.names).index("lnKzz")

    def loglike(Z):
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        out = np.empty(len(Z))
        for i in range(0, len(Z), n_batch):
            Zc = Z[i:i + n_batch]
            m = len(Zc)
            if m < n_batch:
                Zc = np.concatenate([Zc, np.repeat(Zc[-1:], n_batch - m, axis=0)])
            Uj = jnp.asarray(np.log(Zc) - np.log1p(-Zc), pipe.dtype)
            # padded rows and T-P-invalid rows are masked: an out-of-window draw
            # is rejected whatever its chemistry, so its lane takes no step and no
            # RT. Not bitwise for the kept rows: fewer jobs on the lane queue
            # shift the shared refresh cadence.
            tp_in = np.asarray(jax.device_get(tp_ok(Uj)), bool)
            act = jnp.asarray((np.arange(n_batch) < m) & tp_in)
            if prof is not None:
                prof.batch_begin()
            t0 = time.perf_counter()
            warm = anchors is not None and len(anchors) > 0
            if warm:
                idx, dist = anchors.nearest(Zc[:, :k])
                Yw, rw = anchors.columns(idx)
                L, Ynew, rnew, st = fns["warm"](Uj, jnp.asarray(Yw, pipe.dtype),
                                                jnp.asarray(rw, pipe.dtype), act)
            else:
                L, Ynew, rnew, st = fns["cold"](Uj, Y0, refs0, act)
            L = np.asarray(jax.device_get(L), np.float64)[:m]
            acc = np.asarray(jax.device_get(st.acc), np.int64)[:m]
            conv = np.asarray(jax.device_get(st.conv_ok), bool)[:m]
            tp_out = ~tp_in[:m]
            # solver state first: the warm evaluator already floors L on a
            # stalled or exhausted exit, which is not a non-finite forward. The
            # solver's verdict decides: a solve certified at cap or cap + 1 is
            # kept (the runner stops on accept_count > count_max, convergence
            # first)
            exhausted = ~tp_out & ~conv & (acc >= count_max)
            stalled = ~tp_out & ~exhausted & ~conv
            nonfinite = (~tp_out & ~exhausted & conv
                         & (~np.isfinite(L) | (L <= P.REJECT_BELOW)))
            dead = tp_out | exhausted | stalled | nonfinite
            out[i:i + m] = np.where(dead, -np.inf, L)
            if pending[0]:
                th = np.asarray(jax.device_get(theta_of(Uj)), np.float64)[:m]
                counts = {"T-P out": tp_out.sum(), "exhausted": exhausted.sum(),
                          "stalled": stalled.sum(), "non-finite": nonfinite.sum()}
                for line in startup_check(np.log10(kzz_ref) + th[:, i_kzz] / np.log(10.0),
                                          ~tp_out, ~dead, counts):
                    log.info(line)
                pending[0] = False
            if batch_log is not None:
                with open(batch_log, "a") as fh:
                    fh.write(json.dumps({"t": round(time.perf_counter() - t0, 2),
                                         "mode": "warm" if warm else "cold",
                                         "acc": acc.tolist()}) + "\n")
            if n_rej is not None and dead.any():
                cls = np.where(tp_out, 0, np.where(exhausted, 1, np.where(stalled, 2, 3)))
                f = rejects / f"rejected_{n_rej[0]:05d}.npz"
                tmp = f.with_name(f".{f.name}.part")
                with open(tmp, "wb") as fh:
                    np.savez(fh, z=Zc[:m][dead], cls=cls[dead].astype(np.int8),
                             acc=acc[dead].astype(np.int64), warm=np.int8(warm))
                os.replace(tmp, f)
                n_rej[0] += 1
            if n_nui is not None and not dead.all():
                stats = np.asarray(jax.device_get(st.nuis), np.float64)[:m]
                f = nuisance / f"nuisance_{n_nui[0]:05d}.npz"
                tmp = f.with_name(f".{f.name}.part")
                with open(tmp, "wb") as fh:
                    np.savez(fh, z=Zc[:m][~dead], log_l=out[i:i + m][~dead],
                             stats=stats[~dead])
                os.replace(tmp, f)
                n_nui[0] += 1
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


def nuisance_posterior(pipe, folder: Path, queries, rng) -> list:
    """The integrated nuisances' posterior, one conditional draw per row of each
    query ``(z, log_l)`` (pipeline.make_nuisance_marginal's ``draw``), from the
    group statistics logged with the evaluation that gave the point its log L;
    NaN on a rejected row (zero weight). A point logged twice (a batch redrawn
    after a resume) is matched by its log L too; a finite row with no match
    raises."""
    table = {}
    for f in sorted(folder.glob("nuisance_*.npz")):
        with np.load(f) as d:
            for zz, ll, ss in zip(d["z"], d["log_l"], d["stats"]):
                table.setdefault(zz.tobytes(), []).append((float(ll), ss))
    res = []
    for zq, lq in queries:
        zq = np.asarray(zq, np.float64)
        st = np.full((len(zq), 2 * len(pipe.nuisance.groups)), np.nan)
        for i, (zz, ll) in enumerate(zip(zq, np.asarray(lq, np.float64))):
            if not np.isfinite(ll):
                continue
            hit = [s for lg, s in table.get(zz.tobytes(), []) if lg == ll]
            if not hit:
                raise RuntimeError(f"posterior row {i} (log L {ll}) has no logged "
                                   f"nuisance statistics in {folder}")
            st[i] = hit[0]
        nu = np.full((len(zq), len(pipe.nuisance.names)), np.nan)
        ok = np.isfinite(st[:, 0])
        if ok.any():
            nu[ok] = pipe.nuisance.draw(st[ok], pipe.obs_sigma, rng)
        res.append(nu)
    return res


def make_sampler(loglike, n_dim: int, n_live: int, n_batch: int, seed: int,
                 filepath: str | None = None, resume: bool = False):
    """The stock nautilus sampler with run_nautilus's settings. The prior is the
    identity on the unit cube (the pipeline maps u = logit(z) -> theta); loglike
    takes a (n, n_dim) batch (vectorized)."""
    import nautilus
    return nautilus.Sampler(lambda z: z, loglike, n_dim=n_dim, n_live=n_live,
                            n_update=n_live // 2, enlarge_per_dim=ENLARGE_PER_DIM,
                            n_networks=N_NETWORKS, n_batch=n_batch, vectorized=True,
                            pass_dict=False, seed=seed, filepath=filepath, resume=resume)


def main(argv=None) -> None:
    t_start = time.time()
    ap = argparse.ArgumentParser(description="nautilus nested sampling of one run specification")
    ap.add_argument("spec", help="the run specification, runs/<case>/<name>.yaml")
    ap.add_argument("--resume", action="store_true",
                    help="continue the checkpoint in the run's output directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved settings and exit without importing jax")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    cfg = spec.cfg
    if args.dry_run:
        print(C.describe_config(cfg, spec.name))
        print(f"sampler: {spec.sampler}\nruntime: {spec.runtime}\nout_dir: {cfg.out_dir}")
        return
    export_runtime(spec.runtime)       # before the first jax / vulcan3 import
    C.export_network(cfg)
    lanes = int(cfg.cold_lanes)
    if spec.runtime.solver == "ffi":
        lanes *= FFI_BLOCKS_PER_SM
    cfg = replace(cfg, cold_lanes=lanes)
    out = cfg.out_dir
    ckpt = out / "nautilus_checkpoint.hdf5"
    resume = bool(args.resume)
    check_run_dir(out, resume)          # before the minutes of the pipeline build
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "run.log")
    n_live = int(spec.sampler.n_live)
    n_update = n_live // 2
    n_eff = int(spec.sampler.n_eff)
    n_batch = int(spec.sampler.n_batch or BATCH_PER_LANE * lanes)
    warm = bool(spec.sampler.warm_starts)
    log.info(f"spec={spec.path} out_dir={out}")
    log.info(f"nautilus: n_live={n_live} n_update={n_update} n_eff={n_eff} "
             f"n_batch={n_batch} (cold_lanes={lanes}) f_live={F_LIVE} "
             f"enlarge_per_dim={ENLARGE_PER_DIM} networks={N_NETWORKS}x relu (nautilus default) "
             f"discard_exploration={DISCARD_EXPLORATION} "
             f"seed={cfg.seed} starts={'anchored warm (growing)' if warm else 'cold'}")
    log.info(C.describe_config(cfg, spec.name))

    import jax
    import nautilus

    from retrieval_framework import pipeline as P
    log.info(f"jax backend={jax.default_backend()} devices={jax.devices()} "
             f"nautilus {nautilus.__version__} (stock, {nautilus.__file__})")
    log_hardware()

    t0 = time.perf_counter()
    pipe = P.build_pipeline(cfg)
    log.info(f"Built pipeline in {time.perf_counter() - t0:.1f}s | n_dim={pipe.n_dim}: {pipe.names}")
    obs_save = set_observations(cfg, pipe, P)

    # resume identity before anything in the run directory is written; the
    # start policy, the batch width (the lane queue's composition) and the
    # sampler settings (nautilus restores none of them) are part of the
    # target, so a resume cannot switch them
    digest_path = out / "target_digest.txt"
    tally_path = out / "nautilus_tally.json"
    want = (f"{pipe.target_digest}|{'warm' if warm else 'cold'}"
            f"|n_batch={n_batch}|n_live={n_live}|n_update={n_update}|f_live={F_LIVE}"
            f"|enlarge_per_dim={ENLARGE_PER_DIM}|networks={N_NETWORKS}xrelu|stock")
    if resume:
        have = digest_path.read_text().strip() if digest_path.exists() else ""
        if have != want:
            raise RuntimeError(f"nautilus resume refused: {ckpt.name} belongs to a different "
                               f"target ({have or '<absent>'} vs this run's {want}). Restore "
                               "the original target and settings or use a fresh output "
                               "directory.")
    write_run_identity(cfg, pipe, spec, obs_save)
    shutil.copy(spec.path, out / "spec.yaml")       # the run's own record of its spec
    if spec.base is not None:
        shutil.copy(spec.base, out / "spec_base.yaml")
    digest_path.write_text(want + "\n")

    tally = json.loads(tally_path.read_text()) if (resume and tally_path.exists()) else {}
    anchors = Anchors(out / "anchors", int(pipe.n_chem_tp)) if warm else None
    if anchors is not None:
        log.info(f"nautilus: {len(anchors)} anchor column(s) loaded")
    # a fresh run's first chunk is its startup check (a resumed run passed it)
    from retrieval_framework.nautilus_report import _kzz_at_ref
    kzz_ref = None if resume else _kzz_at_ref({"vulcan_cfg_name": cfg.vulcan_cfg_name,
                                               "cfg_overrides": cfg.cfg_overrides})
    like = make_loglike(pipe, n_batch, tally, anchors, out / "batches.jsonl",
                        BatchProfiler(), rejects=out / "rejected",
                        nuisance=out / "nuisance" if pipe.nuisance is not None else None,
                        kzz_ref=kzz_ref)

    def loglike_saved(Z):
        L = like(Z)
        tmp = tally_path.with_name(f".{tally_path.name}.part")
        tmp.write_text(json.dumps(tally, indent=2) + "\n")
        os.replace(tmp, tally_path)
        return L

    sampler = make_sampler(loglike_saved, pipe.n_dim, n_live, n_batch, int(cfg.seed),
                           filepath=str(ckpt), resume=resume)
    budget = float(cfg.walltime_seconds)
    timeout = max(budget - (time.time() - t_start), 60.0) if budget > 0 else float("inf")
    log.info(f"nautilus: starting at n_like={sampler.n_like}; timeout {timeout / 3600:.2f} h")
    t0 = time.perf_counter()
    done = sampler.run(f_live=F_LIVE, n_eff=n_eff, discard_exploration=DISCARD_EXPLORATION,
                       timeout=timeout, verbose=True)
    log.info(f"nautilus: {'finished' if done else 'stopped at the wall budget'} after "
             f"{(time.perf_counter() - t0) / 3600:.2f} h | n_like={sampler.n_like} "
             f"| log_z={sampler.log_z:.3f} | n_eff={sampler.n_eff:.0f} | tally={tally}")
    if not done:
        log.warning("nautilus did not finish: resubmit with --resume (the checkpoint is "
                    f"{ckpt}).")
        return

    z, log_w, log_l = sampler.posterior()
    z_eq, _, log_l_eq = sampler.posterior(equal_weight=True)
    extra = {}
    if pipe.nuisance is not None:
        nu, nu_eq = nuisance_posterior(pipe, out / "nuisance", [(z, log_l), (z_eq, log_l_eq)],
                                       np.random.default_rng(int(cfg.seed) + 2))
        extra = dict(nuisance_names=np.asarray(pipe.nuisance.names, dtype="<U64"),
                     nuisance=nu, samples_nuisance=nu_eq)
    to_theta = jax.jit(jax.vmap(pipe.theta_from_u))
    theta = np.asarray(to_theta(np.log(z) - np.log1p(-z)), np.float64)
    theta_eq = np.asarray(to_theta(np.log(z_eq) - np.log1p(-z_eq)), np.float64)
    P.save_npz(out / "nautilus_posterior.npz",
               param_names=np.asarray(pipe.names, dtype="<U64"),
               target_digest=np.asarray(want),
               theta=theta, log_w=log_w, log_l=log_l, samples=theta_eq,
               u=np.log(z) - np.log1p(-z),     # the rows in u-space: an evaluator's input
               logZ_box=np.asarray(sampler.log_z), n_like=np.asarray(sampler.n_like),
               **extra)
    summary = {
        "logZ_box": float(sampler.log_z), "n_like": int(sampler.n_like),
        "n_eff": float(sampler.n_eff), "n_live": n_live, "n_batch": n_batch,
        "n_update": n_update, "f_live": F_LIVE, "enlarge_per_dim": ENLARGE_PER_DIM,
        "networks": f"{N_NETWORKS}x relu (nautilus default)",
        "discard_exploration": DISCARD_EXPLORATION,
        "nautilus": f"{nautilus.__version__} (stock)",
        "starts": "anchored warm (growing)" if warm else "cold",
        "rejections_and_starts": tally, "spec": spec.name, "seed": int(cfg.seed),
        "chem_rtol": pipe.fwd.chem_rtol,
        "evidence_note": "zero-filled box evidence (log L = -inf on every rejection)"
                         + ("; APPROXIMATE: warm starts move the likelihood and which draws "
                            "are rejected (warm_starts: false for an evidence claim)"
                            if warm else ""),
    }
    (out / "nautilus_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info(f"wrote {out / 'nautilus_posterior.npz'} and nautilus_summary.json")
    from retrieval_framework.nautilus_report import write_certificate
    log.info(f"wrote {write_certificate(out)}")


if __name__ == "__main__":
    main()
