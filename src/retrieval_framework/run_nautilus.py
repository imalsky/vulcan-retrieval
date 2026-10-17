"""run_nautilus.py -- nested sampling (nautilus) on the retrieval's likelihood.

Same case, preset, overrides and observations as run_smc; a different sampler.
Every likelihood call goes through the pipeline's batched chemistry + RT on the
lane queue (``lanes``, below) with the SMC init's rejection rule: a
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

Warm starts (default; NAUTILUS_WARM=0 disables them): every certified column
becomes an anchor keyed by its unit-cube chemistry + T-P coordinates, and each
later solve continues from its nearest anchor (the warm map at the cold
count_max); batches before the first certified column run cold. The certified
state and which draws are rejected then depend on the evaluation order at the
convergence tolerance, so the posterior and evidence are
approximate and the evidence is not an SMC run's logZ_box. An evidence claim
uses NAUTILUS_WARM=0, the cold primal map on the lane queue (a draw's column
depends on its batch at the convergence scale), which compares with SMC's
logZ_box only as stated above.

Anchors are written inside the likelihood call, before nautilus checkpoints
the batch: a job killed in between resumes by redrawing that batch, which then
starts from its own columns (and the tally counts it twice). The set is never
pruned: ~44 KB per certified column, on disk and in memory.

nautilus samples the unit cube; z -> u = logit(z) is the retrieval's own
u-space (pipeline.make_uspace), so theta = theta_from_u(u) follows the box
prior exactly and the posterior is stored in theta.

Output directory: ``<out_dir>_nautilus`` (default runs/<case>/data/<preset>_nautilus;
gpu_nautilus under the PBS):
nautilus_checkpoint.hdf5 (rewritten every iteration), anchors/ (one npz of
certified columns per batch, reloaded on resume), nautilus_tally.json
(rejection and warm-start counts, summed over jobs), and when the run finishes
nautilus_posterior.npz + nautilus_summary.json. ``SMC_RESUME=1`` (the PBS's
RESUME=1) continues the checkpoint; without it an existing checkpoint raises.
The run stops at ``cfg.walltime_seconds`` (the preset's governor) and exits 0
unfinished; resubmit with RESUME=1.

    python -m retrieval_framework.run_nautilus <run_dir>
Env: NAUTILUS_N_LIVE (default N_LIVE; n_update follows at half of it),
NAUTILUS_N_EFF (default N_EFF), NAUTILUS_N_BATCH (default BATCH_PER_LANE x the
lane count, so a lane whose column certifies refills from the same batch; with
cold_lanes = 0, one lockstep batch of config_schema.device_lane_count() columns),
NAUTILUS_WARM (default 1), NAUTILUS_PRECOMPILE (default 1: compile the warm
evaluators -- the first pass at n_batch and the retry pass at every
RETRY_PRECOMPILE bucket -- on a host thread while the cold first batch runs;
0 compiles each one at its first use), and the profiling hook (BatchProfiler, off unless
VULCAN_PROFILE_BATCH is set): VULCAN_PROFILE_BATCH, VULCAN_PROFILE_EXIT_AFTER,
VULCAN_NVTX_LIB.

Sampler settings (the constants below): 2 ReLU + 2 SiLU networks, n_update =
n_live / 2, exploration until f_live = F_LIVE, bounds enlarged by
ENLARGE_PER_DIM, and sampling-phase batches spread over several shells
(``spread_batch``). The SiLU networks and ``spread_batch`` exist only in the
maintainer's nautilus fork; validate_env refuses stock nautilus.

Every likelihood chunk appends one line to ``batches.jsonl``: its wall time,
start mode and the accepted step count of each column.

Chemistry retry (``cfg.chem_retry``, off by default; make_loglike): a draw whose
first solve does not certify under VULCAN's own certificate within count_max
(exhausted or stalled, T-P in window) is re-solved once from the same start by a
second compiled program at ``chem_retry_rtol``, capped at
``chem_retry_count_max``, and kept only if that solve certifies. The rejection
classes then describe each draw's final solve; the tally adds ``retried`` and
``retry_*``, a chunk's batches.jsonl line adds ``retry`` (its rows, widths,
step counts and wall time; ``t`` is both passes, ``acc`` the first), and every
anchor file records the ``rtol`` that certified each column. Off, every path
below is the stock one.

Padding costs no solve: a chunk shorter than n_batch and the retry pass's
bucket (``retry_width``) are filled with copies of their last row that the
evaluator masks (``active``): their lanes take no chemistry step and no RT.
Each pass runs the RT only on the rows whose log L it can keep (T-P in window,
certified under the cap; ``rt_kept_only``), so a row the first pass leaves
uncertified is RT'd once, after its retry.

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
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework.run_smc import (
    log_hardware, make_config, set_observations, setup_logging, write_run_identity)

log = logging.getLogger("retrieval")

N_LIVE = 1000       # live points
N_EFF = 10_000      # effective posterior samples at which nautilus stops
F_LIVE = 0.03       # exploration ends when the live set holds this evidence share
ENLARGE_PER_DIM = 1.3
NETWORKS = [{}, {"activation": "silu"}, {}, {"activation": "silu"}]
BATCH_PER_LANE = 3  # points per lane per batch: certified lanes refill in-batch
# ffi one-buffer blocks resident per SM at ni 89 on compute capability 9.0:
# ~63 KB of shared memory each in the SM's 228 KB (VULCAN-JAX
# block_thomas_cuda.cu, bt_choose picks that variant at this width).
FFI_BLOCKS_PER_SM = 3
# chem_retry: the second pass runs on a width rounded up to a multiple of
# RETRY_BUCKET, padded with masked copies of its last row (no step, no RT). The
# retry program compiles once per width, and the buckets up to
# RETRY_PRECOMPILE[-1] are compiled ahead (NAUTILUS_PRECOMPILE). 132 is the
# GH200's SM count: the compaction rungs (264, 132) are multiples of it, and
# W39b baseline v2 (job 93269, 59 batches) retried 32-586 rows per batch, so
# five buckets cover every batch it ran; a wider pass compiles its own bucket.
RETRY_BUCKET = 132
RETRY_PRECOMPILE = (132, 264, 396, 528, 660)


def retry_width(n: int, bucket: int | None = None) -> int:
    """The padded second-pass width (bucket) of ``n`` retried rows."""
    b = int(RETRY_BUCKET if bucket is None else bucket)
    return int(-(-int(n) // b) * b)


class _Programs:
    """The likelihood evaluators, compiled once per (name, width) and called as
    compiled executables, (U, Y, refs, active) -> outputs. ``start(keys)``
    compiles ``keys`` on one host thread (the GPU keeps running the batch in
    flight); ``call`` waits for a compile in progress, takes a queued one
    out of the queue and compiles it on the spot (it never waits behind
    others), and falls back to compiling in the caller's thread if the
    background compile failed."""

    def __init__(self, fns: dict, n_dim, y_shape: tuple, dtype):
        self.fns, self.n_dim, self.y_shape, self.dtype = fns, n_dim, tuple(y_shape), dtype
        self.exe, self.pending, self.pool = {}, {}, None

    def _avals(self, w):
        import jax
        s = jax.ShapeDtypeStruct
        return (s((w, int(self.n_dim)), self.dtype), s((w,) + self.y_shape, self.dtype),
                s((w, 2), self.dtype), s((w,), bool))

    def _compile(self, key, avals):
        t0 = time.perf_counter()
        exe = self.fns[key[0]].lower(*avals).compile()
        log.info(f"nautilus: compiled the {key[0]} evaluator at width {key[1]} in "
                 f"{time.perf_counter() - t0:.0f} s")
        return exe

    def start(self, keys):
        from concurrent.futures import ThreadPoolExecutor
        keys = [k for k in keys if k not in self.exe and k not in self.pending]
        if not keys:
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="precompile")
        log.info(f"nautilus: precompiling {keys} on a host thread")
        for k in keys:
            self.pending[k] = self.pool.submit(self._compile, k, self._avals(k[1]))

    def call(self, name, w, *args):
        import jax
        key = (name, int(w))
        exe = self.exe.get(key)
        if exe is None:
            avals = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args)
            fut = self.pending.pop(key, None)
            if fut is not None and fut.cancel():    # still queued: compile it here
                fut = None
            t0 = time.perf_counter()
            try:
                exe = fut.result() if fut is not None else self._compile(key, avals)
                if fut is not None and time.perf_counter() - t0 > 1.0:
                    log.info(f"nautilus: waited {time.perf_counter() - t0:.0f} s for the "
                             f"background compile of {key}")
            except Exception as e:  # noqa: BLE001  a failed background compile
                if fut is None:
                    raise
                log.warning(f"nautilus: background compile of {key} failed ({e!r}); "
                            "compiling it now")
                exe = self._compile(key, avals)
            self.exe[key] = exe
        return exe(*args)


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

    def add(self, z, Y, refs, rtol=None):
        """Append certified columns as one new file. ``rtol`` (chem_retry runs):
        per column, the step tolerance of the solve that certified it, stored
        as the file's ``rtol`` (a reload reads z / Y / refs only)."""
        if len(z):
            f = self.folder / f"anchors_{len(self.parts):05d}.npz"
            tmp = f.with_name(f".{f.name}.part")
            with open(tmp, "wb") as fh:
                if rtol is None:
                    np.savez(fh, z=z, Y=Y, refs=refs)
                else:
                    np.savez(fh, z=z, Y=Y, refs=refs, rtol=np.asarray(rtol, np.float64))
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
    unprofiled after the window.
    VULCAN_PROFILE_EXIT_AFTER=n: exit 0 when batch n+1 would start (smoke tests).
    VULCAN_NVTX_LIB: a shared library exporting vp_push(const char*) / vp_pop()
    (an NVTX3 shim) for the phase ranges; without it, or with the `nvtx` package
    absent, no ranges (the window still works). Ranges, every batch:
    batch<k>/anchors (host nearest-anchor lookup), batch<k>/evaluator (ONE XLA
    program: the chemistry lane queue, then the RT; under nsys XLA's own NVTX
    thunk ranges split it by op), batch<k>/retry (cfg.chem_retry only: the
    second-pass solve of the chunk's uncertified rows, inside batch k's window),
    batch<k>/post (rejection classes, anchors), nautilus/update (the sampler's
    work between two batches)."""

    def __init__(self):
        self.k = int(os.environ.get("VULCAN_PROFILE_BATCH", "0") or 0)
        self.exit_after = int(os.environ.get("VULCAN_PROFILE_EXIT_AFTER", "0") or 0)
        self.n = 0
        self.active = False
        self.t_batch = None
        self._push = self._pop = None
        self._open = []
        if self.k <= 0 and self.exit_after <= 0:
            return
        if self.k > 0:
            self._rt = self._cudart()
            atexit.register(self.stop)
            lib = os.environ.get("VULCAN_NVTX_LIB", "").strip()
            try:
                if lib:
                    nv = ctypes.CDLL(lib)
                    nv.vp_push.argtypes = [ctypes.c_char_p]
                    self._push = lambda s: nv.vp_push(s.encode())
                    self._pop = nv.vp_pop
                else:
                    import nvtx  # optional package
                    self._push, self._pop = nvtx.push_range, nvtx.pop_range
            except (OSError, ImportError, AttributeError) as e:
                log.warning(f"profile: no NVTX ranges ({e})")
        log.info(f"profile: capture batch {self.k or '-'} (cudaProfilerStart/Stop), exit after "
                 f"batch {self.exit_after or '-'}, NVTX {'on' if self._push else 'off'}")

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

    def push(self, name):
        if self._push is not None:
            self._push(name)
            self._open.append(name)

    def pop(self):
        if self._pop is not None and self._open:
            self._pop()
            self._open.pop()

    def stop(self):
        if self.active:
            rc = int(self._rt.cudaProfilerStop())
            self.active = False
            log.info(f"profile: cudaProfilerStop after batch {self.k} (rc {rc})")

    def batch_begin(self):
        """Called before every evaluator call; device work of the previous batch
        is complete here (its results were fetched to the host)."""
        self.pop()                                  # nautilus/update, if open
        now = time.perf_counter()
        if self.t_batch is not None and (self.k or self.exit_after):
            log.info(f"profile: batch {self.n} period {now - self.t_batch:.2f} s "
                     "(evaluator + host + nautilus update)")
        self.t_batch = now
        self.n += 1
        if self.k and self.n == self.k + 1:
            self.stop()
        if self.exit_after and self.n == self.exit_after + 1:
            self.stop()
            log.info(f"profile: VULCAN_PROFILE_EXIT_AFTER={self.exit_after}: exiting 0")
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        if self.k and self.n == self.k:
            rc = int(self._rt.cudaProfilerStart())
            self.active = True
            log.info(f"profile: cudaProfilerStart before batch {self.k} (rc {rc})")


def make_loglike(pipe, n_batch: int, tally: dict, anchors: Anchors | None = None,
                 batch_log: Path | None = None, prof: BatchProfiler | None = None,
                 precompile: bool = False):
    """Batched log-likelihood of unit-cube points (n, n_dim) -> (n,), -inf on
    every rejection class. Calls a compiled evaluator at one fixed width
    ``n_batch`` (a short last chunk is padded with masked copies of its last
    row, which take no chemistry step and no RT) and adds this call's counts
    to ``tally``. ``precompile``: from the first chunk on, compile the warm
    evaluators (n_batch, and the retry buckets RETRY_PRECOMPILE) on a host
    thread while the first (cold) chunk runs. With ``anchors``, a chunk starts
    from the nearest certified column (warm, cold count_max) once one exists,
    and every certified column it returns becomes an anchor. With
    ``batch_log``, each chunk appends its wall time, mode and per-column
    accepted step counts there.

    With ``pipe.chem_retry`` (cfg.chem_retry), the rows of a chunk whose first
    solve does not certify within the first cap (exhausted or stalled, T-P in
    window) are re-solved once from the SAME start -- the anchor column and
    references the first solve started from, or the cold two-stage map on a
    cold chunk -- on the retry program, in one second lane-queue call of
    ``retry_width`` rows (masked copies of the last retried row pad it). A
    retried row keeps its second solve, judged at the retry cap: certified ->
    kept (and an anchor), else rejected in the class of that solve."""
    import jax
    import jax.numpy as jnp

    from retrieval_framework import pipeline as P

    tp_ok = jax.jit(jax.vmap(lambda u: pipe.tp_valid(pipe.theta_from_u(u))))
    Y0, refs0 = P._blank_state(pipe, n_batch)
    count_max = int(pipe.fwd.chem.count_max)
    k = int(pipe.n_chem_tp)
    retry = getattr(pipe, "chem_retry", None)
    # every evaluator masks padded rows (active) and RTs only the rows it can keep
    fns = {"cold": jax.jit(pipe._make_batch_eval("cold", False, diag=True,
                                                 rt_kept_only=True))}
    if anchors is not None:
        fns["warm"] = jax.jit(pipe._make_batch_eval("warm", False, mutation_cap=False,
                                                    rt_kept_only=True))
    if retry is not None:
        fns["retry_cold"] = jax.jit(pipe._make_batch_eval("cold", False, diag=True,
                                                          retry=True, rt_kept_only=True))
        if anchors is not None:
            fns["retry_warm"] = jax.jit(pipe._make_batch_eval(
                "warm", False, mutation_cap=False, retry=True, rt_kept_only=True))
        count_max_retry = int(retry.count_max)
        rtol_first = np.nan if retry.rtol_first is None else float(retry.rtol_first)
        rtol_retry = float(retry.rtol)
    progs = _Programs(fns, getattr(pipe, "n_dim", None), Y0.shape[1:], pipe.dtype)
    ahead = []
    if precompile and anchors is not None:
        ahead = [("warm", n_batch)] + (
            [("retry_warm", b) for b in sorted(RETRY_PRECOMPILE, reverse=True)]
            if retry is not None else [])

    def second_pass(rows, warm, Uj, Yw, rw):
        """The retry solve of chunk rows ``rows`` from their first-pass starts:
        host (L, Y, refs, acc, conv) of those rows, the padded width, wall s."""
        w = retry_width(len(rows))
        sel = np.concatenate([rows, np.repeat(rows[-1:], w - len(rows))])
        act = jnp.asarray(np.arange(w) < len(rows))
        t1 = time.perf_counter()
        if warm:
            L2, Y2, r2, st2 = progs.call("retry_warm", w, Uj[sel],
                                         jnp.asarray(Yw[sel], pipe.dtype),
                                         jnp.asarray(rw[sel], pipe.dtype), act)
            acc2, conv2 = st2.acc, st2.conv_ok
        else:
            Yb, rb = P._blank_state(pipe, w)
            L2, Y2, r2, cd2 = progs.call("retry_cold", w, Uj[sel], Yb, rb, act)
            acc2, conv2 = cd2.accept_count, cd2.conv_normal
        n = len(rows)
        out2 = (np.asarray(jax.device_get(L2), np.float64)[:n],
                np.array(jax.device_get(Y2), np.float64)[:n],
                np.array(jax.device_get(r2), np.float64)[:n],
                np.asarray(jax.device_get(acc2), np.int64)[:n],
                np.asarray(jax.device_get(conv2), bool)[:n])
        return out2 + (w, time.perf_counter() - t1)

    def loglike(Z):
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        out = np.empty(len(Z))
        for i in range(0, len(Z), n_batch):
            Zc = Z[i:i + n_batch]
            m = len(Zc)
            if m < n_batch:
                Zc = np.concatenate([Zc, np.repeat(Zc[-1:], n_batch - m, axis=0)])
            Uj = jnp.asarray(np.log(Zc) - np.log1p(-Zc), pipe.dtype)
            act = jnp.asarray(np.arange(n_batch) < m)       # padded rows: masked
            if ahead:
                # compile the warm programs on a host thread from the first chunk
                # on: an evaluator call returns only when its queue loop has
                # finished (the loop's predicate is read on the host each tick)
                progs.start(ahead)
                ahead.clear()
            if prof is not None:
                prof.batch_begin()
                prof.push(f"batch{prof.n}/anchors")
            t0 = time.perf_counter()
            warm = anchors is not None and len(anchors) > 0
            Yw = rw = None
            if warm:
                idx, dist = anchors.nearest(Zc[:, :k])
                Yw, rw = anchors.columns(idx)
                if prof is not None:
                    prof.pop()
                    prof.push(f"batch{prof.n}/evaluator")
                L, Ynew, rnew, st = progs.call("warm", n_batch, Uj,
                                               jnp.asarray(Yw, pipe.dtype),
                                               jnp.asarray(rw, pipe.dtype), act)
                acc, conv = st.acc, st.conv_ok
            else:
                if prof is not None:
                    prof.pop()
                    prof.push(f"batch{prof.n}/evaluator")
                L, Ynew, rnew, cd = progs.call("cold", n_batch, Uj, Y0, refs0, act)
                acc, conv = cd.accept_count, cd.conv_normal
            L = np.asarray(jax.device_get(L), np.float64)[:m]
            acc = np.asarray(jax.device_get(acc), np.int64)[:m]
            conv = np.asarray(jax.device_get(conv), bool)[:m]
            if prof is not None:
                prof.pop()
                prof.push(f"batch{prof.n}/post")
            if batch_log is not None and retry is None:
                with open(batch_log, "a") as fh:
                    fh.write(json.dumps({"t": round(time.perf_counter() - t0, 2),
                                         "mode": "warm" if warm else "cold",
                                         "acc": acc.tolist()}) + "\n")
            t_first = time.perf_counter() - t0
            tp_out = ~np.asarray(jax.device_get(tp_ok(Uj)), bool)[:m]
            cap = count_max
            if retry is not None:
                # second pass: every in-window row the first solve left
                # uncertified (exhausted or stalled), from the same start
                acc_first = acc
                L, acc, conv = L.copy(), acc.copy(), conv.copy()
                Yh = np.array(jax.device_get(Ynew), np.float64)[:m]
                rh = np.array(jax.device_get(rnew), np.float64)[:m]
                retried = np.zeros(m, bool)
                rows = np.flatnonzero(~tp_out & ((acc >= count_max) | ~conv))
                w, t2, acc2, conv2 = 0, 0.0, np.empty(0, np.int64), np.empty(0, bool)
                if rows.size:
                    if prof is not None:     # the second pass gets its own range
                        prof.pop()
                        prof.push(f"batch{prof.n}/retry")
                    L2, Y2, r2, acc2, conv2, w, t2 = second_pass(rows, warm, Uj, Yw, rw)
                    if prof is not None:
                        prof.pop()
                        prof.push(f"batch{prof.n}/post")
                    L[rows], Yh[rows], rh[rows], acc[rows], conv[rows] = L2, Y2, r2, acc2, conv2
                    retried[rows] = True
                cap = np.where(retried, count_max_retry, count_max)
                if batch_log is not None:
                    with open(batch_log, "a") as fh:
                        fh.write(json.dumps({
                            "t": round(t_first + t2, 2), "t_first": round(t_first, 2),
                            "mode": "warm" if warm else "cold", "acc": acc_first.tolist(),
                            "retry": {"t": round(t2, 2), "width": int(w),
                                      "rows": rows.tolist(), "acc": acc2.tolist(),
                                      "conv": conv2.tolist(), "rtol": rtol_retry,
                                      "count_max": count_max_retry}}) + "\n")
            # solver state first: the warm evaluator already floors L on a
            # stalled or exhausted exit, which is not a non-finite forward
            exhausted = ~tp_out & (acc >= cap)
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
            if retry is not None:
                # the final classes above already count each retried row once,
                # by its second solve; these break the second pass out
                kept = retried & ~dead
                for key, v in (("retried", retried.sum()), ("retry_certified", kept.sum()),
                               ("retry_exhausted", (retried & exhausted).sum()),
                               ("retry_stalled", (retried & stalled).sum()),
                               ("retry_nonfinite", (retried & nonfinite).sum()),
                               ("retry_steps_certified", acc[kept].sum()),
                               ("retry_width_sum", w),
                               (f"{mode}_retried", retried.sum()),
                               (f"{mode}_retry_certified", kept.sum())):
                    tally[key] = tally.get(key, 0) + int(v)
                tally["retry_wall_s"] = round(tally.get("retry_wall_s", 0.0) + float(t2), 2)
            if warm:
                tally["warm_anchor_dist_sum"] = (tally.get("warm_anchor_dist_sum", 0.0)
                                                 + float(dist[:m].sum()))
            if anchors is not None:
                alive = ~dead
                if retry is None:
                    anchors.add(Zc[:m][alive][:, :k],
                                np.asarray(jax.device_get(Ynew), np.float64)[:m][alive],
                                np.asarray(jax.device_get(rnew), np.float64)[:m][alive])
                else:
                    anchors.add(Zc[:m][alive][:, :k], Yh[alive], rh[alive],
                                rtol=np.where(retried, rtol_retry, rtol_first)[alive])
            if prof is not None:
                prof.pop()
        return out

    loglike.programs = progs    # the compiled evaluators (tests, diagnostics)
    return loglike


def main() -> None:
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=".",
                    help="retrieval case directory containing case.py (default: cwd)")
    args = ap.parse_args()

    cfg, preset = make_config(Path(args.run_dir))
    lanes = int(cfg.cold_lanes)
    if os.environ.get("VULCAN3_SOLVER") == "ffi":   # set by make_config
        lanes *= FFI_BLOCKS_PER_SM
    cfg = replace(cfg, cold_lanes=lanes,
                  out_dir=cfg.out_dir.parent / f"{cfg.out_dir.name}_nautilus")
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "run.log")
    n_live = int(os.environ.get("NAUTILUS_N_LIVE", str(N_LIVE)))
    n_update = n_live // 2
    n_eff = int(os.environ.get("NAUTILUS_N_EFF", str(N_EFF)))
    n_batch = int(os.environ.get(
        "NAUTILUS_N_BATCH",
        str(BATCH_PER_LANE * lanes if lanes > 0 else C.device_lane_count())))
    warm = os.environ.get("NAUTILUS_WARM", "1").strip() != "0"
    log.info(f"run_dir={Path(args.run_dir).resolve()} preset={preset} out_dir={out}")
    log.info(f"nautilus: n_live={n_live} n_update={n_update} n_eff={n_eff} "
             f"n_batch={n_batch} (cold_lanes={lanes}) f_live={F_LIVE} "
             f"enlarge_per_dim={ENLARGE_PER_DIM} networks={NETWORKS} spread_batch "
             f"seed={cfg.seed} starts={'anchored warm (growing)' if warm else 'cold'}")
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
    if pipe.chem_retry is not None:
        r = pipe.chem_retry
        log.info(f"nautilus: chem_retry ON: a draw uncertified at rtol {r.rtol_first} within "
                 f"{r.count_max_first} accepted steps is re-solved once from the same start "
                 f"at rtol {r.rtol} within {r.count_max} (second pass on a multiple of "
                 f"{RETRY_BUCKET} rows, padding masked); still uncertified -> rejected")

    # resume identity before anything in the run directory is written; the
    # start policy, the batch width (the lane queue's composition) and the
    # sampler settings (nautilus restores none of them) are part of the
    # target, so a resume cannot switch them
    ckpt = out / "nautilus_checkpoint.hdf5"
    digest_path = out / "target_digest.txt"
    tally_path = out / "nautilus_tally.json"
    want = (f"{pipe.target_digest}|{'warm' if warm else 'cold'}"
            f"|n_batch={n_batch}|n_live={n_live}|n_update={n_update}|f_live={F_LIVE}"
            f"|enlarge_per_dim={ENLARGE_PER_DIM}|networks={json.dumps(NETWORKS)}"
            f"|spread_batch")
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
    prof = BatchProfiler()
    precompile = os.environ.get("NAUTILUS_PRECOMPILE", "1").strip() != "0"
    like = make_loglike(pipe, n_batch, tally, anchors, out / "batches.jsonl", prof,
                        precompile=precompile)

    def loglike_saved(Z):
        L = like(Z)
        tmp = tally_path.with_name(f".{tally_path.name}.part")
        tmp.write_text(json.dumps(tally, indent=2) + "\n")
        os.replace(tmp, tally_path)
        prof.push("nautilus/update")    # closed when the next batch begins
        return L

    sampler = nautilus.Sampler(lambda z: z, loglike_saved, n_dim=pipe.n_dim, n_live=n_live,
                               n_update=n_update, enlarge_per_dim=ENLARGE_PER_DIM,
                               n_networks=len(NETWORKS), neural_network_kwargs=NETWORKS,
                               n_batch=n_batch, vectorized=True, pass_dict=False,
                               seed=int(cfg.seed), filepath=str(ckpt), resume=resume,
                               spread_batch=True)
    budget = float(cfg.walltime_seconds)
    timeout = max(budget - (time.time() - t_start), 60.0) if budget > 0 else float("inf")
    log.info(f"nautilus: starting at n_like={sampler.n_like}; timeout {timeout / 3600:.2f} h")
    t0 = time.perf_counter()
    done = sampler.run(f_live=F_LIVE, n_eff=n_eff, discard_exploration=True, timeout=timeout,
                       verbose=True)
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
        "n_update": n_update, "f_live": F_LIVE, "enlarge_per_dim": ENLARGE_PER_DIM,
        "networks": NETWORKS, "spread_batch": True,
        "starts": "anchored warm (growing)" if warm else "cold",
        "rejections_and_starts": tally, "preset": preset, "seed": int(cfg.seed),
        **({"chem_retry": vars(pipe.chem_retry)} if pipe.chem_retry is not None else {}),
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
