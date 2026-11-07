"""Nested sampling with stock nautilus-sampler on the retrieval's likelihood.

    python -m retrieval_framework.run_nautilus runs/<case>/<name>.yaml [--resume] [--dry-run]

Every likelihood call goes through the pipeline's batched chemistry and RT on the lane queue. A non-finite forward, a
T-P draw outside the window, a count_max-exhausted solve or a stall-certified exit gets log L = -inf, so the evidence is
a zero-filled box evidence, ``logZ_box``, which depends on the solver settings. With ``sampler.warm_starts`` each solve
continues from its nearest certified column, so the posterior and evidence are approximate; an evidence claim uses
``warm_starts: false``. docs/method.md describes the output files, the sampler settings, the startup check and the
lanes.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework.batch_profiler import BatchProfiler
from retrieval_framework.run_setup import (
    check_run_dir,
    export_runtime,
    load_spec,
    set_observations,
    setup_logging,
    write_run_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path
    from types import SimpleNamespace

    import nautilus
    from jax.typing import ArrayLike

__all__ = ["Anchors", "main", "make_loglike", "make_sampler", "nuisance_posterior", "startup_check"]

log = logging.getLogger("retrieval")

F_LIVE = 0.03  # exploration ends when the live set holds this evidence share
ENLARGE_PER_DIM = 1.3
N_NETWORKS = 4  # nautilus's default ReLU networks (hidden layers 100-50-20)
DISCARD_EXPLORATION = True
BATCH_PER_LANE = 3  # points per lane per batch: certified lanes refill in-batch
# ffi one-buffer blocks resident per SM at ni 89 on compute capability 9.0: 63 of 228 KB of shared memory each.
FFI_BLOCKS_PER_SM = 3
# In-window cold draws a log10 Kzz decade needs before none certifying refuses the
# run: 0 of 30 bounds that decade's certified fraction below 10% (95%).
STARTUP_MIN_DRAWS = 30
TALLY_FILE = "nautilus_tally.json"


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Write an npz under a temporary name and rename it, so a kill leaves no partial file."""
    tmp = path.with_name(f".{path.name}.part")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)


def startup_check(log10_kzz: ArrayLike, in_window: np.ndarray, kept: np.ndarray, counts: dict) -> list[str]:
    """Return the first chunk's verdict: one line per decade of ``log10_kzz`` with its in-window and kept draws.

    Parameters
    ----------
    log10_kzz : array-like [n]
        log10 Kzz of each draw at the reference pressure.
    in_window : array of bool [n]
        The draws whose T-P is in the window.
    kept : array of bool [n]
        The draws with a certified, finite log L.
    counts : dict
        The chunk's rejection classes, for the message.

    Returns
    -------
    list of str
        The lines for the log.

    Raises
    ------
    RuntimeError
        If nothing was kept, or a decade with ``STARTUP_MIN_DRAWS`` or more in-window draws kept none: the posterior
        there would be cut by the solver, not by the data.
    """
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
            f"({', '.join(f'{v} {c}' for c, v in counts.items())}); the run cannot start. " + " | ".join(lines)
        )
    if cut:
        raise RuntimeError(
            f"startup check: no in-window cold draw certified in log10 Kzz {', '.join(cut)} "
            f"(>= {STARTUP_MIN_DRAWS} draws each): the posterior there would be cut by the "
            "solver, not the data. Narrow prior_lnKzz to exclude it, or measure the "
            "count_max that certifies it before raising the cap. " + " | ".join(lines)
        )
    return lines


class Anchors:
    """The certified columns so far: unit-cube coordinates, the column and its reference composition.

    One npz per added batch in ``folder``; an existing folder is reloaded on resume. A file with two-wide references
    reads as s_o = 0.
    """

    N_REF = 3

    def __init__(self, folder: Path, n_key: int) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.z = np.empty((0, n_key))
        self.refs = np.empty((0, self.N_REF))
        self.parts: list[np.ndarray] = []
        self.part, self.row = np.empty(0, int), np.empty(0, int)
        for f in sorted(folder.glob("anchors_*.npz")):
            with np.load(f) as d:
                self._append(d["z"], d["Y"], d["refs"])

    def __len__(self) -> int:
        """Return the number of anchors."""
        return len(self.z)

    def _append(self, z: np.ndarray, Y: np.ndarray, refs: np.ndarray) -> None:
        """Append one batch in memory."""
        refs = np.asarray(refs, np.float64)
        if refs.shape[1] < self.N_REF:
            refs = np.pad(refs, ((0, 0), (0, self.N_REF - refs.shape[1])))
        self.part = np.concatenate([self.part, np.full(len(z), len(self.parts))])
        self.row = np.concatenate([self.row, np.arange(len(z))])
        self.parts.append(np.asarray(Y))
        self.z = np.concatenate([self.z, z])
        self.refs = np.concatenate([self.refs, refs])

    def add(self, z: np.ndarray, Y: np.ndarray, refs: np.ndarray) -> None:
        """Append certified columns as one new file.

        Parameters
        ----------
        z : numpy.ndarray [n, n_key]
            Unit-cube coordinates.
        Y : numpy.ndarray [n, nz, ni]
            The columns.
        refs : numpy.ndarray [n, 3]
            The reference compositions.
        """
        if len(z):
            _atomic_savez(self.folder / f"anchors_{len(self.parts):05d}.npz", z=z, Y=Y, refs=refs)
            self._append(z, Y, refs)

    def nearest(self, zq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return each query's nearest anchor, Euclidean in the unit cube.

        Parameters
        ----------
        zq : numpy.ndarray [n, n_key]
            The queries.

        Returns
        -------
        tuple of numpy.ndarray
            ``(index, distance)``.
        """
        idx = np.empty(len(zq), int)
        dist = np.empty(len(zq))
        for j in range(0, len(zq), 64):
            d2 = ((zq[j : j + 64, None, :] - self.z[None]) ** 2).sum(-1)
            idx[j : j + 64] = d2.argmin(1)
            dist[j : j + 64] = np.sqrt(d2.min(1))
        return idx, dist

    def columns(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the columns and references of anchors ``idx``.

        Parameters
        ----------
        idx : numpy.ndarray [n]
            The anchor indices.

        Returns
        -------
        tuple of numpy.ndarray
            ``(Y, refs)``.
        """
        return (
            np.stack([self.parts[p][r] for p, r in zip(self.part[idx], self.row[idx], strict=True)]),
            self.refs[idx],
        )


class _Rejections(NamedTuple):
    """A chunk's rejected rows, one mask per class."""

    tp_out: np.ndarray
    exhausted: np.ndarray
    stalled: np.ndarray
    nonfinite: np.ndarray

    @property
    def dead(self) -> np.ndarray:
        """Return the rows of any class."""
        return self.tp_out | self.exhausted | self.stalled | self.nonfinite

    def code(self) -> np.ndarray:
        """Return each row's class: 0 T-P out, 1 exhausted, 2 stalled, 3 otherwise."""
        return np.where(self.tp_out, 0, np.where(self.exhausted, 1, np.where(self.stalled, 2, 3)))


def _classify(
    tp_in: np.ndarray, conv: np.ndarray, acc: np.ndarray, L: np.ndarray, count_max: int, reject_below: float
) -> _Rejections:
    """Sort a chunk's rows into the rejection classes.

    Solver state first: the warm evaluator already floors L on a stalled or exhausted exit, which is not a non-finite
    forward. A solve certified at cap or cap + 1 is kept.
    """
    tp_out = ~tp_in
    exhausted = ~tp_out & ~conv & (acc >= count_max)
    stalled = ~tp_out & ~exhausted & ~conv
    nonfinite = ~tp_out & ~exhausted & conv & (~np.isfinite(L) | (L <= reject_below))
    return _Rejections(tp_out, exhausted, stalled, nonfinite)


class _ChunkLogs:
    """What a run keeps per chunk: the tally, the batch log, and the rejected and nuisance files."""

    def __init__(self, tally: dict, batch_log: Path | None, rejects: Path | None, nuisance: Path | None) -> None:
        self.tally, self.batch_log, self.rejects, self.nuisance = tally, batch_log, rejects, nuisance
        self.n_rej = self.n_nui = 0
        if rejects is not None:
            rejects.mkdir(parents=True, exist_ok=True)
            self.n_rej = len(list(rejects.glob("rejected_*.npz")))
        if nuisance is not None:
            nuisance.mkdir(parents=True, exist_ok=True)
            self.n_nui = len(list(nuisance.glob("nuisance_*.npz")))

    def write(
        self, z: np.ndarray, log_l: np.ndarray, rej: _Rejections, acc: np.ndarray, warm: bool, t: float, nuis: Any
    ) -> None:
        """Log one chunk of ``len(z)`` rows; ``nuis`` holds the evaluator's group statistics, one row per input row."""
        dead = rej.dead
        mode = "warm" if warm else "cold"
        if self.batch_log is not None:
            with self.batch_log.open("a") as fh:
                fh.write(json.dumps({"t": round(t, 2), "mode": mode, "acc": acc.tolist()}) + "\n")
        if self.rejects is not None and dead.any():
            _atomic_savez(
                self.rejects / f"rejected_{self.n_rej:05d}.npz",
                z=z[dead],
                cls=rej.code()[dead].astype(np.int8),
                acc=acc[dead].astype(np.int64),
                warm=np.int8(warm),
            )
            self.n_rej += 1
        if self.nuisance is not None and not dead.all():
            stats = np.asarray(nuis, np.float64)[: len(z)]
            _atomic_savez(
                self.nuisance / f"nuisance_{self.n_nui:05d}.npz", z=z[~dead], log_l=log_l[~dead], stats=stats[~dead]
            )
            self.n_nui += 1
        for key, v in (
            ("n_eval", len(z)),
            ("tp_out", rej.tp_out.sum()),
            ("exhausted", rej.exhausted.sum()),
            ("stalled", rej.stalled.sum()),
            ("nonfinite", rej.nonfinite.sum()),
            (f"n_{mode}", len(z)),
            (f"{mode}_certified", (~dead).sum()),
            (f"{mode}_steps_certified", acc[~dead].sum()),
        ):
            self.tally[key] = self.tally.get(key, 0) + int(v)


# size-limit-ok: 44 of its lines are the signature and numpy docs of nine arguments; the rest is one closure
def make_loglike(
    pipe: SimpleNamespace,
    n_batch: int,
    tally: dict,
    anchors: Anchors | None = None,
    batch_log: Path | None = None,
    prof: BatchProfiler | None = None,
    rejects: Path | None = None,
    nuisance: Path | None = None,
    kzz_ref: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the batched log-likelihood of unit-cube points, -inf on every rejection class.

    Calls a jitted evaluator at one fixed width: a short last chunk is padded with masked copies of its last row, and
    those and the T-P-invalid rows take no chemistry step and no RT.

    Parameters
    ----------
    pipe : types.SimpleNamespace
        The pipeline.
    n_batch : int
        The evaluator's width.
    tally : dict
        Receives this call's counts.
    anchors : Anchors, optional
        A chunk starts from the nearest certified column once one exists, and every certified column becomes an
        anchor.
    batch_log : pathlib.Path, optional
        Each chunk appends its wall time, mode and accepted step counts here.
    prof : BatchProfiler, optional
        The profiling hook.
    rejects : pathlib.Path, optional
        Each chunk writes its rejected rows here as one npz.
    nuisance : pathlib.Path, optional
        With integrated nuisances, each chunk writes its certified rows' ``z``, ``log_l`` and ``stats`` here before
        nautilus sees them.
    kzz_ref : float, optional
        The case's base Kzz at the reference pressure, cm^2/s; the first chunk is then the run's startup check.

    Returns
    -------
    callable
        ``loglike(Z)``, from [n, n_dim] to [n].
    """
    import jax
    import jax.numpy as jnp

    from retrieval_framework import pipeline as P

    tp_ok = jax.jit(jax.vmap(lambda u: pipe.tp_valid(pipe.theta_from_u(u))))
    Y0, refs0 = P._blank_state(pipe, n_batch)
    count_max = int(pipe.fwd.chem.count_max)
    k = int(pipe.n_chem_tp)
    logs = _ChunkLogs(tally, batch_log, rejects, nuisance)
    fns = {"cold": jax.jit(pipe._make_batch_eval("cold"))}
    if anchors is not None:
        fns["warm"] = jax.jit(pipe._make_batch_eval("warm"))
    if kzz_ref is not None:
        theta_of = jax.jit(jax.vmap(pipe.theta_from_u))
        i_kzz = list(pipe.names).index("lnKzz")

    def loglike(Z: np.ndarray) -> np.ndarray:
        """Evaluate unit-cube points in chunks of ``n_batch``."""
        nonlocal kzz_ref  # cleared once the first chunk has passed the startup check
        Z = np.atleast_2d(np.asarray(Z, np.float64))
        out = np.empty(len(Z))
        for i in range(0, len(Z), n_batch):
            Zc = Z[i : i + n_batch]
            m = len(Zc)
            if m < n_batch:
                Zc = np.concatenate([Zc, np.repeat(Zc[-1:], n_batch - m, axis=0)])
            Uj = jnp.asarray(np.log(Zc) - np.log1p(-Zc), pipe.dtype)
            # Not bitwise for the kept rows: fewer jobs on the lane queue shift the shared refresh cadence.
            tp_in = np.asarray(jax.device_get(tp_ok(Uj)), bool)
            act = jnp.asarray((np.arange(n_batch) < m) & tp_in)
            if prof is not None:
                prof.batch_begin()
            t0 = time.perf_counter()
            warm = False
            if anchors is not None and len(anchors) > 0:
                warm = True
                idx, dist = anchors.nearest(Zc[:, :k])
                Yw, rw = anchors.columns(idx)
                L, Ynew, rnew, st = fns["warm"](Uj, jnp.asarray(Yw, pipe.dtype), jnp.asarray(rw, pipe.dtype), act)
            else:
                L, Ynew, rnew, st = fns["cold"](Uj, Y0, refs0, act)
            P._abort_if_frozen(st.reason, st.last_delta, Zc[:m], pipe.fwd.chem_rtol)
            L = np.asarray(jax.device_get(L), np.float64)[:m]
            acc = np.asarray(jax.device_get(st.acc), np.int64)[:m]
            conv = np.asarray(jax.device_get(st.conv_ok), bool)[:m]
            rej = _classify(tp_in[:m], conv, acc, L, count_max, P.REJECT_BELOW)
            dead = rej.dead
            out[i : i + m] = np.where(dead, -np.inf, L)
            if kzz_ref is not None:
                th = np.asarray(jax.device_get(theta_of(Uj)), np.float64)[:m]
                counts = {
                    "T-P out": rej.tp_out.sum(),
                    "exhausted": rej.exhausted.sum(),
                    "stalled": rej.stalled.sum(),
                    "non-finite": rej.nonfinite.sum(),
                }
                log10_kzz = np.log10(kzz_ref) + th[:, i_kzz] / np.log(10.0)
                for line in startup_check(log10_kzz, ~rej.tp_out, ~dead, counts):
                    log.info(line)
                kzz_ref = None
            logs.write(Zc[:m], out[i : i + m], rej, acc, warm, time.perf_counter() - t0, st.nuis if nuisance else None)
            if warm:
                tally["warm_anchor_dist_sum"] = tally.get("warm_anchor_dist_sum", 0.0) + float(dist[:m].sum())
            if anchors is not None:
                alive = ~dead
                anchors.add(
                    Zc[:m][alive][:, :k],
                    np.asarray(jax.device_get(Ynew), np.float64)[:m][alive],
                    np.asarray(jax.device_get(rnew), np.float64)[:m][alive],
                )
        return out

    return loglike


def nuisance_posterior(
    pipe: SimpleNamespace, folder: Path, queries: Iterable[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
) -> list[np.ndarray]:
    """Return the integrated nuisances' posterior, one conditional draw per row of each query.

    Drawn from the group statistics logged with the evaluation that gave the point its log L; NaN on a rejected row. A
    point logged twice, as a batch redrawn after a resume is, matches by its log L too.

    Parameters
    ----------
    pipe : types.SimpleNamespace
        The pipeline.
    folder : pathlib.Path
        The run's nuisance/ directory.
    queries : iterable of tuple
        ``(z, log_l)`` rows.
    rng : numpy.random.Generator
        The generator.

    Returns
    -------
    list of numpy.ndarray
        One array [n, len(names)] per query.

    Raises
    ------
    RuntimeError
        If a finite row has no logged statistics.
    """
    table: dict[bytes, list[tuple[float, np.ndarray]]] = {}
    for f in sorted(folder.glob("nuisance_*.npz")):
        with np.load(f) as d:
            for zz, ll, ss in zip(d["z"], d["log_l"], d["stats"], strict=True):
                table.setdefault(zz.tobytes(), []).append((float(ll), ss))
    res = []
    for z_query, lq in queries:
        zq = np.asarray(z_query, np.float64)
        st = np.full((len(zq), 2 * len(pipe.nuisance.groups)), np.nan)
        for i, (zz, ll) in enumerate(zip(zq, np.asarray(lq, np.float64), strict=True)):
            if not np.isfinite(ll):
                continue
            hit = [s for lg, s in table.get(zz.tobytes(), []) if lg == ll]
            if not hit:
                raise RuntimeError(f"posterior row {i} (log L {ll}) has no logged nuisance statistics in {folder}")
            st[i] = hit[0]
        nu = np.full((len(zq), len(pipe.nuisance.names)), np.nan)
        ok = np.isfinite(st[:, 0])
        if ok.any():
            nu[ok] = pipe.nuisance.draw(st[ok], pipe.obs_sigma, rng)
        res.append(nu)
    return res


def make_sampler(
    loglike: Callable[[np.ndarray], np.ndarray],
    n_dim: int,
    n_live: int,
    n_batch: int,
    seed: int,
    filepath: str | None = None,
    resume: bool = False,
) -> nautilus.Sampler:
    """Return the stock nautilus sampler with this module's settings.

    The prior is the identity on the unit cube, since the pipeline maps u = logit(z) to theta; ``loglike`` is
    vectorized.

    Parameters
    ----------
    loglike : callable
        From [n, n_dim] to [n].
    n_dim : int
        The dimension.
    n_live : int
        Live points; n_update is half of it.
    n_batch : int
        Points per batch.
    seed : int
        The sampler's seed.
    filepath : str, optional
        The checkpoint.
    resume : bool, optional
        Continue the checkpoint.

    Returns
    -------
    nautilus.Sampler
        The sampler.
    """
    import nautilus

    return nautilus.Sampler(
        lambda z: z,
        loglike,
        n_dim=n_dim,
        n_live=n_live,
        n_update=n_live // 2,
        enlarge_per_dim=ENLARGE_PER_DIM,
        n_networks=N_NETWORKS,
        n_batch=n_batch,
        vectorized=True,
        pass_dict=False,
        seed=seed,
        filepath=filepath,
        resume=resume,
    )


def _check_resume(digest_path: Path, want: str, ckpt_name: str) -> None:
    """Refuse a resume whose run directory holds another target's identity, or none."""
    have = digest_path.read_text().strip() if digest_path.exists() else ""
    if have != want:
        raise RuntimeError(
            f"nautilus resume refused: {ckpt_name} belongs to a different "
            f"target ({have or '<absent>'} vs this run's {want}). Restore "
            "the original target and settings or use a fresh output "
            "directory."
        )


def _starts(warm: bool) -> str:
    """Name the start policy."""
    return "anchored warm (growing)" if warm else "cold"


def _log_settings(spec: Any, cfg: C.Config, n_batch: int) -> None:
    """Log the run's sampler settings and the resolved Config."""
    n_live = int(spec.sampler.n_live)
    log.info("spec=%s out_dir=%s", spec.path, cfg.out_dir)
    log.info(
        "nautilus: n_live=%s n_update=%s n_eff=%s n_batch=%s (cold_lanes=%s) f_live=%s enlarge_per_dim=%s "
        "networks=%sx relu (nautilus default) discard_exploration=%s seed=%s starts=%s",
        n_live,
        n_live // 2,
        int(spec.sampler.n_eff),
        n_batch,
        cfg.cold_lanes,
        F_LIVE,
        ENLARGE_PER_DIM,
        N_NETWORKS,
        DISCARD_EXPLORATION,
        cfg.seed,
        _starts(bool(spec.sampler.warm_starts)),
    )
    log.info(C.describe_config(cfg, spec.name))


def _saved_loglike(
    pipe: SimpleNamespace, cfg: C.Config, out: Path, n_batch: int, warm: bool, resume: bool, tally: dict
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the run's log-likelihood, which saves the tally after each call, atomically."""
    from retrieval_framework.nautilus_report import _kzz_at_ref

    anchors = Anchors(out / "anchors", int(pipe.n_chem_tp)) if warm else None
    if anchors is not None:
        log.info("nautilus: %s anchor column(s) loaded", len(anchors))
    kzz_ref = (
        None if resume else _kzz_at_ref({"vulcan_cfg_name": cfg.vulcan_cfg_name, "cfg_overrides": cfg.cfg_overrides})
    )
    like = make_loglike(
        pipe,
        n_batch,
        tally,
        anchors,
        out / "batches.jsonl",
        BatchProfiler(),
        rejects=out / "rejected",
        nuisance=out / "nuisance" if pipe.nuisance is not None else None,
        kzz_ref=kzz_ref,
    )
    tally_path = out / TALLY_FILE

    def loglike_saved(Z: np.ndarray) -> np.ndarray:
        """Evaluate and save the tally."""
        L = like(Z)
        tmp = tally_path.with_name(f".{tally_path.name}.part")
        tmp.write_text(json.dumps(tally, indent=2) + "\n")
        tmp.replace(tally_path)
        return L

    return loglike_saved


def _write_posterior(sampler: nautilus.Sampler, pipe: SimpleNamespace, out: Path, want: str, seed: int) -> None:
    """Write nautilus_posterior.npz: the weighted points, the equal-weight samples and any integrated nuisances."""
    import jax

    from retrieval_framework import pipeline as P

    z, log_w, log_l = sampler.posterior()
    z_eq, _, log_l_eq = sampler.posterior(equal_weight=True)
    extra = {}
    if pipe.nuisance is not None:
        nu, nu_eq = nuisance_posterior(
            pipe, out / "nuisance", [(z, log_l), (z_eq, log_l_eq)], np.random.default_rng(seed + 2)
        )
        extra = {
            "nuisance_names": np.asarray(pipe.nuisance.names, dtype="<U64"),
            "nuisance": nu,
            "samples_nuisance": nu_eq,
        }
    to_theta = jax.jit(jax.vmap(pipe.theta_from_u))
    theta = np.asarray(to_theta(np.log(z) - np.log1p(-z)), np.float64)
    theta_eq = np.asarray(to_theta(np.log(z_eq) - np.log1p(-z_eq)), np.float64)
    P.save_npz(
        out / "nautilus_posterior.npz",
        param_names=np.asarray(pipe.names, dtype="<U64"),
        target_digest=np.asarray(want),
        theta=theta,
        log_w=log_w,
        log_l=log_l,
        samples=theta_eq,
        u=np.log(z) - np.log1p(-z),
        logZ_box=np.asarray(sampler.log_z),
        n_like=np.asarray(sampler.n_like),
        **extra,
    )


def _write_summary(
    sampler: nautilus.Sampler, pipe: SimpleNamespace, spec: Any, out: Path, n_batch: int, tally: dict
) -> None:
    """Write nautilus_summary.json."""
    import nautilus

    n_live = int(spec.sampler.n_live)
    warm = bool(spec.sampler.warm_starts)
    summary = {
        "logZ_box": float(sampler.log_z),
        "n_like": int(sampler.n_like),
        "n_eff": float(sampler.n_eff),
        "n_live": n_live,
        "n_batch": n_batch,
        "n_update": n_live // 2,
        "f_live": F_LIVE,
        "enlarge_per_dim": ENLARGE_PER_DIM,
        "networks": f"{N_NETWORKS}x relu (nautilus default)",
        "discard_exploration": DISCARD_EXPLORATION,
        "nautilus": f"{nautilus.__version__} (stock)",
        "starts": _starts(warm),
        "rejections_and_starts": tally,
        "spec": spec.name,
        "seed": int(spec.cfg.seed),
        "chem_rtol": pipe.fwd.chem_rtol,
        "evidence_note": "zero-filled box evidence (log L = -inf on every rejection)"
        + (
            "; APPROXIMATE: warm starts move the likelihood and which draws "
            "are rejected (warm_starts: false for an evidence claim)"
            if warm
            else ""
        ),
    }
    (out / "nautilus_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _log_versions() -> None:
    """Log the backend, the sampler and the hardware; the first import of jax, so after the runtime export."""
    import jax
    import nautilus

    log.info(
        "jax backend=%s devices=%s nautilus %s (stock, %s)",
        jax.default_backend(),
        jax.devices(),
        nautilus.__version__,
        nautilus.__file__,
    )
    log.info("hardware: %s", C.hardware_profile())


def _copy_specs(spec: Any, out: Path) -> None:
    """Keep the run's spec, and its base if any, in the run directory."""
    shutil.copy(spec.path, out / "spec.yaml")
    if spec.base is not None:
        shutil.copy(spec.base, out / "spec_base.yaml")


def _sample(sampler: nautilus.Sampler, n_eff: int, timeout: float, tally: dict) -> bool:
    """Run the sampler to its stopping rule or the timeout, and log where it ended; True if it finished."""
    log.info("nautilus: starting at n_like=%s; timeout %.2f h", sampler.n_like, timeout / 3600)
    t0 = time.perf_counter()
    done = sampler.run(
        f_live=F_LIVE, n_eff=n_eff, discard_exploration=DISCARD_EXPLORATION, timeout=timeout, verbose=True
    )
    log.info(
        "nautilus: %s after %.2f h | n_like=%s | log_z=%.3f | n_eff=%.0f | tally=%s",
        "finished" if done else "stopped at the wall budget",
        (time.perf_counter() - t0) / 3600,
        sampler.n_like,
        sampler.log_z,
        sampler.n_eff,
        tally,
    )
    return bool(done)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    ap = argparse.ArgumentParser(description="nautilus nested sampling of one run specification")
    ap.add_argument("spec", help="the run specification, runs/<case>/<name>.yaml")
    ap.add_argument("--resume", action="store_true", help="continue the checkpoint in the run's output directory")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved settings and exit without importing jax")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run one specification, from the runtime export to the certificate.

    Refuses a resume across a change of target or sampler settings.

    Parameters
    ----------
    argv : list of str, optional
        The arguments; default ``sys.argv``.
    """
    t_start = time.time()
    args = _parse_args(argv)
    spec = load_spec(args.spec)
    cfg = spec.cfg
    if args.dry_run:
        print(C.describe_config(cfg, spec.name))
        print(f"sampler: {spec.sampler}\nruntime: {spec.runtime}\nout_dir: {cfg.out_dir}")
        return
    export_runtime(spec.runtime)
    C.export_network(cfg)
    lanes = int(cfg.cold_lanes)
    if spec.runtime.solver == "ffi":
        lanes *= FFI_BLOCKS_PER_SM
    cfg = replace(cfg, cold_lanes=lanes)
    out: Path = cfg.out_dir  # type: ignore[assignment]  # load_spec sets it
    ckpt = out / "nautilus_checkpoint.hdf5"
    resume = bool(args.resume)
    check_run_dir(out, resume)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "run.log")
    n_live = int(spec.sampler.n_live)
    n_batch = int(spec.sampler.n_batch or BATCH_PER_LANE * lanes)
    warm = bool(spec.sampler.warm_starts)
    _log_settings(spec, cfg, n_batch)
    _log_versions()

    from retrieval_framework import pipeline as P

    t0 = time.perf_counter()
    pipe = P.build_pipeline(cfg)
    log.info("Built pipeline in %.1fs | n_dim=%s: %s", time.perf_counter() - t0, pipe.n_dim, pipe.names)
    obs_save = set_observations(cfg, pipe, P)

    # Before anything in the run directory is written. The start policy, the batch width and the sampler settings
    # (nautilus restores none of them) are part of the target, so a resume cannot switch them.
    digest_path = out / "target_digest.txt"
    want = (
        f"{pipe.target_digest}|{'warm' if warm else 'cold'}"
        f"|n_batch={n_batch}|n_live={n_live}|n_update={n_live // 2}|f_live={F_LIVE}"
        f"|enlarge_per_dim={ENLARGE_PER_DIM}|networks={N_NETWORKS}xrelu|stock"
    )
    if resume:
        _check_resume(digest_path, want, ckpt.name)
    write_run_identity(cfg, pipe, spec, obs_save)
    _copy_specs(spec, out)
    digest_path.write_text(want + "\n")

    tally_path = out / TALLY_FILE
    tally = json.loads(tally_path.read_text()) if (resume and tally_path.exists()) else {}
    like = _saved_loglike(pipe, cfg, out, n_batch, warm, resume, tally)
    sampler = make_sampler(like, pipe.n_dim, n_live, n_batch, int(cfg.seed), filepath=str(ckpt), resume=resume)
    budget = float(cfg.walltime_seconds)
    timeout = max(budget - (time.time() - t_start), 60.0) if budget > 0 else float("inf")
    done = _sample(sampler, int(spec.sampler.n_eff), timeout, tally)
    if not done:
        log.warning("nautilus did not finish: resubmit with --resume (the checkpoint is %s).", ckpt)
        return

    _write_posterior(sampler, pipe, out, want, int(cfg.seed))
    _write_summary(sampler, pipe, spec, out, n_batch, tally)
    log.info("wrote %s and nautilus_summary.json", out / "nautilus_posterior.npz")
    from retrieval_framework.nautilus_report import write_certificate

    log.info("wrote %s", write_certificate(out))


if __name__ == "__main__":
    main()
