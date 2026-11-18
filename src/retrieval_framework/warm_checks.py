"""A warm-started run's own checks: the guards on its tally and the spot-check at its end, and the caveat they give.

Both only warn: neither changes the likelihood, stops the run or stops its report. docs/method.md "Warm starts" states
them; notes.md "Warm starts" holds the measurements behind their limits.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from retrieval_framework import constants

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = ["SPOT_CHECK_FILE", "caveat", "guards", "spot_check"]

log = logging.getLogger("retrieval")

SPOT_CHECK_FILE = "spot_check.json"


def guards(tally: dict) -> dict[str, Any]:
    """Return a warm run's guards on its tally: the numbers, and one warning for each limit passed.

    Parameters
    ----------
    tally : dict
        The run's tally (``nautilus_tally.json``), summed over its jobs.

    Returns
    -------
    dict
        ``warm_start_none_frac``: the share of the in-window warm rows with no physical start among the anchors tried;
        ``warm_exhausted_frac`` and ``cold_exhausted_frac``: the share of each mode's in-window rows rejected at the
        step cap; ``skipped``: a guard that could not be computed, and why; ``warnings``: the limits passed.
    """
    warm_in, cold_in = tally.get("warm_in_window", 0), tally.get("cold_in_window", 0)
    out: dict[str, Any] = {"warm_in_window": warm_in, "cold_in_window": cold_in, "skipped": [], "warnings": []}
    if warm_in:
        none = out["warm_start_none_frac"] = tally.get("warm_start_none", 0) / warm_in
        if none > constants.WARM_NONE_WARN_FRAC:
            out["warnings"].append(
                f"{none:.2%} of the in-window warm rows had no physical start among the {constants.WARM_START_TRIES} "
                f"nearest anchors and were rejected at step zero (limit {constants.WARM_NONE_WARN_FRAC:.1%})."
            )
    else:
        out["skipped"].append("no-start guard: no in-window warm row")
    if warm_in and cold_in:
        warm = out["warm_exhausted_frac"] = tally.get("warm_exhausted", 0) / warm_in
        cold = out["cold_exhausted_frac"] = tally.get("cold_exhausted", 0) / cold_in
        if warm - cold > constants.WARM_EXHAUSTED_EXCESS:
            out["warnings"].append(
                f"{warm:.1%} of the in-window warm rows ran out of steps against {cold:.1%} of the cold rows (limit "
                f"{constants.WARM_EXHAUSTED_EXCESS:.0%} more): warm starts lose draws a cold solve may keep."
            )
    else:
        out["skipped"].append(f"step-cap guard: {cold_in} in-window cold and {warm_in} warm rows")
    for w in out["warnings"]:
        log.warning("warm-start guard: %s", w)
    return out


def _first_cold_batch_seconds(batch_log: Path) -> float | None:
    """Return the wall time of the run's first cold batch, or None when the batch log holds none."""
    if not batch_log.is_file():
        return None
    for row in map(json.loads, batch_log.read_text().splitlines()):
        if row["mode"] == "cold":
            return float(row["t"])
    return None


def _rejected_at_cap(rejects: Path, n_dim: int) -> np.ndarray:
    """Return the unit-cube points of every warm row the run rejected at the step cap, [n, n_dim]."""
    rows = [np.zeros((0, n_dim))]
    for f in sorted(rejects.glob("rejected_*.npz")):
        with np.load(f) as d:
            if int(d["warm"]):
                rows.append(np.asarray(d["z"], np.float64)[d["cls"] == 1])  # class 1: exhausted (chunks.Rejections)
    return np.concatenate(rows)


def _spot_check(
    out: Path, cold_log_l: Callable[[np.ndarray], np.ndarray], seed: int, n_batch: int, seconds_left: float
) -> dict[str, Any]:
    """Run the spot-check; ``spot_check`` holds the contract."""
    from retrieval_framework.nautilus_report import _wquant, weights

    with np.load(out / "nautilus_posterior.npz") as post:
        u, log_l, theta = (np.asarray(post[k], np.float64) for k in ("u", "log_l", "theta"))
        w, names = weights(post["log_w"]), [str(n) for n in post["param_names"]]
    rng = np.random.default_rng(seed + 3)  # seed + 1 and + 2 draw the report's spectra and the nuisance posterior
    idx = rng.choice(len(w), size=constants.SPOT_CHECK_POSTERIOR_ROWS, p=w)
    capped = _rejected_at_cap(out / "rejected", u.shape[1])
    capped = capped[rng.choice(len(capped), size=min(constants.SPOT_CHECK_REJECTED_ROWS, len(capped)), replace=False)]
    first_cold = _first_cold_batch_seconds(out / "batches.jsonl")
    if first_cold is None:
        return {"status": "skipped", "reason": "batches.jsonl holds no cold batch to estimate its cost from"}
    estimate = first_cold * math.ceil((len(idx) + len(capped)) / n_batch)
    if not estimate < seconds_left:
        return {
            "status": "skipped",
            "reason": f"it needs about {estimate:.0f} s (the run's first cold batch per batch of rows) and "
            f"{max(seconds_left, 0.0):.0f} s are left before half the job's headroom",
        }
    L = cold_log_l(np.concatenate([1.0 / (1.0 + np.exp(-u[idx])), capped]))
    l_cold, l_run, l_capped = L[: len(idx)], log_l[idx], L[len(idx) :]
    ok = np.isfinite(l_cold)
    dl = l_cold[ok] - l_run[ok]
    # A row that does not certify cold has weight 0: it counts against the run.
    rew = np.zeros(len(idx))
    if ok.any():
        rew[ok] = np.exp(dl - dl.max())
    dl = np.abs(dl)
    keep = rew > 0
    mean = w @ theta
    sd = np.sqrt(w @ (theta - mean) ** 2)
    shifts = {
        f"{name} q{round(100 * q)}": (new - ref) / sd[j]
        for j, name in enumerate(names)
        if sd[j] > 0 and keep.any()
        for q, ref, new in zip(
            (0.16, 0.5, 0.84),
            _wquant(theta[idx, j], np.ones(len(idx))),
            _wquant(theta[idx, j][keep], rew[keep]),
            strict=True,
        )
    }
    worst = max(shifts, key=lambda k: abs(shifts[k])) if shifts else None
    fin = np.isfinite(log_l)
    low = _wquant(log_l[fin], w[fin], (0.01,))[0]  # the posterior's 1% log-L point
    rec: dict[str, Any] = {
        "status": "done",
        "estimate_seconds": round(estimate, 1),
        "posterior_rows": len(idx),
        "posterior_certified_cold": int(ok.sum()),
        "median_abs_dlog_l": float(np.median(dl)) if dl.size else None,
        "max_abs_dlog_l": float(dl.max()) if dl.size else None,
        "worst": worst,
        "worst_shift_sigma": None if worst is None else float(shifts[worst]),
        "rejected_rows": len(capped),
        "rejected_certified_cold": int(np.isfinite(l_capped).sum()),
        "rejected_above_posterior_1pct": int((l_capped > low).sum()),
        "warnings": [],
    }
    if worst is None:
        rec["warnings"].append(f"none of the {len(idx)} posterior rows certified cold: no shift could be measured.")
    elif abs(shifts[worst]) > constants.SPOT_CHECK_SHIFT_SIGMA:
        rec["warnings"].append(
            f"reweighted by the cold likelihood, {len(idx)} posterior rows move {worst} by {shifts[worst]:+.3f} sigma "
            f"(limit {constants.SPOT_CHECK_SHIFT_SIGMA})."
        )
    return rec


def spot_check(
    out: Path, cold_log_l: Callable[[np.ndarray], np.ndarray], seed: int, n_batch: int, seconds_left: float
) -> dict[str, Any]:
    """Solve a sample of a finished warm run's rows cold, and write what it finds to ``spot_check.json`` in ``out``.

    ``constants.SPOT_CHECK_POSTERIOR_ROWS`` posterior rows drawn by weight, and up to
    ``constants.SPOT_CHECK_REJECTED_ROWS`` drawn uniformly from the warm rows the run rejected at the step cap, are
    solved together through ``cold_log_l``. The posterior rows give |L_cold - L_run| and the largest shift of a 16th,
    50th or 84th percentile, in posterior sigma, when the rows are reweighted by exp(L_cold - L_run); the rejected rows,
    how many certify cold and how many land above the posterior's 1 % log-L point. It runs only when the run's first
    cold batch time, per ``n_batch`` rows, fits in ``seconds_left``. It never raises.

    Parameters
    ----------
    out : pathlib.Path
        The run's output directory, holding ``nautilus_posterior.npz``, ``batches.jsonl`` and ``rejected/``.
    cold_log_l : callable
        From unit-cube points [n, n_dim] to log L [n], -inf on every rejection: the run's own cold evaluator.
    seed : int
        The run's seed.
    n_batch : int
        The run's batch width.
    seconds_left : float
        Seconds the check may take.

    Returns
    -------
    dict
        ``status`` "done" with the measurements and ``warnings``, "skipped" with a ``reason``, or "failed" with the
        ``error``.
    """
    t0 = time.perf_counter()
    try:
        rec = _spot_check(out, cold_log_l, seed, n_batch, seconds_left)
    except Exception as e:  # the check only warns: its failure is recorded, and the run goes on
        log.exception("spot-check failed; the run is unaffected")
        rec = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    if rec["status"] == "done":
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        log.info("spot-check: %s", {k: v for k, v in rec.items() if k != "warnings"})
        for w in rec["warnings"]:
            log.warning("spot-check: %s", w)
    elif rec["status"] == "skipped":
        log.warning("spot-check skipped: %s", rec["reason"])
    (out / SPOT_CHECK_FILE).write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def _spot_line(spot: dict | None) -> str:
    """Return the caveat's sentence on the spot-check."""
    if spot is None:
        return "The run's summary records no end-of-run spot-check."
    if spot["status"] == "skipped":
        return f"The end-of-run spot-check was skipped: {spot['reason']}."
    if spot["status"] == "failed":
        return f"The end-of-run spot-check failed: {spot['error']}."

    def num(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3g}"

    shift = (
        "no row certified cold"
        if spot["worst"] is None
        else f"{spot['worst_shift_sigma']:+.3f} sigma ({spot['worst']})"
    )
    return (
        f"Spot-check: {spot['posterior_certified_cold']} of {spot['posterior_rows']} weighted posterior rows certified "
        f"cold, |L_cold - L_run| median {num(spot['median_abs_dlog_l'])} and max {num(spot['max_abs_dlog_l'])}; "
        f"reweighted by the cold likelihood, the worst 16th, 50th or 84th percentile shift is {shift}. Of "
        f"{spot['rejected_rows']} draws rejected at the step cap, {spot['rejected_certified_cold']} certified cold, "
        f"{spot['rejected_above_posterior_1pct']} above the posterior's 1% log L."
        + "".join(f" Warning: {w}" for w in spot["warnings"])
    )


def caveat(summ: dict) -> str:
    """Return the report's caveat of a warm run: what warm starts do, the stated limit, the guards and the spot-check.

    Parameters
    ----------
    summ : dict
        The run's summary (``nautilus_summary.json``).

    Returns
    -------
    str
        The caveat.
    """
    g = summ.get("warm_start_guards")
    found = (
        ["The run's summary records no warm-start guards."]
        if g is None
        else [f"Warning: {w}" for w in g["warnings"]] + [f"Not computed: {s}." for s in g["skipped"]]
    )
    how = (
        "Warm starts: each solve continued from the nearest certified column whose start is physical, so the "
        "likelihood and which draws were rejected depend on the evaluation order; logZ_box is approximate, and an "
        "evidence claim needs a cold run (warm_starts: false in the spec). The draws rejected at the step cap are "
        'not solved cold per run; their measured effect on earlier runs is in notes.md "Warm starts".'
    )
    return " ".join([how, *found, _spot_line(summ.get("spot_check"))])
