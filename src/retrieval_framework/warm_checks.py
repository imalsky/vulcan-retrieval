"""A warm-started run's own checks: the guard on its tally and the spot-check at its end, and the caveat they give.

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
    """Return a warm run's guard on its tally: the numbers, and a warning if its limit is passed.

    Parameters
    ----------
    tally : dict
        The run's tally (``nautilus_tally.json``), summed over its jobs.

    Returns
    -------
    dict
        ``warm_start_none_frac``: the share of the in-window warm rows with no physical start among the anchors tried;
        ``skipped``: the guard, if it could not be computed, and why; ``warnings``: the limit passed.
    """
    warm_in = tally.get("warm_in_window", 0)
    out: dict[str, Any] = {"warm_in_window": warm_in, "skipped": [], "warnings": []}
    if warm_in:
        none = out["warm_start_none_frac"] = tally.get("warm_start_none", 0) / warm_in
        if none > constants.WARM_NONE_WARN_FRAC:
            out["warnings"].append(
                f"{none:.2%} of the in-window warm rows had no physical start among the {constants.WARM_START_TRIES} "
                f"nearest anchors and were rejected at step zero (limit {constants.WARM_NONE_WARN_FRAC:.1%})."
            )
    else:
        out["skipped"].append("no-start guard: no in-window warm row")
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
    out: Path,
    cold_log_l: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_batch: int,
    seconds_left: float,
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
    L, frozen = cold_log_l(np.concatenate([1.0 / (1.0 + np.exp(-u[idx])), capped]))
    l_cold, l_run, l_capped = L[: len(idx)], log_l[idx], L[len(idx) :]
    # Every posterior row certified warm. The shift compares only the rows that also certified cold, with the same
    # weights on both sides: a row lost cold would shift the quantiles by its absence alone (notes "Warm starts"), so
    # it is counted apart.
    ok = np.isfinite(l_cold)
    dl = l_cold[ok] - l_run[ok]
    rew = np.exp(dl - dl.max()) if ok.any() else dl
    dl = np.abs(dl)
    th = theta[idx][ok]
    mean = w @ theta
    sd = np.sqrt(w @ (theta - mean) ** 2)
    shifts = {
        f"{name} q{round(100 * q)}": (new - ref) / sd[j]
        for j, name in enumerate(names)
        if sd[j] > 0 and ok.any()
        for q, ref, new in zip(
            (0.16, 0.5, 0.84), _wquant(th[:, j], np.ones(len(th))), _wquant(th[:, j], rew), strict=True
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
        "posterior_not_certified_cold": int((~ok).sum()),
        # Drawn by weight with replacement, each row carries 1 / posterior_rows of the posterior: an estimate.
        "posterior_not_certified_cold_weight_frac": float((~ok).mean()),
        "frozen_rows": int(frozen.sum()),
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
    elif (~ok).mean() > constants.SPOT_CHECK_LOST_WARN_FRAC:
        rec["warnings"].append(
            f"the shift was measured on only {int(ok.sum())} of {len(idx)} posterior rows: ~{(~ok).mean():.1%} of the "
            f"posterior weight (estimated) did not certify cold (limit {constants.SPOT_CHECK_LOST_WARN_FRAC:.0%})."
        )
    if worst is not None and abs(shifts[worst]) > constants.SPOT_CHECK_SHIFT_SIGMA:
        rec["warnings"].append(
            f"reweighted by the cold likelihood, the {int(ok.sum())} posterior rows certified cold move {worst} by "
            f"{shifts[worst]:+.3f} sigma (limit {constants.SPOT_CHECK_SHIFT_SIGMA})."
        )
    return rec


def spot_check(
    out: Path,
    cold_log_l: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_batch: int,
    seconds_left: float,
) -> dict[str, Any]:
    """Solve a sample of a finished warm run's rows cold, and write what it finds to ``spot_check.json`` in ``out``.

    ``constants.SPOT_CHECK_POSTERIOR_ROWS`` posterior rows drawn by weight, and up to
    ``constants.SPOT_CHECK_REJECTED_ROWS`` drawn uniformly from the warm rows the run rejected at the step cap, are
    solved together through ``cold_log_l``. Over the posterior rows that certify cold, they give |L_cold - L_run| and
    the largest shift of a 16th, 50th or 84th percentile, in posterior sigma, when those rows are reweighted by
    exp(L_cold - L_run); the rows that do not certify cold are counted apart, with their estimated share of the
    posterior weight.
    The rejected rows give how many certify cold and how many land above the posterior's 1 % log-L point. A row frozen
    at dt_min does not certify, and is counted. It runs only when the run's first cold batch time, per ``n_batch``
    rows, fits in ``seconds_left``. It never raises.

    Parameters
    ----------
    out : pathlib.Path
        The run's output directory, holding ``nautilus_posterior.npz``, ``batches.jsonl`` and ``rejected/``.
    cold_log_l : callable
        From unit-cube points [n, n_dim] to log L [n], -inf on every rejection, and the rows frozen at dt_min [n]
        (``chunks.cold_log_l``): the run's own cold evaluator.
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
        f"cold; {spot['posterior_not_certified_cold']} did not "
        f"(~{spot['posterior_not_certified_cold_weight_frac']:.1%} of the posterior weight, estimated from "
        f"{spot['posterior_rows']} draws). Over the rows certified "
        f"cold, |L_cold - L_run| median {num(spot['median_abs_dlog_l'])} and max {num(spot['max_abs_dlog_l'])}; "
        f"reweighted by the cold likelihood, the worst 16th, 50th or 84th percentile shift is {shift}. Of "
        f"{spot['rejected_rows']} draws rejected at the step cap, {spot['rejected_certified_cold']} certified cold, "
        f"{spot['rejected_above_posterior_1pct']} above the posterior's 1% log L. {spot['frozen_rows']} of the rows "
        "solved cold froze at dt_min and count as not certified." + "".join(f" Warning: {w}" for w in spot["warnings"])
    )


def caveat(summ: dict) -> str:
    """Return the report's caveat of a warm run: what warm starts do, the stated limit, the guard and the spot-check.

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
