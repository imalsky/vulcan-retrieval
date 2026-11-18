"""A warm-started run's cold re-solve: where its verdict is written and what the run's report makes of it.

``validation/e2e_check.py warmcold`` writes the verdict; docs/method.md "Warm starts" states the check and its gates.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from retrieval_framework import provenance

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["QUOTABLE", "WARM_COLD_DIR", "caveat", "check"]

# Where a run's validation artifacts go, under its output directory.
WARM_COLD_DIR = "validation"
# The verdicts a run's numbers may be quoted under: a cold run needs no check.
QUOTABLE = ("NOT NEEDED", "PASS")


def check(out: Path, summ: dict) -> dict[str, Any]:
    """Return the verdict of the run's warm-against-cold check (``validation/e2e_check.py warmcold``).

    Parameters
    ----------
    out : pathlib.Path
        The run's output directory; the check writes ``validation/warm_cold_<its name>.json`` there.
    summ : dict
        The run's summary.

    Returns
    -------
    dict
        ``status`` is "NOT NEEDED" for a cold run, "NOT RUN" with no check, "STALE" for a check of another posterior
        file than the run's (a resumed run writes a new one), else the check's own "PASS" or "FAIL" with its
        measurements.
    """
    if summ["starts"] == "cold":
        return {"status": "NOT NEEDED"}
    path = out / WARM_COLD_DIR / f"warm_cold_{out.name}.json"
    if not path.exists():
        return {"status": "NOT RUN"}
    art = json.loads(path.read_text())
    m = art["measurements"][0]
    keep = {k: m.get(k) for k in ("rows", "cold_certified", "cold_count_max", "worst", "worst_shift_sigma", "ess")}
    posterior = out / "nautilus_posterior.npz"
    if not posterior.exists() or m.get("posterior_sha256") != provenance._sha256(posterior):
        return {"status": "STALE", **keep}
    return {"status": art["verdict"]["status"], **keep}


def caveat(out: Path, verdict: dict) -> str:
    """Return the caveat of a warm run: what its check found, or that its numbers are not to be quoted.

    Parameters
    ----------
    out : pathlib.Path
        The run's output directory.
    verdict : dict
        What ``check`` returned.

    Returns
    -------
    str
        The caveat; it holds "NOT QUOTABLE" unless the check passed.
    """
    how = (
        "Warm starts: each solve continued from the nearest certified column, so the likelihood and which draws "
        "were rejected depend on the evaluation order; logZ_box is approximate, and an evidence claim needs a cold "
        "run (warm_starts: false in the spec). The draws rejected at the step cap are not checked; their measured "
        'effect on earlier runs is in notes.md "Warm starts". '
    )
    if verdict["status"] == "PASS":
        return how + (
            f"Checked: {verdict['rows']} weighted posterior rows re-solved cold within {verdict['cold_count_max']} "
            f"steps move no 16th, 50th or 84th percentile by more than {abs(verdict['worst_shift_sigma']):.3f} sigma "
            f"({verdict['worst']})."
        )
    return how + (
        f"NOT QUOTABLE: the warm-against-cold check is {verdict['status']}. Run `python validation/e2e_check.py solve "
        f"<spec> --rung production --posterior {out} --draws 1000`, then `python validation/e2e_check.py warmcold "
        f"{out / WARM_COLD_DIR / 'e2e_production_posterior.npz'}`. A FAIL means the run is repeated cold."
    )
