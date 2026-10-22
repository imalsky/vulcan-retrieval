"""Shared helpers for the validation scripts: the production Config and profile,
R-binning, and the JSON result artifact that the two production-fidelity ladders
(`resolution_ladder.py`: art_nlayer; `top_pressure_ladder.py`: the model top)
and `opacity_leave_one_out.py` emit.

Each script's `emit()` writes a JSON artifact under
`output/w39b_smc_retrieval/validation/` (the ignored output tree, beside the
run outputs) with the code and data provenance that ties the number to one state.

Nothing here imports jax, exojax, or the chemistry stack, so it stays cheap and
cannot perturb the measurement.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

# one copy of the git/hash primitives, owned by the provenance module
from retrieval_framework.provenance import (_data_identity, _repo_states, _versions,
                                            science_data_identity)

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent


PRODUCTION_SPEC = REPO / "runs" / "w39b_smc_retrieval" / "v5.yaml"


def production_config():
    """The Config of the PRODUCTION run (PRODUCTION_SPEC); no jax import."""
    from retrieval_framework.run_setup import load_spec
    return load_spec(PRODUCTION_SPEC).cfg


def production_profile(**overrides):
    """The forward profile the PRODUCTION case actually runs.

    A ladder must measure the model production uses, not a hand-copied
    approximation of it, so the profile comes from the case's own
    Config.profile().

    jax/chemistry are imported lazily so this module stays cheap at import.
    """
    prof = production_config().profile()
    prof.update(overrides)
    return prof


RESULTS = REPO / "output" / "w39b_smc_retrieval" / "validation"


def make_r_bins(wl_lo, wl_hi, R):
    n = max(2, int(np.ceil(np.log(wl_hi / wl_lo) * R)))
    return np.geomspace(wl_lo, wl_hi, n + 1)


def bin_trapz(wl, y, edges):
    """d(lambda)-weighted (local-trapezoid) bin means; NaN where empty."""
    w = np.empty_like(wl)
    w[1:-1] = 0.5 * (wl[2:] - wl[:-2]); w[0] = wl[1] - wl[0]; w[-1] = wl[-1] - wl[-2]
    idx = np.digitize(wl, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[b] = float(np.sum(w[sel] * y[sel]) / np.sum(w[sel]))
    return out


def _devices() -> list[str]:
    """JAX devices, if jax is already imported. Never imports it."""
    jax = sys.modules.get("jax")
    if jax is None:
        return []
    try:
        return [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    except RuntimeError:
        return []


def collect_provenance(resolved_config: dict | None = None) -> dict:
    """Everything needed to tie a number to an exact state."""
    repos = _repo_states(REPO.parent)
    # top_pressure_ladder nests its two grids under "production"/"extended"
    rc = (resolved_config or {}).get("production", resolved_config or {})
    prov = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "cwd": str(Path.cwd()),
        "repos": repos,
        "versions": _versions(),
        "jax_devices": _devices(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "hostname": socket.gethostname(),
        },
        # the data root, the VULCAN config and its network (no observations here)
        "data": _data_identity(RESULTS, rc),
        "env_overrides": {
            k: os.environ[k] for k in sorted(os.environ)
            if k.startswith(("SMC_", "VULCAN_", "JAX_", "XLA_", "VALIDATE_"))
        },
    }
    try:
        prov["user"] = getpass.getuser()
    except (KeyError, OSError):
        prov["user"] = None
    if resolved_config is not None:
        blob = json.dumps(resolved_config, sort_keys=True, default=str)
        prov["resolved_config"] = resolved_config
        prov["resolved_config_sha256"] = hashlib.sha256(
            blob.encode()).hexdigest()
        # CONTENT identity of the opacity/CIA files this measurement read, in
        # the same shape the run's target manifest records, so an artifact
        # measured against different data can be told apart from the run's.
        prov["science_data"] = science_data_identity(rc.get("molecules") or ())
    return prov


def emit(name: str, title: str, measurements: list[dict], status: str,
         summary: str, resolved_config: dict | None = None,
         out_dir: Path | None = None) -> Path:
    """Write `<name>.json` under validation/results/.

    `status` is PASS / FAIL / REPORT (REPORT = a measurement with no pass gate).
    Returns the JSON path.
    """
    if status not in ("PASS", "FAIL", "REPORT"):
        raise ValueError(f"status must be PASS/FAIL/REPORT, got {status!r}")
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": name,
        "title": title,
        "verdict": {"status": status, "summary": summary},
        "measurements": measurements,
        "provenance": collect_provenance(resolved_config),
    }
    jpath = out_dir / f"{name}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    logger.info(f"[artifact] wrote {jpath}")
    return jpath
