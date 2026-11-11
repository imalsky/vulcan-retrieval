"""Shared helpers for the validation scripts: a case's Config, R-binning and the JSON artifact.

``emit()`` writes into the folder it is given, beside the case or the run it measured, with the code and data
provenance that ties the number to one state. Imports no jax, exojax or chemistry, so it cannot perturb the measurement.
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
from retrieval_framework.provenance import _data_identity, _repo_states, _sha256, _versions, science_data_identity

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parent.parent


def spec_case(path):
    """Return the Config of the run specification at ``path`` and the folder its validation artifacts go to.

    The folder is ``output/<case>/validation``, beside the case's runs. No jax import.
    """
    from retrieval_framework.run_setup import load_spec, output_base

    spec = load_spec(path)
    return spec.cfg, output_base(spec.path.parent) / "validation"


def make_r_bins(wl_lo, wl_hi, R):
    """Return the edges of constant-resolving-power bins across [wl_lo, wl_hi], um."""
    n = max(2, int(np.ceil(np.log(wl_hi / wl_lo) * R)))
    return np.geomspace(wl_lo, wl_hi, n + 1)


def bin_trapz(wl, y, edges):
    """d(lambda)-weighted (local-trapezoid) bin means; NaN where empty."""
    w = np.empty_like(wl)
    w[1:-1] = 0.5 * (wl[2:] - wl[:-2])
    w[0] = wl[1] - wl[0]
    w[-1] = wl[-1] - wl[-2]
    idx = np.digitize(wl, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[b] = float(np.sum(w[sel] * y[sel]) / np.sum(w[sel]))
    return out


def worst_ppm(a, b) -> float:
    """Return max|a - b| of two binned spectra, ppm: inf if a bin is finite on one side only, or if none is finite.

    A bin empty on both sides is a bin the model does not cover. Dropping a bin finite on one side would pass a NaN.
    """
    fa, fb = np.isfinite(a), np.isfinite(b)
    if np.any(fa != fb) or not fa.any():
        return float("inf")
    return float(1e6 * np.max(np.abs(a[fa] - b[fa])))


def _devices() -> list[str]:
    """Return the JAX devices if jax is already imported; never imports it."""
    jax = sys.modules.get("jax")
    if jax is None:
        return []
    try:
        return [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    except RuntimeError:
        return []


def collect_provenance(resolved_config: dict | None, data_dir: Path) -> dict:
    """Return everything needed to tie a number to an exact code, data and machine state.

    ``data_dir`` is where the measured case's ``observations.npz`` sits, a run's folder; a config that names a data
    source also hashes its product files.
    """
    repos = _repo_states(REPO.parent)
    # top_pressure_ladder nests its two grids under "case"/"extended"
    rc = (resolved_config or {}).get("case", resolved_config or {})
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
        # the data root, the VULCAN config, its network and the observations.npz in data_dir, if any
        "data": _data_identity(Path(data_dir), rc),
        "env_overrides": {
            k: os.environ[k]
            for k in sorted(os.environ)
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
        prov["resolved_config_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
        # Content identity of the opacity and CIA files this measurement read, in the shape the run's target
        # manifest records, so an artifact measured on other data can be told apart. A comparison of saved
        # results names no molecules and reads no table; its solves' provenance is in resolved_config.
        if rc.get("molecules"):
            prov["science_data"] = science_data_identity(rc["molecules"])
        if rc.get("obs_dir") and rc.get("obs_products"):
            files = sorted({f for group in rc["obs_products"].values() for f in group})
            prov["observation_products"] = {f: _sha256(Path(rc["obs_dir"]) / f) for f in files}
    return prov


def emit(
    name: str,
    title: str,
    measurements: list[dict],
    status: str,
    summary: str,
    resolved_config: dict | None = None,
    *,
    out_dir: Path,
) -> Path:
    """Write ``<name>.json`` under ``out_dir`` and return its path.

    ``status`` is PASS, FAIL or REPORT, a measurement with no pass gate.
    """
    if status not in ("PASS", "FAIL", "REPORT"):
        raise ValueError(f"status must be PASS/FAIL/REPORT, got {status!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": name,
        "title": title,
        "verdict": {"status": status, "summary": summary},
        "measurements": measurements,
        "provenance": collect_provenance(resolved_config, out_dir),
    }
    jpath = out_dir / f"{name}.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    logger.info("[artifact] wrote %s", jpath)
    return jpath
