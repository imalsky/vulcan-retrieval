"""Result artifacts for the two production-fidelity ladders
(`resolution_ladder.py`: art_nlayer; `top_pressure_ladder.py`: the model top).

Each script's `emit()` writes a JSON artifact under `validation/results/` with
the code and data provenance that ties the number to one state.

Nothing here imports jax, exojax, or the chemistry stack, so it stays cheap and
cannot perturb the measurement.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

# one copy of the git/hash primitives, owned by the certificate module
from retrieval_framework.certificate import _repo_states, science_data_identity

REPO = Path(__file__).resolve().parent.parent


def production_config():
    """The Config the PRODUCTION case actually runs (gpu preset).

    jax/chemistry are imported lazily so this module stays cheap at import.
    """
    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "gpu")
    from retrieval_framework import run_smc as _R
    cfg, preset = _R.make_config(REPO / "runs" / "w39b_smc_retrieval")
    if preset != "gpu":
        raise RuntimeError(
            f"production_config needs the gpu preset, got {preset!r}; "
            "unset SMC_RETRIEVAL_PRESET or set it to 'gpu'")
    return cfg


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
RESULTS = REPO / "validation" / "results"


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


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for mod in ("jax", "jaxlib", "numpy", "scipy", "exojax"):
        try:
            out[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            out[mod] = None
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


def _data_identity() -> dict:
    """Identity of the opacity trees the RT reads.

    Hashing tens of gigabytes is off the table; the resolved real path plus a
    (count, total bytes, newest mtime) summary changes whenever a tree is
    swapped, extended, or regenerated, which is what matters here.
    """
    out = {}
    for env in ("VULCAN_FORWARD_DATA", "VULCAN_FORWARD_OPACITY_CACHE"):
        out[env] = os.environ.get(env)
    # The env vars above are recorded for transparency only. This repo hands
    # the engine its tree via paths.set_data_root, which takes precedence, so
    # the trees must be resolved through the engine to be the ones a run read.
    tree_dirs = {}
    try:
        # importing this module is what hands the engine this repo's data tree
        from retrieval_framework.forward import config as _fwd_config  # noqa: F401
        from vulcan_forward import paths as _fwd_paths
        out["data_root_resolved"] = str(_fwd_paths.data_root())
        tree_dirs = {"opacity_cache": Path(_fwd_paths.opacity_cache_dir()),
                     "exomolop": Path(_fwd_paths.exomolop_dir())}
    # broad: a provenance collector records any failure instead of raising
    except Exception as exc:                                # pragma: no cover
        out["engine_data_error"] = f"{type(exc).__name__}: {exc}"
        return out
    for sub in ("opacity_cache", "exomolop"):
        p = tree_dirs[sub]
        if not p.is_dir():
            out[sub] = None
            continue
        n = tot = 0
        newest = 0.0
        for f in p.rglob("*"):
            if f.is_file():
                st = f.stat()
                n += 1
                tot += st.st_size
                newest = max(newest, st.st_mtime)
        out[sub] = {"path": str(p.resolve()), "files": n, "bytes": tot,
                    "newest_mtime_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest)) if n else None}
    return out


def collect_provenance(resolved_config: dict | None = None) -> dict:
    """Everything needed to tie a number to an exact state."""
    repos = _repo_states(REPO.parent)
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
        "data": _data_identity(),
        "env_overrides": {
            k: os.environ[k] for k in sorted(os.environ)
            if k.startswith(("SMC_", "VULCAN_", "JAX_", "XLA_", "PROBE_",
                             "CALIBRATE_", "VALIDATE_"))
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
        # the same shape the run's target manifest records, so validate() can
        # refuse an artifact measured against different data. The tree summary
        # in prov["data"] cannot do that job -- it carries mtimes.
        # top_pressure_ladder nests its two grids under "production"/"extended";
        # certificate._validation_artifacts unwraps the same way, so the molecule
        # list is found in both shapes rather than silently reading as empty.
        _rc = resolved_config.get("production", resolved_config)
        prov["science_data"] = science_data_identity(
            _rc.get("molecules") or ())
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
    print(f"\n[artifact] wrote {jpath}")
    return jpath
