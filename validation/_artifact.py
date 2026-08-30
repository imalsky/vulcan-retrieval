"""Provenance-bearing result artifacts for the production-fidelity checks.

The two convergence scripts (`resolution_ladder.py`, `top_pressure_ladder.py`)
measure the two production choices that were made for reasons other than
accuracy:

  * `art_nlayer = 67` (the ART vertical grid; the spectral grid is fixed by the
    ExoMolOP tables) was chosen for GPU gradient MEMORY, not from a convergence
    result;
  * the model top (ART_PTOP_BAR, both grids) was set by band saturation
    before it was set by a convergence result.

A verdict printed to a terminal and lost is not evidence, and
"the script exists" is not the same as "the check passed at production
settings". This module gives each script one `emit()` call that writes a JSON
artifact plus a short Markdown summary under `validation/results/`, carrying
enough provenance to tie the number to an exact code and data state.

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
from retrieval_framework.certificate import (  # noqa: F401
    _git, _repo_states, _sha256, science_data_identity)

REPO = Path(__file__).resolve().parent.parent


def production_config():
    """The Config the PRODUCTION case actually runs (gpu preset).

    jax/chemistry are imported lazily so this module stays cheap at import.
    """
    import os
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
    approximation of it. The shipped artifacts were built from
    `forward.config.FULL` plus a few overrides; FULL carries yconv_cri=1e-3
    while the retrieval schema default (and so production) is 1e-2, and the
    molecule list was copied by hand -- so both ladders certified a different
    convergence behaviour and a different opacity model than any real run.
    Deriving from the case's own Config.profile() removes that whole class.

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
        except Exception:
            out[mod] = None
    return out


def _devices() -> list[str]:
    """JAX devices, if jax is already imported. Never imports it."""
    jax = sys.modules.get("jax")
    if jax is None:
        return []
    try:
        return [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    except Exception:
        return []


def _data_identity() -> dict:
    """Identity of the line-list / opacity trees the RT actually reads.

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
    except Exception:
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


def _md(name: str, payload: dict) -> str:
    prov = payload["provenance"]
    verdict = payload["verdict"]
    lines = [
        f"# {payload['title']}",
        "",
        f"**VERDICT: {verdict['status']}** -- {verdict['summary']}",
        "",
        f"Generated {prov['generated_utc']} by `{prov['command']}`.",
        "",
        "## Measurements",
        "",
        "| quantity | value | gate |",
        "|---|---|---|",
    ]
    for m in payload["measurements"]:
        # measurement names carry |Delta| style math, which would split the cell
        esc = str(m["name"]).replace("|", "\\|")
        val = str(m["value"]).replace("|", "\\|")
        gate = str(m.get("gate", "--")).replace("|", "\\|")
        lines.append(f"| {esc} | {val} | {gate} |")
    lines += ["", "## Provenance", "", "| key | value |", "|---|---|"]
    for repo, state in prov["repos"].items():
        if state is None:
            lines.append(f"| {repo} | MISSING |")
            continue
        mark = " (DIRTY)" if state["dirty"] else ""
        lines.append(f"| {repo} | {state['commit'][:12]}{mark} |")
    for k in ("jax", "jaxlib", "numpy", "exojax", "python"):
        lines.append(f"| {k} | {prov['versions'].get(k)} |")
    lines.append(f"| devices | {', '.join(prov['jax_devices']) or 'none'} |")
    lines.append(f"| host | {prov['hardware']['hostname']} "
                 f"({prov['hardware']['platform']}) |")
    data = prov.get("data", {})
    for sub in ("opacity_cache", "exomolop"):
        d = data.get(sub)
        if isinstance(d, dict):
            lines.append(f"| {sub} | {d['files']} files, {d['bytes']} bytes, "
                         f"newest {d['newest_mtime_utc']} |")
    if prov.get("resolved_config_sha256"):
        lines.append(f"| resolved config sha256 | "
                     f"{prov['resolved_config_sha256'][:16]}... |")
    lines += ["", f"Machine-readable: `{name}.json`", ""]
    return "\n".join(lines)


def emit(name: str, title: str, measurements: list[dict], status: str,
         summary: str, resolved_config: dict | None = None,
         out_dir: Path | None = None) -> Path:
    """Write `<name>.json` and `<name>.md` under validation/results/.

    `status` is PASS / FAIL / REPORT (REPORT = a measurement with no pass gate,
    such as the air-vs-H2/He A/B, whose output is a decision input rather than a
    threshold). Returns the JSON path.
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
    (out_dir / f"{name}.md").write_text(_md(name, payload))
    print(f"\n[artifact] wrote {jpath}")
    print(f"[artifact] wrote {out_dir / (name + '.md')}")
    return jpath
