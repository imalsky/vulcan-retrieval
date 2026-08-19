"""Provenance-bearing result artifacts for the production-fidelity checks.

The three convergence scripts (`resolution_ladder.py`, `top_pressure_ladder.py`,
`broadening_ab.py`) measure the three production choices that were made for
reasons other than accuracy:

  * `nu_pts = 1652` was chosen for GPU gradient MEMORY, not from a convergence
    result;
  * the ART grid extends one decade above the chemistry grid on a
    constant-VMR / isothermal CLAMP;
  * most molecular lines use terrestrial-AIR HITRAN widths even though the
    atmosphere is H2/He.

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
from retrieval_framework.certificate import _git, _sha256  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
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


# The four repositories whose code can move any of these numbers. Resolved
# relative to the workspace root (the directory containing this checkout).
_REPOSITORIES = {
    "jax-vulcan": ("jax-vulcan", "VULCAN-JAX"),
    "vulcan-forward": ("vulcan-forward",),
    "vulcan-retrieval": ("vulcan-retrieval",),
    "vulcan-jwst-tool": ("vulcan-jwst-tool",),
}


def _repo_state(repo: Path) -> dict | None:
    head = _git(repo, "rev-parse", "HEAD")
    if head is None:
        return None
    dirty = _git(repo, "status", "--porcelain") or ""
    return {"commit": head, "dirty": bool(dirty),
            "dirty_files": dirty.splitlines()[:20]}


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
    for env in ("VULCAN_FORWARD_DATA", "VULCAN_FORWARD_LINELISTS",
                "VULCAN_FORWARD_OPACITY_CACHE"):
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
        tree_dirs = {"exojax_linelists": Path(_fwd_paths.linelist_dir()),
                     "opacity_cache": Path(_fwd_paths.opacity_cache_dir())}
    except Exception as exc:                                # pragma: no cover
        out["engine_data_error"] = f"{type(exc).__name__}: {exc}"
        return out
    for sub in ("exojax_linelists", "opacity_cache"):
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
    workspace = REPO.parent
    repos = {}
    for name, directory_names in _REPOSITORIES.items():
        state = None
        for directory in directory_names:
            state = _repo_state(workspace / directory)
            if state is not None:
                state["checkout"] = directory
                break
        repos[name] = state
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
    for sub in ("exojax_linelists", "opacity_cache"):
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
