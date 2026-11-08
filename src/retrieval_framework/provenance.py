"""Run provenance: code, version and data identity, and the target manifest.

The target digest is the sha256 of the canonical manifest (``target_manifest``: resolved config, ordered priors,
observation arrays, opacity and CIA content, code commits and versions, the solver environment). A run writes the
manifest beside its outputs and binds its checkpoint to the digest, so a resume across a change is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable

    from retrieval_framework import config_schema as C
    from retrieval_framework.pipeline import Pipeline

__all__ = [
    "MANIFEST_FILE",
    "REPOSITORIES",
    "TARGET_FREE_KEYS",
    "archived_manifest_digest",
    "case_config",
    "case_data_file",
    "case_inputs",
    "manifest_digest",
    "science_data_identity",
    "target_digest",
    "target_manifest",
]

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
# Repository labels and their accepted checkout names; the NAS clone of VULCAN3 is VULCAN-JAX.
REPOSITORIES = {
    "VULCAN3": ("VULCAN3", "VULCAN-JAX", "jax-vulcan"),
    "vulcan-forward": ("vulcan-forward",),
    "vulcan-retrieval": ("vulcan-retrieval",),
}


def _sha256(path: Path) -> str:
    """Return the sha256 of one file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_raw(repo: Path, *args: str) -> str | None:
    """Return stdout on a zero exit, possibly empty, or None.

    ``git status --porcelain`` on a clean tree exits 0 with no bytes; provenance must not record a repo as clean because
    git was missing.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace", timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _git(repo: Path, *args: str) -> str | None:
    """Return stdout on a zero exit, or None when empty or failed."""
    out = _git_raw(repo, *args)
    return out or None


def _src_state(repo: Path) -> str | None:
    """Return the uncommitted src/ state, tracked diff and untracked files, or None when git could not run."""
    diff = _git_raw(repo, "diff", "HEAD", "--", "src")
    others = _git_raw(repo, "ls-files", "--others", "--exclude-standard", "--", "src")
    if diff is None or others is None:
        return None
    parts = [diff]
    for rel in sorted(p for p in others.splitlines() if p.strip()):
        f = repo / rel
        parts.append(f"?? {rel} {_sha256(f) if f.is_file() else 'gone'}")
    return "\n".join(parts)


def _repo_states(workspace: Path | None = None) -> dict:
    """Return commit and dirty state per repository; ``dirty`` is None when git could not run."""
    workspace = WORKSPACE if workspace is None else Path(workspace)
    out: dict[str, Any] = {}
    for name, directory_names in REPOSITORIES.items():
        repo = next(
            (
                workspace / directory
                for directory in directory_names
                if _git(workspace / directory, "rev-parse", "HEAD")
            ),
            None,
        )
        head = _git(repo, "rev-parse", "HEAD") if repo is not None else None
        if head is None or repo is None:
            out[name] = None
            continue
        dirty = _git_raw(repo, "status", "--porcelain")
        # Scoped to src/: an uncommitted solver edit is a different target, while editing a PBS script must not break a
        # chained resume. Untracked src files count: a new module is invisible to `git diff HEAD`.
        diff = _src_state(repo)
        out[name] = {
            "commit": head,
            "dirty": None if dirty is None else bool(dirty),
            "dirty_files": [] if dirty is None else dirty.splitlines()[:20],
            "src_diff": (None if diff is None else hashlib.sha256(diff.encode()).hexdigest()),
            "checkout": repo.name,
        }
    return out


def _versions() -> dict:
    """Return the interpreter, platform and package versions."""
    out: dict[str, str | None] = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("jax", "jaxlib", "numpy", "scipy", "exojax"):
        try:
            out[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            out[mod] = None
    for dist in ("vulcan3", "vulcan-forward", "vulcan-retrieval"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


CIA_TABLES = ("H2-H2_2011.cia", "H2-He_2011.cia")


def science_data_identity(molecules: Iterable[str]) -> dict[str, Any]:
    """Return the content identity of the data that defines the radiative model, by sha256.

    The part of the target manifest a validation artifact can also record: it binds which opacity data a ladder
    measured. The CIA tables are outside git, and a swapped H2-He table changes the continuum a retrieval fits. An
    absent table raises from its read: a run read these tables, so a target that cannot name them is not that run's.

    Parameters
    ----------
    molecules : iterable of str
        The species with a k-table.

    Returns
    -------
    dict
        ``opacity_sha256`` and ``cia_sha256``.
    """
    from vulcan_forward import exomolop as _exo, paths as _fwd_paths

    from retrieval_framework.forward import config as _fwd_config  # noqa: F401  # hands the engine its data root

    cia = (_fwd_paths.cia_h2h2_file(), _fwd_paths.cia_h2he_file())
    return {
        "opacity_sha256": {str(m): _sha256(_exo.table_path(m)) for m in molecules},
        "cia_sha256": {name: _sha256(p) for name, p in zip(CIA_TABLES, cia, strict=True)},
    }


def _vulcan3_root() -> Path:
    """Return the installed vulcan3 package directory, by find_spec: importing vulcan3 import-locks its network."""
    import importlib.util

    spec = importlib.util.find_spec("vulcan3")
    if spec is None or not spec.origin:
        raise ModuleNotFoundError("vulcan3 is not installed")
    return Path(spec.origin).resolve().parent


def _file_identity(path: Path) -> dict[str, str] | None:
    """Return a file's path and sha256, or None when absent."""
    return {"path": str(path), "sha256": _sha256(path)} if path.is_file() else None


def case_config(name: str) -> tuple[Path, dict]:
    """Return the path and settings of a vulcan3 case config, resolved as vulcan3 does from the working directory.

    A path or ``*.yaml``, else ``./configs/<name>.yaml``, else the packaged one; each key the file omits comes from the
    packaged ``default.yaml``.

    Parameters
    ----------
    name : str
        The case's config name or path.

    Returns
    -------
    tuple
        ``(path, settings)``.

    Raises
    ------
    FileNotFoundError
        If the config is not found.
    """
    pkg = _vulcan3_root()
    p = Path(name)
    if not (p.suffix in (".yaml", ".yml") or p.exists()):
        p = Path("configs") / f"{name}.yaml"
        if not p.is_file():
            p = pkg / "configs" / f"{name}.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"vulcan3 config {name!r} not found (resolved to {p})")
    default = yaml.safe_load((pkg / "configs" / "default.yaml").read_text())
    return p.resolve(), {**default, **(yaml.safe_load(p.read_text()) or {})}


def case_data_file(rel: str | Path) -> Path:
    """Return a case's data path, resolved as ``vulcan3.paths.resolve_data_path`` does.

    Parameters
    ----------
    rel : str or pathlib.Path
        The path in the config.

    Returns
    -------
    pathlib.Path
        Absolute, else under the working directory, else under the package.
    """
    rel = Path(rel)
    return (rel if rel.is_absolute() or rel.exists() else _vulcan3_root() / rel).resolve()


def case_inputs(name: str | None, cfg_overrides: dict) -> dict | None:
    """Return the sha256 of the case's vulcan3 config and of the atm and stellar-flux files it names.

    A path-style case lives outside the trees the code state binds, so these hashes are its only identity.

    Parameters
    ----------
    name : str
        The case's config name or path; empty returns None.
    cfg_overrides : dict
        Overrides of the file's settings.

    Returns
    -------
    dict or None
        ``config``, ``atm_file`` and ``sflux_file`` identities.
    """
    if not name:
        return None
    path, case = case_config(name)
    out = {"config": _file_identity(path)}
    for key in ("atm_file", "sflux_file"):
        out[key] = _file_identity(case_data_file(cfg_overrides.get(key, case[key])))
    return out


def _data_identity(out_dir: Path, cfg_dict: dict) -> dict:
    """Hash the observations and record the resolved data and VULCAN inputs."""
    ident: dict[str, Any] = {}
    obs = out_dir / "observations.npz"
    ident["observations"] = (
        {"path": str(obs), "sha256": _sha256(obs), "bytes": obs.stat().st_size} if obs.is_file() else None
    )

    # Resolve through the engine: paths.set_data_root takes precedence over
    # $VULCAN_FORWARD_DATA. Both are recorded, each under its own name.
    ident["VULCAN_FORWARD_DATA"] = os.environ.get("VULCAN_FORWARD_DATA")
    root = None
    try:
        # importing this module is what hands the engine this repo's data tree
        from vulcan_forward import paths as _fwd_paths

        from retrieval_framework.forward import config as _fwd_config  # noqa: F401  # hands the engine its data root

        root = Path(_fwd_paths.data_root())
    except Exception as exc:
        # Never silently null: a check that cannot run says why it could not.
        ident["engine_data_error"] = f"{type(exc).__name__}: {exc}"
    ident["data_root_resolved"] = str(root) if root is not None else None

    # the reaction network is a vendored VULCAN-JAX input; hash the exact file
    netfile = cfg_dict.get("network")
    if not netfile:
        ident["network_file_error"] = "config.json names no network"
    net = cfg_dict.get("vulcan_cfg_name")
    ident["vulcan_cfg_name"] = net
    try:
        ident["case_inputs"] = case_inputs(net, cfg_dict.get("cfg_overrides") or {})
    # broad: a provenance collector records any failure instead of raising
    except Exception as exc:
        ident["case_inputs_error"] = f"{type(exc).__name__}: {exc}"
    try:
        pkg = _vulcan3_root()
        npath = pkg / netfile if netfile else None
        if npath is not None and npath.is_file():
            ident["network_file"] = {"path": str(npath), "sha256": _sha256(npath)}
    # broad: a provenance collector records any failure instead of raising
    except Exception as exc:
        ident["network_file_error"] = str(exc)
    return ident


# Keys a chained job may differ in: the walltime governor is per job, and the RT chunk size is a batch split,
# numerically identical at any width. Everything else is bound.
TARGET_FREE_KEYS = frozenset(
    {
        "out_dir",
        "run_label",
        "walltime_seconds",
        "smc_rt_chunk",
    }
)


def target_manifest(cfg: C.Config, pipe: Pipeline) -> dict[str, Any]:
    """Return everything that defines the target a checkpoint's numbers belong to.

    A resume that changes any of this splices two densities into one evidence integral, which nothing downstream can
    detect.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.
    pipe : pipeline.Pipeline
        The pipeline, with its observations set.

    Returns
    -------
    dict
        The manifest.
    """
    from dataclasses import asdict

    cfg_bound = {k: v for k, v in asdict(cfg).items() if k not in TARGET_FREE_KEYS}

    # The likelihood's actual inputs, not the file paths that produced them.
    h = hashlib.sha256()
    for arr in (pipe.obs["wl"], pipe.obs["wl_lo"], pipe.obs["wl_hi"], pipe.obs_depth, pipe.obs_sigma):
        h.update(np.ascontiguousarray(np.asarray(arr, np.float64)).tobytes())
    h.update("\x00".join(map(str, pipe.obs.get("group", []))).encode())

    return {
        # Bumped when the manifest's fields change, or the evaluators' keep/reject or start logic changes the target.
        "manifest_version": 6,
        "config": cfg_bound,
        "params": {
            "names": list(pipe.names),
            "prior_types": list(pipe.prior_types),
            "prior_lo": np.asarray(pipe.param_prior_lo, float).tolist(),
            "prior_hi": np.asarray(pipe.param_prior_hi, float).tolist(),
        },
        "groups": list(pipe.groups),
        "observations_sha256": h.hexdigest(),
        "n_bin": int(pipe.n_bin),
        **science_data_identity(cfg.molecules),
        "case_inputs": case_inputs(cfg.vulcan_cfg_name, cfg.cfg_overrides),
        # Not dirty_files: that list churns with any scratch edit, while src_diff moves only when the code changes.
        "code": {
            r: None
            if s is None
            else {"commit": s.get("commit"), "dirty": s.get("dirty"), "src_diff": s.get("src_diff")}
            for r, s in _repo_states().items()
        },
        "versions": _versions(),
        # The solver moves every column at the convergence scale, so a resume must not switch it. The RHS form and the
        # ffi kernel's forced buffer variant are bound the same way.
        "vulcan3_solver": os.environ.get("VULCAN3_SOLVER", "fast"),
        "vulcan3_chem_rhs": os.environ.get("VULCAN3_CHEM_RHS", "codegen"),
        "vulcan3_bt_buffers": os.environ.get("VULCAN3_BT_BUFFERS") or "per-call",
    }


MANIFEST_FILE = "target_manifest.json"


def _canonical(manifest: dict) -> str:
    """Return the one serialization the digest is taken over, shared by the live digest and the archived re-hash.

    From version 6 a repository's ``dirty`` is recorded but not hashed: editing a PBS script must not break a chained
    resume, and ``src_diff`` binds the source. An older archived manifest hashes as it was written.
    """
    if manifest.get("manifest_version", 0) >= 6:
        code = {
            r: s if s is None else {k: v for k, v in s.items() if k != "dirty"} for r, s in manifest["code"].items()
        }
        manifest = {**manifest, "code": code}
    return json.dumps(manifest, sort_keys=True, default=str)


def target_digest(cfg: C.Config, pipe: Pipeline) -> str:
    """Return the sha256 of the canonical target manifest.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.
    pipe : pipeline.Pipeline
        The pipeline, with its observations set.

    Returns
    -------
    str
        The digest.
    """
    return manifest_digest(target_manifest(cfg, pipe))


def manifest_digest(manifest: dict) -> str:
    """Return the sha256 of a target manifest's canonical form."""
    return hashlib.sha256(_canonical(manifest).encode()).hexdigest()


def archived_manifest_digest(out_dir: Path) -> str | None:
    """Return the digest of the manifest archived in a run directory, or None if absent.

    A run started in an old directory can leave a new manifest beside old samples; re-hashing the archived document
    detects that.

    Parameters
    ----------
    out_dir : pathlib.Path
        The run's output directory.

    Returns
    -------
    str or None
        The digest, or a message starting ``unreadable:``.
    """
    p = Path(out_dir) / MANIFEST_FILE
    if not p.is_file():
        return None
    try:
        return manifest_digest(json.loads(p.read_text()))
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        return f"unreadable: {type(exc).__name__}: {exc}"
