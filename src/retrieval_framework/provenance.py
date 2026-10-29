"""Run provenance: code, version and data identity, and the target manifest.

The target digest is the sha256 of the canonical target manifest
(``target_manifest``: resolved config, ordered priors, observation arrays,
opacity and CIA content, code commits and versions, the solver env choices). A
run writes the manifest beside its outputs and binds its checkpoint to the
digest, so a resume across a change is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
# Canonical repository labels plus accepted checkout names: the repository is
# VULCAN3 (formerly jax-vulcan), the NAS clone directory is VULCAN-JAX.
REPOSITORIES = {
    "VULCAN3": ("VULCAN3", "VULCAN-JAX", "jax-vulcan"),
    "vulcan-forward": ("vulcan-forward",),
    "vulcan-retrieval": ("vulcan-retrieval",),
    # jwst-transit-authority is not a dependency of a retrieval run, so it is
    # not bound.
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_GIT_FAILED = object()   # git did not run (or exited non-zero): NOT "clean"


def _git_raw(repo: Path, *args: str):
    """Stdout on a zero exit -- possibly empty -- or `_GIT_FAILED`.

    `git status --porcelain` on a clean tree exits 0 with zero bytes, which is
    indistinguishable from a failed call if both collapse to None. Provenance
    must not record a repo as clean because git was missing.
    """
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, errors="replace",
                           timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return _GIT_FAILED
    return r.stdout.strip() if r.returncode == 0 else _GIT_FAILED


def _git(repo: Path, *args: str) -> str | None:
    out = _git_raw(repo, *args)
    return out or None if out is not _GIT_FAILED else None


def _src_state(repo: Path):
    """Content of the uncommitted src/ state: tracked diff + untracked files.

    Returns `_GIT_FAILED` if git could not run, so "unknown" never reads as
    "unmodified".
    """
    diff = _git_raw(repo, "diff", "HEAD", "--", "src")
    others = _git_raw(repo, "ls-files", "--others", "--exclude-standard", "--", "src")
    if diff is _GIT_FAILED or others is _GIT_FAILED:
        return _GIT_FAILED
    parts = [diff]
    for rel in sorted(p for p in others.splitlines() if p.strip()):
        f = repo / rel
        parts.append(f"?? {rel} {_sha256(f) if f.is_file() else 'gone'}")
    return "\n".join(parts)


def _repo_states(workspace: Path | None = None) -> dict:
    """Commit + dirty state per repository. `dirty` is None when git could not
    be run, so "unknown" never reads as "clean"."""
    workspace = WORKSPACE if workspace is None else Path(workspace)
    out = {}
    for name, directory_names in REPOSITORIES.items():
        repo = next((workspace / directory for directory in directory_names
                     if _git(workspace / directory, "rev-parse", "HEAD")),
                    None)
        head = _git(repo, "rev-parse", "HEAD") if repo is not None else None
        if head is None:
            out[name] = None
            continue
        dirty = _git_raw(repo, "status", "--porcelain")
        failed = dirty is _GIT_FAILED
        # Content hash of the uncommitted SOURCE state, scoped to src/: an
        # uncommitted solver edit is a different target and must move the
        # digest, while editing a PBS or plotting script mid-campaign must not
        # break a chained resume. Untracked src files are hashed too: a new
        # imported module is invisible to `git diff HEAD`.
        diff = _src_state(repo)
        out[name] = {"commit": head,
                     "dirty": None if failed else bool(dirty),
                     "dirty_files": [] if failed else dirty.splitlines()[:20],
                     "src_diff": (None if diff is _GIT_FAILED else
                                  hashlib.sha256(diff.encode()).hexdigest()),
                     "checkout": repo.name}
    return out


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
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


def _cia_identity() -> dict:
    """Exact hashes of the two CIA tables, at the paths the engine reads.

    They are direct radiative-transfer inputs, small enough to hash exactly and
    too important to hide inside a directory file-count: a swapped H2-He table
    leaves the tree summary unchanged while changing the continuum the retrieval
    fits. They and the ExoMolOP k-tables are the only target-affecting inputs NOT tracked in git --
    everything else (network, baseline T-P, Kzz, elemental abundances) is
    vendored in vulcan3 and therefore bound by its commit.
    """
    try:
        from retrieval_framework.forward import config as _fwd_config  # noqa: F401
        from vulcan_forward import paths as _fwd_paths
        files = dict(zip(CIA_TABLES, (_fwd_paths.cia_h2h2_file(),
                                      _fwd_paths.cia_h2he_file())))
    except (ImportError, RuntimeError):                     # pragma: no cover
        files = {}
    out = {}
    for name in CIA_TABLES:
        p = files.get(name)
        out[name] = ({"path": str(p.resolve()), "sha256": _sha256(p),
                      "bytes": p.stat().st_size}
                     if p is not None and p.is_file() else None)
    return out


def science_data_identity(molecules) -> dict:
    """CONTENT identity of the data inputs that define the radiative model.

    The per-molecule k-tables plus the two CIA tables, by sha256. This is the
    part of the target manifest a validation artifact can also record -- an
    artifact has no observations or priors, but it reads exactly these files, so
    binding them is what tells a ladder measured against different opacity data
    from the run's. Tree summaries (counts, mtimes) are not used: they churn on
    any cache write.
    """
    opa = {}
    try:
        from retrieval_framework.forward import config as _fwd_config  # noqa: F401
        from vulcan_forward import exomolop as _exo
        for m in molecules:
            f = _exo.table_path(m)
            opa[str(m)] = _sha256(f) if f.is_file() else None
    # broad: a provenance collector records any failure instead of raising
    except Exception as exc:                                # pragma: no cover
        opa["error"] = f"{type(exc).__name__}: {exc}"
    return {"opacity_sha256": opa,
            "cia_sha256": {k: (v or {}).get("sha256")
                           for k, v in _cia_identity().items()}}


def _data_identity(out_dir: Path, cfg_dict: dict) -> dict:
    """Hash observations and record the resolved data and VULCAN inputs."""
    ident = {}
    obs = out_dir / "observations.npz"
    ident["observations"] = ({"path": str(obs), "sha256": _sha256(obs),
                              "bytes": obs.stat().st_size}
                             if obs.is_file() else None)

    # Resolve through the engine: paths.set_data_root takes precedence over
    # $VULCAN_FORWARD_DATA. Both are recorded, each under its own name.
    ident["VULCAN_FORWARD_DATA"] = os.environ.get("VULCAN_FORWARD_DATA")
    root = None
    try:
        # importing this module is what hands the engine this repo's data tree
        from retrieval_framework.forward import config as _fwd_config  # noqa: F401
        from vulcan_forward import paths as _fwd_paths
        root = Path(_fwd_paths.data_root())
    except Exception as exc:                                # pragma: no cover
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
        # find_spec, NOT import: importing vulcan3 parses and import-locks
        # its reaction network, and this identity collector only needs the
        # package path.
        import importlib.util
        spec = importlib.util.find_spec("vulcan3")
        if spec is None or not spec.origin:
            raise ModuleNotFoundError("vulcan3 is not installed")
        pkg = Path(spec.origin).resolve().parent
        cfg_yaml = pkg / "configs" / f"{net}.yaml"
        if cfg_yaml.is_file():
            ident["vulcan_config_yaml"] = {"path": str(cfg_yaml),
                                           "sha256": _sha256(cfg_yaml)}
        npath = pkg / netfile if netfile else None
        if npath is not None and npath.is_file():
            ident["network_file"] = {"path": str(npath), "sha256": _sha256(npath)}
    # broad: a provenance collector records any failure instead of raising
    except Exception as exc:                                # pragma: no cover
        ident["network_file_error"] = str(exc)
    return ident


# --- target identity ---------------------------------------------------------
# Keys a CHAINED job may legitimately differ in: none of them changes the target
# density or the numbers a resume carries. The walltime governor is per JOB;
# the RT chunk sizes are batch splits that are numerically identical at any
# width; the rest is bookkeeping. Everything NOT listed here is bound.
TARGET_FREE_KEYS = frozenset({
    "out_dir", "run_label", "walltime_seconds",
    "smc_rt_chunk",
})


def target_manifest(cfg, pipe) -> dict:
    """Everything that defines the TARGET a checkpoint's numbers belong to.

    A resume that changes any of this splices two different densities into one
    evidence integral, which nothing downstream can detect.
    """
    from dataclasses import asdict
    cfg_bound = {k: v for k, v in asdict(cfg).items() if k not in TARGET_FREE_KEYS}

    # The likelihood's actual inputs, not the file paths that produced them.
    h = hashlib.sha256()
    for arr in (pipe.obs["wl"], pipe.obs["wl_lo"], pipe.obs["wl_hi"],
                pipe.obs_depth, pipe.obs_sigma):
        h.update(np.ascontiguousarray(np.asarray(arr, np.float64)).tobytes())
    h.update("\x00".join(map(str, pipe.obs.get("group", []))).encode())

    return {
        # bumped when the manifest's fields change, or when the evaluators'
        # keep/reject or start logic changes the target at a fixed config
        "manifest_version": 4,
        "config": cfg_bound,
        "params": {"names": list(pipe.names),
                   "prior_types": list(pipe.prior_types),
                   "prior_lo": np.asarray(pipe.param_prior_lo, float).tolist(),
                   "prior_hi": np.asarray(pipe.param_prior_hi, float).tolist()},
        "groups": list(pipe.groups),
        "observations_sha256": h.hexdigest(),
        "n_bin": int(pipe.n_bin),
        **science_data_identity(cfg.molecules),
        # commit + dirty flag + the src/ diff hash. NOT dirty_files: that list
        # churns with any scratch edit and would refuse every resume in a
        # campaign without identifying a different code state, while src_diff
        # moves only when the code that defines the target actually changes.
        "code": {r: None if s is None else
                 {"commit": s.get("commit"), "dirty": s.get("dirty"),
                  "src_diff": s.get("src_diff")}
                 for r, s in _repo_states().items()},
        "versions": _versions(),
        # the solver moves every column at the convergence scale (fast vs ffi
        # agree to residual, not bitwise), so a resume must not switch it. The
        # chemistry RHS form and the ffi kernel's forced buffer variant are
        # numerical choices of the same solve, bound the same way (the
        # defaults: codegen RHS; unset = the kernel chooses per call).
        "vulcan3_solver": os.environ.get("VULCAN3_SOLVER", "fast"),
        "vulcan3_chem_rhs": os.environ.get("VULCAN3_CHEM_RHS", "codegen"),
        "vulcan3_bt_buffers": os.environ.get("VULCAN3_BT_BUFFERS") or "per-call",
    }


MANIFEST_FILE = "target_manifest.json"


def _canonical(manifest: dict) -> str:
    """The one serialization the digest is taken over.

    Shared by the live digest and by the re-hash of the ARCHIVED manifest, so
    the two can never drift into disagreeing about what the digest covers.
    """
    return json.dumps(manifest, sort_keys=True, default=str)


def target_digest(cfg, pipe) -> str:
    return hashlib.sha256(_canonical(target_manifest(cfg, pipe)).encode()).hexdigest()


def archived_manifest_digest(out_dir: Path) -> str | None:
    """Digest of the manifest ARCHIVED in a run directory, or None if absent.

    The run directory is written before the sampler runs, so a run started in
    an old directory can leave a NEW manifest beside OLD samples. Re-hashing the
    archived document is what detects that: the samples carry the digest of the
    run that wrote them, and only the manifest dissents.
    """
    p = Path(out_dir) / MANIFEST_FILE
    if not p.is_file():
        return None
    try:
        return hashlib.sha256(
            _canonical(json.loads(p.read_text())).encode()).hexdigest()
    except (OSError, ValueError) as exc:                    # pragma: no cover
        return f"unreadable: {type(exc).__name__}: {exc}"
