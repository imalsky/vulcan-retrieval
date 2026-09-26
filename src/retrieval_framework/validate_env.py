#!/usr/bin/env python3
"""validate_env.py -- one-command environment validation for the retrieval stack.

Jobs are READ-ONLY on the environment: all installs happen once in
tools/bootstrap_nas_env.pbs (NAS) or a local editable install, and every PBS
job runs this module first instead of pip. It aggregates ALL failures into one
report (exit 1) that names the remedy.

Checks, in import-order-safe sequence (vulcan_jax / retrieval_framework BEFORE
exojax -- vulcan_forward.vulcan_chem's guard raises if exojax is imported first):

  1. python interpreter version within the supported range (>=3.10);
  2. jax imports; backend + devices reported; GPU asserted with --require-gpu;
  3. vulcan_jax imports, resolves EDITABLE under <PROJECT_ROOT>/VULCAN-JAX, and
     the installed dist version matches the checkout's _version.py (a mismatch
     means the editable install predates a metadata change -- re-bootstrap);
  4. retrieval_framework same, under <PROJECT_ROOT>/vulcan-retrieval;
  4b. vulcan_forward same, under <PROJECT_ROOT>/vulcan-forward -- the shared
     engine every retrieval path imports;
  5. cross-repo pin: the installed vulcan-jax satisfies vulcan-retrieval's
     declared requirement (skipped with a warning if `packaging` is absent);
  6. exojax imports and matches vulcan-forward's pin;
  7. required data under <PROJECT_ROOT>/vulcan-retrieval/data/: the real
     spectrum CSVs, and the ExoMolOP k-tables and H2-H2 + H2-He CIA at the
     paths vulcan_forward.paths resolves ($VULCAN_FORWARD_OPACITY_CACHE wins);
  8. exogibbs imports and meets the floor the equilibrium cold seed needs;
  9. nautilus (nautilus-sampler) imports: run_nautilus needs it.

Usage:
    python -m retrieval_framework.validate_env <PROJECT_ROOT> [--require-gpu]

PROJECT_ROOT is the directory CONTAINING the VULCAN-JAX, vulcan-forward and
vulcan-retrieval checkouts (same meaning as $VULCAN_PROJECT_ROOT).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SUPPORTED_PYTHON = (3, 10)
EXOJAX_PIN = "2.2.3"  # keep in lockstep with vulcan-forward's pyproject.toml
EXOGIBBS_MIN = "0.6.0"  # the Gibbs minimizer behind vulcan_jax.ini_abun.eq_seed

_ERRORS: list[str] = []
_WARNINGS: list[str] = []


def _err(msg: str) -> None:
    _ERRORS.append(msg)
    print(f"ERROR: {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    _WARNINGS.append(msg)
    print(f"WARNING: {msg}")


def _ok(msg: str) -> None:
    print(f"ok: {msg}")


def _repo_version(repo: Path, pkg_dir: str) -> str | None:
    """Read __version__ out of <repo>/src/<pkg_dir>/_version.py without importing."""
    p = repo / "src" / pkg_dir / "_version.py"
    try:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', p.read_text(encoding="utf-8"))
        return m.group(1) if m else None
    except OSError:
        return None


def _check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) < SUPPORTED_PYTHON:
        _err(
            f"python {v.major}.{v.minor}.{v.micro} is older than the supported "
            f"floor {SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]} (pyproject requires-python)."
        )
    else:
        _ok(f"python {v.major}.{v.minor}.{v.micro} ({sys.executable})")


def _check_jax(require_gpu: bool) -> None:
    try:
        import jax
    except Exception as e:  # noqa: BLE001 - aggregate every failure loudly
        _err(f"jax failed to import: {e!r}. Re-run the bootstrap.")
        return
    backend = jax.default_backend()
    _ok(f"jax {jax.__version__} backend={backend} devices={jax.devices()}")
    if require_gpu and backend not in ("gpu", "cuda", "rocm"):
        _err(
            f"JAX backend is '{backend}', not GPU. On NAS: wrong node type, or a "
            "CPU jaxlib shadowed the env's GPU build (re-run the bootstrap, which "
            "pins jax/jaxlib during dependency resolution)."
        )


def _check_editable(pkg_import: str, dist_name: str, repo: Path, pkg_dir: str) -> None:
    try:
        mod = __import__(pkg_import)
    except Exception as e:  # noqa: BLE001
        _err(f"{pkg_import} failed to import: {e!r}. Run the bootstrap to install it.")
        return
    mod_path = Path(mod.__file__).resolve()
    expected = (repo / "src" / pkg_dir).resolve()
    if mod_path.parent != expected:
        _err(
            f"{pkg_import} resolves to {mod_path.parent}, not the checkout at "
            f"{expected}. A stale non-editable install is shadowing the tree; "
            "re-run the bootstrap."
        )
        return
    installed = getattr(mod, "__version__", None)
    checkout = _repo_version(repo, pkg_dir)
    if checkout is not None and installed != checkout:
        # __version__ is read live from the editable tree, so a mismatch here
        # means a half-updated checkout; the dist-metadata drift check is below.
        _err(
            f"{pkg_import} __version__ {installed} != checkout _version.py {checkout}."
        )
    from importlib import metadata

    try:
        dist_version = metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        _err(f"dist '{dist_name}' has no installed metadata; re-run the bootstrap.")
        return
    if checkout is not None and dist_version != checkout:
        _err(
            f"installed dist {dist_name}=={dist_version} but the checkout is at "
            f"{checkout}: the editable install predates a packaging-metadata "
            "change (version/deps/entry points). Re-run the bootstrap."
        )
        return
    _ok(f"{dist_name} {dist_version} editable at {mod_path.parent}")


def _check_cross_repo_pin() -> None:
    """Every sibling-engine requirement vulcan-retrieval declares (vulcan-jax
    AND vulcan-forward) must be satisfied by the INSTALLED sibling."""
    from importlib import metadata

    try:
        reqs = metadata.requires("vulcan-retrieval") or []
    except metadata.PackageNotFoundError:
        return  # already reported by _check_editable
    try:
        from packaging.requirements import Requirement
    except ImportError:
        _warn("`packaging` unavailable; cannot verify the sibling-engine requirements.")
        return
    for dist in ("vulcan-jax", "vulcan-forward"):
        spec = next((r for r in reqs if r.split()[0].startswith(dist)), None)
        if spec is None:
            continue
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue  # already reported by _check_editable
        req = Requirement(spec)
        if not req.specifier.contains(version, prereleases=True):
            _err(
                f"installed {dist} {version} does not satisfy "
                f"vulcan-retrieval's requirement '{spec}'. Pull/update the "
                f"{dist} checkout and re-run the bootstrap."
            )
        else:
            _ok(f"{dist} {version} satisfies '{spec}'")


def _check_exojax() -> None:
    try:
        import exojax
    except Exception as e:  # noqa: BLE001
        _err(f"exojax failed to import: {e!r}. Re-run the bootstrap.")
        return
    if exojax.__version__ != EXOJAX_PIN:
        _err(
            f"exojax {exojax.__version__} != pinned {EXOJAX_PIN} (vulcan-forward). "
            "Re-run the bootstrap."
        )
    else:
        _ok(f"exojax {exojax.__version__}")


def production_molecules(root: Path) -> tuple[str, ...]:
    """The case's own molecule list, read from case.py, never a copy."""
    import os
    from retrieval_framework import run_smc as _R
    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "gpu")
    cfg, _ = _R.make_config(root / "vulcan-retrieval" / "runs" / "w39b_smc_retrieval")
    return tuple(cfg.molecules)


def _check_data_tree(root: Path, prod: tuple[str, ...]) -> None:
    from vulcan_forward import paths

    data = root / "vulcan-retrieval" / "data"
    cm24 = data / "cm24_wasp39b"
    if not any(cm24.glob("*.csv")):
        _err(f"missing real spectrum CSVs in {cm24} (one-time data seed).")
    else:
        _ok(f"real spectrum CSVs present in {cm24}")
    # The engine's own path accessors, so the check reads the files the engine
    # reads ($VULCAN_FORWARD_OPACITY_CACHE included).
    paths.set_data_root(data)
    # Correlated-k tables: the PRODUCTION opacity path. A missing table raises
    # deep inside the RT build, minutes into a job, so catch it in preflight.
    try:
        ckdir = paths.exomolop_dir()
    except RuntimeError as e:
        _err(str(e))
        return
    missing_k = [m for m in prod if not (ckdir / f"{m}.ktable.h5").exists()]
    if missing_k:
        _err(
            f"missing ExoMolOP k-tables for {missing_k} in {ckdir}. Correlated-k "
            "over these tables is the only opacity path. Fetch once with: python -m "
            f"vulcan_forward.fetch_exomolop --molecules {','.join(prod)}"
        )
    else:
        _ok(f"ExoMolOP k-tables present for {len(prod)} molecules in {ckdir}")
    try:
        cia_files = (paths.cia_h2h2_file(), paths.cia_h2he_file())
    except RuntimeError as e:
        _err(str(e))
        return
    for f in cia_files:
        if not f.exists():
            _err(
                f"missing {f.name} under {f.parent} -- H2/He CIA is REQUIRED in "
                "every RT call (exojax_rt raises without it)."
            )


def _check_exogibbs() -> None:
    """The EQ cold seed minimizes the Gibbs energy through exogibbs: pure JAX,
    nothing to build per architecture, but a hard import of every cold solve."""
    remedy = f'pip install --user --no-deps "exogibbs=={EXOGIBBS_MIN}"'
    try:
        import exogibbs
    except Exception as e:  # noqa: BLE001
        _err(f"exogibbs failed to import: {e!r}. Install it: {remedy}")
        return
    got = str(getattr(exogibbs, "__version__", ""))
    parts = tuple(int(n) for n in re.findall(r"\d+", got)[:3])
    if parts < tuple(int(n) for n in EXOGIBBS_MIN.split(".")):
        _err(f"exogibbs {got or '?'} is below the {EXOGIBBS_MIN} floor the "
             f"equilibrium seed needs. Upgrade it: {remedy}")
    else:
        _ok(f"exogibbs {got}")


def _check_nautilus() -> None:
    """run_nautilus (PBS SAMPLER=nautilus) imports nautilus after the pipeline
    build, minutes into a job; catch a missing install here."""
    try:
        import nautilus
    except Exception as e:  # noqa: BLE001
        _err(f"nautilus failed to import: {e!r}. Install it: "
             'pip install --user "nautilus-sampler==1.0.6" "h5py>=3" '
             "(tools/bootstrap_nas_env.pbs does).")
        return
    _ok(f"nautilus {getattr(nautilus, '__version__', '?')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()

    if not (root / "vulcan-retrieval").is_dir() or not (root / "VULCAN-JAX").is_dir():
        _err(
            f"PROJECT_ROOT={root} must contain the vulcan-retrieval and "
            "VULCAN-JAX checkouts (the clone directory must be named VULCAN-JAX; "
            "the GitHub repo is jax-vulcan)."
        )
    else:
        _check_python()
        _check_jax(args.require_gpu)
        # vulcan_jax / retrieval_framework BEFORE exojax: vulcan_forward.vulcan_chem's
        # import-order guard raises if exojax comes first.
        _check_editable("vulcan_jax", "vulcan-jax", root / "VULCAN-JAX", "vulcan_jax")
        # the shared forward engine (chemistry + RT) is its own distribution
        _check_editable("vulcan_forward", "vulcan-forward",
                        root / "vulcan-forward", "vulcan_forward")
        _check_editable(
            "retrieval_framework", "vulcan-retrieval", root / "vulcan-retrieval", "retrieval_framework"
        )
        _check_cross_repo_pin()
        _check_exojax()
        _check_data_tree(root, production_molecules(root))
        _check_exogibbs()
        _check_nautilus()

    print()
    if _ERRORS:
        print(
            f"validate_env: FAIL ({len(_ERRORS)} error(s), {len(_WARNINGS)} warning(s)).\n"
            "Remedy: one-time bootstrap (the editable installs), then resubmit:\n"
            f"  cd {root / 'vulcan-retrieval'}\n"
            "  qsub tools/bootstrap_nas_env.pbs",
            file=sys.stderr,
        )
        return 1
    print(f"validate_env: PASS ({len(_WARNINGS)} warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
