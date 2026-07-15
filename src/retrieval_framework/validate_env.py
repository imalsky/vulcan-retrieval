#!/usr/bin/env python3
"""validate_env.py -- one-command environment validation for the retrieval stack.

Jobs are READ-ONLY on the environment: all installs happen once in
tools/bootstrap_nas_env.pbs (NAS) or a local editable install, and every PBS
job runs this module first instead of pip. It aggregates ALL failures into one
loud report (exit 1) that names the remedy, honoring the no-silent-fallbacks
rule.

Checks, in import-order-safe sequence (vulcan_jax / retrieval_framework BEFORE
exojax -- forward.vulcan_chem's guard raises if exojax is imported first):

  1. python interpreter version within the supported range (>=3.10);
  2. jax imports; backend + devices reported; GPU asserted with --require-gpu;
  3. vulcan_jax imports, resolves EDITABLE under <PROJECT_ROOT>/VULCAN-JAX, and
     the installed dist version matches the checkout's _version.py (a mismatch
     means the editable install predates a metadata change -- re-bootstrap);
  3b. vulcan_jax.conden exposes the live-T(P) condensation builder
     (make_conden_spec + build_conden_profile) -- a capability probe the
     version floor alone cannot guarantee (both pre- and post-conden checkouts
     once reported >=0.1.17-era versions);
  4. retrieval_framework same, under <PROJECT_ROOT>/vulcan-retrieval;
  5. cross-repo pin: the installed vulcan-jax satisfies vulcan-retrieval's
     declared requirement (skipped with a warning if `packaging` is absent);
  6. exojax imports and matches the pyproject pin;
  7. required data files under <PROJECT_ROOT>/vulcan-retrieval/data/ (real
     spectrum CSVs, cached CO ExoMol dir, H2-H2 + H2-He CIA; missing HITRAN
     line-list caches are a warning -- they re-download via the NAS proxy);
  8. a runnable FastChem binary for this node's architecture (exec-probed;
     $VULCAN_JAX_FASTCHEM_DIR first, then the checkout tree).

Usage:
    python -m retrieval_framework.validate_env <PROJECT_ROOT> [--require-gpu]

PROJECT_ROOT is the directory CONTAINING the VULCAN-JAX and vulcan-retrieval
checkouts (same meaning as $VULCAN_PROJECT_ROOT).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SUPPORTED_PYTHON = (3, 10)
EXOJAX_PIN = "2.2.3"  # keep in lockstep with pyproject.toml dependencies

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


def _check_conden_api() -> None:
    """The installed vulcan-jax must expose the live-T(P) condensation builder.

    The dist version alone is insufficient: the pre-conden 0.1.17 and the
    conden-capable 0.1.18 both report a version that satisfies the >=0.1.17
    era floor if an old checkout is shadowing, so probe the actual API. _prep
    rebuilds condensation on-graph via these two functions
    (conden.make_conden_spec + build_conden_profile); their absence means an
    old checkout with no live-T condensation support (re-pull VULCAN-JAX)."""
    try:
        from vulcan_jax import conden
    except Exception as e:  # noqa: BLE001 - aggregate every failure loudly
        _err(f"vulcan_jax.conden failed to import: {e!r}. Re-run the bootstrap.")
        return
    required = ("make_conden_spec", "build_conden_profile")
    missing = [name for name in required if not hasattr(conden, name)]
    if missing:
        _err(
            "installed vulcan-jax lacks live-T(P) condensation support "
            f"(vulcan_jax.conden missing {', '.join(missing)}): the checkout "
            "predates the on-graph conden builder (VULCAN-JAX 0.1.18). "
            "Pull/update the VULCAN-JAX checkout and re-run the bootstrap.")
    else:
        _ok("vulcan_jax.conden exposes make_conden_spec + build_conden_profile")


def _check_config_api() -> None:
    """The installed vulcan-jax must expose the YAML ``load_config`` API.

    ``forward.vulcan_chem`` builds every chemistry model through
    ``vulcan_jax.load_config(name)`` (YAML-only config, gravity from Mp/Rp).
    A checkout predating that migration has no ``load_config`` (it still shipped
    the deleted ``vulcan_cfg`` module), yet reports a version that can satisfy an
    old floor, so probe the actual API instead of trusting the version alone."""
    try:
        import vulcan_jax
    except Exception as e:  # noqa: BLE001 - aggregate every failure loudly
        _err(f"vulcan_jax failed to import: {e!r}. Re-run the bootstrap.")
        return
    if not hasattr(vulcan_jax, "load_config"):
        _err(
            "installed vulcan-jax has no `load_config`: the checkout predates the "
            "YAML-only config migration (gravity from Mp/Rp). forward.vulcan_chem "
            "requires it. Pull/update the VULCAN-JAX checkout and re-run the "
            "bootstrap.")
        return
    try:
        cfg = vulcan_jax.load_config("W39b")
    except Exception as e:  # noqa: BLE001
        _err(f"vulcan_jax.load_config('W39b') failed: {e!r}. Re-run the bootstrap.")
        return
    # Mp/Rp gravity is the load-bearing schema change the chemistry path assumes.
    if not (getattr(cfg, "Mp", None) and getattr(cfg, "Rp", None)):
        _err("vulcan_jax config 'W39b' lacks Mp/Rp (gravity schema); update VULCAN-JAX.")
    else:
        _ok("vulcan_jax.load_config exposes the YAML config API (W39b Mp/Rp present)")


def _check_cross_repo_pin() -> None:
    from importlib import metadata

    try:
        reqs = metadata.requires("vulcan-retrieval") or []
        vj_version = metadata.version("vulcan-jax")
    except metadata.PackageNotFoundError:
        return  # already reported by _check_editable
    spec = next((r for r in reqs if r.split()[0].startswith("vulcan-jax")), None)
    if spec is None:
        return
    try:
        from packaging.requirements import Requirement
    except ImportError:
        _warn(f"`packaging` unavailable; cannot verify '{spec}' against vulcan-jax {vj_version}.")
        return
    req = Requirement(spec)
    if not req.specifier.contains(vj_version, prereleases=True):
        _err(
            f"installed vulcan-jax {vj_version} does not satisfy "
            f"vulcan-retrieval's requirement '{spec}'. Pull/update the "
            "VULCAN-JAX checkout and re-run the bootstrap."
        )
    else:
        _ok(f"vulcan-jax {vj_version} satisfies '{spec}'")


def _check_exojax() -> None:
    try:
        import exojax
    except Exception as e:  # noqa: BLE001
        _err(f"exojax failed to import: {e!r}. Re-run the bootstrap.")
        return
    if exojax.__version__ != EXOJAX_PIN:
        _err(
            f"exojax {exojax.__version__} != pinned {EXOJAX_PIN} (pyproject). "
            "Re-run the bootstrap; the pin is deliberate (see requirements-hpc.txt)."
        )
    else:
        _ok(f"exojax {exojax.__version__}")


def _check_data_tree(root: Path) -> None:
    data = root / "vulcan-retrieval" / "data"
    cm24 = data / "cm24_wasp39b"
    if not any(cm24.glob("*.csv")):
        _err(f"missing real spectrum CSVs in {cm24} (one-time data seed; see CLAUDE.md).")
    else:
        _ok(f"real spectrum CSVs present in {cm24}")
    lldir = data / "exojax_linelists"
    missing = [
        m
        for m in ("H2O", "CO2", "CH4", "SO2", "HCN", "C2H2", "H2S")
        if not (lldir / f"{m}.h5").exists()
    ]
    if missing:
        _warn(
            f"HITRAN caches missing for {missing} in {lldir}; first run will "
            "download via the NAS proxy."
        )
    codir = data / "opacity_cache"
    if not (codir / "CO" / "12C-16O" / "Li2015").is_dir():
        _err(f"missing cached CO ExoMol dir under {codir} (one-time data seed).")
    for cia in ("H2-H2_2011.cia", "H2-He_2011.cia"):
        if not (codir / cia).exists():
            _err(
                f"missing {cia} under {codir} -- H2/He CIA is REQUIRED in every "
                "RT call (exojax_rt raises without it)."
            )


def _fastchem_runnable(tree: Path) -> bool:
    """Exec-probe <tree>/fastchem; an OSError means wrong architecture/missing."""
    binary = tree / "fastchem"
    if not binary.exists():
        return False
    try:
        subprocess.run([str(binary)], cwd=str(tree), timeout=15, capture_output=True)
    except OSError:
        return False
    except Exception:  # noqa: BLE001 - ran but complained (no args): executable is fine
        pass
    return True


def _check_fastchem(root: Path) -> None:
    cands: list[Path] = []
    env_dir = os.environ.get("VULCAN_JAX_FASTCHEM_DIR")
    if env_dir:
        cands.append(Path(env_dir))
    cands.append(root / "VULCAN-JAX" / "src" / "vulcan_jax" / "fastchem_vulcan")
    for c in cands:
        if _fastchem_runnable(c):
            _ok(f"FastChem binary runnable at {c}")
            return
    _err(
        "no runnable FastChem binary for this architecture (probed: "
        + ", ".join(str(c) for c in cands)
        + "). The bootstrap builds it (`make` in VULCAN-JAX/src/vulcan_jax/"
        "fastchem_vulcan on the target node type)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_root", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()

    if not (root / "vulcan-retrieval").is_dir() or not (root / "VULCAN-JAX").is_dir():
        _err(
            f"PROJECT_ROOT={root} must contain the vulcan-retrieval and "
            "VULCAN-JAX checkouts (the VULCAN-JAX clone target name is "
            "load-bearing; the GitHub repo is named jax-vulcan)."
        )
    else:
        _check_python()
        _check_jax(args.require_gpu)
        # vulcan_jax / retrieval_framework BEFORE exojax: forward.vulcan_chem's
        # import-order guard raises if exojax comes first.
        _check_editable("vulcan_jax", "vulcan-jax", root / "VULCAN-JAX", "vulcan_jax")
        _check_config_api()
        _check_conden_api()
        _check_editable(
            "retrieval_framework", "vulcan-retrieval", root / "vulcan-retrieval", "retrieval_framework"
        )
        _check_cross_repo_pin()
        _check_exojax()
        _check_data_tree(root)
        _check_fastchem(root)

    print()
    if _ERRORS:
        print(
            f"validate_env: FAIL ({len(_ERRORS)} error(s), {len(_WARNINGS)} warning(s)).\n"
            "Remedy: one-time bootstrap (installs + FastChem build), then resubmit:\n"
            f"  cd {root / 'vulcan-retrieval'}\n"
            "  qsub tools/bootstrap_nas_env.pbs",
            file=sys.stderr,
        )
        return 1
    print(f"validate_env: PASS ({len(_WARNINGS)} warning(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
