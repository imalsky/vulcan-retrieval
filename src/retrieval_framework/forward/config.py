"""Retrieval-side configuration for the shared vulcan-forward engine.

Pure constants + paths: NO heavy imports here (no jax, no vulcan3, no exojax),
so this module is safe to import before the env-order-sensitive VULCAN-JAX setup
runs.

The shared physics constants and the molecule/opacity table live in
``vulcan_forward.constants``; import them from there. What is local: this
repo's filesystem layout and the WASP-39 b case constants.
Importing this module hands the engine its data root (see the paths
section), so the opacity caches live in this repo's data/ tree.
"""
from __future__ import annotations

import os
from pathlib import Path

from vulcan_forward import paths as _fwd_paths

# Paths
# VULCAN_PROJECT_ROOT = the directory CONTAINING the vulcan-retrieval/ checkout
# (and, for HPC runs + manuscript figures, its siblings VULCAN-JAX/ and jax_paper/).
# The explicit env var wins; otherwise the repo root is inferred from this file's
# location inside an editable checkout. A bare site-packages install cannot infer
# it and must set the env var (checked below).
_env_root = os.environ.get("VULCAN_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).expanduser()
    REPO_DIR = PROJECT_ROOT / "vulcan-retrieval"    # the clone directory must have this name
else:
    # this file lives at <repo>/src/retrieval_framework/forward/config.py,
    # so parents[3] is the repo root -- pinned to that tree layout.
    REPO_DIR = Path(__file__).resolve().parents[3]
    PROJECT_ROOT = REPO_DIR.parent
if not (REPO_DIR / "data" / "cm24_wasp39b").is_dir():    # tracked marker, in every clone
    raise RuntimeError(
        f"vulcan-retrieval data tree not found at {REPO_DIR}/data. Set VULCAN_PROJECT_ROOT "
        "to the directory that contains the vulcan-retrieval/ checkout (a site-packages "
        "install cannot infer it). Large caches (data/opacity_cache, data/exomolop) "
        "are seeded separately -- see the repo README data policy.")

DATA_DIR = REPO_DIR / "data"      # INPUTS: observed spectra + opacity caches

# Hand the shared engine this repo's data tree (exomolop/ + opacity_cache/,
# what data/ already holds); this takes precedence over $VULCAN_FORWARD_DATA,
# and with neither set the engine raises. It then owns every path INSIDE that tree:
# ask paths.opacity_cache_dir / cia_h2h2_file / cia_h2he_file rather than
# rebuilding them here, or the two copies drift and this one wins silently.
_fwd_paths.set_data_root(DATA_DIR)

# WASP-39b radius (cm), the Rp of vulcan3/configs/W39b.yaml. The run specs carry the
# RT geometry (rp_cm, rstar_cm, tp_gravity_cgs) as numbers with their derivation.
R_JUP_CM = 7.1492e9        # upstream VULCAN phy_const.r_jup
RP_CM = 1.279 * R_JUP_CM   # planet radius (cm) at p_ref_bar
