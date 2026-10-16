"""Retrieval-side configuration for the shared vulcan-forward engine.

Pure constants + paths: NO heavy imports here (no jax, no vulcan3, no exojax),
so this module is safe to import before the env-order-sensitive VULCAN-JAX setup
runs.

The shared physics constants and the molecule/opacity table live in
``vulcan_forward.constants``; import them from there. What is local: this
repo's filesystem layout, the WASP-39 b case constants and the SMOKE profile.
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

# WASP-39b geometry (cm): the Rp and r_star of vulcan3/configs/W39b.yaml.
R_SUN_CM = 6.957e10        # = vulcan3.phy_const.r_sun (not imported: import-light)
R_JUP_CM = 7.1492e9        # upstream VULCAN phy_const.r_jup
# The literature radius and the gravity there, G*Mp/Rp^2 with the cfg's Mp.
# They apply at the config's p_ref_bar; the retrieval's free lnR0 scales the
# radius there at fixed planet mass (see transmission_depth_r).
RP_CM = 1.279 * R_JUP_CM   # planet radius (cm) at p_ref_bar
GS_CGS = 422.0             # gravity (cm/s^2) at RP_CM
RSTAR_CM = 0.932 * R_SUN_CM

# Run profile (the condensation test's SMOKE column)
# Wavenumbers in cm^-1. wavelength(um) = 1e4 / nu.
#
# Two requirements for a valid forward-mode tangent:
#   * Photochemistry ON (jvp vs re-converged FD < 0.1% at nz=150): photo-off, the
#     forward model does not settle on a comparable state.
#   * Default count_min/count_max: a pinned step count (dt forced to dt_max) drives
#     the Ros2 step's forward tangent singular.
SMOKE = {
    "use_photo": True,
    "nz": 40,                 # coarse column -> cheaper warm-up + jvps
    "yconv_cri": 1.0e-3,
    "molecules": ["CO"],       # fully offline
    "nu_min": 4280.0,          # ~2.29-2.34 um, the cached CO 2-0 band
    "nu_max": 4360.0,
    "opacity_mode": "exomolop",
    "art_nlayer": 20,
    # planet identity, explicit: the engine requires it rather than
    # defaulting to WASP-39 b (a forgotten key would silently model a
    # different planet).
    "rp_cm": RP_CM, "gs_cgs": GS_CGS, "rstar_cm": RSTAR_CM,
}
