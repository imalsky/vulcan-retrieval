"""Retrieval-side configuration for the shared vulcan-forward engine.

Pure constants + paths: NO heavy imports here (no jax, no vulcan_jax, no exojax),
so this module is safe to import before the env-order-sensitive VULCAN-JAX setup
runs.

The forward-model ENGINE lives in the ``vulcan-forward`` distribution, so the
physics constants and the molecule/opacity table below are re-exported from
``vulcan_forward.constants`` rather than defined twice.
What stays genuinely local: this repo's filesystem layout, the WASP-39 b case
constants, the run profiles, and the parameter-vector labels. This module also
hands the engine its data root (see the paths section), so the opacity caches and
line lists live in this repo's data/ tree.

The demo chains the *live* VULCAN-JAX chemistry forward model into an ExoJax
``ArtTransPure`` transmission model and propagates forward-mode tangents from four
physical parameters -- (ln Z, C/O, ln Kzz, dT) -- all the way to the transit
spectrum, so every wavelength can be colored by d(transit_depth)/d(parameter).

Planet: WASP-39b (matches the validated jax_paper sensitivity scripts + the JWST
SO2/CO2 metallicity story).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared physics constants -- single source of truth is the engine package
# ---------------------------------------------------------------------------
# Re-exported (not redefined) so this repo and vulcan-jwst-tool cannot drift
# apart on molecule masses, opacity sources, the ART grid bounds, or the
# composition tables. vulcan_forward.constants is stdlib-only, so importing it
# here keeps this module import-light.
from vulcan_forward import constants as _fwd
from vulcan_forward import paths as _fwd_paths

MOLECULES = _fwd.MOLECULES
ATOM_COLS = _fwd.ATOM_COLS
ATOMIC_MASSES = _fwd.ATOMIC_MASSES
BULK_H2_VULCAN = _fwd.BULK_H2_VULCAN
CLOUD_NUC0 = _fwd.CLOUD_NUC0
# Value copies for reading. Rebinding one here does NOT reach the engine:
# build_rt_model resolves the ART bounds from the profile
# (profile["art_ptop_bar"] / ["art_pbtm_bar"]), falling back to
# vulcan_forward.constants. Pass an override through the profile.
ART_PTOP_BAR = _fwd.ART_PTOP_BAR
ART_PBTM_BAR = _fwd.ART_PBTM_BAR
T_OPA_MIN_K = _fwd.T_OPA_MIN_K
T_OPA_MAX_K = _fwd.T_OPA_MAX_K

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# VULCAN_PROJECT_ROOT = the directory CONTAINING the vulcan-retrieval/ checkout
# (and, for HPC runs + manuscript figures, its siblings VULCAN-JAX/ and jax_paper/).
# The explicit env var wins; otherwise the repo root is inferred from this file's
# location inside an editable checkout. A bare site-packages install cannot infer
# it and must set the env var -- checked loudly below, no silent fallbacks.
_env_root = os.environ.get("VULCAN_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).expanduser()
    REPO_DIR = PROJECT_ROOT / "vulcan-retrieval"    # NAS clone dir name is load-bearing
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

JP = PROJECT_ROOT / "jax_paper"  # for _common.apply_style (house figure style)
DATA_DIR = REPO_DIR / "data"                                    # INPUTS: observed spectra + opacity caches
OUTPUTS = REPO_DIR / "output"                                   # GENERATED: npz caches from examples/validation/zco
FIGS = JP / "figures"                                           # manuscript figures stay in jax_paper/figures

# Hand the shared engine this repo's data tree (exomolop/ + opacity_cache/,
# exactly what data/ already holds), so the engine never infers
# the location from its own __file__. It then owns every path INSIDE that tree:
# ask paths.opacity_cache_dir / cia_h2h2_file / cia_h2he_file rather than
# rebuilding them here, or the two copies drift and this one wins silently.
_fwd_paths.set_data_root(DATA_DIR)

# ---------------------------------------------------------------------------
# WASP-39b physical constants (from vulcan_jax/configs/W39b.yaml)
# ---------------------------------------------------------------------------
R_SUN_CM = 6.957e10
# Planet radius ASSIGNED to the bottom pressure of the ART grid (7 bar). The
# literature transit radius does not itself specify a 7-bar reference level, so
# this anchoring is a convention; the retrieval's free lnR0 absorbs the offset,
# which is why lnR0 must be interpreted as a pressure-radius normalization
# nuisance rather than a physical radius (see transmission_depth_r).
RP_CM = 1.279 * 7.1492e9   # planet radius (cm) at the bottom pressure P_b
GS_CGS = 422.0             # surface gravity (cm/s^2), held fixed (incl. under lnR0)
RSTAR_CM = 0.932 * R_SUN_CM

# ---------------------------------------------------------------------------
# Run profiles
# ---------------------------------------------------------------------------
# Wavenumbers in cm^-1. wavelength(um) = 1e4 / nu.
#
# Two non-obvious requirements, both about keeping the forward-mode tangent valid:
#   * Photochemistry must be ON. Only in the photo-on regime does the warm-started jvp
#     relax to the true steady-state sensitivity (validated: jvp vs re-converged FD <0.1%
#     at nz=150). With photo OFF the W39b column lands in a regime where the tangent is
#     under-relaxed/unstable.
#   * Let convergence happen naturally (default count_min/count_max). Do NOT pin a fixed
#     step count -- forcing dt to dt_max drives the Ros2 step's forward tangent singular.
SMOKE = {
    "use_photo": True,
    "nz": 40,                  # coarse column -> cheaper warm-up + jvps
    "yconv_cri": 1.0e-3,
    "molecules": ["CO"],       # fully offline
    "nu_min": 4280.0,          # ~2.31-2.34 um, the cached CO 2-0 band (matches smc.py)
    "nu_max": 4360.0,
    "opacity_mode": "exomolop",
    "art_nlayer": 20,
    # planet identity, explicit: the engine requires it rather than
    # defaulting to WASP-39 b (a forgotten key would silently model a
    # different planet).
    "rp_cm": RP_CM, "gs_cgs": GS_CGS, "rstar_cm": RSTAR_CM,
}
FULL = {
    "use_photo": True,         # photo ON -> SO2 chemistry (WASP-39b story)
    "nz": 188,                 # canonical W39b grid: 19 layers/decade over 1e-9..7.6 bar
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": 1923.0,          # ~5.2 um
    "nu_max": 3450.0,          # ~2.9 um  (NIRSpec G395H/PRISM red: CH4 3.3, SO2 4.0, CO2 4.3, CO 4.7)
    "opacity_mode": "exomolop",
    "art_nlayer": 60,
    # planet identity, explicit: the engine requires it rather than
    # defaulting to WASP-39 b (a forgotten key would silently model a
    # different planet).
    "rp_cm": RP_CM, "gs_cgs": GS_CGS, "rstar_cm": RSTAR_CM,
}
# Wide-band overview: 1-15 um (the supported window -- H2-H2 CIA stops at 1 um / 10000
# cm-1 on the short side; the ExoMolOP tables reach 50 um). Computed on the tables'
# R=1000 band grid and displayed at R=100. Used for BOTH the transmission and emission
# figures.
WIDE = {
    "use_photo": True,
    "nz": 188,
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": 667.0,           # 15 um
    "nu_max": 10000.0,         # 1 um  (H2-H2 CIA upper edge)
    "opacity_mode": "exomolop",
    "art_nlayer": 60,
    "display_R": 100,
    # planet identity, explicit: the engine requires it rather than
    # defaulting to WASP-39 b (a forgotten key would silently model a
    # different planet).
    "rp_cm": RP_CM, "gs_cgs": GS_CGS, "rstar_cm": RSTAR_CM,
}

# Parameter vector order: theta = [lnZ, c_o_pert, lnKzz, dT_K]. theta[3] is a
# UNIFORM additive temperature offset applied to every layer -- historically
# (mis)labeled "T_int"; it is not an interior/intrinsic temperature.
THETA_LABELS = ["lnZ", "C/O", "lnKzz", "dT"]
THETA0 = [0.0, 0.0, 0.0, 0.0]   # baseline (no perturbation)
