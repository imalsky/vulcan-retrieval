"""Central configuration for the VULCAN-JAX -> ExoJax transmission-sensitivity demo.

Pure constants only: NO heavy imports here (no jax, no vulcan_jax, no exojax), so
this module is safe to import before the env-order-sensitive VULCAN-JAX setup runs.

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
# Paths
# ---------------------------------------------------------------------------
# VULCAN_PROJECT_ROOT = the directory CONTAINING the vulcan_exojax_run/ checkout
# (and, for HPC runs + manuscript figures, its siblings VULCAN-JAX/ and jax_paper/).
# The explicit env var wins; otherwise the repo root is inferred from this file's
# location inside an editable checkout. A bare site-packages install cannot infer
# it and must set the env var -- checked loudly below, no silent fallbacks.
_env_root = os.environ.get("VULCAN_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).expanduser()
    REPO_DIR = PROJECT_ROOT / "vulcan_exojax_run"    # NAS clone dir name is load-bearing
else:
    # this file lives at <repo>/vulcan-retrieval/src/retrieval_framework/forward/config.py,
    # so parents[4] is the repo root -- pinned to that tree layout.
    REPO_DIR = Path(__file__).resolve().parents[4]
    PROJECT_ROOT = REPO_DIR.parent
if not (REPO_DIR / "data" / "cm24_wasp39b").is_dir():    # tracked marker, in every clone
    raise RuntimeError(
        f"vulcan_exojax_run data tree not found at {REPO_DIR}/data. Set VULCAN_PROJECT_ROOT "
        "to the directory that contains the vulcan_exojax_run/ checkout (a site-packages "
        "install cannot infer it). Large caches (data/opacity_cache, data/exojax_linelists) "
        "are seeded separately -- see the repo README data policy.")

JP = PROJECT_ROOT / "jax_paper"  # for _common.apply_style (house figure style)
DEMO_DIR = REPO_DIR                                             # deprecated pre-0.6 alias
OUTPUTS = REPO_DIR / "data"                                     # npz caches + observed spectra live with the repo
FIGS = JP / "figures"                                           # manuscript figures stay in jax_paper/figures
DEMO_DATABASE = OUTPUTS / "exojax_linelists"                    # HITRAN line lists

# Offline opacity cache (CO ExoMol Li2015 + H2-H2/H2-He CIA), lives IN the repo
# (data/opacity_cache/) so it has no dependency on any sibling project --
# copied in 2026-07-07 from what was previously a reused emulator-demo/ cache.
_CACHE = OUTPUTS / "opacity_cache"
CO_CACHED_DIR = _CACHE / "CO" / "12C-16O" / "Li2015"
CIA_H2H2_FILE = _CACHE / "H2-H2_2011.cia"
# H2-He CIA (He is ~16% by number at 10x solar; real continuum contribution).
# Download once: https://hitran.org/data/CIA/H2-He_2011.cia -> this path.
# exojax_rt REFUSES to build without it, and the He VMR profile is a required
# argument of every depth/flux function (silently skipping the He continuum
# used to be possible and biased the sensitivity-demo spectra).
CIA_H2HE_FILE = _CACHE / "H2-He_2011.cia"

# Reference wavenumber (cm^-1) for the ExoJax powerlaw_clouds retrieval cloud:
# kappa(nu) = kappac0 * (nu/CLOUD_NUC0)^alphac, kappac0 in cm^2 per gram of
# atmosphere (pRT convention; alphac = 0 is a gray cloud). 2857 cm^-1 = 3.5 um,
# mid-band for the 2.0-5.3 um retrieval window. NOTE this is a uniformly mixed
# power-law opacity ("haze slope + gray deck" nuisance), NOT a physical cloud:
# no cloud-top pressure, condensation, sedimentation, or patchiness.
CLOUD_NUC0 = 2857.0

# Pressure-broadening perturber for the HITRAN opacities. "air" = HITRAN's
# terrestrial gamma_air/n_air (a documented approximation for an H2/He envelope);
# "h2he" = HITRAN's planetary-broadener H2/He widths where the database provides
# them, blended with the number-fraction mix below (uncovered lines keep air widths
# for that partner, reported loudly per molecule at build; see
# exojax_rt._blend_h2he_broadening and validation/broadening_ab.py for the A/B).
# Profiles may override per run via profile["broadening"].
BROADENING = "air"
H2HE_BROADENING_MIX = (0.85, 0.15)   # (f_H2, f_He) by number, ~solar envelope

# ---------------------------------------------------------------------------
# VULCAN-JAX network selection (must be set as env vars BEFORE importing vulcan_jax)
# ---------------------------------------------------------------------------
VULCAN_NETWORK = "thermo/SNCHO_photo_network.txt"
VULCAN_ATOM_LIST = "H,O,C,N,S"
W39B_CFG_MODULE = "vulcan_jax.cfg_examples.vulcan_cfg_W39b"

# atom_list column order inside composition.compo_array (probed from the package).
# Index of each element column we touch when building the Z / C/O knobs.
ATOM_COLS = {"H": 0, "O": 1, "C": 2, "He": 3, "N": 4, "S": 5}

# Molar masses (g/mol) for every compo_array column, in column order. Used to turn
# a VMR profile into a mean-molecular-weight profile (compo_array @ this vector).
# atom_list = (H,O,C,He,N,S,P,Na,K,Si,Fe,Ar,Ti,V,Mg,Ca,e)
ATOMIC_MASSES = [
    1.008, 15.999, 12.011, 4.0026, 14.007, 32.06, 30.974, 22.990,
    39.098, 28.085, 55.845, 39.948, 47.867, 50.942, 24.305, 40.078, 5.4858e-4,
]

# ---------------------------------------------------------------------------
# WASP-39b physical constants (from cfg_examples/vulcan_cfg_W39b.py)
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
# Opacity / radiative-transfer grid
# ---------------------------------------------------------------------------
# ART pressure bounds (bar). The bottom stays inside VULCAN's envelope; the TOP is
# set ABOVE VULCAN's 1e-7 bar chemistry top on purpose -- the log-P interpolation
# CLAMPS the topmost VULCAN VMR/T over the extra decade, i.e. a constant-abundance +
# isothermal upper-atmosphere extension (a common transmission-modeling convention,
# NOT chemistry: photochemical species can genuinely vary at sub-microbar pressures).
# Without it, strong bands (CO2 4.3, CO 4.7 um) go optically thick to the model top
# and the transit radius saturates into a flat "wall" at 4.2-5.2 um (saturated
# fraction 4.8% at 1e-6 bar); extending to 1e-8 bar removes it (0.1%), letting the
# bands rise to real peaks. This is an EXPLICIT modeling choice; quantify its effect
# with validation/top_pressure_ladder.py (clamped 1e-7/1e-8/1e-9 tops, and chemistry
# actually extended via cfg P_t) before quoting fine-grained band-peak numbers.
ART_PTOP_BAR = 1.0e-8
ART_PBTM_BAR = 7.0
T_OPA_MIN_K = 300.0
T_OPA_MAX_K = 3000.0

# Each molecule: VULCAN species name, molar mass (g/mol), and opacity source.
# CO is fully offline (cached ExoMol Li2015). H2O/CO2/CH4/SO2 use HITRAN, downloaded
# on first run into .database/<db>/ (small, public, no login -- only the main
# isotopologue, isotope=1; HITRAN intensities already carry the terrestrial isotopic
# abundance factor, so pairing them with the TOTAL molecular VMR is the standard
# slightly-conservative treatment). KNOWN LIMITS for real-data inference, not just
# the methodology demo: (1) HITRAN's 296 K room-T lists under-represent hot bands at
# the ~770-1540 K retrieved limb relative to HITEMP/ExoMol -- absolute abundances
# lean optimistic/biased where hot bands matter; swap sources here (one line per
# molecule) when the multi-GB ExoMol/HITEMP fetch is acceptable. (2) Default
# broadening is terrestrial air; set BROADENING="h2he" (above) for HITRAN's
# planetary H2/He widths where available and run validation/broadening_ab.py for
# the measured difference.
# "source" is one of {"exomol_cached", "exomol", "hitran"}; "db" is the ExoMol
# "<iso>/<list>" path suffix or the per-molecule download dir name under .database.
# molmass is set explicitly (exojax isotope_molmass returns None for CH4).
MOLECULES = {
    "CO":  {"vulcan": "CO",  "molmass": 28.010, "source": "exomol_cached", "db": str(CO_CACHED_DIR)},
    "H2O": {"vulcan": "H2O", "molmass": 18.015, "source": "hitran", "db": "H2O"},
    "CO2": {"vulcan": "CO2", "molmass": 43.990, "source": "hitran", "db": "CO2"},
    "CH4": {"vulcan": "CH4", "molmass": 16.043, "source": "hitran", "db": "CH4"},
    "SO2": {"vulcan": "SO2", "molmass": 64.066, "source": "hitran", "db": "SO2"},
    # High-C/O + sulfur discriminators for the SMC retrieval prior box
    # (runs/w39b_smc_retrieval): the retrieval explores C/O up to ~1 where C2H2/HCN
    # carry the signal; H2S is the reduced-S reservoir. Same HITRAN path as above.
    "HCN":  {"vulcan": "HCN",  "molmass": 27.025, "source": "hitran", "db": "HCN"},
    "C2H2": {"vulcan": "C2H2", "molmass": 26.037, "source": "hitran", "db": "C2H2"},
    "H2S":  {"vulcan": "H2S",  "molmass": 34.081, "source": "hitran", "db": "H2S"},
    # Cool-planet nitrogen carrier (e.g. WASP-107b-class): jwst_tool opt-in molecule.
    "NH3":  {"vulcan": "NH3",  "molmass": 17.031, "source": "hitran", "db": "NH3"},
}

# Bulk gas used for CIA + the dominant background (H2).
BULK_H2_VULCAN = "H2"

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
    "nu_pts": 600,
    "art_nlayer": 20,
}
FULL = {
    "use_photo": True,         # photo ON -> SO2 chemistry (WASP-39b story)
    "nz": 150,                 # canonical W39b grid
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": 1923.0,          # ~5.2 um
    "nu_max": 3450.0,          # ~2.9 um  (NIRSpec G395H/PRISM red: CH4 3.3, SO2 4.0, CO2 4.3, CO 4.7)
    "nu_pts": 6000,
    "art_nlayer": 60,
}
# Wide-band overview: 1-15 um (the supported window -- H2-H2 CIA stops at 1 um / 10000
# cm-1 on the short side, line lists reach ~20 um). Computed on a finer native grid and
# displayed at R=100. Used for BOTH the transmission and emission figures.
WIDE = {
    "use_photo": True,
    "nz": 150,
    "yconv_cri": 1.0e-3,
    "molecules": ["H2O", "CO2", "CO", "CH4", "SO2"],
    "nu_min": 667.0,           # 15 um
    "nu_max": 10000.0,         # 1 um  (H2-H2 CIA upper edge)
    "nu_pts": 8000,            # native R ~ 2950; binned to display_R for the figure
    "art_nlayer": 60,
    "display_R": 100,
}

# Parameter vector order: theta = [lnZ, c_o_pert, lnKzz, dT_K]. theta[3] is a
# UNIFORM additive temperature offset applied to every layer -- historically
# (mis)labeled "T_int"; it is not an interior/intrinsic temperature.
THETA_LABELS = ["lnZ", "C/O", "lnKzz", "dT"]
THETA0 = [0.0, 0.0, 0.0, 0.0]   # baseline (no perturbation)
