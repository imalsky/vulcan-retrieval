"""Fixed numbers of the retrieval: each has a unit and a reason.

A number that belongs to a case is a ``Config`` field stated in the run specification (``config_schema``). A number
here is fixed on purpose and the specification cannot change it, except ``BATCH_PER_LANE``, the batch width when the
specification's ``sampler.n_batch`` is null. Each run's target manifest stores ``fixed_values()`` of this module and of
the engine (``provenance.target_manifest``). The module imports nothing, so every other module can read it before the
engine is imported. The methods behind the numbers are in docs/method.md.
"""

from __future__ import annotations

__all__ = [
    "BATCH_PER_LANE",
    "COND_BINS",
    "COND_PARAMS",
    "DISCARD_EXPLORATION",
    "ENLARGE_PER_DIM",
    "F_LIVE",
    "GUILLOT_F",
    "KZZ_FLOOR_MULTIPLES",
    "LODDERS2020_LOG_EPS",
    "LSF_R_SAFETY",
    "MIN_REAL_BINS",
    "NOT_A_TABLE_ABSORBER",
    "NO_OPACITY_TABLE",
    "NUISANCE_NODES",
    "N_LIVE_PER_UPDATE",
    "N_NETWORKS",
    "PHOTOSPHERE_BAR",
    "PPM",
    "PRIOR_Z_EPS",
    "REJECT_BELOW",
    "REJECT_LOGL",
    "STARTUP_MIN_DRAWS",
    "SYNTH_BAND_PAD_FRAC",
    "SYNTH_R",
    "SYNTH_SIGMA_PPM",
    "TERM_DT_MIN_FROZEN",
    "TP_DRAW_CAP_FACTOR",
    "TP_DRAW_CAP_MIN",
    "TP_DRAW_MIN",
    "T_EDGE_INSET_K",
    "UNDERFLOW_DENOM",
    "UNMODELED_VMR",
    "fixed_values",
]

# ---- 1. Physical constants and unit factors ----

# One part per million of the transit depth (offsets, sigmas).
PPM = 1.0e-6
# Lodders (2020) present-day photospheric log eps (the vulcan3 .dat header cites its Table 8); the total-metals factor
# weighs C and S by their knobs.
LODDERS2020_LOG_EPS = {"C": 8.47, "N": 7.85, "O": 8.73, "S": 7.15}

# ---- 2. Fixed model choices: not settable, each with its reason ----

# Exploration ends when the live set holds this share of the evidence.
F_LIVE = 0.03
# Enlargement of each bounding ellipsoid, per dimension: 1.3 cut the posterior mean error from 0.10-0.14 sd to
# 0.02-0.08 sd against nautilus's 1.1 (notes "Sampler").
ENLARGE_PER_DIM = 1.3
# nautilus's default: four ReLU networks, hidden layers 100-50-20.
N_NETWORKS = 4
# The exploration phase's points are dropped from the posterior: keeping them biased logZ by -0.206 (notes "Do not
# retry").
DISCARD_EXPLORATION = True
# Live points per bound update: nautilus's n_update is n_live // N_LIVE_PER_UPDATE (notes "Sampler").
N_LIVE_PER_UPDATE = 2
# In-window cold draws a log10 Kzz decade needs before none certifying refuses the run: 0 of 30 bounds that decade's
# certified fraction below 10 % at 95 % (notes "Design records", launch guards).
STARTUP_MIN_DRAWS = 30
# A rejected draw's log-likelihood is this finite sentinel on device; run_nautilus maps it to -inf. Anything at or below
# REJECT_BELOW is a rejection.
REJECT_LOGL = -1.0e30
REJECT_BELOW = -1.0e29
# vulcan3.state.TERM_DT_MIN_FROZEN, held here because the pipeline imports no engine: vulcan3's first import must
# follow the network selection. tests/test_run_nautilus_main.py compares the two.
TERM_DT_MIN_FROZEN = 4
# Trapezoid nodes in log10 b for an integrated noise inflation. On the WASP-39b case's bins the error against brute
# force falls as h^2: 3.6e-3 nats at 513 nodes in the worst case tried (b piled at its floor), <= 5e-5 on realistic
# spectra (notes "Design records", the integrated nuisances).
NUISANCE_NODES = 2049
# Guillot f: 1/4 is the whole-planet average of the irradiation.
GUILLOT_F = 0.25
# K. The T-P profile is not clipped, since a clip invents an isothermal wall; the pipeline rejects a profile with a
# layer outside the engine's valid RT window inset by this (pipeline.tp_valid).
T_EDGE_INSET_K = 20.0
# A data source needs at least this many observed bins inside the model band.
MIN_REAL_BINS = 4
# The synthetic bin grid a specification with no data source runs on: resolving power, per-bin sigma (ppm), and the
# fractional inset from the model band edges.
SYNTH_R = 200
SYNTH_SIGMA_PPM = 120.0
SYNTH_BAND_PAD_FRAC = 0.001
# A cell average is a faithful binning operator only when a data bin spans at least this many R1000 model bands. Two
# admits Welbanks et al. (2024)'s native 0.015 um NIRCam bins (R up to 336, about three bands).
LSF_R_SAFETY = 2.0
# Kzz weights near the prior floor are taken at these multiples of the specification's prior_lnKzz lower edge (x0.15
# and x0.2 on a x0.1 floor).
KZZ_FLOOR_MULTIPLES = (1.5, 2.0)
# The conditioning map: equal bins over each prior box, for these parameters when the case infers them.
COND_BINS = 10
COND_PARAMS = ("lnKzz", "T1bar", "Tint", "lnZ")
# A network species with no opacity table, abundant in the observable layers, is reported with its abundance: the rule
# and thresholds of jwst-transit-authority's certify.unmodeled_absorbers. Mean VMR over the transmission photosphere
# PHOTOSPHERE_BAR (bar) above UNMODELED_VMR, an order of magnitude below the bulk carriers. NO_OPACITY_TABLE are the
# engine molecules ExoMolOP publishes no k-table for; NOT_A_TABLE_ABSORBER are absent on purpose: atoms (no IR bands),
# the CIA and Rayleigh gases H2 and He, and the dipole-free homonuclear N2, O2 and S2.
UNMODELED_VMR = 1.0e-4
PHOTOSPHERE_BAR = (1.0e-5, 1.0e-2)
NO_OPACITY_TABLE = ("CS2", "C2H6")
NOT_A_TABLE_ABSORBER = (
    *("H", "O", "C", "N", "S", "O_1", "N_2D", "H2", "He", "N2", "O2", "S2"),
    *("Li", "F", "Na", "Mg", "Si", "P", "Cl", "K", "Ti", "V", "Cr", "Fe", "Rb", "atCs"),
)

# ---- 3. Numerical tolerances ----

# Floor for a denominator or log argument that may be exactly zero.
UNDERFLOW_DENOM = 1e-300
# sample_prior_u draws z = sigmoid(u) inside [eps, 1 - eps], so u stays finite.
PRIOR_Z_EPS = 1e-6
# T-P window rejection sampling: the floor on candidates per round, and the cap (max(factor * n, min)) where it raises.
TP_DRAW_MIN = 16
TP_DRAW_CAP_FACTOR = 64
TP_DRAW_CAP_MIN = 4096

# ---- 4. Defaults a caller may override through the run specification ----

# Points per lane per batch when the specification sets no n_batch: certified lanes refill inside the batch.
BATCH_PER_LANE = 3


def fixed_values() -> dict[str, object]:
    """Return every constant of this module by name, for a run's record.

    Returns
    -------
    dict
        Name to value, tuples as lists, in ``__all__`` order.
    """
    out: dict[str, object] = {}
    for name in __all__:
        value = globals()[name]
        if callable(value):
            continue
        out[name] = list(value) if isinstance(value, tuple) else value
    return out
