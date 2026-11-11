"""The editable-install guard, the smoke pipeline fixtures and the target-digest stand-in.

With the src layout pytest imports the installed retrieval_framework, so a non-editable install would test stale code:
collection fails instead. Fix: pip install --no-deps -e . from this repo's root. Session fixtures are built once per
xdist worker.
"""

import dataclasses
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

# Tests run on CPU unless the caller sets JAX_PLATFORMS: on a GPU host the
# CUDA plugin would otherwise open every visible card in each worker.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# The engine sets this at its import, but a test module that imports scipy first has already started scipy's BLAS with
# one thread per core. Those threads busy-wait between calls, and two test processes on shared cores then starve each
# other (notes "Current state", Tests).
os.environ.setdefault("OMP_NUM_THREADS", "1")
# One network for the session: test modules import vulcan3 at collection, before any case is loaded, and vulcan3
# freezes the network and its atoms at that first import.
os.environ["VULCAN3_NETWORK"] = "thermo/SNCHO_photo_network_2025.txt"
os.environ["VULCAN3_ATOM_LIST"] = "H,O,C,N,S"

import pytest

import retrieval_framework

_SRC = Path(__file__).resolve().parent.parent / "src" / "retrieval_framework"
_IMPORTED = Path(retrieval_framework.__file__).resolve().parent

if _SRC.is_dir() and _IMPORTED != _SRC:
    raise RuntimeError(
        f"retrieval_framework imports from {_IMPORTED}, not this checkout's {_SRC}. "
        "A non-editable install is shadowing the checkout -- the suite would test stale "
        "code. Fix: pip install --no-deps -e . (from this repo's root)."
    )

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "w39b_smc_retrieval"

# A complete case for the tests that build a Config directly: every key a case states, at WASP-39b-like values (the
# schema held these as defaults before they moved into the specifications), every mode's priors and truths included.
CASE = {
    "seed": 20260704,
    "vulcan_cfg_name": "W39b",
    "network": "thermo/SNCHO_photo_network_2025.txt",
    "atom_list": ("H", "O", "C", "N", "S"),
    "rp_cm": 9.1438268e9,
    "rstar_cm": 6.483924e10,
    "tp_gravity_cgs": 422.0,
    "p_ref_bar": 1.0e-3,
    "art_pbtm_bar": 7.0,
    "molecules": ("H2O", "CO2", "CO", "CH4", "NH3", "HCN", "H2S", "SO2"),
    "nu_min": 1923.0,
    "nu_max": 3450.0,
    "cloud_model": "powerlaw",
    "count_max": 5000,
    "cold_lanes": 8,
    "baseline_metallicity_solar": 10.0,
    "combo": ("NIRISS", "G395H"),
    "obs_wl_lo": 1.0,
    "obs_wl_hi": 5.28,
    "tp_Tint_K": 150.0,
    "tp_gamma_fixed": 0.4,
    "tp_deep_p_bar": 1.0,
    "tp_nabla_ad": 0.29,
    "kzz_p_ref_bar": 1.0e-3,
    "prior_lnZ": (-2.302585092994046, 2.302585092994046),
    "prior_c_o": (-1.6, 0.45),
    "prior_lnKzz": (-5.0, 5.0),
    "prior_s_o": (-2.302585092994046, 2.302585092994046),
    "prior_kzz_slope": (-1.0, 1.0),
    "prior_Tirr": (800.0, 2600.0),
    "prior_log10kappa": (-3.5, 0.5),
    "prior_log10gamma": (-2.0, 0.7),
    "prior_Tint": (100.0, 500.0),
    "prior_T1bar": (320.0, 2980.0),
    "prior_log10P_rcb": (-1.0, 0.88),
    "prior_lnR0": (-0.08, 0.08),
    "prior_lnM_sigma": 0.11,
    "prior_log10kappa_cloud": (-7.0, 1.0),
    "prior_cloud_alpha": (0.0, 6.0),
    "prior_log10P_deck": (-6.0, 0.8),
    "prior_offset_ppm": (-800.0, 800.0),
    "prior_noise_inflation": (0.5, 3.0),
    **{
        f"truth_{k}": v
        for k, v in {
            "lnZ": 0.0,
            "c_o": 0.0,
            "lnKzz": 0.0,
            "s_o": 0.0,
            "kzz_slope": 0.0,
            "Tirr": 1650.0,
            "log10kappa": -2.0,
            "log10gamma": -0.4,
            "Tint": 150.0,
            "T1bar": 1500.0,
            "log10P_rcb": 0.5,
            "lnR0": 0.0,
            "lnM": 0.0,
            "log10kappa_cloud": -6.5,
            "cloud_alpha": 1.0,
            "log10P_deck": 0.5,
            "offset_ppm": 0.0,
            "noise_inflation": 1.0,
        }.items()
    },
}


def case_config(**over):
    """Return the Config of ``CASE`` with ``over`` applied."""
    from retrieval_framework import config_schema as C

    return C.Config(**{**CASE, **over})


CAPPED_COLD_CMAX = 50  # < count_min: no solve can certify, its column stays finite


def build_smoke_pipe(**overrides):
    """Return the real smoke pipeline (chemistry and RT, offline) with ``overrides`` and zero-depth, unit-sigma data.

    Skips only when exojax or a table the case reads is absent; any other failure of the build fails the test.
    """
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    if importlib.util.find_spec("exojax") is None:
        pytest.skip("exojax not installed")
    import jax
    import numpy as np

    from retrieval_framework import pipeline as P
    from retrieval_framework import run_setup as R
    from retrieval_framework.forward import config as _config  # noqa: F401  # hands the engine its data root
    from vulcan_forward import exomolop, paths

    jax.config.update("jax_enable_x64", True)

    cfg = dataclasses.replace(R.load_spec(RUN_DIR / "smoke.yaml").cfg, **overrides)
    needed = [exomolop.table_path(m) for m in cfg.molecules] + [paths.cia_h2h2_file(), paths.cia_h2he_file()]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        pytest.skip(f"data absent: {missing}")
    pipe = P.build_pipeline(cfg)
    pipe.set_observations(np.zeros(pipe.n_bin), np.ones(pipe.n_bin))
    return pipe


@pytest.fixture(scope="session")
def capped_smoke_pipe():
    """The smoke pipeline at count_max=50: no solve can certify at this cap."""
    return build_smoke_pipe(count_max=CAPPED_COLD_CMAX)


# Every v5 model option at once (tests/test_v5_options.py): sulfur free, the
# Kzz power law, T at 1 bar as the deep parameter with the adiabat below a free
# boundary, the Gaussian mass nuisance, the grey deck.
V5_SMOKE = {
    "infer_s_o": True,
    "prior_s_o": (-2.302585092994046, 2.302585092994046),  # a factor of ten either way
    "truth_s_o": 0.0,
    "kzz_mode": "powerlaw",
    "kzz_p_ref_bar": 1.0e-3,
    "prior_kzz_slope": (-1.0, 1.0),
    "truth_kzz_slope": 0.0,
    "tp_infer_Tint": True,
    "tp_deep_param": "T1bar",
    "prior_T1bar": (600.0, 2500.0),
    "truth_T1bar": 1500.0,
    "tp_adiabat": True,
    "tp_nabla_ad": 0.29,
    "prior_log10P_rcb": (-1.0, 0.88),
    "truth_log10P_rcb": 0.5,
    "infer_lnM": True,
    "truth_lnM": 0.0,
    "cloud_model": "deck",
    "prior_log10P_deck": (-6.0, 0.8),
    "truth_log10P_deck": 0.5,
}


@pytest.fixture(scope="session")
def smoke_pipe():
    """The smoke pipeline as the case defines it."""
    return build_smoke_pipe()


@pytest.fixture(scope="session")
def v5_smoke_pipe():
    """The smoke pipeline with every v5 option on (V5_SMOKE)."""
    return build_smoke_pipe(**V5_SMOKE)


@pytest.fixture
def digest_pipe(monkeypatch):
    """What provenance.target_manifest reads from a pipeline, with the code
    state and the version record frozen (the version record imports exojax,
    which must not precede vulcan_chem in a worker).
    """
    import numpy as np

    from retrieval_framework import provenance

    monkeypatch.setattr(provenance, "_repo_states", lambda *_a, **_k: {"r": {"commit": "c" * 40, "dirty": False}})
    monkeypatch.setattr(provenance, "_versions", lambda: {"jax": "0.6.2"})
    monkeypatch.setattr(provenance, "science_data_identity", lambda _m: {"opacity_sha256": {}, "cia_sha256": {}})
    return SimpleNamespace(
        obs={
            "wl": np.array([1.0, 2.0]),
            "wl_lo": np.array([0.9, 1.9]),
            "wl_hi": np.array([1.1, 2.1]),
            "group": ["A", "A"],
        },
        obs_depth=np.array([1e-2, 1.1e-2]),
        obs_sigma=np.array([1e-4, 1e-4]),
        names=["lnZ", "c_o"],
        prior_types=["uniform", "uniform"],
        param_prior_lo=np.array([-1.0, -1.0]),
        param_prior_hi=np.array([1.0, 1.0]),
        groups=["A"],
        n_bin=2,
    )
