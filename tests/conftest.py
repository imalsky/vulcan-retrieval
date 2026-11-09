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
    "kzz_mode": "powerlaw",
    "tp_infer_Tint": True,
    "tp_deep_param": "T1bar",
    "prior_T1bar": (600.0, 2500.0),
    "tp_adiabat": True,
    "infer_lnM": True,
    "cloud_model": "deck",
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
