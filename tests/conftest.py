"""Editable-install guard: the suite must exercise THIS checkout's code.

Because of the src layout, pytest imports the *installed* retrieval_framework. A
non-editable install (pip install . or a wheel) shadows the checkout, so the suite
would silently test stale code. Fail collection loudly instead (same convention as
VULCAN-JAX's conftest). Fix: pip install --no-deps -e . (from this repo's root)

Also the real smoke pipeline builder (its capped session fixture is built
once per session, per xdist worker), the shared ConvDiag stub and the
target-digest stand-in.
"""
import dataclasses
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

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
CAPPED_COLD_CMAX = 50   # < count_min: no solve can certify, its column stays finite


class StubConvDiag(NamedTuple):
    """Pytree stand-in for forward.vulcan_chem.ConvDiag: the init reads only
    .accept_count and .conv_normal (importing the real one would pull the heavy
    VULCAN env into unit tests)."""
    accept_count: Any
    conv_normal: Any


def build_smoke_pipe(**overrides):
    """The REAL smoke pipeline (chemistry + RT, offline) with Config fields
    ``overrides`` replaced and observations set to zeros/ones. Skips only when
    the stack or its data is absent."""
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    import jax
    import numpy as np

    from retrieval_framework import pipeline as P
    from retrieval_framework import run_setup as R
    jax.config.update("jax_enable_x64", True)

    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "smoke")
    try:
        cfg, preset = R.make_config(RUN_DIR)
        if preset != "smoke":
            pytest.skip(f"preset resolved to {preset!r}, not smoke")
        pipe = P.build_pipeline(dataclasses.replace(cfg, **overrides))
    # Skip ONLY on a missing-data or missing-dependency environment; a broader
    # handler reports a real forward-model break as a green skip.
    except (FileNotFoundError, OSError, ImportError) as e:
        pytest.skip(f"cannot build real smoke pipeline ({type(e).__name__}: {e})")
    pipe.set_observations(np.zeros(pipe.n_bin), np.ones(pipe.n_bin))
    return pipe


@pytest.fixture(scope="session")
def capped_smoke_pipe():
    """The smoke pipeline at count_max=50: no solve can certify at this cap."""
    return build_smoke_pipe(count_max=CAPPED_COLD_CMAX)


@pytest.fixture
def digest_pipe(monkeypatch):
    """What provenance.target_manifest reads from a pipeline, with the code
    state and the version record frozen (the version record imports exojax,
    which must not precede vulcan_chem in a worker)."""
    import numpy as np

    from retrieval_framework import provenance
    monkeypatch.setattr(provenance, "_repo_states",
                        lambda *a, **k: {"r": {"commit": "c" * 40, "dirty": False}})
    monkeypatch.setattr(provenance, "_versions", lambda: {"jax": "0.6.2"})
    return SimpleNamespace(
        obs={"wl": np.array([1.0, 2.0]), "wl_lo": np.array([0.9, 1.9]),
             "wl_hi": np.array([1.1, 2.1]), "group": ["A", "A"]},
        obs_depth=np.array([1e-2, 1.1e-2]), obs_sigma=np.array([1e-4, 1e-4]),
        names=["lnZ", "c_o"], prior_types=["uniform", "uniform"],
        param_prior_lo=np.array([-1.0, -1.0]), param_prior_hi=np.array([1.0, 1.0]),
        groups=["A"], n_bin=2)
