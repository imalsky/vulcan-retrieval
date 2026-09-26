"""Editable-install guard: the suite must exercise THIS checkout's code.

Because of the src layout, pytest imports the *installed* retrieval_framework. A
non-editable install (pip install . or a wheel) shadows the checkout, so the suite
would silently test stale code. Fail collection loudly instead (same convention as
VULCAN-JAX's conftest). Fix: pip install --no-deps -e . (from this repo's root)

Also the shared pipeline builders: a chemistry-free stub for the SMC-core tests
and the real smoke pipeline (the session fixture the two rejection-gate files
share is built once per session, per xdist worker).
"""
import dataclasses
import os
from pathlib import Path

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
CAPPED_WARM_CMAX = 5    # well below the cold cap, so the warm cap is what binds


def stub_pipeline(cfg, specs, loglik_theta):
    """A chemistry-free Pipeline on the box prior ``specs`` with log-likelihood
    ``loglik_theta(theta)``: the SMC core runs through its real code path
    without the forward stack."""
    import jax.numpy as jnp
    import numpy as np

    from retrieval_framework import pipeline as P
    theta_from_u, log_prior_u, sample_prior_u = P.make_uspace(specs, jnp.float64)
    return P.Pipeline(
        cfg=cfg, dtype=jnp.float64, npdtype=np.float64, n_dim=len(specs),
        theta_from_u=theta_from_u, log_prior_u=log_prior_u,
        sample_prior_u=sample_prior_u,
        log_likelihood_u=lambda u: loglik_theta(theta_from_u(u)))


def build_smoke_pipe(**overrides):
    """The REAL smoke pipeline (chemistry + RT, offline) with Config fields
    ``overrides`` replaced and observations set to zeros/ones. Skips only when
    the stack or its data is absent."""
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    import jax
    import numpy as np

    from retrieval_framework import pipeline as P
    from retrieval_framework import run_smc as R
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
    """The smoke pipeline at count_max=50 and warm_count_max=5, in WARM chem
    mode. No solve can certify at these caps. test_cold_reject reads its cold
    evaluators (chem-mode independent), test_warm_reject its warm ones."""
    return build_smoke_pipe(count_max=CAPPED_COLD_CMAX,
                            warm_count_max=CAPPED_WARM_CMAX,
                            smc_chem_mode="warm")
