"""Every likelihood entry point rejects a FINITE spectrum from an uncertified cold
solve: the scalar log_likelihood_u, both gradient routes and the batched cold
evaluator apply one certificate (retrieval_forward.native_depth_aux's ok bit ==
pipeline._proposal_converged plus the count_max cap). Real smoke pipeline at
count_max=50 < count_min, so no cold solve can certify while its column, and
hence its spectrum, stays finite. Slow: real chemistry + RT build, ~1-3 min."""
import dataclasses
import os
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from retrieval_framework import pipeline as P
from retrieval_framework import run_smc as R

pytestmark = pytest.mark.slow
RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "w39b_smc_retrieval"


@pytest.fixture(scope="module")
def pipe():
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "smoke")
    cfg, preset = R.make_config(RUN_DIR)
    if preset != "smoke":
        pytest.skip(f"preset resolved to {preset!r}, not smoke")
    cfg = dataclasses.replace(cfg, count_max=50, warm_count_max=5)
    try:
        pipe = P.build_pipeline(cfg)
    except (FileNotFoundError, OSError, ImportError) as e:
        pytest.skip(f"cannot build real smoke pipeline ({type(e).__name__}: {e})")
    pipe.set_observations(np.zeros(pipe.n_bin), np.ones(pipe.n_bin))
    return pipe


def test_every_entry_point_rejects_uncertified_finite_spectrum(pipe):
    u0 = pipe.sample_prior_u(jax.random.PRNGKey(0), 1)[0]      # T-P-valid draw
    theta = pipe.theta_from_u(u0)
    # precondition: the forward is finite -- the certificate, not a NaN, rejects
    assert np.all(np.isfinite(np.asarray(pipe.observed_depth_model_jit(theta))))
    assert float(pipe.log_likelihood_u(u0)) <= -1e29
    for vg in (pipe.value_and_grad_block, pipe.value_and_grad_naive):
        L, G = vg(u0)
        assert float(L) <= -1e29 and np.all(np.asarray(G) == 0.0)
    U = pipe.sample_prior_u(jax.random.PRNGKey(1), 2)
    Y0, refs0, _S1 = P._blank_state(pipe, 2)
    L = pipe.batch_eval_cold_l(U, Y0, refs0)[0]
    assert np.all(np.asarray(L) <= -1e29)
