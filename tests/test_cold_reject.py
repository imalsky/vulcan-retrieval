"""Every likelihood entry point rejects a FINITE spectrum from an uncertified cold
solve: the scalar log_likelihood_u and the batched cold evaluator run_nautilus
calls apply one certificate (the conv_normal gate plus the count_max cap). Real
smoke pipeline at count_max=50 < count_min (conftest.capped_smoke_pipe), so no
cold solve can certify while its column, and hence its spectrum, stays finite.
Slow (full tier): builds and compiles the real chemistry."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from retrieval_framework import pipeline as P

pytestmark = pytest.mark.slow


def test_every_entry_point_rejects_uncertified_finite_spectrum(capped_smoke_pipe):
    pipe = capped_smoke_pipe
    u0 = pipe.sample_prior_u(jax.random.PRNGKey(0), 1)[0]      # T-P-valid draw
    theta = pipe.theta_from_u(u0)
    # precondition: the forward is finite -- the certificate, not a NaN, rejects
    assert np.all(np.isfinite(np.asarray(pipe.observed_depth_model_jit(theta))))
    assert float(pipe.log_likelihood_u(u0)) <= P.REJECT_BELOW
    U = pipe.sample_prior_u(jax.random.PRNGKey(1), 2)
    Y0, refs0 = P._blank_state(pipe, 2)
    ev = jax.jit(pipe._make_batch_eval("cold", diag=True, rt_kept_only=True))
    L = ev(U, Y0, refs0)[0]
    assert np.all(np.asarray(L) <= P.REJECT_BELOW)
