"""No entry point accepts a finite spectrum from an uncertified cold solve.

The batched cold evaluator rejects it and the synthetic injection refuses it as a truth, both on the solver's
conv_normal (False on every cap exit). The smoke pipeline at count_max=50 < count_min (conftest.capped_smoke_pipe) has
no solve that can certify, while its column and spectrum stay finite. Slow: builds and compiles the real chemistry.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from retrieval_framework import pipeline as P

pytestmark = pytest.mark.slow


def test_every_entry_point_rejects_uncertified_finite_spectrum(capped_smoke_pipe):
    pipe = capped_smoke_pipe
    u0 = pipe.sample_prior_u(jax.random.PRNGKey(0), 1)[0]  # T-P-valid draw
    theta = pipe.theta_from_u(u0)
    # precondition: the forward is finite -- the certificate, not a NaN, rejects
    mu, ok = pipe.binned_ok_jit(theta)
    assert np.all(np.isfinite(np.asarray(mu)))
    assert not bool(ok)
    U = pipe.sample_prior_u(jax.random.PRNGKey(1), 2)
    Y0, refs0 = P._blank_state(pipe, 2)
    ev = jax.jit(pipe._make_batch_eval("cold"))
    L, _, _, st = ev(U, Y0, refs0, np.ones(2, bool))
    assert not np.any(np.asarray(st.conv_ok))
    assert np.all(np.asarray(L) <= P.REJECT_BELOW)
    # the truth is refused before set_observations would be called a second time
    with pytest.raises(RuntimeError, match="truth chemistry does not certify"):
        P.generate_observations(pipe, seed=0)
