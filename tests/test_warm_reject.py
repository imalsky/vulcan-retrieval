"""Regression test for the warm-proposal convergence gate
(pipeline._make_batch_eval, warm + want_grad; retrieval_forward.chem_solve_warm_diag).

A warm_count_max-exhausted (non-converged) warm MALA proposal must be REJECTED (-1e30
L, dropped from n_bad_grad), NOT fed into the jvp/RT-vjp as a finite-likelihood MH
candidate, which surfaces as a spurious n_bad_grad RuntimeError (or a NaN
gradient) at SMC stage 0 and fails the timing calibration. See CLAUDE.md
"Init / mutation handling".

Also covers the warm-cap plumbing: the warm solvers run a TWIN runner
capped at warm_count_max < count_max, so a doomed proposal is cut off at the warm cap
(here 5) instead of marching to the cold cap (here 50) -- asserted via the observed
accept_count landing at the warm cap, far below the cold one.

This uses the REAL smoke pipeline (CO-only, fully offline; conftest.capped_smoke_pipe)
at warm_count_max=5, so every warm continuation from the baseline column is guaranteed
non-converged (the readiness floor count_min=120 alone forbids convergence in 5 steps).
It is a heavier integration test than the rest of the suite (~1-3 min: real
chemistry+RT build + a couple of XLA compiles) and SKIPS cleanly when the VULCAN-JAX /
ExoJax stack or its data/env is unavailable.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import pipeline as P  # noqa: E402

# SLOW: this module builds a REAL chemistry + RT pipeline (ExoJAX RT model,
# line lists, chemistry converged to steady state), so it costs
# minutes, not seconds. `pytest tests` still runs it; `pytest -m "not slow"`
# is the opt-in fast inner loop.
pytestmark = pytest.mark.slow

WARM_CMAX = 5     # conftest.CAPPED_WARM_CMAX: a warm continuation from baseline cannot converge
COLD_CMAX = 50    # conftest.CAPPED_COLD_CMAX: well above WARM_CMAX, proves the warm cap cut the loop
N = 4


@pytest.fixture(scope="module")
def smoke(capped_smoke_pipe):
    """(pipe, ACC, L_gated, G, n_bad, L_ungated) from the shared real smoke
    pipeline at warm_count_max=5 / count_max=50 in WARM mode (smc_chem_mode
    defaults to "cold", which would build the cold evaluators and test nothing
    about the warm cap)."""
    pipe = capped_smoke_pipe
    assert int(pipe.fwd.chem.warm_count_max) == WARM_CMAX
    assert int(pipe.fwd.chem.count_max) == COLD_CMAX
    U = pipe.sample_prior_u(jax.random.PRNGKey(0), N)
    Y0, refs0 = P._blank_state(pipe, N)
    C_ = jax.vmap(pipe.theta_from_u)(U)[:, : pipe.n_chem_tp]

    def _ac(cc, yw, rf):
        _y, cd = pipe.fwd.chem_solve_warm_diag(cc, yw, rf[0], rf[1])
        return jnp.asarray(cd.accept_count, jnp.int32)

    ACC = np.asarray(jax.vmap(_ac)(C_, Y0, refs0))
    L_g, G, _Yn, _rn, n_bad, _stats = jax.jit(pipe.batch_eval_move_vg)(U, Y0, refs0)
    L_u = jax.jit(pipe.batch_eval_move_l)(U, Y0, refs0)[0]   # gated too, since this pass
    return dict(pipe=pipe, cmax=int(pipe.fwd.chem.warm_count_max), ACC=ACC,
                L_gated=np.asarray(L_g), G=np.asarray(G), n_bad=int(n_bad),
                L_ungated=np.asarray(L_u))


def test_warm_diag_detects_exhaustion(smoke):
    # every warm continuation from the baseline column exhausts warm_count_max=5
    assert np.all(smoke["ACC"] >= smoke["cmax"])


def test_warm_cap_binds_not_cold_cap(smoke):
    # the warm cap cut the loop AT warm_count_max (accept_count lands just past
    # it), nowhere near the cold count_max -- the wall-clock point of the cap
    assert np.all(smoke["ACC"] <= WARM_CMAX + 1)
    assert np.all(smoke["ACC"] < COLD_CMAX)


def test_move_vg_rejects_nonconverged_without_raising(smoke):
    # a non-converged warm proposal is an MH rejection, not an AD pathology:
    assert smoke["n_bad"] == 0                       # ... it must not trip n_bad_grad
    assert np.all(np.isfinite(smoke["G"]))           # ... no NaN leaks into the gradient
    assert np.all(smoke["L_gated"] <= -1.0e29)       # ... and it is rejected (MH -inf)


def test_init_eval_is_uncapped(smoke):
    """The INIT gradient path must NOT run under the mutation cap: a phase-1 survivor
    that needs more than warm_count_max steps to re-certify is a healthy particle, not
    a doomed proposal (5 of 96 survivors gated at the warm cap raise a spurious
    'crippled cloud' RuntimeError). chem_solve_warm_diag_full must run the
    UNCAPPED runner: from the baseline column (which cannot certify in either budget
    here) the capped solve stops at WARM_CMAX while the full solve marches on to the
    cold cap."""
    pipe = smoke["pipe"]
    assert pipe.batch_eval_init_vg is not pipe.batch_eval_move_vg
    U = pipe.sample_prior_u(jax.random.PRNGKey(1), 1)
    C_ = jax.vmap(pipe.theta_from_u)(U)[:, : pipe.n_chem_tp]
    Y0, refs0 = P._blank_state(pipe, 1)
    _y, cd_cap = pipe.fwd.chem_solve_warm_diag(C_[0], Y0[0], refs0[0, 0], refs0[0, 1])
    _y, cd_full = pipe.fwd.chem_solve_warm_diag_full(C_[0], Y0[0], refs0[0, 0], refs0[0, 1])
    assert int(cd_cap.accept_count) <= WARM_CMAX + 1
    assert int(cd_full.accept_count) > WARM_CMAX + 1   # kept going past the mutation cap
    assert int(cd_full.accept_count) >= COLD_CMAX      # ... all the way to the cold cap


def test_gate_is_load_bearing(smoke):
    """The rejected proposals have a perfectly FINITE forward -- the gate, not a
    blown solve, is what rejects them.

    Both batched evaluators gate (the primal-only one is the FD reference for
    the gradient, the certificate's cold replay, and validate_warm's comparison
    arm, so it has to be the same likelihood function the sampler targets), so
    the load-bearing claim is made directly: run the RAW warm map, show its
    spectrum is finite, and show both evaluators reject it anyway."""
    pipe = smoke["pipe"]
    U = pipe.sample_prior_u(jax.random.PRNGKey(0), N)
    Theta = jax.vmap(pipe.theta_from_u)(U)
    C_ = Theta[:, : pipe.n_chem_tp]
    Y0, refs0 = P._blank_state(pipe, N)

    def _raw_depth(cc, th, yw, rf):
        y = pipe.fwd.chem_solve_warm_diag(cc, yw, rf[0], rf[1])[0]   # ungated, non-converged
        aux = pipe.fwd.aux_from_y(y, cc)
        cloud = th[pipe.cloud_idx[0]:pipe.cloud_idx[0] + pipe.n_cloud]
        return pipe.fwd.rt_depth(aux, th[pipe.lnR0_idx], cloud)

    depth = np.asarray(jax.vmap(_raw_depth)(C_, Theta, Y0, refs0))
    assert np.all(np.isfinite(depth)), "raw warm map blew up; this tests nothing"
    # ...and yet every one of them is rejected, by BOTH evaluators
    assert np.all(smoke["L_gated"] <= -1.0e29)
    assert np.all(smoke["L_ungated"] <= -1.0e29)
