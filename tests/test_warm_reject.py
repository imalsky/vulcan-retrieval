"""Regression test for the warm-proposal convergence gate
(pipeline._make_batch_eval, warm + want_grad; retrieval_forward.chem_solve_warm_diag).

A warm_count_max-exhausted (non-converged) warm MALA proposal must be REJECTED (-1e30
L, dropped from n_bad_grad), not fed into the jvp/RT-vjp as a finite-likelihood MH
candidate.

Also covers the warm cap: it rides the runner carry per lane, so a doomed
proposal stops at warm_count_max (here 5), not at the cold cap (here 50).

This uses the REAL smoke pipeline (CO-only, fully offline; conftest.capped_smoke_pipe)
at warm_count_max=5, so every warm continuation from the baseline column is guaranteed
non-converged (the readiness floor count_min=120 alone forbids convergence in 5 steps).
The claims are ONE test: xdist builds a module or session fixture once per
worker, so separate tests sharing one rebuilt the pipeline and its compiles
(~5 min, ~14 GB) on every worker they landed on. SKIPS cleanly when the
VULCAN-JAX / ExoJax stack or its data/env is unavailable.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from conftest import CAPPED_COLD_CMAX as COLD_CMAX, CAPPED_WARM_CMAX as WARM_CMAX  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402

# SLOW (full tier only): this module builds a REAL chemistry + RT pipeline
# (ExoJAX RT model, k-tables, chemistry converged to steady state), so it
# costs minutes, not seconds.
pytestmark = pytest.mark.slow

N = 4


def test_warm_cap_rejects_mutation_proposals_not_init(capped_smoke_pipe):
    """The shared smoke pipeline at warm_count_max=5 / count_max=50 in WARM
    mode (smc_chem_mode defaults to "cold", which would build the cold
    evaluators and test nothing about the warm cap)."""
    pipe = capped_smoke_pipe
    assert int(pipe.fwd.chem.warm_count_max) == WARM_CMAX
    assert int(pipe.fwd.chem.count_max) == COLD_CMAX
    U = pipe.sample_prior_u(jax.random.PRNGKey(0), N)
    Y0, refs0 = P._blank_state(pipe, N)
    Theta = jax.vmap(pipe.theta_from_u)(U)
    C_ = Theta[:, : pipe.n_chem_tp]

    def _ac(cc, yw, rf):
        _y, cd = pipe.fwd.chem_solve_warm_diag(cc, yw, rf[0], rf[1])
        return jnp.asarray(cd.accept_count, jnp.int32)

    ACC = np.asarray(jax.vmap(_ac)(C_, Y0, refs0))
    L_g, G, _Yn, _rn, n_bad, _stats = jax.jit(pipe.batch_eval_move_vg)(U, Y0, refs0)
    L_gated, G = np.asarray(L_g), np.asarray(G)
    L_ungated = np.asarray(jax.jit(pipe.batch_eval_move_l)(U, Y0, refs0)[0])

    # every warm continuation from the baseline column exhausts warm_count_max=5
    assert np.all(ACC >= WARM_CMAX)
    # the warm cap cut the loop AT warm_count_max (accept_count lands just past
    # it), nowhere near the cold count_max -- the wall-clock point of the cap
    assert np.all(ACC <= WARM_CMAX + 1)
    assert np.all(ACC < COLD_CMAX)

    # a non-converged warm proposal is an MH rejection, not an AD pathology:
    assert int(n_bad) == 0                           # ... it must not trip n_bad_grad
    assert np.all(np.isfinite(G))                    # ... no NaN leaks into the gradient
    assert np.all(L_gated <= P.REJECT_BELOW)         # ... and it is rejected (MH -inf)

    # The gate is load-bearing: the rejected proposals have a perfectly FINITE
    # forward, so the gate, not a blown solve, rejects them. Both batched
    # evaluators gate (the primal-only one is the FD reference for the
    # gradient, the certificate's cold replay and validate_warm's comparison
    # arm, so it has to be the likelihood the sampler targets): run the RAW
    # warm map, show its spectrum is finite, and show both evaluators reject it.
    def _raw_depth(cc, th, yw, rf):
        y = pipe.fwd.chem_solve_warm_diag(cc, yw, rf[0], rf[1])[0]   # ungated, non-converged
        aux = pipe.fwd.aux_from_y(y, cc)
        cloud = th[pipe.cloud_idx[0]:pipe.cloud_idx[0] + pipe.n_cloud]
        return pipe.fwd.rt_depth(aux, th[pipe.lnR0_idx], cloud)

    depth = np.asarray(jax.vmap(_raw_depth)(C_, Theta, Y0, refs0))
    assert np.all(np.isfinite(depth)), "raw warm map blew up; this tests nothing"
    # ...and yet the primal evaluator rejects every one too (the gated one above)
    assert np.all(L_ungated <= P.REJECT_BELOW)

    # The INIT gradient path must NOT run under the mutation cap: a phase-1
    # survivor that needs more than warm_count_max steps to re-certify is a
    # healthy particle, not a doomed proposal. chem_solve_warm_diag_full runs
    # the UNCAPPED runner: from the baseline column (which cannot certify in
    # either budget here) the capped solve stops at WARM_CMAX while the full
    # solve marches on to the cold cap.
    assert pipe.batch_eval_init_vg is not pipe.batch_eval_move_vg
    U1 = pipe.sample_prior_u(jax.random.PRNGKey(1), 1)
    C1 = jax.vmap(pipe.theta_from_u)(U1)[:, : pipe.n_chem_tp]
    Y1, refs1 = P._blank_state(pipe, 1)
    _y, cd_cap = pipe.fwd.chem_solve_warm_diag(C1[0], Y1[0], refs1[0, 0], refs1[0, 1])
    _y, cd_full = pipe.fwd.chem_solve_warm_diag_full(C1[0], Y1[0], refs1[0, 0], refs1[0, 1])
    assert int(cd_cap.accept_count) <= WARM_CMAX + 1
    assert int(cd_full.accept_count) > WARM_CMAX + 1   # kept going past the mutation cap
    assert int(cd_full.accept_count) >= COLD_CMAX      # ... all the way to the cold cap
