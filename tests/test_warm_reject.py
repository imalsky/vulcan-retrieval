"""Regression test for the warm-proposal convergence gate
(pipeline._make_batch_eval, warm + want_grad; retrieval_forward.chem_solve_warm_diag).

A warm_count_max-exhausted (non-converged) warm MALA proposal must be REJECTED (-1e30
L, dropped from n_bad_grad), NOT fed into the jvp/RT-vjp as a finite-likelihood MH
candidate -- the pre-fix behavior that surfaced as a spurious n_bad_grad RuntimeError
(or a NaN gradient) at SMC stage 0 and made the timing calibration fail. See CLAUDE.md
"Cold-init reject-and-cull" (the warm-mutation analogue that was previously deferred).

Also covers the 2026-07-09 warm-cap plumbing: the warm solvers run a TWIN runner
capped at warm_count_max < count_max, so a doomed proposal is cut off at the warm cap
(here 5) instead of marching to the cold cap (here 50) -- asserted via the observed
accept_count landing at the warm cap, far below the cold one.

This builds the REAL smoke pipeline (CO-only, fully offline) with warm_count_max=5 so
every warm continuation from the baseline column is guaranteed non-converged (the
readiness floor count_min=120 alone forbids convergence in 5 steps). It is a heavier
integration test than the rest of the suite (~1-3 min: real chemistry+RT build + a
couple of XLA compiles) and SKIPS cleanly when the VULCAN-JAX / ExoJax stack or its
data/env is unavailable.
"""
import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework import run_smc as R  # noqa: E402

# tests/ and runs/ are siblings inside the vulcan-retrieval package dir
RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "w39b_smc_retrieval"
WARM_CMAX = 5     # so a warm continuation from baseline cannot converge (<= 5 steps)
COLD_CMAX = 50    # well above WARM_CMAX: proves the warm cap (not this) cut the loop
N = 4


@pytest.fixture(scope="module")
def smoke():
    """(pipe, ACC, L_gated, G, n_bad, L_ungated) from the real smoke pipeline at
    warm_count_max=5 / count_max=50; built once and shared. Skips if the chem
    stack/env is unavailable."""
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "smoke")
    try:
        cfg, preset = R.make_config(RUN_DIR)
        if preset != "smoke":
            pytest.skip(f"preset resolved to {preset!r}, not smoke")
        cfg = dataclasses.replace(cfg, count_max=COLD_CMAX, warm_count_max=WARM_CMAX)
        pipe = P.build_pipeline(cfg)
    except Exception as e:                       # missing fastchem / data / env / import
        pytest.skip(f"cannot build real smoke pipeline ({type(e).__name__}: {e})")
    pipe.set_observations(np.zeros(pipe.n_bin), np.ones(pipe.n_bin))

    U = pipe.sample_prior_u(jax.random.PRNGKey(0), N)
    Y0, refs0 = P._blank_state(pipe, N)
    C_ = jax.vmap(pipe.theta_from_u)(U)[:, : pipe.n_chem_tp]

    def _ac(cc, yw, rf):
        _y, ac = pipe.fwd.chem_solve_warm_diag(cc, yw, rf[0], rf[1])
        return jnp.asarray(ac, jnp.int32)

    ACC = np.asarray(jax.vmap(_ac)(C_, Y0, refs0))
    L_g, G, _Yn, _rn, n_bad, _dy, _ncap = jax.jit(pipe.batch_eval_move_vg)(U, Y0, refs0)
    L_u = jax.jit(pipe.batch_eval_move_l)(U, Y0, refs0)[0]
    return dict(pipe=pipe, cmax=int(pipe.fwd.chem.warm_count_max), ACC=ACC,
                L_gated=np.asarray(L_g), G=np.asarray(G), n_bad=int(n_bad),
                L_ungated=np.asarray(L_u))


def test_warm_diag_detects_exhaustion(smoke):
    # every warm continuation from the baseline column exhausts warm_count_max=5
    assert np.all(smoke["ACC"] >= smoke["cmax"])


def test_warm_cap_binds_not_cold_cap(smoke):
    # the twin warm-capped runner cut the loop AT warm_count_max (accept_count lands
    # just past it), nowhere near the cold count_max -- the wall-clock point of the cap
    assert np.all(smoke["ACC"] <= WARM_CMAX + 1)
    assert np.all(smoke["ACC"] < COLD_CMAX)
    assert int(smoke["pipe"].fwd.chem.count_max) == COLD_CMAX
    assert int(smoke["pipe"].fwd.chem.warm_count_max) == WARM_CMAX


def test_move_vg_rejects_nonconverged_without_raising(smoke):
    # a non-converged warm proposal is an MH rejection, not an AD pathology:
    assert smoke["n_bad"] == 0                       # ... it must not trip n_bad_grad
    assert np.all(np.isfinite(smoke["G"]))           # ... no NaN leaks into the gradient
    assert np.all(smoke["L_gated"] <= -1.0e29)       # ... and it is rejected (MH -inf)


def test_init_eval_is_uncapped(smoke):
    """The INIT gradient path must NOT run under the mutation cap: a phase-1 survivor
    that needs more than warm_count_max steps to re-certify is a healthy particle, not
    a doomed proposal (NAS job 64854 regression: 5/96 survivors gated at the warm cap
    -> spurious 'crippled cloud' RuntimeError). chem_solve_warm_diag_full must run the
    UNCAPPED runner: from the baseline column (which cannot certify in either budget
    here) the capped solve stops at WARM_CMAX while the full solve marches on to the
    cold cap."""
    pipe = smoke["pipe"]
    assert pipe.batch_eval_init_vg is not pipe.batch_eval_move_vg
    U = pipe.sample_prior_u(jax.random.PRNGKey(1), 1)
    C_ = jax.vmap(pipe.theta_from_u)(U)[:, : pipe.n_chem_tp]
    Y0, refs0 = P._blank_state(pipe, 1)
    _y, ac_cap = pipe.fwd.chem_solve_warm_diag(C_[0], Y0[0], refs0[0, 0], refs0[0, 1])
    _y, ac_full = pipe.fwd.chem_solve_warm_diag_full(C_[0], Y0[0], refs0[0, 0], refs0[0, 1])
    assert int(ac_cap) <= WARM_CMAX + 1
    assert int(ac_full) > WARM_CMAX + 1          # kept going past the mutation cap
    assert int(ac_full) >= COLD_CMAX             # ... all the way to the cold cap


def test_gate_is_load_bearing(smoke):
    # the ungated primal likelihood (move_l) carries these non-converged proposals as
    # FINITE MH candidates; the gated gradient evaluator (move_vg) rejects them. Every
    # finite-forward proposal here is non-converged (see test_warm_diag_detects_...), so
    # each must be gated out -- proving the gate changes behavior, not just documents it.
    finite_ungated = smoke["L_ungated"] > -1.0e29
    if not np.any(finite_ungated):
        pytest.skip("no finite-forward non-converged proposal in this batch to compare")
    assert np.all(smoke["L_gated"][finite_ungated] <= -1.0e29)
