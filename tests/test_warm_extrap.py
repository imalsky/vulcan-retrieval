"""Tangent-extrapolated warm starts (config warm_extrapolate; pipeline._make_batch_eval
want_dy + _make_mutation seed).

Physics invariant: the extrapolated seed Y + DY.(theta_new - theta_cur), solved with
refs = the PROPOSAL's (lnZ, c_o) (the no-double-scaling recipe), must relax to the SAME
certified steady state as the plain warm seed (Y, refs) -- the extrapolation changes
wall time, never the target. Also covers the DY plumbing end-to-end (eval return slot,
mutation carry/select) and the config validation (extrapolation requires the warm map).

Builds the real smoke pipeline once (CO-only, offline; a few compiles, ~minutes) and
SKIPS cleanly when the chem stack/env is unavailable, same pattern as test_warm_reject.
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
from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import run_smc as R  # noqa: E402

# tests/ and runs/ are siblings inside the vulcan-retrieval package dir
RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "w39b_smc_retrieval"


def _smoke_cfg():
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "smoke")
    cfg, preset = R.make_config(RUN_DIR)
    if preset != "smoke":
        pytest.skip(f"preset resolved to {preset!r}, not smoke")
    return cfg


def test_extrapolate_requires_warm_map():
    cfg = dataclasses.replace(_smoke_cfg(), warm_extrapolate=True, smc_chem_mode="cold")
    with pytest.raises(ValueError, match="warm_extrapolate"):
        C.validate_config(cfg)


@pytest.fixture(scope="module")
def extrap():
    """One converged particle + its tangents from the real smoke pipeline built with
    warm_extrapolate=True: (pipe, U0, Y_cold, refs_cold, DY, L0, G0)."""
    cfg = dataclasses.replace(_smoke_cfg(), warm_extrapolate=True)
    try:
        pipe = P.build_pipeline(cfg)
    except Exception as e:                       # missing fastchem / data / env / import
        pytest.skip(f"cannot build real smoke pipeline ({type(e).__name__}: {e})")
    pipe.set_observations(np.zeros(pipe.n_bin), np.ones(pipe.n_bin))

    # box-center draw (a mid-prior planet, far from the non-convergent corners)
    U0 = jnp.zeros((1, pipe.n_dim), jnp.float64)
    Y0, refs0 = P._blank_state(pipe, 1)
    L_c, Y_cold, refs_cold = jax.jit(pipe.batch_eval_cold_l)(U0, Y0, refs0)
    assert float(L_c[0]) > -1.0e29, "box-center draw failed to cold-converge"
    L0, G0, _, _, n_bad, DY, _ncap = jax.jit(pipe.batch_eval_move_vg)(U0, Y_cold, refs_cold)
    assert int(n_bad) == 0
    return dict(pipe=pipe, U0=U0, Y=Y_cold, refs=refs_cold, DY=DY,
                L0=L0, G0=G0)


def test_dy_shape_and_finiteness(extrap):
    pipe, DY = extrap["pipe"], extrap["DY"]
    nz, ni = np.asarray(extrap["Y"]).shape[1:]
    assert np.asarray(DY).shape == (1, pipe.n_chem_tp, nz, ni)
    assert np.all(np.isfinite(np.asarray(DY)))
    assert np.any(np.asarray(DY) != 0.0)         # tangents actually relaxed, not zeros


def test_extrapolated_seed_reaches_same_likelihood(extrap):
    """The load-bearing check: plain warm seed and extrapolated seed (with proposal
    refs -- no double scaling) converge to the same likelihood at a MALA-sized move."""
    pipe = extrap["pipe"]
    U0, Y, refs, DY = extrap["U0"], extrap["Y"], extrap["refs"], extrap["DY"]
    U1 = U0 + 0.05                                                  # MALA-sized move
    C0 = jax.vmap(pipe.theta_from_u)(U0)[:, :pipe.n_chem_tp]
    C1 = jax.vmap(pipe.theta_from_u)(U1)[:, :pipe.n_chem_tp]
    Y_seed = jnp.maximum(Y + jnp.einsum("nkij,nk->nij", DY, C1 - C0), 0.0)
    move_l = jax.jit(pipe.batch_eval_move_l)
    L_plain = float(move_l(U1, Y, refs)[0][0])
    L_ex = float(move_l(U1, Y_seed, C1[:, :2])[0][0])
    assert L_plain > -1.0e29 and L_ex > -1.0e29, "warm solve failed to converge"
    assert abs(L_ex - L_plain) < 1e-2, (
        f"extrapolated seed changed the converged likelihood: {L_ex} vs {L_plain} "
        "(double-scaling or a broken seed)")


def test_mutation_kernel_carries_tangents(extrap):
    pipe = extrap["pipe"]
    U0, Y, refs, DY = extrap["U0"], extrap["Y"], extrap["refs"], extrap["DY"]
    L0, G0 = extrap["L0"], extrap["G0"]
    mutate = P._make_mutation(pipe, 1)
    out = mutate(jax.random.PRNGKey(3), U0, Y, refs, L0, G0, DY,
                 jnp.asarray(0.5, jnp.float64), jnp.asarray(0.01, jnp.float64),
                 jnp.ones((pipe.n_dim,), jnp.float64))
    U1, Y1, refs1, L1, G1, DY1, acc, n_bad, n_capped = out
    assert int(n_bad) == 0
    assert int(n_capped) == 0
    assert np.asarray(DY1).shape == np.asarray(DY).shape
    assert np.all(np.isfinite(np.asarray(DY1)))
    assert np.all(np.isfinite(np.asarray(L1)))
