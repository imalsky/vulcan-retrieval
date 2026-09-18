"""The cold two-stage gradient path with its stages SPLIT must be the same program.

pipeline._make_batch_eval runs the cold two-stage chemistry as two explicit jvps
(stage 1 at baseline composition, then stage 2 warm-started from it) and returns
the stage-1 result and its lnKzz/T-P tangents as a Stage1Cache value. That is the
single-chain jvp regrouped, not a different map: the primal (Y, L) must be
bit-identical and the gradient may differ only by XLA fusion.

The BLOCK evaluator (stage2_only) reads that cache instead of re-solving stage 1.
It is only correct where the proposal's theta[2:n_chem_tp] is bit-identical to
the cached column's -- there is no hit/miss branch, the block MALA sweep's
invariant is what guarantees it -- so the tests below pin both sides: a z-only
move reproduces the full evaluator, and a moved stage-1 dim reads as a miss.

Builds the REAL smoke pipeline (chemistry + RT, fully offline) at the case's own
caps, so it costs minutes and SKIPS cleanly when the stack or its data is absent.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework import run_smc as R  # noqa: E402

# SLOW: builds a real chemistry + RT pipeline (see CLAUDE.md, "Layout / entry points").
pytestmark = pytest.mark.slow

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "w39b_smc_retrieval"


@pytest.fixture(scope="module")
def smoke():
    """(pipe, U) from the real smoke pipeline at its own caps; built once."""
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "smoke")
    try:
        cfg, preset = R.make_config(RUN_DIR)
        if preset != "smoke":
            pytest.skip(f"preset resolved to {preset!r}, not smoke")
        pipe = P.build_pipeline(cfg)
    # Skip ONLY on a missing-data or missing-dependency environment; a broader
    # handler reports a real forward-model break as a green skip.
    except (FileNotFoundError, OSError, ImportError) as e:
        pytest.skip(f"cannot build real smoke pipeline ({type(e).__name__}: {e})")
    if not bool(pipe.fwd.two_stage):
        pytest.skip("case is single-stage: there is no stage-1 result to split out")
    pipe.set_observations(np.zeros(pipe.n_bin), np.ones(pipe.n_bin))
    # the same three probe particles smoke_retrieval uses: u0 and u0 +- du
    u0 = jnp.asarray(np.linspace(-0.35, 0.4, pipe.n_dim))
    du = jnp.asarray(np.linspace(-0.06, 0.09, pipe.n_dim))
    return pipe, jnp.stack([u0, u0 + du, u0 - du])


def test_stage_split_matches_single_chain(smoke):
    pipe, U = smoke
    Y0, refs0, _S1 = P._blank_state(pipe, int(U.shape[0]))
    legacy = jax.jit(pipe._make_batch_eval("cold", True, split_stage1=False))
    L_new, G_new, Y_new, _r, S1, n_bad, _s = jax.jit(pipe.batch_eval_cold_vg)(
        U, Y0, refs0)
    L_ref, G_ref, Y_ref, _rr, S1_ref, _nb, _sr = legacy(U, Y0, refs0)

    assert int(n_bad) == 0
    assert S1_ref is None and S1 is not None

    Y_new, Y_ref = np.asarray(Y_new), np.asarray(Y_ref)
    if not np.array_equal(Y_new, Y_ref):
        d = np.abs(Y_new - Y_ref)
        rel = float(np.max(d / np.maximum(np.abs(Y_ref), 1e-300)))
        ulp = float(np.max(d / np.maximum(np.spacing(np.abs(Y_ref)), 1e-300)))
        pytest.fail(f"stage-split primal is NOT bit-exact: max rel {rel:.3e}, "
                    f"max {ulp:.3g} ulp")
    assert np.array_equal(np.asarray(L_new), np.asarray(L_ref))

    # Norm-relative, never componentwise (CLAUDE.md, "Opacity: correlated-k").
    # CALIBRATED, not assumed: the two routes batch the stage-1 tangent at
    # different widths (3 lanes vs 5), so XLA fuses the jvp'd while_loop
    # differently. Measured 1.1e-10 on these three probe points; the gate sits
    # ~90x above it and still leaves five orders of margin against the wiring
    # bug it exists to catch (the repo's staged-vs-block gate is 1e-5).
    G_new, G_ref = np.asarray(G_new), np.asarray(G_ref)
    dg = float(np.max(np.abs(G_new - G_ref)) / max(float(np.max(np.abs(G_ref))), 1e-300))
    assert dg < 1e-8, f"split-vs-legacy gradient disagrees at {dg:.3e}"

    # The cache records the theta[2:n_chem_tp] each stage-1 column was solved at.
    # Reference it through jit: the pipeline computes theta inside a jit, and an
    # EAGER vmap of the same theta_from_u lands up to 4 ulp away.
    h = np.asarray(jax.jit(jax.vmap(pipe.theta_from_u))(U))[:, 2:pipe.n_chem_tp]
    assert np.array_equal(np.asarray(S1.h), h)
    assert S1.y1.shape == Y_new.shape
    assert S1.dy1.shape == (int(U.shape[0]), pipe.n_chem_tp - 2) + Y_new.shape[1:]


@pytest.fixture(scope="module")
def block_eval(smoke):
    """(L_z, L_c) at a z-block-only displacement, plus the arrays test 2 checks.

    One shared fixture: each entry costs a real stage-2 solve + RT, and tests 2
    and 3 gate the SAME two likelihood vectors at different thresholds.
    """
    pipe, U = smoke
    if getattr(pipe, "batch_eval_move_z_vg", None) is None:
        pytest.skip("no block evaluator (cold two-stage gradient path only)")
    n = int(U.shape[0])
    Y0, refs0, _ = P._blank_state(pipe, n)
    cold_vg = jax.jit(pipe.batch_eval_cold_vg)
    move_z = jax.jit(pipe.batch_eval_move_z_vg)

    _L, _G, _Y, _r, S1, n_bad, _s = cold_vg(U, Y0, refs0)
    assert int(n_bad) == 0 and S1 is not None
    # displace ONLY the dims stage 1 does not depend on: the cache stays exact
    z_idx = jnp.asarray([i for i in range(pipe.n_dim) if i not in pipe.h_idx])
    U2 = U.at[:, z_idx].add(0.05)

    L_z, G_z, Y_z, _rz, S1_z, n_bad_z, stats = move_z(U2, S1)
    L_f, G_f, Y_f, _rf, S1_f, _nbf, _sf = cold_vg(U2, Y0, refs0)
    L_c = jax.jit(pipe.batch_eval_cold_l)(U2, Y0, refs0)[0]
    # a stage-1 dim moved -> the cache is stale and the hit bit must say so
    U3 = U2.at[:, 2].set(jnp.round(U2[:, 2], 6))
    stats_miss = move_z(U3, S1)[6]
    return dict(S1=S1, S1_z=S1_z, S1_f=S1_f, n_bad_z=int(n_bad_z), stats=stats,
                stats_miss=stats_miss, L_z=np.asarray(L_z), L_f=np.asarray(L_f),
                L_c=np.asarray(L_c), G_z=np.asarray(G_z), G_f=np.asarray(G_f),
                Y_z=np.asarray(Y_z), Y_f=np.asarray(Y_f))


def test_block_evaluator_reuses_the_cache_and_is_the_same_program(block_eval):
    """A move that leaves theta[2:n_chem_tp] fixed must give the FULL evaluator's
    answer while skipping stage 1: same primal, same gradient, cached column."""
    b = block_eval
    assert b["n_bad_z"] == 0
    assert bool(np.all(np.asarray(b["stats"].s1_hit))), "cache miss on a z-only move"
    assert all(np.array_equal(np.asarray(getattr(b["S1_z"], f)),
                              np.asarray(getattr(b["S1"], f)))
               for f in ("y1", "dy1", "h")), (
        "the block evaluator must return its INPUT cache, unchanged")

    for name in ("Y", "G"):
        got, ref = b[f"{name}_z"], b[f"{name}_f"]
        if not np.array_equal(got, ref):
            # norm-relative, never componentwise (CLAUDE.md, "Opacity: correlated-k")
            rel = float(np.max(np.abs(got - ref))
                        / max(float(np.max(np.abs(ref))), 1e-300))
            print(f"block-vs-full {name} norm-relative difference {rel:.3e}")
            assert rel < 1e-12, f"block-vs-full {name} disagrees at {rel:.3e}"

    dl = float(np.max(np.abs(b["L_z"] - b["L_c"])))
    print(f"max|logL block - logL cold-primal| = {dl:.3e}")
    assert dl < 1e-10

    # stage 1 depends on theta[2:] alone, so the displaced full eval re-solved
    # the SAME column the cache holds
    assert np.array_equal(np.asarray(b["S1_f"].y1), np.asarray(b["S1"].y1))
    assert not bool(np.any(np.asarray(b["stats_miss"].s1_hit))), (
        "a moved stage-1 dim must read as a cache MISS on every particle")


def test_block_likelihood_is_inside_the_warm_validator_gate(block_eval):
    """The shipped PASS gate on a likelihood remapping (validate_warm), applied to
    the block move: it is the number a run's certificate would have to defend."""
    from retrieval_framework import validate_warm as VW
    dl = float(np.max(np.abs(block_eval["L_z"] - block_eval["L_c"])))
    print(f"max|dlogL| = {dl:.3e} against DLOGL_MAX_PASS = {VW.DLOGL_MAX_PASS}")
    assert dl < VW.DLOGL_MAX_PASS
