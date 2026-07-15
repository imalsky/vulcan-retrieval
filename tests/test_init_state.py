"""Unit tests for the cold-init reject-and-cull path (pipeline._init_state /
_init_draw_count). Best practice for a full-kinetics forward: draws whose chemistry
does not converge within count_max are REJECTED (not carried, not raised on), and the
init OVERSAMPLES so the culled cloud still holds exactly N healthy particles. See
README.md sec K and CLAUDE.md.

No VULCAN/ExoJax here -- a tiny chem-LIKE fake pipe (has_chem_state=True) provides the
two batched evaluators _init_state calls, with count_max exhaustion made a deterministic
function of the draw so the test controls exactly which particles are rejected.
"""
import types
from typing import NamedTuple

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework.config_schema import ParamSpec  # noqa: E402


class _StubConvDiag(NamedTuple):
    """Pytree stand-in for forward.vulcan_chem.ConvDiag: _init_state phase 1 reads
    only .accept_count and .conv_normal (importing the real one would pull the
    heavy VULCAN env into these unit tests)."""
    accept_count: jnp.ndarray
    conv_normal: jnp.ndarray


def _chem_like_pipe(count_max=100, oversample=1.6, y_shape=(4, 3)):
    """A minimal has_chem_state pipe. A draw is 'exhausted' (hit count_max, i.e. did not
    converge) iff its first coordinate is >= 0 -- deterministic, so tests know exactly
    which particles _init_state must reject. A draw 'stall-certifies' (under the cap
    but not canonically certified) iff its first coordinate is exactly 0.5.
    Likelihoods/gradients are finite Gaussians."""
    cfg = C.Config(smc_num_particles=8, init_oversample=oversample)
    y_baseline = jnp.zeros(y_shape, jnp.float64)

    def cold_l_diag(U, Y0, refs0):
        n = U.shape[0]
        worst_accept = jnp.where(U[:, 0] >= 0.0, count_max, 1).astype(jnp.int32)
        stalled = U[:, 0] == 0.5
        worst_accept = jnp.where(stalled, 1, worst_accept).astype(jnp.int32)
        L = -0.5 * jnp.sum(U ** 2, axis=1)                       # always finite
        Y = jnp.broadcast_to(y_baseline[None], (n,) + y_shape)
        refs = jnp.zeros((n, 2), jnp.float64)
        return L, Y, refs, _StubConvDiag(worst_accept, ~stalled)

    def move_vg(U, Y, refs):
        L = -0.5 * jnp.sum(U ** 2, axis=1)
        G = -U
        stats = P._zero_eval_stats(U.shape[0], jnp.float64)
        return L, G, Y, refs, jnp.int32(0), None, stats          # survivors: no AD pathology

    def init_vg(U, Y, refs):
        """Phase-2 stub (7-tuple with an EvalStats tail, like a real pipeline's
        batch_eval_init_vg). Second coordinate > 0 -> cannot RE-certify warm
        (dead, ACC=count_max: cull); third coordinate > 0 -> RT/AD death (dead,
        certified + finite ACC: must raise)."""
        L = -0.5 * jnp.sum(U ** 2, axis=1)
        G = -U
        recert = U[:, 1] > 0.0
        rtdead = U[:, 2] > 0.0
        L = jnp.where(recert | rtdead, -1.0e30, L)
        ACC = jnp.where(recert, count_max, 1).astype(jnp.int32)
        stats = P._zero_eval_stats(U.shape[0], jnp.float64)._replace(acc=ACC)
        return L, G, Y, refs, jnp.int32(0), None, stats

    def _unused(*a, **k):                                        # never called on this path
        raise AssertionError("unexpected evaluator call")

    return P.Pipeline(
        cfg=cfg, dtype=jnp.float64, npdtype=np.float64, has_chem_state=True,
        y_baseline=y_baseline,
        fwd=types.SimpleNamespace(chem=types.SimpleNamespace(count_max=count_max)),
        batch_eval_cold_l_diag=cold_l_diag,
        batch_eval_cold_vg=_unused, batch_eval_cold_l=_unused,
        batch_eval_move_vg=move_vg, batch_eval_init_vg=init_vg,
        batch_eval_move_l=_unused,
    )


def _U(first_coords):
    """(n, 3) draws with a chosen first coordinate (controls exhaustion) and zeros else."""
    a = np.asarray(first_coords, float)
    return jnp.asarray(np.column_stack([a, np.zeros_like(a), np.zeros_like(a)]))


def test_init_draw_count_oversamples_chem_not_stub():
    pipe = _chem_like_pipe(oversample=1.6)
    assert P._init_draw_count(pipe, 50) == 80          # ceil(50 * 1.6)
    assert P._init_draw_count(pipe, 1) == 2            # never below the target

    specs = [ParamSpec("p", "p", "uniform", -8.0, 8.0, 0.0, "chem")]
    tf, lp, sp = P.make_uspace(specs, jnp.float64)
    stub = P.Pipeline(cfg=C.Config(smc_num_particles=8), dtype=jnp.float64,
                      npdtype=np.float64, n_dim=1, theta_from_u=tf, log_prior_u=lp,
                      sample_prior_u=sp, log_likelihood_u=lambda u: 0.0,
                      loglik_fwd=lambda u: 0.0)
    assert P._init_draw_count(stub, 50) == 50          # no chemistry -> no oversample


def test_init_state_rejects_nonconverged_and_keeps_target_n():
    pipe = _chem_like_pipe(count_max=100)
    # 12 draws: first 4 exhausted (>=0), last 8 healthy (<0). Ask for 8.
    U = _U([+1, +1, +1, +1, -1, -2, -3, -4, -5, -6, -7, -8])
    U_keep, L, G, Y, refs, DY, stats = P._init_state(pipe, U, target_n=8)

    assert U_keep.shape[0] == 8 and L.shape[0] == 8 and G.shape[0] == 8
    assert np.all(np.isfinite(np.asarray(L)))
    assert np.all(np.isfinite(np.asarray(G)))
    # exactly the healthy (first-coord < 0) draws survive, in draw order
    kept0 = np.asarray(U_keep)[:, 0]
    assert np.all(kept0 < 0)
    assert np.allclose(np.sort(kept0), np.sort([-1, -2, -3, -4, -5, -6, -7, -8]))


def test_init_state_culls_extra_survivors_to_exactly_target_n():
    pipe = _chem_like_pipe(count_max=100)
    # 10 healthy draws but only 6 requested -> keep the first 6, no rejection needed
    U = _U([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
    U_keep, L, G, Y, refs, DY, stats = P._init_state(pipe, U, target_n=6)
    assert U_keep.shape[0] == 6
    assert np.allclose(np.asarray(U_keep)[:, 0], [-1, -2, -3, -4, -5, -6])


def test_init_state_rejects_stall_certified_draws():
    """A draw whose cold solve exits under the cap WITHOUT the canonical
    certification (conv_normal False: the stall-fallback class behind NAS job
    65200's non-finite mutation gradients) is rejected in phase 1 exactly like a
    count_max exhaustion."""
    pipe = _chem_like_pipe(count_max=100)
    # 10 draws: draws 0 and 3 stall-certify (first coord == 0.5), rest healthy
    U = _U([0.5, -1, -2, 0.5, -3, -4, -5, -6, -7, -8])
    U_keep, L, G, Y, refs, DY, stats = P._init_state(pipe, U, target_n=8)
    kept0 = np.asarray(U_keep)[:, 0]
    assert np.all(kept0 < 0)                       # stall-certified draws rejected
    assert stats["n_stalled_init"] == 2
    assert stats["n_exhausted"] == 0


def test_init_state_raises_when_too_few_survivors():
    pipe = _chem_like_pipe(count_max=100)
    # 10 draws, 8 exhausted, only 2 healthy; asking for 8 must raise (systemic)
    U = _U([+1] * 8 + [-1, -2])
    with pytest.raises(RuntimeError, match="ran out of survivors"):
        P._init_state(pipe, U, target_n=8)


def test_init_state_all_healthy_default_target_is_len_u():
    pipe = _chem_like_pipe(count_max=100)
    U = _U([-1, -2, -3, -4])
    U_keep, L, G, Y, refs, DY, stats = P._init_state(pipe, U)   # target_n=None -> len(U)
    assert U_keep.shape[0] == 4
    assert np.all(np.isfinite(np.asarray(L)))


def test_init_phase2_culls_recert_failures_and_backfills():
    pipe = _chem_like_pipe(count_max=100)
    # 12 phase-1-healthy draws; draws 2 and 5 certify cold but cannot re-certify warm
    a = np.column_stack([-np.arange(1.0, 13.0), np.zeros(12), np.zeros(12)])
    a[2, 1] = 1.0
    a[5, 1] = 1.0
    U_keep, L, G, Y, refs, DY, stats = P._init_state(pipe, jnp.asarray(a), target_n=8)
    assert U_keep.shape[0] == 8 and L.shape[0] == 8
    # culled draws (first coords -3, -6) replaced by the next spares, order preserved
    assert np.allclose(np.asarray(U_keep)[:, 0], [-1, -2, -4, -5, -7, -8, -9, -10])
    assert np.all(np.isfinite(np.asarray(L))) and np.all(np.isfinite(np.asarray(G)))


def test_init_phase2_rt_death_raises():
    pipe = _chem_like_pipe(count_max=100)
    a = np.column_stack([-np.arange(1.0, 11.0), np.zeros(10), np.zeros(10)])
    a[3, 2] = 1.0                          # non-finite forward, NON-exhausted ACC
    with pytest.raises(RuntimeError, match="RT/AD"):
        P._init_state(pipe, jnp.asarray(a), target_n=8)


def test_init_phase2_spares_exhausted_raises():
    pipe = _chem_like_pipe(count_max=100)
    # 9 healthy phase-1 draws, target 8: killing 2 at phase 2 leaves only 7
    a = np.column_stack([-np.arange(1.0, 10.0), np.zeros(9), np.zeros(9)])
    a[0, 1] = 1.0
    a[1, 1] = 1.0
    with pytest.raises(RuntimeError, match="Spares exhausted"):
        P._init_state(pipe, jnp.asarray(a), target_n=8)
