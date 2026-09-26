"""The cold two-stage gradient path with its stages SPLIT must be the same program.

pipeline._make_batch_eval runs the cold two-stage chemistry as two explicit jvps
(stage 1 at baseline composition on the lnKzz/T-P directions only, then stage 2
warm-started from it). That is the single-chain jvp regrouped, not a different
map: the primal (Y, L) must be bit-identical and the gradient may differ only by
XLA fusion. The single-chain reference lives here: one jvp per chemistry
direction through the whole cold map (fwd.chem_solve_cold_diag_batch), then the
pipeline's own RT stage.

Builds the REAL smoke pipeline (chemistry + RT, fully offline) at the case's own
caps, so it costs minutes and SKIPS cleanly when the stack or its data is absent.
"""
import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from conftest import build_smoke_pipe  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402

# SLOW: builds a real chemistry + RT pipeline (see CLAUDE.md, "Layout / entry points").
pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def smoke():
    """(pipe, U) from the real smoke pipeline at its own caps; built once."""
    pipe = build_smoke_pipe()
    # the same three probe particles smoke_retrieval uses: u0 and u0 +- du
    u0 = jnp.asarray(np.linspace(-0.35, 0.4, pipe.n_dim))
    du = jnp.asarray(np.linspace(-0.06, 0.09, pipe.n_dim))
    return pipe, jnp.stack([u0, u0 + du, u0 - du])


def _single_chain_vg(pipe, U):
    """(L, G, Y) with the cold two-stage chemistry differentiated as ONE chain:
    each chemistry direction's tangent runs through stage 1 and on into stage 2
    inside the same jvp; then the pipeline's RT vjp, chained to u-space."""
    fwd = pipe.fwd
    Theta = jax.vmap(pipe.theta_from_u)(U)
    _, dTh = jax.vmap(lambda u: jax.jvp(pipe.theta_from_u, (u,), (jnp.ones_like(u),)))(U)
    C_ = Theta[:, :pipe.n_chem_tp]

    def chain(c):
        Y, _cd = fwd.chem_solve_cold_diag_batch(c)
        return jax.vmap(fwd.aux_from_y)(Y, c), Y

    (AUX_l, Y_l), (DAUX_l, _dY) = jax.vmap(
        lambda v: jax.jvp(chain, (C_,), (jnp.broadcast_to(v, C_.shape),))
    )(jnp.eye(pipe.n_chem_tp, dtype=pipe.dtype))
    AUX = jax.tree_util.tree_map(lambda x: x[0], AUX_l)
    DAUX = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), DAUX_l)
    L, g, _bad = jax.vmap(pipe._rt_val_grad)((AUX, DAUX, Theta))
    return L, g * dTh, Y_l[0]


def test_stage_split_matches_single_chain(smoke):
    pipe, U = smoke
    Y0, refs0 = P._blank_state(pipe, int(U.shape[0]))
    L_new, G_new, Y_new, _r, n_bad, _s = jax.jit(pipe.batch_eval_cold_vg)(
        U, Y0, refs0)
    L_ref, G_ref, Y_ref = jax.jit(lambda U_: _single_chain_vg(pipe, U_))(U)

    assert int(n_bad) == 0
    # certified probe draws, or the comparison below is between two rejections
    assert np.all(np.asarray(L_new) > P.REJECT_BELOW)

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
    print(f"split-vs-single-chain max|dG|/max|G| = {dg:.3e}")
    assert dg < 1e-8, f"split-vs-single-chain gradient disagrees at {dg:.3e}"
