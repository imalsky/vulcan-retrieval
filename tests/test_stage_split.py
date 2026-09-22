"""The cold two-stage gradient path with its stages SPLIT must be the same program.

pipeline._make_batch_eval runs the cold two-stage chemistry as two explicit jvps
(stage 1 at baseline composition on the lnKzz/T-P directions only, then stage 2
warm-started from it). That is the single-chain jvp regrouped, not a different
map: the primal (Y, L) must be bit-identical and the gradient may differ only by
XLA fusion.

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
    Y0, refs0 = P._blank_state(pipe, int(U.shape[0]))
    legacy = jax.jit(pipe._make_batch_eval("cold", True, split_stage1=False))
    L_new, G_new, Y_new, _r, n_bad, _s = jax.jit(pipe.batch_eval_cold_vg)(
        U, Y0, refs0)
    L_ref, G_ref, Y_ref, _rr, _nb, _sr = legacy(U, Y0, refs0)

    assert int(n_bad) == 0

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
