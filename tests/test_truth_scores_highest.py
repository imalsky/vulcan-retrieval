"""The forward map finds its own truth, with no sampler (oracle.md, "Success: the repository's tests").

Synthetic data are made at the smoke case's truth, and the likelihood scores the truth against one perturbation per
inferred parameter, each a large step across its prior and inside it, every point solved cold through the chemistry
and the RT and certified. The data are noise-free: on the smoke case's three CO bins, a lnKzz step of 0.3 prior widths
moves chi^2 by about 0.002, far below the random term of the 120 ppm synthetic noise, so with noise the comparison would
test the draw, not the forward map.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest

from conftest import RUN_DIR

pytestmark = pytest.mark.slow

# A perturbation moves one parameter by this fraction of its prior width, toward the prior's interior.
PERTURB_FRACTION = 0.3
# The step cap the job-path test uses: the truth's second stage certifies within it.
COUNT_MAX = 2000
# The cloud of the smoke truth (log10 kappa -6.5) is too thin for cloud_alpha to move the spectrum; at -3.0 it does.
TRUTH_LOG10KAPPA_CLOUD = -3.0


@pytest.mark.skipif(importlib.util.find_spec("exojax") is None, reason="needs the chemistry and RT stack")
def test_the_truth_scores_highest_against_every_one_parameter_perturbation():
    import jax
    import jax.numpy as jnp

    from retrieval_framework import pipeline as P
    from retrieval_framework import run_setup as R
    from retrieval_framework.forward import config as _config  # noqa: F401  # hands the engine its data root

    jax.config.update("jax_enable_x64", True)
    cfg = dataclasses.replace(
        R.load_spec(RUN_DIR / "smoke.yaml").cfg, count_max=COUNT_MAX, truth_log10kappa_cloud=TRUTH_LOG10KAPPA_CLOUD
    )
    pipe = P.build_pipeline(cfg)
    truth = np.asarray(pipe.param_truth, np.float64)
    width = pipe.param_prior_hi - pipe.param_prior_lo
    rows = [truth]
    for i in range(truth.size):
        row = truth.copy()
        up = truth[i] + PERTURB_FRACTION * width[i]
        row[i] = up if up < pipe.param_prior_hi[i] else truth[i] - PERTURB_FRACTION * width[i]
        rows.append(row)
    scored = [pipe.binned_ok_jit(jnp.asarray(r)) for r in rows]
    for name, r, (_, ok) in zip(["truth", *pipe.names], rows, scored, strict=True):
        assert bool(ok), f"{name}: not certified"
        assert bool(pipe.tp_valid(jnp.asarray(r))), f"{name}: T-P outside the window"

    pipe.set_observations(np.asarray(scored[0][0]), np.asarray(pipe.obs["sigma"]))
    log_l = [
        float(pipe._gauss_loglik(jnp.asarray(mu), jnp.asarray(r))) for r, (mu, _) in zip(rows, scored, strict=True)
    ]
    worse = {name: log_l[0] - lp for name, lp in zip(pipe.names, log_l[1:], strict=True)}
    assert all(d > 0.0 for d in worse.values()), f"log L(truth) - log L(perturbed): {worse}"
