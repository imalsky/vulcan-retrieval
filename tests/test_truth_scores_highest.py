"""The forward map finds its own truth, with no sampler (oracle.md, "Success: the repository's tests").

``generate_observations`` makes synthetic data at the smoke case's truth, and the run's own batched log-likelihood
(``run_nautilus.make_loglike``, the cold evaluator nautilus calls) scores the truth against one perturbation per
inferred parameter, each a large step across its prior and inside it, every point solved cold through the chemistry and
the RT and certified. The injected noise is scaled to nothing, so the data are the truth spectrum: on the smoke case's
three CO bins a lnKzz step of 0.3 prior widths moves chi^2 by about 0.002, far below the random term of the 120 ppm
synthetic noise, so with noise the comparison would test the draw, not the forward map.
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
# Scales the injected noise far below one ulp of the depth, so the injected data equal the truth spectrum bit for bit.
NOISE_FREE_INFLATION = 1.0e-30


@pytest.mark.skipif(importlib.util.find_spec("exojax") is None, reason="needs the chemistry and RT stack")
def test_the_truth_scores_highest_against_every_one_parameter_perturbation():
    import jax

    from retrieval_framework import pipeline as P
    from retrieval_framework import run_setup as R
    from retrieval_framework.forward import config as _config  # noqa: F401  # hands the engine its data root
    from retrieval_framework.run_nautilus import make_loglike

    jax.config.update("jax_enable_x64", True)
    cfg = dataclasses.replace(
        R.load_spec(RUN_DIR / "smoke.yaml").cfg,
        count_max=COUNT_MAX,
        truth_log10kappa_cloud=TRUTH_LOG10KAPPA_CLOUD,
        truth_noise_inflation=NOISE_FREE_INFLATION,
    )
    pipe = P.build_pipeline(cfg)
    obs = P.generate_observations(pipe, seed=int(cfg.seed))
    np.testing.assert_array_equal(obs["depth"], obs["flux_true"])

    assert set(pipe.prior_types) == {"uniform"}, "the unit-cube points below assume uniform priors"
    lo, hi = pipe.param_prior_lo, pipe.param_prior_hi
    truth = (np.asarray(pipe.param_truth, np.float64) - lo) / (hi - lo)
    rows = [truth]
    for i in range(truth.size):
        row = truth.copy()
        row[i] = truth[i] + PERTURB_FRACTION if truth[i] + PERTURB_FRACTION < 1.0 else truth[i] - PERTURB_FRACTION
        rows.append(row)
    tally: dict = {}
    log_l = make_loglike(pipe, len(rows), tally)(np.stack(rows))
    assert np.all(np.isfinite(log_l)), f"a point did not certify or left the T-P window: {dict(tally)}"
    worse = {name: float(log_l[0] - lp) for name, lp in zip(pipe.names, log_l[1:], strict=True)}
    assert all(d > 0.0 for d in worse.values()), f"log L(truth) - log L(perturbed): {worse}"
