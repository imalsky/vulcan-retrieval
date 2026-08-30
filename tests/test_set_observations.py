"""Observation-injection validation.

pipeline.validate_observations is the API-boundary guard behind
pipe.set_observations: the Gaussian likelihood divides by sigma and logs it, so
a non-finite depth or a non-positive/non-finite sigma must RAISE here rather than
silently produce NaN/Inf likelihoods. Pure-numpy except for the opt-in
production preflight at the bottom, which builds the real forward.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from retrieval_framework import pipeline as P


N = 5
GOOD_DEPTH = np.linspace(0.01, 0.02, N)
GOOD_SIGMA = np.full(N, 1e-4)


def test_valid_vector_is_preserved_exactly():
    d, s = P.validate_observations(GOOD_DEPTH, GOOD_SIGMA, N, np.float64)
    assert np.array_equal(d, GOOD_DEPTH)
    assert np.array_equal(s, GOOD_SIGMA)
    assert d.shape == (N,) and s.shape == (N,)
    # tiny but positive-finite sigmas are legal
    _, out = P.validate_observations(GOOD_DEPTH, np.full(N, 1e-30), N,
                                     np.float64)
    assert np.all(out > 0.0)


def test_invalid_observations_raise():
    with pytest.raises(ValueError, match="length must be n_bin"):
        P.validate_observations(GOOD_DEPTH[:-1], GOOD_SIGMA[:-1], N,
                                np.float64)
    for bad in (np.nan, np.inf, -np.inf):
        d = GOOD_DEPTH.copy()
        d[2] = bad
        with pytest.raises(ValueError, match="depths must all be finite"):
            P.validate_observations(d, GOOD_SIGMA, N, np.float64)
    for bad in (0.0, -1e-4, np.nan, np.inf, -np.inf):
        s = GOOD_SIGMA.copy()
        s[1] = bad
        with pytest.raises(ValueError,
                           match="sigmas must all be finite and strictly"):
            P.validate_observations(GOOD_DEPTH, s, N, np.float64)


# --- production preflight (RC-01) --------------------------------------------

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "w39b_smc_retrieval"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_PRODUCTION_PREFLIGHT") != "1",
                    reason="opt-in: builds the exact production forward "
                           "(set RUN_PRODUCTION_PREFLIGHT=1)")
def test_production_case_assembles_and_evaluates_one_finite_likelihood(
        monkeypatch, tmp_path):
    """Assemble the EXACT shipped case and evaluate one likelihood.

    The shipped gpu case could not build: one NIRISS order-1 bin is a truncation
    remnant at R=464 against a model R=1000, and the binning operator is a pure
    cell average with no LSF, so observations refused it -- after the batch job
    had been submitted. No test caught it because the forward E2E never assembled
    the production observation operator. This is that test: real products, real
    binning and offset operators, one finite likelihood, before a submit.
    """
    from retrieval_framework import run_smc as R
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    # monkeypatch, not os.environ: test_warm_*.py select their preset with
    # setdefault, so a leaked "gpu" here would silently SKIP them
    monkeypatch.setenv("SMC_RETRIEVAL_PRESET", "gpu")
    cfg, preset = R.make_config(RUN_DIR)
    assert preset == "gpu"

    import jax

    pipe = P.build_pipeline(cfg)                 # raises if any operator refuses
    P.load_real_into_pipe(pipe)
    obs = pipe.obs

    r_data = 0.5 * (obs["wl_lo"] + obs["wl_hi"]) / (obs["wl_hi"] - obs["wl_lo"])
    assert r_data.max() <= float(cfg.obs_max_bin_R), "an unresolved bin survived"
    assert pipe.n_bin == obs["wl"].size >= 4
    assert pipe.target_digest, "the run carries no target identity"

    logl = float(pipe.log_likelihood_u(
        pipe.sample_prior_u(jax.random.PRNGKey(0), 1)[0]))
    assert np.isfinite(logl) and logl > -1e29, f"non-finite/rejected likelihood: {logl}"

    # The RC-02 opacity screen selects its states from this same pipe, and its
    # first run costs GH200 hours -- prove the selection here, on the real object.
    sys.path.insert(0, str(RUN_DIR.parent.parent / "validation"))
    from opacity_leave_one_out import states_for
    st = states_for(pipe, n_prior=2)
    assert st.shape == (3, pipe.n_dim)                     # nominal + 2 prior draws
    assert np.all(st >= pipe.param_prior_lo) and np.all(st <= pipe.param_prior_hi)
    assert all(bool(pipe.tp_valid(t)) for t in st), "a screened state is off-window"

    post = tmp_path / "posterior_samples.npz"
    np.savez(post, param_names=np.asarray(pipe.names, dtype="<U64"),
             samples=np.broadcast_to(st[0], (1, 8, pipe.n_dim)))
    assert states_for(pipe, 2, post).shape == (5, pipe.n_dim)
    np.savez(post, param_names=np.asarray(list(pipe.names)[::-1], dtype="<U64"),
             samples=np.broadcast_to(st[0], (1, 8, pipe.n_dim)))
    with pytest.raises(RuntimeError, match="different target"):
        states_for(pipe, 2, post)
