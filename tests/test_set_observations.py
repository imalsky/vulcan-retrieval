"""Observation injection: ``pipeline.validate_observations`` and the production case's assembly.

The Gaussian likelihood divides by sigma and logs it, so a non-finite depth or a non-positive or non-finite sigma
raises at the boundary. Pure numpy except the production case's tests (slow), which build the real forward.
"""

import dataclasses

import numpy as np
import pytest

from conftest import RUN_DIR
from retrieval_framework import pipeline as P


N = 5
# Of sample_prior_u(PRNGKey(0), 16) on the production case. Stage 2 starts draw 7 with S4 far below its balance with S2,
# and its first step must be shorter than 1e-14 s.
CERTIFYING_DRAWS, FROZEN_DRAW = [3, 7], 7
GOOD_DEPTH = np.linspace(0.01, 0.02, N)
GOOD_SIGMA = np.full(N, 1e-4)


def test_valid_vector_is_preserved_exactly():
    d, s = P.validate_observations(GOOD_DEPTH, GOOD_SIGMA, N)
    assert np.array_equal(d, GOOD_DEPTH)
    assert np.array_equal(s, GOOD_SIGMA)
    assert d.shape == (N,)
    assert s.shape == (N,)
    # tiny but positive-finite sigmas are legal
    _, out = P.validate_observations(GOOD_DEPTH, np.full(N, 1e-30), N)
    assert np.all(out > 0.0)


def test_invalid_observations_raise():
    with pytest.raises(ValueError, match="length must be n_bin"):
        P.validate_observations(GOOD_DEPTH[:-1], GOOD_SIGMA[:-1], N)
    for bad in (np.nan, np.inf, -np.inf):
        d = GOOD_DEPTH.copy()
        d[2] = bad
        with pytest.raises(ValueError, match="depths must all be finite"):
            P.validate_observations(d, GOOD_SIGMA, N)
    for bad in (0.0, -1e-4, np.nan, np.inf, -np.inf):
        s = GOOD_SIGMA.copy()
        s[1] = bad
        with pytest.raises(ValueError, match="sigmas must all be finite and strictly"):
            P.validate_observations(GOOD_DEPTH, s, N)


# --- the production case -------------------------------------------------------


@pytest.mark.slow
def test_production_case_assembles_and_evaluates_one_finite_likelihood(monkeypatch, tmp_path):
    """Assemble the shipped case (real products, binning and offset operators)
    and evaluate one finite likelihood, so a product the binning operator
    refuses fails here rather than after a submit.
    """
    from retrieval_framework import run_setup as R

    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    cfg = R.load_spec(RUN_DIR / "wasp39_test1.yaml").cfg

    import jax

    pipe = P.build_pipeline(cfg)  # raises if any operator refuses
    P.load_real_into_pipe(pipe)
    obs = pipe.obs

    r_data = 0.5 * (obs["wl_lo"] + obs["wl_hi"]) / (obs["wl_hi"] - obs["wl_lo"])
    assert r_data.max() <= float(cfg.obs_max_bin_R), "an unresolved bin survived"
    assert pipe.n_bin == obs["wl"].size >= 4
    assert pipe.target_digest, "the run carries no target identity"

    # Each certifies in both stages. Some others of the 16 exit at the cap.
    U = pipe.sample_prior_u(jax.random.PRNGKey(0), 16)[np.asarray(CERTIFYING_DRAWS)]
    Y0, refs0 = P._blank_state(pipe, len(CERTIFYING_DRAWS))
    logl = np.asarray(jax.jit(pipe._make_batch_eval("cold"))(U, Y0, refs0, np.ones(len(CERTIFYING_DRAWS), bool))[0])
    assert np.all(np.isfinite(logl)), f"non-finite likelihood: {logl}"
    assert np.all(logl > P.REJECT_BELOW), f"rejected likelihood: {logl}"

    # The opacity leave-one-out screen selects its states from this same pipe;
    # prove the selection on the real object before a GPU run.
    monkeypatch.syspath_prepend(str(RUN_DIR.parent.parent / "validation"))
    from opacity_leave_one_out import states_for

    st = states_for(pipe, n_prior=2)
    # the nominal state (u = 0) only when its T-P is in the window (the WASP-39b case's is not)
    nominal = int(bool(pipe.tp_valid(pipe.theta_from_u(jax.numpy.zeros(pipe.n_dim)))))
    assert st.shape == (nominal + 2, pipe.n_dim)  # nominal + 2 prior draws
    box = np.asarray(pipe.prior_types) != "gaussian"  # gaussian: lo, hi = mean, sigma
    assert np.all(st[:, box] >= pipe.param_prior_lo[box])
    assert np.all(st[:, box] <= pipe.param_prior_hi[box])
    assert all(bool(pipe.tp_valid(t)) for t in st), "a screened state is off-window"

    post = tmp_path / "nautilus_posterior.npz"
    np.savez(post, param_names=np.asarray(pipe.names, dtype="<U64"), samples=np.broadcast_to(st[0], (8, pipe.n_dim)))
    assert states_for(pipe, 2, post).shape == (nominal + 4, pipe.n_dim)
    np.savez(
        post,
        param_names=np.asarray(list(pipe.names)[::-1], dtype="<U64"),
        samples=np.broadcast_to(st[0], (8, pipe.n_dim)),
    )
    with pytest.raises(RuntimeError, match="different target"):
        states_for(pipe, 2, post)


@pytest.mark.slow
def test_a_column_frozen_at_dt_min_aborts_the_run_rather_than_rejecting_the_draw(tmp_path):
    """At a floor of 1e-14 s draw 7 of the first 16 freezes on stage 2. The raise reaches the sampler through the job's
    own likelihood: no rejection is logged and no tally is kept for it.
    """
    if not RUN_DIR.exists():
        pytest.skip(f"run dir {RUN_DIR} not present")
    from vulcan_forward import vulcan_chem  # noqa: F401  # before vulcan3, which it configures
    import vulcan3
    import jax

    from retrieval_framework import run_setup as R
    from retrieval_framework.run_nautilus import make_loglike

    cfg = R.load_spec(RUN_DIR / "wasp39_test1.yaml").cfg
    cfg = dataclasses.replace(cfg, cfg_overrides={**cfg.cfg_overrides, "dt_min": 1e-14})
    pipe = P.build_pipeline(cfg)
    P.load_real_into_pipe(pipe)
    u = pipe.sample_prior_u(jax.random.PRNGKey(0), 16)[FROZEN_DRAW : FROZEN_DRAW + 1]
    tally = {}
    rejects = tmp_path / "rejected"
    like = make_loglike(pipe, 1, tally, rejects=rejects)
    with pytest.raises(vulcan3.DtMinFrozenError, match="froze at dt_min"):
        like(np.asarray(jax.nn.sigmoid(u), np.float64))
    assert tally == {}
    assert list(rejects.iterdir()) == []
