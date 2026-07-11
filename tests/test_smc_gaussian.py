"""Validate the self-contained adaptive-tempered SMC + preconditioned-MALA core on an
analytic Gaussian posterior (flat box prior x independent Gaussian likelihood), where
the posterior is known exactly. No VULCAN/ExoJax -- pipeline's forward import is lazy.
"""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework.config_schema import ParamSpec  # noqa: E402

M = np.array([1.0, -0.5, 0.3])
S = np.array([0.40, 0.60, 0.25])
SPECS = [ParamSpec(f"p{i}", f"p{i}", "uniform", -8.0, 8.0, float(M[i]), "chem")
         for i in range(3)]


def _stub_pipe(cfg):
    theta_from_u, log_prior_u, sample_prior_u = P.make_uspace(SPECS, jnp.float64)
    m = jnp.asarray(M)
    s = jnp.asarray(S)

    def log_likelihood_u(u):
        th = theta_from_u(u)
        return -0.5 * jnp.sum(((th - m) / s) ** 2)

    return P.Pipeline(
        cfg=cfg, dtype=jnp.float64, npdtype=np.float64, n_dim=3,
        theta_from_u=theta_from_u, log_prior_u=log_prior_u, sample_prior_u=sample_prior_u,
        log_likelihood_u=log_likelihood_u, loglik_fwd=log_likelihood_u,
    )


def test_smc_recovers_gaussian_posterior(tmp_path):
    cfg = C.Config(
        smc_num_particles=256, smc_num_mcmc_steps=10, smc_max_steps=40,
        smc_target_ess_frac=0.6, mcmc_stage_adapt=True, mala_step_size=0.2,
        num_samples=256, num_chains=2,
    )
    pipe = _stub_pipe(cfg)
    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(1), progress=False,
                         checkpoint_path=tmp_path / "ck.npz")
    assert res["reached_beta1"], f"did not reach beta=1: {res['final_beta']}"
    assert (tmp_path / "ck.npz").exists()

    th = res["theta_draws"].reshape(-1, 3)
    mean = th.mean(axis=0)
    std = th.std(axis=0)
    assert np.all(np.abs(mean - M) < 0.20 * S), (mean, M)
    assert np.all(std / S > 0.70) and np.all(std / S < 1.40), (std, S)

    # sane diagnostics: final acceptance not collapsed, particle diversity retained
    assert 0.05 < res["acceptance_rate"][-1] < 0.98
    assert res["unique_particles"][-1] > cfg.smc_num_particles // 4
    # betas strictly increasing to 1
    b = res["betas"]
    assert np.all(np.diff(b) > 0) and abs(b[-1] - 1.0) < 1e-8


def test_walltime_governor_stops_cleanly(tmp_path):
    cfg = C.Config(smc_num_particles=64, smc_num_mcmc_steps=4, smc_max_steps=40,
                         num_samples=32, num_chains=1)
    pipe = _stub_pipe(cfg)
    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(2), progress=False,
                         checkpoint_path=tmp_path / "ck.npz",
                         walltime_seconds=1e-9)     # exceeded after the first stage
    assert len(res["betas"]) == 2                   # exactly one stage ran
    assert not res["reached_beta1"]
    assert (tmp_path / "ck.npz").exists()           # partial output usable
    assert res["theta_draws"].shape == (1, 32, 3)


def test_resume_from_checkpoint_completes_the_ladder(tmp_path):
    """Kill a run early via the governor, resume from its checkpoint, and verify the
    resumed ladder reaches beta=1 with the correct posterior and a longer history."""
    cfg = C.Config(smc_num_particles=192, smc_num_mcmc_steps=8, smc_max_steps=40,
                         smc_target_ess_frac=0.6, num_samples=192, num_chains=1)
    pipe = _stub_pipe(cfg)
    ck = tmp_path / "ck.npz"
    part = P.run_smc_loop(pipe, key=jax.random.PRNGKey(3), progress=False,
                          checkpoint_path=ck, walltime_seconds=1e-9)
    assert not part["reached_beta1"]
    n_done = len(part["betas"]) - 1

    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(4), progress=False,
                         checkpoint_path=ck, resume_from=ck)
    assert res["reached_beta1"]
    assert len(res["betas"]) - 1 > n_done           # prior stages retained + new ones
    assert res["betas"][n_done] == part["betas"][n_done]
    th = res["theta_draws"].reshape(-1, 3)
    assert np.all(np.abs(th.mean(axis=0) - M) < 0.25 * S)
