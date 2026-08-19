"""External-oracle regression test for the SMC/MALA core: the repo's
run_smc_loop and BlackJAX's adaptive tempered SMC must agree with the ANALYTIC
log-evidence of the same Gaussian-box target (and with each other) within
seed scatter.

Rationale (audit + collaborator review): the custom SMC core
is kept because BlackJAX cannot carry per-particle chemistry state, but its
generic machinery (tempering, resampling, evidence increments, MALA) should
never drift from an external oracle unnoticed. The audit measured the two
indistinguishable (24 seeds each, two-sample t p=0.275, repo bias
+0.009 +/- 0.015); this test pins a cheaper 6-seed version of that measurement.

Opt-in (RUN_BLACKJAX_ORACLE=1): needs the `blackjax` package (not part of the
runtime deps -- dev-only) and ~2-4 min of CPU. Analytic target: flat box prior
U(-8,8)^3 x independent Gaussian likelihood, for which

    lnZ = sum_i ln( sigma_i * sqrt(2*pi) ) - 3*ln(16)      (box fully contains
                                                            the likelihood mass)
"""
import math
import os

import numpy as np
import pytest

if os.environ.get("RUN_BLACKJAX_ORACLE", "") != "1":
    pytest.skip("external-oracle SMC test: set RUN_BLACKJAX_ORACLE=1 (needs "
                "blackjax installed; ~2-4 min)", allow_module_level=True)

blackjax = pytest.importorskip("blackjax")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework.config_schema import ParamSpec  # noqa: E402

M = np.array([1.0, -0.5, 0.3])
S = np.array([0.40, 0.60, 0.25])
LO, HI = -8.0, 8.0
SPECS = [ParamSpec(f"p{i}", f"p{i}", "uniform", LO, HI, float(M[i]), "chem")
         for i in range(3)]
LNZ_EXACT = sum(math.log(s * math.sqrt(2 * math.pi)) for s in S) \
    - 3 * math.log(HI - LO)
N_PART, N_MCMC, ESS_FRAC, N_SEEDS = 512, 8, 0.6, 6


def _loglik_theta(th):
    return -0.5 * jnp.sum(((th - jnp.asarray(M)) / jnp.asarray(S)) ** 2)


def _repo_lnz(seed):
    theta_from_u, log_prior_u, sample_prior_u = P.make_uspace(SPECS, jnp.float64)
    cfg = C.Config(smc_num_particles=N_PART, smc_num_mcmc_steps=N_MCMC,
                   smc_max_steps=60, smc_target_ess_frac=ESS_FRAC,
                   mcmc_stage_adapt=True, mala_step_size=0.2,
                   num_samples=N_PART, num_chains=1)
    pipe = P.Pipeline(
        cfg=cfg, dtype=jnp.float64, npdtype=np.float64, n_dim=3,
        theta_from_u=theta_from_u, log_prior_u=log_prior_u,
        sample_prior_u=sample_prior_u,
        log_likelihood_u=lambda u: _loglik_theta(theta_from_u(u)),
        loglik_fwd=lambda u: _loglik_theta(theta_from_u(u)),
    )
    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(seed), progress=False)
    assert res["reached_beta1"]
    return float(res["logZ"])


def _blackjax_lnz(seed, step_size=0.10):
    """BlackJAX adaptive tempered SMC on the IDENTICAL u-space target (the same
    bounded->unconstrained transform and logistic prior the repo samples), so
    the two engines integrate the same function. u-space keeps the prior smooth
    everywhere -- a hard -inf box boundary in theta space would break BlackJAX's
    MALA gradient. Invocation matches the audit's measured-working call
    (verify_smc_evidence.py; blackjax 1.x)."""
    import blackjax.smc.resampling as resampling

    theta_from_u, log_prior_u, sample_prior_u = P.make_uspace(SPECS, jnp.float64)
    alg = blackjax.adaptive_tempered_smc(
        logprior_fn=log_prior_u,
        loglikelihood_fn=lambda u: _loglik_theta(theta_from_u(u)),
        mcmc_step_fn=blackjax.mala.build_kernel(),
        mcmc_init_fn=blackjax.mala.init,
        mcmc_parameters=dict(step_size=jnp.full((1,), step_size)),
        resampling_fn=resampling.systematic,
        target_ess=ESS_FRAC,
        num_mcmc_steps=N_MCMC,
    )
    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    state = alg.init(sample_prior_u(sub, N_PART))
    lnz = 0.0
    for _ in range(200):
        if float(state.tempering_param) >= 1.0:
            return lnz
        key, sub = jax.random.split(key)
        state, info = alg.step(sub, state)
        lnz += float(info.log_likelihood_increment)
    raise RuntimeError("BlackJAX tempering did not reach lambda=1 in 200 stages")


def test_repo_and_blackjax_agree_with_analytic_lnz():
    repo = np.array([_repo_lnz(1000 + s) for s in range(N_SEEDS)])
    bj = np.array([_blackjax_lnz(2000 + s) for s in range(N_SEEDS)])

    # each estimator must be consistent with the analytic value (5-sigma of its
    # own seed scatter, floored at 0.05 log-units against tiny-variance flukes)
    for name, x in (("repo", repo), ("blackjax", bj)):
        sem = max(float(x.std(ddof=1)) / math.sqrt(N_SEEDS), 0.01)
        bias = float(x.mean()) - LNZ_EXACT
        assert abs(bias) < max(5.0 * sem, 0.05), (
            f"{name} lnZ biased: mean={x.mean():.4f} vs exact {LNZ_EXACT:.4f} "
            f"(bias {bias:+.4f}, SEM {sem:.4f})")

    # and with each other (Welch two-sample; generous 5-sigma gate -- this is a
    # drift alarm, not a precision measurement)
    se = math.sqrt(repo.var(ddof=1) / N_SEEDS + bj.var(ddof=1) / N_SEEDS)
    diff = float(repo.mean() - bj.mean())
    assert abs(diff) < max(5.0 * se, 0.05), (
        f"repo vs BlackJAX lnZ diverged: {repo.mean():.4f} vs {bj.mean():.4f} "
        f"(diff {diff:+.4f}, SE {se:.4f})")