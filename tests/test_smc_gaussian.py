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


def test_init_checkpoint_recovers_stage0_death(tmp_path, monkeypatch):
    """The init-level checkpoint (written right after _init_state, last_step=-1)
    must survive a stage-0 death and let RESUME skip the init entirely (NAS job
    65200: a bad-gradient raise at stage 0 threw away a 2.1 h init because the
    only checkpoint was per-stage)."""
    import pytest
    cfg = C.Config(smc_num_particles=64, smc_num_mcmc_steps=4, smc_max_steps=40,
                   smc_target_ess_frac=0.6, mcmc_stage_adapt=True, mala_step_size=0.2,
                   num_samples=64, num_chains=1)
    ck = tmp_path / "ck.npz"

    # run 1: the mutation kernel dies at stage 0 (simulating the bad-grad raise)
    real_make_mutation = P._make_mutation

    def dying_make_mutation(pipe_, n_mcmc):
        def mutate(*a, **k):
            raise RuntimeError("simulated stage-0 bad-gradient death")
        return mutate

    monkeypatch.setattr(P, "_make_mutation", dying_make_mutation)
    with pytest.raises(RuntimeError, match="simulated stage-0"):
        P.run_smc_loop(_stub_pipe(cfg), key=jax.random.PRNGKey(3), progress=False,
                       checkpoint_path=ck)
    assert ck.exists()
    d = np.load(ck)
    assert int(d["init_checkpoint"]) == 1 and int(d["last_step"]) == -1
    assert list(d["betas"]) == [0.0]
    assert "y_state" in d.files and "loglik" in d.files and "grad_u" in d.files

    # run 2: resume from the init checkpoint -- _init_state must NOT run again,
    # and the ladder completes to beta=1 with the recovered init_stats
    monkeypatch.setattr(P, "_make_mutation", real_make_mutation)

    def no_init(*a, **k):
        raise AssertionError("_init_state must not run on an init-checkpoint resume")

    monkeypatch.setattr(P, "_init_state", no_init)
    res = P.run_smc_loop(_stub_pipe(cfg), key=jax.random.PRNGKey(3), progress=False,
                         checkpoint_path=ck, resume_from=ck)
    assert res["reached_beta1"]
    assert res["init_stats"]                       # survived the round-trip
    th = res["theta_draws"].reshape(-1, 3)
    assert np.all(np.abs(th.mean(axis=0) - M) < 0.25 * S)


def _init_ck_then_poison(cfg, tmp_path, monkeypatch, seed=5):
    """Produce an init-level checkpoint (mutation dies immediately), then return a
    fresh pipe whose gradient evaluator poisons particle 0 with a NaN gradient on
    every eval -- exercised on the MUTATION path only via resume (the init has its
    own badgrad handling). The stub mirrors the real evaluators' contract: the
    non-finite gradient entries are ZEROED and the particle is flagged in
    stats.bad_grad (pipeline._rt_val_grad zeroes; the flag drives the zero-drift
    handling + forensics)."""
    import pytest
    ck = tmp_path / "ck.npz"

    def dying_make_mutation(pipe_, n_mcmc):
        def mutate(*a, **k):
            raise RuntimeError("die before any sweep")
        return mutate

    real_make_mutation = P._make_mutation
    monkeypatch.setattr(P, "_make_mutation", dying_make_mutation)
    with pytest.raises(RuntimeError, match="die before any sweep"):
        P.run_smc_loop(_stub_pipe(cfg), key=jax.random.PRNGKey(seed), progress=False,
                       checkpoint_path=ck)
    monkeypatch.setattr(P, "_make_mutation", real_make_mutation)
    assert ck.exists() and int(np.load(ck)["init_checkpoint"]) == 1

    pipe = _stub_pipe(cfg)
    evg, el, _, _ = P._get_batch_evals(pipe)

    def bad_evg(U, Y, refs):
        L, G, Y_, refs_, n_bad, DY, stats = evg(U, Y, refs)
        G = G.at[0, 0].set(jnp.nan)
        bad = jnp.isfinite(L) & ~jnp.all(jnp.isfinite(G), axis=1)
        # mirror the real evaluators: flag, then zero the non-finite entries
        # (the zeroed drift is what the MH correction sees on both sides)
        G = jnp.where(jnp.isfinite(G), G, 0.0)
        stats = stats._replace(bad_grad=bad)
        return L, G, Y_, refs_, jnp.sum(bad.astype(jnp.int32)), DY, stats

    pipe._stub_evals = (bad_evg, el)
    return pipe, ck


def test_tangent_blown_proposal_zero_drift_not_fatal(tmp_path, monkeypatch):
    """A finite-likelihood/non-finite-tangent proposal WITHIN the backstop is
    handled as a ZERO-DRIFT MALA move (zeroed gradient entries used consistently
    in both proposal densities; the certified likelihood decides acceptance),
    its forensics are dumped, and the RUN COMPLETES. The NAS 65815 lesson: the
    class is theta-DEPENDENT (dense in the high-Z/low-C-O corner the posterior
    favors), so the pre-65815 MH-reject-with-floored-L handling was a
    theta-correlated suppression of the posterior bulk AND its 5% per-sweep
    abort tripped with near-certainty over a full ladder."""
    cfg = C.Config(smc_num_particles=32, smc_num_mcmc_steps=3, smc_max_steps=40,
                   smc_target_ess_frac=0.6, mcmc_stage_adapt=True, mala_step_size=0.2,
                   num_samples=32, num_chains=1)   # default backstop 0.25 -> 8/sweep
    pipe, ck = _init_ck_then_poison(cfg, tmp_path, monkeypatch)
    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(5), progress=False,
                         checkpoint_path=ck, resume_from=ck)
    assert res["reached_beta1"]
    assert int(np.sum(res["tangent_rejected"])) > 0     # class visible per stage
    dumps = sorted(tmp_path.glob("bad_grad_stage*_sweep*.npz"))
    assert dumps, "forensics npz was not written next to the checkpoint"
    d = np.load(dumps[0])
    for k in ("bad_grad", "chem_tan_bad", "acc", "longdy", "conv_ok",
              "theta_proposal", "loglik_proposal"):
        assert k in d.files
    assert np.flatnonzero(d["bad_grad"]).tolist() == [0]
    # the carried state stays finite whether or not badgrad proposals were
    # accepted: the eval zeroes the non-finite gradient entries, so nothing
    # non-finite can enter U/G/L through the zero-drift acceptance path
    assert np.all(np.isfinite(np.asarray(res["U"])))
    assert np.all(np.isfinite(np.asarray(res["theta_draws"])))


def test_tangent_blown_over_threshold_raises(tmp_path, monkeypatch):
    """Above the per-sweep backstop the loud raise is intact: a systematic AD
    breakage must never be absorbed as a zero-drift class."""
    import pytest
    cfg = C.Config(smc_num_particles=32, smc_num_mcmc_steps=3, smc_max_steps=40,
                   smc_target_ess_frac=0.6, mcmc_stage_adapt=True, mala_step_size=0.2,
                   num_samples=32, num_chains=1,
                   smc_tangent_bad_max_frac=0.0)   # zero tolerance
    pipe, ck = _init_ck_then_poison(cfg, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="non-finite-gradient") as ei:
        P.run_smc_loop(pipe, key=jax.random.PRNGKey(5), progress=False,
                       checkpoint_path=ck, resume_from=ck)
    assert "sweep 1/3" in str(ei.value)
    assert "indices [0]" in str(ei.value)
    assert sorted(tmp_path.glob("bad_grad_stage000_sweep1.npz"))


def test_calibrate_benchmarks_stage0_conditions(tmp_path):
    """Regression for NAS job 64961: calibrate() must benchmark the mutation at the
    ladder's own stage-0 conditions (ESS-bisected first beta, stage-0 resample,
    cloud-width preconditioner, clamped step). The old hard-coded
    (beta=0.5, step=mala_step_size, scale=1) proposal made drift moves
    ~step*beta*|G| with prior-cloud gradients -- proposals the production ladder
    never launches -- and aborted the calibration on a spurious AD-pathology raise."""
    from retrieval_framework import run_smc
    cfg = C.Config(smc_num_particles=64, smc_num_mcmc_steps=3,
                   smc_target_ess_frac=0.6, mcmc_stage_adapt=True,
                   num_samples=64, num_chains=1, out_dir=tmp_path)
    pipe = _stub_pipe(cfg)
    pipe.n_chem_tp = 0
    pipe.gradient_mode = "stub"
    pipe.chem_mode = "stub"
    proj = run_smc.calibrate(cfg, pipe, P, jax)

    assert 0.0 < proj["calibration_beta_stage0"] <= 1.0
    assert cfg.mcmc_step_size_min <= proj["calibration_step"] <= cfg.mcmc_step_size_max
    # preconditioner is the resampled cloud's per-dim width, never unit scale
    assert proj["calibration_scale_min"] >= 1e-3
    assert proj["calibration_scale_max"] <= cfg.mcmc_scale_clip
    assert (tmp_path / "timing.json").exists()
