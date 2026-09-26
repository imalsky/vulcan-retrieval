"""Validate the self-contained adaptive-tempered SMC + preconditioned-MALA core on an
analytic Gaussian posterior (flat box prior x independent Gaussian likelihood), where
the posterior is known exactly. No VULCAN/ExoJax -- pipeline's forward import is lazy.
"""
import math
from dataclasses import replace

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from conftest import stub_pipeline  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework.config_schema import ParamSpec  # noqa: E402

M = np.array([1.0, -0.5, 0.3])
S = np.array([0.40, 0.60, 0.25])
SPECS = [ParamSpec(f"p{i}", f"p{i}", "uniform", -8.0, 8.0, float(M[i]), "chem")
         for i in range(3)]


def _dying_make_mutation(_pipe, n_mcmc):
    """A mutation kernel that dies before its first sweep (a stage-0 death)."""
    def mutate(*a, **k):
        raise RuntimeError("simulated stage-0 death")
    return mutate


def _stub_pipe(cfg):
    m = jnp.asarray(M)
    s = jnp.asarray(S)
    return stub_pipeline(cfg, SPECS, lambda th: -0.5 * jnp.sum(((th - m) / s) ** 2))


def test_smc_recovers_gaussian_posterior(tmp_path):
    cfg = C.Config(
        smc_num_particles=256, smc_num_mcmc_steps=10, smc_max_steps=40,
        smc_target_ess_frac=0.6, num_samples=256, num_chains=2,
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
    assert np.all(np.diff(b) > 0) and abs(b[-1] - 1.0) < P._BETA_DONE_TOL


@pytest.mark.parametrize("kernel, n_sweeps", [("mala", 8), ("rwm", 24)])
def test_full_covariance_preconditioner_on_a_correlated_posterior(kernel, n_sweeps):
    """Both mutation kernels share the cloud's full Cholesky factor, so the MALA MH
    correction whitens with L^-1 rather than dividing by a per-dim width and the
    rwm proposal draws from 2*step*C. A wrong whitening (or a preconditioner that
    is not shared) biases the posterior SHAPE and the evidence, neither of which
    the uncorrelated test above can see. The gates are identical across kernels;
    only the sweep count differs, because a random walk needs more sweeps for the
    same mixing. The gates sit ~3-5 sigma of the measured single-seed scatter
    (MALA over 24 seeds; rwm at 24 sweeps, the smallest count with margin)."""
    sd = np.array([0.40, 0.60, 0.25])
    corr = np.array([[1.0, 0.95, 0.30], [0.95, 1.0, 0.20], [0.30, 0.20, 1.0]])
    sig = corr * np.outer(sd, sd)
    lo, hi = SPECS[0].lo, SPECS[0].hi
    lnz_exact = (0.5 * 3 * math.log(2 * math.pi)
                 + 0.5 * float(np.linalg.slogdet(sig)[1]) - 3 * math.log(hi - lo))
    sigi, mu = jnp.asarray(np.linalg.inv(sig)), jnp.asarray(M)

    def loglik(th):
        d = th - mu
        return -0.5 * (d @ sigi @ d)

    cfg = C.Config(smc_num_particles=384, smc_num_mcmc_steps=n_sweeps, smc_max_steps=60,
                   smc_target_ess_frac=0.6, num_samples=384, num_chains=1,
                   smc_mcmc_kernel=kernel)
    pipe = stub_pipeline(cfg, SPECS, loglik)
    res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(100), progress=False)
    assert res["reached_beta1"]

    th = res["theta_draws"].reshape(-1, 3)
    assert np.all(np.abs(th.mean(axis=0) - M) < 0.25 * sd)
    got = np.corrcoef(th, rowvar=False)
    assert abs(got[0, 1] - 0.95) < 0.03, got
    assert abs(got[0, 2] - 0.30) < 0.15 and abs(got[1, 2] - 0.20) < 0.20, got
    assert abs(res["logZ"] - lnz_exact) < 0.4, (res["logZ"], lnz_exact)
    # a correlated target is exactly where a diagonal proposal degenerates
    assert res["unique_particles"][-1] == cfg.smc_num_particles


def test_proposal_scale_reduces_to_the_diagonal_at_full_shrinkage():
    """shrink=1 must reproduce the per-dimension (diagonal) preconditioner."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)) @ np.array([[1.0, 0.9, 0.0, 0.0],
                                              [0.0, 0.5, 0.0, 0.0],
                                              [0.0, 0.0, 2.0, 0.3],
                                              [0.0, 0.0, 0.0, 0.1]])
    got = P._proposal_scale(x, cap=C.SCALE_CLIP, shrink=1.0)
    assert np.allclose(got, np.diag(np.clip(x.std(axis=0), C.SCALE_FLOOR, C.SCALE_CLIP)))


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


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    """A killed-and-resumed ladder must be bit-identical to an uninterrupted one:
    per-stage randomness is fold_in(seed_key, absolute stage index), so a resumed
    job never replays a killed job's resample offsets or MALA noise. The
    statistical claims are covered by test_smc_recovers_gaussian_posterior and
    test_smc_blackjax_oracle.py."""
    cfg = C.Config(smc_num_particles=128, smc_num_mcmc_steps=4, smc_max_steps=40,
                   smc_target_ess_frac=0.6, num_samples=128, num_chains=1)
    key = jax.random.PRNGKey(11)              # the SAME seed on both legs
    full = P.run_smc_loop(_stub_pipe(cfg), key=key, progress=False,
                          checkpoint_path=tmp_path / "a.npz")
    assert full["reached_beta1"]

    ck = tmp_path / "b.npz"
    part = P.run_smc_loop(_stub_pipe(cfg), key=key, progress=False,
                          checkpoint_path=ck, walltime_seconds=1e-9)
    assert not part["reached_beta1"] and len(part["betas"]) == 2
    res = P.run_smc_loop(_stub_pipe(cfg), key=key, progress=False,
                         checkpoint_path=ck, resume_from=ck)
    assert res["reached_beta1"]
    assert list(res["betas"]) == list(full["betas"])
    assert res["logZ"] == full["logZ"]
    assert np.array_equal(res["U"], full["U"])
    assert np.array_equal(res["theta_draws"], full["theta_draws"])


def _costed_stub_pipe(cfg, position_dependent):
    """The stub with a per-proposal cost (accept count) that varies with theta,
    so the slowest-first queue order is not the identity. With
    ``position_dependent`` the likelihood also moves by 1e-9 per batch position,
    the stand-in for the real queue, where the tick a proposal enters its lane
    moves its column at the convergence scale."""
    pipe = _stub_pipe(cfg)
    evg, el = P._get_batch_evals(pipe)[2:]

    def cost(U):
        return (jnp.abs(U[:, 0]) * 1000.0).astype(jnp.int32)

    def pos(U):
        return 1e-9 * jnp.arange(U.shape[0]) if position_dependent else 0.0

    def vg(U, Y, refs):
        L, G, Y2, r2, nb, st = evg(U, Y, refs)
        return L + pos(U), G, Y2, r2, nb, st._replace(acc=cost(U))

    def lik(U, Y, refs):
        L, Y2, r2, st = el(U, Y, refs)
        return L + pos(U), Y2, r2, st._replace(acc=cost(U))

    pipe._stub_evals = (vg, lik)
    return pipe


def _ladder(pipe, **kw):
    return P.run_smc_loop(pipe, key=jax.random.PRNGKey(11), progress=False, **kw)


def test_slowest_first_order_leaves_the_kernel_unchanged():
    """More particles than lanes sends each sweep's proposals to the queue
    slowest first and restores particle order after. On evaluators that do not
    depend on batch position the run is bitwise the unordered one: the order
    moves only a proposal's queue entry, never which particle gets which
    result."""
    base = dict(smc_num_particles=128, smc_num_mcmc_steps=4, smc_max_steps=40,
                smc_target_ess_frac=0.6, num_samples=128, num_chains=1)
    runs = [_ladder(_costed_stub_pipe(C.Config(cold_lanes=k, **base), False))
            for k in (0, 40)]
    assert runs[0]["reached_beta1"]
    assert runs[0]["logZ"] == runs[1]["logZ"]
    assert np.array_equal(runs[0]["U"], runs[1]["U"])


def test_slowest_first_order_resumes_bit_identically(tmp_path):
    """The order comes from each particle's last proposal count, which the
    checkpoint carries, so a resumed ordered run reproduces the uninterrupted
    one even where batch position moves the likelihood (the real queue)."""
    cfg = C.Config(smc_num_particles=128, smc_num_mcmc_steps=4, smc_max_steps=40,
                   smc_target_ess_frac=0.6, num_samples=128, num_chains=1,
                   cold_lanes=40)
    full = _ladder(_costed_stub_pipe(cfg, True))
    unordered = _ladder(_costed_stub_pipe(replace(cfg, cold_lanes=0), True))
    assert unordered["logZ"] != full["logZ"], "the order was never applied"
    ck = tmp_path / "ck.npz"
    part = _ladder(_costed_stub_pipe(cfg, True), checkpoint_path=ck,
                   walltime_seconds=1e-9)
    assert not part["reached_beta1"]
    res = _ladder(_costed_stub_pipe(cfg, True), checkpoint_path=ck,
                  resume_from=ck)
    assert res["logZ"] == full["logZ"]
    assert np.array_equal(res["U"], full["U"])


@pytest.mark.parametrize("field, bad, match", [
    ("chem_mode", "warm", "chem_mode"),
    ("target_digest", "0" * 64, "target digest"),
])
def test_resume_refuses_a_checkpoint_from_a_different_target(tmp_path, field, bad, match):
    """A checkpoint may only resume against the target it was written for.

    Adopting betas/logZ/loglik/gradients across a target change splices two
    densities into one evidence integral, and the certificate cannot detect it --
    it reads the last job only.
    """
    cfg = C.Config(smc_num_particles=64, smc_num_mcmc_steps=2, smc_max_steps=20,
                   smc_target_ess_frac=0.6, num_samples=64, num_chains=1)
    key = jax.random.PRNGKey(3)
    ck = tmp_path / "ck.npz"

    def pipe():                      # the real path sets this in set_observations
        p = _stub_pipe(cfg)
        p.target_digest = "d" * 64
        return p

    P.run_smc_loop(pipe(), key=key, progress=False,
                   checkpoint_path=ck, walltime_seconds=1e-9)
    # unchanged target: the same checkpoint resumes
    P.run_smc_loop(pipe(), key=key, progress=False,
                   checkpoint_path=tmp_path / "ok.npz", resume_from=ck)

    d = {k: v for k, v in np.load(ck, allow_pickle=True).items()}
    d[field] = np.asarray(bad)
    np.savez(ck, **d)

    with pytest.raises(ValueError, match=match):
        P.run_smc_loop(pipe(), key=key, progress=False,
                       checkpoint_path=tmp_path / "out.npz", resume_from=ck)


def test_resume_refuses_a_checkpoint_with_no_target_digest(tmp_path):
    """A checkpoint without a target digest is refused."""
    cfg = C.Config(smc_num_particles=64, smc_num_mcmc_steps=2, smc_max_steps=20,
                   smc_target_ess_frac=0.6, num_samples=64, num_chains=1)
    key = jax.random.PRNGKey(3)
    ck = tmp_path / "ck.npz"
    P.run_smc_loop(_stub_pipe(cfg), key=key, progress=False,
                   checkpoint_path=ck, walltime_seconds=1e-9)
    d = {k: v for k, v in np.load(ck, allow_pickle=True).items()}
    del d["target_digest"]
    np.savez(ck, **d)

    pipe = _stub_pipe(cfg)
    pipe.target_digest = "d" * 64
    with pytest.raises(ValueError, match="absent"):
        P.run_smc_loop(pipe, key=key, progress=False,
                       checkpoint_path=tmp_path / "out.npz", resume_from=ck)


def test_init_checkpoint_recovers_stage0_death(tmp_path, monkeypatch):
    """The init-level checkpoint (written right after _init_state, last_step=-1)
    must survive a stage-0 death and let RESUME skip the init entirely (with
    only per-stage checkpoints a bad-gradient raise at stage 0 throws away an
    hours-scale init)."""
    cfg = C.Config(smc_num_particles=64, smc_num_mcmc_steps=4, smc_max_steps=40,
                   smc_target_ess_frac=0.6, num_samples=64, num_chains=1)
    ck = tmp_path / "ck.npz"

    # run 1: the mutation kernel dies at stage 0 (simulating the bad-grad raise)
    real_make_mutation = P._make_mutation
    monkeypatch.setattr(P, "_make_mutation", _dying_make_mutation)
    with pytest.raises(RuntimeError, match="simulated stage-0"):
        P.run_smc_loop(_stub_pipe(cfg), key=jax.random.PRNGKey(3), progress=False,
                       checkpoint_path=ck)
    assert ck.exists()
    d = np.load(ck)
    assert int(d["last_step"]) == -1
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
    assert np.all(np.isfinite(res["theta_draws"]))


def _init_ck_then_poison(cfg, tmp_path, monkeypatch, seed=5):
    """Produce an init-level checkpoint (mutation dies immediately), then return a
    fresh pipe whose gradient evaluator poisons particle 0 with a NaN gradient on
    every eval -- exercised on the MUTATION path only via resume (the init has its
    own badgrad handling). The stub mirrors the real evaluators' contract: the
    non-finite gradient entries are ZEROED and the particle is flagged in
    stats.bad_grad (pipeline._rt_val_grad zeroes; the flag drives the zero-drift
    handling + forensics)."""
    ck = tmp_path / "ck.npz"
    real_make_mutation = P._make_mutation
    monkeypatch.setattr(P, "_make_mutation", _dying_make_mutation)
    with pytest.raises(RuntimeError, match="simulated stage-0"):
        P.run_smc_loop(_stub_pipe(cfg), key=jax.random.PRNGKey(seed), progress=False,
                       checkpoint_path=ck)
    monkeypatch.setattr(P, "_make_mutation", real_make_mutation)
    assert ck.exists() and int(np.load(ck)["last_step"]) == -1

    pipe = _stub_pipe(cfg)
    evg, el, _, _ = P._get_batch_evals(pipe)

    def bad_evg(U, Y, refs):
        L, G, Y_, refs_, n_bad, stats = evg(U, Y, refs)
        G = G.at[0, 0].set(jnp.nan)
        bad = jnp.isfinite(L) & ~jnp.all(jnp.isfinite(G), axis=1)
        # mirror the real evaluators: flag, then zero the non-finite entries
        # (the zeroed drift is what the MH correction sees on both sides)
        G = jnp.where(jnp.isfinite(G), G, 0.0)
        stats = stats._replace(bad_grad=bad)
        return L, G, Y_, refs_, jnp.sum(bad.astype(jnp.int32)), stats

    pipe._stub_evals = (bad_evg, el)
    return pipe, ck


def test_tangent_blown_proposal_zero_drift_not_fatal(tmp_path, monkeypatch):
    """A finite-likelihood/non-finite-tangent proposal WITHIN the backstop is
    handled as a ZERO-DRIFT MALA move (zeroed gradient entries used consistently
    in both proposal densities; the certified likelihood decides acceptance),
    its forensics are dumped, and the RUN COMPLETES. The class is
    theta-DEPENDENT (dense in the high-Z/low-C-O corner the posterior favors),
    so MH-rejecting it with a floored L suppresses the posterior bulk."""
    cfg = C.Config(smc_num_particles=32, smc_num_mcmc_steps=3, smc_max_steps=40,
                   smc_target_ess_frac=0.6, num_samples=32,
                   num_chains=1)   # default backstop 0.25 -> 8/sweep
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
    cfg = C.Config(smc_num_particles=32, smc_num_mcmc_steps=3, smc_max_steps=40,
                   smc_target_ess_frac=0.6, num_samples=32, num_chains=1,
                   smc_tangent_bad_max_frac=0.0)   # zero tolerance
    pipe, ck = _init_ck_then_poison(cfg, tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="non-finite-gradient") as ei:
        P.run_smc_loop(pipe, key=jax.random.PRNGKey(5), progress=False,
                       checkpoint_path=ck, resume_from=ck)
    assert "sweep 1/3" in str(ei.value)
    assert "indices [0]" in str(ei.value)
    assert sorted(tmp_path.glob("bad_grad_stage000_sweep1.npz"))


def test_calibrate_benchmarks_stage0_conditions(tmp_path):
    """calibrate() must benchmark the mutation at the ladder's own stage-0
    conditions (ESS-bisected first beta, stage-0 resample, cloud-width
    preconditioner, clamped step). A hard-coded (beta=0.5, step=C.MALA_STEP0,
    scale=1) proposal makes drift moves ~step*beta*|G| with prior-cloud
    gradients -- proposals the production ladder never launches -- and aborts
    the calibration on a spurious AD-pathology raise."""
    from retrieval_framework import run_smc
    cfg = C.Config(smc_num_particles=64, smc_num_mcmc_steps=3,
                   smc_target_ess_frac=0.6, num_samples=64, num_chains=1,
                   out_dir=tmp_path)
    pipe = _stub_pipe(cfg)
    pipe.n_chem_tp = 0
    pipe.chem_mode = "stub"
    proj = run_smc.calibrate(cfg, pipe, P, jax)

    assert 0.0 < proj["calibration_beta_stage0"] <= 1.0
    assert C.STEP_MIN <= proj["calibration_step"] <= C.STEP_MAX
    # preconditioner is the resampled cloud's per-dim width, never unit scale
    assert proj["calibration_scale_min"] >= C.SCALE_FLOOR
    assert proj["calibration_scale_max"] <= C.SCALE_CLIP
    assert (tmp_path / "timing.json").exists()
