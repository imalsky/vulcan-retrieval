"""run_nautilus.main on a stub pipeline and a stub sampler, no chemistry: a finished run's files, the identity a resume
is checked against, and the order of that check against everything the run writes.
"""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_run_nautilus import D, _FrozenStats, _stub_pipe

from retrieval_framework import config_schema as C
from retrieval_framework import constants
from retrieval_framework import hardware
from retrieval_framework import nautilus_report as NR
from retrieval_framework import pipeline as P
from retrieval_framework import run_nautilus as RN
from retrieval_framework import run_setup as R
from retrieval_framework import warm_checks
from retrieval_framework.chunks import cold_log_l
from conftest import case_config

DIGEST = "d" * 64
N_BATCH, N_LIVE = 8, 16
IDENTITY = (
    f"{DIGEST}|warm|n_batch={N_BATCH}|n_live={N_LIVE}|n_update={N_LIVE // 2}|f_live=0.03"
    "|enlarge_per_dim=1.3|networks=4xrelu|stock"
)
LOG_Z, N_EFF = -4.5, 321.0


class _Sampler:
    """Stands for nautilus: ``run`` evaluates seeded points through the run's own likelihood, and hands them back."""

    def __init__(self, like, done=True, explored=True, shell_n=(24,)):
        self.like, self.done, self.explored, self.shell_n = like, done, explored, np.asarray(shell_n)
        self.n_like, self.log_z, self.n_eff = 0, LOG_Z, N_EFF

    def run(self, **kw):
        self.run_kw = kw
        self.z = np.random.default_rng(5).uniform(0.05, 0.95, size=(24, D))
        self.log_l = self.like(self.z)
        self.n_like += len(self.z)
        return self.done

    def posterior(self, equal_weight=False):
        if equal_weight:
            return self.z[:4], None, self.log_l[:4]
        return self.z, np.linspace(-3.0, 0.0, len(self.z)), self.log_l


@pytest.fixture
def job(tmp_path, monkeypatch):
    """A spec on disk and main's collaborators replaced, recording the order of what the run does."""
    monkeypatch.chdir(R.__file__.rsplit("/src/", 1)[0])
    spec = tmp_path / "stub_main.yaml"
    spec.write_text(
        f"base: {R.__file__.rsplit('/src/', 1)[0]}/runs/w39b_smc_retrieval/smoke.yaml\n"
        f"sampler: {{n_live: {N_LIVE}, n_batch: {N_BATCH}}}\n"
        "config: {cold_lanes: 8}\n"
    )
    pipe = _stub_pipe()
    pipe.names = ["p0", "p1", "lnKzz"]
    pipe.nuisance = None
    pipe.target_digest = DIGEST
    pipe.fwd.chem_rtol = 0.05
    events = []
    sampler = {}

    def make_sampler(like, n_dim, settings, filepath=None, resume=False):
        events.append("sampler")
        sampler["args"] = (n_dim, settings.n_live, settings.n_batch, settings.seed, filepath, resume)
        sampler["s"] = _Sampler(like, sampler.get("done", True), **sampler.get("state", {}))
        return sampler["s"]

    monkeypatch.setattr(RN, "export_runtime", lambda _rt: None)
    monkeypatch.setattr(RN, "setup_logging", lambda _path: None)
    monkeypatch.setattr(C, "export_network", lambda _cfg: None)
    monkeypatch.setattr(hardware, "hardware_profile", lambda: "stub")
    monkeypatch.setattr(P, "build_pipeline", lambda _cfg: pipe)
    monkeypatch.setattr(RN, "set_observations", lambda _cfg, _pipe, _P: {})
    monkeypatch.setattr(RN, "write_run_identity", lambda *_a: events.append("identity"))
    monkeypatch.setattr(RN, "make_sampler", make_sampler)
    monkeypatch.setattr(NR, "kzz_at_ref", lambda _name, _over, _p_bar=NR.KZZ_P_REF_BAR: 1.0e9)
    monkeypatch.setattr(NR, "write_certificate", lambda out: events.append("certificate") or out / "cert.json")
    out = tmp_path / "output" / "stub_main_nautilus"
    return SimpleNamespace(spec=str(spec), out=out, pipe=pipe, events=events, sampler=sampler)


def test_the_startup_kzz_decades_are_the_report_s(job, monkeypatch):
    """The launch guard's absolute log10 Kzz is the report's: the case's base at the report's level times e^lnKzz."""
    seen = {}
    real = RN.make_loglike

    def capture(*a, **kw):
        seen["startup_kzz"] = kw["startup_kzz"]
        return real(*a, **kw)

    monkeypatch.setattr(RN, "make_loglike", capture)
    monkeypatch.setattr(NR, "kzz_at_ref", lambda _name, _over, p_bar=NR.KZZ_P_REF_BAR: 1.0e9 * (p_bar / 0.1))
    job.pipe.names = ["p0", "lnKzz"]
    cfg = case_config(vulcan_cfg_name="W39b")
    RN._saved_loglike(job.pipe, cfg, job.out, N_BATCH, warm=False, resume=False, tally={})
    th = np.array([[0.0, np.log(10.0)]])
    # 1e9 at 0.1 bar is 1e7 at 1 mbar, x10 by lnKzz
    np.testing.assert_allclose(seen["startup_kzz"](th), [8.0], rtol=1e-12)


@pytest.mark.parametrize("solver", ["fast", "ffi"])
def test_the_run_builds_its_forward_at_the_spec_s_lanes_under_every_solver(job, monkeypatch, solver):
    """The lane count that reaches the forward model at execution is the spec's cold_lanes, under ffi too (the forward's
    dispatch width is cold_lanes: tests/test_retrieval_forward_stub.py).
    """
    seen = []
    monkeypatch.setattr(P, "build_pipeline", lambda cfg: seen.append(cfg.cold_lanes) or job.pipe)
    spec = Path(job.spec).with_name(f"stub_{solver}.yaml")
    runtime = f"runtime: {{solver: {solver}}}\nconfig: {{cold_lanes: 8}}"
    spec.write_text(Path(job.spec).read_text().replace("config: {cold_lanes: 8}", runtime))
    RN.main([str(spec)])
    assert seen == [8]


def test_a_finished_run_writes_its_posterior_summary_and_identity(job):
    RN.main([job.spec])
    s = job.sampler["s"]
    assert job.events == ["identity", "sampler", "certificate"]
    assert job.sampler["args"][:3] == (D, N_LIVE, N_BATCH)
    assert (job.out / "target_digest.txt").read_text() == IDENTITY + "\n"
    assert (job.out / "spec.yaml").exists()
    assert (job.out / "spec_base.yaml").exists()
    assert s.run_kw["f_live"] == constants.F_LIVE
    assert s.run_kw["discard_exploration"] is constants.DISCARD_EXPLORATION

    post = np.load(job.out / "nautilus_posterior.npz")
    assert [str(n) for n in post["param_names"]] == ["p0", "p1", "lnKzz"]
    assert str(post["target_digest"]) == IDENTITY
    u = np.log(s.z) - np.log1p(-s.z)
    np.testing.assert_array_equal(post["u"], u)
    # the stub's prior is uniform on [0, 1]: theta is z back, through the pipeline's own map
    np.testing.assert_allclose(post["theta"], s.z, rtol=1e-12, atol=0)
    np.testing.assert_allclose(post["samples"], s.z[:4], rtol=1e-12, atol=0)
    np.testing.assert_array_equal(post["log_w"], np.linspace(-3.0, 0.0, len(s.z)))
    np.testing.assert_array_equal(post["log_l"], s.log_l)
    assert float(post["logZ_box"]) == LOG_Z
    assert int(post["n_like"]) == len(s.z)
    assert "nuisance" not in post.files

    summary = json.loads((job.out / "nautilus_summary.json").read_text())
    tally = json.loads((job.out / RN.TALLY_FILE).read_text())
    assert tally["n_eval"] == len(s.z)
    assert summary["rejections_and_starts"] == tally
    assert (summary["logZ_box"], summary["n_eff"], summary["n_like"]) == (LOG_Z, N_EFF, len(s.z))
    assert (summary["n_live"], summary["n_batch"], summary["n_update"]) == (N_LIVE, N_BATCH, N_LIVE // 2)
    assert summary["chem_rtol"] == 0.05
    assert summary["starts"] == "anchored warm (growing)"
    assert summary["finished"] is True
    assert "APPROXIMATE" in summary["evidence_note"]


def _tilt_cold(pipe, tilt):
    """Make the stub's cold solve score each certified row ``tilt * p0`` higher than its warm solve does."""
    make = pipe._make_batch_eval

    def tilted(mode):
        ev = make(mode)
        if mode != "cold":
            return ev

        def cold(U, Y, refs, active):
            L, Ynew, rnew, st = ev(U, Y, refs, active)
            return L + tilt * jax.nn.sigmoid(U[:, 0]), Ynew, rnew, st

        return cold

    pipe._make_batch_eval = tilted


@pytest.mark.parametrize("tilt", [0.0, 20.0])
def test_a_finished_warm_run_spot_checks_its_rows_cold(job, monkeypatch, caplog, tilt):
    """At the end of a warm run, posterior rows drawn by weight and the warm rows rejected at the step cap are solved by
    the run's own cold evaluator: a cold likelihood equal to the warm one moves nothing, one that rises with p0 moves
    the posterior's quantiles past the limit, and that is a warning in the log, the file and the summary, never a stop.
    """
    _tilt_cold(job.pipe, tilt)
    stub = _Sampler.posterior

    def posterior(self, equal_weight=False):
        """As nautilus's: a rejected row has no weight."""
        z, log_w, log_l = stub(self, equal_weight)
        return z, None if log_w is None else np.where(np.isfinite(log_l), log_w, -np.inf), log_l

    monkeypatch.setattr(_Sampler, "posterior", posterior)
    with caplog.at_level("INFO", logger="retrieval"):
        RN.main([job.spec])
    assert job.events[-1] == "certificate"
    spot = json.loads((job.out / "spot_check.json").read_text())
    summary = json.loads((job.out / "nautilus_summary.json").read_text())
    assert summary["spot_check"] == spot
    assert summary["warm_start_guards"]["warm_in_window"] == summary["rejections_and_starts"]["warm_in_window"] > 0
    capped = 0
    for f in (job.out / "rejected").glob("rejected_*.npz"):
        with np.load(f) as d:
            capped += int(d["warm"]) * int((d["cls"] == 1).sum())
    assert capped > 0
    assert spot["status"] == "done"
    assert (spot["posterior_rows"], spot["rejected_rows"]) == (constants.SPOT_CHECK_POSTERIOR_ROWS, capped)
    assert spot["rejected_certified_cold"] == spot["rejected_above_posterior_1pct"] == 0  # the stub caps them cold too
    assert spot["posterior_certified_cold"] == spot["posterior_rows"]
    assert (spot["posterior_not_certified_cold"], spot["frozen_rows"]) == (0, 0)
    assert spot["estimate_seconds"] > 0
    if tilt == 0.0:
        assert (spot["median_abs_dlog_l"], spot["max_abs_dlog_l"], spot["worst_shift_sigma"]) == (0.0, 0.0, 0.0)
        assert spot["warnings"] == []
    else:
        assert spot["max_abs_dlog_l"] > 1.0
        assert abs(spot["worst_shift_sigma"]) > constants.SPOT_CHECK_SHIFT_SIGMA
        assert len(spot["warnings"]) == 1
        assert f"spot-check: {spot['warnings'][0]}" in caplog.text


@pytest.mark.parametrize("why", ["no time", "failure"])
def test_a_spot_check_with_no_time_or_a_failure_is_recorded_and_the_run_goes_on(job, monkeypatch, why):
    if why == "no time":
        monkeypatch.setattr(constants, "JOB_HEADROOM_S", 0)  # the governor is the whole budget, and it is spent
        Path(job.spec).write_text(
            Path(job.spec).read_text().replace("cold_lanes: 8", "cold_lanes: 8, walltime_seconds: 1.0e-9")
        )
    else:

        def broken(*_a):
            raise ValueError("a broken posterior file")

        monkeypatch.setattr(warm_checks, "_spot_check", broken)
    RN.main([job.spec])
    assert job.events[-1] == "certificate"
    spot = json.loads((job.out / "nautilus_summary.json").read_text())["spot_check"]
    assert spot == json.loads((job.out / "spot_check.json").read_text())
    if why == "no time":
        assert spot["status"] == "skipped"
        assert " s are left before half the job's headroom" in spot["reason"]
    else:
        assert spot == {"status": "failed", "error": "ValueError: a broken posterior file"}


# Posterior rows that certify on the stub's cold solve (T-P window, p1 >= 0.35, p0 <= 0.6), with theta = z.
SPOT_Z = np.array([[0.45, 0.5, 0.5], [0.5, 0.45, 0.4], [0.55, 0.55, 0.6], [0.5, 0.6, 0.5]])


def _spot_run(out, z, log_l, capped=None):
    """Write the files the spot-check reads: equal-weight posterior rows at ``z``, one cold batch, and ``capped`` as
    warm rows rejected at the step cap.
    """
    np.savez(
        out / "nautilus_posterior.npz",
        u=np.log(z) - np.log1p(-z),
        log_l=log_l,
        theta=z,
        log_w=np.zeros(len(z)),
        param_names=np.array(["p0", "p1", "p2"]),
    )
    (out / "batches.jsonl").write_text(json.dumps({"t": 1.0, "mode": "cold", "acc": []}) + "\n")
    if capped is not None:
        (out / "rejected").mkdir()
        np.savez(out / "rejected" / "rejected_00000.npz", z=capped, cls=np.ones(len(capped), np.int8), warm=np.int8(1))


def test_the_shift_compares_matched_rows_reweighted_and_counts_a_row_lost_cold(tmp_path):
    """Rows A and B certify warm and cold, C only warm. The cold log L is 50 above the run's on B alone, so reweighted,
    every quantile of p0 sits at B; unweighted, the 16th sits at A. The worst shift is then (p0_B - p0_A) / sd(p0) at
    the 16th, positive, whatever the draw; p1 differs only on C, so it moves only if C enters the comparison.
    """
    pipe = _stub_pipe()
    cold = jax.jit(pipe._make_batch_eval("cold"))
    z = np.array([[0.42, 0.5, 0.5], [0.55, 0.5, 0.5], [0.4, 0.3, 0.5]])  # A, B, and C, which stalls cold (p1 < 0.35)
    log_l, _ = cold_log_l(pipe, cold, N_BATCH, z)
    assert np.isfinite(log_l[:2]).all()
    assert np.isneginf(log_l[2])
    _spot_run(tmp_path, z, np.array([log_l[0], log_l[1] - 50.0, log_l[0]]))
    spot = warm_checks.spot_check(tmp_path, partial(cold_log_l, pipe, cold, N_BATCH), 0, N_BATCH, np.inf)
    assert spot["status"] == "done"
    assert spot["worst"] == "p0 q16"
    assert spot["worst_shift_sigma"] == pytest.approx((0.55 - 0.42) / np.std(z[:, 0]), rel=1e-12)
    assert spot["max_abs_dlog_l"] == pytest.approx(50.0, rel=1e-12)
    # The draw, as the spot-check makes it: seed + 3, three rows of equal weight.
    drawn = np.random.default_rng(0 + 3).choice(3, size=constants.SPOT_CHECK_POSTERIOR_ROWS, p=np.full(3, 1 / 3))
    lost = int((drawn == 2).sum())
    assert 0 < lost == spot["posterior_not_certified_cold"]
    assert spot["posterior_not_certified_cold_weight_frac"] == pytest.approx(lost / constants.SPOT_CHECK_POSTERIOR_ROWS)
    assert spot["frozen_rows"] == 0


def test_no_posterior_row_certified_cold_is_a_warning_with_no_shift(tmp_path):
    pipe = _stub_pipe()
    cold = jax.jit(pipe._make_batch_eval("cold"))
    z = np.array([[0.4, 0.3, 0.5], [0.5, 0.2, 0.4]])  # both stall cold
    _spot_run(tmp_path, z, np.zeros(len(z)))
    spot = warm_checks.spot_check(tmp_path, partial(cold_log_l, pipe, cold, N_BATCH), 0, N_BATCH, np.inf)
    n = constants.SPOT_CHECK_POSTERIOR_ROWS
    assert spot["status"] == "done"
    assert (spot["posterior_certified_cold"], spot["posterior_not_certified_cold"]) == (0, n)
    assert spot["posterior_not_certified_cold_weight_frac"] == 1.0
    assert (spot["worst"], spot["worst_shift_sigma"], spot["median_abs_dlog_l"]) == (None, None, None)
    assert spot["warnings"] == [f"none of the {n} posterior rows certified cold: no shift could be measured."]


@pytest.mark.parametrize("lost", [20, 30, 40])
def test_a_spot_check_on_too_few_matched_rows_warns(tmp_path, lost):
    """With more than SPOT_CHECK_LOST_WARN_FRAC of the posterior rows lost cold, the shift rests on few rows: a warning
    naming the share, though the shift itself is zero. 30 of 200 is at the limit.
    """
    _spot_run(tmp_path, SPOT_Z, np.zeros(len(SPOT_Z)))

    def cold(Z):  # the first ``lost`` posterior rows do not certify; the rest score the run's log L
        return np.where(np.arange(len(Z)) < lost, -np.inf, 0.0), np.zeros(len(Z), bool)

    spot = warm_checks.spot_check(tmp_path, cold, 0, N_BATCH, np.inf)
    n = constants.SPOT_CHECK_POSTERIOR_ROWS
    assert (spot["status"], spot["posterior_not_certified_cold"], spot["worst_shift_sigma"]) == ("done", lost, 0.0)
    msg = (
        f"the shift was measured on only {n - lost} of {n} posterior rows: ~{lost / n:.1%} of the posterior weight "
        "(estimated) did not certify cold (limit 15%)."
    )
    assert spot["warnings"] == ([] if lost <= 30 else [msg])


def _flag_frozen(fn, where):
    """Wrap a stub evaluator: rows where ``where(U)`` report a freeze at dt_min, and are otherwise as ``fn`` gives."""

    def ev(U, Y, refs, active):
        L, Ynew, rnew, st = fn(U, Y, refs, active)
        fz = jnp.asarray(active, bool) & where(U)
        reason = jnp.where(fz, constants.TERM_DT_MIN_FROZEN, 1)
        return L, Ynew, rnew, _FrozenStats(st.acc, st.conv_ok, reason, jnp.zeros_like(L))

    return ev


def test_a_row_frozen_in_the_spot_check_is_counted_and_the_rest_computed(tmp_path):
    """A column frozen at dt_min in the spot-check's cold solve is a row that does not certify, by its freeze alone
    (the stub certifies it otherwise), and is counted: the check still finishes, where a run's own chunk would stop.
    """
    pipe = _stub_pipe()
    plain = jax.jit(pipe._make_batch_eval("cold"))
    cold = jax.jit(_flag_frozen(plain, lambda U: jnp.abs(jax.nn.sigmoid(U[:, 1]) - 0.52) < 1e-9))
    # The last posterior row freezes, as does the rejected row; nine rows keep its share under the lost-row limit.
    z = np.vstack([SPOT_Z, SPOT_Z + 0.03, [[0.5, 0.52, 0.45]]])
    stuck = np.array([[0.5, 0.52, 0.5]])
    want, none = cold_log_l(pipe, plain, N_BATCH, np.vstack([z, stuck]))
    assert np.isfinite(want).all()
    assert not none.any()
    log_l, frozen = cold_log_l(pipe, cold, N_BATCH, np.vstack([z, stuck]))
    assert frozen.tolist() == [False] * (len(z) - 1) + [True, True]
    assert np.array_equal(log_l, np.where(frozen, -np.inf, want))
    _spot_run(tmp_path, z, want[: len(z)], capped=stuck)
    spot = warm_checks.spot_check(tmp_path, partial(cold_log_l, pipe, cold, N_BATCH), 0, N_BATCH, np.inf)
    assert spot["status"] == "done"
    assert (spot["rejected_rows"], spot["rejected_certified_cold"]) == (1, 0)
    lost = spot["posterior_not_certified_cold"]
    assert 0 < lost == spot["frozen_rows"] - 1
    assert spot["posterior_certified_cold"] == constants.SPOT_CHECK_POSTERIOR_ROWS - lost
    assert (spot["worst_shift_sigma"], spot["warnings"]) == (0.0, [])


def test_a_changed_target_is_refused_before_the_run_writes_anything(job):
    job.out.mkdir(parents=True)
    (job.out / "nautilus_checkpoint.hdf5").write_text("")
    (job.out / "target_digest.txt").write_text("another target\n")
    with pytest.raises(RuntimeError, match="resume refused"):
        RN.main([job.spec, "--resume"])
    assert job.events == []
    assert (job.out / "target_digest.txt").read_text() == "another target\n"
    assert not (job.out / "spec.yaml").exists()
    assert not (job.out / RN.TALLY_FILE).exists()


def test_a_resume_continues_the_tally_and_skips_the_startup_check(job, monkeypatch):
    job.out.mkdir(parents=True)
    (job.out / "nautilus_checkpoint.hdf5").write_text("")
    (job.out / "target_digest.txt").write_text(IDENTITY + "\n")
    (job.out / RN.TALLY_FILE).write_text(json.dumps({"n_eval": 100}))

    def no_check(*_a, **_k):
        raise AssertionError("the startup check is the first chunk of a fresh run only")

    monkeypatch.setattr(RN, "startup_check", no_check)
    job.sampler["done"] = False  # the wall budget ends it while exploring: a checkpoint, and no posterior
    job.sampler["state"] = {"explored": False}
    RN.main([job.spec, "--resume"])
    assert job.sampler["args"][5] is True
    assert json.loads((job.out / RN.TALLY_FILE).read_text())["n_eval"] == 124
    assert not (job.out / "nautilus_posterior.npz").exists()
    assert "certificate" not in job.events


def test_a_run_stopped_while_its_shells_refill_writes_no_posterior(job):
    """After exploration nautilus discards its points, then samples each shell again in turn: a posterior from before
    the last of them holds points is missing that shell.
    """
    job.sampler["done"] = False
    job.sampler["state"] = {"shell_n": (24, 0)}
    RN.main([job.spec])
    assert not (job.out / "nautilus_posterior.npz").exists()
    assert "certificate" not in job.events


def test_a_run_the_wall_budget_stops_after_exploration_writes_what_it_has(job):
    job.sampler["done"] = False
    RN.main([job.spec])
    assert (job.out / "nautilus_posterior.npz").exists()
    summary = json.loads((job.out / "nautilus_summary.json").read_text())
    assert summary["finished"] is False
    assert (summary["n_eff"], summary["n_eff_target"]) == (N_EFF, RN.nautilus_settings(RN.load_spec(job.spec)).n_eff)
    assert job.events[-1] == "certificate"


def test_a_fresh_run_over_a_previous_one_is_refused(job):
    job.out.mkdir(parents=True)
    (job.out / "nautilus_checkpoint.hdf5").write_text("")
    with pytest.raises(FileExistsError, match="--resume"):
        RN.main([job.spec])
    assert job.events == []


class _NuisStats(NamedTuple):
    acc: jnp.ndarray
    conv_ok: jnp.ndarray
    nuis: jnp.ndarray

    @property
    def reason(self):
        return jnp.zeros_like(self.acc)

    @property
    def last_delta(self):
        return jnp.zeros_like(self.acc, dtype=float)


def test_integrated_nuisances_reach_the_posterior_file_from_the_logged_statistics(job):
    """Each row's conditional draw comes from the statistics logged with its own log L; a rejected row has none."""
    make = job.pipe._make_batch_eval

    def make_with_stats(mode):
        ev = make(mode)

        def with_stats(U, Y, refs, active):
            L, Ynew, rnew, st = ev(U, Y, refs, active)
            return L, Ynew, rnew, _NuisStats(st.acc, st.conv_ok, jnp.stack([U[:, 0], U[:, 1]], axis=1))

        return with_stats

    job.pipe._make_batch_eval = make_with_stats
    job.pipe.obs_sigma = np.ones(2)
    job.pipe.nuisance = SimpleNamespace(
        groups=["g"],
        names=["offset_g"],
        draw=lambda st, _sigma, _rng: 2.0 * st[:, :1],
    )
    RN.main([job.spec])
    s = job.sampler["s"]
    post = np.load(job.out / "nautilus_posterior.npz")
    assert [str(n) for n in post["nuisance_names"]] == ["offset_g"]
    assert post["nuisance"].shape == (len(s.z), 1)
    alive = np.isfinite(s.log_l)
    assert alive.any()
    assert not alive.all()
    u0 = np.log(s.z[:, 0]) - np.log1p(-s.z[:, 0])
    np.testing.assert_allclose(post["nuisance"][alive, 0], 2.0 * u0[alive], rtol=1e-12, atol=0)
    assert np.all(np.isnan(post["nuisance"][~alive]))
    assert post["samples_nuisance"].shape == (4, 1)


@pytest.mark.parametrize("frozen_row", [0, 2])
def test_a_frozen_lane_aborts_the_run_naming_the_row_the_ratio_and_rtol(job, frozen_row):
    """The evaluator returns normally under jit; a row's reason must stop the run before anything is logged of it."""

    def make(_mode):
        def frozen(U, Y, refs, _active):
            n = U.shape[0]
            reason = jnp.zeros(n, int).at[frozen_row].set(constants.TERM_DT_MIN_FROZEN)
            delta = jnp.full(n, 0.01).at[frozen_row].set(0.0808)
            stats = _FrozenStats(jnp.full(n, 2), jnp.zeros(n, bool), reason, delta)
            return jnp.full(n, constants.REJECT_LOGL), Y, refs, stats

        return frozen

    job.pipe._make_batch_eval = make
    from vulcan3 import DtMinFrozenError

    with pytest.raises(DtMinFrozenError) as e:
        RN.main([job.spec])
    text = str(e.value)
    assert f"at rows [{frozen_row}] of this batch" in text
    assert "ratio 0.0808 against rtol 0.05" in text
    assert "Point of row" in text
    assert "certificate" not in job.events
    assert not (job.out / RN.TALLY_FILE).exists()
    assert list((job.out / "rejected").iterdir()) == []
    assert not (job.out / "nautilus_posterior.npz").exists()


def test_the_local_frozen_code_is_vulcan3_s():
    from vulcan3 import state

    assert constants.TERM_DT_MIN_FROZEN == state.TERM_DT_MIN_FROZEN


def test_a_solver_error_aborts_the_run_and_is_not_taken_for_a_rejection(job):
    """A column frozen at dt_min raises out of the evaluator (under jit, as a runtime error): it must stop the run."""

    def make(_mode):
        def frozen(*_a):
            raise RuntimeError("DtMinFrozenError: 1 of 8 columns froze")

        return frozen

    job.pipe._make_batch_eval = make
    with pytest.raises(RuntimeError, match="DtMinFrozenError"):
        RN.main([job.spec])
    assert "certificate" not in job.events
    assert not (job.out / RN.TALLY_FILE).exists()
    assert list((job.out / "rejected").iterdir()) == []
    assert not (job.out / "nautilus_posterior.npz").exists()
