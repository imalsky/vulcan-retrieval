"""run_nautilus.main on a stub pipeline and a stub sampler, no chemistry: a finished run's files, the identity a resume
is checked against, and the order of that check against everything the run writes.
"""

import json
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from test_run_nautilus import D, _stub_pipe

from retrieval_framework import config_schema as C
from retrieval_framework import constants
from retrieval_framework import nautilus_report as NR
from retrieval_framework import pipeline as P
from retrieval_framework import run_nautilus as RN
from retrieval_framework import run_setup as R

DIGEST = "d" * 64
N_BATCH, N_LIVE = 8, 16
IDENTITY = (
    f"{DIGEST}|warm|n_batch={N_BATCH}|n_live={N_LIVE}|n_update={N_LIVE // 2}|f_live=0.03"
    "|enlarge_per_dim=1.3|networks=4xrelu|stock"
)
LOG_Z, N_EFF = -4.5, 321.0


class _Sampler:
    """Stands for nautilus: ``run`` evaluates seeded points through the run's own likelihood, and hands them back."""

    def __init__(self, like, done=True):
        self.like, self.done = like, done
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

    def make_sampler(like, n_dim, n_live, n_batch, seed, filepath=None, resume=False):
        events.append("sampler")
        sampler["args"] = (n_dim, n_live, n_batch, seed, filepath, resume)
        sampler["s"] = _Sampler(like, sampler.get("done", True))
        return sampler["s"]

    monkeypatch.setattr(RN, "export_runtime", lambda _rt: None)
    monkeypatch.setattr(RN, "setup_logging", lambda _path: None)
    monkeypatch.setattr(C, "export_network", lambda _cfg: None)
    monkeypatch.setattr(C, "hardware_profile", lambda: "stub")
    monkeypatch.setattr(P, "build_pipeline", lambda _cfg: pipe)
    monkeypatch.setattr(RN, "set_observations", lambda _cfg, _pipe, _P: {})
    monkeypatch.setattr(RN, "write_run_identity", lambda *_a: events.append("identity"))
    monkeypatch.setattr(RN, "make_sampler", make_sampler)
    monkeypatch.setattr(NR, "_kzz_at_ref", lambda _cfg: 1.0e9)
    monkeypatch.setattr(NR, "write_certificate", lambda out: events.append("certificate") or out / "cert.json")
    out = tmp_path / "output" / "stub_main_nautilus"
    return SimpleNamespace(spec=str(spec), out=out, pipe=pipe, events=events, sampler=sampler)


def test_the_startup_kzz_decades_follow_the_case_s_kzz_mode(job, monkeypatch):
    """The launch guard's absolute log10 Kzz is the report's: under the power law, the base at kzz_p_ref_bar tilted by
    each draw's slope to the report's level.
    """
    seen = {}
    real = RN.make_loglike

    def capture(*a, **kw):
        seen["startup_kzz"] = kw["startup_kzz"]
        return real(*a, **kw)

    monkeypatch.setattr(RN, "make_loglike", capture)
    monkeypatch.setattr(NR, "_kzz_at_ref", lambda _cfg, p_bar=NR.KZZ_P_REF_BAR: 1.0e9 * (p_bar / 0.1))
    job.pipe.names = ["p0", "kzz_slope", "lnKzz"]
    cfg = C.Config(kzz_mode="powerlaw", kzz_p_ref_bar=0.1, vulcan_cfg_name="W39b")
    RN._saved_loglike(job.pipe, cfg, job.out, N_BATCH, warm=False, resume=False, tally={})
    th = np.array([[0.0, 0.5, np.log(10.0)]])
    # 1e9 at 0.1 bar, x10 by lnKzz, x10^(0.5 x 2) by the slope to 1 mbar
    np.testing.assert_allclose(seen["startup_kzz"](th), [11.0], rtol=1e-12)


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
    assert "APPROXIMATE" in summary["evidence_note"]


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
    job.sampler["done"] = False  # the wall budget ends it: a checkpoint, and no posterior
    RN.main([job.spec, "--resume"])
    assert job.sampler["args"][5] is True
    assert json.loads((job.out / RN.TALLY_FILE).read_text())["n_eval"] == 124
    assert not (job.out / "nautilus_posterior.npz").exists()
    assert "certificate" not in job.events


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


class _FrozenStats(NamedTuple):
    acc: jnp.ndarray
    conv_ok: jnp.ndarray
    reason: jnp.ndarray
    last_delta: jnp.ndarray


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
