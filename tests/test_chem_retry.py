"""cfg.chem_retry, run_nautilus's rtol retry of uncertified solves.

Off (the default) it leaves the target manifest -- so every existing run's
digest -- and the banner as they were; on, it binds the manifest, validates
its settings and builds the retry profile by changing only rtol and the cap.
The make_loglike logic is pinned on a stub pipeline in test_run_nautilus.py.
The slow test builds the REAL smoke pipeline with the retry on
(conftest.build_smoke_pipe at count_max=50 < count_min, so no first solve
certifies) and checks the retry program differs from the main one only in rtol
and the cap, and that make_loglike sends every in-window draw through it.
"""
import json
import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from retrieval_framework import provenance  # noqa: E402
from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402

VALID = dict(vulcan_cfg_name="W39b", rp_cm=1.0, rstar_cm=10.0)
RETRY_KEYS = provenance.SWITCHED_KEYS["chem_retry"]
SWITCHED = {k for keys in provenance.SWITCHED_KEYS.values() for k in keys}


def _digest_pipe():
    """What target_manifest reads (test_provenance._digest_pipe)."""
    return SimpleNamespace(
        obs={"wl": np.array([1.0, 2.0]), "wl_lo": np.array([0.9, 1.9]),
             "wl_hi": np.array([1.1, 2.1]), "group": ["A", "A"]},
        obs_depth=np.array([1e-2, 1.1e-2]), obs_sigma=np.array([1e-4, 1e-4]),
        names=["lnZ", "c_o"], prior_types=["uniform", "uniform"],
        param_prior_lo=np.array([-1.0, -1.0]), param_prior_hi=np.array([1.0, 1.0]),
        groups=["A"], n_bin=2)


@pytest.fixture
def frozen_code(monkeypatch):
    monkeypatch.setattr(provenance, "_repo_states",
                        lambda *a, **k: {"r": {"commit": "c" * 40, "dirty": False}})
    monkeypatch.setattr(provenance, "_versions", lambda: {"jax": "0.6.2"})


def test_off_leaves_the_manifest_as_before(frozen_code):
    pipe, base = _digest_pipe(), C.Config(molecules=("H2O",))
    man = provenance.target_manifest(base, pipe)
    # exactly the manifest of a Config that never had the retry / presolve fields
    assert man["config"] == {k: v for k, v in asdict(base).items()
                             if k not in provenance.TARGET_FREE_KEYS and k not in SWITCHED}
    d0 = provenance.target_digest(base, pipe)
    assert provenance.target_digest(
        replace(base, chem_retry_rtol=0.01, chem_retry_count_max=9000), pipe) == d0
    # the presolve switch off (its default) changes nothing, with the retry on too
    on = replace(base, chem_retry=True)
    assert "chem_presolve" not in provenance.target_manifest(on, pipe)["config"]
    assert provenance.target_digest(replace(on, chem_presolve=False), pipe) == \
        provenance.target_digest(on, pipe)


def test_presolve_on_binds_the_manifest(frozen_code):
    pipe = _digest_pipe()
    on = C.Config(molecules=("H2O",), chem_retry=True, chem_presolve=True)
    man = provenance.target_manifest(on, pipe)
    assert man["config"]["chem_presolve"] is True
    assert {k: man["config"][k] for k in RETRY_KEYS} == {
        "chem_retry": True, "chem_retry_rtol": 0.05, "chem_retry_count_max": 5000}
    assert provenance.target_digest(on, pipe) != provenance.target_digest(
        replace(on, chem_presolve=False), pipe)


def test_validate_config_refuses_a_presolve_without_the_retry():
    C.validate_config(C.Config(**VALID, chem_retry=True, chem_presolve=True))
    with pytest.raises(ValueError, match="chem_presolve=True needs chem_retry"):
        C.validate_config(C.Config(**VALID, chem_presolve=True))
    with pytest.raises(ValueError, match="chem_presolve must be"):
        C.validate_config(C.Config(**VALID, chem_retry=True, chem_presolve=1))


def test_banner_names_the_presolve_only_when_on():
    off = C.describe_config(C.Config(**VALID, count_max=3000, chem_retry=True))
    on = C.describe_config(C.Config(**VALID, count_max=3000, chem_retry=True,
                                    chem_presolve=True))
    assert "chem_presolve" not in off and "chem_retry ON" in off
    assert "chem_presolve ON" in on and "chem_retry ON" not in on


def test_on_binds_the_manifest(frozen_code):
    pipe, on = _digest_pipe(), C.Config(molecules=("H2O",), chem_retry=True)
    man = provenance.target_manifest(on, pipe)
    assert {k: man["config"][k] for k in RETRY_KEYS} == {
        "chem_retry": True, "chem_retry_rtol": 0.05, "chem_retry_count_max": 5000}
    d = provenance.target_digest(on, pipe)
    for ch in (dict(chem_retry=False), dict(chem_retry_rtol=0.1),
               dict(chem_retry_count_max=8000)):
        assert provenance.target_digest(replace(on, **ch), pipe) != d, ch


@pytest.mark.parametrize("bad, msg", [
    (dict(chem_retry_rtol=0.0), "chem_retry_rtol"),
    (dict(chem_retry_rtol=1.0), "chem_retry_rtol"),
    (dict(chem_retry_rtol=math.nan), "chem_retry_rtol"),
    (dict(chem_retry_count_max=1000), "chem_retry_count_max"),   # < warm_count_max
])
def test_validate_config_refuses_bad_retry_settings(bad, msg):
    C.validate_config(C.Config(**VALID, chem_retry=True))          # the defaults pass
    with pytest.raises(ValueError, match=msg):
        C.validate_config(C.Config(**VALID, chem_retry=True, **bad))
    C.validate_config(C.Config(**VALID, **bad))                     # off: inert


def test_validate_config_refuses_a_non_bool_switch():
    with pytest.raises(ValueError, match="chem_retry must be"):
        C.validate_config(C.Config(**VALID, chem_retry=1))


def test_banner_names_the_retry_only_when_on():
    assert "chem_retry" not in C.describe_config(C.Config(**VALID))
    assert "chem_retry ON" in C.describe_config(C.Config(**VALID, count_max=3000,
                                                         chem_retry=True))


def test_retry_profile_changes_only_rtol_and_the_cap():
    cfg = C.Config(**VALID, count_max=3000, chem_retry=True, cfg_overrides={"Rp": 1.0})
    p0, p1 = cfg.profile(), C.retry_profile(cfg)
    assert p1.pop("cfg_overrides") == {"Rp": 1.0, "rtol": 0.05}
    assert (p1.pop("count_max"), p1.pop("skip_warmup")) == (5000, True)
    p0.pop("cfg_overrides")
    p0.pop("count_max")
    assert p1 == p0
    assert cfg.profile()["cfg_overrides"] == {"Rp": 1.0}            # untouched


@pytest.mark.slow
def test_real_pipeline_retries_every_uncertified_draw(tmp_path, monkeypatch):
    from conftest import CAPPED_COLD_CMAX, build_smoke_pipe
    from retrieval_framework import retrieval_forward as RF
    from retrieval_framework import run_nautilus as RN
    retry_cmax = CAPPED_COLD_CMAX + 10            # still < count_min: cannot certify
    pipe = build_smoke_pipe(count_max=CAPPED_COLD_CMAX, warm_count_max=5, chem_retry=True,
                            chem_retry_rtol=0.05, chem_retry_count_max=retry_cmax)
    r = pipe.chem_retry
    assert (r.count_max_first, r.count_max, r.rtol) == (CAPPED_COLD_CMAX, retry_cmax, 0.05)
    st0, st1 = pipe.fwd.chem._integ._statics, pipe.fwd.chem_retry._integ._statics
    diff = set(RF.statics_diff(st0, st1))
    assert "count_max" in diff and diff <= RF.RETRY_STATICS_FREE, diff
    assert float(st1.rtol_accept) == 0.05 and r.rtol_first == float(st0.rtol_accept)

    monkeypatch.setattr(RN, "RETRY_BUCKET", 2)     # keep the CPU second pass at 2 lanes
    U = np.asarray(pipe.sample_prior_u(jax.random.PRNGKey(0), 2))   # T-P in window
    Z = 1.0 / (1.0 + np.exp(-U))
    tally, log = {}, tmp_path / "batches.jsonl"
    L = RN.make_loglike(pipe, 2, tally, None, log)(Z)
    assert np.all(np.isneginf(L))
    assert tally["retried"] == tally["retry_exhausted"] == tally["exhausted"] == 2
    assert tally["retry_certified"] == tally["tp_out"] == 0
    row = json.loads(log.read_text().splitlines()[0])
    assert row["mode"] == "cold" and min(row["acc"]) >= CAPPED_COLD_CMAX
    assert row["retry"]["rows"] == [0, 1] and row["retry"]["width"] == 2
    assert min(row["retry"]["acc"]) >= retry_cmax and not any(row["retry"]["conv"])


@pytest.mark.slow
def test_real_pipeline_presolve_sends_every_draw_through_the_retry(tmp_path):
    """chem_presolve on the REAL smoke pipeline: the presolve evaluator is the
    main program's chemistry (no RT), and every in-window draw is solved again
    on the retry program at the fixed width n_batch (here no solve certifies:
    both caps sit below count_min)."""
    from conftest import CAPPED_COLD_CMAX, build_smoke_pipe
    from retrieval_framework import run_nautilus as RN
    retry_cmax = CAPPED_COLD_CMAX + 10
    pipe = build_smoke_pipe(count_max=CAPPED_COLD_CMAX, warm_count_max=5, chem_retry=True,
                            chem_retry_rtol=0.05, chem_retry_count_max=retry_cmax,
                            chem_presolve=True)
    assert pipe.chem_retry.presolve is True
    U = np.asarray(pipe.sample_prior_u(jax.random.PRNGKey(0), 2))   # T-P in window
    Y, refs = P._blank_state(pipe, 2)
    Yp, rp, acc, conv = jax.jit(pipe._make_presolve_eval("cold"))(U, Y, refs, np.ones(2, bool))
    th = np.asarray(jax.vmap(pipe.theta_from_u)(U))
    assert np.allclose(np.asarray(rp), th[:, :2]) and np.all(np.isfinite(np.asarray(Yp)))
    assert np.all(np.asarray(acc) >= CAPPED_COLD_CMAX) and not np.any(np.asarray(conv))
    Z = 1.0 / (1.0 + np.exp(-U))
    tally, log = {}, tmp_path / "batches.jsonl"
    L = RN.make_loglike(pipe, 2, tally, None, log)(Z)
    assert np.all(np.isneginf(L))
    assert tally["retried"] == tally["retry_exhausted"] == tally["exhausted"] == 2
    assert tally["presolve_certified"] == 0 and tally["retry_certified"] == 0
    row = json.loads(log.read_text().splitlines()[0])
    assert (row["presolve"]["certified"], row["presolve"]["count_max"]) == (0, CAPPED_COLD_CMAX)
    assert row["retry"]["rows"] == [0, 1] and row["retry"]["width"] == 2
    assert min(row["retry"]["acc"]) >= retry_cmax and not any(row["retry"]["conv"])
