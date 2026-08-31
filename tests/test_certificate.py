"""The production certificate must fail closed.

The certificate is what turns "the run finished" into "these numbers may be
reported", so every one of its gates is exercised here on synthetic payloads --
no sampler, no chemistry.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest

from retrieval_framework import certificate
from retrieval_framework.certificate import (
    REQUIRED_VALIDATION_ARTIFACTS,
    collect,
    validate,
)


# The state a validation artifact records. Every key here is compared against the
# run's resolved config, so this doubles as the list of manifest classes a stale
# artifact can drift in.
_ART_CFG = {"art_ptop_bar": 1e-9, "nz": 62, "art_nlayer": 67,
            "yconv_cri": 0.01, "molecules": ["H2O", "CO2"]}
_ART_REPOS = {name: {"commit": "a" * 40}
              for name in ("jax-vulcan", "vulcan-forward", "vulcan-retrieval")}
# CONTENT identity of the opacity/CIA files; the artifact and the run must match
_ART_DATA = {"opacity_sha256": {"H2O": "f" * 64, "CO2": "e" * 64},
             "cia_sha256": {"H2-H2_2011.cia": "1" * 64,
                            "H2-He_2011.cia": "2" * 64}}
_DIGEST = "7" * 64


def _passing_cert():
    return {
        "code": {
            "repos": {name: {"commit": "a" * 40, "dirty": False,
                             "dirty_files": []}
                      for name in ("jax-vulcan", "vulcan-forward",
                                   "vulcan-retrieval")},
            "versions": {"jax": "0.6.2", "numpy": "1.26.4"},
        },
        "data": {"observations": {"sha256": "b" * 64, "bytes": 1234}},
        "resolved_config": {"smc_chem_mode": "cold", "opacity_mode": "exomolop",
                            **_ART_CFG},
        "resolved_config_sha256": "c" * 64,
        "target": {"smc_chem_mode": "cold",
                   "approximate_history_dependent_target": False,
                   "warm_extrapolate": False,
                   "digest": _DIGEST, "digest_samples": _DIGEST,
                   "digest_checkpoint": _DIGEST, "digest_manifest": _DIGEST,
                   "science_data": {k: dict(v) for k, v in _ART_DATA.items()}},
        "convergence": {"reached_beta1": True, "final_beta": 1.0,
                        "n_stages": 27},
        "evidence": {"smc_logZ": -123.4, "smc_logZ_box": -125.0,
                     "log_support_fraction": -1.6,
                     "log_support_fraction_err": 0.04,
                     "f_tp": 0.8, "f_conv": 0.7,
                     "f_c1": 0.995, "f_c2": 1.0,
                     "log_conv_attrition": math.log(0.995)},
        "diagnostics": {
            "ess": [110.0, 98.0, 105.0], "acceptance_rate": [0.55, 0.52, 0.49],
            "unique_particles": [140, 120, 96],
            "warm_capped": [12, 1, 0], "warm_stalled": [4, 0, 0],
            "badgrad": [3, 1, 2],
            "n_particles": 144, "n_mcmc_steps": 4,
        },
        "posterior": [
            {"name": "lnZ", "median": 0.4, "q05": -0.2, "q95": 1.0,
             "prior_lo": -2.303, "prior_hi": 2.303, "prior_position": 0.59},
            {"name": "lnR0", "median": 0.0, "q05": -0.01, "q95": 0.01,
             "prior_lo": -0.08, "prior_hi": 0.08, "prior_position": 0.5},
        ],
        "warm_validation": None,
        "mala_reversibility": None,
        "validation_artifacts": {
            "resolution_ladder": {"status": "PASS", "summary": "", "sha256": "d",
                                  "resolved_config": dict(_ART_CFG),
                                  "repos": dict(_ART_REPOS),
                                  "science_data": {k: dict(v)
                                                   for k, v in _ART_DATA.items()}},
            "top_pressure_ladder": {"status": "PASS", "summary": "", "sha256": "e",
                                    "resolved_config": dict(_ART_CFG),
                                    "repos": dict(_ART_REPOS),
                                    "science_data": {k: dict(v)
                                                     for k, v in _ART_DATA.items()}},
        },
        "checkpoint_present": True,
    }


def _replay(passed=True):
    return {"ran": True, "passed": passed, "max_abs_dlogl": 1e-9,
            "detail": "" if passed else "max |dlogL| 3.0e+00 vs gate 0.1"}


def test_a_clean_cold_run_passes():
    assert validate(_passing_cert(), _replay()) == []


def test_tempered_cloud_is_refused():
    """beta < 1 is a tempered intermediate, never a posterior."""
    c = _passing_cert()
    c["convergence"] = {"reached_beta1": False, "final_beta": 0.83,
                        "n_stages": 19}
    problems = validate(c, _replay())
    assert any("TEMPERED" in p for p in problems), problems


def test_beta_just_below_one_is_refused():
    c = _passing_cert()
    c["convergence"]["final_beta"] = 0.999
    assert any("not 1 within" in p for p in validate(c, _replay()))


def test_dirty_repo_is_refused():
    """A run that cannot be attributed to a committed state is not evidence."""
    c = _passing_cert()
    c["code"]["repos"]["jax-vulcan"]["dirty"] = True
    assert any("DIRTY" in p for p in validate(c, _replay()))


def test_missing_validation_artifact_is_refused():
    for name in REQUIRED_VALIDATION_ARTIFACTS:
        c = _passing_cert()
        c["validation_artifacts"][name] = None
        problems = validate(c, _replay())
        assert any(name in p and "missing" in p for p in problems), (name, problems)


def test_an_unrecorded_or_removed_opacity_mode_is_refused():
    """Only correlated-k runs are certifiable: no record at all, or a record
    of the removed sampled line-by-line path, is a problem."""
    c = _passing_cert()
    c["resolved_config"].pop("opacity_mode")
    assert any("opacity_mode" in p for p in validate(c, _replay()))
    c = _passing_cert()
    c["resolved_config"]["opacity_mode"] = "lbl"
    assert any("removed" in p for p in validate(c, _replay()))


def test_failed_validation_artifact_is_refused():
    c = _passing_cert()
    c["validation_artifacts"]["resolution_ladder"] = {
        "status": "FAIL", "summary": "not converged", "sha256": "d"}
    assert any("FAILED" in p for p in validate(c, _replay()))


@pytest.mark.parametrize("key, value", [
    ("art_ptop_bar", 1e-8),          # pressure domain
    ("nz", 80),                      # chemistry grid
    ("art_nlayer", 101),             # RT grid
    ("yconv_cri", 0.001),            # convergence tolerance
    ("molecules", ["H2O"]),          # opacity list
])
def test_artifact_measured_at_a_different_state_is_refused(key, value):
    """A PASS artifact certifies only the state it was measured at. Every key it
    recorded is bound, not a hand-picked three -- a ladder run at a different
    chemistry tolerance or molecule list measured a different model, whatever its
    grid says (the shipped artifacts were made at yconv_cri=0.001 vs production
    0.01 and rode a three-key comparison)."""
    c = _passing_cert()
    c["resolved_config"][key] = value
    assert any("different state" in p for p in validate(c, _replay()))


def test_artifact_from_different_code_is_refused():
    """The artifact measured what the code of its day computed."""
    c = _passing_cert()
    c["validation_artifacts"]["resolution_ladder"]["repos"] = {
        "vulcan-retrieval": {"commit": "f" * 40}}
    assert any("different state" in p for p in validate(c, _replay()))


def test_artifact_without_a_recorded_config_is_refused():
    c = _passing_cert()
    c["validation_artifacts"]["resolution_ladder"]["resolved_config"] = {}
    assert any("nothing binds it" in p for p in validate(c, _replay()))
    assert not [p for p in validate(_passing_cert(), _replay())
                if "different state" in p]


def test_report_status_is_refused_for_a_gated_artifact():
    """REPORT (a decisive test skipped) is refused for a gated artifact."""
    c = _passing_cert()
    c["validation_artifacts"]["top_pressure_ladder"] = {
        "status": "REPORT", "summary": "decisive test not run", "sha256": "e"}
    assert any("REPORT, not PASS" in p for p in validate(c, _replay()))


def test_missing_cold_replay_is_refused():
    problems = validate(_passing_cert(), None)
    assert any("cold replay not run" in p for p in problems), problems


def test_failed_cold_replay_is_refused():
    problems = validate(_passing_cert(), _replay(passed=False))
    assert any("cold replay MISMATCH" in p for p in problems), problems


def test_missing_support_fraction_error_is_refused():
    c = _passing_cert()
    c["evidence"]["log_support_fraction_err"] = None
    assert any("no log_support_fraction_err recorded" in p
               for p in validate(c, _replay()))


@pytest.mark.parametrize("key", ["smc_logZ", "smc_logZ_box",
                                 "log_support_fraction",
                                 "log_support_fraction_err"])
def test_every_evidence_semantics_field_is_required(key):
    c = _passing_cert()
    c["evidence"][key] = None
    assert any(f"no {key} recorded" in p for p in validate(c, _replay()))


def test_unknown_validation_artifact_status_is_refused():
    c = _passing_cert()
    c["validation_artifacts"]["resolution_ladder"]["status"] = None
    assert any("expected PASS" in p for p in validate(c, _replay()))


# --- run health: a diagnostic that is PRESENT is not a diagnostic that PASSED --

def test_particle_degeneracy_is_refused():
    """N rows of a handful of distinct states still draws a smooth corner plot."""
    c = _passing_cert()
    c["diagnostics"]["unique_particles"] = [140, 120, 9]
    assert any("particle degeneracy" in p for p in validate(c, _replay()))


def test_ess_collapse_is_refused():
    c = _passing_cert()
    c["diagnostics"]["ess"] = [110.0, 12.0, 105.0]
    assert any("ESS collapsed" in p for p in validate(c, _replay()))


@pytest.mark.parametrize("acc", [[0.55, 0.5, 0.01], [0.55, 0.5, 0.99]])
def test_acceptance_outside_the_band_is_refused(acc):
    c = _passing_cert()
    c["diagnostics"]["acceptance_rate"] = acc
    assert any("acceptance" in p for p in validate(c, _replay()))


def test_late_ladder_convergence_rejections_are_refused():
    """warmcap/stalled late in the ladder mean the posterior sits on the
    convergence cliff, so the target is set by count_max, not by the physics."""
    c = _passing_cert()
    c["diagnostics"]["warm_capped"] = [12, 1, 40]
    assert any("convergence cliff" in p for p in validate(c, _replay()))


def test_a_prior_railed_median_is_refused():
    c = _passing_cert()
    c["posterior"][0]["prior_position"] = 0.995
    problems = validate(c, _replay())
    assert any("prior edge" in p and "lnZ" in p for p in problems), problems


def test_a_prior_railed_TAIL_is_refused_even_with_a_central_median():
    """A bimodal marginal can pin a mode to an edge with the median mid-box."""
    c = _passing_cert()
    c["posterior"][0]["prior_position"] = 0.50
    c["posterior"][0]["q95_position"] = 0.995
    problems = validate(c, _replay())
    assert any("95th percentile" in p and "lnZ" in p for p in problems), problems


def test_a_missing_posterior_summary_is_refused():
    c = _passing_cert()
    c["posterior"] = None
    assert any("no posterior summary" in p for p in validate(c, _replay()))


# --- warm runs ---------------------------------------------------------------

def _warm_cert(**wv):
    c = _passing_cert()
    c["target"].update(smc_chem_mode="warm",
                       approximate_history_dependent_target=True,
                       warm_extrapolate=True)
    base = {"dlogl_max": 1e-3, "spectrum_dppm_max": 0.4,
            "atom_ratio_rel_max": 1e-9, "grad_rel_max_gated": 0.01,
            "grad_zeroed_frac": 0.02}
    base.update(wv)
    c["warm_validation"] = base
    c["mala_reversibility"] = {
        "status": "PASS", "summary": "", "checkpoint_matches": True,
        "pairs_requested": 24, "pairs_tested": 24, "asymmetric_pairs": 0,
    }
    return c


def test_a_validated_warm_run_passes():
    """Warm stays usable -- it just has to prove it."""
    assert validate(_warm_cert(), _replay()) == []


def test_warm_run_without_validate_warm_is_refused():
    c = _warm_cert()
    c["warm_validation"] = None
    assert any("UNMEASURED" in p for p in validate(c, _replay()))


def test_warm_run_missing_the_stamp_is_refused():
    c = _warm_cert()
    c["target"]["approximate_history_dependent_target"] = False
    assert any("NOT stamped" in p for p in validate(c, _replay()))


def test_cold_replay_does_not_substitute_for_mala_reversibility():
    c = _warm_cert()
    c["mala_reversibility"] = None
    problems = validate(c, _replay())
    assert any("mala_reversibility.json" in p for p in problems), problems


def test_stale_mala_reversibility_artifact_is_refused():
    c = _warm_cert()
    c["mala_reversibility"]["checkpoint_matches"] = False
    problems = validate(c, _replay())
    assert any("does not match" in p for p in problems), problems


@pytest.mark.parametrize("key, bad", [
    ("dlogl_max", 5.0),
    ("spectrum_dppm_max", 50.0),
    ("grad_rel_max_gated", 0.9),
    ("grad_zeroed_frac", 0.9),
])
def test_each_warm_axis_can_fail_the_certificate(key, bad):
    problems = validate(_warm_cert(**{key: bad}), _replay())
    assert any(key in p for p in problems), (key, problems)


def test_unmeasured_warm_axis_is_not_a_pass():
    """NaN means unmeasured, which must not read as within-gate."""
    problems = validate(_warm_cert(grad_rel_max_gated=float("nan")), _replay())
    assert any("not measured" in p for p in problems), problems


def test_unknown_chem_mode_is_refused():
    c = _passing_cert()
    c["target"]["smc_chem_mode"] = "lukewarm"
    assert any("expected 'cold'" in p for p in validate(c, _replay()))


def test_collect_reads_the_files_and_keys_written_by_run_smc(tmp_path,
                                                              monkeypatch):
    """An actual beta=1 output schema must not look like an empty run."""
    (tmp_path / "config.json").write_text(json.dumps({
        "smc_chem_mode": "cold", "warm_extrapolate": False,
        "inferred_param_names": ["lnZ", "noise_inflation"],
        "inferred_param_prior_types": ["uniform", "log10_uniform"],
        "inferred_param_prior_lo": [-2.0, 0.5],
        "inferred_param_prior_hi": [2.0, 3.0],
    }))
    # the digest the npz copies carry is the RE-HASH of the archived manifest,
    # so this exercises archived_manifest_digest rather than a copied constant
    (tmp_path / certificate.MANIFEST_FILE).write_text(json.dumps(
        {"manifest_version": 2, "config": {"nz": 62}}, indent=2,
        sort_keys=True) + "\n")
    dig = certificate.archived_manifest_digest(tmp_path)
    rng = np.random.default_rng(0)
    draws = np.stack([rng.normal(1.0, 0.1, 64), rng.normal(1.0, 0.05, 64)], 1)
    np.savez(tmp_path / "posterior_samples.npz",
             samples=draws.reshape(1, 64, 2),
             final_beta=np.asarray(1.0), reached_beta1=np.asarray(1),
             smc_target_digest=np.asarray(dig),
             approximate_history_dependent_target=np.asarray(0))
    np.savez(tmp_path / "smc_checkpoint.npz",
             target_digest=np.asarray(dig), last_step=np.asarray(3))
    np.savez(tmp_path / "smc_extra_fields.npz",
             smc_betas=np.asarray([0.0, 0.4, 1.0]),
             smc_logZ=np.asarray(-123.4), smc_logZ_box=np.asarray(-125.0),
             smc_log_support_fraction=np.asarray(-1.6),
             smc_log_support_fraction_err=np.asarray(0.04),
             smc_log_support_physical=np.asarray(-0.2),
             smc_log_support_physical_err=np.asarray(0.01),
             smc_log_conv_attrition=np.asarray(-1.4),
             smc_log_conv_attrition_err=np.asarray(0.03),
             smc_target_digest=np.asarray(dig),
             init_stats_keys=np.asarray(
                 ["n_drawn", "n_alive_phase1", "n_phase2", "n_recert_fail"]),
             init_stats_vals=np.asarray([200, 150, 152, 8], np.int64))
    monkeypatch.setattr(certificate, "_repo_states", lambda: {})
    monkeypatch.setattr(certificate, "_versions", lambda: {})
    monkeypatch.setattr(certificate, "_validation_artifacts", lambda: {})

    cert = collect(tmp_path)
    assert cert["convergence"] == {
        "reached_beta1": True, "final_beta": 1.0, "n_stages": 2}
    assert cert["evidence"]["smc_logZ"] == pytest.approx(-123.4)
    assert cert["evidence"]["log_support_fraction"] == pytest.approx(-1.6)
    assert cert["evidence"]["log_support_fraction_err"] == pytest.approx(0.04)
    # the prior-rail gate reads THESE keys; a rename must not turn it into a no-op
    post = {r["name"]: r for r in cert["posterior"]}
    assert post["lnZ"]["prior_position"] == pytest.approx(0.75, abs=0.05)
    # a log10_uniform prior is flat in log10, so the position is measured there
    assert post["noise_inflation"]["prior_position"] == pytest.approx(
        (0.0 - np.log10(0.5)) / (np.log10(3.0) - np.log10(0.5)), abs=0.05)
    assert certificate.rail_problems(cert["posterior"]) == []
    # the target identity travels with ALL THREE artifacts, and both survival
    # fractions reach the certificate
    assert (cert["target"]["digest"] == cert["target"]["digest_samples"]
            == cert["target"]["digest_checkpoint"]
            == cert["target"]["digest_manifest"] == dig)
    assert not [p for p in certificate.validate(cert) if "target digest" in p]
    assert cert["evidence"]["f_c1"] == pytest.approx(150 / 200)
    assert cert["evidence"]["f_c2"] == pytest.approx((152 - 8) / 152)


def test_data_identity_never_reports_null_opacity_without_saying_why(
        tmp_path, monkeypatch):
    """The certificate must not silently claim a run read no opacity data.

    This repo never sets $VULCAN_FORWARD_DATA -- it hands the engine its tree
    through paths.set_data_root -- so resolving the trees from the environment
    yields null for a run that read gigabytes of them. Holds in any
    environment: either the root resolves, or the certificate says why not.
    """
    monkeypatch.delenv("VULCAN_FORWARD_DATA", raising=False)
    ident = certificate._data_identity(tmp_path, {})
    assert not (ident["data_root_resolved"] is None
                and "engine_data_error" not in ident), (
        "data identity is null with no stated reason -- the environment-only "
        "resolution regressed")
    # the variable's own field reports the variable, never the resolved root
    assert ident["VULCAN_FORWARD_DATA"] is None


def test_certificate_keeps_science_hashes_only_under_target(tmp_path, monkeypatch):
    """Tree summaries and duplicate CIA records are not authoritative identity."""
    (tmp_path / "config.json").write_text("{}")
    expected = {k: dict(v) for k, v in _ART_DATA.items()}
    monkeypatch.setattr(certificate, "science_data_identity", lambda _m: expected)
    monkeypatch.setattr(certificate, "_repo_states", lambda: {})
    monkeypatch.setattr(certificate, "_versions", lambda: {})
    monkeypatch.setattr(certificate, "_validation_artifacts", lambda: {})

    cert = collect(tmp_path)

    assert not ({"opacity_cache", "exomolop", *certificate.CIA_TABLES}
                & cert["data"].keys())
    assert cert["target"]["science_data"] == expected


# --- target identity (RC-03/RC-04): what a checkpoint's numbers belong to -----

def _digest_pipe():
    """Minimal stand-in carrying exactly what target_manifest reads."""
    return SimpleNamespace(
        obs={"wl": np.array([1.0, 2.0]), "wl_lo": np.array([0.9, 1.9]),
             "wl_hi": np.array([1.1, 2.1]), "group": ["A", "A"]},
        obs_depth=np.array([1e-2, 1.1e-2]), obs_sigma=np.array([1e-4, 1e-4]),
        names=["lnZ", "c_o"], prior_types=["uniform", "uniform"],
        param_prior_lo=np.array([-1.0, -1.0]),
        param_prior_hi=np.array([1.0, 1.0]),
        groups=["A"], n_bin=2)


@pytest.mark.parametrize("field, value", [
    ("yconv_cri", 0.001),            # convergence tolerance
    ("molecules", ("H2O",)),         # opacity list
    ("nz", 80),                      # chemistry grid
    ("art_ptop_bar", 1e-8),          # pressure domain
    ("prior_lnZ", (-1.0, 1.0)),      # prior definition
    ("smc_chem_mode", "warm"),       # target semantics
    ("seed", 999),                   # RNG identity (bit-identical resume)
    ("count_max", 4000),             # solver-defined support
])
def test_target_digest_moves_with_every_bound_class(field, value, monkeypatch):
    """Each manifest class must change the digest, or a resume could carry
    numbers from a different density (the shipped stamp was chem_mode only)."""
    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: {"r": {"commit": "c" * 40, "dirty": False}})
    monkeypatch.setattr(certificate, "_versions", lambda: {"jax": "0.6.2"})
    from retrieval_framework import config_schema as _C
    pipe = _digest_pipe()
    base = _C.Config(molecules=("H2O", "CO2"))
    assert getattr(base, field) != value, f"{field} probe equals the default"
    changed = replace(base, **{field: value})
    assert (certificate.target_digest(base, pipe)
            != certificate.target_digest(changed, pipe))


@pytest.mark.parametrize("field, value", [
    ("smc_max_steps", 999),          # per-JOB cap, documented
    ("walltime_seconds", 3600.0),    # per-JOB governor
    ("smc_rt_vjp_chunk", 24),        # batch split, numerically identical
    ("out_dir", Path("/tmp/elsewhere")),
])
def test_target_digest_ignores_per_job_settings(field, value, monkeypatch):
    """A chained RESUME job legitimately changes these; binding them would refuse
    the documented NAS chaining workflow for no scientific reason."""
    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: {"r": {"commit": "c" * 40, "dirty": False}})
    monkeypatch.setattr(certificate, "_versions", lambda: {"jax": "0.6.2"})
    from retrieval_framework import config_schema as _C
    pipe = _digest_pipe()
    base = _C.Config(molecules=("H2O", "CO2"))
    assert (certificate.target_digest(base, pipe)
            == certificate.target_digest(replace(base, **{field: value}), pipe))


def test_target_digest_moves_with_code_and_with_data(monkeypatch):
    from retrieval_framework import config_schema as _C
    monkeypatch.setattr(certificate, "_versions", lambda: {"jax": "0.6.2"})
    pipe, cfg = _digest_pipe(), _C.Config(molecules=("H2O",))
    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: {"r": {"commit": "a" * 40, "dirty": False}})
    a = certificate.target_digest(cfg, pipe)
    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: {"r": {"commit": "b" * 40, "dirty": False}})
    assert certificate.target_digest(cfg, pipe) != a, "code identity not bound"

    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: {"r": {"commit": "a" * 40, "dirty": False}})
    pipe.obs_sigma = pipe.obs_sigma * 2.0
    assert certificate.target_digest(cfg, pipe) != a, "observations not bound"


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: d.update(badgrad=[3, 1, 400]), "zero-drift"),
    (lambda d: d.update(warm_stalled=[4, 0]), "mismatched lengths"),
    (lambda d: d.update(ess=[110.0, float("nan"), 105.0]), "non-finite"),
])
def test_diagnostic_arrays_are_gated_not_just_present(mutate, expect):
    """A ragged or non-finite series must FAIL, not silently disable its gate.

    len(warm_capped) != len(warm_stalled) used to skip the convergence-rejection
    gate entirely, which reads exactly like passing it.
    """
    c = _passing_cert()
    mutate(c["diagnostics"])
    assert any(expect in p for p in validate(c, _replay()))


@pytest.mark.parametrize("survived, justified, refused", [
    (0.50, False, True),    # 50% removed: fatal whatever is claimed
    (0.50, True,  True),
    (0.98, False, True),    # 2%: needs the independent demonstration
    (0.98, True,  False),
    (0.999, False, False),  # 0.1%: below the justification band
])
def test_convergence_attrition_gate(survived, justified, refused):
    """Conditioning on convergence removes prior mass. Above CONV_ATTRITION_FAIL
    the run is unreportable; between CONV_ATTRITION_JUSTIFY and that, only with
    the recorded independent evidence that the removed region is empty."""
    c = _passing_cert()
    c["evidence"].update(log_conv_attrition=math.log(survived),
                         f_c1=survived, f_c2=1.0)
    if justified:
        c["resolved_config"]["attrition_justification"] = "job 65999 re-solve"
    assert bool([p for p in validate(c, _replay()) if "attrition" in p]) is refused


@pytest.mark.parametrize("mutate, expect", [
    (lambda t: t.update(digest=None), "no target digest"),
    (lambda t: t.update(digest_samples=None), "no target digest"),
    (lambda t: t.update(digest_checkpoint=None), "no target digest"),
    (lambda t: t.update(digest_samples="0" * 64), "DISAGREES"),
    (lambda t: t.update(digest_checkpoint="0" * 64), "DISAGREES"),
])
def test_target_digest_must_be_present_and_agree(mutate, expect):
    """RC-03: the checkpoint, the samples and the diagnostics must all name the
    SAME target. A missing copy is not a pass -- an unbound number is exactly
    what lets two targets be reported as one run."""
    c = _passing_cert()
    mutate(c["target"])
    assert any(expect in p for p in validate(c, _replay()))


def test_absent_checkpoint_does_not_demand_its_digest():
    """A finished run may be certified after its checkpoint is cleaned up.

    The manifest is NOT excused the same way: it is written before sampling and
    is the copy that dissents when a refused resume rewrites the run directory.
    """
    c = _passing_cert()
    c["checkpoint_present"] = False
    c["target"]["digest_checkpoint"] = None
    assert not [p for p in validate(c, _replay()) if "target digest" in p]
    c["target"]["digest_manifest"] = None
    assert any("target_manifest.json" in p for p in validate(c, _replay()))


@pytest.mark.parametrize("mutate, expect", [
    (lambda e: e.update(log_conv_attrition=None), "missing or non-finite"),
    (lambda e: e.update(log_conv_attrition=float("nan")), "missing or non-finite"),
    (lambda e: e.update(f_c1=0.5), "does not equal"),
])
def test_attrition_evidence_fails_closed(mutate, expect):
    """An UNMEASURED support fraction is not a small one, and an aggregate that
    disagrees with its own parts describes a different run."""
    c = _passing_cert()
    mutate(c["evidence"])
    assert any(expect in p for p in validate(c, _replay()))


def test_artifact_measured_against_different_opacity_data_is_refused():
    """A swapped k-table leaves every config key identical while changing the
    model the ladder certified."""
    c = _passing_cert()
    art = c["validation_artifacts"]["resolution_ladder"]
    art["science_data"]["opacity_sha256"]["H2O"] = "9" * 64
    assert any("data:opacity_sha256:H2O" in p for p in validate(c, _replay()))
    c2 = _passing_cert()
    c2["validation_artifacts"]["top_pressure_ladder"]["science_data"] = {}
    assert any("data:not recorded" in p for p in validate(c2, _replay()))


def test_both_survival_fractions_must_reach_the_certificate():
    """Their product hides WHICH cull removed the prior mass (RC-06)."""
    c = _passing_cert()
    c["evidence"]["f_c2"] = None
    assert any("survival fractions" in p for p in validate(c, _replay()))


def _repo(**over):
    st = {"commit": "a" * 40, "dirty": True, "dirty_files": ["a.py"],
          "src_diff": "d" * 64}
    st.update(over)
    return {"r": st}


@pytest.mark.parametrize("state, moves", [
    ({"dirty_files": ["a.py", "b.py"]}, False),   # the LIST churns; not an identity
    ({"dirty": False, "dirty_files": [], "src_diff": None}, True),
    ({"src_diff": "e" * 64}, True),               # a DIFFERENT uncommitted src edit
])
def test_target_digest_binds_code_state_not_file_churn(state, moves, monkeypatch):
    """A scratch edit must not refuse a resume, but a different source diff must.

    `dirty_files` churns with any untracked file; `src_diff` is the content hash
    of `git diff HEAD -- src`, so it moves only when the code that defines the
    target changed.
    """
    from retrieval_framework import config_schema as _C
    monkeypatch.setattr(certificate, "_versions", lambda: {"jax": "0.6.2"})
    pipe, cfg = _digest_pipe(), _C.Config(molecules=("H2O",))
    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: _repo())
    d0 = certificate.target_digest(cfg, pipe)
    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: _repo(**state))
    assert (certificate.target_digest(cfg, pipe) != d0) is moves


def test_target_digest_binds_the_cia_tables(monkeypatch):
    """The CIA tables are the only target-affecting input outside git, so nothing
    else can notice a swapped continuum."""
    from retrieval_framework import config_schema as _C
    monkeypatch.setattr(certificate, "_versions", lambda: {"jax": "0.6.2"})
    monkeypatch.setattr(certificate, "_repo_states", lambda *a, **k: _repo())
    pipe, cfg = _digest_pipe(), _C.Config(molecules=("H2O",))
    monkeypatch.setattr(certificate, "_cia_identity",
                        lambda *a, **k: {n: {"sha256": "1" * 64}
                                         for n in certificate.CIA_TABLES})
    d0 = certificate.target_digest(cfg, pipe)
    monkeypatch.setattr(certificate, "_cia_identity",
                        lambda *a, **k: {n: {"sha256": "2" * 64}
                                         for n in certificate.CIA_TABLES})
    assert certificate.target_digest(cfg, pipe) != d0


def test_profile_aliases_cover_every_derived_key():
    """A validation artifact records Config.profile(); the certificate holds the
    flat Config. Any profile key with no Config counterpart must be aliased, or
    validate() reads it as drift and rejects every freshly generated artifact --
    which is exactly what an unaliased `gs_cgs` did.
    """
    from dataclasses import asdict
    from retrieval_framework import config_schema as _C
    cfg = _C.Config(molecules=("H2O",))
    flat = set(asdict(cfg))
    derived = set(cfg.profile()) - flat
    assert derived == set(certificate._PROFILE_ALIASES), (
        "profile key(s) with no Config counterpart and no alias: "
        f"{sorted(derived - set(certificate._PROFILE_ALIASES))}")
    assert set(certificate._PROFILE_ALIASES.values()) <= flat


def test_a_fresh_ladder_artifact_is_accepted():
    """The ladders emit production_profile(); after aliasing it must show ZERO
    drift against the run that produced it."""
    from dataclasses import asdict
    from retrieval_framework import config_schema as _C
    cfg = _C.Config(molecules=("H2O", "CO2"))
    c = _passing_cert()
    c["resolved_config"] = {"smc_chem_mode": "cold", "opacity_mode": "exomolop",
                            **asdict(cfg)}
    art = {"status": "PASS", "summary": "", "sha256": "d",
           "resolved_config": cfg.profile(), "repos": dict(_ART_REPOS),
           "science_data": {k: dict(v) for k, v in _ART_DATA.items()}}
    c["validation_artifacts"] = {n: dict(art) for n in REQUIRED_VALIDATION_ARTIFACTS}
    assert not [p for p in validate(c, _replay()) if "measured at a different" in p]


# --- run-directory identity (RC-03): a refused resume must not rewrite it -----

@pytest.mark.parametrize("stored, want, refused", [
    ("a" * 64, "a" * 64, False),   # same target: resume proceeds
    ("a" * 64, "b" * 64, True),    # different target
    (None, "b" * 64, True),        # legacy checkpoint, no digest at all
])
def test_resume_is_refused_before_the_run_directory_is_written(
        stored, want, refused, tmp_path):
    """run_smc_loop makes the same check, but only AFTER config.json,
    observations.npz and the manifest have been overwritten -- which leaves the
    killed run's samples beside a different run's recorded identity."""
    from retrieval_framework.run_smc import refuse_mismatched_resume
    ck = tmp_path / "smc_checkpoint.npz"
    np.savez(ck, last_step=np.asarray(3),
             **({} if stored is None else {"target_digest": np.asarray(stored)}))
    if refused:
        with pytest.raises(RuntimeError, match="DIFFERENT target"):
            refuse_mismatched_resume(ck, want)
    else:
        refuse_mismatched_resume(ck, want)


def test_resume_with_no_checkpoint_is_not_this_gate(tmp_path):
    """Absent checkpoint is run_smc's own fail-loud path, not a target mismatch."""
    from retrieval_framework.run_smc import refuse_mismatched_resume
    refuse_mismatched_resume(tmp_path / "nope.npz", "a" * 64)


def test_untracked_source_moves_the_code_state(tmp_path, monkeypatch):
    """`git diff HEAD` cannot see an untracked module, but importing one changes
    the code that defines the target."""
    import subprocess
    from retrieval_framework.certificate import _src_state
    env = {**os.environ, "HOME": str(tmp_path), "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_CONFIG_NOSYSTEM": "1"}
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t",
                  "commit", "-qm", "x", "--allow-empty"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                       capture_output=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    clean = _src_state(repo)
    (repo / "src" / "new_module.py").write_text("x = 1\n")
    one = _src_state(repo)
    (repo / "src" / "new_module.py").write_text("x = 2\n")
    two = _src_state(repo)
    assert clean != one != two and clean != two, "untracked src content unbound"
    (repo / "src" / "new_module.py").unlink()
    assert _src_state(repo) == clean


