"""The production certificate must fail closed.

The certificate is what turns "the run finished" into "these numbers may be
reported", so every one of its gates is exercised here on synthetic payloads --
no sampler, no chemistry.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from retrieval_framework import certificate
from retrieval_framework.certificate import (
    REQUIRED_VALIDATION_ARTIFACTS,
    collect,
    validate,
)


def _passing_cert():
    return {
        "code": {
            "repos": {name: {"commit": "a" * 40, "dirty": False,
                             "dirty_files": []}
                      for name in ("jax-vulcan", "vulcan-forward",
                                   "vulcan-retrieval", "vulcan-jwst-tool")},
            "versions": {"jax": "0.6.2", "numpy": "1.26.4"},
        },
        "data": {"observations": {"sha256": "b" * 64, "bytes": 1234}},
        "resolved_config": {"smc_chem_mode": "cold", "opacity_mode": "exomolop"},
        "resolved_config_sha256": "c" * 64,
        "target": {"smc_chem_mode": "cold",
                   "approximate_history_dependent_target": False,
                   "warm_extrapolate": False},
        "convergence": {"reached_beta1": True, "final_beta": 1.0,
                        "n_stages": 27},
        "evidence": {"smc_logZ": -123.4, "smc_logZ_box": -125.0,
                     "log_support_fraction": -1.6,
                     "log_support_fraction_err": 0.04,
                     "f_tp": 0.8, "f_conv": 0.7},
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
            "resolution_ladder": {"status": "PASS", "summary": "", "sha256": "d"},
            "top_pressure_ladder": {"status": "PASS", "summary": "", "sha256": "e"},
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


def test_report_status_is_refused_for_a_gated_artifact():
    """top_pressure_ladder REPORT means --extend-chem was not run."""
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


def test_a_missing_posterior_summary_is_refused():
    c = _passing_cert()
    c["posterior"] = None
    assert any("no posterior summary" in p for p in validate(c, _replay()))


# --- warm runs ---------------------------------------------------------------

def _warm_cert(**wv):
    c = _passing_cert()
    c["target"] = {"smc_chem_mode": "warm",
                   "approximate_history_dependent_target": True,
                   "warm_extrapolate": True}
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
    rng = np.random.default_rng(0)
    draws = np.stack([rng.normal(1.0, 0.1, 64), rng.normal(1.0, 0.05, 64)], 1)
    np.savez(tmp_path / "posterior_samples.npz",
             samples=draws.reshape(1, 64, 2),
             final_beta=np.asarray(1.0), reached_beta1=np.asarray(1),
             approximate_history_dependent_target=np.asarray(0))
    np.savez(tmp_path / "smc_extra_fields.npz",
             smc_betas=np.asarray([0.0, 0.4, 1.0]),
             smc_logZ=np.asarray(-123.4), smc_logZ_box=np.asarray(-125.0),
             smc_log_support_fraction=np.asarray(-1.6),
             smc_log_support_fraction_err=np.asarray(0.04),
             smc_log_support_physical=np.asarray(-0.2),
             smc_log_support_physical_err=np.asarray(0.01),
             smc_log_conv_attrition=np.asarray(-1.4),
             smc_log_conv_attrition_err=np.asarray(0.03))
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
