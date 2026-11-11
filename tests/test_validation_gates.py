"""The validation scripts' gates, on arrays and stubs: an invalid number never reads as a pass, a screen runs the
forward's cloud and mass, and a forward check records the state that solved its spectra.
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import case_config

VALIDATION = Path(__file__).resolve().parents[1] / "validation"


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(VALIDATION))
    import _artifact
    import e2e_check
    import opacity_leave_one_out
    import posterior_forward_check
    import resolution_ladder

    return SimpleNamespace(
        artifact=_artifact,
        e2e=e2e_check,
        loo=opacity_leave_one_out,
        check=posterior_forward_check,
        ladder=resolution_ladder,
    )


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        ([0.02, 0.03], [0.02, 0.03], 0.0),
        ([0.02, 0.03], [0.02, 0.031], 1000.0),
        ([0.02, np.nan], [0.02, np.nan], 0.0),  # a bin the model does not cover, on both sides
        ([0.02, np.nan], [0.02, 0.03], math.inf),  # finite on one side: a passing 0 ppm if it were dropped
        ([np.nan, np.nan], [np.nan, np.nan], math.inf),  # nothing compared
    ],
)
def test_worst_ppm_never_passes_a_nan(scripts, a, b, want):
    got = scripts.artifact.worst_ppm(np.array(a), np.array(b))
    assert got == pytest.approx(want)


def _metrics(**over):
    good = {"max_bin_sigma": 0.01, "D_sigma_max": 0.1, "dex_p95": 0.01, "dex_max": 0.01, "dlogL_good_fit_max": 0.1}
    return {**good, **over}


def test_e2e_gates_fail_a_nan_metric(scripts):
    over = scripts.e2e._over_the_gates
    both, none = np.array([True]), np.array([], int)
    assert over(_metrics(), none, both) == []
    assert over(_metrics(max_bin_sigma=np.nan), none, both) == ["max_bin_sigma"]
    assert over(_metrics(dlogL_good_fit_max=1.0), none, both) == ["dlogL"]
    assert over(_metrics(), np.array([3]), both) == ["certificate_mismatch"]
    assert over(_metrics(), none, np.array([False])) == ["no_jointly_certified_draw"]


def test_the_screen_runs_the_forward_s_cloud_and_mass(scripts):
    """The deck goes through the forward's keywords with the sampled lnM, as the likelihood's RT does."""
    seen = {}

    def depth_r(*_a, **kw):
        seen.update(kw)
        return np.array([1.0, 2.0]), np.array([[1.5, 2.0]])

    import jax.numpy as jnp

    from retrieval_framework.pipeline import Pipeline

    n_chem_tp, i_r0, i_m, i_deck = 2, 2, 3, 4
    pipe = SimpleNamespace(
        n_chem_tp=n_chem_tp,
        dtype=jnp.float64,
        lnR0_idx=i_r0,
        lnM_idx=i_m,
        _cloud_lo=i_deck,
        n_cloud=1,
        B=np.eye(2),
        fwd=SimpleNamespace(
            chem_solve_cold_diag=lambda _c: (np.ones(1), SimpleNamespace(conv_normal=True)),
            aux_from_y=lambda _y, _c: ({"H2O": 1.0}, 2.0, 3.0, 4.0, 5.0),
            rt_kwargs=lambda cloud, lnM: {"deck": float(cloud[0]), "lnM": float(lnM)},
            rt=SimpleNamespace(transmission_depth_r=depth_r),
        ),
    )
    pipe.rt_args = lambda th: Pipeline.rt_args(pipe, th)
    d = scripts.loo.deltas_at(pipe, np.array([0.0, 0.0, 0.01, 0.2, -3.0]), ["H2O"])
    assert seen["deck"] == pytest.approx(-3.0)
    assert seen["lnM"] == pytest.approx(0.2)
    assert seen["wo_mols"] == ["H2O"]
    assert "cloud" not in seen
    np.testing.assert_allclose(d, [[0.5, 0.0]])


def test_a_forward_check_comparison_keeps_what_solved_each_side(scripts, tmp_path):
    wl = np.linspace(3.9, 4.5, 8)
    common = {
        "theta": np.zeros((3, 2)),
        "wl": wl,
        "sigma": np.full(8, 1e-4),
        "certified": np.ones(3, bool),
        "model": np.zeros((3, 8)),
    }
    base, variant = tmp_path / "forward_check_base.npz", tmp_path / "forward_check_mass.npz"
    np.savez(base, settings=np.asarray("{}"), provenance=np.asarray(json.dumps({"commit": "base"})), **common)
    np.savez(variant, settings=np.asarray("{}"), provenance=np.asarray(json.dumps({"commit": "variant"})), **common)
    scripts.check.compare(SimpleNamespace(base=str(base), variant=str(variant)))
    got = json.loads((tmp_path / "forward_check_mass.json").read_text())["provenance"]["resolved_config"]
    assert got["base_solve_provenance"] == {"commit": "base"}
    assert got["variant_solve_provenance"] == {"commit": "variant"}


def test_a_saved_forward_check_records_its_own_solve(scripts, tmp_path, monkeypatch):
    from vulcan_forward import paths as fwd_paths

    # CI has no data folder, and provenance only hashes these tables.
    root = tmp_path / "data"
    for table in ("exomolop/H2O.ktable.h5", "opacity_cache/H2-H2_2011.cia", "opacity_cache/H2-He_2011.cia"):
        (root / table).parent.mkdir(parents=True, exist_ok=True)
        (root / table).write_bytes(b"table")
    monkeypatch.setattr(fwd_paths, "data_root", lambda: root)
    cfg = case_config(molecules=("H2O",), rp_cm=1.0, rstar_cm=10.0)
    run = tmp_path / "output" / "survey" / "hd189733b_nautilus"
    path = scripts.check._save_spectra("base", cfg, {"variant": "base"}, {"model": np.zeros(2)}, run)
    assert path == run / "validation" / "forward_check_base.npz"  # beside its run, never another case's folder
    with np.load(path) as z:
        got = scripts.check._solve_provenance(z)
    assert got["resolved_config"]["molecules"] == ["H2O"]
    assert set(got["repos"]) == {"VULCAN3", "vulcan-forward", "vulcan-retrieval"}


def test_the_ladder_writes_its_artifact_beside_the_case_on_rungs_from_its_layers(scripts, tmp_path, monkeypatch):
    """The ladder's writing path on stub chemistry and RT: rungs from the case's art_nlayer, the artifact in the case's
    validation folder.
    """
    import jax
    import vulcan_forward

    from retrieval_framework import config_schema as C

    jax.config.update("jax_enable_x64", True)  # what the real vulcan_chem import does
    # a stub engine module: importing the real vulcan_chem after exojax raises in a worker that imported exojax
    stub = SimpleNamespace(build_chem_model=lambda _p: SimpleNamespace(converged_y=lambda _t: None))
    monkeypatch.setitem(sys.modules, "vulcan_forward.vulcan_chem", stub)
    monkeypatch.setattr(vulcan_forward, "vulcan_chem", stub, raising=False)
    cfg = case_config(molecules=("H2O",), art_nlayer=20, nu_min=2000.0, nu_max=3000.0)
    monkeypatch.setattr(scripts.artifact, "spec_case", lambda _p: (cfg, tmp_path / "validation"))
    monkeypatch.setattr(scripts.artifact, "collect_provenance", lambda _rc, _d: {})
    monkeypatch.setattr(C, "export_network", lambda _cfg: None)
    rungs = []
    monkeypatch.setattr(
        scripts.ladder, "_run_rung", lambda prof, *_a: rungs.append(prof["art_nlayer"]) or {"binned": np.ones(4)}
    )
    monkeypatch.setattr(sys, "argv", ["resolution_ladder.py", "--spec", "x.yaml"])
    assert scripts.ladder.main() == 0
    assert rungs == [20, 30, 45]
    got = json.loads((tmp_path / "validation" / "resolution_ladder.json").read_text())
    assert got["verdict"]["status"] == "REPORT"


def test_the_elemental_audit_records_a_refused_start_as_a_fail_row(scripts, tmp_path, monkeypatch):
    """The engine refuses a start whose repair factor is not > 0: the audit records that draw and fails, not crash."""
    import elemental_audit
    import jax
    import vulcan_forward

    from retrieval_framework import config_schema as C

    jax.config.update("jax_enable_x64", True)

    def audit_init(th):
        if th[0] > 0.0:
            raise ValueError("the chemistry start column is not physical: stub")
        return {
            "ratio_max_rel_err": 0.0,
            "density_closure_max_rel": 0.0,
            "atom_ini_max_rel_err": 0.0,
            "dln_CO_achieved": float(th[1]),
            "min_repair_factor": 1.0,
        }

    stub = SimpleNamespace(build_chem_model=lambda *_a, **_k: SimpleNamespace(audit_init=audit_init))
    monkeypatch.setitem(sys.modules, "vulcan_forward.vulcan_chem", stub)
    monkeypatch.setattr(vulcan_forward, "vulcan_chem", stub, raising=False)
    monkeypatch.setattr(scripts.artifact, "spec_case", lambda _p: (case_config(), tmp_path))
    monkeypatch.setattr(scripts.artifact, "collect_provenance", lambda _rc, _d: {})
    monkeypatch.setattr(C, "export_network", lambda _cfg: None)
    monkeypatch.setattr(sys, "argv", ["elemental_audit.py", "--spec", "x.yaml", "--n", "8"])
    assert elemental_audit.main() == 1
    got = json.loads((tmp_path / "elemental_audit.json").read_text())
    refused = got["measurements"][0]["refused"]
    assert refused
    assert all("not physical" in r["refusal"] for r in refused)
    assert got["verdict"]["status"] == "FAIL"


def test_a_file_from_before_provenance_says_so(scripts, tmp_path):
    np.savez(tmp_path / "old.npz", settings=np.asarray("{}"))
    with np.load(tmp_path / "old.npz") as z:
        assert scripts.check._solve_provenance(z) == "not recorded"


def test_the_ladder_gates_fail_a_bin_finite_on_one_side(scripts, capsys):
    finite = np.array([0.02, 0.03])
    holed = np.array([0.02, np.nan])
    knob = "art_nlayer"
    results = {67: {"binned": finite}, 101: {"binned": finite}}
    assert scripts.ladder._depth_gates(results, [67, 101], (67, 101), knob)[1]
    results = {67: {"binned": finite}, 101: {"binned": holed}}
    measurements, ok = scripts.ladder._depth_gates(results, [67, 101], (67, 101), knob)
    assert not ok
    assert measurements[0]["passed"] is False
    results = {67: {"jac_lnZ": finite}, 101: {"jac_lnZ": finite * 1.001}}
    assert scripts.ladder._jacobian_gate(results, (67, 101), knob)[1]
    for ja, jb in ((holed, finite), (finite, holed)):
        results = {67: {"jac_lnZ": ja}, 101: {"jac_lnZ": jb}}
        measurement, ok = scripts.ladder._jacobian_gate(results, (67, 101), knob)
        assert not ok
        assert measurement["passed"] is False
    capsys.readouterr()


def test_the_e2e_solve_aborts_on_a_frozen_column(scripts):
    """Its rows carry a certificate and no reason, so a frozen column would read as a plain uncertified draw."""
    import jax.numpy as jnp
    from vulcan3 import DtMinFrozenError

    from retrieval_framework import constants, pipeline as P

    def frozen(U, Y, refs, _active):
        n = U.shape[0]
        reason = jnp.zeros(n, int).at[1].set(constants.TERM_DT_MIN_FROZEN)
        stats = P.EvalStats(jnp.zeros(n, int), jnp.zeros(n, bool), reason, jnp.full(n, 0.08))
        return jnp.zeros(n), Y, refs, stats

    pipe = SimpleNamespace(
        _make_batch_eval=lambda _mode: frozen,
        y_baseline=jnp.zeros((1, 1)),
        dtype=jnp.float64,
        fwd=SimpleNamespace(chem_rtol=0.05),
    )
    with pytest.raises(DtMinFrozenError, match=r"at rows \[1\] of this batch"):
        scripts.e2e._solve_rows(pipe, jnp.zeros((3, 2)), True)


def test_the_spot_variant_needs_its_star_and_the_kzz_cut_its_scale_mode(scripts, monkeypatch, tmp_path):
    """The spotted star is the case's, named on the command line; the solver-cut test refuses a power-law run, whose
    absolute Kzz is not lnKzz times one base.
    """
    import rejected_resolve

    monkeypatch.setattr(sys, "argv", ["posterior_forward_check.py", "run", str(tmp_path), "--variant", "spot"])
    with pytest.raises(SystemExit):
        scripts.check.main()
    (tmp_path / "config.json").write_text(json.dumps({"kzz_mode": "powerlaw", "inferred_param_names": ["lnKzz"]}))
    with pytest.raises(RuntimeError, match="power-law"):
        rejected_resolve._run(tmp_path)


def test_an_artifact_hashes_the_tables_of_a_measurement_and_not_of_a_comparison(scripts, monkeypatch):
    """A comparison of saved results reads no table, so it runs on a machine without them."""
    hashed = []
    monkeypatch.setattr(scripts.artifact, "science_data_identity", lambda m: hashed.append(m) or {"opacity_sha256": {}})
    monkeypatch.setattr(scripts.artifact, "_data_identity", lambda *_a: {})
    monkeypatch.setattr(scripts.artifact, "_versions", lambda: {})
    assert "science_data" not in scripts.artifact.collect_provenance({"base": "x", "variant": "y"}, Path("."))
    assert hashed == []
    assert "science_data" in scripts.artifact.collect_provenance({"molecules": ["H2O"]}, Path("."))
    assert hashed == [["H2O"]]


def test_an_artifact_hashes_the_case_s_own_observations(scripts, tmp_path, monkeypatch):
    """The products a case's config names are hashed, so an artifact measured on another planet's data shows it."""
    monkeypatch.setattr(scripts.artifact, "_data_identity", lambda *_a: {})
    monkeypatch.setattr(scripts.artifact, "_versions", lambda: {})
    (tmp_path / "a.csv").write_text("a\n")
    rc = {"obs_dir": str(tmp_path), "obs_products": {"G": ["a.csv"]}}
    first = scripts.artifact.collect_provenance(rc, tmp_path)["observation_products"]
    (tmp_path / "a.csv").write_text("b\n")
    assert scripts.artifact.collect_provenance(rc, tmp_path)["observation_products"] != first
