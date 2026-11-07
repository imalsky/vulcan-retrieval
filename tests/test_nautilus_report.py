"""nautilus_report on a tiny synthetic run_nautilus output directory (no
chemistry): the certificate's fractions and caveats follow from the tally, the
summary and the identity files, and the CLI writes the weighted corner and T-P
figures; given model spectra, the spectrum fit too.
"""

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from retrieval_framework import config_schema as C
from retrieval_framework import nautilus_report as NR
from retrieval_framework import provenance
from retrieval_framework import report_figures

NAMES = ["lnZ", "c_o", "lnKzz", "Tirr", "log10kappa", "log10gamma", "lnR0", "noise_inflation"]
MEAN = np.array([0.2, -0.5, 0.0, 1000.0, -1.0, -1.0, 0.03, 1.2])
SD = np.array([0.1, 0.1, 0.5, 50.0, 0.2, 0.2, 0.005, 0.05])
# 1000 draws: 100 off-window, 45 of the 900 in-window rejected, 855 certified
TALLY = {
    "n_eval": 1000,
    "tp_out": 100,
    "exhausted": 40,
    "stalled": 5,
    "nonfinite": 0,
    "n_cold": 8,
    "cold_certified": 6,
    "n_warm": 992,
    "warm_certified": 849,
}


def _run_dir(out):
    rng = np.random.default_rng(0)
    cfg = asdict(
        C.Config(
            molecules=("H2O",), vulcan_cfg_name="W39b", network="thermo/SNCHO_photo_network_2025.txt", cold_lanes=8
        )
    )
    cfg.update(
        inferred_param_names=NAMES, inferred_param_labels=NAMES, inferred_param_truth=[float("nan")] * len(NAMES)
    )
    (out / "config.json").write_text(json.dumps(cfg, default=str))
    (out / provenance.MANIFEST_FILE).write_text(json.dumps({"manifest_version": 2, "code": {}}))
    digest = provenance.archived_manifest_digest(out)
    (out / "target_digest.txt").write_text(f"{digest}|warm|n_batch=8\n")
    (out / "nautilus_tally.json").write_text(json.dumps(TALLY))
    (out / "nautilus_summary.json").write_text(
        json.dumps(
            {
                "logZ_box": -12.3,
                "n_like": 990,
                "n_eff": 1234.0,
                "n_live": 100,
                "n_batch": 8,
                "n_update": 50,
                "f_live": 0.03,
                "enlarge_per_dim": 1.3,
                "starts": "anchored warm (growing)",
                "rejections_and_starts": TALLY,
                "seed": 1,
                "chem_rtol": 0.2,
            }
        )
    )
    theta = MEAN + SD * rng.normal(size=(400, len(NAMES)))
    log_w = rng.normal(size=400)
    np.savez(
        out / "nautilus_posterior.npz",
        param_names=np.asarray(NAMES),
        theta=theta,
        log_w=log_w - np.log(np.exp(log_w).sum()),
    )
    # the conditioning map's inputs: 90 certified columns, 9 per lnKzz bin (the anchors
    # keep the first 6 dims), and 10 in-window rejections in the lowest bin beside 5
    # T-P-out rows (class 0, not in the map)
    (out / "anchors").mkdir()
    za = np.full((90, 6), 0.5)
    za[:, 2] = (np.arange(90) % 10 + 0.5) / 10
    np.savez(out / "anchors" / "anchors_00000.npz", z=za, Y=np.zeros((90, 1)), refs=np.zeros((90, 3)))
    (out / "rejected").mkdir()
    zr = np.full((15, len(NAMES)), 0.5)
    zr[:10, 2], zr[10:, 2] = 0.05, 0.95
    np.savez(
        out / "rejected" / "rejected_00000.npz",
        z=zr,
        cls=np.r_[[1] * 10, [0] * 5],
        acc=np.zeros(15, int),
        warm=np.int8(1),
    )
    wl = np.linspace(3.0, 5.0, 20)
    np.savez(
        out / "observations.npz",
        wl=wl,
        depth=0.02 + 1e-4 * np.sin(wl),
        sigma=np.full(20, 1e-4),
        group=np.asarray(["NIRISS"] * 8 + ["G395H"] * 12),
    )
    return theta, 0.02 + 1e-4 * np.sin(wl)


@pytest.mark.parametrize(
    "key",
    ["baseline_metallicity_solar", "kzz_mode", "cfg_overrides", "tp_deep_p_bar", "integrate_nuisance", "cold_lanes"],
)
def test_a_config_without_a_key_the_report_reads_is_refused(tmp_path, monkeypatch, key):
    """A run's config.json holds every Config field, so a missing one must not be read as today's default."""
    _run_dir(tmp_path)
    monkeypatch.setattr(provenance, "_versions", lambda: {"jax": "0.6.2"})
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    cfgj = json.loads((tmp_path / "config.json").read_text())
    del cfgj[key]
    (tmp_path / "config.json").write_text(json.dumps(cfgj))
    with pytest.raises(KeyError, match=key):
        NR.certificate(tmp_path)


@pytest.mark.parametrize("key", ["chem_presolve", "chem_retry"])
def test_a_second_program_run_is_refused_rather_than_re_solved_without_it(tmp_path, key):
    _run_dir(tmp_path)
    cfgj = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**cfgj, key: True}))
    with pytest.raises(RuntimeError, match="0a2dbf3"):
        NR._config_from_json(tmp_path / "config.json")


def test_a_frozen_draw_stops_the_report_naming_the_row(tmp_path, monkeypatch):
    """A column frozen at dt_min is a solver failure on a finished run too: not one more uncertified spectrum."""
    from vulcan_forward import vulcan_chem  # noqa: F401  # before vulcan3

    import jax.numpy as jnp
    import vulcan3

    from retrieval_framework import pipeline as P

    if not hasattr(vulcan3, "DtMinFrozenError"):
        pytest.skip("this vulcan3 has no guard")

    _run_dir(tmp_path)
    n = 2

    class Diag(NamedTuple):
        conv_normal: jnp.ndarray
        accept_count: jnp.ndarray
        termination_reason: jnp.ndarray
        last_delta: jnp.ndarray

    def solve(_th):
        reason = jnp.zeros(n, int).at[1].set(P._TERM_DT_MIN_FROZEN)
        return jnp.zeros((n, 1)), Diag(jnp.ones(n), jnp.ones(n, int), reason, jnp.full(n, 0.08))

    pipe = SimpleNamespace(
        names=NAMES,
        dtype=jnp.float64,
        n_chem_tp=6,
        nuisance=None,
        observed_depth_from_y_jit=lambda *_a: pytest.fail("the RT ran after a frozen solve"),
        fwd=SimpleNamespace(chem_solve_cold_diag_batch=solve, chem_rtol=0.05),
    )
    monkeypatch.setattr(C, "export_network", lambda _cfg: None)
    monkeypatch.setattr(C, "choose_solver", lambda: "fast")
    monkeypatch.setattr(P, "build_pipeline", lambda _cfg: pipe)
    with pytest.raises(vulcan3.DtMinFrozenError, match=r"at rows \[1\] of this batch"):
        NR._model_spectra(tmp_path, n)
    assert not (tmp_path / "nautilus_spectra.npz").exists()


def test_certificate_and_figures(tmp_path, monkeypatch):
    pytest.importorskip("corner")
    theta, depth = _run_dir(tmp_path)
    # the version record imports exojax, which must not precede vulcan_chem in a worker
    monkeypatch.setattr(provenance, "_versions", lambda: {"jax": "0.6.2"})
    NR.main([str(tmp_path), "--spectra", "0"])

    cert = json.loads((tmp_path / NR.CERT_FILE).read_text())
    rej = cert["rejections"]
    assert rej["tp_out_of_n_eval"] == pytest.approx(0.1)
    assert rej["rejected_of_in_window"] == pytest.approx(45 / 900)
    assert rej["exhausted_of_in_window"] == pytest.approx(40 / 900)
    assert rej["warm_certified_of_warm"] == pytest.approx(849 / 992)
    assert cert["result"] == {"logZ_box": -12.3, "logZ_err": None, "n_eff": 1234.0, "n_like": 990}
    assert cert["starts"]["mode"] == "warm"
    assert cert["chemistry"]["rtol"] == 0.2
    # the derived quantities: weighted 16/50/84% of what the paper quotes
    der = cert["derived"]
    z_o = der["metallicity_O_H_solar"]
    assert len(z_o) == 3
    assert z_o[0] < z_o[1] < z_o[2]
    assert abs(z_o[1] - 10.0 * np.exp(MEAN[0])) < 0.5  # the baseline is 10x solar
    assert der["metallicity_total_solar"][1] < z_o[1]  # C/O below solar lowers total metals
    assert der["S_to_O_solar"] is None  # sulfur was not freed in this run
    # the Kzz block: weights at multiples of the prior floor, the absolute Kzz on the
    # case's own profile, and the x10 ceiling only on W39b's; a constant-Kzz case
    # (1e9) shifts the absolute Kzz by its base and has no ceiling
    assert der["truncated_kzz_le_x10"]["weight"] == pytest.approx(1.0 - der["kzz_weight_above_x10"])
    assert der["kzz_weight_below_1.5x_floor"] <= der["kzz_weight_below_2x_floor"]
    cfgj = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**cfgj, "vulcan_cfg_name": "data/survey/w107b/W107b.yaml"}))
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    d107 = NR.derived_quantities(tmp_path)
    assert not any("x10" in k for k in d107)
    np.testing.assert_allclose(
        np.subtract(d107["log10_kzz_ref_cgs"], der["log10_kzz_ref_cgs"]),
        9.0 - np.log10(NR._kzz_at_ref({"vulcan_cfg_name": "W39b", "cfg_overrides": {}})),
        rtol=0,
        atol=1e-12,
    )
    assert der["T_deep_p_bar"] == 1.0
    assert len(der["T_deep_K"]) == 3 == len(der["Tint_K"])
    cond = cert["conditioning"]
    assert cond["rejected_of_in_window"] == pytest.approx(10 / 100)
    kzz = cond["lnKzz"]
    assert kzz["n_in_window"] == [19] + [9] * 9
    assert kzz["rejected_frac"] == pytest.approx([10 / 19] + [0.0] * 9)
    assert sum(kzz["posterior_weight"]) == pytest.approx(1.0)
    assert "lnZ" in cond
    assert "Tint" not in cond
    text = " ".join(cert["caveats"])
    for phrase in (
        "5.0% of the draws",
        "no logZ error",
        "Warm starts",
        "Lane queue",
        "counted twice",
        "52.6% of the in-window draws in the lowest bin",
    ):
        assert phrase in text, phrase
    assert "different run" not in text  # the manifest hashes to the digest
    (tmp_path / "target_digest.txt").write_text("0" * 64 + "|warm\n")
    assert any("different run" in c for c in NR.certificate(tmp_path)["caveats"])

    assert sorted(p.name for p in (tmp_path / "plots").iterdir()) == ["corner.png", "tp_posterior.png"]
    mu = depth + 2e-5 * np.random.default_rng(1).normal(size=(6, depth.size))
    paths = report_figures.plot(tmp_path, (theta[:6], mu, np.arange(6) < 5))
    assert [p.name for p in paths] == ["corner.png", "tp_posterior.png", "spectrum_fit.png"]
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in paths)


def test_derived_quantiles_are_weighted(tmp_path, monkeypatch):
    """90% of the weight on the third of the rows at lnZ = -1: every quantile sits there, not at the majority's +1."""
    _run_dir(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    theta = MEAN + np.zeros((300, len(NAMES)))
    theta[:100, 0], theta[100:, 0] = -1.0, 1.0
    w = np.r_[np.full(100, 0.9 / 100), np.full(200, 0.1 / 200)]
    np.savez(tmp_path / "nautilus_posterior.npz", param_names=np.asarray(NAMES), theta=theta, log_w=np.log(w))
    z_o = NR.derived_quantities(tmp_path)["metallicity_O_H_solar"]
    assert z_o == pytest.approx([10.0 * np.exp(-1.0)] * 3, rel=1e-12)


@pytest.mark.skipif(importlib.util.find_spec("exojax") is None, reason="needs exojax (the Guillot profile)")
@pytest.mark.parametrize(
    "over",
    [
        {},
        {"tp_infer_gamma": False},
        {"tp_infer_Tint": True},
        {"tp_infer_Tint": True, "tp_deep_param": "T1bar", "tp_deep_p_bar": 1.0},
        {"tp_infer_Tint": True, "tp_deep_param": "T1bar", "tp_deep_p_bar": 3.0, "tp_adiabat": True},
    ],
)
def test_the_reported_tp_curve_is_the_model_s_profile(over):
    """The report's numpy twin against the fitted evaluator, on every T-P option."""
    from vulcan_forward import vulcan_chem  # noqa: F401  # before exojax
    import jax

    from retrieval_framework import tp_profile

    cfg = C.Config(rp_cm=1.0, rstar_cm=10.0, **over)
    specs = C.specs_from_config(cfg)
    names = [s.name for s in specs]
    values = {
        "Tirr": 1400.0,
        "log10kappa": -2.2,
        "log10gamma": -1.1,
        "Tint": 250.0,
        "T1bar": 2500.0,
        "log10P_rcb": 0.2,
    }
    t = np.array([values.get(n, 0.0) for n in names])
    tp = t[[names.index(s.name) for s in specs if s.kind == "tp"]]
    model = tp_profile.build_tp_model(cfg)
    p_bar = np.logspace(-8.0, 1.0, 40)
    want = np.asarray(model.eval(jax.numpy.asarray(tp), jax.numpy.asarray(p_bar)))
    assert np.all(np.isfinite(want))
    got = NR._tp_curve(t, names, asdict(cfg), p_bar)
    np.testing.assert_allclose(got, want, rtol=1e-11, atol=0)
    assert NR._tp_params(t, names, asdict(cfg))[3] == pytest.approx(
        float(model.tint_of(jax.numpy.asarray(tp))), rel=1e-11
    )


def test_the_reported_kzz_follows_the_power_law_anchor_and_slope(tmp_path, monkeypatch):
    """A base of 1e9 at 0.1 bar, sloped 0.5: at 1 mbar the law gives 10^9 x 10^(0.5 x 2) = 10^10, through the report."""
    _run_dir(tmp_path)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    # the stub's base depends on the level it is read at, so reading it at the wrong level shows
    monkeypatch.setattr(NR, "_kzz_at_ref", lambda _cfg, p_bar=NR.KZZ_P_REF_BAR: 1.0e9 * (p_bar / 0.1))
    cfgj = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**cfgj, "kzz_mode": "powerlaw", "kzz_p_ref_bar": 0.1}))
    names = [*NAMES, "kzz_slope"]
    theta = np.tile(np.r_[MEAN, 0.5], (40, 1))
    theta[:, names.index("lnKzz")] = 0.0
    np.savez(
        tmp_path / "nautilus_posterior.npz",
        param_names=np.asarray(names),
        theta=theta,
        log_w=np.full(40, -np.log(40.0)),
    )
    got = NR.derived_quantities(tmp_path)["log10_kzz_ref_cgs"]
    np.testing.assert_allclose(got, 10.0, rtol=1e-12, atol=0)
    # at the default anchor the slope tilts nothing, and lnKzz scales it
    at_default = NR._log10_kzz_at_report_level(
        {**cfgj, "kzz_mode": "powerlaw"}, np.array([np.log(10.0)]), np.array([0.5])
    )
    np.testing.assert_allclose(at_default, np.log10(1.0e9 * (NR.KZZ_P_REF_BAR / 0.1) * 10.0), rtol=1e-12, atol=0)
    got = NR._log10_kzz_at_report_level({**cfgj, "kzz_mode": "scale"}, np.array([np.log(100.0)]), None)
    np.testing.assert_allclose(got, np.log10(1.0e9 * (NR.KZZ_P_REF_BAR / 0.1) * 100.0), rtol=1e-12, atol=0)
