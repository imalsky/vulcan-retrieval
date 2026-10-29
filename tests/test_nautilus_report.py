"""nautilus_report on a tiny synthetic run_nautilus output directory (no
chemistry): the certificate's fractions and caveats follow from the tally, the
summary and the identity files, and the CLI writes the weighted corner and T-P
figures; given model spectra, the spectrum fit too."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from retrieval_framework import config_schema as C
from retrieval_framework import nautilus_report as NR
from retrieval_framework import provenance

NAMES = ["lnZ", "c_o", "lnKzz", "Tirr", "log10kappa", "log10gamma", "lnR0", "noise_inflation"]
MEAN = np.array([0.2, -0.5, 0.0, 1000.0, -1.0, -1.0, 0.03, 1.2])
SD = np.array([0.1, 0.1, 0.5, 50.0, 0.2, 0.2, 0.005, 0.05])
# 1000 draws: 100 off-window, 45 of the 900 in-window rejected, 855 certified
TALLY = {"n_eval": 1000, "tp_out": 100, "exhausted": 40, "stalled": 5, "nonfinite": 0,
         "n_cold": 8, "cold_certified": 6, "n_warm": 992, "warm_certified": 849}


def _run_dir(out):
    rng = np.random.default_rng(0)
    cfg = asdict(C.Config(molecules=("H2O",), vulcan_cfg_name="W39b",
                          network="thermo/SNCHO_photo_network_2025.txt", cold_lanes=8))
    cfg.update(inferred_param_names=NAMES, inferred_param_labels=NAMES,
               inferred_param_truth=[float("nan")] * len(NAMES))
    (out / "config.json").write_text(json.dumps(cfg, default=str))
    (out / provenance.MANIFEST_FILE).write_text(json.dumps({"manifest_version": 2, "code": {}}))
    digest = provenance.archived_manifest_digest(out)
    (out / "target_digest.txt").write_text(f"{digest}|warm|n_batch=8\n")
    (out / "nautilus_tally.json").write_text(json.dumps(TALLY))
    (out / "nautilus_summary.json").write_text(json.dumps({
        "logZ_box": -12.3, "n_like": 990, "n_eff": 1234.0, "n_live": 100, "n_batch": 8,
        "n_update": 50, "f_live": 0.03, "enlarge_per_dim": 1.3, "starts": "anchored warm (growing)",
        "rejections_and_starts": TALLY, "seed": 1, "chem_rtol": 0.2}))
    theta = MEAN + SD * rng.normal(size=(400, len(NAMES)))
    log_w = rng.normal(size=400)
    np.savez(out / "nautilus_posterior.npz", param_names=np.asarray(NAMES), theta=theta,
             log_w=log_w - np.log(np.exp(log_w).sum()))
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
    np.savez(out / "rejected" / "rejected_00000.npz", z=zr, cls=np.r_[[1] * 10, [0] * 5],
             acc=np.zeros(15, int), warm=np.int8(1))
    wl = np.linspace(3.0, 5.0, 20)
    np.savez(out / "observations.npz", wl=wl, depth=0.02 + 1e-4 * np.sin(wl),
             sigma=np.full(20, 1e-4), group=np.asarray(["NIRISS"] * 8 + ["G395H"] * 12))
    return theta, 0.02 + 1e-4 * np.sin(wl)


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
    assert cert["result"] == {"logZ_box": -12.3, "logZ_err": None, "n_eff": 1234.0,
                              "n_like": 990}
    assert cert["starts"]["mode"] == "warm" and cert["chemistry"]["rtol"] == 0.2
    # the derived quantities: weighted 16/50/84% of what the paper quotes
    der = cert["derived"]
    z_o = der["metallicity_O_H_solar"]
    assert len(z_o) == 3 and z_o[0] < z_o[1] < z_o[2]
    assert abs(z_o[1] - 10.0 * np.exp(MEAN[0])) < 0.5          # the baseline is 10x solar
    assert der["metallicity_total_solar"][1] < z_o[1]           # C/O below solar lowers total metals
    assert der["S_to_O_solar"] is None                          # sulfur was not freed in this run
    # the Kzz block: weights at multiples of the prior floor, the absolute Kzz on the
    # case's own profile, and the x10 ceiling only on W39b's; a constant-Kzz case
    # (1e9) shifts the absolute Kzz by its base and has no ceiling
    assert der["truncated_kzz_le_x10"]["weight"] == pytest.approx(
        1.0 - der["kzz_weight_above_x10"])
    assert der["kzz_weight_below_1.5x_floor"] <= der["kzz_weight_below_2x_floor"]
    cfgj = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(
        json.dumps({**cfgj, "vulcan_cfg_name": "data/w107b/W107b.yaml"}))
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    d107 = NR.derived_quantities(tmp_path)
    assert not any("x10" in k for k in d107)
    np.testing.assert_allclose(
        np.subtract(d107["log10_kzz_ref_cgs"], der["log10_kzz_ref_cgs"]),
        9.0 - np.log10(NR._kzz_at_ref({"vulcan_cfg_name": "W39b"})), rtol=0, atol=1e-12)
    assert der["T_deep_p_bar"] == 1.0 and len(der["T_deep_K"]) == 3 == len(der["Tint_K"])
    cond = cert["conditioning"]
    assert cond["rejected_of_in_window"] == pytest.approx(10 / 100)
    kzz = cond["lnKzz"]
    assert kzz["n_in_window"] == [19] + [9] * 9
    assert kzz["rejected_frac"] == pytest.approx([10 / 19] + [0.0] * 9)
    assert sum(kzz["posterior_weight"]) == pytest.approx(1.0)
    assert "lnZ" in cond and "Tint" not in cond                # Tint is not inferred here
    text = " ".join(cert["caveats"])
    for phrase in ("5.0% of the draws", "no logZ error", "Warm starts", "Lane queue",
                   "counted twice", "52.6% of the in-window draws in the lowest bin"):
        assert phrase in text, phrase
    assert "different run" not in text                 # the manifest hashes to the digest
    (tmp_path / "target_digest.txt").write_text("0" * 64 + "|warm\n")
    assert any("different run" in c for c in NR.certificate(tmp_path)["caveats"])

    assert sorted(p.name for p in (tmp_path / "plots").iterdir()) == [
        "corner.png", "tp_posterior.png"]
    mu = depth + 2e-5 * np.random.default_rng(1).normal(size=(6, depth.size))
    paths = NR.plot(tmp_path, (theta[:6], mu, np.arange(6) < 5))
    assert [p.name for p in paths] == ["corner.png", "tp_posterior.png", "spectrum_fit.png"]
    assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in paths)
