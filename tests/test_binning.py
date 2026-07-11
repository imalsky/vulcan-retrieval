"""The binning MATRIX must reproduce the d(lambda)-weighted trapezoidal bin average
(zco_lib.bin_to_obs's operation) exactly -- that equivalence is what makes the binned
depth's jvp exact and free."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from retrieval_framework import observations as OBS

# np.trapz was renamed np.trapezoid in NumPy 2.0 (and trapz removed); support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _reference_bin(wl_model, y, lo_all, hi_all):
    """Trapezoidal bin average, the zco_lib.bin_to_obs operation."""
    order = np.argsort(wl_model)
    wl = wl_model[order]
    Y = y[order]
    out = np.full(lo_all.size, np.nan)
    for b in range(lo_all.size):
        lo, hi = lo_all[b], hi_all[b]
        if not (wl[0] <= lo < hi <= wl[-1]):
            continue
        inside = (wl > lo) & (wl < hi)
        x = np.concatenate([[lo], wl[inside], [hi]])
        yy = np.interp(x, wl, Y)
        out[b] = _trapezoid(yy, x) / (hi - lo)
    return out


def test_binning_matrix_matches_trapezoid_reference():
    rng = np.random.default_rng(7)
    wl_model = np.sort(rng.uniform(2.0, 5.2, size=800))
    # smooth + wiggly test spectra
    specs = [
        np.sin(3.0 * wl_model) + 0.2 * wl_model,
        np.exp(-0.5 * ((wl_model - 3.3) / 0.05) ** 2),
        rng.standard_normal(wl_model.size).cumsum() * 1e-3,
    ]
    # observed-like bins, mixed widths, some outside the model span
    lo = np.array([1.8, 2.05, 2.5, 3.29, 3.301, 4.0, 5.15, 5.25])
    hi = np.array([2.1, 2.10, 2.9, 3.31, 3.309, 4.4, 5.19, 5.40])
    obs = dict(wl=0.5 * (lo + hi), wl_lo=lo, wl_hi=hi)

    keep, B = OBS.build_binning_matrix(wl_model, obs)
    # bins crossing the model edge must be dropped
    assert not keep[0] and not keep[-1] and keep[1:-1].all()

    for y in specs:
        ref = _reference_bin(wl_model, y, lo, hi)
        got = B @ y
        assert np.allclose(got, ref[keep], rtol=1e-12, atol=1e-14)


def test_binning_matrix_on_real_cm24_bins():
    """Same equivalence on the actual Carter & May NIRISS+G395H bins, loaded through
    the framework's config-driven product loader (repo data/cm24_wasp39b)."""
    from retrieval_framework.forward import config as fwd_config
    cm24 = fwd_config.DATA_DIR / "cm24_wasp39b"
    cfg = SimpleNamespace(
        obs_dir=cm24,
        obs_products={"NIRISS": ("NIRISS_O1_R100.csv", "NIRISS_O2_R100.csv"),
                      "G395H": ("G395H_NRS1_R100.csv", "G395H_NRS2_R100.csv")},
        combo=("NIRISS", "G395H"), obs_wl_lo=2.02, obs_wl_hi=5.24)
    obs = OBS.load_real_observations(cfg)
    wl_model = np.linspace(2.0, 5.26, 6000)
    y = 0.021 + 1e-3 * np.sin(4.0 * wl_model)
    keep, B = OBS.build_binning_matrix(wl_model, obs)
    # both instrument groups must survive the band cut (29 NIRISS + 59 G395H bins):
    # NIRISS is the reference group and G395H carries the inter-instrument offset
    g = np.asarray(obs["group"])[keep]
    assert (g == "NIRISS").sum() >= 20 and (g == "G395H").sum() >= 50
    ref = _reference_bin(wl_model, y, np.asarray(obs["wl_lo"]), np.asarray(obs["wl_hi"]))
    assert np.allclose(B @ y, ref[keep], rtol=1e-12, atol=1e-14)
    # row weights of a bin average must sum to 1
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-10)


def test_offset_design_groups():
    obs = dict(group=np.array(["NIRISS", "NIRISS", "G395H", "G395H", "G395H"]),
               groups=["NIRISS", "G395H"])
    O = OBS.build_offset_design(obs)
    assert O.shape == (5, 1)
    assert np.array_equal(O[:, 0], [0.0, 0.0, 1.0, 1.0, 1.0])
