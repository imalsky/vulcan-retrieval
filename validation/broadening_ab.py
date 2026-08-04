#!/usr/bin/env python3
"""A/B: terrestrial-air vs H2/He pressure broadening on the binned spectrum.

HITRAN's default gamma_air/n_air describe broadening by terrestrial air; the
W39b envelope is H2/He. exojax_rt's broadening="h2he" swaps in HITRAN's
planetary-broadener H2/He widths where the database provides them (per-line
coverage is printed per molecule; uncovered lines keep air widths for that
partner). This script builds the RT both ways over the production band on the
SAME converged chemistry and reports the binned-depth difference -- the number
that decides whether "air" is acceptable for a given precision target.

The h2he build downloads into separate <db>_h2he cache dirs on first use
(network required once). Run:

    python validation/broadening_ab.py

Reported, not gated: the acceptable difference depends on the target precision.
As a guide, differences below ~5 ppm are invisible under the CM24 error bars;
tens of ppm mean the production runs should switch to broadening="h2he" (or the
molecule's ExoMol list with its own H2/He .broad files).
"""
from __future__ import annotations

import sys
import time

import numpy as np

# PYTHONSAFEPATH strips the script's own directory from sys.path in some
# sandboxes, so be explicit rather than relying on it.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import _artifact  # noqa: E402


BIN_R = 100.0
# Guide, not a gate: below this the air-vs-H2/He difference is invisible under
# the Carter & May 2024 error bars this retrieval fits.
DECISION_INVISIBLE_PPM = 5.0
BAND = (1900.0, 9900.0)


def make_r_bins(wl_lo, wl_hi, R):
    n = max(2, int(np.ceil(np.log(wl_hi / wl_lo) * R)))
    return np.geomspace(wl_lo, wl_hi, n + 1)


def bin_trapz(wl, y, edges):
    w = np.empty_like(wl)
    w[1:-1] = 0.5 * (wl[2:] - wl[:-2]); w[0] = wl[1] - wl[0]; w[-1] = wl[-1] - wl[-2]
    idx = np.digitize(wl, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[b] = float(np.sum(w[sel] * y[sel]) / np.sum(w[sel]))
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-artifact", action="store_true",
                    help="skip writing the provenance-bearing result under "
                         "validation/results/ (exploration only)")
    args = ap.parse_args()

    from retrieval_framework.forward import config
    from vulcan_forward import interp_map
    # import order is load-bearing: vulcan_chem before exojax
    from vulcan_forward import vulcan_chem
    from vulcan_forward import exojax_rt
    import jax.numpy as jnp

    profile = dict(config.FULL)
    profile.update(nz=50, count_max=5000, dt_max=1.0e11,
                   abundance_mode="elemental", co_mode="fixed_O",
                   molecules=["H2O", "CO2", "CO", "CH4", "SO2", "HCN", "C2H2", "H2S"],
                   nu_min=BAND[0], nu_max=BAND[1], nu_pts=1652,
                   art_nlayer=60, use_rayleigh=True)
    chem = vulcan_chem.build_chem_model(profile)
    y = chem.converged_y(jnp.zeros(4, dtype=jnp.float64))
    ymix = y / jnp.sum(y, axis=1, keepdims=True)
    he, h2 = chem.sidx["He"], chem.sidx[config.BULK_H2_VULCAN]
    edges = make_r_bins(1e4 / BAND[1], 1e4 / BAND[0], BIN_R)

    binned = {}
    for mode in ("air", "h2he"):
        t0 = time.time()
        prof = dict(profile); prof["broadening"] = mode
        rt = exojax_rt.build_rt_model(prof)
        to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)
        vmr = {k: to_art(ymix[:, chem.sidx[config.MOLECULES[k]["vulcan"]]])
               for k in rt.molecules}
        d = rt.transmission_depth(vmr, to_art(ymix[:, h2]),
                                  to_art(jnp.asarray(chem.T_base)),
                                  to_art(ymix @ chem.species_masses),
                                  vmr_he=to_art(ymix[:, he]))
        wl = np.asarray(rt.wl_um, np.float64)
        o = np.argsort(wl)
        binned[mode] = bin_trapz(wl[o], np.asarray(d, np.float64)[o], edges)
        print(f"[ab] {mode} done ({time.time()-t0:.0f}s)", flush=True)

    da, db = binned["air"], binned["h2he"]
    m = np.isfinite(da) & np.isfinite(db)
    diff = 1e6 * (db[m] - da[m])
    centers = np.sqrt(edges[:-1] * edges[1:])[m]
    i = int(np.argmax(np.abs(diff)))
    max_ppm = float(np.max(np.abs(diff)))
    med_ppm = float(np.median(np.abs(diff)))
    print("\n==== air vs h2he broadening, binned R=100 depth ====")
    print(f"max |Delta| = {max_ppm:.2f} ppm at {centers[i]:.2f} um; "
          f"median |Delta| = {med_ppm:.2f} ppm")
    print("(guide: <5 ppm invisible under CM24 errors; tens of ppm -> switch the "
          "production broadening to 'h2he' / ExoMol)")

    # This A/B has no pass/fail gate: it is a DECISION INPUT. The production
    # atmosphere is H2/He, so air widths are a known approximation, and the
    # question is only whether the approximation is visible at the quoted
    # precision. Recorded as REPORT with an explicit recommendation so the
    # decision is made from a measurement rather than from habit.
    if max_ppm < DECISION_INVISIBLE_PPM:
        rec = (f"max {max_ppm:.2f} ppm is below the {DECISION_INVISIBLE_PPM:.0f} "
               "ppm guide, i.e. invisible under the CM24 error bars. Keeping "
               "air-broadened HITRAN widths is defensible for THIS band and "
               "precision; the limitation stays documented.")
    else:
        rec = (f"max {max_ppm:.2f} ppm is at or above the "
               f"{DECISION_INVISIBLE_PPM:.0f} ppm guide, so the broadening "
               "choice is visible in the binned spectrum. Select the existing "
               "broadening='h2he' mode for production molecules that have "
               "coverage, and fail loudly for a requested molecule that does "
               "not -- never mix a claim of H2/He broadening with all-air data.")
    print(f"\nRECOMMENDATION: {rec}")

    if not args.no_artifact:
        _artifact.emit(
            name="broadening_ab",
            title="Air vs H2/He line broadening, binned R=100 transit depth",
            measurements=[
                {"name": "max |Delta binned depth| (h2he - air)",
                 "value": f"{max_ppm:.2f} ppm at {centers[i]:.2f} um",
                 "value_raw": max_ppm, "unit": "ppm",
                 "gate": f"guide: < {DECISION_INVISIBLE_PPM:.0f} ppm invisible",
                 "decisive": True, "passed": None},
                {"name": "median |Delta binned depth|",
                 "value": f"{med_ppm:.2f} ppm", "value_raw": med_ppm,
                 "unit": "ppm", "gate": "(informational)",
                 "decisive": False, "passed": None},
            ],
            status="REPORT",
            summary=rec,
            resolved_config=profile,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
