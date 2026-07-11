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

BIN_R = 100.0
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
    from retrieval_framework.forward import config
    from retrieval_framework.forward import interp_map
    # import order is load-bearing: vulcan_chem before exojax
    from retrieval_framework.forward import vulcan_chem
    from retrieval_framework.forward import exojax_rt
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
    print("\n==== air vs h2he broadening, binned R=100 depth ====")
    print(f"max |Delta| = {np.max(np.abs(diff)):.2f} ppm at {centers[i]:.2f} um; "
          f"median |Delta| = {np.median(np.abs(diff)):.2f} ppm")
    print("(guide: <5 ppm invisible under CM24 errors; tens of ppm -> switch the "
          "production broadening to 'h2he' / ExoMol)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
