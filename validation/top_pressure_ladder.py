#!/usr/bin/env python3
"""Model-top convergence: clamped upper-atmosphere extension vs REAL chemistry.

The production ART grid tops at 1e-8 bar while the chemistry tops at 1e-7 bar;
the decade in between uses the constant-VMR/isothermal clamp (interp_map). That
choice prevents strong bands from saturating into a model-top wall, but a clamp
is an assumption, not chemistry -- photochemical abundances can genuinely vary at
sub-microbar pressures. This script measures both effects on the binned depth:

  A. clamp ladder: same chemistry (P_t = 1e-7 bar), ART top in {1e-7, 1e-8, 1e-9}
     -- how much the extension choice itself moves the R=100 spectrum;
  B. --extend-chem: chemistry actually SOLVED to 1e-8 bar (cfg P_t=0.01 dyn/cm2,
     proportionally deeper nz), ART top 1e-8 -- clamped-vs-real chemistry over
     the extra decade, the review's decisive comparison.

Run on the GPU node (two chemistry solves with --extend-chem):

    python validation/top_pressure_ladder.py --extend-chem

PASS gate: |Delta binned depth| < 5 ppm between the clamped production choice
(top 1e-8) and the extended-chemistry run; the pure clamp ladder is reported for
context (1e-7 vs 1e-8 is EXPECTED to differ -- that is the wall being removed).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

GATE_PPM = 5.0
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


def binned_depth(chem, rt, config, interp_map):
    import jax.numpy as jnp
    to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)
    y = chem.converged_y(jnp.zeros(4, dtype=jnp.float64))
    ymix = y / jnp.sum(y, axis=1, keepdims=True)
    he, h2 = chem.sidx["He"], chem.sidx[config.BULK_H2_VULCAN]
    vmr = {k: to_art(ymix[:, chem.sidx[config.MOLECULES[k]["vulcan"]]])
           for k in rt.molecules}
    d = rt.transmission_depth(vmr, to_art(ymix[:, h2]),
                              to_art(jnp.asarray(chem.T_base)),
                              to_art(ymix @ chem.species_masses),
                              vmr_he=to_art(ymix[:, he]))
    wl = np.asarray(rt.wl_um, np.float64)
    o = np.argsort(wl)
    return wl[o], np.asarray(d, np.float64)[o]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tops", type=float, nargs="+", default=[1e-7, 1e-8, 1e-9])
    ap.add_argument("--extend-chem", action="store_true")
    args = ap.parse_args()

    from retrieval_framework.forward import config
    from retrieval_framework.forward import interp_map
    # import order is load-bearing: vulcan_chem before exojax
    from retrieval_framework.forward import vulcan_chem
    from retrieval_framework.forward import exojax_rt

    base_profile = dict(config.FULL)
    base_profile.update(nz=50, count_max=5000, dt_max=1.0e11,
                        abundance_mode="elemental", co_mode="fixed_O",
                        molecules=["H2O", "CO2", "CO", "CH4", "SO2", "HCN", "C2H2", "H2S"],
                        nu_min=BAND[0], nu_max=BAND[1], nu_pts=1652,
                        art_nlayer=60, use_rayleigh=True)
    chem = vulcan_chem.build_chem_model(base_profile)
    edges = make_r_bins(1e4 / BAND[1], 1e4 / BAND[0], BIN_R)

    # A: clamp ladder (same chemistry, different ART tops)
    binned = {}
    for top in args.tops:
        t0 = time.time()
        # ART grid construction reads config.ART_PTOP_BAR (module constant);
        # override it for this build only, restoring after.
        old = config.ART_PTOP_BAR
        config.ART_PTOP_BAR = float(top)
        try:
            rt = exojax_rt.build_rt_model(base_profile)
            wl, d = binned_depth(chem, rt, config, interp_map)
        finally:
            config.ART_PTOP_BAR = old
        binned[top] = bin_trapz(wl, d, edges)
        print(f"[topP] ART top {top:.0e} bar done ({time.time()-t0:.0f}s)", flush=True)

    tops = sorted(binned, reverse=True)   # coarse (1e-7) -> deep (1e-9)
    print("\n==== clamp-extension ladder (same chemistry) ====")
    for a, b in zip(tops[:-1], tops[1:]):
        da, db = binned[a], binned[b]
        m = np.isfinite(da) & np.isfinite(db)
        print(f"ART top {a:.0e} -> {b:.0e} bar: max |Delta| = "
              f"{1e6 * np.max(np.abs(da[m] - db[m])):.2f} ppm")

    ok = True
    if args.extend_chem:
        print("\n[topP] solving EXTENDED chemistry to 1e-8 bar ...", flush=True)
        ext = dict(base_profile)
        # deepen the grid by one decade at matching per-decade layer density
        n_dec_base = np.log10(7.6e6 / 0.1)       # cfg P_b=7.6e6, P_t=0.1 dyn/cm2
        nz_ext = int(round(base_profile["nz"] * (n_dec_base + 1.0) / n_dec_base))
        ext.update(nz=nz_ext, cfg_overrides={"P_t": 0.01})
        chem_ext = vulcan_chem.build_chem_model(ext)
        old = config.ART_PTOP_BAR
        config.ART_PTOP_BAR = 1e-8
        try:
            rt = exojax_rt.build_rt_model(ext)
            wl, d = binned_depth(chem_ext, rt, config, interp_map)
        finally:
            config.ART_PTOP_BAR = old
        b_ext = bin_trapz(wl, d, edges)
        b_clamp = binned.get(1e-8)
        m = np.isfinite(b_ext) & np.isfinite(b_clamp)
        dppm = 1e6 * np.max(np.abs(b_ext[m] - b_clamp[m]))
        print("\n==== clamped extension vs REAL chemistry over 1e-7..1e-8 bar ====")
        print(f"max |Delta binned depth| = {dppm:.2f} ppm  (gate {GATE_PPM} ppm)")
        ok = dppm < GATE_PPM
        msg = ("PASS -- the clamp is a faithful stand-in at the quoted precision"
               if ok else
               "FAIL -- extend the production chemistry grid (cfg P_t) instead of clamping")
        print(f"\nVERDICT: {msg}")
    else:
        print("\n(no --extend-chem: clamp ladder reported, decisive test skipped)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
