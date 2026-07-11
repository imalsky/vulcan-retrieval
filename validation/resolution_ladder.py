#!/usr/bin/env python3
"""Native-spectral-resolution convergence of the BINNED transit depth.

Transmission is nonlinear in optical depth (T = e^-tau), so sampling the opacity
on a coarse native grid and averaging into R~100 bins is NOT automatically the
correct binned spectrum -- narrow saturated lines and low-opacity windows must be
resolved (or treated with a validated k-distribution). The production choice
(nu_pts=1652, R~1000 over the retrieval band) was set by GPU gradient MEMORY, not
by a convergence test; this script supplies the test.

Method: converge the W39b chemistry ONCE (baseline theta), then rebuild the RT at
each rung of a nu_pts ladder over the production band, bin every native spectrum
onto the SAME R=100 bins, and difference against the finest rung. Optionally
convolve with a Gaussian LSF (--lsf-r) before binning to show LSF insensitivity
at R=100 products, and optionally check a chemistry Jacobian column (--jacobian:
d(binned depth)/d lnZ via a warm-started jvp per rung).

Run on the GPU node (primal RT only -- the vjp memory wall does not apply; the
finest rung is still line-count-heavy):

    python validation/resolution_ladder.py --ladder 1652 3304 6608 13216

PASS gates (from the review): binned-depth change < 5 ppm between the top two
rungs; Jacobian direction change < 1% where the depth response is significant.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

GATE_PPM = 5.0
GATE_JAC_REL = 0.01
BIN_R = 100.0
BAND = (1900.0, 9900.0)          # production retrieval band (cm^-1)


def make_r_bins(wl_lo, wl_hi, R):
    n = max(2, int(np.ceil(np.log(wl_hi / wl_lo) * R)))
    return np.geomspace(wl_lo, wl_hi, n + 1)


def bin_trapz(wl, y, edges):
    """d(lambda)-weighted (local-trapezoid) bin means; NaN where empty."""
    w = np.empty_like(wl)
    w[1:-1] = 0.5 * (wl[2:] - wl[:-2]); w[0] = wl[1] - wl[0]; w[-1] = wl[-1] - wl[-2]
    idx = np.digitize(wl, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.any():
            out[b] = float(np.sum(w[sel] * y[sel]) / np.sum(w[sel]))
    return out


def gaussian_lsf(wl, y, R_lsf):
    """Gaussian LSF of resolving power R_lsf applied in ln-lambda (host-side)."""
    ln = np.log(wl)
    s = 1.0 / (R_lsf * 2.3548200450309493)      # FWHM = 1/R in ln-lambda
    out = np.empty_like(y)
    for i, l0 in enumerate(ln):
        w = np.exp(-0.5 * ((ln - l0) / s) ** 2)
        out[i] = np.sum(w * y) / np.sum(w)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=int, nargs="+", default=[1652, 3304, 6608])
    ap.add_argument("--lsf-r", type=float, default=0.0,
                    help="optional Gaussian LSF resolving power before binning")
    ap.add_argument("--jacobian", action="store_true",
                    help="also compare d(binned)/dlnZ per rung (jvp; expensive)")
    args = ap.parse_args()

    from retrieval_framework.forward import config
    from retrieval_framework.forward import interp_map
    # import order is load-bearing: vulcan_chem before exojax
    from retrieval_framework.forward import vulcan_chem
    from retrieval_framework.forward import exojax_rt
    import jax
    import jax.numpy as jnp

    profile = dict(config.FULL)
    profile.update(nz=50, count_max=5000, dt_max=1.0e11,
                   abundance_mode="elemental", co_mode="fixed_O",
                   molecules=["H2O", "CO2", "CO", "CH4", "SO2", "HCN", "C2H2", "H2S"],
                   nu_min=BAND[0], nu_max=BAND[1], art_nlayer=60,
                   use_rayleigh=True)
    chem = vulcan_chem.build_chem_model(profile)
    theta0 = jnp.zeros(4, dtype=jnp.float64)
    y0 = chem.converged_y(theta0)
    he, h2 = chem.sidx["He"], chem.sidx[config.BULK_H2_VULCAN]

    edges = make_r_bins(1e4 / BAND[1], 1e4 / BAND[0], BIN_R)
    results = {}
    for nu_pts in sorted(args.ladder):
        t0 = time.time()
        prof = dict(profile); prof["nu_pts"] = int(nu_pts)
        rt = exojax_rt.build_rt_model(prof)
        to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)

        def depth_of(y):
            ymix = y / jnp.sum(y, axis=1, keepdims=True)
            mmw = to_art(ymix @ chem.species_masses)
            vmr = {k: to_art(ymix[:, chem.sidx[config.MOLECULES[k]["vulcan"]]])
                   for k in rt.molecules}
            T_art = to_art(jnp.asarray(chem.T_base))
            return rt.transmission_depth(vmr, to_art(ymix[:, h2]), T_art, mmw,
                                         vmr_he=to_art(ymix[:, he]))

        d = np.asarray(depth_of(y0), np.float64)
        wl = np.asarray(rt.wl_um, np.float64)
        o = np.argsort(wl); wl, d = wl[o], d[o]
        if args.lsf_r > 0:
            d = gaussian_lsf(wl, d, args.lsf_r)
        binned = bin_trapz(wl, d, edges)
        entry = dict(binned=binned)
        if args.jacobian:
            def f(th):
                return depth_of(chem.converged_y(th, warm_y=y0,
                                                 lnZ_ref=0.0, c_o_ref=0.0))
            _, jd = jax.jvp(f, (theta0,), (jnp.array([1.0, 0.0, 0.0, 0.0]),))
            jd = np.asarray(jd, np.float64)[o]
            entry["jac_lnZ"] = bin_trapz(wl, jd, edges)
        results[nu_pts] = entry
        print(f"[ladder] nu_pts={nu_pts}: native R~{nu_pts / np.log(BAND[1]/BAND[0]):.0f}, "
              f"{time.time()-t0:.0f}s", flush=True)

    rungs = sorted(results)
    ok = True
    print("\n==== binned-depth convergence (vs next rung) ====")
    for a, b in zip(rungs[:-1], rungs[1:]):
        da, db = results[a]["binned"], results[b]["binned"]
        m = np.isfinite(da) & np.isfinite(db)
        dppm = 1e6 * np.max(np.abs(da[m] - db[m]))
        print(f"nu_pts {a} -> {b}: max |Delta binned depth| = {dppm:.2f} ppm")
        if b == rungs[-1]:
            ok &= dppm < GATE_PPM
    if args.jacobian:
        print("\n==== Jacobian (dlnZ) convergence ====")
        a, b = rungs[-2], rungs[-1]
        ja, jb = results[a]["jac_lnZ"], results[b]["jac_lnZ"]
        m = np.isfinite(ja) & np.isfinite(jb) & (np.abs(jb) > 0.01 * np.nanmax(np.abs(jb)))
        jrel = float(np.max(np.abs(ja[m] - jb[m]) / np.abs(jb[m])))
        print(f"nu_pts {a} -> {b}: max rel Jacobian change (significant bins) = {jrel:.3%}")
        ok &= jrel < GATE_JAC_REL
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} (top-rung gate {GATE_PPM} ppm"
          + (f", Jacobian {GATE_JAC_REL:.0%}" if args.jacobian else "") + ")")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
