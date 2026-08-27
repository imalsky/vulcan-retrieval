#!/usr/bin/env python3
"""Model-top convergence of the production forward.

Chemistry and RT grids both end at the model top (constants.ART_PTOP_BAR, 1e-9 bar;
vulcan_chem sets the chemistry P_t from it, interp_map refuses a clamped top).
This script extends BOTH grids one decade higher at the same layers per decade,
solves the chemistry there, and compares the R=100 binned depth with production.

    python validation/top_pressure_ladder.py

PASS gate: |Delta binned depth| < 5 ppm. The former one-decade constant-VMR
clamp above a 1e-7 bar chemistry top measured 73.47 ppm against chemistry
solved there (2026-08-27 artifact) and was replaced by extending the grid.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# PYTHONSAFEPATH strips the script's own directory from sys.path in some
# sandboxes, so be explicit rather than relying on it.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import _artifact  # noqa: E402

GATE_PPM = 5.0
BIN_R = 100.0
BAND = (1900.0, 9900.0)


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
    ap.add_argument("--no-artifact", action="store_true",
                    help="skip writing the provenance-bearing result under "
                         "validation/results/ (exploration only)")
    args = ap.parse_args()

    from retrieval_framework.forward import config
    from vulcan_forward import interp_map
    # import order is load-bearing: vulcan_chem before exojax
    from vulcan_forward import vulcan_chem
    from vulcan_forward import exojax_rt

    prod = dict(config.FULL)
    prod.update(nz=62,   # = runs/w39b_smc_retrieval/case.py gpu_config nz; keep in step
                count_max=5000, dt_max=1.0e11,
                abundance_mode="elemental", co_mode="fixed_O",
                molecules=["H2O", "CO2", "CO", "CH4", "SO2", "HCN", "C2H2", "H2S"],
                nu_min=BAND[0], nu_max=BAND[1],
                art_nlayer=67, art_ptop_bar=config.ART_PTOP_BAR, use_rayleigh=True)
    edges = _artifact.make_r_bins(1e4 / BAND[1], 1e4 / BAND[0], BIN_R)

    chem = vulcan_chem.build_chem_model(prod)
    rt = exojax_rt.build_rt_model(prod)
    p_top, p_btm = float(np.min(chem.p_bar)), float(np.max(chem.p_bar))
    wl, d = binned_depth(chem, rt, config, interp_map)
    b_prod = _artifact.bin_trapz(wl, d, edges)
    print(f"[topP] production: model top {p_top:.0e} bar, nz={prod['nz']}, "
          f"art_nlayer={prod['art_nlayer']}", flush=True)

    # one decade higher on BOTH grids, same layers per decade
    f = (np.log10(p_btm / p_top) + 1.0) / np.log10(p_btm / p_top)
    ext = dict(prod, nz=int(round(prod["nz"] * f)),
               art_nlayer=int(round(prod["art_nlayer"] * f)),
               art_ptop_bar=p_top / 10.0)
    t0 = time.time()
    chem_x = vulcan_chem.build_chem_model(ext)
    rt_x = exojax_rt.build_rt_model(ext)
    wl, d = binned_depth(chem_x, rt_x, config, interp_map)
    b_ext = _artifact.bin_trapz(wl, d, edges)
    print(f"[topP] extended: model top {ext['art_ptop_bar']:.0e} bar, nz={ext['nz']}, "
          f"art_nlayer={ext['art_nlayer']} ({time.time() - t0:.0f}s)", flush=True)

    m = np.isfinite(b_prod) & np.isfinite(b_ext)
    dppm = 1e6 * np.max(np.abs(b_ext[m] - b_prod[m]))
    ok = bool(dppm < GATE_PPM)
    msg = (f"PASS -- the {p_top:.0e} bar model top is converged at the quoted precision"
           if ok else
           f"FAIL -- the {p_top:.0e} bar model top is NOT converged; a higher top needs "
           "a measured T-P/Kzz there, not a constant fill")
    print(f"\nmax |Delta binned depth| = {dppm:.2f} ppm  (gate {GATE_PPM} ppm)")
    print(f"\nVERDICT: {msg}")
    if not args.no_artifact:
        _artifact.emit(
            name="top_pressure_ladder",
            title="Model top: production vs one decade higher on both grids",
            measurements=[{
                "name": f"production ({p_top:.0e} bar) vs extended ({p_top / 10:.0e} bar) model top",
                "value": f"{dppm:.2f} ppm", "value_raw": float(dppm), "unit": "ppm",
                "gate": f"< {GATE_PPM} ppm", "decisive": True, "passed": ok,
            }],
            status="PASS" if ok else "FAIL", summary=msg,
            resolved_config={"production": prod, "extended": ext},
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
