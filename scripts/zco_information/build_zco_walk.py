"""Build the Gaussian-validity-walk data for the WASP-39b Z / C/O geometry figure (fig 3).

The Fisher ellipse assumes the log-likelihood is locally quadratic. Vallisneri (2008)'s
standard validity test: walk the REAL forward model along the most-degenerate (flattest)
eigendirection of the reported (lnZ, ln C/O) marginal covariance, and compare the true
delta-chi^2 to the Fisher parabola (delta-chi^2 = s^2 in units of the 1-sigma step). Where
they diverge, the ellipse is not trustworthy.

At each step we perturb only (lnZ, ln C/O) -- along the marginal major axis -- run the
kinetics->transmission forward model (reanchor_atom_ini ON + tight convergence, because
these are FINITE Z/CO steps, not infinitesimal AD), bin to the observed grid, and profile
out the LINEAR nuisances actually in the design (lnR0 + inter-instrument offsets) by least
squares against their fiducial Jacobian columns. The nonlinear nuisances (lnKzz, T_int)
are held at the fiducial (stated caveat). Cache -> data/zco_walk.npz for fig_zco_geometry.

Run (base env, ~15-20 heavy forward runs, from the repo root):
    python scripts/zco_information/build_zco_walk.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
# import order is load-bearing: vulcan_chem before exojax (sets env + jax x64)
from retrieval_framework.forward import vulcan_chem
import jax.numpy as jnp
from retrieval_framework.forward import exojax_rt
from retrieval_framework.forward import interp_map

import zco_lib  # sibling in zco_information/ (script dir is on sys.path)

OUT = config.OUTPUTS / "zco_walk.npz"
TIER = "P"                       # walk the fiducial (photochemistry-on) tier
STEPS = np.array([-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3], float)  # in sigma
CO_ABS_CAP = 0.50                # fixed-O knob stays valid for dln(C/O) < ~0.57


def main():
    # --- flattest marginal eigendirection from the cached tier-P Fisher + real noise ---
    wl_model, tiers, meta = zco_lib.load_jacobians()
    obs = zco_lib.load_combined()
    des = zco_lib.build_design(tiers[TIER], wl_model, obs)
    F, _ = zco_lib.fisher(des["J"], des["sigma"])
    evals, evecs, C2 = zco_lib.eigendecompose_marginal(F, des["interest"])
    v_flat = evecs[:, 0]                       # smallest eigenvalue => most degenerate
    sig_flat = 1.0 / np.sqrt(evals[0])         # 1-sigma step length along it (in ln units)
    # orient so the lnZ component is >= 0 (nicer labels)
    if v_flat[0] < 0:
        v_flat = -v_flat
    print(f"[walk] flattest marginal eigendir (dlnZ,dlnCO)=({v_flat[0]:+.3f},{v_flat[1]:+.3f})  "
          f"sigma_step={sig_flat:.4f} ln  (={sig_flat/np.log(10):.4f} dex); "
          f"axis ratio sqrt(l_max/l_min)={np.sqrt(evals[1]/evals[0]):.2f}", flush=True)

    # cap the step so the largest |dln C/O| stays inside the fixed-O knob's valid range
    max_s = min(np.max(np.abs(STEPS)), CO_ABS_CAP / (abs(v_flat[1]) * sig_flat + 1e-12))
    steps = STEPS[np.abs(STEPS) <= max_s + 1e-9]
    print(f"[walk] using steps (sigma): {list(steps)}  (cap {max_s:.2f} sigma from CO validity)",
          flush=True)

    # --- forward model with FINITE-step-safe chemistry (reanchor + tight convergence) ---
    prof = dict(config.FULL, co_mode="fixed_O", nz=150, yconv_cri=1.0e-4, yconv_min=1.0e-4,
                slope_cri=1.0e-7, count_min=300, count_max=12000, reanchor_atom_ini=True,
                molecules=["H2O", "CO2", "CO", "CH4", "SO2"],
                nu_min=1893.0, nu_max=10000.0, nu_pts=6000, art_nlayer=60,
                **zco_lib_tier_cfg(TIER))
    t0 = time.time()
    chem = vulcan_chem.build_chem_model(prof)
    trt = exojax_rt.build_rt_model(prof)
    to_art = interp_map.make_to_art(chem.p_bar, trt.p_art_bar)
    mol_cols = {k: chem.sidx[config.MOLECULES[k]["vulcan"]] for k in trt.molecules}
    h2 = chem.sidx[config.BULK_H2_VULCAN]
    he = chem.sidx["He"]        # H2-He CIA partner (required by the RT)
    T_base = jnp.asarray(chem.T_base); masses = chem.species_masses

    def depth_of(theta):
        ymix = chem.converged_ymix(jnp.asarray(theta, dtype=jnp.float64))
        T_art = to_art(T_base + theta[3]); mmw_art = to_art(ymix @ masses)
        vmr = {k: to_art(ymix[:, c]) for k, c in mol_cols.items()}
        return np.asarray(trt.transmission_depth(vmr, to_art(ymix[:, h2]), T_art,
                                                 mmw_art, vmr_he=to_art(ymix[:, he])))

    wl_m = np.asarray(trt.wl_um)
    depth0 = depth_of(config.THETA0)
    print(f"[walk] built + primal in {time.time()-t0:.0f}s; "
          f"depth {depth0.min()*1e6:.0f}-{depth0.max()*1e6:.0f} ppm", flush=True)

    # --- profile ALL nuisances (linearly) at each step, matching the marginal Fisher ---
    # CRITICAL: the s^2 prediction comes from the FULL marginal Fisher (lnKzz, T_int, lnR0,
    # offsets all marginalized). So the walk's chi^2 must marginalize the SAME nuisances,
    # else a constant scale mismatch (~8x, since lnKzz/T_int/lnR0 held fixed is much tighter)
    # masquerades as nonlinearity. We profile every nuisance COLUMN of the binned design
    # linearly (their fiducial gradients) -- identical to how the Fisher treats them -- so any
    # residual deviation from s^2 is the genuine nonlinearity in the walked (Z, C/O) direction.
    des = zco_lib.build_design(tiers[TIER], wl_model, obs)
    keep = des["keep"]
    sigma = des["sigma"]; W = 1.0 / sigma ** 2
    nuis = [c for c in range(des["J"].shape[1]) if c not in des["interest"]]
    Nmat = des["J"][:, nuis]                                    # lnKzz, T_int, lnR0, offsets
    print(f"[walk] profiling nuisances: {[des['keys'][c] for c in nuis]}", flush=True)
    NtWN_inv = np.linalg.inv(Nmat.T @ (W[:, None] * Nmat))

    def profiled_chi2(dm):
        """min over ALL linear nuisances a of sum W (dm - N a)^2, on the observed grid."""
        NtWd = Nmat.T @ (W * dm)
        return float((W * dm ** 2).sum() - NtWd @ (NtWN_inv @ NtWd))

    def bin_depth(dvec):
        _, b = zco_lib.bin_to_obs(wl_m, dvec[:, None], obs)
        return b[keep, 0]

    chi2_true, chi2_quad, dm_all = [], [], []
    for s in steps:
        d = s * sig_flat * v_flat                    # (dlnZ, dlnCO)
        theta = [float(d[0]), float(d[1]), 0.0, 0.0]
        tk = time.time()
        dm = bin_depth(depth_of(theta) - depth0)
        c2 = profiled_chi2(dm)
        chi2_true.append(c2); chi2_quad.append(float(s) ** 2); dm_all.append(dm)
        print(f"[walk] s={s:+.1f}  dlnZ={d[0]:+.3f} dlnCO={d[1]:+.3f}  "
              f"chi2_true={c2:.2f} vs s^2={s*s:.2f}  ({time.time()-tk:.0f}s)", flush=True)

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, steps=steps, chi2_true=np.asarray(chi2_true),
             chi2_quad=np.asarray(chi2_quad), v_flat=v_flat, sig_flat=sig_flat,
             evals=evals, evecs=evecs, C2=C2, tier=TIER,
             dm_all=np.asarray(dm_all), sigma=sigma)   # dm_all: re-profile without rerun
    print(f"[walk] DONE -> {OUT}", flush=True)
    return 0


def zco_lib_tier_cfg(tier):
    # Shared with the Jacobian builder: tier config merged over the pinned
    # validated-baseline recipe (vm_mol off, dt_max cap, photo-off convergence
    # recipe) so the walk runs the SAME chemistry scheme as the cached Jacobians.
    from build_zco_jacobians import tier_profile_overrides
    return tier_profile_overrides(tier)


if __name__ == "__main__":
    sys.exit(main())
