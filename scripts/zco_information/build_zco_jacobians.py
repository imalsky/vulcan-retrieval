"""Build the per-tier transmission Jacobians for the WASP-39b Z / C/O Fisher figures.

For each chemistry tier -- E(quilibrium) -> Q(uench, +transport) -> P(hotochem, +photo) --
compute, by forward-mode AD straight through VULCAN-JAX kinetics + ExoJax transmission,

    J_chem = d(depth)/d[ lnZ, dln(C/O)_fixedO, lnKzz, T_int ]     (n_nu, 4)
    J_lnR0 = d(depth)/d(lnR0)   at that tier's converged profiles  (n_nu,)

and cache everything (+ the baseline depth per tier) to data/zco_jacobians.npz for the
pure-numpy Fisher figures (zco_lib.py + fig_zco_*.py).

The three tiers share ONE ExoJax opacity build (RT depends only on the molecule set +
grid, not the chemistry); only the VULCAN-JAX chemistry config differs:
    E : use_photo=False, Kzz->0, use_moldiff=False   (no-transport kinetic fixed point,
        approximately local thermochemical equilibrium; ~3% element drift, see below)
    Q : use_photo=False                              (+ vertical transport => quenching)
    P : use_photo=True                               (+ photochemistry; the fiducial)

C/O uses the fixed-O knob (vulcan_chem co_mode="fixed_O"): dln(C/O) at constant O.
Jacobians are AD at theta0, so the reanchor-atom-ini fix is NOT needed (the atom-loss
snap-back only bites at FINITE Z/CO steps; an infinitesimal AD perturbation stays under
the threshold -- see vulcan_chem._prep). The finite-step validity walk (fig 3) is built
separately by build_zco_walk.py, which DOES turn reanchor on.

Run (base env, from the repo root):
    python scripts/zco_information/build_zco_jacobians.py --smoke        # nz=40, CO-only, fast full-pipeline check
    python scripts/zco_information/build_zco_jacobians.py                # full nz=150 build (slow; ~1-2 h background)
    python scripts/zco_information/build_zco_jacobians.py --tier P       # (re)build one tier only, merge into cache
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
# import order is load-bearing: vulcan_chem before exojax (sets env + jax x64)
from retrieval_framework.forward import vulcan_chem
import jax
import jax.numpy as jnp
from retrieval_framework.forward import exojax_rt
from retrieval_framework.forward import interp_map

OUT = config.OUTPUTS / "zco_jacobians.npz"

# 1.0-5.28 um: covers NIRISS water bands + NIRSpec SO2/CO2/CO. 1 um = H2-H2 CIA short edge.
_ZCO_BAND = dict(nu_min=1893.0, nu_max=10000.0, nu_pts=6000, art_nlayer=60)

TIER_CFG = {
    # Equilibrium: photo off + eddy mixing off (Kzz=0) + molecular diffusion off, so each
    # layer relaxes to LOCAL thermochemical equilibrium (the fixed point of the reaction
    # network). Caveat: with transport off, VULCAN's atom-conservation step (which lives in
    # the diffusion operator) is bypassed, so the converged column drifts ~3% in element
    # totals -- a documented numerical floor of the no-transport limit. (Leaving moldiff ON
    # with Kzz=0 is WORSE: molecular diffusion then separates species vertically, ~7% drift,
    # and is no longer equilibrium.) The 3% fiducial shift does not change the RELATIVE
    # (tier-to-tier) comparison. Kzz=0 makes d/dlnKzz==0, dropped as an uninformative column.
    "E": dict(use_photo=False, zero_Kzz=True, cfg_overrides={"use_moldiff": False}),
    "Q": dict(use_photo=False),
    "P": dict(use_photo=True),
}
TIER_ORDER = ["E", "Q", "P"]
TIER_NAME = {"E": "no-transport (~equilibrium)", "Q": "quench(+transport)", "P": "photochem(+photo)"}


def _profile(smoke: bool, tier: str) -> dict:
    if smoke:
        base = dict(config.SMOKE, co_mode="fixed_O",
                    nu_min=4280.0, nu_max=4360.0, nu_pts=600, art_nlayer=20,
                    molecules=["CO"], nz=40, yconv_cri=1.0e-3)
    else:
        base = dict(config.FULL, co_mode="fixed_O", nz=150, yconv_cri=1.0e-3,
                    molecules=["H2O", "CO2", "CO", "CH4", "SO2"], **_ZCO_BAND)
    return dict(base, **TIER_CFG[tier])


def build_tier(tier: str, trt, smoke: bool):
    """Return (depth (n,), J_chem (n,4), J_lnR0 (n,)) for one chemistry tier."""
    prof = _profile(smoke, tier)
    print(f"\n===== tier {tier} = {TIER_NAME[tier]}  {TIER_CFG[tier]} =====", flush=True)
    t0 = time.time()
    chem = vulcan_chem.build_chem_model(prof)
    to_art = interp_map.make_to_art(chem.p_bar, trt.p_art_bar)
    mol_cols = {k: chem.sidx[config.MOLECULES[k]["vulcan"]] for k in trt.molecules}
    h2 = chem.sidx[config.BULK_H2_VULCAN]
    he = chem.sidx["He"]        # H2-He CIA partner (required by the RT)
    T_base = jnp.asarray(chem.T_base)
    masses = chem.species_masses

    def g(theta):
        ymix = chem.converged_ymix(theta)
        T_art = to_art(T_base + theta[3])
        mmw_art = to_art(ymix @ masses)
        vmr = {k: to_art(ymix[:, c]) for k, c in mol_cols.items()}
        vmr_h2 = to_art(ymix[:, h2])
        vmr_he = to_art(ymix[:, he])
        return (vmr, vmr_h2, T_art, mmw_art, vmr_he)

    def trans_of(gg):
        return trt.transmission_depth(*gg)

    theta0 = jnp.asarray(config.THETA0, dtype=jnp.float64)
    g0 = g(theta0)
    depth0 = np.asarray(trans_of(g0))
    print(f"  primal depth {depth0.min()*1e6:.0f}-{depth0.max()*1e6:.0f} ppm  "
          f"finite={np.all(np.isfinite(depth0))}  ({time.time()-t0:.0f}s)", flush=True)

    # 4 chemistry columns: expensive shared-chemistry jvp, then cheap RT jvp.
    Jc = []
    for k, lab in enumerate(config.THETA_LABELS):
        tk = time.time()
        _, dg = jax.jvp(g, (theta0,), (jnp.zeros(4).at[k].set(1.0),))
        _, dt = jax.jvp(trans_of, (g0,), (dg,))
        dt = np.asarray(dt)
        print(f"  d/d{lab:6s} {time.time()-tk:.0f}s finite={np.all(np.isfinite(dt))} "
              f"max|.|={np.nanmax(np.abs(dt)):.2e}", flush=True)
        Jc.append(dt)
    J_chem = np.stack(Jc, axis=1)

    # lnR0 column: RT-only jvp of the reference-radius-scaled depth at this tier's g0.
    # g0 is (vmr, vmr_h2, T_art, mmw_art, vmr_he) -- unpack explicitly: a bare *g0
    # splat would land vmr_he in the positional lnR0 slot.
    def trans_r(lnR0):
        vmr_, vmr_h2_, T_art_, mmw_art_, vmr_he_ = g0
        return trt.transmission_depth_r(vmr_, vmr_h2_, T_art_, mmw_art_, lnR0,
                                        vmr_he=vmr_he_)
    _, dR = jax.jvp(trans_r, (jnp.float64(0.0),), (jnp.float64(1.0),))
    J_lnR0 = np.asarray(dR)
    print(f"  d/dlnR0 finite={np.all(np.isfinite(J_lnR0))} max|.|={np.nanmax(np.abs(J_lnR0)):.2e}"
          f"  (tier {tier} total {time.time()-t0:.0f}s)", flush=True)
    return depth0, J_chem, J_lnR0


def _load_existing():
    if OUT.exists():
        d = np.load(OUT, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return {}


def main():
    smoke = "--smoke" in sys.argv
    only = None
    if "--tier" in sys.argv:
        only = sys.argv[sys.argv.index("--tier") + 1]
    tiers = [only] if only else TIER_ORDER

    prof0 = _profile(smoke, tiers[0])
    trt = exojax_rt.build_rt_model(prof0)      # shared opacity build
    wl = np.asarray(trt.wl_um)

    store = _load_existing()
    store["wl_um"] = wl
    store["theta0"] = np.asarray(config.THETA0)
    store["molecules"] = np.array(trt.molecules)
    done = list(store.get("tiers", []))
    for t in tiers:
        depth, Jc, JR = build_tier(t, trt, smoke)
        store[f"depth_{t}"] = depth
        store[f"Jchem_{t}"] = Jc
        store[f"JlnR0_{t}"] = JR
        if t not in done:
            done.append(t)
        store["tiers"] = np.array([x for x in TIER_ORDER if x in done])
        config.OUTPUTS.mkdir(parents=True, exist_ok=True)
        np.savez(OUT, **store)          # incremental save so a crash keeps finished tiers
        print(f"  [saved] {OUT.name}: tiers so far = {list(store['tiers'])}", flush=True)
    print(f"\nDONE. wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
