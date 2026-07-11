"""Fast chemistry-only validation for the Z / C/O Fisher figures (no RT, no line lists).

Answers the three questions that decide the expensive build's design:

  (1) fixed-O C/O knob conserves O while scaling C. Perturb c_o and check, on the
      CONVERGED column's element totals: C-total scales ~ e^c_o, O-total ~ invariant.
  (2) forward-mode AD gives FINITE tangents for all three chemistry tiers -- in
      particular photochemistry OFF (config.py warns the warm-started jvp can be
      unstable there; here we re-converge from the FastChem init, so we must check).
  (3) AD matches central finite differences at the fiducial (the derivative-verification
      the Fisher figure needs), on a scalar chemistry readout (SO2 column density).

Run (base env, ~a few min at nz=40):  python validation/smoke_zco.py
"""
from __future__ import annotations

import time

import numpy as np

from retrieval_framework.forward import config
from retrieval_framework.forward import vulcan_chem
import jax
import jax.numpy as jnp

NZ = 40
# Tiers: P = full (photo+transport), Q = transport-only (quench, no photo),
#        E = equilibrium (no photo, no transport).
TIERS = {
    "P_photo":  dict(use_photo=True),
    "Q_quench": dict(use_photo=False),
    "E_equil":  dict(use_photo=False, zero_Kzz=True, cfg_overrides={"use_moldiff": False}),
}
BASE = dict(config.SMOKE, nz=NZ, co_mode="fixed_O",
            molecules=["CO"],           # opacity set is irrelevant here (chemistry only)
            yconv_cri=1.0e-3)


def element_totals(chem, ymix):
    """Column-summed element totals (C, O) from a linear-VMR profile (nz, ni)."""
    compo = np.asarray(chem.compo_array, dtype=np.float64)
    nC = compo[:, config.ATOM_COLS["C"]]
    nO = compo[:, config.ATOM_COLS["O"]]
    y = np.asarray(ymix, dtype=np.float64)
    return float((y * nC[None, :]).sum()), float((y * nO[None, :]).sum())


def main():
    theta0 = jnp.asarray([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    so2 = None
    results = {}
    for name, over in TIERS.items():
        prof = dict(BASE, **over)
        print(f"\n===== tier {name}  {over} =====", flush=True)
        t0 = time.time()
        chem = vulcan_chem.build_chem_model(prof)
        so2 = chem.sidx["SO2"]
        f = chem.converged_ymix

        # (2) AD finiteness: one jvp per chem param
        ok_finite = True
        for k, lab in enumerate(["lnZ", "c_o", "lnKzz", "T_int"]):
            e = jnp.zeros(4).at[k].set(1.0)
            y, dy = jax.jvp(f, (theta0,), (e,))
            fin = bool(np.all(np.isfinite(np.asarray(dy))))
            ok_finite &= fin
            print(f"  jvp d/d{lab:5s} finite={fin} max|dy|={np.nanmax(np.abs(np.asarray(dy))):.2e}", flush=True)

        # (1) fixed-O conservation: compare element totals at c_o = 0 vs +0.20 (finite)
        y0 = f(theta0)
        C0, O0 = element_totals(chem, y0)
        yC = f(jnp.asarray([0.0, 0.20, 0.0, 0.0]))
        C1, O1 = element_totals(chem, yC)
        print(f"  fixed-O check: C ratio {C1/C0:.4f} (expect {np.exp(0.20):.4f});  "
              f"O ratio {O1/O0:.5f} (expect 1.00000)", flush=True)

        # (3) AD vs central-FD on log10 SO2 column density, for lnZ and c_o
        def so2_col(th):
            y = f(th)  # linear VMR; use as proxy column (sum over layers)
            return jnp.log10(jnp.sum(y[:, so2]) + 1e-300)
        for k, lab in enumerate(["lnZ", "c_o"]):
            e = jnp.zeros(4).at[k].set(1.0)
            _, g_ad = jax.jvp(so2_col, (theta0,), (e,))
            h = 0.02
            thp = theta0.at[k].add(h); thm = theta0.at[k].add(-h)
            g_fd = (np.asarray(so2_col(thp)) - np.asarray(so2_col(thm))) / (2 * h)
            g_ad = float(np.asarray(g_ad))
            rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-12)
            print(f"  dlog10(SO2col)/d{lab:4s}: AD={g_ad:+.4f} FD={g_fd:+.4f} rel={rel:.2e}", flush=True)
            results[(name, lab)] = (g_ad, g_fd, rel)
        print(f"  tier {name} done in {time.time()-t0:.0f}s  AD_all_finite={ok_finite}", flush=True)

    print("\n===== SUMMARY =====")
    worst = max(v[2] for v in results.values())
    print(f"worst AD-vs-FD rel err across tiers/params: {worst:.2e}")
    print("PASS" if worst < 0.05 else "CHECK (AD-FD disagreement > 5% somewhere)")


if __name__ == "__main__":
    main()
