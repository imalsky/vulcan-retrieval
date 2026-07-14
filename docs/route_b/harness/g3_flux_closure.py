"""B0C G3: sulfur-flux closure at the G1 endpoint (D6 budget).

Analyzes the latest (or named) G1 artifact -- no solve, cheap, host-only.

The D6 open-system budget at the converged endpoint, per open element E in
{S, H}: dN_E/dt = Phi_bc(E) - Phi_top(E) - Phi_rain(E), where
  Phi_bc   = led_bc / led_dt      (the implied bottom source, measured at
                                   the pinned cells; the D3b diagnostic)
  Phi_rain = led_rain             (instantaneous elemental rainout rate at
                                   the committed state)
  Phi_top  = 0                    (zero-flux top boundary -- explicit G0
                                   statement, not an omission)
  dN_E/dt  = (led_step + led_renorm + led_bc) / led_dt
                                  (net inventory drift at the last accepted
                                   step; ~0 at a true steady state)
Closure metric per element: |dN_E/dt| / max(|Phi_bc|, |Phi_rain|, floor) --
the fraction of the boundary/rain throughput left unaccounted. The gate
passes when this is below the documented tolerance (ROUTE_B_G3_TOL; without
it the verdict is INCOMPLETE and the measured numbers stand for the record).
Also checked: led_bc H = 2 x led_bc S EXACTLY (H2S pin stoichiometry) and
Phi_rain > 0 (the gate is only meaningful with ACTIVE rainout -- a zero-rain
endpoint must FAIL loudly here, per the directive's "nonzero flux" demand).

Run: python docs/route_b/harness/g3_flux_closure.py [g1_artifact.json]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402


def latest_g1() -> Path:
    cands = sorted(gc.RESULTS_DIR.glob("w107b_g1_*.json"))
    if not cands:
        raise SystemExit("no w107b_g1_*.json artifact under results/; run "
                         "g1_convergence.py first")
    return cands[-1]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_g1()
    g1 = json.loads(path.read_text())
    led = g1["ledger"]
    atoms = led["atom_order"]
    dt = float(led["led_dt_s"])
    tol_env = os.environ.get("ROUTE_B_G3_TOL")

    per_element = {}
    for e in ("S", "H"):
        if e not in atoms:
            raise SystemExit(f"element {e} missing from atom order {atoms}")
        phi_bc = led["led_bc"][e] / dt
        phi_rain = led["led_rain"][e]
        drift = (led["led_step"][e] + led["led_renorm"][e]
                 + led["led_bc"][e]) / dt
        scale = max(abs(phi_bc), abs(phi_rain), 1e-300)
        per_element[e] = {
            "Phi_bc_atoms_cm^-2_s^-1": phi_bc,
            "Phi_rain_atoms_cm^-2_s^-1": phi_rain,
            "Phi_top_atoms_cm^-2_s^-1": 0.0,
            "dN_dt_atoms_cm^-2_s^-1": drift,
            "closure_frac": abs(drift) / scale,
        }

    s_bc = led["led_bc"]["S"]
    h_bc = led["led_bc"]["H"]
    stoich_exact = (h_bc == 2.0 * s_bc)
    rain_active = per_element["S"]["Phi_rain_atoms_cm^-2_s^-1"] > 0.0

    worst = max(v["closure_frac"] for v in per_element.values())
    if not rain_active:
        verdict = ("FAIL (Phi_rain,S = 0: no active rainout at the endpoint;"
                   " the source-bearing fixture demand is not met)")
    elif tol_env is not None:
        verdict = ("PASS" if (worst <= float(tol_env) and stoich_exact)
                   else "FAIL")
    else:
        verdict = ("INCOMPLETE (rain active, worst closure_frac "
                   f"{worst:.3e}; no documented tolerance -- set "
                   "ROUTE_B_G3_TOL after the record review)")

    payload = {
        "gate": "G3",
        "verdict": verdict,
        "closure_tolerance": float(tol_env) if tol_env is not None else None,
        "per_element": per_element,
        "led_bc_H_equals_2x_S_exactly": stoich_exact,
        "telescoping_rel": led["telescoping_rel"],
        "g1_artifact": path.name,
        "g1_termination": g1["solve"]["termination"],
        "provenance": g1["provenance"],
    }
    gc.save_artifact("w107b_g3", payload)
    for e, v in per_element.items():
        print(f"[g3] {e}: Phi_bc={v['Phi_bc_atoms_cm^-2_s^-1']:.4e} "
              f"Phi_rain={v['Phi_rain_atoms_cm^-2_s^-1']:.4e} "
              f"dN/dt={v['dN_dt_atoms_cm^-2_s^-1']:.4e} "
              f"closure_frac={v['closure_frac']:.3e}")
    print(f"[g3] H=2S stoich exact: {stoich_exact}; verdict={verdict}")


if __name__ == "__main__":
    main()
