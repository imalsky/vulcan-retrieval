"""B0C G2: multiple-seed agreement on the W107b fixture.

The plan's five materially distinct initializations, all solved to the SAME
fiducial theta on the SAME smooth-rainout model:

  1. equilibrium      -- the default cold init (FastChem EQ + elemental map)
  2. master_pin       -- the converged column of a SECOND model built in
                         legacy master_pin mode (certified window+pin recipe)
                         used as a warm seed
  3. sulfur-rich      -- seed 1's converged y with every S-bearing species
                         x10 (vertical S distribution displaced; the column
                         S/H ratio is re-projected by _prep's elemental
                         repair, so what survives is the DISTRIBUTION
                         perturbation -- documented, not a bug: the open S
                         budget means the solver, not the init, must set the
                         final S inventory via boundary + rain)
  4. sulfur-poor      -- same with /10
  5. continuation     -- warm start from the converged column at a
                         neighboring theta (lnZ + 0.2)

Agreement metric (plan): |y1 - y2| <= atol + rtol*max(|y1|,|y2|) on
RT-relevant gas columns, plus rainout-flux and budget-residual deltas.
NOT measured here (logged, per the no-silent-caps rule): the binned-spectrum
agreement column -- it needs the W107b RT harness (spectrum-derivative work
item); this artifact must be extended before G2 is declared closed.

Run (scheduled, heavy -- ~7 solves + 2 builds):
    ROUTE_B_W107B=1 python docs/route_b/harness/g2_seeds.py
Env: W107B_NZ, W107B_COUNT_MAX (default 15000), ROUTE_B_G2_RTOL /
ROUTE_B_G2_ATOL (documented tolerances; verdict INCOMPLETE without them).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w107b_fixture as wf  # noqa: E402
import gate_common as gc  # noqa: E402

# RT-relevant gas columns for the agreement gate (the tool's sulfur +
# background observables on the SNCHO network).
RT_SPECIES = ["H2O", "CO", "CO2", "CH4", "NH3", "H2S", "SO2", "SO", "S8"]

MASTER_PIN_EXTRA = {
    "conden_mode": "master_pin",
    "fix_species": ["S8", "S8_l_s"],
    "fix_species_from_coldtrap_lev": False,
    "start_conden_time": 0.0,
    "stop_conden_time": 1.0e6,
    "trun_min": 1.0e6,
    "mtol_conv": 1.0e-15,
    "conver_ignore": ["C6H6", "C2H2", "C6H5", "C2H", "C2H4", "C2H5", "C2H6",
                      "C3H2", "C3H3", "C4H5", "CH2NH", "CH3NH2", "H2CCO",
                      "S", "S2", "S3", "S4"],
}


def _solve(chem, theta, warm_y=None):
    init, atm_T = chem.prep_state(theta, warm_y=warm_y)
    final = chem._integ._runner(init, atm_T)
    final.y.block_until_ready()
    return final


def main():
    wf.require_env_gate()
    import importlib

    from retrieval_framework.forward import config

    nz = int(os.environ.get("W107B_NZ", "100"))
    count_max = int(os.environ.get("W107B_COUNT_MAX", "15000"))
    rtol_env = os.environ.get("ROUTE_B_G2_RTOL")
    atol_env = os.environ.get("ROUTE_B_G2_ATOL")

    chem, meta = wf.build(nz=nz, count_max=count_max)
    cfgmod = importlib.import_module(config.W39B_CFG_MODULE)
    runtime = float(cfgmod.runtime)
    theta = wf.FIDUCIAL_THETA
    s_mask = np.asarray(chem.compo_array[:, config.ATOM_COLS["S"]] > 0)

    solves = {}

    print("[g2] seed 1: equilibrium cold init")
    solves["equilibrium"] = _solve(chem, theta)
    y_eq = np.asarray(solves["equilibrium"].y, dtype=np.float64)

    print("[g2] seed 2: master_pin twin build + converge (legacy recipe)")
    chem_mp, _ = wf.build(nz=nz, count_max=count_max,
                          cfg_extra=dict(MASTER_PIN_EXTRA))
    fin_mp = _solve(chem_mp, theta)
    # rebuild the SMOOTH model (the twin build mutated the shared cfg module)
    chem, meta = wf.build(nz=nz, count_max=count_max)
    solves["master_pin_seed"] = _solve(chem, theta,
                                       warm_y=np.asarray(fin_mp.y))

    print("[g2] seeds 3/4: sulfur-rich / sulfur-poor displacements")
    for name, fac in (("sulfur_rich", 10.0), ("sulfur_poor", 0.1)):
        y_seed = y_eq * np.where(s_mask[None, :], fac, 1.0)
        solves[name] = _solve(chem, theta, warm_y=y_seed)

    print("[g2] seed 5: continuation from lnZ + 0.2")
    theta_n = np.array(theta, dtype=np.float64)
    theta_n[0] += 0.2
    fin_n = _solve(chem, theta_n)
    solves["continuation"] = _solve(chem, theta, warm_y=np.asarray(fin_n.y))

    ref = "equilibrium"
    y_ref = np.asarray(solves[ref].y, dtype=np.float64)
    dz = np.asarray(chem.dz, dtype=np.float64)
    idx_rt = [chem.sidx[s] for s in RT_SPECIES if s in chem.sidx]
    names_rt = [s for s in RT_SPECIES if s in chem.sidx]
    col_ref = (y_ref[:, idx_rt] * dz[:, None]).sum(axis=0)

    comparisons = {}
    worst_pair = 0.0
    for name, fin in solves.items():
        if name == ref:
            continue
        y = np.asarray(fin.y, dtype=np.float64)
        col = (y[:, idx_rt] * dz[:, None]).sum(axis=0)
        # plan metric on RT-relevant gas columns
        rel = np.abs(col - col_ref) / np.maximum(np.abs(col), np.abs(col_ref))
        led = gc.ledger_report(chem, fin)
        led_ref = gc.ledger_report(chem, solves[ref])
        comparisons[name] = {
            "solve": gc.solve_report(chem, fin, count_max, runtime),
            "rt_column_rel_delta": dict(zip(names_rt, rel.tolist())),
            "max_rt_column_rel_delta": float(rel.max()),
            "phi_rain_S": led["led_rain"]["S"],
            "phi_rain_S_ref": led_ref["led_rain"]["S"],
            "cell_delta": gc.species_delta_report(chem, y_ref, y),
        }
        worst_pair = max(worst_pair, float(rel.max()))

    if rtol_env is not None and atol_env is not None:
        rtol, atol = float(rtol_env), float(atol_env)
        ok = all(
            np.all(np.abs((y := np.asarray(solves[n].y)) - y_ref)
                   <= atol + rtol * np.maximum(np.abs(y), np.abs(y_ref)))
            for n in solves if n != ref)
        verdict = "PASS" if ok else "FAIL"
    else:
        rtol = atol = None
        verdict = ("INCOMPLETE (worst RT-column rel delta "
                   f"{worst_pair:.3e}; documented rtol/atol pending -- set "
                   "ROUTE_B_G2_RTOL/ATOL; binned-spectrum agreement column "
                   "NOT yet measured, needs the W107b RT harness)")

    payload = {
        "gate": "G2",
        "verdict": verdict,
        "rtol": rtol, "atol": atol,
        "reference_seed": ref,
        "theta": np.asarray(theta).tolist(),
        "reference_solve": gc.solve_report(chem, solves[ref], count_max,
                                           runtime),
        "comparisons": comparisons,
        "spectrum_agreement": "NOT MEASURED (RT harness pending; G2 not "
                              "closable without it)",
        "fixture": meta,
        "provenance": gc.provenance(),
    }
    gc.save_artifact("w107b_g2", payload, arrays={
        f"y_{n}": np.asarray(f.y) for n, f in solves.items()})
    for n, c in comparisons.items():
        print(f"[g2] {n}: max RT-column rel delta "
              f"{c['max_rt_column_rel_delta']:.3e}, "
              f"termination {c['solve']['termination']}")
    print(f"[g2] verdict={verdict}")



if __name__ == "__main__":
    main()
