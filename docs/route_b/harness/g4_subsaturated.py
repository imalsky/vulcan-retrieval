"""B0C G4: hot/subsaturated limit -- smooth_rainout == condensation-off.

On a column safely below S8 saturation everywhere, smooth_rainout must
reproduce the condensation-OFF endpoint (identical within solver noise),
with the rainout flux EXACTLY zero and no sulfur removed (plan section 6;
the kernel-level guarantee was already measured at 1.6e-16 in B0B's
VULCAN-JAX runtime test -- this gate re-measures it on the production
photo-on fixture path).

Hot variant: Tirr raised to 1560 K (W39b-like; Teq ~ 1103 K). The script
first MEASURES the S8 saturation ratio profile at the smooth model's init
and REFUSES if any cell is above ROUTE_B_G4_SAT_CEILING (default 0.1 --
"safely below", not merely below). The condensation-off twin keeps the SAME
H2S bottom pin (use_fix_sp_bot is mode-independent), so the ONLY difference
is the sink machinery.

Run (scheduled, heavy -- 2 builds + 2 solves):
    ROUTE_B_W107B=1 python docs/route_b/harness/g4_subsaturated.py
Env: W107B_NZ, W107B_COUNT_MAX, ROUTE_B_G4_SAT_CEILING,
ROUTE_B_G4_RTOL/ATOL (documented agreement tolerances; INCOMPLETE without).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w107b_fixture as wf  # noqa: E402
import gate_common as gc  # noqa: E402

TIRR_HOT = 1560.0

CONDEN_OFF_EXTRA = {
    "use_condense": False,
    "conden_mode": "master_pin",   # irrelevant with conden off; explicit
    "condense_sp": [],
    "non_gas_sp": [],
}


def main():
    wf.require_env_gate()
    import importlib

    from retrieval_framework.forward import config

    nz = int(os.environ.get("W107B_NZ", "100"))
    count_max = int(os.environ.get("W107B_COUNT_MAX", "15000"))
    sat_ceiling = float(os.environ.get("ROUTE_B_G4_SAT_CEILING", "0.1"))
    rtol_env = os.environ.get("ROUTE_B_G4_RTOL")
    atol_env = os.environ.get("ROUTE_B_G4_ATOL")

    theta = np.array(wf.FIDUCIAL_THETA, dtype=np.float64)
    theta[3] = TIRR_HOT

    chem_s, meta_s = wf.build(nz=nz, count_max=count_max)
    cfgmod = importlib.import_module(config.W39B_CFG_MODULE)
    runtime = float(cfgmod.runtime)

    # measured subsaturation refusal (loud, before any solve)
    init, atm_T = chem_s.prep_state(theta)
    i_s8 = chem_s.sidx["S8"]
    n_s8 = np.asarray(init.y[:, i_s8], dtype=np.float64)
    n_sat = np.asarray(init.pv.c_sat_n_per_re[0], dtype=np.float64)
    sat_ratio = n_s8 / n_sat
    if sat_ratio.max() >= sat_ceiling:
        raise SystemExit(
            f"[g4] REFUSED: max S8 saturation ratio {sat_ratio.max():.3e} "
            f">= ceiling {sat_ceiling} at init (z={int(sat_ratio.argmax())})"
            " -- the hot fixture is not safely subsaturated; raise Tirr or "
            "lower the ceiling deliberately.")
    print(f"[g4] subsaturation confirmed: max S8 sat ratio "
          f"{sat_ratio.max():.3e} (ceiling {sat_ceiling})")

    fin_s = chem_s._integ._runner(init, atm_T)
    fin_s.y.block_until_ready()
    led_s = gc.ledger_report(chem_s, fin_s)
    solve_s = gc.solve_report(chem_s, fin_s, count_max, runtime)

    print("[g4] condensation-off twin build + solve")
    # The twin has no on-graph lookup (smooth-only plumbing), so its STATIC
    # bottom pin must be set to the smooth model's LIVE pin at the G4 theta
    # -- same boundary in both columns, or the comparison measures the
    # boundary difference instead of the sink.
    pin_at_theta = chem_s.pin_value(theta)["x_pin"]
    chem_o, meta_o = wf.build(
        nz=nz, count_max=count_max,
        cfg_extra=dict(CONDEN_OFF_EXTRA,
                       use_fix_sp_bot={"H2S": float(pin_at_theta)}))
    init_o, atm_o = chem_o.prep_state(theta)
    fin_o = chem_o._integ._runner(init_o, atm_o)
    fin_o.y.block_until_ready()
    solve_o = gc.solve_report(chem_o, fin_o, count_max, runtime)

    y_s = np.asarray(fin_s.y, dtype=np.float64)
    y_o = np.asarray(fin_o.y, dtype=np.float64)
    # subsaturation must also hold at the ENDPOINT ("safely below
    # saturation everywhere" is a property of the solution, not the init)
    sat_ratio_end = y_s[:, i_s8] / n_sat
    if sat_ratio_end.max() >= sat_ceiling:
        raise SystemExit(
            f"[g4] REFUSED: endpoint S8 saturation ratio "
            f"{sat_ratio_end.max():.3e} >= ceiling {sat_ceiling}: the solve "
            "moved the column out of the subsaturated regime.")
    delta = gc.species_delta_report(chem_s, y_s, y_o)
    rain_S = led_s["led_rain"]["S"]
    s_col_s = float((y_s[:, i_s8]).sum())
    s_col_o = float((y_o[:, i_s8]).sum())

    rain_zero = rain_S == 0.0
    both_converged = (solve_s["termination"] == "converged-gate"
                      and solve_o["termination"] == "converged-gate")
    if not both_converged:
        rtol = float(rtol_env) if rtol_env is not None else None
        atol = float(atol_env) if atol_env is not None else None
        verdict = (f"TAINTED (smooth: {solve_s['termination']}, conden-off: "
                   f"{solve_o['termination']} -- the identical-endpoint "
                   "claim needs converged endpoints)")
    elif rtol_env is not None and atol_env is not None:
        rtol, atol = float(rtol_env), float(atol_env)
        agree = bool(np.all(np.abs(y_s - y_o)
                            <= atol + rtol * np.maximum(np.abs(y_s),
                                                        np.abs(y_o))))
        verdict = "PASS" if (rain_zero and agree) else "FAIL"
    else:
        rtol = atol = None
        verdict = ("INCOMPLETE (rain exactly zero: "
                   f"{rain_zero}; max |dln n| {delta['max_abs_dln_n']:.3e}; "
                   "documented rtol/atol pending -- set ROUTE_B_G4_RTOL/ATOL)")

    payload = {
        "gate": "G4",
        "verdict": verdict,
        "rtol": rtol, "atol": atol,
        "theta": theta.tolist(),
        "Tirr_hot_K": TIRR_HOT,
        "sat_ceiling": sat_ceiling,
        "max_init_sat_ratio": float(sat_ratio.max()),
        "max_endpoint_sat_ratio": float(sat_ratio_end.max()),
        "phi_rain_S_exactly_zero": rain_zero,
        "phi_rain_S": rain_S,
        "endpoint_delta": delta,
        "S8_column_smooth": s_col_s,
        "S8_column_condenoff": s_col_o,
        "solve_smooth": solve_s,
        "solve_condenoff": solve_o,
        "fixture_smooth": meta_s,
        "fixture_condenoff": meta_o,
        "provenance": gc.provenance(),
    }
    gc.save_artifact("w107b_g4", payload, arrays={
        "y_smooth": y_s, "y_condenoff": y_o, "sat_ratio_init": sat_ratio})
    print(f"[g4] rain_S={rain_S:.3e} (exactly zero: {rain_zero}); "
          f"max |dln n| = {delta['max_abs_dln_n']:.3e} at "
          f"{delta['argmax_species']} z={delta['argmax_z']}; "
          f"verdict={verdict}")


if __name__ == "__main__":
    main()
