"""B0C G5: regularization-width kill test + physical-knob ladders.

G5a (KILL criterion, plan section 6): endpoint stability over a documented
usable range of the smoothing width w = conden_smooth_width. Ladder
{0.05, 0.1, 0.2} (half / default / double). Instability across this range
IS a numerical failure. Endpoint metric here = chemistry endpoints (max
|dln n| + RT-relevant gas columns); the binned-SPECTRUM stability column is
NOT yet measured (needs the W107b RT harness) and is logged as pending --
G5a is not closable without it.

G5b (CHARACTERIZE, never auto-kill): 3-point ladders in rainout_rate_scale
{0.1, 1, 10} (model rebuilds -- statics-baked), deep supply via lnZ
{-0.5, 0, +0.5} and Kzz via lnKzz {-ln10/2, 0, +ln10/2} (theta-only, no
rebuild). Strong dependence can be scientifically real; the artifact
records the measured sensitivities for scenario forecasts / Fisher
bracketing decisions.

Lookup-cell probe (host-only, cheap): pin values and cell indices along the
T ladder of visited points, plus an assert_points_same_cell demonstration
on the FD-step point set at the fiducial -- records which gate points share
lookup cells (the C0-derivative disclosure feeding the G6 design).

Run (scheduled, VERY heavy -- ~5 builds + ~11 solves):
    ROUTE_B_W107B=1 python docs/route_b/harness/g5_ladders.py
Env: W107B_NZ, W107B_COUNT_MAX, ROUTE_B_G5A_RTOL/ATOL (documented G5a
tolerances; INCOMPLETE without).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w107b_fixture as wf  # noqa: E402
import gate_common as gc  # noqa: E402
from g2_seeds import RT_SPECIES  # noqa: E402  (same observable set)

W_LADDER = (0.05, 0.1, 0.2)
SCALE_LADDER = (0.1, 1.0, 10.0)
LNZ_LADDER = (-0.5, 0.0, 0.5)
LNKZZ_LADDER = (-0.5 * np.log(10.0), 0.0, 0.5 * np.log(10.0))


def _solve(chem, theta):
    init, atm_T = chem.prep_state(theta)
    final = chem._integ._runner(init, atm_T)
    final.y.block_until_ready()
    return final


def _endpoint(chem, fin, count_max, runtime):
    y = np.asarray(fin.y, dtype=np.float64)
    dz = np.asarray(chem.dz, dtype=np.float64)
    idx = [chem.sidx[s] for s in RT_SPECIES if s in chem.sidx]
    names = [s for s in RT_SPECIES if s in chem.sidx]
    led = gc.ledger_report(chem, fin)
    return {
        "solve": gc.solve_report(chem, fin, count_max, runtime),
        "rt_columns": dict(zip(names,
                               ((y[:, idx] * dz[:, None]).sum(axis=0)).tolist())),
        "phi_rain_S": led["led_rain"]["S"],
    }, y


def main():
    wf.require_env_gate()
    import importlib

    from retrieval_framework.forward import config, h2s_boundary as hb

    nz = int(os.environ.get("W107B_NZ", "100"))
    count_max = int(os.environ.get("W107B_COUNT_MAX", "15000"))
    rtol_env = os.environ.get("ROUTE_B_G5A_RTOL")
    atol_env = os.environ.get("ROUTE_B_G5A_ATOL")
    theta = wf.FIDUCIAL_THETA
    cfgmod = importlib.import_module(config.W39B_CFG_MODULE)

    # ---- G5a: w ladder (kill criterion) --------------------------------
    g5a = {}
    y_by_w = {}
    saved_arrays = {}
    for w in W_LADDER:
        print(f"[g5a] build + solve at w = {w}")
        chem, meta = wf.build(nz=nz, count_max=count_max,
                              cfg_extra={"conden_smooth_width": float(w)})
        runtime = float(cfgmod.runtime)
        fin = _solve(chem, theta)
        g5a[str(w)], y_by_w[w] = _endpoint(chem, fin, count_max, runtime)
        saved_arrays[f"y_w_{w}"] = y_by_w[w]
    w_ref = 0.1
    chem_ref = chem  # last build (w=0.2); species map identical across builds
    deltas_w = {
        str(w): gc.species_delta_report(chem_ref, y_by_w[w_ref], y_by_w[w])
        for w in W_LADDER if w != w_ref
    }
    worst_w = max(d["max_abs_dln_n"] for d in deltas_w.values())
    if rtol_env is not None and atol_env is not None:
        rtol, atol = float(rtol_env), float(atol_env)
        ok = all(
            bool(np.all(np.abs(y_by_w[w] - y_by_w[w_ref])
                        <= atol + rtol * np.maximum(np.abs(y_by_w[w]),
                                                    np.abs(y_by_w[w_ref]))))
            for w in W_LADDER if w != w_ref)
        verdict_a = "PASS" if ok else "FAIL (w-instability IS numerical failure)"
    else:
        rtol = atol = None
        verdict_a = ("INCOMPLETE (worst endpoint |dln n| across w: "
                     f"{worst_w:.3e}; documented tolerances pending -- set "
                     "ROUTE_B_G5A_RTOL/ATOL; SPECTRUM stability column NOT "
                     "yet measured, needs the W107b RT harness)")

    # ---- G5b: physical ladders (characterize) ---------------------------
    g5b = {"rainout_rate_scale": {}, "lnZ": {}, "lnKzz": {}}
    for s in SCALE_LADDER:
        if s == 1.0:
            g5b["rainout_rate_scale"][str(s)] = g5a[str(w_ref)]
            continue
        print(f"[g5b] build + solve at rainout_rate_scale = {s}")
        chem, _ = wf.build(nz=nz, count_max=count_max,
                           cfg_extra={"rainout_rate_scale": float(s)})
        fin = _solve(chem, theta)
        g5b["rainout_rate_scale"][str(s)], _y = _endpoint(
            chem, fin, count_max, float(cfgmod.runtime))
        saved_arrays[f"y_scale_{s}"] = _y

    print("[g5b] rebuilding the default model for theta-only ladders")
    chem, meta = wf.build(nz=nz, count_max=count_max)
    runtime = float(cfgmod.runtime)
    for lnz in LNZ_LADDER:
        th = np.array(theta, dtype=np.float64)
        th[0] = lnz
        print(f"[g5b] solve at lnZ = {lnz}")
        fin = _solve(chem, th)
        g5b["lnZ"][f"{lnz:+.2f}"], _y = _endpoint(chem, fin, count_max,
                                                  runtime)
        saved_arrays[f"y_lnZ_{lnz:+.2f}"] = _y
    for lnk in LNKZZ_LADDER:
        th = np.array(theta, dtype=np.float64)
        th[2] = lnk
        print(f"[g5b] solve at lnKzz = {lnk:+.3f}")
        fin = _solve(chem, th)
        g5b["lnKzz"][f"{lnk:+.3f}"], _y = _endpoint(chem, fin, count_max,
                                                    runtime)
        saved_arrays[f"y_lnKzz_{lnk:+.3f}"] = _y

    # ---- lookup-cell probe (host-only) ----------------------------------
    table = hb.load_h2s_boundary_table(config.H2S_BOUNDARY_TABLE)
    visited = []
    for lnz in LNZ_LADDER:
        th = np.array(theta, dtype=np.float64)
        th[0] = lnz
        d = chem.pin_value(th)
        visited.append({
            **d, "cell": list(hb.cell_of(table, d["T_bottom"], d["lnZ"],
                                         d["c_o"]))})
    fid = chem.pin_value(theta)
    fd_pts = []
    for h_t in (2.0, 1.0):        # FD steps h and h/2 in T_bottom via Tirr
        for sgn in (1, -1):
            fd_pts.append([fid["T_bottom"] + sgn * h_t, fid["lnZ"],
                           fid["c_o"]])
    try:
        cell = hb.assert_points_same_cell(table,
                                          [[fid["T_bottom"], fid["lnZ"],
                                            fid["c_o"]]] + fd_pts)
        fd_cell = {"same_cell": True, "cell": list(cell),
                   "fd_T_steps": [2.0, 1.0]}
    except ValueError as e:
        fd_cell = {"same_cell": False, "error": str(e),
                   "fd_T_steps": [2.0, 1.0]}

    payload = {
        "gate": "G5",
        "verdict_G5a": verdict_a,
        "G5a_rtol": rtol, "G5a_atol": atol,
        "G5a_w_ladder": g5a,
        "G5a_endpoint_deltas_vs_w0.1": deltas_w,
        "G5a_spectrum_stability": "NOT MEASURED (RT harness pending; G5a "
                                  "not closable without it)",
        "verdict_G5b": "CHARACTERIZED (never auto-kill; see ladders)",
        "G5b_ladders": g5b,
        "theta": np.asarray(theta).tolist(),
        "lookup_cells_visited": visited,
        "fd_point_knot_guard": fd_cell,
        "fixture": meta,
        "provenance": gc.provenance(),
    }
    gc.save_artifact("w107b_g5", payload, arrays=saved_arrays)
    print(f"[g5] G5a: {verdict_a}")
    for knob, ladder in g5b.items():
        for k, v in ladder.items():
            print(f"[g5b] {knob}={k}: phi_rain_S={v['phi_rain_S']:.3e} "
                  f"termination={v['solve']['termination']}")


if __name__ == "__main__":
    main()
