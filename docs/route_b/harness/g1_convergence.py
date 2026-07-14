"""B0C G1: normal convergence + small direct residual on the W107b fixture.

Gate criterion (plan section 6): the longdy gate is reached WITHOUT runtime
cap or stall certification within count_max <= 15000, AND the B0-5 scaled
direct residual ||F(y*, theta)|| is below a documented threshold. A
slowly-changing state with a large residual is a FAIL. No tuning is
permitted here beyond the G5 knobs (w, rainout_rate_scale, boundary supply).

Run (scheduled, heavy -- photo-on W107b build + cold solve):
    ROUTE_B_W107B=1 python docs/route_b/harness/g1_convergence.py

Env knobs: W107B_NZ (default 100), W107B_COUNT_MAX (default 15000, the
plan's G1 bound), ROUTE_B_G1_MAX_R (documented residual threshold in s^-1;
without it the verdict is INCOMPLETE and the measured max_R is recorded for
the threshold to be set in the B0C record).

Writes results/w107b_g1_<stamp>.json + .npz (converged y, ledger, pin
diagnostics) -- G2/G3 consume the npz instead of re-solving.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w107b_fixture as wf  # noqa: E402
import gate_common as gc  # noqa: E402


def main():
    wf.require_env_gate()
    import importlib

    from retrieval_framework.forward import config

    nz = int(os.environ.get("W107B_NZ", "100"))
    count_max = int(os.environ.get("W107B_COUNT_MAX", "15000"))
    max_r_env = os.environ.get("ROUTE_B_G1_MAX_R")

    t0 = time.time()
    chem, meta = wf.build(nz=nz, count_max=count_max)
    cfgmod = importlib.import_module(config.W39B_CFG_MODULE)
    runtime = float(cfgmod.runtime)
    yconv_cri = float(cfgmod.yconv_cri)

    theta = wf.FIDUCIAL_THETA
    pin = chem.pin_value(theta)   # domain-validates the visited point (loud)
    print(f"[g1] fiducial pin: x_H2S = {pin['x_pin']:.6e} at "
          f"T_bottom = {pin['T_bottom']:.1f} K")

    t1 = time.time()
    init, atm_T = chem.prep_state(theta)
    final = chem._integ._runner(init, atm_T)
    final.y.block_until_ready()
    solve_s = time.time() - t1

    solve = gc.solve_report(chem, final, count_max, runtime)
    # Stall discrimination: terminated before the caps but above the longdy
    # criterion means the stall-certification path fired, not the gate.
    if (solve["termination"] == "converged-gate"
            and solve["longdy"] > yconv_cri):
        solve["termination"] = "stalled-certification"
    resid = gc.residual_report(chem, final, atm_T)
    ledger = gc.ledger_report(chem, final)

    term_ok = solve["termination"] == "converged-gate"
    if max_r_env is not None:
        resid_ok = resid["max_R_s^-1"] <= float(max_r_env)
        verdict = "PASS" if (term_ok and resid_ok) else "FAIL"
        threshold = float(max_r_env)
    else:
        verdict = ("INCOMPLETE (termination "
                   + ("ok" if term_ok else f"FAIL: {solve['termination']}")
                   + "; no documented max_R threshold -- set ROUTE_B_G1_MAX_R"
                   " after the record review)")
        threshold = None

    payload = {
        "gate": "G1",
        "verdict": verdict,
        "residual_threshold_s^-1": threshold,
        "theta": np.asarray(theta).tolist(),
        "pin": pin,
        "solve": solve,
        "solve_wall_s": solve_s,
        "build_wall_s": t1 - t0,
        "yconv_cri": yconv_cri,
        "residual": resid,
        "ledger": ledger,
        "fixture": meta,
        "provenance": gc.provenance(),
    }
    gc.save_artifact("w107b_g1", payload, arrays={
        "y_final": np.asarray(final.y),
        "theta": np.asarray(theta),
        "led_step": np.asarray(final.led_step),
        "led_renorm": np.asarray(final.led_renorm),
        "led_bc": np.asarray(final.led_bc),
        "led_rain": np.asarray(final.led_rain),
        "led_dt": np.asarray(final.led_dt),
        "n_0": np.asarray(final.pv.n_0),
        "dz": np.asarray(chem.dz),
    })
    print(f"[g1] termination={solve['termination']} "
          f"accept_count={solve['accept_count']} t_end={solve['t_end_s']:.3e}s "
          f"longdy={solve['longdy']:.3e} max_R={resid['max_R_s^-1']:.3e}/s "
          f"verdict={verdict}")


if __name__ == "__main__":
    main()
