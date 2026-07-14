"""B0C G1 extended REFERENCE trajectory + two-stage workflow (collaborator
directive 2026-07-14).

The formal B0C verdict stays NO-GO. This establishes ONE checkpointed cold
reference endpoint driven to PHYSICAL sulfur-flux balance (not a step
count), then tests whether the deterministic two-stage warm workflow
reproduces it -- the prerequisite for replacing the 15000-step cold
requirement.

STAGE 1 (reference drive, looser operational criteria are explicitly
sanctioned for this one-time certification): a reference model built with
`trun_min` raised above any physical balance time, so the runner INTEGRATES
rather than early-certifies (its `ready` gate blocks both normal and stall
certification), driven in fixed step-budget chunks. Each chunk resets the
carry's `accept_count` to 0 and sets `chunk_target`, so the baked
`count_max` (which bounds the build warm-up and each chunk, safely) never
caps the TOTAL trajectory. The physics carry continues bit-for-bit across
chunk boundaries -- y, ymix, t, dt, the (y, t) convergence ring buffers,
`longdy_seen_min`, and `count_since_new_min` all persist; only the step
counter, its ring WRITE index (ring CONTENT persists, so the longdy
lookback still uses real history), the `count_min` ready-gate, and the
`accept_count % frq` cadences reset -- the last re-fire photo / atm-refresh
/ adapt-rtol at the boundary, which recompute from the UNCHANGED carry and
so do not perturb the state. Balance is judged HERE, by the sulfur ledger,
not by the runner: drive until the closure fraction
|dN_S/dt| / max(|Phi_bc|, |Phi_rain|) (= |Phi_bc - Phi_rain| / max(...),
since Phi_top = 0) stays below ROUTE_B_G1REF_CLOSURE for CONSEC consecutive
chunks.

STAGE 2 (strict certification, the SEPARATE production runner with the
NORMAL fixture config -- default trun_min, exact target tolerances, no
altered chemistry, no pin): warm-start three strict re-certifications from
the Stage-1 endpoint and require they return to the same state (RT-species
columns, sulfur inventory, rainout flux, binned spectrum, direct residual).

Reference-endpoint criteria, all measured and recorded:
  1. sulfur ledger closes (Stage-1 balance criterion above);
  2. inventory tendency near zero (closure for S AND H);
  3. Phi_bottom ~= Phi_rain + Phi_top(=0, explicit);
  4. direct steady-state residual at the endpoint;
  5. spectrum stable under further integration (last checkpoints + the
     strict recert endpoint vs the reference, binned);
  6. repeated STRICT warm re-certifications return to the same state;
  7. seed convergence (G2) runs SEPARATELY against this endpoint
     (g2_seeds.py, once the reference .npz exists).

Flux reconciliation (directive): the chunk history IS the sulfur arc, so
the earlier single-run 2.99e14 (t~8.9e11) and continuation-path 1.97e14
numbers are bracketed by the settled value recorded here. They differed
because the continuation probe re-projected state through `_prep` each
round (rebuilding the initial column) while the single run did not; THIS
reference does neither mid-trajectory -- one uninterrupted carry.

Nitrogen-radical irrelevance battery (directive: measure before touching
any floor -- mtol is NOT raised here): at the endpoint, for the measured
gating radicals (N, N2H2, N2H4) -- absolute |F| production-loss rate, share
of the N-element column inventory, peak mixing ratio, longdy contribution;
then a zero-and-strict-recertify perturbation measuring their effect on
major-species columns, sulfur throughput, and the binned spectrum, each
against the solver-noise floor from the strict recert twins. No chemistry
is removed.

Run (scheduled, VERY heavy -- 2 builds + up to ~150k steps + RT + recerts):
    ROUTE_B_W107B=1 python docs/route_b/harness/g1_reference.py
Env: W107B_NZ (100), ROUTE_B_G1REF_CLOSURE (0.02), G1REF_CHUNK (5000),
G1REF_MAX_CHUNKS (30), G1REF_CONSEC (2), G1REF_TRUN_MIN (1e18).
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
import w107b_spectrum as ws  # noqa: E402

N_RADICALS = ("N", "N2H2", "N2H4")
RT_SPECIES = ["H2O", "CO", "CO2", "CH4", "NH3", "H2S", "SO2", "SO", "S8"]


def ledger_rates(chem, state):
    led = gc.ledger_report(chem, state)
    dt = led["led_dt_s"]
    out = {}
    for e in ("S", "H"):
        phi_bc = led["led_bc"][e] / dt
        phi_rain = led["led_rain"][e]
        drift = (led["led_step"][e] + led["led_renorm"][e]
                 + led["led_bc"][e]) / dt
        out[e] = {"phi_bc": phi_bc, "phi_rain": phi_rain, "dN_dt": drift,
                  "closure_frac": abs(drift) / max(abs(phi_bc),
                                                   abs(phi_rain), 1e-300)}
    return out


def rt_columns(chem, y):
    y = np.asarray(y, dtype=np.float64)
    dz = np.asarray(chem.dz, dtype=np.float64)
    idx = [chem.sidx[s] for s in RT_SPECIES if s in chem.sidx]
    names = [s for s in RT_SPECIES if s in chem.sidx]
    return dict(zip(names, ((y[:, idx] * dz[:, None]).sum(axis=0)).tolist()))


def main():
    wf.require_env_gate()
    import jax.numpy as jnp

    nz = int(os.environ.get("W107B_NZ", "100"))
    chunk = int(os.environ.get("G1REF_CHUNK", "5000"))
    max_chunks = int(os.environ.get("G1REF_MAX_CHUNKS", "30"))
    closure_tol = float(os.environ.get("ROUTE_B_G1REF_CLOSURE", "0.02"))
    consec_need = int(os.environ.get("G1REF_CONSEC", "2"))
    trun_min = float(os.environ.get("G1REF_TRUN_MIN", "1e18"))

    # --- STAGE 1: reference-drive build (integrate, do not early-certify) --
    print(f"[ref] building reference-drive model (trun_min={trun_min:.1e} s "
          "-> runner integrates, never certifies)", flush=True)
    chem, meta = wf.build(nz=nz, count_max=15000,
                          cfg_extra={"trun_min": trun_min})
    theta = wf.FIDUCIAL_THETA
    integ = chem._integ

    print(f"[ref] cold start; chunks of {chunk} steps, closure tol "
          f"{closure_tol}, need {consec_need} consecutive", flush=True)
    state, atm_T = chem.prep_state(theta)
    total_steps = 0
    consec = 0
    history = []
    checkpoints = {}
    t0 = time.time()
    balanced = False
    for ck in range(max_chunks):
        state = state._replace(accept_count=jnp.int32(0),
                               chunk_target=jnp.int32(chunk))
        state = integ._runner(state, atm_T)
        state.y.block_until_ready()
        y_now = np.asarray(state.y, dtype=np.float64)
        if not np.all(np.isfinite(y_now)):
            print(f"[ref ck{ck:02d}] NON-FINITE state -- aborting drive",
                  flush=True)
            history.append({"chunk": ck, "non_finite": True})
            break
        n_acc = int(np.asarray(state.accept_count))
        total_steps += n_acc
        rates = ledger_rates(chem, state)
        cf = max(rates["S"]["closure_frac"], rates["H"]["closure_frac"])
        row = {"chunk": ck, "steps": n_acc, "total_steps": total_steps,
               "t_s": float(state.t), "dt_s": float(state.dt),
               "longdy": float(state.longdy),
               "S": rates["S"], "H": rates["H"], "worst_closure_frac": cf}
        history.append(row)
        checkpoints[f"y_ck{ck:02d}"] = y_now
        print(f"[ref ck{ck:02d}] steps+{n_acc} tot={total_steps} "
              f"t={row['t_s']:.3e} dt={row['dt_s']:.2e} "
              f"longdy={row['longdy']:.2e} "
              f"phi_bc_S={rates['S']['phi_bc']:.3e} "
              f"phi_rain_S={rates['S']['phi_rain']:.3e} "
              f"closure={cf:.3e} wall={time.time()-t0:.0f}s", flush=True)
        consec = consec + 1 if cf < closure_tol else 0
        if consec >= consec_need:
            balanced = True
            print(f"[ref] FLUX BALANCE reached at chunk {ck} "
                  f"(closure {cf:.3e} < {closure_tol} x{consec})", flush=True)
            break
    y_ref = np.asarray(state.y, dtype=np.float64)

    # endpoint residual on the reference model
    resid = gc.residual_report(chem, state, atm_T)
    endpoint_rates = ledger_rates(chem, state)
    print(f"[ref] endpoint max_R={resid['max_R_s^-1']:.3e}/s at "
          f"{resid['argmax_species']} z={resid['argmax_z']}; "
          f"closure S={endpoint_rates['S']['closure_frac']:.3e}", flush=True)

    # --- STAGE 2: strict certification build (NORMAL fixture config) -------
    print("[ref] building STRICT model (normal config; the certification "
          "runner)", flush=True)
    chem_s, meta_s = wf.build(nz=nz, count_max=15000)
    integ_s = chem_s._integ
    import importlib
    from retrieval_framework.forward import config as fcfg
    cfgmod = importlib.import_module(fcfg.W39B_CFG_MODULE)
    runtime = float(cfgmod.runtime)

    recert = []
    y_prev = y_ref
    for r in range(3):
        init_w, atm_w = chem_s.prep_state(theta, warm_y=y_prev)
        fin_w = integ_s._runner(init_w, atm_w)
        fin_w.y.block_until_ready()
        y_w = np.asarray(fin_w.y, dtype=np.float64)
        rates_w = ledger_rates(chem_s, fin_w)
        rep_w = gc.solve_report(chem_s, fin_w, 15000, runtime)
        recert.append({
            "round": r, "solve": rep_w, "rates": rates_w,
            "delta_vs_reference": gc.species_delta_report(chem_s, y_ref, y_w),
            "delta_vs_prev": gc.species_delta_report(chem_s, y_prev, y_w),
            "rt_columns": rt_columns(chem_s, y_w),
        })
        print(f"[ref recert{r}] {rep_w['termination']} "
              f"steps={rep_w['accept_count']} longdy={rep_w['longdy']:.2e} "
              f"rain_S={rates_w['S']['phi_rain']:.3e} "
              f"closure_S={rates_w['S']['closure_frac']:.3e} "
              f"max|dln n| vs ref={recert[-1]['delta_vs_reference']['max_abs_dln_n']:.3e}",
              flush=True)
        y_prev = y_w
    # solver-noise floor = last strict recert's own motion (rounds 2->3)
    noise_floor = recert[-1]["delta_vs_prev"]["max_abs_dln_n"]

    # --- spectrum stability across the last checkpoints + strict endpoint --
    print("[ref] building RT for spectrum-stability check", flush=True)
    rt, to_art = ws.build_rt(chem_s)
    B = ws.band_matrix(rt)
    last_keys = sorted(checkpoints)[-3:]
    spectra = {k: ws.spectrum_from_y(chem_s, rt, to_art, B, theta,
                                     checkpoints[k]) for k in last_keys}
    s_ref = ws.spectrum_from_y(chem_s, rt, to_art, B, theta, y_ref)
    s_recert = ws.spectrum_from_y(chem_s, rt, to_art, B, theta, y_prev)
    spec_stab = {k: float(np.max(np.abs(v - s_ref))) for k, v in spectra.items()}
    spec_stab["recert_final"] = float(np.max(np.abs(s_recert - s_ref)))
    print(f"[ref] spectrum stability (max |ddepth| vs endpoint): "
          f"{spec_stab}", flush=True)

    # --- nitrogen-radical irrelevance battery ------------------------------
    from vulcan_jax.steady_residual import residual_from_state
    rep_full = residual_from_state(chem_s._integ, fin_w, atm_w)
    F = np.asarray(rep_full.F, dtype=np.float64)
    dz = np.asarray(chem_s.dz, dtype=np.float64)
    compo = np.asarray(chem_s.compo_array, dtype=np.float64)
    nN = compo[:, fcfg.ATOM_COLS["N"]]
    N_col_total = float(((y_prev * nN[None, :]) * dz[:, None]).sum())
    nrad = {}
    for sp in N_RADICALS:
        i = chem_s.sidx[sp]
        col = float((y_prev[:, i] * dz).sum())
        nrad[sp] = {
            "column_cm^-2": col,
            "share_of_N_inventory": col * float(nN[i]) / N_col_total,
            "max_abs_F_cm^-3_s^-1": float(np.abs(F[:, i]).max()),
            "max_mix": float((y_prev[:, i] / y_prev.sum(axis=1)).max()),
        }
    # zero-and-strict-recertify perturbation
    y_zero = y_prev.copy()
    for sp in N_RADICALS:
        y_zero[:, chem_s.sidx[sp]] = 0.0
    init_z, atm_z = chem_s.prep_state(theta, warm_y=y_zero)
    fin_z = integ_s._runner(init_z, atm_z)
    fin_z.y.block_until_ready()
    y_z = np.asarray(fin_z.y, dtype=np.float64)
    rates_z = ledger_rates(chem_s, fin_z)
    spec_z = ws.spectrum_from_y(chem_s, rt, to_art, B, theta, y_z)
    zero_effect = {
        "major_delta_vs_recert": gc.species_delta_report(chem_s, y_prev, y_z),
        "rain_S_after_zeroing": rates_z["S"]["phi_rain"],
        "rain_S_recert": recert[-1]["rates"]["S"]["phi_rain"],
        "spectrum_max_abs_delta": float(np.max(np.abs(spec_z - s_recert))),
        "solver_noise_floor_recert": noise_floor,
    }
    zmax = zero_effect["major_delta_vs_recert"]["max_abs_dln_n"]
    print(f"[ref] N-radical zeroing: max|dln n|={zmax:.3e} "
          f"(noise floor {noise_floor:.3e}); "
          f"rain {zero_effect['rain_S_after_zeroing']:.3e} vs "
          f"{zero_effect['rain_S_recert']:.3e}; "
          f"spec delta {zero_effect['spectrum_max_abs_delta']:.3e}", flush=True)

    # --- verdict ----------------------------------------------------------
    recert_returns = all(
        r["delta_vs_reference"]["max_abs_dln_n"] < 5.0 * max(noise_floor, 1e-12)
        for r in recert)
    if balanced and recert_returns:
        verdict = ("REFERENCE ESTABLISHED + strict warm workflow reproduces "
                   "it (flux-balanced; recerts return within 5x noise floor)")
    elif balanced:
        verdict = ("REFERENCE ESTABLISHED but strict recerts do NOT cleanly "
                   "return -- inspect recertifications before adopting the "
                   "two-stage recipe")
    else:
        verdict = ("REFERENCE NOT REACHED (budget exhausted before flux "
                   "balance) -- more chunks or a continuation strategy needed")

    payload = {
        "gate": "G1-reference + two-stage workflow (directive 2026-07-14)",
        "verdict": verdict,
        "flux_balanced": bool(balanced),
        "strict_recerts_return": bool(recert_returns),
        "closure_tol": closure_tol,
        "trun_min_drive_s": trun_min,
        "total_steps": total_steps,
        "t_end_s": float(state.t),
        "chunk_history": history,
        "endpoint_rates": endpoint_rates,
        "endpoint_residual": resid,
        "recertifications": recert,
        "solver_noise_floor_recert": noise_floor,
        "spectrum_stability_max_abs_ddepth": spec_stab,
        "bands_um": ws.BANDS,
        "rt_columns_reference": rt_columns(chem_s, y_ref),
        "nitrogen_radicals": nrad,
        "nitrogen_zeroing_effect": zero_effect,
        "flux_reconciliation": (
            "chunk_history IS the sulfur arc; the earlier 2.99e14 "
            "(single-run t~8.9e11) and 1.97e14 (re-projected continuation "
            "path) are bracketed by the settled value here -- they differed "
            "because the continuation probe re-prepped state each round and "
            "this reference does not"),
        "theta": np.asarray(theta).tolist(),
        "fixture": meta,
        "strict_fixture": meta_s,
        "provenance": gc.provenance(),
    }
    arrays = dict(checkpoints)
    arrays["y_reference"] = y_ref
    arrays["y_recert_final"] = y_prev
    gc.save_artifact("w107b_g1ref", payload, arrays=arrays)
    print(f"[ref] {verdict}", flush=True)


if __name__ == "__main__":
    main()
