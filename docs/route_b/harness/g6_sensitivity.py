"""B0C G6: solver-map sensitivity vs independently reconverged FD (W107b).

The D9 production derivative route, run on the ACTIVE-RAINOUT fixture:
VULCAN-JAX's validated solver-map machinery (renorm map, log-abundance
coordinates, measured-null deflation, host LGMRES) extended for smooth
rainout -- `make_body_terms` packs the RainoutTerm into the body map, the
theta-map `rebuild(theta)` reproduces the retrieval's `_prep` on-graph
(rates, atmosphere, conden arrays, lookup boundary pin) and supplies the
smooth extras so dG/dtheta carries d(n_sat)/dT, d(C)/dDzz, and the boundary
derivative. One adjoint solve per loss yields dL/dtheta for ALL theta
components; FD re-converges every endpoint independently.

Gate criteria (plan sections 6/8): AD vs FD agreement on the chemistry
endpoint losses AND one binned-spectrum row with no qualitative
disagreement (sign, order of magnitude); FD stable across h and h/2 with
"converged" termination at EVERY endpoint (each recorded separately);
FD signal >= 10x the measured convergence-noise floor; spectral
correlation > 0.99 and amplitude within 10% (target 5%) on the spectrum
row. The diagnostic unrolled jvp is NOT run here (measured invalid 6-9
orders on the cold fixture, B0-6; recorded in the record).

Losses: ln column N_S8, ln column N_H2S, ln Phi_rain,S, and one binned
spectrum band (band index SPEC_BAND of w107b_spectrum.BANDS; the spectrum
loss adds the direct RT-theta term -- jacfwd at fixed y* -- to the
adjoint's chemistry path before comparing against FD, which measures the
total).

REFUSES to run unless the nominal solve terminates converged-gate with
Phi_rain,S > 0 (G6 is only meaningful on a G1-quality active-rainout
state). Run (scheduled, VERY heavy -- ~18 solves + 4 adjoint ensembles):
    ROUTE_B_W107B=1 python docs/route_b/harness/g6_sensitivity.py
Env: W107B_NZ, W107B_COUNT_MAX, ROUTE_B_G6_BODY_DT (default 1e7).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w107b_fixture as wf  # noqa: E402
import gate_common as gc  # noqa: E402
import w107b_spectrum as ws  # noqa: E402

SPEC_BAND = 4                      # 3.90-3.9625 um (SO2/H2S-sensitive)
FD_PARAMS = {"lnZ": (0, 0.05), "c_o": (1, 0.05),
             "lnKzz": (2, 0.2), "Tirr": (3, 2.0)}
LOSSES = ("lnN_S8", "lnN_H2S", "lnPhi_rain_S", "spec_band")


def main():
    wf.require_env_gate()
    import importlib

    import jax
    import jax.numpy as jnp

    from retrieval_framework.forward import config

    nz = int(os.environ.get("W107B_NZ", "100"))
    count_max = int(os.environ.get("W107B_COUNT_MAX", "15000"))
    body_dt = float(os.environ.get("ROUTE_B_G6_BODY_DT", "1e7"))

    chem, meta = wf.build(nz=nz, count_max=count_max)
    cfgmod = importlib.import_module(config.W39B_CFG_MODULE)
    runtime = float(cfgmod.runtime)
    theta0 = np.asarray(wf.FIDUCIAL_THETA, dtype=np.float64)

    from vulcan_jax import chem_funs
    from vulcan_jax.conden import RainoutTerm, smooth_rainout_loss
    from vulcan_jax.steady_state_grad import (
        make_body_terms,
        steady_state_input_sensitivity,
    )

    integ = chem._integ
    st = integ._statics

    def solve(theta):
        chem.pin_value(theta)     # host domain guard on every visited point
        init, atm_T = chem.prep_state(theta)
        fin = integ._runner(init, atm_T)
        fin.y.block_until_ready()
        return fin, atm_T, gc.solve_report(chem, fin, count_max, runtime)

    print("[g6] nominal solve")
    fin, atm_T0, rep0 = solve(theta0)
    led0 = gc.ledger_report(chem, fin)
    rain0 = led0["led_rain"]["S"]
    if rep0["termination"] != "converged-gate" or rain0 <= 0.0:
        raise SystemExit(
            f"[g6] REFUSED: nominal termination={rep0['termination']}, "
            f"Phi_rain,S={rain0:.3e}. G6 requires a converged ACTIVE-rainout "
            "state (G1 first).")

    # --- body map at the converged state (rainout packed) ----------------
    atm_step, terms = make_body_terms(integ, fin, atm_T0)
    if terms.rainout is None:
        raise RuntimeError("make_body_terms returned no RainoutTerm on a "
                           "smooth-rainout state")
    y_star, k_star = fin.y, fin.k_arr
    net_jax = chem_funs._NET_JAX

    # measured-null deflation: closed elements only (open S/H budgets are
    # fixed by the boundary + sink, their directions are NOT null here)
    gate_mask = np.asarray(st.gate_atom_mask, dtype=bool)
    compo_run = np.asarray(integ._compo_arr, dtype=np.float64)
    compo_deflate = jnp.asarray(compo_run * gate_mask[None, :].astype(np.float64))
    atoms = list(integ._atom_order)
    i_s_atom = atoms.index("S")
    rain_atoms_S = float(np.asarray(st.rainout_atoms)[i_s_atom])  # 8 for S8

    # photo rows stay frozen from the converged k (their theta-dependence
    # is host-baked cross sections; the y-feedback dJ/dy rides
    # photo_recompute_k="auto")
    from vulcan_jax import network as net_mod
    from vulcan_jax._paths import resolve_data_path
    network = net_mod.parse_network(str(resolve_data_path(cfgmod.network)))
    photo_rows = jnp.asarray(np.asarray(network.is_photo, dtype=bool))

    row = int(st.rainout_re_row)
    r_scale = float(st.rainout_scale) * float(st.rainout_coeff)

    def rebuild(theta):
        init, atm_p = chem.prep_state(theta)
        k_p = jnp.where(photo_rows[:, None], k_star, init.k_arr)
        # converged refresh geometry spliced (frozen-by-design cascade,
        # second-order; exactly the fields make_body_terms spliced)
        atm_p = atm_p._replace(g=atm_step.g, dzi=atm_step.dzi,
                               Hpi=atm_step.Hpi, top_flux=atm_step.top_flux,
                               vs=atm_step.vs, vm=atm_step.vm)
        extras = {
            "rainout": RainoutTerm(
                C=r_scale * init.pv.c_Dg_per_re[row],
                n_sat=init.pv.c_sat_n_per_re[row],
                w=float(st.rainout_w),
                sp_mask=st.rainout_sp_mask,
            ),
            "bot_val": init.pv.bot_pin_mix * init.pv.n_0[0],
        }
        return k_p, atm_p, extras

    # --- losses -----------------------------------------------------------
    dz_j = jnp.asarray(np.asarray(chem.dz, dtype=np.float64))
    i_s8 = chem.sidx["S8"]
    i_h2s = chem.sidx["H2S"]
    rt, to_art = ws.build_rt(chem)
    B = ws.band_matrix(rt)
    B_row = jnp.asarray(B[SPEC_BAND])
    term0 = terms.rainout

    def loss_lnN_S8(y):
        return jnp.log(jnp.sum(y[:, i_s8] * dz_j))

    def loss_lnN_H2S(y):
        return jnp.log(jnp.sum(y[:, i_h2s] * dz_j))

    def loss_lnPhi_rain(y):
        L, _ = smooth_rainout_loss(y[:, i_s8], term0.C, term0.n_sat, term0.w)
        return jnp.log(rain_atoms_S * jnp.sum(L * dz_j))

    tp_eval = wf.make_tp_eval()
    p_art_j = jnp.asarray(np.asarray(rt.p_art_bar))
    mols = list(rt.molecules)

    def _band_depth(y, th):
        n_tot = jnp.sum(y, axis=1)
        vmr = {m: to_art(y[:, chem.sidx[m]] / n_tot) for m in mols}
        vmr_h2 = to_art(y[:, chem.sidx["H2"]] / n_tot)
        vmr_he = to_art(y[:, chem.sidx["He"]] / n_tot)
        mmw = to_art((y @ chem.species_masses) / n_tot)
        T_art = tp_eval(th[3:3 + wf.N_TP], p_art_j)
        depth = rt.transmission_depth(vmr, vmr_h2, T_art, mmw, vmr_he=vmr_he)
        return jnp.sum(B_row * depth)

    th0_j = jnp.asarray(theta0)

    def loss_spec(y):
        return _band_depth(y, th0_j)

    loss_fns = {"lnN_S8": loss_lnN_S8, "lnN_H2S": loss_lnN_H2S,
                "lnPhi_rain_S": loss_lnPhi_rain, "spec_band": loss_spec}

    # --- adjoint solves (chemistry path), one per loss --------------------
    ad = {}
    infos = {}
    for name, lf in loss_fns.items():
        print(f"[g6] adjoint solve: {name}")
        g, info = steady_state_input_sensitivity(
            lf, y_star, k_star, atm_step, net_jax, th0_j, rebuild,
            compo_array=compo_deflate, dz=dz_j, body_dt=body_dt,
            photo_recompute_k="auto", converged_state=fin, integ=integ,
            runner_photo_static=getattr(integ._odesolver, "_photo_static",
                                        None),
            return_info=True)
        ad[name] = np.asarray(g, dtype=np.float64)
        infos[name] = {k: (float(v) if np.isscalar(v) or hasattr(v, "item")
                           else v)
                       for k, v in info.items()
                       if k in ("fp_err", "null_quality", "resid",
                                "resid_median", "ensemble_spread",
                                "n_matvec", "body_dt", "solver_map",
                                "photo_feedback")}
        print(f"[g6]   resid={infos[name]['resid']:.3e} "
              f"fp_err={infos[name]['fp_err']:.3e} "
              f"null_q={infos[name]['null_quality']:.3e}")

    # direct RT-theta term for the spectrum loss (fixed y*)
    d_direct = np.asarray(
        jax.jacfwd(lambda th: _band_depth(y_star, th))(th0_j),
        dtype=np.float64)
    ad["spec_band"] = ad["spec_band"] + d_direct

    # --- noise floor: warm re-certification twin at theta0 ----------------
    print("[g6] noise-floor twin (warm re-certification at theta0)")
    init_b, atm_b = chem.prep_state(theta0, warm_y=np.asarray(y_star))
    fin_b = integ._runner(init_b, atm_b)
    fin_b.y.block_until_ready()
    floor = {n: abs(float(lf(fin_b.y)) - float(lf(y_star)))
             for n, lf in loss_fns.items()}
    floor["spec_band"] = abs(float(_band_depth(fin_b.y, th0_j))
                             - float(_band_depth(y_star, th0_j)))

    # --- independently reconverged FD --------------------------------------
    fd = {n: {} for n in LOSSES}
    endpoints = {"nominal": rep0}
    for pname, (pi, h) in FD_PARAMS.items():
        for hh in (h, 0.5 * h):
            Lp, Lm = {}, {}
            for sgn, out in ((1.0, Lp), (-1.0, Lm)):
                th = theta0.copy()
                th[pi] += sgn * hh
                print(f"[g6] FD solve {pname} {'+' if sgn > 0 else '-'}{hh}")
                f, _a, rep = solve(th)
                endpoints[f"{pname}{'+' if sgn > 0 else '-'}{hh}"] = rep
                for n, lf in loss_fns.items():
                    out[n] = float(lf(f.y)) if n != "spec_band" else float(
                        _band_depth(f.y, jnp.asarray(th)))
            for n in LOSSES:
                fd[n].setdefault(pname, {})[f"h={hh}"] = (
                    (Lp[n] - Lm[n]) / (2.0 * hh))

    # --- comparison table ---------------------------------------------------
    table = {}
    worst_rel = 0.0
    qualitative_fail = []
    for n in LOSSES:
        table[n] = {}
        for pname, (pi, h) in FD_PARAMS.items():
            f1 = fd[n][pname][f"h={h}"]
            f2 = fd[n][pname][f"h={0.5 * h}"]
            a = float(ad[n][pi])
            # signal-vs-floor: |L(+h/2)-L(-h/2)| ~ 2(h/2)|dL| vs twin floor
            signal = 2.0 * (0.5 * h) * abs(f2)
            floor_ok = signal >= 10.0 * floor[n] if floor[n] > 0 else True
            rel = abs(a - f2) / max(abs(f2), 1e-300)
            h_stab = abs(f1 - f2) / max(abs(f2), 1e-300)
            entry = {"ad": a, "fd_h": f1, "fd_h2": f2,
                     "ad_vs_fd_rel": rel, "h_vs_h2_rel": h_stab,
                     "fd_signal": signal, "noise_floor": floor[n],
                     "signal_ge_10x_floor": bool(floor_ok)}
            table[n][pname] = entry
            if floor_ok:
                worst_rel = max(worst_rel, rel)
                if (a * f2 < 0 and abs(f2) > 0) or (
                        abs(f2) > 0 and not (0.1 < abs(a) / abs(f2) < 10.0)):
                    qualitative_fail.append((n, pname))

    nonconv = {k: v["termination"] for k, v in endpoints.items()
               if v["termination"] != "converged-gate"}
    if nonconv:
        verdict = f"TAINTED (non-converged FD endpoints: {nonconv})"
    elif qualitative_fail:
        verdict = (f"FAIL (qualitative disagreement -- sign or order of "
                   f"magnitude -- on {qualitative_fail})")
    else:
        verdict = (f"MEASURED (worst floor-passing AD-vs-FD rel "
                   f"{worst_rel:.3e}; amplitude gate 10%/target 5% applies "
                   "per-entry; see table)")

    payload = {
        "gate": "G6",
        "verdict": verdict,
        "theta0": theta0.tolist(),
        "phi_rain_S_nominal": rain0,
        "nominal_solve": rep0,
        "body_dt": body_dt,
        "losses": list(LOSSES),
        "spec_band_um": ws.BANDS[SPEC_BAND],
        "adjoint_dL_dtheta": {n: ad[n].tolist() for n in LOSSES},
        "adjoint_info": infos,
        "spec_direct_rt_theta_term": d_direct.tolist(),
        "fd": fd,
        "comparison": table,
        "noise_floor_twin": floor,
        "endpoint_solves": endpoints,
        "nonconverged_endpoints": nonconv,
        "unrolled_jvp": "NOT RUN (measured invalid 6-9 orders on the cold "
                        "fixture, B0-6; diagnostic-only per D9)",
        "theta_names": wf.THETA_NAMES,
        "fixture": meta,
        "provenance": gc.provenance(),
    }
    gc.save_artifact("w107b_g6", payload, arrays={"y_star": np.asarray(y_star)})
    for n in LOSSES:
        for pname in FD_PARAMS:
            e = table[n][pname]
            print(f"[g6] {n:14s} d/d{pname:6s}: AD {e['ad']:+.4e} "
                  f"FD(h/2) {e['fd_h2']:+.4e} rel {e['ad_vs_fd_rel']:.2e} "
                  f"h-stab {e['h_vs_h2_rel']:.2e} "
                  f"floor-ok {e['signal_ge_10x_floor']}")
    print(f"[g6] verdict: {verdict}")


if __name__ == "__main__":
    main()
