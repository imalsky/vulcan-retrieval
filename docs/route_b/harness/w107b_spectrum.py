"""B0C spectrum harness: binned W107b transmission spectra + FD derivative.

Three duties (directive items G2/G5a spectrum columns + K):

1. `spectrum_from_y(chem, rt, theta, y)` -- binned transit depths for a
   CONVERGED chemistry state, through the retrieval's production RT
   (`exojax_rt.build_rt_model` transmission depth: H2-He CIA required,
   Rayleigh on) with the live Guillot T(P) on the ART grid. Binning is the
   documented harness operator: top-hat band averages (BANDS below). The
   tool's count-space instrument operator is B1-11 production wiring; every
   artifact states this substitution explicitly.

2. npz mode -- `python w107b_spectrum.py npz <g2_or_g5_artifact>.npz`:
   compute spectra for every saved y_* state of a G2/G4/G5 artifact
   (NO chemistry solves) and report the pairwise band-depth agreement.
   This is the measured spectrum column that G2 and G5a need to close.
   T(P) is the FIDUCIAL profile unless the states were solved at a theta
   with different T-P parameters (G5 lnZ/lnKzz rungs leave T(P) untouched;
   G2 seeds share the fiducial theta; G4's hot state must NOT be fed here).

3. fd mode -- `ROUTE_B_W107B=1 python w107b_spectrum.py fd`:
   the full binned-spectrum derivative on the ACTIVE-RAINOUT fixture
   (directive K; the zero-rainout cold-spectrum harness does not count):
   independently reconverged centered FD at h and h/2 for lnZ and Tirr,
   with per-endpoint termination + residual metadata (checkpoint-review
   requirement: nominal, +h, -h, +h/2, -h/2 each recorded). This is the FD
   side of the G6 spectrum row; the solver-map AD side compares against it.

Requires data/opacity_cache/ + data/exojax_linelists/ (same as the B0-6
spectrum probe). Heavy in fd mode (9 cold solves); npz mode is cheap.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w107b_fixture as wf  # noqa: E402
import gate_common as gc  # noqa: E402

# Documented harness bands (um): eight top-hats over the H2S/SO2-sensitive
# 3.65-4.15 um window the opacity cache covers (H2O/CO/CH4/H2S/SO2).
BANDS = [(3.65 + 0.0625 * i, 3.65 + 0.0625 * (i + 1)) for i in range(8)]
RT_PROFILE = dict(
    molecules=["H2O", "CO", "CH4", "H2S", "SO2"],
    nu_min=2400.0, nu_max=2800.0, nu_pts=800,   # ~3.57-4.17 um, R ~ 5000
    art_nlayer=60,
    use_rayleigh=True,
)

FD_PARAMS = {  # theta index, step h (h/2 runs too)
    "lnZ": (0, 0.05),
    "Tirr": (3, 2.0),
}


def build_rt(chem):
    """Production RT model + VULCAN->ART interpolation map for the fixture."""
    from retrieval_framework.forward import config as fcfg
    from retrieval_framework.forward import exojax_rt, interp_map

    prof = dict(RT_PROFILE)
    prof["broadening"] = str(getattr(fcfg, "BROADENING", "air"))
    prof["rp_cm"] = wf.RP_RJUP * wf.R_JUP_CM
    prof["gs_cgs"] = wf.GS_CGS
    prof["rstar_cm"] = wf.RSTAR_RSUN * wf.R_SUN_CM
    rt = exojax_rt.build_rt_model(prof)
    to_art = interp_map.make_to_art(np.asarray(chem.p_bar),
                                    np.asarray(rt.p_art_bar))
    return rt, to_art


def band_matrix(rt) -> np.ndarray:
    """(n_bands, n_wl) top-hat averaging operator over BANDS."""
    wl = np.asarray(rt.wl_um)
    B = np.zeros((len(BANDS), wl.shape[0]))
    for b, (lo, hi) in enumerate(BANDS):
        m = (wl >= lo) & (wl < hi)
        if not m.any():
            raise RuntimeError(f"band {lo}-{hi} um has no wavelength cells")
        B[b, m] = 1.0 / m.sum()
    return B


def spectrum_from_y(chem, rt, to_art, B, theta, y):
    """Binned transit depths (n_bands,) for a converged y at theta."""
    import jax.numpy as jnp

    tp_eval = wf.make_tp_eval()
    y = jnp.asarray(y)
    n_tot = jnp.sum(y, axis=1)
    mols = list(rt.molecules)
    vmr = {m: to_art(y[:, chem.sidx[m]] / n_tot) for m in mols}
    vmr_h2 = to_art(y[:, chem.sidx["H2"]] / n_tot)
    vmr_he = to_art(y[:, chem.sidx["He"]] / n_tot)   # He CIA partner, REQUIRED
    # mean molecular weight from composition (g/mol); state-only, no carry
    mmw_v = (y @ chem.species_masses) / n_tot
    mmw = to_art(mmw_v)
    th = jnp.asarray(theta, dtype=jnp.float64)
    T_art = tp_eval(th[3:3 + wf.N_TP], jnp.asarray(np.asarray(rt.p_art_bar)))
    depth = rt.transmission_depth(vmr, vmr_h2, T_art, mmw, vmr_he=vmr_he)
    return np.asarray(B @ np.asarray(depth), dtype=np.float64)


def run_npz(npz_path: Path):
    """Spectrum-agreement columns for a saved gate artifact (cheap)."""
    dat = np.load(npz_path)
    names = [k for k in dat.files if k.startswith("y_")]
    if not names:
        raise SystemExit(f"{npz_path} carries no y_* state arrays")
    jpath = Path(str(npz_path).replace(".npz", ".json"))
    theta = np.asarray(json.loads(jpath.read_text())["theta"]) \
        if jpath.exists() else wf.FIDUCIAL_THETA

    print("[spec-npz] building fixture chem model for RT wiring ...")
    chem, meta = wf.build(nz=int(os.environ.get("W107B_NZ", "100")))
    rt, to_art = build_rt(chem)
    B = band_matrix(rt)

    spectra = {n: spectrum_from_y(chem, rt, to_art, B, theta, dat[n])
               for n in names}
    ref = names[0]
    comps = {}
    for n in names[1:]:
        d = np.abs(spectra[n] - spectra[ref])
        r = d / np.maximum(np.abs(spectra[n]), np.abs(spectra[ref]))
        comps[n] = {"max_abs_depth_delta": float(d.max()),
                    "max_rel_depth_delta": float(r.max())}
        print(f"[spec-npz] {n} vs {ref}: max |ddepth| {d.max():.3e} "
              f"(rel {r.max():.3e})")
    payload = {
        "harness": "w107b_spectrum npz",
        "source_artifact": npz_path.name,
        "bands_um": BANDS,
        "binning_operator": "top-hat band average (harness; tool "
                            "count-space operator is B1-11)",
        "theta": np.asarray(theta).tolist(),
        "reference_state": ref,
        "spectra": {n: s.tolist() for n, s in spectra.items()},
        "comparisons": comps,
        "fixture": meta,
        "provenance": gc.provenance(),
    }
    gc.save_artifact(f"w107b_spectrum_{npz_path.stem}", payload)


def run_fd():
    """Directive K: reconverged-FD binned-spectrum derivative rows."""
    wf.require_env_gate()
    import importlib

    from retrieval_framework.forward import config

    nz = int(os.environ.get("W107B_NZ", "100"))
    count_max = int(os.environ.get("W107B_COUNT_MAX", "15000"))
    chem, meta = wf.build(nz=nz, count_max=count_max)
    cfgmod = importlib.import_module(config.W39B_CFG_MODULE)
    runtime = float(cfgmod.runtime)
    rt, to_art = build_rt(chem)
    B = band_matrix(rt)

    def solve(theta):
        chem.pin_value(theta)   # per-point domain validation (loud)
        init, atm_T = chem.prep_state(theta)
        fin = chem._integ._runner(init, atm_T)
        fin.y.block_until_ready()
        rep = gc.solve_report(chem, fin, count_max, runtime)
        led = gc.ledger_report(chem, fin)
        return (np.asarray(fin.y, dtype=np.float64), rep,
                led["led_rain"]["S"])

    theta0 = np.asarray(wf.FIDUCIAL_THETA, dtype=np.float64)
    print("[spec-fd] nominal solve")
    y0, rep0, rain0 = solve(theta0)
    if rain0 <= 0.0:
        raise SystemExit(
            f"[spec-fd] REFUSED: Phi_rain,S = {rain0:.3e} at the nominal "
            "endpoint -- directive K requires the ACTIVE-rainout fixture; "
            "a zero-rainout spectrum derivative does not count.")
    s0 = spectrum_from_y(chem, rt, to_art, B, theta0, y0)

    rows = {}
    endpoints = {"nominal": rep0}
    for pname, (pi, h) in FD_PARAMS.items():
        rows[pname] = {}
        for hh in (h, 0.5 * h):
            sp, sm = None, None
            for sgn, tag in ((1.0, "+"), (-1.0, "-")):
                th = theta0.copy()
                th[pi] += sgn * hh
                print(f"[spec-fd] solve {pname} {tag}{hh}")
                y, rep, _rain = solve(th)
                endpoints[f"{pname}{tag}{hh}"] = rep
                s = spectrum_from_y(chem, rt, to_art, B, th, y)
                if sgn > 0:
                    sp = s
                else:
                    sm = s
            rows[pname][f"h={hh}"] = ((sp - sm) / (2.0 * hh)).tolist()
        d1 = np.asarray(rows[pname][f"h={h}"])
        d2 = np.asarray(rows[pname][f"h={0.5 * h}"])
        rel = np.abs(d1 - d2) / np.maximum(np.abs(d2), 1e-300)
        rows[pname]["h_vs_h2_max_rel"] = float(rel.max())
        print(f"[spec-fd] d(depth)/d{pname}: h-vs-h/2 max rel "
              f"{rel.max():.3e}")

    nonconv = {k: v["termination"] for k, v in endpoints.items()
               if v["termination"] != "converged-gate"}
    payload = {
        "harness": "w107b_spectrum fd (directive K; FD side of G6 spectrum)",
        "bands_um": BANDS,
        "binning_operator": "top-hat band average (harness; tool "
                            "count-space operator is B1-11)",
        "theta0": theta0.tolist(),
        "phi_rain_S_nominal": rain0,
        "spectrum_nominal": s0.tolist(),
        "fd_rows": rows,
        "endpoint_solves": endpoints,
        "nonconverged_endpoints": nonconv,
        "verdict": ("MEASURED" if not nonconv else
                    f"TAINTED (non-converged endpoints: {nonconv})"),
        "fixture": meta,
        "provenance": gc.provenance(),
    }
    gc.save_artifact("w107b_spectrum_fd", payload)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("npz", "fd"):
        raise SystemExit("usage: w107b_spectrum.py npz <artifact.npz> | fd")
    if sys.argv[1] == "npz":
        if len(sys.argv) != 3:
            raise SystemExit("usage: w107b_spectrum.py npz <artifact.npz>")
        run_npz(Path(sys.argv[2]))
    else:
        run_fd()


if __name__ == "__main__":
    main()
