"""Old-vs-corrected offset-model comparison for the Z-C/O Fisher forecasts.

For every instrument combination and chemistry tier, report under both offset
models (reference_fixed = pre-2026-07-20 figures, first group assumed to define
the absolute transit-depth baseline; all_groups = corrected default, every
group carries a depth offset):

    sigma(log10 Z), sigma(log10 C/O)  [dex, marginalized]
    rho(Z, C/O)                       [marginal correlation]
    sigma(lnR0)                       [ln units]
    whitened Fisher rank / dimension, eigenvalue extremes

Reads output/zco_jacobians.npz. Writes a markdown table to stdout and to
output/zco_offset_model_comparison.md.

Run (base env, from the repo root):
    python scripts/zco_information/compare_offset_models.py
"""
from __future__ import annotations

import sys

import numpy as np

import zco_lib as Z

COMBOS = [
    ["PRISM"], ["NIRISS"], ["G395H"], ["NIRCam"],
    ["NIRISS", "G395H"],
    ["PRISM", "NIRISS", "G395H", "NIRCam"],
]
LN10 = np.log(10.0)


def _row(payload, wl_model, obs, offset_model):
    des = Z.build_design(payload, wl_model, obs, offset_model=offset_model)
    dg = {}
    F, C = Z.fisher(des["J"], des["sigma"], diag=dg)
    C2 = C[np.ix_(des["interest"], des["interest"])]
    out = dict(params=len(des["keys"]), rank=dg["rank"], dim=dg["dimension"],
               wmin=float(dg["eigenvalues"].min()), wmax=float(dg["eigenvalues"].max()))
    if np.all(np.isfinite(C2)):
        out["sZ"] = float(np.sqrt(C2[0, 0]) / LN10)
        out["sCO"] = float(np.sqrt(C2[1, 1]) / LN10)
        out["rho"] = float(C2[0, 1] / np.sqrt(C2[0, 0] * C2[1, 1]))
    else:
        out["sZ"] = out["sCO"] = out["rho"] = float("inf")
    if "lnR0" in des["keys"]:
        v = C[des["keys"].index("lnR0")][des["keys"].index("lnR0")]
        out["sR0"] = float(np.sqrt(v)) if np.isfinite(v) else float("inf")
    else:
        out["sR0"] = float("nan")
    return out


def _fmt(x, n=3):
    if not np.isfinite(x):
        return "inf"
    return f"{x:.{n}f}"


def main():
    wl_model, tiers, meta = Z.load_jacobians()
    lines = []
    lines.append("# Z-C/O Fisher: offset-model comparison (2026-07-20 correction)")
    lines.append("")
    lines.append("ref = reference_fixed (old: first group assumed absolutely "
                 "depth-calibrated); all = all_groups (corrected default). "
                 "sigma in dex (marginalized); sigma(lnR0) in ln units; "
                 "whitened rank and eigenvalue extremes from the rank-aware inversion.")
    lines.append("")
    lines.append("| combo | tier | sZ ref | sZ all | dsZ % | sCO ref | sCO all | dsCO % "
                 "| rho ref | rho all | s(lnR0) ref | s(lnR0) all | rank all | wmin all | wmax all |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for combo in COMBOS:
        obs = Z.load_combined(combo)
        for tier in meta["tiers"]:
            r_ref = _row(tiers[tier], wl_model, obs, "reference_fixed")
            r_all = _row(tiers[tier], wl_model, obs, "all_groups")
            dz = (100.0 * (r_all["sZ"] / r_ref["sZ"] - 1.0)
                  if np.isfinite(r_all["sZ"]) and np.isfinite(r_ref["sZ"]) else float("inf"))
            dco = (100.0 * (r_all["sCO"] / r_ref["sCO"] - 1.0)
                   if np.isfinite(r_all["sCO"]) and np.isfinite(r_ref["sCO"]) else float("inf"))
            lines.append(
                f"| {'+'.join(combo)} | {tier} "
                f"| {_fmt(r_ref['sZ'])} | {_fmt(r_all['sZ'])} | {_fmt(dz, 1)} "
                f"| {_fmt(r_ref['sCO'])} | {_fmt(r_all['sCO'])} | {_fmt(dco, 1)} "
                f"| {_fmt(r_ref['rho'], 3)} | {_fmt(r_all['rho'], 3)} "
                f"| {_fmt(r_ref['sR0'], 4)} | {_fmt(r_all['sR0'], 4)} "
                f"| {r_all['rank']}/{r_all['dim']} "
                f"| {r_all['wmin']:.2e} | {r_all['wmax']:.2e} |")
    text = "\n".join(lines) + "\n"
    print(text)
    out = Z.DATA / "zco_offset_model_comparison.md"
    out.write_text(text)
    print(f"[written] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
