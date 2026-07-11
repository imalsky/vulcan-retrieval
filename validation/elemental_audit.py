#!/usr/bin/env python3
"""Elemental-inventory audit of the abundance map over random prior draws.

For each draw theta = (lnZ, c_o, lnKzz) this verifies, at initialization, the four
quantities the science review requires at every retrieval point:

    (1) column elemental ratios He/H, O/H, C/H, N/H, S/H == the exact theta targets
    (2) dln(C/O) achieved == c_o
    (3) per-layer density closure sum_i n_i == P/(kB T)
    (4) conserved-atom anchor pv.atom_ini == atoms(y_init) in the runner's basis

plus the smallest elemental-repair factor (must stay > 0: a repair species driven
negative would mean the guess left the physical simplex). With --converge it also
re-converges each draw and reports the post-convergence drift of the column totals
against atom_ini (the runner's own conservation metric, now anchored exactly).

Run (GPU node or a patient workstation; ~minutes without --converge, chemistry-
solve-bound with it):

    python validation/elemental_audit.py --n 30 [--converge] [--mode elemental|masks]

Exit code 0 = all gates pass (elemental mode); masks mode reports the documented
leakage without failing (it exists to MEASURE the legacy knob's error).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# gates (elemental mode): the projection is exact up to the fixed-iteration
# residual; see vulcan_chem._ELEMENTAL_REPAIR_ITERS
GATE_RATIO = 1.0e-6
GATE_DENSITY = 1.0e-10
GATE_ATOM_INI = 1.0e-10
GATE_REPAIR_POSITIVE = 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="prior draws to audit")
    ap.add_argument("--mode", default="elemental", choices=["elemental", "masks"])
    ap.add_argument("--converge", action="store_true",
                    help="also re-converge each draw and report elemental drift")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from retrieval_framework.forward import config
    from retrieval_framework.forward import vulcan_chem

    profile = dict(config.FULL)
    profile.update(nz=50, count_max=5000, dt_max=1.0e11,
                   abundance_mode=args.mode, co_mode="fixed_O",
                   reanchor_atom_ini=True)
    chem = vulcan_chem.build_chem_model(profile)

    rng = np.random.default_rng(args.seed)
    # W39b production prior box for the abundance/mixing knobs (case.py::_W39B)
    lo = np.array([-2.303, -1.70, -4.6, 0.0])
    hi = np.array([+2.303, +0.24, +4.6, 0.0])
    draws = lo + (hi - lo) * rng.random((args.n, 4))

    worst = dict(ratio=0.0, dco=0.0, dens=0.0, atom=0.0, repair=np.inf)
    fails = 0
    for k, th in enumerate(draws):
        a = chem.audit_init(th)
        r = a.get("ratio_max_rel_err", np.nan)
        dco = abs(a.get("dln_CO_achieved", np.nan) - th[1])
        dens = a["density_closure_max_rel"]
        atom = a["atom_ini_max_rel_err"]
        rep = a.get("min_repair_factor", np.nan)
        if args.mode == "elemental":
            ok = (r < GATE_RATIO and dco < GATE_RATIO and dens < GATE_DENSITY
                  and atom < GATE_ATOM_INI and rep > GATE_REPAIR_POSITIVE)
        else:
            ok = True   # masks mode: measurement, not a gate
        fails += (not ok)
        worst["ratio"] = max(worst["ratio"], 0.0 if np.isnan(r) else r)
        worst["dco"] = max(worst["dco"], 0.0 if np.isnan(dco) else dco)
        worst["dens"] = max(worst["dens"], dens)
        worst["atom"] = max(worst["atom"], atom)
        worst["repair"] = min(worst["repair"], np.inf if np.isnan(rep) else rep)
        line = (f"[{k:02d}] lnZ={th[0]:+.2f} c_o={th[1]:+.2f} lnKzz={th[2]:+.2f} | "
                f"ratio_err={r:.2e} dCO={dco:.2e} dens={dens:.2e} atom_ini={atom:.2e}"
                + (f" min_repair={rep:.4f}" if not np.isnan(rep) else "")
                + ("" if ok else "  <-- FAIL"))
        print(line, flush=True)

        if args.converge:
            final, _init = chem.run_diag(np.asarray(th, np.float64))
            # drift of the converged column totals vs the (now-exact) anchor --
            # the runner's own conservation metric atom_loss = (atoms - atom_ini)/atom_ini
            a_run = np.asarray(final.atom_loss, np.float64)
            print(f"      converged: accept={int(final.accept_count)} "
                  f"max|atom_loss|={np.max(np.abs(a_run)):.3e} "
                  f"(runner drift metric vs exact atom_ini)", flush=True)

    print("\n==== elemental audit summary ====")
    print(f"mode={args.mode} draws={args.n} | worst ratio_err={worst['ratio']:.3e} "
          f"dCO={worst['dco']:.3e} density={worst['dens']:.3e} "
          f"atom_ini={worst['atom']:.3e} min_repair={worst['repair']:.4f}")
    if args.mode == "elemental":
        verdict = fails == 0
        print(f"VERDICT: {'PASS' if verdict else f'FAIL ({fails}/{args.n} draws)'} "
              f"(gates: ratio<{GATE_RATIO:g}, density<{GATE_DENSITY:g}, "
              f"atom_ini<{GATE_ATOM_INI:g}, repair>{GATE_REPAIR_POSITIVE:g})")
        return 0 if verdict else 1
    print("masks mode: leakage measured (no gate) -- compare against the elemental run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
