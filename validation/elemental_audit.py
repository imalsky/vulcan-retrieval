"""Elemental-inventory audit of the abundance map over random prior draws.

    python validation/elemental_audit.py --spec runs/<case>/<name>.yaml --n 30 [--converge]

Draws the case's chemistry block (lnZ, c_o, lnKzz and the extras it infers, s_o among them) from its priors, and checks
the column ratios to H against the theta targets, dln(C/O) against c_o, the per-layer density closure, atom_ini
against the initial column, and that the smallest repair factor stays above 0. With --converge it also reports the
post-convergence drift of the column totals against atom_ini. Exit code 0 means all gates pass.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# PYTHONSAFEPATH strips the script's own directory from sys.path in some
# sandboxes, so be explicit rather than relying on it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifact

# gates: the projection is exact up to the fixed-iteration
# residual; see vulcan_forward._elemental._ELEMENTAL_REPAIR_ITERS
GATE_RATIO = 1.0e-6
GATE_DENSITY = 1.0e-10
GATE_ATOM_INI = 1.0e-10
GATE_REPAIR_POSITIVE = 0.0


def _report_drift(chem, th) -> None:
    """Re-converge one draw and print the runner's conservation metric, atom_loss = (atoms - atom_ini) / atom_ini."""
    final, _init = chem.run_diag(np.asarray(th, np.float64))
    a_run = np.asarray(final.atom_loss, np.float64)
    print(
        f"      converged: accept={int(final.accept_count)} "
        f"max|atom_loss|={np.max(np.abs(a_run)):.3e} "
        f"(runner drift metric vs exact atom_ini)",
        flush=True,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="prior draws to audit")
    ap.add_argument("--converge", action="store_true", help="also re-converge each draw and report elemental drift")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--spec", required=True, help="the run specification whose case to audit")
    args = ap.parse_args()

    from retrieval_framework import config_schema as C

    cfg, results = _artifact.spec_case(args.spec)
    C.export_network(cfg)  # before vulcan3 is imported
    from vulcan_forward import vulcan_chem

    # The case supplies the profile and the chemistry prior box. The chemistry is built without the T-P hook, so every
    # draw uses the VULCAN baseline T-P (the last theta slot, a 0 K shift), not the case's Guillot T-P.
    profile = cfg.profile()
    chem = vulcan_chem.build_chem_model(profile, chem_extras=C.chem_extras(cfg))
    chem_specs = [s for s in C.specs_from_config(cfg) if s.kind == "chem"]
    names = [s.name for s in chem_specs]

    rng = np.random.default_rng(args.seed)
    lo = np.array([s.lo for s in chem_specs] + [0.0])
    hi = np.array([s.hi for s in chem_specs] + [0.0])
    draws = lo + (hi - lo) * rng.random((args.n, len(lo)))

    worst = {"ratio": 0.0, "dco": 0.0, "dens": 0.0, "atom": 0.0, "repair": np.inf}
    fails = 0
    for k, th in enumerate(draws):
        a = chem.audit_init(th)
        r = a.get("ratio_max_rel_err", np.nan)
        dco = abs(a.get("dln_CO_achieved", np.nan) - th[1])
        dens = a["density_closure_max_rel"]
        atom = a["atom_ini_max_rel_err"]
        rep = a.get("min_repair_factor", np.nan)
        ok = (
            r < GATE_RATIO
            and dco < GATE_RATIO
            and dens < GATE_DENSITY
            and atom < GATE_ATOM_INI
            and rep > GATE_REPAIR_POSITIVE
        )
        fails += not ok
        worst["ratio"] = max(worst["ratio"], 0.0 if np.isnan(r) else r)
        worst["dco"] = max(worst["dco"], 0.0 if np.isnan(dco) else dco)
        worst["dens"] = max(worst["dens"], dens)
        worst["atom"] = max(worst["atom"], atom)
        worst["repair"] = min(worst["repair"], np.inf if np.isnan(rep) else rep)
        line = (
            f"[{k:02d}] " + " ".join(f"{n}={v:+.2f}" for n, v in zip(names, th, strict=False)) + " | "
            f"ratio_err={r:.2e} dCO={dco:.2e} dens={dens:.2e} atom_ini={atom:.2e}"
            + (f" min_repair={rep:.4f}" if not np.isnan(rep) else "")
            + ("" if ok else "  <-- FAIL")
        )
        print(line, flush=True)

        if args.converge:
            _report_drift(chem, th)

    print("\n==== elemental audit summary ====")
    print(
        f"case={args.spec} draws={args.n} chemistry block {names} | worst ratio_err={worst['ratio']:.3e} "
        f"dCO={worst['dco']:.3e} density={worst['dens']:.3e} "
        f"atom_ini={worst['atom']:.3e} min_repair={worst['repair']:.4f}"
    )
    verdict = fails == 0
    summary = (
        f"{'PASS' if verdict else f'FAIL ({fails}/{args.n} draws)'} "
        f"(gates: ratio<{GATE_RATIO:g}, density<{GATE_DENSITY:g}, "
        f"atom_ini<{GATE_ATOM_INI:g}, repair>{GATE_REPAIR_POSITIVE:g})"
    )
    print(f"VERDICT: {summary}")
    _artifact.emit(
        name="elemental_audit",
        title="Elemental inventory of the abundance map over prior draws",
        measurements=[{"draws": args.n, "seed": args.seed, "chem_block": names, "failed": fails, "worst": worst}],
        status="PASS" if verdict else "FAIL",
        summary=summary,
        resolved_config=profile,
        out_dir=results,
    )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
