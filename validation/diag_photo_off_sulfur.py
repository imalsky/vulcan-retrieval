"""Does the known sulfur-allotrope convergence recipe let photo-OFF W39b reach a
PHYSICAL steady state -- or only pass the longdy gate while sulfur stays drained?

The culprit probe showed photo-off non-convergence is entirely sulfur: S3/S2/S2O/HS2/CS2
allotropes drive longdy and S drains ~84%. The project's condensation path already handles
the slow sulfur allotropes with conver_ignore(S/S2/S3/S4) + mtol_conv=1e-15. This tests whether
that recipe works for plain photo-off, and -- crucially -- whether the resulting state is
PHYSICAL (sulfur conserved, SO2/H2S intact) or a gate-passing but sulfur-depleted artifact.

Configs (photo-off, theta0, cold run_diag), each reporting longdy / longdy_seen_min /
per-element atom_loss / remaining top-culprit species:
  A. baseline (no recipe)                      -- reference
  B. + conver_ignore sulfur allotropes         -- gate no longer trips on S/S2/S3/S4/S8
  C. B + mtol_conv=1e-15                        -- close the sub-femto allotrope drift branch
  D. C + dt_max=1e11                            -- capped step (no blow-up)

Run:  (vulcan env, from vulcan-retrieval/)  python validation/diag_photo_off_sulfur.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
from retrieval_framework.forward import vulcan_chem

# The stiff sulfur allotropes + minor S carriers that the culprit probe flagged as gating.
# S/S2/S3/S4 are the documented condensation-path conver_ignore set; add S8 and the minor
# oscillators the probe surfaced. SO2/SO/H2S are NOT ignored -- those carry the science signal.
SULFUR_IGNORE = ["S", "S2", "S3", "S4", "S8", "S2O", "HS2", "CS2"]


def _report(tag, chem):
    inv = {v: k for k, v in chem.sidx.items()}
    t0 = time.time()
    final, _ = chem.run_diag(np.asarray(config.THETA0, dtype=np.float64))
    ac = int(final.accept_count); cmax = int(chem.count_max)
    longdy = float(final.longdy); seen = float(final.longdy_seen_min)
    aloss = np.asarray(final.atom_loss)
    wvm = np.abs(np.asarray(final.where_varies_most))
    sp_worst = wvm.max(axis=0)
    top = np.argsort(sp_worst)[::-1][:6]
    held = (ac < cmax) and (longdy < 0.1)
    cmin = int(getattr(chem._integ._cfg, "count_min", 120))
    hyb = "  <-- ~count_min+2000 (hybrid phase-flip cap; forward._check_converged accepts it!)" \
        if abs(ac - (cmin + 2000)) < 30 else ""
    print(f"[S] {tag}", flush=True)
    print(f"[S]   ac={ac}/{cmax}{hyb} longdy={longdy:.3e} seen_min={seen:.3e} "
          f"held={held} atom_loss={np.array2string(aloss, precision=3)} ({time.time()-t0:.0f}s)",
          flush=True)
    print("[S]   remaining top-culprit species: " +
          ", ".join(f"{inv.get(int(i),'?')}={sp_worst[i]:.2e}" for i in top), flush=True)
    return dict(ac=ac, longdy=longdy, seen=seen, aloss=aloss, held=held)


def _build(extra):
    prof = dict(config.WIDE); prof["use_photo"] = False; prof.update(extra)
    return vulcan_chem.build_chem_model(prof)


def main() -> int:
    import vulcan_jax
    base_ci = list(getattr(vulcan_jax.load_config(config.W39B_CFG_NAME), "conver_ignore", []))
    ci_ext = base_ci + [s for s in SULFUR_IGNORE if s not in base_ci]
    print(f"[S] base conver_ignore ({len(base_ci)}) -> extended (+{len(ci_ext)-len(base_ci)} "
          f"sulfur): {SULFUR_IGNORE}", flush=True)

    t0 = time.time()
    configs = [
        ("A baseline (no recipe)", {}),
        ("B +conver_ignore(S allotropes)", {"cfg_overrides": {"conver_ignore": ci_ext}}),
        ("C B+mtol_conv=1e-15", {"cfg_overrides": {"conver_ignore": ci_ext, "mtol_conv": 1e-15}}),
        ("D C+dt_max=1e11", {"dt_max": 1e11,
                             "cfg_overrides": {"conver_ignore": ci_ext, "mtol_conv": 1e-15}}),
        ("E hybrid vm_mol OFF (honest termination)",
         {"cfg_overrides": {"use_vm_mol": False, "use_hybrid_vm_mol": False}}),
        ("F recipe + hybrid OFF", {"dt_max": 1e11,
         "cfg_overrides": {"conver_ignore": ci_ext, "mtol_conv": 1e-15,
                           "use_vm_mol": False, "use_hybrid_vm_mol": False}}),
    ]
    for tag, extra in configs:
        print("=" * 78, flush=True)
        chem = _build(extra)
        _report(tag, chem)
    print("=" * 78, flush=True)
    print(f"[S] DONE {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
