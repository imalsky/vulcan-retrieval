"""A1 -- Does the production W39b column reach a steady state with photolysis OFF?

This is the decisive question behind the jwst-tool "Fisher forecast requires photo ON"
gate. The gate says the warm-started forward-mode jvp is "under-relaxed/unstable" photo-off.
But if the *primal* never converges photo-off, there is no fixed point y* to linearize and
no AD scheme can help -- the gate would be an honest forward-model non-convergence limit,
just mislabeled as a tangent-stability problem.

We build the SAME production column as the Fisher forecast (config.WIDE, nz=150, yconv 1e-3),
once photo-ON (known-good control, ~1300 steps) and once photo-OFF, and run the COLD solver
(run_diag -> integ._runner) at theta0 and a small ring. We judge convergence from
accept_count vs count_max and longdy vs the convergence gate. NOTE: is_done /
termination_reason are batched-runner-only (seeded (False,0) and never set on the
single-profile path -- outer_loop.py:211-220), so we do NOT read them here.

Run:  (vulcan env, from vulcan-retrieval/)  python validation/diag_photo_off_convergence.py
Cost: 2 warm-up solves (build) + ~6 cold solves. Photo-off solves may march to count_max
(minutes each). A normal local run, not a "big run."
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
# import order is load-bearing: vulcan_chem before exojax/jax (sets env + jax x64)
from retrieval_framework.forward import vulcan_chem


def _cfg_gates(chem):
    """Pull the convergence-gate knobs off the built model's resolved cfg."""
    c = chem._integ._cfg
    return {
        "yconv_cri": float(getattr(c, "yconv_cri", np.nan)),
        "yconv_min": float(getattr(c, "yconv_min", np.nan)),
        "slope_cri": float(getattr(c, "slope_cri", np.nan)),
        "runtime": float(getattr(c, "runtime", np.nan)),
        "count_min": int(getattr(c, "count_min", -1)),
        "count_max": int(chem.count_max),
    }


def _report(label, chem, theta, gates):
    """One cold solve; print the convergence diagnostics and a verdict."""
    t0 = time.time()
    final, _init = chem.run_diag(np.asarray(theta, dtype=np.float64))
    ac = int(final.accept_count)
    longdy = float(final.longdy)
    longdydt = float(final.longdydt)
    seen_min = float(final.longdy_seen_min)
    t_elapsed = float(final.t)
    dt = float(final.dt)
    cmax = gates["count_max"]
    yc = gates["yconv_cri"]
    ymin = gates["yconv_min"]
    slope = gates["slope_cri"]
    runtime = gates["runtime"]

    hit_cap = ac >= cmax
    hit_runtime = (not hit_cap) and np.isfinite(runtime) and t_elapsed >= 0.999 * runtime
    # The runner's actual convergence gate is an OR: longdy < yconv_cri (tight) OR
    # (longdy < yconv_min AND longdydt < slope_cri) (slope branch). The photo-ON W39b
    # column converges via the slope branch at longdy~0.06, NOT below yconv_cri=1e-3.
    conv_gate = (longdy < yc) or (longdy < ymin and longdydt < slope)
    converged = (not hit_cap) and conv_gate
    if converged:
        verdict = "CONVERGED"
    elif hit_cap:
        verdict = "NOT CONVERGED (hit count_max)"
    elif hit_runtime:
        verdict = "NOT CONVERGED (runtime cap; longdy stuck)"
    else:
        verdict = "UNCLEAR (stopped early, longdy>yconv_cri)"

    print(f"[A1] {label:28s} theta={list(np.round(np.asarray(theta),3))}", flush=True)
    print(f"[A1]     accept_count={ac:6d} / count_max={cmax:<6d}  "
          f"longdy={longdy:.3e} (gate yconv_cri={yc:.1e})  longdy_seen_min={seen_min:.3e}",
          flush=True)
    print(f"[A1]     longdydt={longdydt:.3e}  t={t_elapsed:.3e}s (runtime={runtime:.2e})  "
          f"dt={dt:.3e}  solve={time.time()-t0:.0f}s  -> {verdict}", flush=True)
    return verdict


def main() -> int:
    t0 = time.time()
    wide_on = dict(config.WIDE)
    wide_on["use_photo"] = True
    wide_off = dict(config.WIDE)
    wide_off["use_photo"] = False

    theta0 = config.THETA0
    # Small ring to see whether any non-convergence is baseline-specific or regime-wide.
    # lnZ +-0.3 (theta[0]); dT +-100 K (theta[3]) -- both safely inside the T-window.
    ring = [
        ("theta0", theta0),
        ("lnZ=+0.3", [0.3, 0.0, 0.0, 0.0]),
        ("lnZ=-0.3", [-0.3, 0.0, 0.0, 0.0]),
        ("dT=+100K", [0.0, 0.0, 0.0, 100.0]),
        ("dT=-100K", [0.0, 0.0, 0.0, -100.0]),
    ]

    print("=" * 78, flush=True)
    print("[A1] Building photo-ON control model (config.WIDE, nz=150) ...", flush=True)
    chem_on = vulcan_chem.build_chem_model(wide_on)
    gates_on = _cfg_gates(chem_on)
    print(f"[A1] photo-ON gates: {gates_on}", flush=True)
    _report("photo-ON  theta0 (control)", chem_on, theta0, gates_on)

    print("=" * 78, flush=True)
    print("[A1] Building photo-OFF model (config.WIDE, nz=150, use_photo=False) ...",
          flush=True)
    chem_off = vulcan_chem.build_chem_model(wide_off)
    gates_off = _cfg_gates(chem_off)
    print(f"[A1] photo-OFF gates: {gates_off}", flush=True)

    verdicts = {}
    for name, theta in ring:
        verdicts[name] = _report(f"photo-OFF {name}", chem_off, theta, gates_off)

    print("=" * 78, flush=True)
    print("[A1] SUMMARY (photo-OFF):", flush=True)
    for name, _ in ring:
        print(f"[A1]     {name:12s} -> {verdicts[name]}", flush=True)
    n_conv = sum(v == "CONVERGED" for v in verdicts.values())
    print(f"[A1] photo-OFF converged at {n_conv}/{len(ring)} points; "
          f"total {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
