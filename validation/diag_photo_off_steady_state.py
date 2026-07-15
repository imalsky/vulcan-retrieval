"""What is needed to reach a photo-OFF steady state on the production W39b column?

A1 established that photo-OFF W39b (config.WIDE, nz=150) does NOT hold a steady state: the
cold solve approaches a fixed point (longdy_seen_min ~ 0.002-0.05, below the 0.1 gate) but
longdy bounces back up (1 to 3.8e8) -- a marginally-stable / oscillatory fixed point, and the
runner stall-terminates at ~2100 steps with accept_count < count_max (so forward.py's
_check_converged waves the non-steady state through). dt ballooned to 1e16-1e18 s because the
WIDE profile leaves dt_max at 1e17 (production retrieval caps it at 1e11).

This script sweeps the convergence levers to determine what, if anything, makes it hold:

  * dt_max cap (1e11, 1e9) -- does bounding the step stop the longdy blow-ups / atom loss?
  * conv_stall_window huge -- disable the early stall so we see the uninterrupted long-run
    longdy behavior (settle vs oscillate).
  * WARM-START from the photo-ON converged column + a fixed-point residual (re-converge from
    the warm result and measure how far it moves) -- THE test for whether a photo-off steady
    state EXISTS near the physical atmosphere and can be held from a good initial guess.

Cold runs use run_diag (full JaxIntegState: longdy, longdy_seen_min, atom_loss). Warm runs use
converged_y(return_diag) for accept_count + the self-consistency residual.

Run:  (vulcan env, from vulcan-retrieval/)  python validation/diag_photo_off_steady_state.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
from retrieval_framework.forward import vulcan_chem


def _gates(chem):
    c = chem._integ._cfg
    return dict(yconv_cri=float(c.yconv_cri), yconv_min=float(getattr(c, "yconv_min", 0.1)),
                slope_cri=float(getattr(c, "slope_cri", 1e-4)),
                dt_max=float(getattr(c, "dt_max", np.nan)),
                conv_stall_window=int(getattr(c, "conv_stall_window", 200)),
                count_max=int(chem.count_max))


def _cold(chem, theta, gates):
    t0 = time.time()
    final, _ = chem.run_diag(np.asarray(theta, dtype=np.float64))
    ac = int(final.accept_count); cmax = gates["count_max"]
    longdy = float(final.longdy); seen = float(final.longdy_seen_min)
    aloss = float(np.max(np.abs(np.asarray(final.atom_loss))))
    csnm = int(final.count_since_new_min)
    held = (ac < cmax) and (longdy < gates["yconv_min"])
    print(f"[SS]   COLD  ac={ac:6d}/{cmax}  longdy={longdy:.3e}  seen_min={seen:.3e}  "
          f"csnm={csnm}  max|atomloss|={aloss:.2e}  t={float(final.t):.2e}s  "
          f"{time.time()-t0:.0f}s  held={held}", flush=True)
    return dict(ac=ac, longdy=longdy, seen=seen, aloss=aloss, held=held)


def _warm_holds(chem, theta, y_on, gates):
    """Warm-start photo-off from the photo-ON column, then re-converge from the result.
    A small self-consistency residual + accept_count < count_max = a held photo-off fixed point."""
    t0 = time.time()
    th = np.asarray(theta, dtype=np.float64)
    y1, ac1 = chem.converged_y(th, warm_y=y_on, return_diag=True)
    y2, ac2 = chem.converged_y(th, warm_y=y1, return_diag=True)
    v1 = np.asarray(y1); v1 = v1 / v1.sum(axis=1, keepdims=True)
    v2 = np.asarray(y2); v2 = v2 / v2.sum(axis=1, keepdims=True)
    # residual on cells with non-negligible VMR (ignore trace-species float noise)
    mask = v1 > 1e-8
    resid = float(np.max(np.abs(v2[mask] / v1[mask] - 1.0))) if mask.any() else np.nan
    cmax = gates["count_max"]
    print(f"[SS]   WARM(from photo-on)  ac1={int(ac1):6d} ac2={int(ac2):6d} /{cmax}  "
          f"self-consistency max|dVMR|={resid:.3e}  {time.time()-t0:.0f}s  "
          f"(small resid + ac<cmax = held fixed point)", flush=True)
    return dict(ac1=int(ac1), ac2=int(ac2), resid=resid)


def _build_off(extra):
    prof = dict(config.WIDE)
    prof["use_photo"] = False
    prof.update(extra)
    return vulcan_chem.build_chem_model(prof)


def main() -> int:
    t0 = time.time()
    theta0 = config.THETA0

    print("=" * 78, flush=True)
    print("[SS] building photo-ON model for the warm-start seed y_on ...", flush=True)
    chem_on = _build_off({"use_photo": True})  # extra overrides use_photo back to True
    y_on = np.asarray(chem_on.converged_y(np.asarray(theta0, dtype=np.float64)))
    print(f"[SS] photo-ON y_on obtained ({time.time()-t0:.0f}s)", flush=True)

    # (label, profile-extra). dt_max unset in WIDE -> default 1e17 (the ballooning baseline).
    configs = [
        ("dtmax=1e11  stall=200",  {"dt_max": 1e11}),
        ("dtmax=1e9   stall=200",  {"dt_max": 1e9}),
        ("dtmax=1e11  NOSTALL",    {"dt_max": 1e11,
                                    "cfg_overrides": {"conv_stall_window": 10**9}}),
    ]
    for label, extra in configs:
        print("=" * 78, flush=True)
        print(f"[SS] === photo-OFF  [{label}] ===", flush=True)
        chem = _build_off(extra)
        g = _gates(chem)
        print(f"[SS]   gates: dt_max={g['dt_max']:.1e} stall_window={g['conv_stall_window']} "
              f"count_max={g['count_max']} yconv_min={g['yconv_min']}", flush=True)
        _cold(chem, theta0, g)
        _warm_holds(chem, theta0, y_on, g)

    print("=" * 78, flush=True)
    print(f"[SS] DONE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
