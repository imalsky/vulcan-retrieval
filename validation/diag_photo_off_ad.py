"""Does forward-mode AD work on the recipe-CONVERGED photo-off W39b state?

Once the photo-off primal is made to converge (config F: central diffusion + dt_max=1e11 +
conver_ignore sulfur allotropes + mtol_conv=1e-15), there is a real fixed point to linearize.
This checks the actual claim behind the gate: warm jvp (the production Fisher path) and cold jvp
vs re-converged central FD, per Fisher parameter, per-responding-cell.

If warm/cold jvp match FD here, the gate's premise ("photo-off tangent is wrong") is false once
the primal converges -- the only real blocker was forward-model convergence.

Run:  (vulcan env, from vulcan-retrieval/)  python validation/diag_photo_off_ad.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
from retrieval_framework.forward import vulcan_chem
import jax
import jax.numpy as jnp

KNOBS = [(0, 0.02, "lnZ"), (1, 0.02, "C/O"), (2, 0.5, "lnKzz"), (3, 1.0, "dT")]
TRACERS = ["SO2", "CO2", "CO", "H2O", "CH4"]
SULFUR_IGNORE = ["S", "S2", "S3", "S4", "S8", "S2O", "HS2", "CS2"]


def _cell_metric(deriv, fd):
    a, b = np.asarray(deriv).ravel(), np.asarray(fd).ravel()
    mag = np.abs(b); hi = mag >= np.quantile(mag, 0.99)
    if hi.sum() < 2:
        return np.nan, np.nan, np.nan
    rel = np.abs(a[hi] - b[hi]) / np.maximum(np.abs(b[hi]), 1e-300)
    return float(np.corrcoef(a[hi], b[hi])[0, 1]), float(np.median(rel)), float(np.max(rel))


def main() -> int:
    import vulcan_jax
    base_ci = list(getattr(vulcan_jax.load_config(config.W39B_CFG_NAME), "conver_ignore", []))
    ci_ext = base_ci + [s for s in SULFUR_IGNORE if s not in base_ci]
    prof = dict(config.WIDE)
    prof.update({"use_photo": False, "dt_max": 1e11,
                 "cfg_overrides": {"conver_ignore": ci_ext, "mtol_conv": 1e-15,
                                   "use_vm_mol": False, "use_hybrid_vm_mol": False}})
    t0 = time.time()
    chem = vulcan_chem.build_chem_model(prof)
    sidx = chem.sidx
    theta0 = jnp.asarray(config.THETA0, dtype=jnp.float64)
    lnZ0, co0 = float(config.THETA0[0]), float(config.THETA0[1])

    y_sol, ac = chem.converged_y(theta0, return_diag=True)
    print(f"[AD] recipe photo-off baseline: accept_count={int(ac)}/{chem.count_max} "
          f"({time.time()-t0:.0f}s) -- must be << count_max for a valid fixed point", flush=True)

    def f_warm(th):
        y = chem.converged_y(th, warm_y=y_sol, lnZ_ref=lnZ0, c_o_ref=co0)
        return y / jnp.sum(y, axis=1, keepdims=True)

    def f_cold(th):
        return chem.converged_ymix(th)

    for k, eps, name in KNOBS:
        tk = time.time()
        e = jnp.zeros(4, dtype=jnp.float64).at[k].set(1.0)
        _, warm = jax.jvp(f_warm, (theta0,), (e,))
        _, cold = jax.jvp(f_cold, (theta0,), (e,))
        fp = np.asarray(f_cold(theta0.at[k].add(eps)))
        fm = np.asarray(f_cold(theta0.at[k].add(-eps)))
        fd = (fp - fm) / (2 * eps)
        cw, mw, xw = _cell_metric(warm, fd)
        cc, mc, xc = _cell_metric(cold, fd)
        print(f"[AD] d(VMR)/d{name} ({time.time()-tk:.0f}s): "
              f"WARM vs FD corr={cw:.4f} med_rel={mw:.2e} | COLD vs FD corr={cc:.4f} "
              f"med_rel={mc:.2e}", flush=True)
        wj = np.asarray(warm)
        for s in TRACERS:
            if s in sidx:
                j = sidx[s]; kk = int(np.nanargmax(np.abs(fd[:, j])))
                print(f"[AD]     {s:4s} FD={fd[kk,j]:+.3e} warm={wj[kk,j]:+.3e} @L{kk}", flush=True)
    print(f"[AD] DONE {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
