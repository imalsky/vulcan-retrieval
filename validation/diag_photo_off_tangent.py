"""A2/A3.1 -- warm-jvp (the production Fisher tangent) vs re-converged FD, photo on/off.

The jwst-tool Fisher forecast differentiates the WARM-started continuation
``converged_y(theta, warm_y=y_sol)`` (forward.py:590-603), NOT the cold ``converged_ymix``
that validate_wide_chem checks. This script tests the actual production tangent:

  * warm jvp   = jax.jvp of  theta -> VMR(converged_y(theta, warm_y=y_sol, lnZ_ref, c_o_ref))
  * cold jvp   = jax.jvp of  converged_ymix   (full relaxation from the elemental guess)
  * FD truth   = central difference of the COLD re-converged VMR field (the ground truth)

per Fisher parameter {lnZ, C/O, lnKzz, dT}, photo-ON (known-good control) and photo-OFF.
Reporting is per-responding-cell (top-1% by |FD|), not column-mean, so localized failures
show. warm-vs-FD tests the gate's claim; cold-vs-FD localizes whether any photo-off failure
is warm-start truncation (cold agrees, warm doesn't) or something deeper (both disagree).

NOTE on the C1 subtlety: lnZ and C/O do NOT enter k(theta)/atm(theta); their sensitivity is
carried entirely by the initial-column / conserved inventory (dG/dtheta=0 for them). So expect
lnKzz/dT (driven by dG/dtheta) and lnZ/C-O (driven by the initial-state tangent) to behave
differently -- reported separately, never pooled.

Run:  (vulcan env, from vulcan-retrieval/)  python validation/diag_photo_off_tangent.py [on|off|both]
Cost: per mode 1 cold baseline + per param {warm jvp, cold jvp, 2 FD} ~= 17 solves.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
# import order is load-bearing: vulcan_chem before exojax/jax
from retrieval_framework.forward import vulcan_chem
import jax
import jax.numpy as jnp

# (theta index, FD half-step, label). Steps clear the re-converged central-difference
# noise floor (same lnZ/C-O/lnKzz steps as validate_wide_chem; dT in Kelvin).
KNOBS = [(0, 0.02, "lnZ"), (1, 0.02, "C/O"), (2, 0.5, "lnKzz"), (3, 1.0, "dT")]
TRACERS = ["SO2", "CO2", "CO", "H2O", "CH4"]


def _cell_metric(deriv, fd):
    """Top-1%-|FD|-cell agreement of a (nz,ni) derivative field vs FD truth."""
    a, b = np.asarray(deriv).ravel(), np.asarray(fd).ravel()
    mag = np.abs(b)
    hi = mag >= np.quantile(mag, 0.99)
    if hi.sum() < 2:
        return np.nan, np.nan, np.nan
    rel = np.abs(a[hi] - b[hi]) / np.maximum(np.abs(b[hi]), 1e-300)
    corr = np.corrcoef(a[hi], b[hi])[0, 1]
    return corr, float(np.median(rel)), float(np.max(rel))


def _run_mode(mode: str) -> dict:
    photo = (mode == "on")
    prof = dict(config.WIDE)
    prof["use_photo"] = photo
    print("=" * 78, flush=True)
    print(f"[A2] building photo-{'ON' if photo else 'OFF'} model ...", flush=True)
    chem = vulcan_chem.build_chem_model(prof)
    sidx = chem.sidx
    theta0 = jnp.asarray(config.THETA0, dtype=jnp.float64)
    lnZ0, co0 = float(config.THETA0[0]), float(config.THETA0[1])

    # baseline converged column -> warm-start seed for the production continuation
    tb = time.time()
    y_sol = chem.converged_y(theta0)
    print(f"[A2] baseline y_sol converged in {time.time()-tb:.0f}s", flush=True)

    def f_warm(th):
        y = chem.converged_y(th, warm_y=y_sol, lnZ_ref=lnZ0, c_o_ref=co0)
        return y / jnp.sum(y, axis=1, keepdims=True)

    def f_cold(th):
        return chem.converged_ymix(th)

    results = {}
    for k, eps, name in KNOBS:
        tk = time.time()
        e = jnp.zeros(4, dtype=jnp.float64).at[k].set(1.0)
        _, warm_jvp = jax.jvp(f_warm, (theta0,), (e,))     # production tangent (warm)
        _, cold_jvp = jax.jvp(f_cold, (theta0,), (e,))     # full-relaxation tangent (cold)
        fp = np.asarray(f_cold(theta0.at[k].add(eps)))     # FD truth (cold re-converge)
        fm = np.asarray(f_cold(theta0.at[k].add(-eps)))
        fd = (fp - fm) / (2 * eps)
        warm_jvp = np.asarray(warm_jvp); cold_jvp = np.asarray(cold_jvp)

        cw, mw, xw = _cell_metric(warm_jvp, fd)
        cc, mc, xc = _cell_metric(cold_jvp, fd)
        results[name] = dict(warm=(cw, mw, xw), cold=(cc, mc, xc))
        print(f"[A2] --- d(VMR)/d{name}  ({time.time()-tk:.0f}s) "
              f"[photo-{'ON' if photo else 'OFF'}] ---", flush=True)
        print(f"[A2]   WARM jvp vs FD: corr={cw:.4f} median_rel={mw:.2e} max_rel={xw:.2e}",
              flush=True)
        print(f"[A2]   COLD jvp vs FD: corr={cc:.4f} median_rel={mc:.2e} max_rel={xc:.2e}",
              flush=True)
        for s in TRACERS:
            if s in sidx:
                j = sidx[s]
                kk = int(np.nanargmax(np.abs(fd[:, j])))
                print(f"[A2]     {s:4s} peak FD={fd[kk,j]:+.3e} warm={warm_jvp[kk,j]:+.3e} "
                      f"cold={cold_jvp[kk,j]:+.3e} @layer {kk}", flush=True)
    return results


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    modes = {"on": ["on"], "off": ["off"], "both": ["on", "off"]}[which]
    t0 = time.time()
    allres = {m: _run_mode(m) for m in modes}
    print("=" * 78, flush=True)
    print("[A2] SUMMARY (WARM jvp median rel-err on top-1% cells):", flush=True)
    for m in modes:
        row = "  ".join(f"{n}={allres[m][n]['warm'][1]:.1e}" for _, _, n in KNOBS)
        print(f"[A2]   photo-{m.upper():3s}: {row}", flush=True)
    print(f"[A2] DONE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
