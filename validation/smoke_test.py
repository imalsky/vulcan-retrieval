"""Fast, fully-offline pre-flight for the VULCAN-JAX -> ExoJax chain.

CO-only opacity (cached), coarse column (nz=40), loose convergence, photo on. Proves:
  1. the composed forward(theta) runs and returns a finite, physically-sane spectrum;
  2. all four parameter tangents (lnZ, C/O, lnKzz, dT) are finite through the whole
     chain (chemistry lax.while_loop -> log-P bridge -> ArtTransPure);
  3. the end-to-end forward-mode derivative matches a re-converged central finite
     difference at the most-sensitive wavelength for lnZ and dT (the two knobs that
     exercise, respectively, the y0-direction path and the on-graph rates_jax T path).

Run:  (vulcan env)  python validation/smoke_test.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

# import order is load-bearing: sensitivity (vulcan_chem) before any exojax/jax use
from retrieval_framework.forward import config
from retrieval_framework.forward.sensitivity import build_forward
import jax
import jax.numpy as jnp

# (param index, eps, name): eps large enough that the signal clears the convergence-
# noise floor of the central difference (a per-wavelength check is noisier than the
# column-integrated checks in the reference scripts).
FD_CHECKS = [(0, 0.02, "lnZ"), (3, 1.0, "dT")]
REL_TOL = 0.05  # dT matches FD to ~1e-5; lnZ to ~2% at this noisier per-wavelength check


def main() -> int:
    t0 = time.time()
    fb = build_forward(config.SMOKE)
    forward = fb.forward
    theta0 = jnp.asarray(config.THETA0, dtype=jnp.float64)

    print("[smoke] evaluating primal ...", flush=True)
    tp = time.time()
    primal = np.asarray(forward(theta0))
    base_depth = float(np.median(primal))
    print(f"[smoke] primal {time.time()-tp:.1f}s  n_nu={primal.size}  "
          f"finite={np.all(np.isfinite(primal))}  median depth={base_depth:.4e} "
          f"(expect ~0.02)", flush=True)
    if not np.all(np.isfinite(primal)):
        print("[smoke] FAIL: non-finite primal", flush=True)
        return 1

    print("[smoke] forward-mode jvp columns ...", flush=True)
    cols = []
    for k in range(4):
        e = jnp.zeros(4, dtype=jnp.float64).at[k].set(1.0)
        tj = time.time()
        _, dv = jax.jvp(forward, (theta0,), (e,))
        dv = np.asarray(dv)
        ok = np.all(np.isfinite(dv))
        print(f"[smoke]   d/d{config.THETA_LABELS[k]:6s} {time.time()-tj:.1f}s  "
              f"finite={ok}  max|.|={np.nanmax(np.abs(dv)):.3e}", flush=True)
        cols.append(dv)
        if not ok:
            print(f"[smoke] FAIL: non-finite tangent for {config.THETA_LABELS[k]}", flush=True)
            return 1
    J = np.stack(cols, axis=1)  # (n_nu, 4)

    print("[smoke] end-to-end finite-difference validation ...", flush=True)
    ok_all = True
    for k, eps, name in FD_CHECKS:
        i = int(np.nanargmax(np.abs(J[:, k])))
        ep = np.asarray(forward(theta0.at[k].add(eps)))[i]
        em = np.asarray(forward(theta0.at[k].add(-eps)))[i]
        fd = (ep - em) / (2.0 * eps)
        jv = J[i, k]
        rel = abs(jv - fd) / max(abs(fd), 1e-30)
        flag = "OK" if rel < REL_TOL else "WARN"
        if rel >= REL_TOL:
            ok_all = False
        print(f"[smoke]   {name:6s} @ lambda={fb.rt.wl_um[i]:.3f}um  "
              f"jvp={jv:+.4e}  fd={fd:+.4e}  rel={rel:.2e}  [{flag}]", flush=True)

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.savez(config.OUTPUTS / "smoke.npz",
             wl_um=fb.rt.wl_um, depth=primal, J=J, theta0=np.asarray(config.THETA0))
    print(f"[smoke] done in {time.time()-t0:.1f}s; "
          f"{'PASS' if ok_all else 'FD-WARN (chain finite, see rel above)'}", flush=True)
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
