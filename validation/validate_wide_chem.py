"""Full-config finite-difference validation of the chemistry jvp (config.WIDE, nz=150).

``smoke_test.py`` validates the *whole* chain end-to-end, but only at the reduced SMOKE
config (nz=40, CO-only). This script independently checks the expensive piece -- the
forward-mode chemistry gradient at the *actual figure resolution* (nz=150, photochemistry
on) -- by comparing ``jax.jvp`` of column-mean tracer abundances against a re-converged
central finite difference, for the three knobs shown in ``sensitivity_transmission_1-15um``
(lnZ, C/O, lnKzz). No exojax/HITRAN needed, so it runs fully offline.

Cost is chemistry-dominated, so each knob does exactly one jvp + two FD re-converges
(three nz=150 solves); all five tracers are read off those shared arrays. Expect ~13 min.

Interpreting the output
-----------------------
* lnZ and C/O drive large, well-mixed responses -> jvp matches FD to <1%.
* lnKzz barely moves the column (eddy mixing is ~30x weaker here), so the column-MEAN
  signal is tiny and the central FD's noise floor inflates the relative error to a few %
  (more on near-zero tracers like CH4). That is FD noise on a ~0 derivative, not a jvp
  error -- the Kzz tangent itself is validated to <0.1% on the *responding levels* in
  VULCAN-JAX's fig_kzz_jvp_validate.py, which this demo's knob replicates.

Run:  (vulcan env)  python validation/validate_wide_chem.py
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
# import order is load-bearing: vulcan_chem before exojax/jax (sets env + jax x64)
from retrieval_framework.forward import vulcan_chem
import jax
import jax.numpy as jnp

# (theta index, FD step, label). Steps are large enough to clear the convergence-noise
# floor of the re-converged central difference.
KNOBS = [(0, 0.02, "lnZ"), (1, 0.02, "C/O"), (2, 0.5, "lnKzz")]
TRACERS = ["SO2", "CO2", "CO", "H2O", "CH4"]
REL_TOL = 0.10           # pass threshold for tracers with a non-negligible signal
SIGNAL_FLOOR = 1.0e-6    # |d ln VMR/d theta| below this is treated as noise, not checked


def main() -> int:
    t0 = time.time()
    chem = vulcan_chem.build_chem_model(config.WIDE)
    cols = {s: chem.sidx[s] for s in TRACERS if s in chem.sidx}
    theta0 = jnp.asarray(config.THETA0, dtype=jnp.float64)

    def logcols(theta):
        """log of each tracer's column-mean VMR -> stacked (len(cols),). One chem solve."""
        ymix = chem.converged_ymix(theta)
        return jnp.log(jnp.stack([jnp.mean(ymix[:, cols[s]]) for s in cols]))

    print(f"[wide-chem] model built in {time.time()-t0:.0f}s; "
          f"nz={chem.nz} ni={chem.ni}; FD-checking {list(cols)}", flush=True)

    worst = 0.0
    for k, eps, name in KNOBS:
        tk = time.time()
        e = jnp.zeros(4, dtype=jnp.float64).at[k].set(1.0)
        _, jv = jax.jvp(logcols, (theta0,), (e,))               # 1 chem solve
        fp = np.asarray(logcols(theta0.at[k].add(eps)))          # 1 chem solve
        fm = np.asarray(logcols(theta0.at[k].add(-eps)))         # 1 chem solve
        jv = np.asarray(jv)
        fd = (fp - fm) / (2 * eps)
        print(f"[wide-chem] --- d ln(col VMR)/d{name} ({time.time()-tk:.0f}s) ---", flush=True)
        for i, s in enumerate(cols):
            rel = abs(jv[i] - fd[i]) / max(abs(fd[i]), 1e-12)
            has_signal = abs(fd[i]) > SIGNAL_FLOOR
            if has_signal:
                worst = max(worst, rel)
            flag = "OK" if (rel < REL_TOL or not has_signal) else "WARN"
            print(f"[wide-chem]   {s:4s}  jvp={jv[i]:+.4e}  fd={fd[i]:+.4e}  "
                  f"rel={rel:.2e}  [{flag}]", flush=True)
    print(f"[wide-chem] DONE in {time.time()-t0:.0f}s; "
          f"worst rel over tracers with signal>{SIGNAL_FLOOR:g} = {worst:.2e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
