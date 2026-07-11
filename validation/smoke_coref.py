"""Fast check that the c_o_ref continuation preserves C/O along a metallicity march.

March Z up at a FIXED C/O row using warm-start continuation (c_o_ref = c_o after the first
cell). At each cell, recompute the converged column's C/O ratio -- it must stay ~constant
along the row (only metallicity changes). A bug that re-applies c_o every step would make C/O
drift multiplicatively. nz=40, ~a few min.
"""
from __future__ import annotations
import numpy as np
from retrieval_framework.forward import config
from retrieval_framework.forward import vulcan_chem
import jax.numpy as jnp

PROF = dict(config.SMOKE, co_mode="fixed_O", nz=40, yconv_cri=1.0e-4, yconv_min=1.0e-4,
            slope_cri=1.0e-7, count_min=200, count_max=8000, reanchor_atom_ini=True,
            fastchem_met_scale=1.0, molecules=["CO"])


def co_ratio(chem, y):
    compo = np.asarray(chem.compo_array, float)
    nC = compo[:, config.ATOM_COLS["C"]]; nO = compo[:, config.ATOM_COLS["O"]]
    y = np.asarray(y, float)
    return float((y * nC[None, :]).sum() / (y * nO[None, :]).sum())


def main():
    chem = vulcan_chem.build_chem_model(PROF)
    co_base = co_ratio(chem, chem.converged_y(jnp.array([0.0, 0.0, 0.0, 0.0])))
    target = 0.35
    c_o = float(np.log(target / co_base))
    print(f"baseline C/O={co_base:.4f}; target row C/O={target} -> c_o={c_o:+.3f}", flush=True)
    prev = None; lnZ_prev = 0.0
    for z in np.linspace(0.0, np.log(20.0), 5):
        y = chem.converged_y(jnp.array([z, c_o, 0.0, 0.0]), warm_y=prev,
                             lnZ_ref=lnZ_prev, c_o_ref=(0.0 if prev is None else c_o))
        co = co_ratio(chem, y)
        print(f"  Z={np.exp(z):5.1f}x  C/O={co:.4f}  (target {target}, err {abs(co-target)/target*100:.1f}%)", flush=True)
        prev = y; lnZ_prev = float(z)
    print("PASS if C/O stayed ~constant near the target (not drifting up each step).")


if __name__ == "__main__":
    main()
