"""Which species/layers drive the photo-OFF non-convergence?

The final JaxIntegState carries `where_varies_most` (nz, ni) -- the per-cell convergence
ratio the runner maximizes into `longdy`. Its top cells name the exact species + layer that
keep the photo-off W39b column from settling. If it is a handful of specific radicals, an
extended `conver_ignore` (the runner's own mechanism for slow-but-benign trace radicals) or a
targeted damping might let it converge; if it is a bulk carrier, it is a real transport-limited
non-steady state.

Reports the top offending cells (species, layer, pressure, value) cold at theta0, plus the
elemental atom loss per element (the ~0.9 drain seen in the long __call__ run).

Run:  (vulcan env, from vulcan-retrieval/)  python validation/diag_photo_off_culprit.py [on|off]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
from retrieval_framework.forward import vulcan_chem


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "off"
    prof = dict(config.WIDE)
    prof["use_photo"] = (which == "on")
    prof["dt_max"] = 1e11   # capped, so the blow-up doesn't dominate where_varies_most
    t0 = time.time()
    chem = vulcan_chem.build_chem_model(prof)
    inv = {v: k for k, v in chem.sidx.items()}
    p_bar = np.asarray(chem.p_bar)

    final, _ = chem.run_diag(np.asarray(config.THETA0, dtype=np.float64))
    wvm = np.abs(np.asarray(final.where_varies_most))     # (nz, ni)
    print(f"[culprit] photo-{which.upper()}  ac={int(final.accept_count)}  "
          f"longdy={float(final.longdy):.3e}  seen_min={float(final.longdy_seen_min):.3e}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    flat = np.argsort(wvm.ravel())[::-1][:15]
    print("[culprit] top cells driving longdy (species @ layer, pressure, ratio):", flush=True)
    for f in flat:
        z, i = np.unravel_index(f, wvm.shape)
        print(f"[culprit]   {inv.get(int(i), '?'):8s} @ layer {int(z):3d}  "
              f"P={p_bar[z]:.2e} bar   ratio={wvm[z, i]:.3e}", flush=True)

    # per-species worst ratio (which species is worst anywhere in the column)
    sp_worst = wvm.max(axis=0)
    order = np.argsort(sp_worst)[::-1][:12]
    print("[culprit] worst per-species (max over layers):", flush=True)
    for i in order:
        z = int(np.argmax(wvm[:, i]))
        print(f"[culprit]   {inv.get(int(i), '?'):8s}  max_ratio={sp_worst[i]:.3e} "
              f"@ layer {z} (P={p_bar[z]:.2e} bar)", flush=True)

    aloss = np.asarray(final.atom_loss)
    print(f"[culprit] atom_loss per element (runner order): "
          f"{np.array2string(aloss, precision=3)}", flush=True)
    # Clip-driven mass sink check: nega_y/small_y are cumulative |y| of cells the
    # post-step clip zeroed (a closed column has no physical escape, so any drain is
    # numerical). Compare to the total column density.
    Mtot = float(np.sum(np.asarray(final.pv.n_0)))
    print(f"[culprit] clipped mass: nega_y(sum|y| of clipped-negative)={float(final.nega_y):.3e}  "
          f"small_y={float(final.small_y):.3e}  total_col_density~{Mtot:.3e}  "
          f"nega_count={int(final.nega_count)} loss_count={int(final.loss_count)}", flush=True)
    T = np.asarray(chem.T_base)
    print(f"[culprit] T profile: top(P={p_bar[-1]:.1e}bar)={T[-1]:.0f}K  "
          f"bottom(P={p_bar[0]:.1e}bar)={T[0]:.0f}K  min={T.min():.0f}K max={T.max():.0f}K",
          flush=True)
    print(f"[culprit] DONE {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
