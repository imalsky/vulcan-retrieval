#!/usr/bin/env python3
"""Single-panel forward-mode spectral-sensitivity figure: the WASP-39b
transmission spectrum colored by d(transit depth)/d(ln Z).

This is the metallicity panel of the four-parameter VULCAN-JAX -> ExoJax demo
(this repo), kept on its own because it is the most informative of
the four knobs: metallicity is the only one whose spectral sensitivity changes
sign, and over the NIRSpec PRISM band it isolates the SO2 (~4 um) feature while
the rest of the spectrum responds the other way.

The same forward-mode tangent that gives the per-species metallicity power laws
(``fig_metallicity_sens.py``) is pushed one step further, through the converged
chemistry and an ExoJax ``ArtTransPure`` transmission model, all the way to the
transit depth at every wavelength. We read the cached jvp output the demo wrote
(no recompute); the heavy chemistry+RT+line-list work lives in the demo project.

Range/resolution match the WASP-39b NIRSpec PRISM spectrum of Tsai et al. (2023)
(0.5-5.5 um, R~100) as closely as the demo allows: the H2-H2 CIA opacity stops
at 1 um, so the window is 1.0-5.5 um (the <1 um region is featureless for these
molecules anyway), binned to R=100.

Input : config.OUTPUTS/wide_sensitivity.npz  (written by examples/run_figs.py)
Output: jax_paper/figures/exojax_sensitivity.png
"""
from __future__ import annotations

import sys

import numpy as np

from retrieval_framework.forward import config

# shared _common lives in jax_paper/scripts (sibling repo)
if not (config.JP / "scripts" / "_common.py").is_file():
    raise RuntimeError(f"jax_paper sibling not found at {config.JP} -- manuscript figure scripts require it (set VULCAN_PROJECT_ROOT)")
sys.path.insert(0, str(config.JP / "scripts"))
from _common import FIGS, apply_style, require_input   # FIGS = jax_paper/figures (manuscript figures stay)

DATA = config.OUTPUTS                                  # run-bundle npz caches
NPZ = DATA / "wide_sensitivity.npz"
OUT = FIGS / "exojax_sensitivity.png"
PARAM_COL = 0                                   # theta order is [lnZ, C/O, lnKzz, dT]
WL_LO, WL_HI, R = 1.0, 5.5, 100                 # NIRSpec PRISM range (1 um CIA floor) + resolution


def bin_constant_R(wl_um, depth_ppm, sens_ppm, R, wl_lo, wl_hi):
    """Bin (depth, sensitivity) onto a constant-R log-wavelength grid over [lo, hi]."""
    order = np.argsort(wl_um)
    wl, d, s = wl_um[order], depth_ppm[order], sens_ppm[order]
    sel = (wl >= wl_lo) & (wl <= wl_hi)
    wl, d, s = wl[sel], d[sel], s[sel]
    nb = int(round(np.log(wl_hi / wl_lo) * R))          # d(ln lambda) = 1/R per bin
    edges = wl_lo * np.exp(np.arange(nb + 1) / R)
    idx = np.clip(np.digitize(wl, edges) - 1, 0, nb - 1)
    wb = np.full(nb, np.nan); db = np.full(nb, np.nan); sb = np.full(nb, np.nan)
    for b in range(nb):
        m = idx == b
        if m.any():
            wb[b] = wl[m].mean(); db[b] = d[m].mean(); sb[b] = s[m].mean()
    keep = np.isfinite(wb)
    return wb[keep], db[keep], sb[keep]


def main() -> int:
    require_input(
        NPZ,
        "python examples/run_figs.py  "
        "(writes wide_sensitivity.npz to config.OUTPUTS)",
    )
    apply_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    d = np.load(NPZ, allow_pickle=True)
    Jkey = "J_trans" if "J_trans" in d.files else "J"     # wide vs full demo schema
    wl_um = np.asarray(d["wl_um"])
    depth_ppm = np.asarray(d["depth"]) * 1e6
    sens_ppm = np.asarray(d[Jkey])[:, PARAM_COL] * 1e6    # d(depth)/d(ln Z), ppm per e-fold in Z

    wl, dppm, sppm = bin_constant_R(wl_um, depth_ppm, sens_ppm, R, WL_LO, WL_HI)
    pts = np.array([wl, dppm]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    segc = 0.5 * (sppm[:-1] + sppm[1:])                   # per-segment color

    # berlin runs dark-center -> bright-ends, so |sensitivity| maps to brightness:
    # dark = insensitive (ignore), bright = informative (look here). The P88 bound
    # saturates the informative bands to the bright ends and fades the flat
    # continuum into the dark center.
    a = float(np.nanpercentile(np.abs(segc), 88)) or 1.0

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    lc = LineCollection(segs, cmap="berlin", norm=Normalize(vmin=-a, vmax=+a), lw=2.0)
    lc.set_array(segc)
    ax.add_collection(lc)
    span = dppm.max() - dppm.min()
    pad = 0.04 * span
    ax.set_xlim(WL_LO, WL_HI)
    ax.set_ylim(dppm.min() - pad, dppm.max() + pad)
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(r"Transit depth $(R_p/R_\star)^2$  [ppm]")

    cb = fig.colorbar(lc, ax=ax, pad=0.015, extend="both")
    cb.set_label(r"$\partial d / \partial \ln Z$  [ppm]", fontsize=13)
    cb.ax.tick_params(labelsize=11)
    cb.ax.yaxis.get_offset_text().set_fontsize(10)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"[fig] wrote {OUT}  (R={R}, {WL_LO}-{WL_HI} um, panel col {PARAM_COL} = d/dlnZ)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
