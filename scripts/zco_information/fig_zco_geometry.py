"""Figure 3 -- the Z-C/O degeneracy geometry, and where the Gaussian ellipse can be trusted.

(a) The (log Z, log C/O) marginal 68% ellipse with its two principal axes drawn: the
    eigendecomposition AUTOMATICALLY surfaces which parameter COMBINATION the data constrain
    well and which is degenerate (the flat axis). No guessing -- the Hessian tells you.
(b) The Gaussian-validity walk (Vallisneri 2008): step the REAL forward model along the
    flattest (most degenerate) eigendirection and compare the true delta-chi^2 (profiled over
    the linear nuisances) to the Fisher parabola (delta-chi^2 = s^2). Where the points leave
    the parabola, the quadratic/Gaussian error bar is no longer trustworthy -- a check the
    exact second derivatives let us make instead of assume.

Reads data/zco_jacobians.npz (fig a) + data/zco_walk.npz (fig b, build_zco_walk.py). PNG only.
"""
from __future__ import annotations

import sys

import numpy as np

from retrieval_framework.forward import config

# shared _common (apply_style) lives in the sibling jax_paper repo
if not (config.JP / "scripts" / "_common.py").is_file():
    raise RuntimeError(f"jax_paper sibling not found at {config.JP} -- manuscript figure scripts require it (set VULCAN_PROJECT_ROOT)")
sys.path.insert(0, str(config.JP / "scripts"))
import zco_lib as Z  # sibling in zco_information/ (script dir is on sys.path)

LN10 = np.log(10.0)
WALK = Z.DATA / "zco_walk.npz"


def main(out=Z.FIGS / "zco_geometry.png", combo=Z.DEFAULT_COMBO, tier="P"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _common import apply_style
    apply_style()

    wl_model, tiers, meta = Z.load_jacobians()
    obs = Z.load_combined(combo)
    des = Z.build_design(tiers[tier], wl_model, obs)
    F, _ = Z.fisher(des["J"], des["sigma"])
    evals, evecs, C2 = Z.eigendecompose_marginal(F, des["interest"])
    # eigenvectors of the MARGINAL Fisher: column 0 = smallest eigenvalue = flattest (degenerate)
    sig_axis = 1.0 / np.sqrt(evals)                      # 1-sigma length along each eigenvector (ln)
    v_flat, v_stiff = evecs[:, 0], evecs[:, 1]
    if v_flat[0] < 0:
        v_flat = -v_flat
    if v_stiff[0] < 0:
        v_stiff = -v_stiff

    fig = plt.figure(figsize=(12.4, 5.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.26)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # (a) ellipse + principal axes (in dex)
    ex, ey = Z.ellipse_xy(C2)
    axA.plot(ex / LN10, ey / LN10, color="#333333", lw=2.0, zorder=3)
    axA.fill(ex / LN10, ey / LN10, color="#cc3311", alpha=0.10, zorder=1)
    # eigenvector arrows (both directions), no per-arrow text -- numbers go in the corner box
    for v, s, col in [(v_stiff, sig_axis[1], "#2b6cb0"), (v_flat, sig_axis[0], "#cc3311")]:
        dx, dy = (v[0] * s) / LN10, (v[1] * s) / LN10
        for sgn in (+1, -1):
            axA.annotate("", xy=(sgn * dx, sgn * dy), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2), zorder=4)
    axA.axhline(0, color="0.9", lw=0.6); axA.axvline(0, color="0.9", lw=0.6)
    axA.plot(0, 0, "k+", ms=8)
    axA.set_xlabel(r"$\Delta\,\log_{10} Z$  [dex]")
    axA.set_ylabel(r"$\Delta\,\log_{10}(\mathrm{C/O})$  [dex]")
    ratio = np.sqrt(evals[1] / evals[0])
    axA.set_title("(a) which Z–C/O combination is least constrained", fontsize=12.5)
    box = (f"$\\sigma_{{\\log Z}}$ = {sig_axis[0]/LN10:.3f} dex  (red, least constrained)\n"
           f"$\\sigma_{{\\log C/O}}$ = {sig_axis[1]/LN10:.3f} dex  (blue)\n"
           f"$\\rho$ = {C2[0,1]/np.sqrt(C2[0,0]*C2[1,1]):+.2f},  axis ratio {ratio:.1f}\n"
           f"$\\Rightarrow$ C/O measured {ratio:.1f}$\\times$ better; nearly independent")
    axA.text(0.03, 0.03, box, transform=axA.transAxes, ha="left", va="bottom", fontsize=8.4,
             color="0.2", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", alpha=0.9))
    axA.set_aspect("equal", adjustable="datalim")
    axA.margins(0.16)

    # (b) validity walk
    if WALK.exists():
        w = np.load(WALK, allow_pickle=True)
        s = np.asarray(w["steps"], float)
        c2t = np.asarray(w["chi2_true"], float)
        c2q = np.asarray(w["chi2_quad"], float)
        ss = np.linspace(s.min(), s.max(), 200)
        axB.plot(ss, ss ** 2, color="0.55", lw=1.8, ls="--",
                 label=r"Fisher / Gaussian  ($\Delta\chi^2 = s^2$)")
        axB.plot(s, c2t, "o-", color="#cc3311", ms=5.5, lw=1.4,
                 label="true forward model (profiled)")
        for lv, lab in [(1, "1$\\sigma$"), (4, "2$\\sigma$"), (9, "3$\\sigma$")]:
            axB.axhline(lv, color="0.85", lw=0.6, ls=":")
            axB.text(s.min(), lv, lab, fontsize=7.5, color="0.5", va="bottom", ha="left")
        axB.set_xlabel(r"steps along the least-constrained (metallicity) axis  [$\sigma$]")
        axB.set_ylabel(r"$\Delta\chi^2$ from the fiducial")
        axB.set_title("(b) is the ellipse trustworthy? (validity walk)", fontsize=12.5)
        axB.legend(fontsize=8.6, loc="upper left")
        # the skew is the science: SO2 ~ Z^3.3 makes high Z easier to rule out than low Z
        axB.annotate("high $Z$: SO$_2$ rises steeply with $Z$\n$\\to$ better constrained than Gaussian",
                     xy=(2.0, 5.44), xytext=(0.15, 0.62), textcoords="axes fraction",
                     fontsize=8.0, color="#8a1a0d", ha="left",
                     arrowprops=dict(arrowstyle="->", color="#8a1a0d", lw=0.9))
        axB.annotate("low $Z$: SO$_2$ fades\n$\\to$ looser than Gaussian",
                     xy=(-2.0, 3.06), xytext=(0.04, 0.30), textcoords="axes fraction",
                     fontsize=8.0, color="#234e7d", ha="left",
                     arrowprops=dict(arrowstyle="->", color="#234e7d", lw=0.9))
    else:
        axB.text(0.5, 0.5, "run build_zco_walk.py\nto generate the validity walk",
                 transform=axB.transAxes, ha="center", va="center", fontsize=11, color="0.5")
        axB.set_xticks([]); axB.set_yticks([])

    fig.suptitle("How well the spectrum separates Z and C/O, and where the Gaussian error "
                 "bar holds", fontsize=12.8, y=1.0)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.9, bottom=0.17)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
