"""Figure 2 -- the (Z, C/O) information that DISEQUILIBRIUM chemistry creates.

Same radiative transfer, same real Carter & May (2024) errors; the only thing that changes
is which chemistry physics is switched on when VULCAN-JAX converges the column:

    E  no-transport (~equilibrium: kinetic fixed point, ~3% element drift caveat)
    Q  + transport (quenching)  (no photochemistry)
    P  + photochemistry         (the fiducial WASP-39b model)

(a) The joint 68% (log Z, log C/O) marginal error ellipse for each tier -- everything else
    (lnKzz, T_int, lnR0, inter-instrument offsets) marginalized. Photochemistry tightens and
    de-correlates the constraint.
(b) WHERE the new metallicity information comes from: the per-wavelength unique ln Z
    information for equilibrium vs photochemistry. The SO2 band at ~4.05 um -- absent in
    equilibrium -- is the photochemical metallicity anchor.

Reads data/zco_jacobians.npz. PNG only.
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
TIER_STYLE = {
    "E": dict(color="#9aa0a6", label="no-transport (~equilibrium)"),
    "Q": dict(color="#e08214", label="+ transport (quench)"),
    "P": dict(color="#cc3311", label="+ photochemistry"),
}


def main(out=Z.FIGS / "zco_disequilibrium.png", combo=Z.DEFAULT_COMBO):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _common import apply_style
    apply_style()

    wl_model, tiers, meta = Z.load_jacobians()
    obs = Z.load_combined(combo)
    order = [t for t in ["E", "Q", "P"] if t in tiers]

    fig = plt.figure(figsize=(12.6, 5.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.22], wspace=0.28)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # (a) ellipses per tier. The quench ellipse is ~5x larger than equilibrium/photochem,
    # so an inset zooms on the two small ones.
    des_by, ell = {}, {}
    for t in order:
        des = Z.build_design(tiers[t], wl_model, obs)
        des_by[t] = des
        F, _ = Z.fisher(des["J"], des["sigma"])
        C2, _ = Z.marginal_cov(F, des["interest"])
        ex, ey = Z.ellipse_xy(C2)
        sZ, sCO = np.sqrt(C2[0, 0]) / LN10, np.sqrt(C2[1, 1]) / LN10
        rho = C2[0, 1] / np.sqrt(C2[0, 0] * C2[1, 1])
        ell[t] = (ex / LN10, ey / LN10, sZ, sCO, rho)
        st = TIER_STYLE[t]
        axA.plot(ex / LN10, ey / LN10, color=st["color"], lw=2.2,
                 label=f"{st['label']}:  $\\sigma_{{\\log Z}}$={sZ:.3f}, "
                       f"$\\sigma_{{\\log C/O}}$={sCO:.3f}, $\\rho$={rho:+.2f}")
    axA.axhline(0, color="0.9", lw=0.6); axA.axvline(0, color="0.9", lw=0.6)
    axA.plot(0, 0, "k+", ms=8)
    axA.set_xlabel(r"$\Delta\,\log_{10} Z$  [dex]")
    axA.set_ylabel(r"$\Delta\,\log_{10}(\mathrm{C/O})$  [dex]")
    axA.set_title("(a) joint 68% error on Z and C/O, by chemistry tier", fontsize=12.5)
    axA.legend(fontsize=8.0, loc="upper left")
    axA.set_aspect("equal", adjustable="datalim")

    # inset: zoom on equilibrium + photochemistry (the small ellipses)
    small = [t for t in order if t != "Q"]
    if small:
        zx = 1.25 * max(ell[t][2] for t in small)
        zy = 1.25 * max(ell[t][3] for t in small)
        axins = axA.inset_axes([0.60, 0.06, 0.37, 0.37])
        for t in small:
            ex, ey, *_ = ell[t]
            axins.plot(ex, ey, color=TIER_STYLE[t]["color"], lw=1.8)
        axins.axhline(0, color="0.9", lw=0.5); axins.axvline(0, color="0.9", lw=0.5)
        axins.plot(0, 0, "k+", ms=5)
        axins.set_xlim(-zx, zx); axins.set_ylim(-zy, zy)
        axins.tick_params(labelsize=6.5)
        axins.set_title("zoom: equil. vs photochem.", fontsize=7.5)
        axA.indicate_inset_zoom(axins, edgecolor="0.5", lw=0.8)

    # (b) where the metallicity information comes from: unique lnZ info, E vs P
    def uniqueZ(des):
        others = [j for j in range(des["J"].shape[1]) if j != 0]
        return des["wl"], Z.marginal_info_per_bin(des["J"], des["sigma"], 0, others)
    wlP, iP = uniqueZ(des_by["P"])
    scale = iP.max()
    for t in [x for x in ["E", "Q", "P"] if x in des_by]:
        wl_t, i_t = uniqueZ(des_by[t])
        st = TIER_STYLE[t]
        axB.fill_between(wl_t, 0, i_t / scale, color=st["color"],
                         alpha=0.25 if t == "P" else 0.12, lw=0, zorder=1 if t != "P" else 2)
        axB.plot(wl_t, i_t / scale, color=st["color"], lw=1.8, label=st["label"], zorder=3)
    for name, lam in Z.BANDS.items():
        if wlP.min() <= lam <= wlP.max():
            axB.axvline(lam, color="0.8", lw=0.7, ls=":", zorder=0)
            axB.text(lam, 1.02, name, fontsize=8.5, color="0.42", ha="center", va="bottom")
    axB.set_xlim(wlP.min(), wlP.max()); axB.set_ylim(0, 1.14)
    axB.set_xlabel(r"Wavelength ($\mu$m)")
    axB.set_ylabel("unique $\\ln Z$ information\n(relative to photochem peak)")
    axB.set_title("(b) where the metallicity information comes from", fontsize=12.5)
    axB.legend(fontsize=8.6, loc="upper left")
    axB.annotate("SO$_2$: photochemistry\nonly", xy=(4.05, 0.98), xytext=(2.9, 0.66),
                 fontsize=9, color="#8a1a0d", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#8a1a0d", lw=1.0))

    fig.suptitle("Photochemistry doesn't just add a feature: it adds measurable "
                 "metallicity information", fontsize=12.8, y=1.0)
    fig.text(0.5, -0.02, "Same radiative transfer + real errors for all three; only the "
             "VULCAN-JAX chemistry physics differs. Kzz, $T_{\\rm int}$, ln$R_0$, offsets "
             "marginalized. Absolute $\\sigma$ are best-case (no clouds / free T-P).",
             ha="center", fontsize=8.3, style="italic", color="0.35")
    fig.subplots_adjust(left=0.075, right=0.975, top=0.9, bottom=0.14)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
