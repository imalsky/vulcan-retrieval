"""Figure 1 -- WHERE the metallicity and C/O information lives in WASP-39b's spectrum.

Top:    the real combined JWST spectrum (Carter & May 2024: NIRISS SOSS + NIRSpec G395H),
        coloured by instrument, molecular bands marked.
Bottom: the per-wavelength MARGINAL 'unique information' for ln Z (red) and ln(C/O) (blue)
        -- each parameter's noise-whitened sensitivity after the OTHER parameter and every
        nuisance (lnKzz, T_int, lnR0, inter-instrument offset) are projected out. Metallicity
        lights up on the photochemical SO2 feature (~4.05 um); C/O lights up on the carbon
        carriers (CH4 3.3, CO 4.66) and the H2O bands -- different molecules, so both are
        measurable.

Reads data/zco_jacobians.npz (build_zco_jacobians.py) + data/cm24_wasp39b/*.csv. PNG only.
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

C_Z, C_CO = "#cc3311", "#2b6cb0"
INST_COLOR = {"NIRISS": "#4c4c9d", "G395H": "#3a8a3a", "PRISM": "0.25", "NIRCam": "#b8860b"}


def _bands(ax, y, only=None, text=True):
    x0, x1 = ax.get_xlim()
    for name, lam in Z.BANDS.items():
        if (only is None or name in only) and x0 <= lam <= x1:
            ax.axvline(lam, color="0.78", lw=0.7, ls=":", zorder=0)
            if text:
                ax.text(lam, y, name, fontsize=8.5, color="0.42", ha="center", va="bottom")


def main(out=Z.FIGS / "zco_information.png", combo=Z.DEFAULT_COMBO, tier="P"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _common import apply_style
    apply_style()

    wl_model, tiers, meta = Z.load_jacobians()
    obs = Z.load_combined(combo)
    des = Z.build_design(tiers[tier], wl_model, obs)
    J, sigma = des["J"], des["sigma"]
    wl, group = des["wl"], des["group"]

    others_Z = [j for j in range(J.shape[1]) if j != 0]
    others_CO = [j for j in range(J.shape[1]) if j != 1]
    iZ = Z.marginal_info_per_bin(J, sigma, 0, others_Z); iZ = iZ / iZ.max()
    iC = Z.marginal_info_per_bin(J, sigma, 1, others_CO); iC = iC / iC.max()

    fig, (axs, axi) = plt.subplots(2, 1, figsize=(9.8, 7.2), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.07))
    # top: measured spectrum, per instrument (kept bins only, aligned with the info panel)
    depthO_ppm = des["depthO"] * 1e6
    sigmaO_ppm = sigma * 1e6
    for g in obs["groups"]:
        m = group == g
        axs.errorbar(wl[m], depthO_ppm[m], yerr=sigmaO_ppm[m], fmt="o", ms=3.0, lw=0,
                     elinewidth=0.6, color=INST_COLOR.get(g, "0.3"), ecolor="0.75",
                     zorder=3, label=g)
    axs.set_xlim(wl.min(), wl.max())
    dall = depthO_ppm
    pad = 0.08 * (dall.max() - dall.min())
    axs.set_ylim(dall.min() - pad, dall.max() + 2.2 * pad)
    _bands(axs, dall.max() + 1.15 * pad)
    axs.set_ylabel("Transit depth [ppm]")
    axs.legend(loc="upper left", fontsize=9, frameon=True, ncol=len(obs["groups"]))
    axs.set_title("WASP-39b transmission spectrum  (real JWST: Carter & May 2024)", fontsize=12.5)

    # bottom: unique information
    axi.fill_between(wl, 0, iZ, color=C_Z, alpha=0.48, lw=0, zorder=2)
    axi.plot(wl, iZ, color=C_Z, lw=1.6, label=r"metallicity  $\ln Z$", zorder=3)
    axi.fill_between(wl, 0, iC, color=C_CO, alpha=0.34, lw=0, zorder=1)
    axi.plot(wl, iC, color=C_CO, lw=1.6, label=r"carbon-to-oxygen  $\ln(\mathrm{C/O})$", zorder=3)
    axi.set_ylim(0, 1.32)
    _bands(axi, 1.05, text=False)
    wZ = wl[np.argmax(iZ)]
    axi.annotate("SO$_2$ (photochemical)\n$\\to$ metallicity anchor", xy=(wZ, 1.0),
                 xytext=(wZ - 0.9, 0.60), fontsize=9, color="#8a1a0d", ha="left",
                 arrowprops=dict(arrowstyle="->", color="#8a1a0d", lw=1.0))
    axi.set_xlabel(r"Wavelength ($\mu$m)")
    axi.set_ylabel("unique information\n(relative, marginalized)")
    axi.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle("Metallicity and C/O are read from different molecules, so a broad spectrum "
                 "measures both", fontsize=12.8, y=0.975)
    fig.text(0.5, 0.005, "'Unique information' = each parameter's whitened sensitivity after the "
             "other and all nuisances (lnKzz, $T_{\\rm int}$, ln$R_0$, offsets) are projected out. "
             "Real Carter & May (2024) errors, 1 transit.",
             ha="center", fontsize=8.2, style="italic", color="0.35")
    fig.subplots_adjust(left=0.115, right=0.965, top=0.905, bottom=0.13)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
