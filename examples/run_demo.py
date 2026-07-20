"""Headline demo: gradients of a WASP-39b transmission spectrum, straight from physics.

Builds the full chain (live VULCAN-JAX chemistry -> ExoJax ArtTransPure), evaluates the
transit spectrum at the baseline, and pushes one forward-mode tangent per physical
parameter (ln Z, C/O, ln Kzz, dT -- a uniform T offset) all the way to data space. Renders a 2x2 figure
where every spectrum point is colored by d(transit_depth)/d(parameter): the colored
regions are exactly the wavelengths that best constrain that parameter -- a quantitative
observation-planning view.

Run:  (vulcan env)  python examples/run_demo.py
First run downloads ExoMol/HITEMP line lists for H2O/CO2/CH4/SO2 (CO is cached).
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


def compute_jacobian(forward, theta0):
    """Return (primal depth (n_nu,), J (n_nu, 4)) via one forward-mode jvp per parameter."""
    primal = np.asarray(forward(theta0))
    cols = []
    for k in range(4):
        e = jnp.zeros(4, dtype=jnp.float64).at[k].set(1.0)
        tk = time.time()
        _, dv = jax.jvp(forward, (theta0,), (e,))
        dv = np.asarray(dv)
        print(f"[demo]   d/d{config.THETA_LABELS[k]:6s} {time.time()-tk:.1f}s  "
              f"finite={np.all(np.isfinite(dv))}  max|.|={np.nanmax(np.abs(dv)):.3e}", flush=True)
        cols.append(dv)
    return primal, np.stack(cols, axis=1)


def _bin_constant_R(wl_um, depth, J, R=2000, wl_lo=3.0, wl_hi=5.0):
    """Bin onto a constant-resolving-power grid (log-spaced in wavelength, R=lambda/dlambda)."""
    order = np.argsort(wl_um)
    wl = wl_um[order]; d = depth[order]; Jo = J[order]
    sel = (wl >= wl_lo) & (wl <= wl_hi)
    wl, d, Jo = wl[sel], d[sel], Jo[sel]
    nb = int(round(np.log(wl_hi / wl_lo) * R))
    edges = wl_lo * np.exp(np.arange(nb + 1) / R)        # d(ln lambda) = 1/R per bin
    idx = np.clip(np.digitize(wl, edges) - 1, 0, nb - 1)
    wl_b = np.full(nb, np.nan); d_b = np.full(nb, np.nan)
    J_b = np.full((nb, J.shape[1]), np.nan)
    for b in range(nb):
        m = idx == b
        if m.any():
            wl_b[b] = wl[m].mean(); d_b[b] = d[m].mean(); J_b[b] = Jo[m].mean(axis=0)
    keep = np.isfinite(wl_b)
    return wl_b[keep], d_b[keep], J_b[keep]


def make_figure(wl_um, depth, J, molecules, out_png):
    """2x2 transmission spectrum (binned to R=2000, 3-5 um) as a line colored by
    d(depth)/d(parameter) on a symmetric-log color scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    if not (config.JP / "scripts" / "_common.py").is_file():
        raise RuntimeError(f"jax_paper sibling not found at {config.JP} -- manuscript figure scripts require it (set VULCAN_PROJECT_ROOT)")
    sys.path.insert(0, str(config.JP / "scripts"))
    from _common import apply_style
    apply_style()

    wl, d_ppm, Jb = _bin_constant_R(wl_um, depth * 1e6, J, R=2000)
    pts = np.array([wl, d_ppm]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    segc = 0.5 * (Jb[:-1] + Jb[1:])                       # per-segment color

    titles = [r"$\partial d / \partial \ln Z$",
              r"$\partial d / \partial (\mathrm{C/O})$",
              r"$\partial d / \partial \ln K_{zz}$",
              r"$\partial d / \partial (\Delta T)$"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), sharex=True, sharey=True)
    pad = 0.04 * (d_ppm.max() - d_ppm.min())
    for k, ax in enumerate(axes.flat):
        c = segc[:, k]
        # Highlight WHERE to look: importance is |sensitivity| (sign only gives the
        # direction the feature moves). berlin runs dark-center -> bright-ends, so
        # importance maps to brightness: dark = ignore, bright = look here. The P88
        # bound saturates the informative regions to the bright ends and fades the
        # insensitive continuum into the dark center.
        a = float(np.nanpercentile(np.abs(c), 88)) or 1.0
        lc = LineCollection(segs, cmap="berlin",
                            norm=Normalize(vmin=-a, vmax=+a), lw=2.0)
        lc.set_array(c)
        ax.add_collection(lc)
        ax.set_xlim(3.0, 5.0); ax.set_ylim(d_ppm.min() - pad, d_ppm.max() + pad)
        cb = fig.colorbar(lc, ax=ax, pad=0.01, extend="both")
        cb.ax.tick_params(labelsize=14)
        cb.ax.yaxis.get_offset_text().set_fontsize(13)
        ax.set_title(titles[k], fontsize=18)
    for ax in axes[-1]:
        ax.set_xlabel(r"Wavelength ($\mu$m)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Transit depth $(R_p/R_\star)^2$  [ppm]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"[demo] wrote {out_png}", flush=True)


def replot() -> int:
    """Re-render the figure from outputs/sensitivity.npz (no recompute)."""
    d = np.load(config.OUTPUTS / "sensitivity.npz", allow_pickle=True)
    make_figure(d["wl_um"], d["depth"], d["J"], list(d["molecules"]),
                config.FIGS / "sensitivity_spectrum.png")
    return 0


def main() -> int:
    if "--replot" in sys.argv:
        return replot()
    t0 = time.time()
    fb = build_forward(config.FULL)
    theta0 = jnp.asarray(config.THETA0, dtype=jnp.float64)

    print("[demo] computing primal + 4 forward-mode jvps ...", flush=True)
    primal, J = compute_jacobian(fb.forward, theta0)
    if not np.all(np.isfinite(primal)) or not np.all(np.isfinite(J)):
        print("[demo] FAIL: non-finite output", flush=True)
        return 1

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    np.savez(config.OUTPUTS / "sensitivity.npz",
             wl_um=fb.rt.wl_um, nu_grid=fb.rt.nu_grid, depth=primal, J=J,
             theta_labels=np.array(config.THETA_LABELS), theta0=np.asarray(config.THETA0),
             molecules=np.array(fb.rt.molecules))
    make_figure(fb.rt.wl_um, primal, J, fb.rt.molecules,
                config.FIGS / "sensitivity_spectrum.png")
    print(f"[demo] done in {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
