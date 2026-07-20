"""Wide-band (1-15 um, R=100) parameter-sensitivity figures: transmission AND emission.

The 4 expensive forward-mode jvps go through chemistry+bridge ONCE via
``g(theta) -> (ART-grid VMR, VMR_H2, T, mmw)``; the transmission and emission spectra
then each apply a CHEAP rt-jvp to the shared tangent. By the chain rule this is identical
to a full-forward jvp, so both modes cost ~one run's chemistry. Figures are berlin,
importance-highlighted (dark = ignore, bright = look here), binned to R=100, in figs/.

Run:  (vulcan env)  python examples/run_figs.py     # first run downloads HITRAN over 1-15 um
"""
from __future__ import annotations

import sys
import time

import numpy as np

from retrieval_framework.forward import config
# import order is load-bearing: vulcan_chem before exojax (sets env + jax x64)
from retrieval_framework.forward import vulcan_chem
import jax
import jax.numpy as jnp
from retrieval_framework.forward import exojax_rt
from retrieval_framework.forward import interp_map


def bin_R(wl, vals, J, R):
    """Bin (wl, vals, J) onto a constant-resolving-power grid (R = lambda/dlambda)."""
    o = np.argsort(wl); wl = wl[o]; vals = vals[o]; J = J[o]
    lo, hi = float(wl.min()), float(wl.max())
    nb = int(round(np.log(hi / lo) * R))
    edges = lo * np.exp(np.arange(nb + 1) / R)
    idx = np.clip(np.digitize(wl, edges) - 1, 0, nb - 1)
    wb = np.full(nb, np.nan); vb = np.full(nb, np.nan); Jb = np.full((nb, J.shape[1]), np.nan)
    for b in range(nb):
        m = idx == b
        if m.any():
            wb[b] = wl[m].mean(); vb[b] = vals[m].mean(); Jb[b] = J[m].mean(axis=0)
    k = np.isfinite(wb)
    return wb[k], vb[k], Jb[k]


def make_fig(wl_um, spec, J, kind, out_png, display_R):
    """Same-style berlin importance figure; kind in {'transmission','emission'}."""
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

    if kind == "transmission":
        yv = spec * 1e6; ylab = r"Transit depth $(R_p/R_\star)^2$  [ppm]"; sym = "d"
    else:
        yv = spec; ylab = r"Emergent flux  [erg s$^{-1}$ cm$^{-2}$/cm$^{-1}$]"; sym = "F"
    wl, yb, Jb = bin_R(wl_um, yv, J, display_R)
    pts = np.array([wl, yb]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    segc = 0.5 * (Jb[:-1] + Jb[1:])

    # T_int dropped (knobs ln Z, C/O, ln Kzz). Panels are placed at explicit inch-based
    # positions so each is exactly s x s (square); the colorbar is a thin axes of the
    # same height right beside it. No layout engine -> the square aspect is guaranteed.
    titles = [rf"$\partial {sym} / \partial \ln Z$",
              rf"$\partial {sym} / \partial (\mathrm{{C/O}})$",
              rf"$\partial {sym} / \partial \ln K_{{zz}}$"]
    s, lm, bm, tm, rm = 3.5, 0.85, 0.62, 0.42, 0.45   # inches: panel side + margins
    cw, cg, gp = 0.15, 0.05, 0.78                      # colorbar width, gap, panel gap
    fig_w = lm + 3 * s + 3 * (cg + cw) + 2 * gp + rm
    fig_h = bm + s + tm
    fig = plt.figure(figsize=(fig_w, fig_h))
    pad = 0.04 * (yb.max() - yb.min())
    for k in range(3):
        x0 = lm + k * (s + cg + cw + gp)
        ax = fig.add_axes([x0 / fig_w, bm / fig_h, s / fig_w, s / fig_h])
        cax = fig.add_axes([(x0 + s + cg) / fig_w, bm / fig_h, cw / fig_w, s / fig_h])
        c = segc[:, k]
        a = float(np.nanpercentile(np.abs(c), 88)) or 1.0   # bright = look here
        lc = LineCollection(segs, cmap="berlin", norm=Normalize(vmin=-a, vmax=+a), lw=1.8)
        lc.set_array(c)
        ax.add_collection(lc)
        ax.set_xlim(wl.min(), wl.max()); ax.set_ylim(yb.min() - pad, yb.max() + pad)
        ax.set_title(titles[k], fontsize=17)
        ax.set_xlabel(r"Wavelength ($\mu$m)")
        if k == 0:
            ax.set_ylabel(ylab)
        else:
            ax.set_yticklabels([])
        cb = fig.colorbar(lc, cax=cax, extend="both")
        cb.ax.tick_params(labelsize=11); cb.ax.yaxis.get_offset_text().set_fontsize(10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"[figs] wrote {out_png}", flush=True)


def replot() -> int:
    """Re-render both figures from outputs/wide_sensitivity.npz (no recompute)."""
    d = np.load(config.OUTPUTS / "wide_sensitivity.npz", allow_pickle=True)
    R = config.WIDE["display_R"]
    make_fig(d["wl_um"], d["depth"], d["J_trans"], "transmission",
             config.FIGS / "sensitivity_transmission_1-15um.png", R)
    make_fig(d["wl_um"], d["flux"], d["J_emis"], "emission",
             config.FIGS / "sensitivity_emission_1-15um.png", R)
    return 0


def main() -> int:
    if "--replot" in sys.argv:
        return replot()
    t0 = time.time()
    prof = config.WIDE
    chem = vulcan_chem.build_chem_model(prof)
    trt = exojax_rt.build_rt_model(prof)
    ert = exojax_rt.build_emis_model(trt, prof)
    to_art = interp_map.make_to_art(chem.p_bar, trt.p_art_bar)
    mol_cols = {k: chem.sidx[config.MOLECULES[k]["vulcan"]] for k in trt.molecules}
    h2 = chem.sidx[config.BULK_H2_VULCAN]
    he = chem.sidx["He"]            # H2-He CIA partner (required by both RTs now)
    T_base = jnp.asarray(chem.T_base); masses = chem.species_masses

    def g(theta):
        ymix = chem.converged_ymix(theta)
        T_art = to_art(T_base + theta[3])
        mmw_art = to_art(ymix @ masses)
        vmr = {k: to_art(ymix[:, c]) for k, c in mol_cols.items()}
        vmr_h2 = to_art(ymix[:, h2])
        vmr_he = to_art(ymix[:, he])
        return (vmr, vmr_h2, T_art, mmw_art, vmr_he)

    def trans_of(gg):
        return trt.transmission_depth(*gg)

    def emis_of(gg):
        return ert.emission_flux(*gg)

    theta0 = jnp.asarray(config.THETA0, dtype=jnp.float64)
    print("[figs] primal (shared chemistry + both RTs) ...", flush=True)
    g0 = g(theta0)
    trans0 = np.asarray(trans_of(g0)); emis0 = np.asarray(emis_of(g0))
    ok0 = np.all(np.isfinite(trans0)) and np.all(np.isfinite(emis0))
    print(f"[figs] primal {time.time()-t0:.0f}s  trans {trans0.min()*1e6:.0f}-{trans0.max()*1e6:.0f} ppm  "
          f"emis {emis0.min():.2e}-{emis0.max():.2e}  finite={ok0}", flush=True)
    if not ok0:
        print("[figs] FAIL: non-finite primal", flush=True); return 1

    Jt, Je = [], []
    for k in range(4):
        e = jnp.zeros(4, dtype=jnp.float64).at[k].set(1.0)
        tk = time.time()
        _, dg = jax.jvp(g, (theta0,), (e,))              # expensive: chemistry
        _, dt = jax.jvp(trans_of, (g0,), (dg,))          # cheap
        _, de = jax.jvp(emis_of, (g0,), (dg,))           # cheap
        dt = np.asarray(dt); de = np.asarray(de)
        ok = np.all(np.isfinite(dt)) and np.all(np.isfinite(de))
        print(f"[figs]   d/d{config.THETA_LABELS[k]:6s} {time.time()-tk:.0f}s  finite={ok}  "
              f"|trans|max={np.nanmax(np.abs(dt)):.2e} |emis|max={np.nanmax(np.abs(de)):.2e}", flush=True)
        if not ok:
            print(f"[figs] FAIL: non-finite tangent {config.THETA_LABELS[k]}", flush=True); return 1
        Jt.append(dt); Je.append(de)
    Jt = np.stack(Jt, axis=1); Je = np.stack(Je, axis=1)

    config.OUTPUTS.mkdir(parents=True, exist_ok=True)
    config.FIGS.mkdir(parents=True, exist_ok=True)
    np.savez(config.OUTPUTS / "wide_sensitivity.npz", wl_um=trt.wl_um, depth=trans0,
             flux=emis0, J_trans=Jt, J_emis=Je, theta0=np.asarray(config.THETA0),
             molecules=np.array(trt.molecules))
    make_fig(trt.wl_um, trans0, Jt, "transmission",
             config.FIGS / "sensitivity_transmission_1-15um.png", prof["display_R"])
    make_fig(trt.wl_um, emis0, Je, "emission",
             config.FIGS / "sensitivity_emission_1-15um.png", prof["display_R"])
    print(f"[figs] done in {time.time()-t0:.0f}s -> {config.FIGS}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
