"""Observed-spectrum handling: load a real product-CSV spectrum described by the
Config's ``obs_dir``/``obs_products``/``combo`` fields, and build the *linear*
operators the differentiable likelihood needs.

Two operators are precomputed once (static numpy) and then used as constant jnp arrays
inside the jitted likelihood:

    B  (n_bin, n_native)   bin-averaging matrix: binned_depth = B @ native_depth. The
                           binning is a d(lambda)-weighted trapezoidal average
                           expressed as a matrix so its forward-mode/JVP derivative is
                           exact and free (numerically identical to zco_lib.bin_to_obs,
                           the reference implementation it was validated against).
    O  (n_bin, G-1)        instrument-offset design: bin i in group g>0 gets a flat
                           depth offset. depth_with_offset = binned + O @ (offset_ppm*1e-6).

Product CSV format (the Carter & May 2024 / Zenodo convention): one header line, then
rows of  [index, wave, wave_low, wave_hig, rp/rs, rp/rs_err_low, rp/rs_err_hih].
depth = (rp/rs)^2 and sigma_depth = 2*(rp/rs)*sigma_rprs with the near-symmetric
low/high errors averaged. A case with a different upstream format should convert its
files to this layout once, next to the originals.

For synthetic runs where no real bin overlaps the model band (the offline CO-only
smoke), or when ``obs_dir`` is unset, a constant-R synthetic bin grid is generated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

OFFSET_UNIT = 1.0e-6   # offset parameter is in ppm -> fractional depth


def read_rprs_csv(path: Path) -> Tuple[np.ndarray, ...]:
    """Read one (Rp/Rs)-format product CSV -> (wl, wl_lo, wl_hi, depth_frac, sigma_frac)."""
    a = np.genfromtxt(path, delimiter=",", skip_header=1)
    wl, wlo, whi, rprs, el, eh = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    sig_rprs = 0.5 * (np.abs(el) + np.abs(eh))
    depth = rprs ** 2
    sigma = 2.0 * rprs * sig_rprs
    lo = np.minimum(wlo, whi)
    hi = np.maximum(wlo, whi)
    good = np.isfinite(wl) & np.isfinite(sigma) & (sigma > 0) & (hi > lo)
    return wl[good], lo[good], hi[good], depth[good], sigma[good]


def load_real_observations(cfg: Any) -> Dict[str, np.ndarray]:
    """Concatenate the case's product CSVs (cfg.obs_dir / cfg.obs_products, groups
    selected by cfg.combo) into one combined spectrum, clipped to the band.

    Returns a dict of per-bin arrays sorted by wavelength: wl, wl_lo, wl_hi (um),
    depth, sigma (fractional (Rp/Rs)^2), group (label per bin), groups (ordered
    unique labels; reference = groups[0]).
    """
    obs_dir = Path(cfg.obs_dir)
    products = dict(cfg.obs_products)
    missing = [g for g in cfg.combo if g not in products]
    if missing:
        raise ValueError(f"combo groups {missing} not in obs_products "
                         f"(known: {sorted(products)})")
    W, LO, HI, D, S, G = [], [], [], [], [], []
    for grp in cfg.combo:
        for fname in products[grp]:
            wl, lo, hi, d, s = read_rprs_csv(obs_dir / fname)
            W.append(wl); LO.append(lo); HI.append(hi); D.append(d); S.append(s)
            G.append(np.array([grp] * len(wl)))
    wl = np.concatenate(W); lo = np.concatenate(LO); hi = np.concatenate(HI)
    depth = np.concatenate(D); sigma = np.concatenate(S); group = np.concatenate(G)
    sel = (wl >= float(cfg.obs_wl_lo)) & (wl <= float(cfg.obs_wl_hi))
    wl, lo, hi, depth, sigma, group = (wl[sel], lo[sel], hi[sel], depth[sel],
                                       sigma[sel], group[sel])
    o = np.argsort(wl)
    groups = list(dict.fromkeys(group[o].tolist()))   # ordered-unique; ref = groups[0]
    return dict(wl=wl[o], wl_lo=lo[o], wl_hi=hi[o], depth=depth[o], sigma=sigma[o],
                group=group[o], groups=groups)


def _synthetic_bin_grid(wl_lo: float, wl_hi: float, R: int = 200,
                        sigma_ppm: float = 120.0) -> Dict[str, np.ndarray]:
    """A simple constant-R bin grid across [wl_lo, wl_hi] (single instrument group).
    Used only when no real bin overlaps the model band (the CO-only smoke)."""
    edges = [wl_lo]
    while edges[-1] < wl_hi:
        edges.append(edges[-1] * (1.0 + 1.0 / R))
    edges = np.asarray(edges)
    lo, hi = edges[:-1], edges[1:]
    wl = 0.5 * (lo + hi)
    n = wl.size
    return dict(wl=wl, wl_lo=lo, wl_hi=hi,
                depth=np.full(n, np.nan), sigma=np.full(n, sigma_ppm * 1e-6),
                group=np.array(["SYNTH"] * n), groups=["SYNTH"])


def restrict_to_model_band(obs: Dict[str, np.ndarray], wl_model_um: np.ndarray
                           ) -> Dict[str, np.ndarray]:
    """Keep only observed bins whose [lo,hi] sits strictly inside the model wavelength
    span (the binning operator is undefined outside it)."""
    wl = np.asarray(wl_model_um, float)
    wmin, wmax = float(wl.min()), float(wl.max())
    lo, hi = np.asarray(obs["wl_lo"], float), np.asarray(obs["wl_hi"], float)
    keep = (lo >= wmin) & (hi <= wmax) & (hi > lo)
    out = {k: (np.asarray(v)[keep] if isinstance(v, np.ndarray) and np.asarray(v).shape == lo.shape else v)
           for k, v in obs.items()}
    # recompute ordered-unique groups after the cut
    if "group" in out and np.asarray(out["group"]).size:
        out["groups"] = list(dict.fromkeys(np.asarray(out["group"]).tolist()))
    return out


def get_observation_grid(cfg: Any, wl_model_um: np.ndarray) -> Tuple[Dict[str, np.ndarray], bool]:
    """Return (obs, real_bins). Real product bins are used when a source is configured
    and overlaps the model band; a synthetic grid (real_bins=False) is built ONLY for
    a synthetic run (generate_synthetic_data=True). A real-data run that finds no
    usable bin RAISES rather than silently fabricating a synthetic grid to fit."""
    wl = np.asarray(wl_model_um, float)
    synthetic = bool(cfg.generate_synthetic_data)
    have_source = bool(cfg.obs_dir) and bool(cfg.obs_products)

    if have_source:
        real = load_real_observations(cfg)
        real = restrict_to_model_band(real, wl)
        n_real = int(np.asarray(real["wl"]).size)
        if n_real >= 4:
            return real, True
        # source configured but the band clips out (nearly) all of it
        if not synthetic:
            raise ValueError(
                f"only {n_real} of the configured observed bins fall inside the model "
                f"band [{wl.min():.3f}, {wl.max():.3f}] um (need >=4) -- the model band "
                "does not overlap the data. Fix obs_wl_lo/hi or nu_min/nu_max; refusing "
                "to silently fabricate a synthetic grid for a real-data run.")
        # synthetic run with a deliberately non-overlapping band (e.g. the CO smoke)
    elif not synthetic:
        raise ValueError("no observation source configured (obs_dir/obs_products unset) "
                         "and generate_synthetic_data=False -- nothing to fit")

    # synthetic run: build a grid across the model span to inject onto
    pad = 0.001
    grid = _synthetic_bin_grid(float(wl.min()) * (1 + pad), float(wl.max()) * (1 - pad))
    return grid, False


def build_binning_matrix(wl_model_um: np.ndarray, obs: Dict[str, np.ndarray]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Linear bin-averaging matrix. Returns (keep_mask (n_bin,), B (n_keep, n_native)).

    B[b] are the native-grid weights whose dot with the native depth equals the
    d(lambda)-weighted trapezoidal bin average of zco_lib.bin_to_obs -- exact, so the
    JVP of a binned depth is B @ (JVP of native depth).
    """
    wl = np.asarray(wl_model_um, float)
    order = np.argsort(wl)
    wl_s = wl[order]
    n_native = wl.size
    lo_all = np.asarray(obs["wl_lo"], float)
    hi_all = np.asarray(obs["wl_hi"], float)
    nb = lo_all.size

    keep = np.zeros(nb, dtype=bool)
    rows = []
    for b in range(nb):
        lo, hi = lo_all[b], hi_all[b]
        if not (wl_s[0] <= lo < hi <= wl_s[-1]):
            continue
        inside = np.where((wl_s > lo) & (wl_s < hi))[0]
        x = np.concatenate([[lo], wl_s[inside], [hi]])
        dx = np.diff(x)
        wnode = np.zeros(x.size)
        wnode[:-1] += 0.5 * dx
        wnode[1:] += 0.5 * dx
        wnode /= (hi - lo)                       # bin average
        row = np.zeros(n_native)
        # interior nodes map to their native points directly (interp weight 1)
        for k, j in enumerate(inside):
            row[order[j]] += wnode[1 + k]
        # the lo / hi endpoints are linear interpolations of the bracketing native points
        for xend, wend in ((x[0], wnode[0]), (x[-1], wnode[-1])):
            jj = int(np.clip(np.searchsorted(wl_s, xend), 1, n_native - 1))
            xl, xr = wl_s[jj - 1], wl_s[jj]
            t = 0.0 if xr == xl else (xend - xl) / (xr - xl)
            row[order[jj - 1]] += wend * (1.0 - t)
            row[order[jj]] += wend * t
        rows.append(row)
        keep[b] = True
    B = np.asarray(rows) if rows else np.zeros((0, n_native))
    return keep, B


def build_offset_design(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """(n_bin, G-1) indicator matrix: column k is 1 for bins in groups[k+1] (relative to
    the reference group groups[0]). Empty (n_bin, 0) if <2 groups."""
    groups = list(obs["groups"])
    group_per_bin = np.asarray(obs["group"])
    if len(groups) <= 1:
        return np.zeros((group_per_bin.size, 0))
    cols = [(group_per_bin == g).astype(float) for g in groups[1:]]
    return np.column_stack(cols)
