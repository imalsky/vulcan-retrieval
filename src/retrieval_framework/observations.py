"""Observed spectra: read the product CSVs and build the linear operators the likelihood needs.

``B`` [n_bin, n_native] is the d(lambda)-weighted trapezoidal bin average, so a binned depth's JVP is exact. ``O``
[n_bin,
G-1] gives each non-reference group a flat offset; ``N`` [n_bin, G] is the group indicator behind per-group noise.

A product CSV follows the Carter & May (2024) convention: one header line, then ``index, wave, wave_low, wave_hig,
rp/rs, rp/rs_err_low, rp/rs_err_hih``. depth = (rp/rs)^2 and sigma = 2 (rp/rs) sigma_rprs, the two errors averaged. A
synthetic constant-R grid is built only for a synthetic run with no source or too few bins in the model band.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jax.typing import ArrayLike

__all__ = [
    "MIN_REAL_BINS",
    "NOISE_PREFIX",
    "PPM",
    "build_binning_matrix",
    "build_group_design",
    "build_offset_design",
    "get_observation_grid",
    "load_real_observations",
    "noise_map",
    "noise_scale_per_bin",
    "read_rprs_csv",
    "restrict_to_model_band",
]

logger = logging.getLogger(__name__)

PPM = 1.0e-6  # one part per million of the transit depth (offsets, sigmas)
# The scalar noise-inflation parameter's name; the per-group ones are f"{NOISE_PREFIX}_{group}".
NOISE_PREFIX = "noise_inflation"
# A real-data run needs at least this many observed bins inside the model band.
MIN_REAL_BINS = 4
# Synthetic bin grid: resolving power, per-bin sigma, and the fractional inset from the model band edges.
SYNTH_R = 200
SYNTH_SIGMA_PPM = 120.0
SYNTH_BAND_PAD_FRAC = 0.001


def _validated_model_wavelengths(wl_model_um: np.ndarray) -> np.ndarray:
    """Return a finite, positive, duplicate-free 1-D model grid, or raise."""
    wl = np.asarray(wl_model_um, float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("model wavelength grid must be 1-D with at least two points")
    if not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError("model wavelength grid must be finite and positive")
    if np.unique(wl).size != wl.size:
        raise ValueError("model wavelength grid contains duplicate coordinates")
    return wl


def read_rprs_csv(path: Path) -> tuple[np.ndarray, ...]:
    """Read one (Rp/Rs)-format product CSV.

    Scoped to raw upstream products: rows with non-finite wl or sigma, or sigma <= 0, are dropped (published products
    carry padding rows), and swapped wl_lo/wl_hi edges are repaired. Everything downstream validates.

    Parameters
    ----------
    path : pathlib.Path
        The CSV.

    Returns
    -------
    tuple of numpy.ndarray
        ``(wl, wl_lo, wl_hi, depth, sigma)``: um, and fractional (Rp/Rs)^2.
    """
    a = np.genfromtxt(path, delimiter=",", skip_header=1)
    wl, wlo, whi, rprs, el, eh = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    sig_rprs = 0.5 * (np.abs(el) + np.abs(eh))
    depth = rprs**2
    sigma = 2.0 * rprs * sig_rprs
    lo = np.minimum(wlo, whi)
    hi = np.maximum(wlo, whi)
    good = np.isfinite(wl) & np.isfinite(sigma) & (sigma > 0) & (hi > lo)
    return wl[good], lo[good], hi[good], depth[good], sigma[good]


def load_real_observations(cfg: Any) -> dict[str, Any]:
    """Concatenate the case's product CSVs into one spectrum, clipped to the fit window.

    Parameters
    ----------
    cfg : config_schema.Config
        Reads ``obs_dir``, ``obs_products``, ``combo``, ``obs_wl_lo``, ``obs_wl_hi``, ``obs_max_bin_R``.

    Returns
    -------
    dict
        Per-bin arrays sorted by wavelength: ``wl``, ``wl_lo``, ``wl_hi`` (um), ``depth``, ``sigma`` (fractional),
        ``group``, and ``groups``, the ordered unique labels; the reference is ``groups[0]``.

    Raises
    ------
    ValueError
        If ``combo`` names a group not in ``obs_products``.
    """
    obs_dir = Path(cfg.obs_dir)
    products = dict(cfg.obs_products)
    missing = [g for g in cfg.combo if g not in products]
    if missing:
        raise ValueError(f"combo groups {missing} not in obs_products (known: {sorted(products)})")
    W, LO, HI, D, S, G = [], [], [], [], [], []
    for grp in cfg.combo:
        for fname in products[grp]:
            wl, lo, hi, d, s = read_rprs_csv(obs_dir / fname)
            W.append(wl)
            LO.append(lo)
            HI.append(hi)
            D.append(d)
            S.append(s)
            G.append(np.array([grp] * len(wl)))
    wl = np.concatenate(W)
    lo = np.concatenate(LO)
    hi = np.concatenate(HI)
    depth = np.concatenate(D)
    sigma = np.concatenate(S)
    group = np.concatenate(G)
    sel = (wl >= float(cfg.obs_wl_lo)) & (wl <= float(cfg.obs_wl_hi))
    r_max = float(cfg.obs_max_bin_R)
    if r_max > 0.0:
        drop = sel & (0.5 * (lo + hi) / (hi - lo) > r_max)
        if drop.any():
            logger.warning(
                "obs_max_bin_R=%g drops %d truncation-remnant bin(s) at wl %s um",
                r_max,
                int(drop.sum()),
                np.round(wl[drop], 5).tolist(),
            )
        sel &= ~drop
    wl, lo, hi, depth, sigma, group = (wl[sel], lo[sel], hi[sel], depth[sel], sigma[sel], group[sel])
    o = np.argsort(wl)
    groups = list(dict.fromkeys(group[o].tolist()))  # ordered-unique; ref = groups[0]
    return {
        "wl": wl[o],
        "wl_lo": lo[o],
        "wl_hi": hi[o],
        "depth": depth[o],
        "sigma": sigma[o],
        "group": group[o],
        "groups": groups,
    }


def _synthetic_bin_grid(
    wl_lo: float, wl_hi: float, R: int = SYNTH_R, sigma_ppm: float = SYNTH_SIGMA_PPM
) -> dict[str, Any]:
    """Return a constant-R bin grid across [wl_lo, wl_hi], one instrument group, for a synthetic run."""
    edge_list = [wl_lo]
    while edge_list[-1] < wl_hi:
        edge_list.append(edge_list[-1] * (1.0 + 1.0 / R))
    edges = np.asarray(edge_list)
    lo, hi = edges[:-1], edges[1:]
    wl = 0.5 * (lo + hi)
    n = wl.size
    return {
        "wl": wl,
        "wl_lo": lo,
        "wl_hi": hi,
        "depth": np.full(n, np.nan),
        "sigma": np.full(n, sigma_ppm * PPM),
        "group": np.array(["SYNTH"] * n),
        "groups": ["SYNTH"],
    }


def restrict_to_model_band(obs: dict[str, Any], wl_model_um: np.ndarray) -> dict[str, Any]:
    """Keep the observed bins whose [lo, hi] sits inside the model wavelength span.

    The binning operator is undefined outside it.

    Parameters
    ----------
    obs : dict
        The observations.
    wl_model_um : numpy.ndarray
        The model wavelengths, um.

    Returns
    -------
    dict
        The observations, cut, with ``groups`` recomputed.
    """
    wl = np.asarray(wl_model_um, float)
    wmin, wmax = float(wl.min()), float(wl.max())
    lo, hi = np.asarray(obs["wl_lo"], float), np.asarray(obs["wl_hi"], float)
    keep = (lo >= wmin) & (hi <= wmax) & (hi > lo)
    out = {
        k: (np.asarray(v)[keep] if isinstance(v, np.ndarray) and np.asarray(v).shape == lo.shape else v)
        for k, v in obs.items()
    }
    if "group" in out and np.asarray(out["group"]).size:
        out["groups"] = list(dict.fromkeys(np.asarray(out["group"]).tolist()))
    return out


def get_observation_grid(cfg: Any, wl_model_um: np.ndarray) -> tuple[dict[str, Any], bool]:
    """Return the observed or synthetic bins inside the model band.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.
    wl_model_um : numpy.ndarray
        The model wavelengths, um.

    Returns
    -------
    tuple
        ``(obs, real_bins)``; ``real_bins`` is False for a synthetic grid.

    Raises
    ------
    ValueError
        If a real-data run finds no usable bin, rather than fabricate a grid.
    """
    wl = np.asarray(wl_model_um, float)
    synthetic = bool(cfg.generate_synthetic_data)
    have_source = bool(cfg.obs_dir) and bool(cfg.obs_products)

    if have_source:
        real = load_real_observations(cfg)
        real = restrict_to_model_band(real, wl)
        n_real = int(np.asarray(real["wl"]).size)
        if n_real >= MIN_REAL_BINS:
            return real, True
        if not synthetic:
            raise ValueError(
                f"only {n_real} of the configured observed bins fall inside the model "
                f"band [{wl.min():.3f}, {wl.max():.3f}] um (need >={MIN_REAL_BINS}) -- the model band "
                "does not overlap the data. Fix obs_wl_lo/hi or nu_min/nu_max; refusing "
                "to silently fabricate a synthetic grid for a real-data run."
            )
    elif not synthetic:
        raise ValueError(
            "no observation source configured (obs_dir/obs_products unset) "
            "and generate_synthetic_data=False -- nothing to fit"
        )

    pad = SYNTH_BAND_PAD_FRAC
    grid = _synthetic_bin_grid(float(wl.min()) * (1 + pad), float(wl.max()) * (1 - pad))
    return restrict_to_model_band(grid, wl), False


# A cell average is a faithful binning operator only when a data bin spans at least this many R1000 model bands. Two
# admits
# Welbanks et al. (2024)'s native 0.015 um NIRCam bins (R up to 336, about three bands).
_LSF_R_SAFETY = 2.0


def _refuse_unresolved_products(wl_model_um: np.ndarray, obs: dict[str, Any]) -> None:
    """Refuse data bins fine enough that the missing LSF matters."""
    lo = np.asarray(obs["wl_lo"], float)
    hi = np.asarray(obs["wl_hi"], float)
    inside = (lo > 0) & (hi > lo)
    if not inside.any():
        return
    r_data = 0.5 * (lo[inside] + hi[inside]) / (hi[inside] - lo[inside])
    wl_s = np.sort(np.asarray(wl_model_um, float))
    r_model = float(np.median(0.5 * (wl_s[1:] + wl_s[:-1]) / np.diff(wl_s)))
    limit = r_model / _LSF_R_SAFETY
    bad = int(np.sum(r_data > limit))
    if bad:
        raise ValueError(
            f"{bad} of {r_data.size} data bins have R up to {r_data.max():.0f}, "
            f"within a factor {_LSF_R_SAFETY:g} of the model band R "
            f"({r_model:.0f}). The binning operator is a pure cell average with "
            "no instrument line-spread function, so it would model those bins "
            "wrong. Bin the product down, or add an LSF before the cell average."
        )


def build_binning_matrix(wl_model_um: np.ndarray, obs: dict[str, Any]) -> np.ndarray:
    """Return the bin-averaging matrix for bins already inside the model span.

    Its rows dotted with the native depth give the d(lambda)-weighted trapezoidal bin average. No instrument line-spread
    function is applied: that is defensible only while each bin spans at least ``_LSF_R_SAFETY`` model bands, and a
    product approaching the model's own R is refused.

    Parameters
    ----------
    wl_model_um : numpy.ndarray [n_native]
        The model wavelengths, um.
    obs : dict
        Reads ``wl_lo`` and ``wl_hi``.

    Returns
    -------
    numpy.ndarray [n_bin, n_native]
        The matrix.

    Raises
    ------
    ValueError
        If a bin lies outside the model span, or is too fine for the missing line-spread function.
    """
    wl = _validated_model_wavelengths(wl_model_um)
    _refuse_unresolved_products(wl, obs)
    order = np.argsort(wl)
    wl_s = wl[order]
    n_native = wl.size
    lo_all = np.asarray(obs["wl_lo"], float)
    hi_all = np.asarray(obs["wl_hi"], float)
    nb = lo_all.size
    out = ~((wl_s[0] <= lo_all) & (lo_all < hi_all) & (hi_all <= wl_s[-1]))
    if out.any():
        raise ValueError(
            f"{int(out.sum())} bin(s) at wl_lo {lo_all[out].tolist()} um are not inside the "
            f"model span [{wl_s[0]:g}, {wl_s[-1]:g}] um: restrict_to_model_band first"
        )

    rows = []
    for b in range(nb):
        lo, hi = lo_all[b], hi_all[b]
        inside = np.where((wl_s > lo) & (wl_s < hi))[0]
        x = np.concatenate([[lo], wl_s[inside], [hi]])
        dx = np.diff(x)
        wnode = np.zeros(x.size)
        wnode[:-1] += 0.5 * dx
        wnode[1:] += 0.5 * dx
        wnode /= hi - lo  # bin average
        row = np.zeros(n_native)
        for k, j in enumerate(inside):
            row[order[j]] += wnode[1 + k]
        # The endpoints are linear interpolations of the bracketing native points.
        for xend, wend in ((x[0], wnode[0]), (x[-1], wnode[-1])):
            jj = int(np.clip(np.searchsorted(wl_s, xend), 1, n_native - 1))
            xl, xr = wl_s[jj - 1], wl_s[jj]
            t = (xend - xl) / (xr - xl)
            row[order[jj - 1]] += wend * (1.0 - t)
            row[order[jj]] += wend * t
        rows.append(row)
    return np.asarray(rows) if rows else np.zeros((0, n_native))


def build_offset_design(obs: dict[str, Any]) -> np.ndarray:
    """Return the offset design: column k is 1 for the bins of ``groups[k + 1]``.

    Parameters
    ----------
    obs : dict
        Reads ``groups`` and ``group``.

    Returns
    -------
    numpy.ndarray [n_bin, G-1]
        The design; empty with fewer than two groups.
    """
    groups = list(obs["groups"])
    group_per_bin = np.asarray(obs["group"])
    if len(groups) <= 1:
        return np.zeros((group_per_bin.size, 0))
    cols = [(group_per_bin == g).astype(float) for g in groups[1:]]
    return np.column_stack(cols)


def build_group_design(obs: dict[str, Any]) -> np.ndarray:
    """Return the indicator matrix over every group: column k is 1 for the bins of ``groups[k]``.

    Parameters
    ----------
    obs : dict
        Reads ``groups`` and ``group``.

    Returns
    -------
    numpy.ndarray [n_bin, G]
        The indicator, so N @ b gives every bin its group's noise inflation.

    Raises
    ------
    ValueError
        If a bin's group is not listed.
    """
    groups = list(obs["groups"])
    group_per_bin = np.asarray(obs["group"]).astype(str)
    N = (
        np.column_stack([(group_per_bin == g).astype(float) for g in groups])
        if groups
        else np.zeros((group_per_bin.size, 0))
    )
    if not np.all(N.sum(axis=1) == 1.0):
        raise ValueError(f"bins of groups {sorted(set(group_per_bin) - set(groups))} are not in groups={groups}")
    return N


def noise_map(names: Iterable[str], group_per_bin: ArrayLike) -> tuple[int, np.ndarray | None] | None:
    """Say where theta's noise inflation sits and which bins each entry scales, by parameter name.

    The one map the likelihood (``pipeline.make_noise_model``) and the predictive band (``noise_scale_per_bin``)
    apply.

    Parameters
    ----------
    names : sequence of str
        The parameter names.
    group_per_bin : array-like [n_bin]
        Each bin's group.

    Returns
    -------
    tuple or None
        None with no noise parameter; ``(i, None)`` for the scalar at theta[i]; ``(lo, N)`` for one per group,
        contiguous from theta[lo], with N [n_bin, G] the bins' group indicator.

    Raises
    ------
    ValueError
        If the per-group parameters are not contiguous, or a bin's group has none.
    """
    names = list(names)
    if NOISE_PREFIX in names:
        return names.index(NOISE_PREFIX), None
    per = [(i, n[len(NOISE_PREFIX) + 1 :]) for i, n in enumerate(names) if n.startswith(NOISE_PREFIX + "_")]
    if not per:
        return None
    idx = [i for i, _ in per]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        raise ValueError(f"per-group noise parameters {[names[i] for i in idx]} at {idx} are not contiguous in theta")
    return idx[0], build_group_design({"groups": [g for _, g in per], "group": group_per_bin})


def noise_scale_per_bin(theta: ArrayLike, names: Iterable[str], group_per_bin: ArrayLike) -> np.ndarray:
    """Return the likelihood's per-bin sigma multiplier for parameter vectors, in numpy, for the posterior predictive.

    Parameters
    ----------
    theta : array-like [..., n_dim]
        The parameter vectors.
    names : sequence of str
        The parameter names.
    group_per_bin : array-like [n_bin]
        Each bin's group.

    Returns
    -------
    numpy.ndarray [..., n_bin]
        The multiplier.
    """
    theta = np.asarray(theta, dtype=float)
    out = np.ones((*theta.shape[:-1], np.asarray(group_per_bin).size))
    m = noise_map(names, group_per_bin)
    if m is None:
        return out
    lo, N = m
    if N is None:
        return out * theta[..., lo, None]
    return theta[..., lo + N.argmax(axis=1)]  # each bin's own group entry
