"""The three figures of a finished run: the weighted corner plot, the T-P band and the spectrum fit."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from retrieval_framework import constants, observations as OBS
from retrieval_framework.nautilus_report import read_json, tp_curve, weights

if TYPE_CHECKING:
    from jax.typing import ArrayLike

__all__ = ["plot", "predictive_replicates"]

log = logging.getLogger("retrieval")

DPI = 300
_TO_PPM = 1.0 / constants.PPM
# T-P band: log10 of its top (bar), its points down to the run's art_pbtm_bar, and the weighted draws it is taken
# over.
TP_PLOT_LOG10_P_TOP = -8
TP_PLOT_N_P = 120
TP_PLOT_DRAWS = 300
# corner contour levels, in sigma; corner's own default stops at 2.
CORNER_SIGMAS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

# Okabe-Ito blue for the model, charcoal for the data.
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Palatino", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": "large",
    "axes.linewidth": 1.5,
    "axes.xmargin": 0,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "savefig.bbox": "tight",
}
MODEL = "#0072B2"
INK = "#2b2b2b"
TRUTH = "#CC3311"
MARKERS = "os^Dv"  # one data-point shape per instrument group


def _pastel(c: str, keep: float) -> tuple:
    """Return ``c`` blended toward white (keep = 1 is the colour itself)."""
    from matplotlib.colors import to_rgb

    return tuple(1 - keep + keep * v for v in to_rgb(c))


def predictive_replicates(
    mu_draws: np.ndarray, obs_sigma: ArrayLike, infl: ArrayLike, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return replicated observations under the run's own likelihood.

    The likelihood is diagonal Gaussian with sigma_eff = obs_sigma * noise_inflation and mu already carries the offsets,
    so a replicate is mu + N(0, sigma_eff^2). The spread of ``mu_draws`` alone is the latent band, not comparable to
    data.

    Parameters
    ----------
    mu_draws : numpy.ndarray [n_draw, n_bin]
        The model depths.
    obs_sigma : array-like [n_bin]
        The observed sigmas.
    infl : array-like [n_draw, 1] or [n_draw, n_bin]
        The sigma multiplier.
    rng : numpy.random.Generator
        The generator.

    Returns
    -------
    tuple of numpy.ndarray
        ``(y_rep, sigma_eff)``.
    """
    sig_eff = np.asarray(obs_sigma)[None, :] * np.asarray(infl)
    return mu_draws + rng.normal(size=mu_draws.shape) * sig_eff, sig_eff


def _corner_figure(path: Path, theta: np.ndarray, w: np.ndarray, labels: list[str], truth: np.ndarray | None) -> None:
    """Draw the weighted corner plot."""
    import corner as corner_mod
    import matplotlib.pyplot as plt

    fig = corner_mod.corner(
        # An unweighted scatter of weighted points would show the low-weight shells as posterior mass.
        theta,
        weights=w,
        plot_datapoints=False,
        color=INK,
        levels=[1.0 - np.exp(-0.5 * s * s) for s in CORNER_SIGMAS],
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        truths=truth,
        truth_color=TRUTH,
    )
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _tp_figure(
    path: Path, theta: np.ndarray, w: np.ndarray, names: list[str], cfgj: dict, truth: np.ndarray | None
) -> None:
    """Draw the T-P posterior: the median and 5-95% interval of weighted draws."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    sel = theta[rng.choice(len(theta), size=TP_PLOT_DRAWS, p=w)]
    p_bar = np.logspace(TP_PLOT_LOG10_P_TOP, np.log10(float(cfgj["art_pbtm_bar"])), TP_PLOT_N_P)

    def T_of(t: np.ndarray) -> np.ndarray:
        """Return T on the band's pressure grid for one row."""
        return tp_curve(t, names, cfgj, p_bar)

    qs = np.nanpercentile(np.stack([T_of(t) for t in sel]), [5, 50, 95], axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.fill_betweenx(p_bar, qs[0], qs[2], color=_pastel(MODEL, 0.3), lw=0, label="5-95% interval")
    ax.plot(qs[1], p_bar, color=MODEL, lw=1.5, label="median")
    if truth is not None:
        ax.plot(T_of(truth), p_bar, color=TRUTH, ls="--", lw=1.2, label="truth")
    ax.set_yscale("log")
    ax.set_ylim(p_bar.max(), p_bar.min())
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("pressure (bar)")
    ax.legend(loc="best")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _spectrum_figure(path: Path, out: Path, spectra: tuple[Any, ...], names: list[str], synthetic: bool) -> None:
    """Draw the spectrum fit: the model band, the posterior predictive and the data."""
    import matplotlib.pyplot as plt

    th_s, mu, ok, mult = spectra
    if not np.any(ok):
        raise RuntimeError("no model-spectrum draw certified cold: nothing to plot")
    log.info("spectrum fit: %s/%s draws certified cold", int(ok.sum()), len(ok))
    obs = np.load(out / "observations.npz")
    wl = np.asarray(obs["wl"], np.float64)
    sigma = np.asarray(obs["sigma"], np.float64)
    group = np.asarray(obs["group"]).astype(str)
    infl = OBS.noise_scale_per_bin(th_s[ok], names, group) if mult is None else mult[ok]
    y_rep, _ = predictive_replicates(mu[ok], sigma, infl, np.random.default_rng(4))
    o = np.argsort(wl)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    pred = _TO_PPM * np.percentile(y_rep, [5, 95], axis=0)[:, o]
    band = _TO_PPM * np.percentile(mu[ok], [5, 95], axis=0)[:, o]
    ax.fill_between(
        wl[o], pred[0], pred[1], color=_pastel(MODEL, 0.15), lw=0, label="posterior predictive 5-95%", zorder=1
    )
    ax.fill_between(wl[o], band[0], band[1], color=_pastel(MODEL, 0.3), lw=0, label="model 5-95%", zorder=2)
    ax.plot(wl[o], _TO_PPM * np.median(mu[ok], axis=0)[o], color=MODEL, lw=1.5, label="median model", zorder=3)
    for g, mk in zip(dict.fromkeys(group.tolist()), MARKERS * len(group), strict=False):
        m = group == g
        ax.errorbar(
            wl[m],
            _TO_PPM * np.asarray(obs["depth"], np.float64)[m],
            yerr=_TO_PPM * sigma[m],
            fmt=mk,
            ms=3.5,
            elinewidth=0.8,
            capsize=0,
            color=INK,
            label=g,
            zorder=4,
        )
    if synthetic and "flux_true" in obs.files:
        ax.plot(
            wl[o],
            _TO_PPM * np.asarray(obs["flux_true"], np.float64)[o],
            color=TRUTH,
            ls="--",
            lw=1.2,
            label="injected truth",
            zorder=5,
        )
    ax.set_xlabel("wavelength ($\\mu$m)")
    ax.set_ylabel("transit depth (ppm)")
    ax.legend(loc="best")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot(out_dir: str | Path, spectra: tuple[Any, ...] | None = None) -> list[Path]:
    """Draw corner.png and tp_posterior.png in ``out_dir/plots``, and spectrum_fit.png with ``spectra``.

    Refuses spectra none of which certified (RuntimeError).

    Parameters
    ----------
    out_dir : str or pathlib.Path
        The run's output directory.
    spectra : tuple, optional
        ``_model_spectra``'s result.

    Returns
    -------
    list of pathlib.Path
        The files written.

    Raises
    ------
    ValueError
        If the posterior's parameters are not config.json's.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir).resolve()
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    cfgj = read_json(out, "config.json")
    post = np.load(out / "nautilus_posterior.npz")
    names = [str(s) for s in post["param_names"]]
    if names != list(cfgj["inferred_param_names"]):
        raise ValueError(
            f"nautilus_posterior.npz parameters {names} are not config.json's {cfgj['inferred_param_names']}"
        )
    theta = np.asarray(post["theta"], np.float64)
    w = weights(post["log_w"])
    truth = np.asarray(cfgj["inferred_param_truth"], np.float64)
    synthetic = bool(cfgj["generate_synthetic_data"])
    shown_truth = truth if synthetic and np.isfinite(truth).any() else None
    plt.rcParams.update(STYLE)

    _corner_figure(plots / "corner.png", theta, w, list(cfgj["inferred_param_labels"]), shown_truth)
    _tp_figure(plots / "tp_posterior.png", theta, w, names, cfgj, shown_truth)
    written = [plots / "corner.png", plots / "tp_posterior.png"]
    if spectra is not None:
        _spectrum_figure(plots / "spectrum_fit.png", out, spectra, names, synthetic)
        written.append(plots / "spectrum_fit.png")
    return written
