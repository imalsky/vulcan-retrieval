"""nautilus_report.py -- certificate and figures for a run_nautilus output directory.

    python -m retrieval_framework.nautilus_report <out_dir> [--spectra N]

Reads what a finished run_nautilus run wrote (config.json, observations.npz,
target_manifest.json, target_digest.txt, nautilus_summary.json,
nautilus_tally.json, nautilus_posterior.npz) and writes

    nautilus_certificate.json   provenance, sampler settings, logZ_box and n_eff,
                                the rejection tally as fractions, the start mode,
                                the chemistry settings and plain-English caveats
    plots/corner.png            corner plot of the weighted posterior
    plots/tp_posterior.png      Guillot T-P 5-95% band from weighted draws
    plots/spectrum_fit.png      data, model and posterior-predictive 5-95% bands
                                and the median model (with --spectra N > 0)

There is no pass/fail verdict: the caveats say what limits the numbers.
run_nautilus writes the certificate itself when a run finishes.

Model spectra (``--spectra N``, default 32; 0 skips them and the spectrum
figure): N weighted posterior draws are re-solved COLD on the run's own
pipeline, rebuilt from config.json, so its data paths must resolve on this
machine. The cost is the pipeline build and compile plus one batched cold
two-stage chemistry solve of N columns on the lane queue
(``fwd.chem_solve_cold_diag_batch``, at the run's first-pass rtol and
count_max) and N RTs (``observed_depth_from_y_jit``): about one likelihood
batch of the run at width N. A warm or chem_retry run certified its columns
from other starts or at chem_retry_rtol, so these spectra agree with the run's
at the convergence scale. A draw whose cold solve does not certify within
count_max is left out of the bands and counted in the log. The draws and
spectra are saved to nautilus_spectra.npz.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
from retrieval_framework import provenance

log = logging.getLogger("retrieval")

CERT_FILE = "nautilus_certificate.json"
DPI = 300
_TO_PPM = 1.0 / OBS.PPM
_BAR_TO_CGS = 1.0e6          # dyn cm^-2 per bar
# T-P band pressure grid: log10 of its top (bar), down to ART_PBTM_BAR; and the
# number of weighted draws the band is taken over.
TP_PLOT_LOG10_P_TOP = -8
TP_PLOT_N_P = 120
TP_PLOT_DRAWS = 300

# Figure style: serif, ticks in on all four sides, no grid, frameless legend,
# square panels; Okabe-Ito blue for the model, charcoal for data.
STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Palatino", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": "large",
    "axes.linewidth": 1.5, "axes.xmargin": 0, "legend.frameon": False,
    "legend.fontsize": 10, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "xtick.labelsize": "large", "ytick.labelsize": "large",
    "savefig.bbox": "tight",
}
MODEL = "#0072B2"
INK = "#2b2b2b"
TRUTH = "#CC3311"
MARKERS = "os^Dv"       # one data-point shape per instrument group


def _pastel(c: str, keep: float) -> tuple:
    """``c`` blended toward white (keep = 1 is the colour itself)."""
    from matplotlib.colors import to_rgb
    return tuple(1 - keep + keep * v for v in to_rgb(c))


def _read(out: Path, name: str):
    return json.loads((out / name).read_text())


def _weights(log_w) -> np.ndarray:
    w = np.exp(np.asarray(log_w, np.float64) - np.max(log_w))
    return w / w.sum()


def certificate(out_dir) -> dict:
    """The certificate of the finished run in ``out_dir``."""
    out = Path(out_dir).resolve()
    cfg = _read(out, "config.json")
    summ = _read(out, "nautilus_summary.json")
    tally = _read(out, "nautilus_tally.json")
    manifest = _read(out, provenance.MANIFEST_FILE)
    digest = (out / "target_digest.txt").read_text().strip()
    manifest_digest = provenance.archived_manifest_digest(out)
    run_code = manifest.get("code") or {}
    code_now = provenance._repo_states()

    def frac(key, denom):
        return tally.get(key, 0) / denom if denom else None

    n = tally.get("n_eval", 0)
    in_window = n - tally.get("tp_out", 0)
    rejected = sum(tally.get(k, 0) for k in ("exhausted", "stalled", "nonfinite"))
    warm = summ["starts"] != "cold"
    retry = summ.get("chem_retry")
    rejections = {
        "n_eval": n,
        "tp_out_of_n_eval": frac("tp_out", n),
        "rejected_of_in_window": rejected / in_window if in_window else None,
        "exhausted_of_in_window": frac("exhausted", in_window),
        "stalled_of_in_window": frac("stalled", in_window),
        "nonfinite_of_in_window": frac("nonfinite", in_window),
        "cold_certified_of_cold": frac("cold_certified", tally.get("n_cold")),
        "warm_certified_of_warm": frac("warm_certified", tally.get("n_warm")),
    }
    if retry is not None:
        rejections.update(retried_of_in_window=frac("retried", in_window),
                          retry_certified_of_retried=frac("retry_certified",
                                                          tally.get("retried")))

    caveats = [
        f"{rejections['rejected_of_in_window'] or 0.0:.1%} of the draws with a T-P "
        "profile in the window were rejected (chemistry not certified within its cap, "
        "or a non-finite forward). Rejected draws have zero likelihood, so logZ_box is "
        "the box-prior evidence with those regions removed and depends on count_max "
        "and the rtol.",
        "nautilus reports no logZ error: quote the spread over independent seeds.",
    ]
    if warm:
        caveats.append(
            "Warm starts: each solve continued from the nearest certified column, so the "
            "likelihood and which draws were rejected depend on the evaluation order. The "
            "posterior and logZ_box are approximate; an evidence claim needs a cold run "
            "(NAUTILUS_WARM=0).")
    if int(cfg.get("cold_lanes") or 0) > 0:
        caveats.append(
            "Lane queue: a draw that starts in a freed lane starts at that tick, so its "
            "column and likelihood move with its batch at the convergence scale.")
    if retry is not None and not retry.get("presolve"):
        caveats.append(
            f"chem_retry: draws not certified at rtol {retry['rtol_first']} were re-solved "
            f"at rtol {retry['rtol']}, so the kept columns carry either tolerance.")
    if n > int(summ["n_like"]):
        caveats.append(
            f"The tally counts {n - int(summ['n_like'])} more evaluations than nautilus "
            "(n_like): a batch redrawn after a resume is counted twice, so the fractions "
            "are approximate.")
    if manifest_digest != digest.split("|")[0]:
        caveats.append(
            "target_manifest.json does not hash to this run's target digest: the identity "
            "files in this directory belong to a different run.")
    dirty = sorted(r for r, s in run_code.items() if (s or {}).get("dirty"))
    if dirty:
        caveats.append(f"The run's {', '.join(dirty)} tree(s) had uncommitted changes.")
    moved = sorted(r for r, s in run_code.items()
                   if (s or {}).get("commit") != (code_now.get(r) or {}).get("commit"))
    if moved:
        caveats.append(f"code_now differs from the run's code for {', '.join(moved)}: "
                       "this certificate was written at another commit.")

    return {
        "certificate_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": str(out),
        "provenance": {
            "target_digest": digest,
            "target_manifest_digest": manifest_digest,
            "run_code": run_code,
            "run_versions": manifest.get("versions"),
            "vulcan3_solver": manifest.get("vulcan3_solver"),
            "science_data": {k: manifest.get(k) for k in ("opacity_sha256", "cia_sha256")},
            "data": provenance._data_identity(out, cfg),
            "code_now": code_now,
            "versions_now": provenance._versions(),
        },
        "sampler": {k: summ.get(k) for k in (
            "nautilus", "n_live", "n_update", "n_batch", "f_live", "enlarge_per_dim",
            "networks", "discard_exploration", "seed", "preset")},
        "result": {"logZ_box": summ["logZ_box"], "logZ_err": None,
                   "n_eff": summ["n_eff"], "n_like": summ["n_like"]},
        "starts": {"mode": "warm" if warm else "cold",
                   "warm_of_n_eval": frac("n_warm", n), "cold_of_n_eval": frac("n_cold", n)},
        "rejections": rejections,
        "chemistry": {"rtol": summ.get("chem_rtol"), "chem_retry": retry,
                      **{k: cfg.get(k) for k in ("count_max", "yconv_cri", "dt_max", "nz",
                                                 "cold_lanes", "cfg_overrides")}},
        "caveats": caveats,
    }


def write_certificate(out_dir) -> Path:
    """Write ``nautilus_certificate.json`` into ``out_dir``; returns its path."""
    path = Path(out_dir) / CERT_FILE
    path.write_text(json.dumps(certificate(out_dir), indent=2, default=str) + "\n")
    return path


def predictive_replicates(mu_draws, obs_sigma, infl, rng):
    """Replicated observations under the run's OWN likelihood.

    The likelihood is diagonal Gaussian with sigma_eff = obs_sigma *
    noise_inflation, and mu already carries the instrument offsets, so a
    posterior-predictive replicate is mu + N(0, sigma_eff^2). ``infl`` is
    (n_draw, 1) for one b or (n_draw, n_bin) per bin
    (observations.noise_scale_per_bin, either noise mode). Returns
    (y_rep, sigma_eff). The spread of ``mu_draws`` alone is the LATENT band and
    is not comparable to data -- it ignores the inferred noise entirely.
    """
    sig_eff = np.asarray(obs_sigma)[None, :] * np.asarray(infl)
    return mu_draws + rng.normal(size=mu_draws.shape) * sig_eff, sig_eff


def guillot_T(p_bar, gravity, kappa, gamma, Tint, Tirr, f):
    """Guillot (2010) Eq. 29 -- numpy twin of exojax.atm.atmprof.atmprof_Guillot."""
    tau = np.asarray(p_bar) * _BAR_TO_CGS * kappa / gravity
    invsq3 = 1.0 / np.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (1.0 / gamma + (gamma - 1.0 / gamma) * np.exp(-gamma * tau / invsq3))
    return (0.75 * Tint ** 4 * (2.0 / 3.0 + tau) + 0.75 * Tirr ** 4 * f * fac) ** 0.25


def _config_from_json(path: Path) -> C.Config:
    """The run's Config from its config.json; keys that are not Config fields
    (the parameter table, the hardware record) are dropped."""
    d = json.loads(Path(path).read_text())
    names = {f.name for f in fields(C.Config)}
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names}
    for k in ("out_dir", "obs_dir"):
        if kw.get(k) is not None:
            kw[k] = Path(kw[k])
    return C.Config(**kw)


def model_spectra(out_dir, n: int):
    """(theta (n, n_dim), binned model depth (n, n_bin), certified (n,)) for n
    weighted posterior draws, re-solved cold (module docstring)."""
    import jax
    import jax.numpy as jnp

    from retrieval_framework import pipeline as P
    out = Path(out_dir).resolve()
    cfg = _config_from_json(out / "config.json")
    post = np.load(out / "nautilus_posterior.npz")
    rng = np.random.default_rng(int(cfg.seed) + 1)
    theta = np.asarray(post["theta"], np.float64)
    theta = theta[rng.choice(len(theta), size=int(n), p=_weights(post["log_w"]))]
    log.info(f"solver: {C.choose_solver()}")     # before vulcan3 is imported
    pipe = P.build_pipeline(cfg)
    if list(pipe.names) != [str(s) for s in post["param_names"]]:
        raise ValueError(f"the rebuilt pipeline's parameters {pipe.names} are not the "
                         f"posterior's {list(post['param_names'])}")
    th = jnp.asarray(theta, pipe.dtype)
    t0 = time.perf_counter()
    Y, cd = jax.jit(pipe.fwd.chem_solve_cold_diag_batch)(th[:, :pipe.n_chem_tp])
    mu = np.asarray(pipe.observed_depth_from_y_jit(Y, th), np.float64)
    # the runs' certificate: conv_normal alone (a solve can certify at cap + 1)
    ok = (np.asarray(cd.conv_normal) > 0.5) & np.all(np.isfinite(mu), axis=1)
    log.info(f"model spectra: {int(ok.sum())}/{len(ok)} draws certified cold in "
             f"{time.perf_counter() - t0:.0f} s")
    P.save_npz(out / "nautilus_spectra.npz", theta=theta, model=mu, certified=ok)
    return theta, mu, ok


def plot(out_dir, spectra=None) -> list:
    """corner.png and tp_posterior.png in ``out_dir/plots`` from the weighted
    posterior; with ``spectra`` (model_spectra's tuple) also spectrum_fit.png.
    Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import corner as corner_mod
    import matplotlib.pyplot as plt
    from vulcan_forward.constants import ART_PBTM_BAR

    out = Path(out_dir).resolve()
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    cfgj = _read(out, "config.json")
    post = np.load(out / "nautilus_posterior.npz")
    names = [str(s) for s in post["param_names"]]
    if names != list(cfgj["inferred_param_names"]):
        raise ValueError(f"nautilus_posterior.npz parameters {names} are not config.json's "
                         f"{cfgj['inferred_param_names']}")
    theta = np.asarray(post["theta"], np.float64)
    w = _weights(post["log_w"])
    truth = np.asarray(cfgj["inferred_param_truth"], np.float64)
    synthetic = bool(cfgj.get("generate_synthetic_data", False))
    show_truth = synthetic and np.isfinite(truth).any()
    written = []
    plt.rcParams.update(STYLE)

    # ---------------- corner ----------------
    fig = corner_mod.corner(
        # the points carry weights: an unweighted scatter of them would show
        # the low-weight shells as if they were posterior mass
        theta, weights=w, plot_datapoints=False, color=INK,
        labels=list(cfgj["inferred_param_labels"]), quantiles=[0.16, 0.5, 0.84],
        truths=(truth if show_truth else None), truth_color=TRUTH)
    fig.savefig(plots / "corner.png", dpi=DPI)
    plt.close(fig)
    written.append(plots / "corner.png")

    # ---------------- T-P posterior ----------------
    rng = np.random.default_rng(3)
    sel = theta[rng.choice(len(theta), size=TP_PLOT_DRAWS, p=w)]
    g_cgs, Tint = float(cfgj["tp_gravity_cgs"]), float(cfgj["tp_Tint_K"])
    gam_fix = float(cfgj["tp_gamma_fixed"])
    iT, ik = names.index("Tirr"), names.index("log10kappa")
    ig = names.index("log10gamma") if "log10gamma" in names else None
    iti = names.index("Tint") if "Tint" in names else None
    p_bar = np.logspace(TP_PLOT_LOG10_P_TOP, np.log10(ART_PBTM_BAR), TP_PLOT_N_P)

    def T_of(t):
        return guillot_T(p_bar, g_cgs, 10.0 ** t[ik],
                         (10.0 ** t[ig] if ig is not None else gam_fix),
                         (t[iti] if iti is not None else Tint), t[iT], C.GUILLOT_F)

    qs = np.nanpercentile(np.stack([T_of(t) for t in sel]), [5, 50, 95], axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.fill_betweenx(p_bar, qs[0], qs[2], color=_pastel(MODEL, 0.3), lw=0,
                     label="5-95% interval")
    ax.plot(qs[1], p_bar, color=MODEL, lw=1.5, label="median")
    if show_truth:
        ax.plot(T_of(truth), p_bar, color=TRUTH, ls="--", lw=1.2, label="truth")
    ax.set_yscale("log")
    ax.set_ylim(p_bar.max(), p_bar.min())
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("pressure (bar)")
    ax.legend(loc="best")
    fig.savefig(plots / "tp_posterior.png", dpi=DPI)
    plt.close(fig)
    written.append(plots / "tp_posterior.png")

    # ---------------- spectrum fit ----------------
    if spectra is not None:
        th_s, mu, ok = spectra
        if not np.any(ok):
            raise RuntimeError("no model-spectrum draw certified cold: nothing to plot")
        log.info(f"spectrum fit: {int(ok.sum())}/{len(ok)} draws certified cold")
        obs = np.load(out / "observations.npz")
        wl = np.asarray(obs["wl"], np.float64)
        sigma = np.asarray(obs["sigma"], np.float64)
        group = np.asarray(obs["group"]).astype(str)
        y_rep, _ = predictive_replicates(mu[ok], sigma,
                                         OBS.noise_scale_per_bin(th_s[ok], names, group),
                                         np.random.default_rng(4))
        o = np.argsort(wl)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        pred = _TO_PPM * np.percentile(y_rep, [5, 95], axis=0)[:, o]
        band = _TO_PPM * np.percentile(mu[ok], [5, 95], axis=0)[:, o]
        ax.fill_between(wl[o], pred[0], pred[1], color=_pastel(MODEL, 0.15), lw=0,
                        label="posterior predictive 5-95%", zorder=1)
        ax.fill_between(wl[o], band[0], band[1], color=_pastel(MODEL, 0.3), lw=0,
                        label="model 5-95%", zorder=2)
        ax.plot(wl[o], _TO_PPM * np.median(mu[ok], axis=0)[o], color=MODEL, lw=1.5,
                label="median model", zorder=3)
        for g, mk in zip(dict.fromkeys(group.tolist()), MARKERS * len(group)):
            m = group == g
            ax.errorbar(wl[m], _TO_PPM * np.asarray(obs["depth"], np.float64)[m],
                        yerr=_TO_PPM * sigma[m], fmt=mk, ms=3.5, elinewidth=0.8, capsize=0,
                        color=INK, label=g, zorder=4)
        if synthetic and "flux_true" in obs.files:
            ax.plot(wl[o], _TO_PPM * np.asarray(obs["flux_true"], np.float64)[o],
                    color=TRUTH, ls="--", lw=1.2, label="injected truth", zorder=5)
        ax.set_xlabel("wavelength ($\\mu$m)")
        ax.set_ylabel("transit depth (ppm)")
        ax.legend(loc="best")
        fig.savefig(plots / "spectrum_fit.png", dpi=DPI)
        plt.close(fig)
        written.append(plots / "spectrum_fit.png")
    return written


def main(argv=None) -> None:
    from retrieval_framework.run_setup import setup_logging
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="a run_nautilus output directory")
    ap.add_argument("--spectra", type=int, default=32,
                    help="posterior draws to re-solve for the spectrum figure "
                         "(0: no forward model, no spectrum figure)")
    args = ap.parse_args(argv)
    setup_logging()
    out = Path(args.out_dir).resolve()
    # the forward build comes first: the certificate's version record imports
    # exojax, which must not precede vulcan_chem
    spectra = model_spectra(out, args.spectra) if args.spectra > 0 else None
    log.info(f"wrote {write_certificate(out)}")
    for p in plot(out, spectra):
        log.info(f"wrote {p}")


if __name__ == "__main__":
    main()
