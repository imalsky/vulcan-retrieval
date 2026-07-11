#!/usr/bin/env python3
"""plot_smc.py -- figures for one SMC retrieval run (PNG only, dpi=200).

Reads the .npz bundles run_smc.py wrote into an output dir and produces, in
<out_dir>/plots (or SMC_PLOTS_DIR):

    corner.png            posterior corner plot (truth lines for synthetic runs)
    spectrum_fit.png      real/synthetic data per instrument + PPC band + median model
    tp_posterior.png      retrieved Guillot T-P credible band (+ truth for synthetic)
    smc_diagnostics.png   beta ladder / ESS / acceptance + step size / unique + logZ

Self-contained: numpy + matplotlib + corner only -- no jax, no VULCAN, no exojax
(the Guillot curve is re-evaluated here in numpy from the same Guillot 2010 Eq. 29
that exojax.atm.atmprof.atmprof_Guillot implements).

Usage:  python -m retrieval_framework.plot_smc <out_dir>
        (relative paths resolve against the cwd; a run dir's outputs live in
        <run_dir>/data/<preset>)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 200

GROUP_COLORS = {"NIRISS": "#1f77b4", "G395H": "#d62728", "PRISM": "#2ca02c",
                "NIRCam": "#9467bd", "SYNTH": "#7f7f7f"}


def _out_dir() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    preset = os.environ.get("SMC_RETRIEVAL_PRESET", "smoke").strip().lower()
    return (Path(".") / "data" / preset).resolve()


def guillot_T(p_bar, gravity, kappa, gamma, Tint, Tirr, f):
    """Guillot (2010) Eq. 29 -- numpy twin of exojax.atm.atmprof.atmprof_Guillot."""
    tau = np.asarray(p_bar) * 1.0e6 * kappa / gravity
    invsq3 = 1.0 / np.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (1.0 / gamma + (gamma - 1.0 / gamma) * np.exp(-gamma * tau / invsq3))
    return (0.75 * Tint ** 4 * (2.0 / 3.0 + tau) + 0.75 * Tirr ** 4 * f * fac) ** 0.25


def main() -> None:
    out = _out_dir()
    plots = Path(os.environ.get("SMC_PLOTS_DIR", out / "plots"))
    plots.mkdir(parents=True, exist_ok=True)
    print(f"[plot] reading {out}")

    cfgj = json.loads((out / "config.json").read_text())
    label = str(cfgj.get("run_label") or "").strip() or "retrieval"
    names = list(cfgj["inferred_param_names"])
    labels = list(cfgj["inferred_param_labels"])
    truth = np.asarray(cfgj["inferred_param_truth"], float)
    synthetic = bool(cfgj.get("generate_synthetic_data", False))

    s = np.load(out / "posterior_samples.npz", allow_pickle=True)
    theta = np.asarray(s["samples"], float).reshape(-1, len(names))
    obs = np.load(out / "observations.npz", allow_pickle=True)
    # a governor-stopped run saves a TEMPERED cloud; stamp every headline figure so it
    # can never be mistaken for the posterior
    final_beta = float(s["final_beta"]) if "final_beta" in s.files else 1.0
    tempered_tag = ("" if final_beta >= 1.0 - 1e-6
                    else f"  [TEMPERED beta={final_beta:.3f} -- NOT the posterior]")

    # ---------------- corner ----------------
    try:
        import corner as corner_mod
        fig = corner_mod.corner(
            theta, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True,
            title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 10},
            truths=(truth if synthetic and np.isfinite(truth).any() else None),
            truth_color="#d62728")
        fig.suptitle(f"{label} VULCAN-JAX retrieval ({'synthetic' if synthetic else 'real data'})"
                     f"{tempered_tag}", y=1.02, fontsize=11)
        fig.savefig(plots / "corner.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("[plot] corner.png")
    except Exception as e:  # pragma: no cover
        print(f"[plot] corner failed: {e}")

    # ---------------- spectrum fit ----------------
    wl = np.asarray(obs["wl"], float)
    depth = np.asarray(obs["depth"], float)
    sigma = np.asarray(obs["sigma"], float)
    group = np.asarray(obs["group"]).astype(str)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ppc_path = out / "posterior_predictive.npz"
    if ppc_path.exists():
        ppc = np.load(ppc_path)
        o = np.argsort(np.asarray(ppc["wl"], float))
        wlm = np.asarray(ppc["wl"], float)[o]
        ax.fill_between(wlm, 1e6 * np.asarray(ppc["p05"], float)[o],
                        1e6 * np.asarray(ppc["p95"], float)[o],
                        color="#ff7f0e", alpha=0.30, lw=0, label="PPC 5-95%", zorder=2)
        ax.plot(wlm, 1e6 * np.asarray(ppc["mu_at_median"], float)[o],
                color="#ff7f0e", lw=1.4, label="model @ posterior median", zorder=3)
    for g in dict.fromkeys(group.tolist()):
        m = group == g
        ax.errorbar(wl[m], 1e6 * depth[m], yerr=1e6 * sigma[m], fmt="o", ms=2.6,
                    lw=0.8, capsize=0, alpha=0.85, color=GROUP_COLORS.get(g, "k"),
                    label=f"{g} ({int(m.sum())} bins)", zorder=4)
    if synthetic and "flux_true" in obs.files:
        oo = np.argsort(wl)
        ax.plot(wl[oo], 1e6 * np.asarray(obs["flux_true"], float)[oo], "k--", lw=0.9,
                label="injected truth", zorder=5)
    ax.set_xlabel("wavelength [$\\mu$m]"); ax.set_ylabel("transit depth [ppm]")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.set_title(f"{label} transmission: data vs posterior predictive{tempered_tag}")
    fig.tight_layout(); fig.savefig(plots / "spectrum_fit.png", dpi=DPI); plt.close(fig)
    print("[plot] spectrum_fit.png")

    # ---------------- T-P posterior ----------------
    if cfgj.get("tp_model", "guillot") == "guillot":
        g_cgs = float(cfgj["tp_gravity_cgs"]); f_g = float(cfgj["tp_f"]); Tint = float(cfgj["tp_Tint_K"])
        iT = names.index("Tirr"); ik = names.index("log10kappa")
        ig = names.index("log10gamma") if "log10gamma" in names else None
        gam_fix = float(cfgj.get("tp_gamma_fixed", 0.4))
        p_bar = np.logspace(-8, np.log10(7.0), 120)
        rng = np.random.default_rng(3)
        sel = theta[rng.choice(theta.shape[0], size=min(300, theta.shape[0]), replace=False)]
        curves = np.stack([
            guillot_T(p_bar, g_cgs, 10.0 ** t[ik], (10.0 ** t[ig] if ig is not None else gam_fix),
                      Tint, t[iT], f_g) for t in sel])
        qs = np.nanpercentile(curves, [5, 50, 95], axis=0)
        fig, ax = plt.subplots(figsize=(4.6, 5.4))
        ax.fill_betweenx(p_bar, qs[0], qs[2], alpha=0.30, color="#1f77b4", lw=0, label="5-95%")
        ax.plot(qs[1], p_bar, color="#1f77b4", lw=1.6, label="median")
        if synthetic and np.isfinite(truth).any():
            ax.plot(guillot_T(p_bar, g_cgs, 10.0 ** truth[ik],
                              (10.0 ** truth[ig] if ig is not None else gam_fix), Tint, truth[iT], f_g),
                    p_bar, "k--", lw=1.2, label="truth")
        ax.set_yscale("log"); ax.invert_yaxis()
        ax.set_xlabel("T [K]"); ax.set_ylabel("P [bar]")
        ax.legend(fontsize=8, frameon=False)
        # same tempered stamp as corner/spectrum: this band is NOT a posterior band
        # when the ladder stopped at beta<1 (this figure used to be the one unstamped
        # headline panel)
        ax.set_title("Retrieved Guillot T-P" + tempered_tag, fontsize=9)
        fig.tight_layout(); fig.savefig(plots / "tp_posterior.png", dpi=DPI); plt.close(fig)
        print("[plot] tp_posterior.png")

    # ---------------- SMC diagnostics ----------------
    xp = out / "smc_extra_fields.npz"
    if xp.exists():
        x = np.load(xp, allow_pickle=True)
        betas = np.asarray(x["smc_betas"], float)
        stages = np.arange(1, betas.size)
        fig, axs = plt.subplots(2, 2, figsize=(9.4, 6.4))
        a = axs[0, 0]; a.semilogy(np.arange(betas.size), np.maximum(betas, 1e-12), "o-", ms=3)
        a.set_xlabel("stage"); a.set_ylabel(r"$\beta$"); a.set_title("tempering ladder")
        a.axhline(1.0, color="k", lw=0.6, ls=":")
        a = axs[0, 1]; a.plot(stages, np.asarray(x["smc_ess"], float), "o-", ms=3)
        a.axhline(float(x["smc_num_particles"]) * float(cfgj["smc_target_ess_frac"]),
                  color="r", lw=0.7, ls="--", label="target ESS")
        a.set_xlabel("stage"); a.set_ylabel("ESS"); a.set_title("effective sample size"); a.legend(fontsize=8)
        a = axs[1, 0]
        a.plot(stages, np.asarray(x["smc_acceptance_rate"], float), "o-", ms=3, label="acceptance")
        a.axhline(float(cfgj["mcmc_target_accept_mala"]), color="r", lw=0.7, ls="--", label="target")
        a2 = a.twinx(); a2.semilogy(stages, np.asarray(x["smc_step_size_history"], float),
                                    "s-", ms=2.5, color="#2ca02c", alpha=0.7, label="step size")
        a.set_xlabel("stage"); a.set_ylabel("MALA acceptance"); a2.set_ylabel("step size")
        a.set_title("mutation kernel adaptation"); a.legend(fontsize=8, loc="upper left")
        a = axs[1, 1]
        a.plot(stages, np.asarray(x["smc_unique_particles"], float), "o-", ms=3, label="unique particles")
        a.set_xlabel("stage"); a.set_ylabel("unique particles")
        a2 = a.twinx()
        a2.plot(stages, np.cumsum(np.nan_to_num(np.asarray(x["smc_logZ_increment"], float))),
                "s-", ms=2.5, color="#9467bd", alpha=0.7)
        a2.set_ylabel("cumulative logZ")
        a.set_title("diversity + evidence"); a.legend(fontsize=8, loc="lower left")
        sup = ""
        if ("smc_log_support_fraction" in x.files
                and np.isfinite(float(x["smc_log_support_fraction"]))):
            sup = (f" | prior support f={np.exp(float(x['smc_log_support_fraction'])):.2f} "
                   f"(logZ conditioned; box {float(x['smc_logZ_box']):.1f})")
        fig.suptitle(f"SMC diagnostics (reached beta=1: {bool(int(x['reached_beta1']))})"
                     f"{sup}", fontsize=11)
        fig.tight_layout(); fig.savefig(plots / "smc_diagnostics.png", dpi=DPI); plt.close(fig)
        print("[plot] smc_diagnostics.png")

    print(f"[plot] all figures in {plots}")


if __name__ == "__main__":
    main()
