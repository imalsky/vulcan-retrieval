#!/usr/bin/env python3
"""Forward checks at posterior draws: one model change at a time, the same
draws, and the binned-depth shift against the data's sigma.

    python validation/posterior_forward_check.py run <out_dir> --variant base
    python validation/posterior_forward_check.py run <out_dir> --variant dayside
    python validation/posterior_forward_check.py compare <base.npz> <variant.npz>

``run`` draws ``--draws`` weighted posterior rows from ``<out_dir>``'s
nautilus_posterior.npz (seeded, so every variant sees the same rows), solves
them cold on the run's own pipeline rebuilt from its config.json with the
variant's change, and writes ``forward_check_<variant>.npz`` under
``output/w39b_smc_retrieval/validation/`` (theta, binned model depth, certified
flag, the variant's settings). ``compare`` reads two of those and reports, over
the draws both certified, the worst per-bin |shift| / sigma, its median over
draws, and the band-mean shifts in the SO2 (3.95-4.15 um) and CO2 (4.2-4.45 um)
bands in ppm and in sigma, as a REPORT artifact (``forward_check_<variant>.json``
in the same directory).

Variants (one per process; vulcan3 freezes the network at import):
  base         the run's own model
  network2025  thermo/SNCHO_photo_network_2025.txt (VULCAN3_NETWORK, set before import)
  dayside      dayside-average photochemistry geometry: sl_angle acos(0.5), f_diurnal 0.5
  twolimb      two Guillot limbs at Tirr -/+ TWO_LIMB_DTIRR_K (about 180 K apart at
               the skin, Espinoza+2024), depths averaged
  mass         the planet mass at +-1 sigma in the RT (lnM = +-MASS_SIGMA at frozen
               chemistry; the column keeps M0): two spectra, the file holds the + and
               the - as model and model_minus
  spot         the transit-light-source factor of a spotted star, blackbody ratio at
               (SPOT_T_PHOT_K, SPOT_T_PHOT_K - SPOT_DT_K, SPOT_F); no chemistry change

A one-sided rate-range clamp is not a variant here: vulcan3 has no switch for it
(the recorded +1.5% SO2 came from a study patch).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifact

NETWORK_2025 = "thermo/SNCHO_photo_network_2025.txt"
TWO_LIMB_DTIRR_K = 150.0      # skin T ~ 0.59 Tirr: +-150 K in Tirr is ~180 K between the limbs
MASS_SIGMA = 0.11             # the literature mass error in ln M
SPOT_T_PHOT_K, SPOT_DT_K, SPOT_F = 5400.0, 400.0, 0.03   # a quiet G8 star
SO2_BAND_UM, CO2_BAND_UM = (3.95, 4.15), (4.20, 4.45)
H_PLANCK, C_LIGHT, K_B = 6.62607015e-27, 2.99792458e10, 1.380649e-16
VARIANTS = ("base", "network2025", "dayside", "twolimb", "mass", "spot")


def _planck_ratio(wl_um, t_a, t_b):
    """B(lambda, t_a) / B(lambda, t_b)."""
    x = H_PLANCK * C_LIGHT / (wl_um * 1e-4 * K_B)
    return np.expm1(x / t_b) / np.expm1(x / t_a)


def spot_factor(wl_um, t_phot=SPOT_T_PHOT_K, dt=SPOT_DT_K, f=SPOT_F):
    """Transit-light-source factor (Rackham et al. 2018): the observed depth is
    the true depth times 1 / (1 - f (1 - B_spot / B_phot))."""
    return 1.0 / (1.0 - f * (1.0 - _planck_ratio(wl_um, t_phot - dt, t_phot)))


def _draws(out: Path, n: int, seed: int):
    post = np.load(out / "nautilus_posterior.npz")
    w = np.exp(post["log_w"] - np.max(post["log_w"]))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), size=int(n), p=w / w.sum())
    return np.asarray(post["theta"], np.float64)[idx], [str(s) for s in post["param_names"]]


def _solve(pipe, theta):
    """(column, binned model depth, certified) of the rows ``theta``: the cold
    two-stage solve, certified as production certifies."""
    import jax
    import jax.numpy as jnp
    th = jnp.asarray(theta, pipe.dtype)
    C = th[:, :pipe.n_chem_tp]
    Y, cd = jax.jit(pipe.fwd.chem_solve_cold_diag_batch)(C)
    mu = np.asarray(pipe.observed_depth_from_y_jit(Y, th), np.float64)
    ok = (np.asarray(cd.conv_normal) > 0.5) & np.all(np.isfinite(mu), axis=1)
    return Y, mu, ok


def run(args) -> int:
    out = Path(args.out_dir).resolve()
    # a presolve-chain run (v4) certified at a second program this tree no longer
    # builds; its config.json would load without it (unknown keys are dropped)
    if json.loads((out / "config.json").read_text()).get("chem_presolve"):
        raise RuntimeError(f"{out} ran the presolve chain, removed after vulcan-retrieval "
                           "0a2dbf3: check its draws at that commit")
    from retrieval_framework import config_schema as C
    from retrieval_framework.nautilus_report import _config_from_json

    cfg = _config_from_json(out / "config.json")
    if args.variant == "network2025":
        cfg = replace(cfg, network=NETWORK_2025)
    C.export_network(cfg)                   # before any vulcan3 import
    from vulcan_forward import vulcan_chem  # noqa: F401  (import order)
    from retrieval_framework import pipeline as P

    settings = {"variant": args.variant, "run": str(out), "draws": int(args.draws),
                "seed": int(args.seed)}
    if args.variant == "dayside":
        cfg = replace(cfg, cfg_overrides={**cfg.cfg_overrides,
                                          "sl_angle": math.acos(0.5), "f_diurnal": 0.5})
        settings.update(sl_angle_deg=60.0, f_diurnal=0.5)
    elif args.variant == "network2025":
        settings.update(network=NETWORK_2025)
    theta, names = _draws(out, args.draws, args.seed)
    logging.info(f"solver: {C.choose_solver()}")
    pipe = P.build_pipeline(cfg)
    if list(pipe.names) != names:
        raise RuntimeError(f"the rebuilt pipeline's parameters {pipe.names} are not the "
                           f"posterior's {names}")
    t0 = time.perf_counter()
    payload = dict(theta=theta, names=np.asarray(names), wl=np.asarray(pipe.obs["wl"]),
                   sigma=np.asarray(pipe.obs["sigma"]))
    if args.variant == "twolimb":
        i = names.index("Tirr")
        th_m, th_e = theta.copy(), theta.copy()
        th_m[:, i] -= TWO_LIMB_DTIRR_K
        th_e[:, i] += TWO_LIMB_DTIRR_K
        _, mu_m, ok_m = _solve(pipe, th_m)
        _, mu_e, ok_e = _solve(pipe, th_e)
        payload.update(model=0.5 * (mu_m + mu_e), certified=ok_m & ok_e,
                       model_morning=mu_m, model_evening=mu_e)
        settings.update(dTirr_K=TWO_LIMB_DTIRR_K)
    elif args.variant == "mass":
        import jax.numpy as jnp
        Y, mu, ok = _solve(pipe, theta)
        th = jnp.asarray(theta, pipe.dtype)

        def at(lnM):
            rows = []
            for k in range(theta.shape[0]):
                aux = pipe.fwd.aux_from_y(Y[k], th[k, :pipe.n_chem_tp])
                cloud = th[k, pipe.cloud_idx[0]:pipe.cloud_idx[0] + pipe.n_cloud]
                base_lnM = th[k, pipe.lnM_idx] if pipe.lnM_idx is not None else 0.0
                d = pipe.fwd.rt_depth(aux, th[k, pipe.lnR0_idx], cloud, base_lnM + lnM)
                rows.append(np.asarray(pipe.B, np.float64) @ np.asarray(d, np.float64))
            return np.asarray(rows)
        # offsets are the same on both sides, so the shift is the binned RT shift
        payload.update(model=mu + (at(+MASS_SIGMA) - at(0.0)), certified=ok,
                       model_minus=mu + (at(-MASS_SIGMA) - at(0.0)))
        settings.update(lnM_sigma=MASS_SIGMA)
    elif args.variant == "spot":
        _, mu, ok = _solve(pipe, theta)
        payload.update(model=mu * spot_factor(payload["wl"])[None, :], certified=ok,
                       model_unspotted=mu)
        settings.update(t_phot_K=SPOT_T_PHOT_K, dt_K=SPOT_DT_K, f_spot=SPOT_F)
    else:
        _, mu, ok = _solve(pipe, theta)
        payload.update(model=mu, certified=ok)
    logging.info(f"{args.variant}: {int(payload['certified'].sum())}/{len(theta)} draws "
                 f"certified in {time.perf_counter() - t0:.0f} s")
    _artifact.RESULTS.mkdir(parents=True, exist_ok=True)
    path = _artifact.RESULTS / f"forward_check_{args.variant}.npz"
    np.savez(path, settings=np.asarray(str(settings)), **payload)
    print(f"wrote {path}")
    return 0


def _band(wl, lo, hi):
    return (wl >= lo) & (wl <= hi)


def compare(args) -> int:
    a, b = np.load(args.base), np.load(args.variant)
    if not np.array_equal(a["theta"], b["theta"]):
        raise RuntimeError("the two files hold different draws; run both with the same "
                           "--draws and --seed on the same run")
    wl, sigma = np.asarray(a["wl"]), np.asarray(a["sigma"])
    ok = np.asarray(a["certified"]) & np.asarray(b["certified"])
    if not ok.any():
        raise RuntimeError("no draw certified in both files")
    d = (np.asarray(b["model"]) - np.asarray(a["model"]))[ok]        # (n, n_bin)
    worst = np.abs(d) / sigma[None, :]
    rows = []
    for name, (lo, hi) in (("SO2", SO2_BAND_UM), ("CO2", CO2_BAND_UM)):
        m = _band(wl, lo, hi)
        mean_ppm = 1e6 * d[:, m].mean(axis=1)
        band_sig = np.sqrt(np.sum(sigma[m] ** 2)) / m.sum()        # sigma of the band mean
        rows.append((name, float(np.median(mean_ppm)), float(np.max(np.abs(mean_ppm))),
                     float(np.median(np.abs(mean_ppm)) * 1e-6 / band_sig)))
    label = str(b["settings"])
    print(f"variant: {label}")
    print(f"draws compared: {int(ok.sum())} of {len(ok)}")
    print(f"worst per-bin |shift|/sigma: max {worst.max():.3f}, median over draws "
          f"{np.median(worst.max(axis=1)):.3f}")
    for name, med, mx, sig in rows:
        print(f"{name} band mean shift: median {med:+.1f} ppm, max |{mx:.1f}| ppm, "
              f"median {sig:.2f} sigma of the band mean")
    measurements = [
        {"name": "worst per-bin |shift| / sigma over the compared draws",
         "value": f"{worst.max():.3f}", "value_raw": float(worst.max()), "unit": "sigma",
         "gate": "report", "decisive": False, "passed": True},
        {"name": "median over draws of the worst per-bin |shift| / sigma",
         "value": f"{np.median(worst.max(axis=1)):.3f}",
         "value_raw": float(np.median(worst.max(axis=1))), "unit": "sigma",
         "gate": "report", "decisive": False, "passed": True},
    ] + [
        {"name": f"{name} band mean shift (median over draws)", "value": f"{med:+.1f} ppm",
         "value_raw": med, "unit": "ppm", "gate": "report", "decisive": False, "passed": True,
         "band_sigma_units": sig, "max_abs_ppm": mx}
        for name, med, mx, sig in rows]
    variant = Path(args.variant).stem.replace("forward_check_", "")
    _artifact.emit(f"forward_check_{variant}", f"Forward check at posterior draws: {variant}",
                   measurements, "REPORT",
                   f"{int(ok.sum())} draws; worst per-bin shift {worst.max():.2f} sigma; "
                   f"SO2 band {rows[0][1]:+.1f} ppm, CO2 band {rows[1][1]:+.1f} ppm (medians)",
                   resolved_config={"base": str(a["settings"]), "variant": label})
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out_dir", help="a finished run_nautilus output directory")
    r.add_argument("--variant", choices=VARIANTS, required=True)
    r.add_argument("--draws", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("variant")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
