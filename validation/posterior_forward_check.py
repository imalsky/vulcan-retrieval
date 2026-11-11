"""Forward checks at posterior draws: one model change at a time, the same draws, the shift against the data's sigma.

    python validation/posterior_forward_check.py run <out_dir> --variant base
    python validation/posterior_forward_check.py run <out_dir> --variant spot --spot-t-phot 5400 --spot-dt 400 \
        --spot-f 0.03
    python validation/posterior_forward_check.py compare <base.npz> <variant.npz>

``run`` solves seeded weighted posterior rows cold on the run's own pipeline with the variant's change. ``compare``
reports the worst per-bin |shift| / sigma and the band-mean shifts in the SO2 and CO2 bands. The mass variant moves lnM
by the run's ``prior_lnM_sigma``; the spotted star is the user's, named on the command line. The variants are in
docs/method.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifact

TWO_LIMB_DTIRR_K = 150.0  # skin T ~ 0.59 Tirr: +-150 K in Tirr is ~180 K between the limbs
# Dayside-average photochemistry: the stellar zenith angle acos(0.5), 60 deg, and the diurnal factor 0.5, the
# hemisphere-mean geometry of a dayside column.
DAYSIDE_SL_ANGLE_RAD, DAYSIDE_F_DIURNAL = math.acos(0.5), 0.5
SO2_BAND_UM, CO2_BAND_UM = (3.95, 4.15), (4.20, 4.45)
H_PLANCK, C_LIGHT, K_B = 6.62607015e-27, 2.99792458e10, 1.380649e-16
VARIANTS = ("base", "dayside", "twolimb", "mass", "spot")


def _planck_ratio(wl_um, t_a, t_b):
    """B(lambda, t_a) / B(lambda, t_b)."""
    x = H_PLANCK * C_LIGHT / (wl_um * 1e-4 * K_B)
    return np.expm1(x / t_b) / np.expm1(x / t_a)


def spot_factor(wl_um, t_phot, dt, f):
    """Transit-light-source factor (Rackham et al. 2018): the observed depth is
    the true depth times 1 / (1 - f (1 - B_spot / B_phot)).
    """
    return 1.0 / (1.0 - f * (1.0 - _planck_ratio(wl_um, t_phot - dt, t_phot)))


def _draws(out: Path, n: int, seed: int):
    from retrieval_framework.nautilus_report import _weights

    post = np.load(out / "nautilus_posterior.npz")
    w = _weights(post["log_w"])
    idx = np.random.default_rng(seed).choice(len(w), size=int(n), p=w)
    return np.asarray(post["theta"], np.float64)[idx], [str(s) for s in post["param_names"]]


def _twolimb(pipe, theta, names):
    """The two limbs' mean depth at Tirr +- TWO_LIMB_DTIRR_K, certified where both certify."""
    i = names.index("Tirr")
    th_m, th_e = theta.copy(), theta.copy()
    th_m[:, i] -= TWO_LIMB_DTIRR_K
    th_e[:, i] += TWO_LIMB_DTIRR_K
    _, mu_m, ok_m = pipe.solve_cold(th_m)
    _, mu_e, ok_e = pipe.solve_cold(th_e)
    return {"model": 0.5 * (mu_m + mu_e), "certified": ok_m & ok_e, "model_morning": mu_m, "model_evening": mu_e}


def _mass(pipe, theta, sigma):
    """The depth at lnM +- sigma on the base columns: the offsets match on both sides, so the shift is the RT's."""
    import jax.numpy as jnp

    Y, mu, ok = pipe.solve_cold(theta)
    th = jnp.asarray(theta, pipe.dtype)

    def at(lnM):
        rows = []
        for k in range(theta.shape[0]):
            aux = pipe.fwd.aux_from_y(Y[k], th[k, : pipe.n_chem_tp])
            lnR0, cloud, base_lnM = pipe.rt_args(th[k])
            d = pipe.fwd.rt_depth(aux, lnR0, cloud, base_lnM + lnM)
            rows.append(np.asarray(pipe.B, np.float64) @ np.asarray(d, np.float64))
        return np.asarray(rows)

    base = at(0.0)
    return {"model": mu + (at(+sigma) - base), "certified": ok, "model_minus": mu + (at(-sigma) - base)}


def run(args) -> int:
    out = Path(args.out_dir).resolve()
    from retrieval_framework import config_schema as C
    from retrieval_framework.nautilus_report import _config_from_json

    cfg = _config_from_json(out / "config.json")
    C.export_network(cfg)  # before any vulcan3 import
    from vulcan_forward import vulcan_chem  # noqa: F401  (import order)
    from retrieval_framework import pipeline as P

    settings = {"variant": args.variant, "run": str(out), "draws": int(args.draws), "seed": int(args.seed)}
    if args.variant == "dayside":
        over = {"sl_angle": DAYSIDE_SL_ANGLE_RAD, "f_diurnal": DAYSIDE_F_DIURNAL}
        cfg = replace(cfg, cfg_overrides={**cfg.cfg_overrides, **over})
        settings.update(over)
    theta, names = _draws(out, args.draws, args.seed)
    logging.info("solver: %s", C.choose_solver())
    pipe = P.build_pipeline(cfg)
    if list(pipe.names) != names:
        raise RuntimeError(f"the rebuilt pipeline's parameters {pipe.names} are not the posterior's {names}")
    t0 = time.perf_counter()
    payload = {
        "theta": theta,
        "names": np.asarray(names),
        "wl": np.asarray(pipe.obs["wl"]),
        "sigma": np.asarray(pipe.obs["sigma"]),
    }
    if args.variant == "twolimb":
        payload.update(_twolimb(pipe, theta, names))
        settings.update(dTirr_K=TWO_LIMB_DTIRR_K)
    elif args.variant == "mass":
        payload.update(_mass(pipe, theta, float(cfg.prior_lnM_sigma)))
        settings.update(lnM_sigma=float(cfg.prior_lnM_sigma))
    elif args.variant == "spot":
        _, mu, ok = pipe.solve_cold(theta)
        star = (args.spot_t_phot, args.spot_dt, args.spot_f)
        payload.update(model=mu * spot_factor(payload["wl"], *star)[None, :], certified=ok, model_unspotted=mu)
        settings.update(t_phot_K=star[0], dt_K=star[1], f_spot=star[2])
    else:
        _, mu, ok = pipe.solve_cold(theta)
        payload.update(model=mu, certified=ok)
    logging.info(
        "%s: %s/%s draws certified in %.0f s",
        args.variant,
        int(payload["certified"].sum()),
        len(theta),
        time.perf_counter() - t0,
    )
    print(f"wrote {_save_spectra(args.variant, cfg, settings, payload, out)}")
    return 0


def _save_spectra(variant, cfg, settings, payload, out):
    """Save one variant's solve beside its run, ``<run>/validation/``, with the state that produced it."""
    folder = Path(out) / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"forward_check_{variant}.npz"
    provenance = json.dumps(_artifact.collect_provenance(asdict(cfg)), default=str)
    np.savez(path, settings=np.asarray(str(settings)), provenance=np.asarray(provenance), **payload)
    return path


def _solve_provenance(z):
    """The provenance a solve saved; a file from before it was recorded says so."""
    return json.loads(str(z["provenance"])) if "provenance" in z.files else "not recorded"


def _band(wl, lo, hi):
    return (wl >= lo) & (wl <= hi)


def _shift_measurements(worst, rows):
    """The artifact's report-only rows: the worst per-bin shift and each band's mean shift."""
    return [
        {
            "name": "worst per-bin |shift| / sigma over the compared draws",
            "value": f"{worst.max():.3f}",
            "value_raw": float(worst.max()),
            "unit": "sigma",
            "gate": "report",
            "decisive": False,
            "passed": True,
        },
        {
            "name": "median over draws of the worst per-bin |shift| / sigma",
            "value": f"{np.median(worst.max(axis=1)):.3f}",
            "value_raw": float(np.median(worst.max(axis=1))),
            "unit": "sigma",
            "gate": "report",
            "decisive": False,
            "passed": True,
        },
    ] + [
        {
            "name": f"{name} band mean shift (median over draws)",
            "value": f"{med:+.1f} ppm",
            "value_raw": med,
            "unit": "ppm",
            "gate": "report",
            "decisive": False,
            "passed": True,
            "band_sigma_units": sig,
            "max_abs_ppm": mx,
        }
        for name, med, mx, sig in rows
    ]


def compare(args) -> int:
    a, b = np.load(args.base), np.load(args.variant)
    if not np.array_equal(a["theta"], b["theta"]):
        raise RuntimeError(
            "the two files hold different draws; run both with the same --draws and --seed on the same run"
        )
    wl, sigma = np.asarray(a["wl"]), np.asarray(a["sigma"])
    ok = np.asarray(a["certified"]) & np.asarray(b["certified"])
    if not ok.any():
        raise RuntimeError("no draw certified in both files")
    d = (np.asarray(b["model"]) - np.asarray(a["model"]))[ok]  # (n, n_bin)
    worst = np.abs(d) / sigma[None, :]
    rows = []
    for name, (lo, hi) in (("SO2", SO2_BAND_UM), ("CO2", CO2_BAND_UM)):
        m = _band(wl, lo, hi)
        mean_ppm = 1e6 * d[:, m].mean(axis=1)
        band_sig = np.sqrt(np.sum(sigma[m] ** 2)) / m.sum()  # sigma of the band mean
        rows.append(
            (
                name,
                float(np.median(mean_ppm)),
                float(np.max(np.abs(mean_ppm))),
                float(np.median(np.abs(mean_ppm)) * 1e-6 / band_sig),
            )
        )
    label = str(b["settings"])
    print(f"variant: {label}")
    print(f"draws compared: {int(ok.sum())} of {len(ok)}")
    print(f"worst per-bin |shift|/sigma: max {worst.max():.3f}, median over draws {np.median(worst.max(axis=1)):.3f}")
    for name, med, mx, sig in rows:
        print(
            f"{name} band mean shift: median {med:+.1f} ppm, max |{mx:.1f}| ppm, "
            f"median {sig:.2f} sigma of the band mean"
        )
    measurements = _shift_measurements(worst, rows)
    variant = Path(args.variant).stem.replace("forward_check_", "")
    _artifact.emit(
        f"forward_check_{variant}",
        f"Forward check at posterior draws: {variant}",
        measurements,
        "REPORT",
        f"{int(ok.sum())} draws; worst per-bin shift {worst.max():.2f} sigma; "
        f"SO2 band {rows[0][1]:+.1f} ppm, CO2 band {rows[1][1]:+.1f} ppm (medians)",
        resolved_config={
            "base": str(a["settings"]),
            "variant": label,
            "base_solve_provenance": _solve_provenance(a),
            "variant_solve_provenance": _solve_provenance(b),
        },
        out_dir=Path(args.variant).resolve().parent,  # beside the solves, in their run's validation/
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out_dir", help="a finished run_nautilus output directory")
    r.add_argument("--variant", choices=VARIANTS, required=True)
    r.add_argument("--draws", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--spot-t-phot", type=float, help="spot variant: the photosphere's temperature, K")
    r.add_argument("--spot-dt", type=float, help="spot variant: how much cooler the spots are, K")
    r.add_argument("--spot-f", type=float, help="spot variant: the spot covering fraction")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("variant")
    args = ap.parse_args()
    if args.cmd == "run" and args.variant == "spot" and None in (args.spot_t_phot, args.spot_dt, args.spot_f):
        ap.error("--variant spot needs --spot-t-phot, --spot-dt and --spot-f: the star is the case's, with no default")
    return run(args) if args.cmd == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
