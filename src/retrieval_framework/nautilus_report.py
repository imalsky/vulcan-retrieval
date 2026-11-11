"""Certificate and figures for a run_nautilus output directory.

    python -m retrieval_framework.nautilus_report <out_dir> [--spectra N]

Writes ``nautilus_certificate.json`` (provenance, sampler settings, logZ_box, the rejection tally, plain-English
caveats, the derived quantities, the conditioning map) and ``plots/`` (corner, T-P band, spectrum fit). There is no
pass/fail verdict. ``--spectra N`` re-solves N weighted posterior draws cold on the run's own pipeline, rebuilt from
config.json; a warm run certified its columns from other starts, so these spectra agree with the run's at the
convergence scale. docs/method.md lists the derived quantities.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from vulcan_forward.constants import BAR_CGS, KZZ_P_REF_BAR

from retrieval_framework import config_schema as C, constants, hardware, provenance

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

__all__ = [
    "CERT_FILE",
    "certificate",
    "config_from_json",
    "derived_quantities",
    "kzz_at_ref",
    "log10_kzz_at_report_level",
    "main",
    "read_json",
    "tp_curve",
    "weights",
    "write_certificate",
]

log = logging.getLogger("retrieval")

CERT_FILE = "nautilus_certificate.json"


def read_json(out: Path, name: str) -> Any:
    """Read one JSON file of the run."""
    return json.loads((out / name).read_text())


def weights(log_w: ArrayLike) -> np.ndarray:
    """Return the normalized weights from log weights."""
    w = np.exp(np.asarray(log_w, np.float64) - np.max(log_w))
    return w / w.sum()


def _wquant(x: ArrayLike, w: np.ndarray, qs: tuple[float, ...] = (0.16, 0.5, 0.84)) -> list[float]:
    """Return weighted quantiles of ``x``, interpolated on the cumulative weight."""
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw = (cw - 0.5 * w[o]) / cw[-1]
    return np.interp(qs, cw, np.asarray(x, np.float64)[o]).tolist()


# Keys with no default that a run written before they were Config keys does not record, and the line that restores the
# value it ran with, the engine's default of then. Any other missing key with no default is refused for its own value.
_ARCHIVED_LINES = {
    "network": '"network": "thermo/SNCHO_photo_network.txt"',
    "atom_list": '"atom_list": ["H", "O", "C", "N", "S"]',
    "art_pbtm_bar": '"art_pbtm_bar": 7.0',
}


def _tp_params(t: np.ndarray, names: list[str], cfgj: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return (Tirr, kappa, gamma, Tint) of one posterior row; Tint follows from a retrieved deep temperature.

    This and ``tp_curve`` are numpy twins of ``tp_profile``'s evaluator, so the report reads a finished run's rows
    without building the model; tests/test_nautilus_report.py holds them to it on every option.
    """
    g = float(cfgj["tp_gravity_cgs"])
    Tirr = float(t[names.index("Tirr")])
    kappa = 10.0 ** float(t[names.index("log10kappa")])
    gamma = 10.0 ** float(t[names.index("log10gamma")]) if "log10gamma" in names else float(cfgj["tp_gamma_fixed"])
    if "T1bar" in names:
        from retrieval_framework.tp_profile import tint_from_deep

        T_inv, p_inv = float(t[names.index("T1bar")]), float(cfgj["tp_deep_p_bar"])
        if "log10P_rcb" in names:
            # the deep level may sit on the adiabat (tp_profile._phys)
            p_rcb = 10.0 ** float(t[names.index("log10P_rcb")])
            T_inv = T_inv * (min(p_rcb, p_inv) / p_inv) ** float(cfgj["tp_nabla_ad"])
            p_inv = min(p_rcb, p_inv)
        Tint = float(np.asarray(tint_from_deep(T_inv, Tirr, kappa, gamma, g, constants.GUILLOT_F, p_inv)))
    elif "Tint" in names:
        Tint = float(t[names.index("Tint")])
    else:
        Tint = float(cfgj["tp_Tint_K"])
    return Tirr, kappa, gamma, Tint


def tp_curve(t: np.ndarray, names: list[str], cfgj: dict[str, Any], p_bar: ArrayLike) -> np.ndarray:
    """Return T(p_bar), K, of one posterior row: the Guillot profile, with the adiabat below the retrieved boundary."""
    g = float(cfgj["tp_gravity_cgs"])
    Tirr, kappa, gamma, Tint = _tp_params(t, names, cfgj)
    T = _guillot_T(p_bar, g, kappa, gamma, Tint, Tirr, constants.GUILLOT_F)
    if "log10P_rcb" in names:
        p_rcb = 10.0 ** float(t[names.index("log10P_rcb")])
        T_rcb = _guillot_T(p_rcb, g, kappa, gamma, Tint, Tirr, constants.GUILLOT_F)
        nabla = float(cfgj["tp_nabla_ad"])
        T = np.where(np.asarray(p_bar) > p_rcb, T_rcb * (np.asarray(p_bar) / p_rcb) ** nabla, T)
    return T


def kzz_at_ref(vulcan_cfg_name: str, cfg_overrides: dict[str, Any], p_bar: float = KZZ_P_REF_BAR) -> float:
    """Return the case's base Kzz at a pressure, the value e^lnKzz scales.

    vulcan3's ``const_Kzz``, or its atm file's Kzz column interpolated linearly in pressure, as the case's
    ``Kzz_prof`` says.

    Parameters
    ----------
    vulcan_cfg_name : str
        The case's vulcan3 config name or path.
    cfg_overrides : dict
        Overrides of the file's settings.
    p_bar : float, optional
        The pressure, bar; by default ``KZZ_P_REF_BAR``.

    Returns
    -------
    float
        Kzz, cm^2/s.

    Raises
    ------
    ValueError
        If the profile is neither ``const`` nor ``file``, or the file does not span ``p_bar``.
    """
    over = cfg_overrides
    _, case = provenance.case_config(vulcan_cfg_name)
    prof = over.get("Kzz_prof", case["Kzz_prof"])
    if prof == "const":
        return float(over.get("const_Kzz", case["const_Kzz"]))
    if prof != "file":
        raise ValueError(f"Kzz_prof {prof!r}: the report reads only 'file' or 'const'")
    atm = provenance.case_data_file(over.get("atm_file", case["atm_file"]))
    t = np.genfromtxt(atm, names=True, dtype=None, skip_header=1)
    p = np.asarray(t["Pressure"], np.float64) / BAR_CGS
    if not p.min() <= p_bar <= p.max():
        raise ValueError(f"{atm} does not span {p_bar} bar")
    o = np.argsort(p)
    return float(np.interp(p_bar, p[o], np.asarray(t["Kzz"], np.float64)[o]))


def log10_kzz_at_report_level(
    vulcan_cfg_name: str,
    cfg_overrides: dict[str, Any],
    powerlaw_p_ref_bar: float | None,
    lnkzz: np.ndarray,
    slope: np.ndarray | None,
) -> np.ndarray:
    """Return log10 Kzz at ``KZZ_P_REF_BAR`` of each row, on the engine's profile.

    The power law anchors the base at its reference pressure and tilts it as (P / P_anchor)^-slope; at the default
    anchor the tilt is 1. Otherwise e^lnKzz scales the case's profile.

    Parameters
    ----------
    vulcan_cfg_name : str
        The case's vulcan3 config name or path.
    cfg_overrides : dict
        Overrides of the file's settings.
    powerlaw_p_ref_bar : float or None
        The power law's ``kzz_p_ref_bar``, bar; None for ``kzz_mode: scale``.
    lnkzz : numpy.ndarray [n]
        Each row's lnKzz.
    slope : numpy.ndarray [n] or None
        Each row's power-law slope; None under the scale mode.

    Returns
    -------
    numpy.ndarray [n]
        log10 Kzz, cm^2/s.
    """
    kzz = np.exp(lnkzz)
    if powerlaw_p_ref_bar is None:
        return np.log10(kzz_at_ref(vulcan_cfg_name, cfg_overrides) * kzz)
    anchor = float(powerlaw_p_ref_bar)
    tilt = (KZZ_P_REF_BAR / anchor) ** (-(0.0 if slope is None else slope))
    return np.log10(kzz_at_ref(vulcan_cfg_name, cfg_overrides, anchor) * kzz * tilt)


def _powerlaw_p_ref(cfgj: dict[str, Any]) -> float | None:
    """Return a config.json's power-law reference pressure, None under the scale mode."""
    return float(cfgj["kzz_p_ref_bar"]) if cfgj["kzz_mode"] == "powerlaw" else None


def derived_quantities(out_dir: str | Path) -> dict[str, Any] | None:
    """Return weighted 16/50/84% quantiles of the quantities the paper quotes.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        The run's output directory.

    Returns
    -------
    dict or None
        See docs/method.md; None before the run finishes.
    """
    out = Path(out_dir).resolve()
    if not (out / "nautilus_posterior.npz").exists():
        return None
    cfgj = read_json(out, "config.json")
    post = np.load(out / "nautilus_posterior.npz")
    names = [str(s) for s in post["param_names"]]
    theta = np.asarray(post["theta"], np.float64)
    w = weights(post["log_w"])

    def col(n: str) -> np.ndarray:
        """Return one parameter's column of theta."""
        return theta[:, names.index(n)]

    base = float(cfgj["baseline_metallicity_solar"])
    eps = {k: 10.0 ** (v - 12.0) for k, v in constants.LODDERS2020_LOG_EPS.items()}
    lnZ, c_o = col("lnZ"), col("c_o")
    s_o = col("s_o") if "s_o" in names else np.zeros_like(lnZ)
    c_to_o = provenance.baseline_c_to_o(cfgj["vulcan_cfg_name"], cfgj["cfg_overrides"])
    z_o = base * np.exp(lnZ)
    tot = eps["O"] + eps["N"] + eps["S"] * np.exp(s_o) + eps["C"] * np.exp(c_o)
    kzz = np.exp(col("lnKzz"))
    floor = float(np.exp(cfgj["prior_lnKzz"][0]))
    d = {
        "quantiles": [0.16, 0.5, 0.84],
        "metallicity_O_H_solar": _wquant(z_o, w),
        "metallicity_total_solar": _wquant(z_o * tot / sum(eps.values()), w),
        "C_to_O": _wquant(c_to_o * np.exp(c_o), w),
        "S_to_O_solar": _wquant(np.exp(s_o), w) if "s_o" in names else None,
        "kzz_factor": _wquant(kzz, w),
        "kzz_ref_p_bar": KZZ_P_REF_BAR,
        "log10_kzz_ref_cgs": _wquant(
            log10_kzz_at_report_level(
                cfgj["vulcan_cfg_name"],
                cfgj["cfg_overrides"],
                _powerlaw_p_ref(cfgj),
                col("lnKzz"),
                col("kzz_slope") if "kzz_slope" in names else None,
            ),
            w,
        ),
        "kzz_prior_floor_factor": floor,
        **{f"kzz_weight_below_{m:g}x_floor": float(w[kzz < m * floor].sum()) for m in constants.KZZ_FLOOR_MULTIPLES},
    }
    if "report_kzz_ceiling_x" not in cfgj:
        log.info("config.json has no report_kzz_ceiling_x (written before the key): no truncated Kzz block")
    ceil = cfgj.get("report_kzz_ceiling_x")
    keep = kzz <= (np.inf if ceil is None else ceil)
    if ceil is not None and keep.any():
        wt = w[keep] / w[keep].sum()
        d[f"kzz_weight_above_x{ceil:g}"] = float(w[~keep].sum())
        d[f"truncated_kzz_le_x{ceil:g}"] = {
            "weight": float(w[keep].sum()),
            "metallicity_O_H_solar": _wquant(z_o[keep], wt),
            "C_to_O": _wquant(c_to_o * np.exp(c_o[keep]), wt),
            "kzz_factor": _wquant(kzz[keep], wt),
        }
    if "Tirr" in names:
        p_deep = np.array([float(cfgj["tp_deep_p_bar"])])
        # the full profile (adiabat included) at the deep level, and the Tint
        # behind it
        T_deep = np.asarray([tp_curve(t, names, cfgj, p_deep)[0] for t in theta])
        Tint = np.asarray([_tp_params(t, names, cfgj)[3] for t in theta])
        d.update(T_deep_p_bar=float(p_deep[0]), T_deep_K=_wquant(T_deep, w), Tint_K=_wquant(Tint, w))
    if "nuisance" in post.files:
        # integrated offsets (ppm) and inflations: one conditional draw per row,
        # NaN on a rejected (zero-weight) row
        nu = np.asarray(post["nuisance"], np.float64)
        ok = np.isfinite(nu).all(axis=1)
        d["nuisance"] = {str(n): _wquant(nu[ok, j], w[ok] / w[ok].sum()) for j, n in enumerate(post["nuisance_names"])}
    return d


def _conditioning_map(out_dir: str | Path) -> dict[str, Any] | None:
    """Return where the rejections sit, for lnKzz, the deep T-P parameter and lnZ, one at a time.

    ``constants.COND_BINS`` equal bins over the prior box, each with the in-window proposals evaluated, the rejected
    fraction and the posterior weight. A batch redrawn after a resume counts twice. None without anchors/ (a cold run
    keeps no certified points) or rejected/.
    """
    out = Path(out_dir).resolve()
    afiles = sorted((out / "anchors").glob("anchors_*.npz"))
    if not (afiles and (out / "rejected").is_dir() and (out / "nautilus_posterior.npz").exists()):
        return None
    cfgj = read_json(out, "config.json")
    post = np.load(out / "nautilus_posterior.npz")
    names = [str(s) for s in post["param_names"]]
    theta, w = np.asarray(post["theta"], np.float64), weights(post["log_w"])

    def z_of(files: list[Path], keep: Callable[[Any], Any]) -> list[np.ndarray]:
        """Return the kept unit-cube points of each file."""
        zs = []
        for f in files:
            with np.load(f) as d:
                zs.append(d["z"][keep(d)])
        return zs

    ok = z_of(afiles, lambda _d: slice(None))
    bad = z_of(sorted((out / "rejected").glob("rejected_*.npz")), lambda d: d["cls"] > 0)
    edges_z = np.linspace(0.0, 1.0, constants.COND_BINS + 1)
    n_ok, n_bad = sum(len(z) for z in ok), sum(len(z) for z in bad)
    m: dict[str, Any] = {
        "rejected_of_in_window": n_bad / (n_ok + n_bad) if n_ok + n_bad else None,
        "rejected_in_log": n_bad,
    }
    for name in constants.COND_PARAMS:
        if name not in names:
            continue
        j = names.index(name)  # the anchors keep the chemistry block, which holds j
        lo, hi = cfgj[f"prior_{name}"]
        c_ok = sum(np.histogram(z[:, j], edges_z)[0] for z in ok)
        c_bad = sum((np.histogram(z[:, j], edges_z)[0] for z in bad), np.zeros(constants.COND_BINS, int))
        edges = lo + (hi - lo) * edges_z
        n = c_ok + c_bad
        m[name] = {
            "edges": edges.tolist(),
            "n_in_window": n.tolist(),
            "rejected_frac": [float(b / t) if t else None for b, t in zip(c_bad, n, strict=True)],
            "posterior_weight": np.histogram(theta[:, j], edges, weights=w)[0].tolist(),
        }
    return m


def _rejection_fractions(tally: dict) -> dict[str, Any]:
    """Return the certificate's rejection block from the tally."""

    def frac(key: str, denom: int | None) -> float | None:
        """Return one tally count over ``denom``, or None with no denominator."""
        return tally.get(key, 0) / denom if denom else None

    n = tally.get("n_eval", 0)
    in_window = n - tally.get("tp_out", 0)
    return {
        "n_eval": n,
        "tp_out_of_n_eval": frac("tp_out", n),
        "rejected_of_in_window": _n_rejected(tally) / in_window if in_window else None,
        "exhausted_of_in_window": frac("exhausted", in_window),
        "stalled_of_in_window": frac("stalled", in_window),
        "nonfinite_of_in_window": frac("nonfinite", in_window),
        "cold_certified_of_cold": frac("cold_certified", tally.get("n_cold")),
        "warm_certified_of_warm": frac("warm_certified", tally.get("n_warm")),
    }


def _n_rejected(tally: dict) -> int:
    """Return the in-window rejections in the tally."""
    return sum(tally.get(k, 0) for k in ("exhausted", "stalled", "nonfinite"))


def _model_caveats(cfg: dict, summ: dict, tally: dict, rejections: dict, cond: dict | None) -> list[str]:
    """Return the caveats on what the run's numbers mean, from its settings, tally and conditioning map."""
    rejected = _n_rejected(tally)
    caveats = [
        (
            f"{rejections['rejected_of_in_window'] or 0.0:.1%} of the draws with a T-P "
            "profile in the window were rejected (chemistry not certified within its cap, "
            "or a non-finite forward). Rejected draws have zero likelihood, so logZ_box is "
            "the box-prior evidence with those regions removed and depends on count_max "
            "and the rtol."
        ),
        "nautilus reports no logZ error: quote the spread over independent seeds.",
    ]
    if summ["starts"] != "cold":
        caveats.append(
            "Warm starts: each solve continued from the nearest certified column, so the "
            "likelihood and which draws were rejected depend on the evaluation order. The "
            "posterior and logZ_box are approximate; an evidence claim needs a cold run "
            "(warm_starts: false in the spec)."
        )
    if cond is None:
        caveats.append("No conditioning map: the run kept no anchors/ or no rejected/ log.")
    elif "lnKzz" in cond:
        k = cond["lnKzz"]
        f0 = k["rejected_frac"][0]
        caveats.append(
            f"Rejections by Kzz: {'no' if f0 is None else f'{f0:.1%} of the'} in-window "
            f"draws in the lowest bin (lnKzz {k['edges'][0]:.2f} to {k['edges'][1]:.2f}) "
            f"{'were evaluated' if f0 is None else 'were rejected'}, against "
            f"{cond['rejected_of_in_window']:.1%} overall; that bin holds "
            f"{k['posterior_weight'][0]:.3f} of the posterior weight (evaluated proposals; "
            f"{cond['rejected_in_log']} in the rejected log against {rejected} in the tally)."
        )
    if cfg["integrate_nuisance"]:
        caveats.append(
            "The offsets and noise inflations were integrated out of the likelihood "
            "(the evidence and the other parameters' posterior are those of the sampled "
            "model); their posteriors are one conditional draw per row from the statistics "
            "the run logged, on the log10-b quadrature grid."
        )
    if cfg["infer_lnM"]:
        caveats.append(
            "lnM moves the planet mass in the RT only: the chemistry column keeps the "
            "baseline mass M0, so its gravity does not vary with lnM."
        )
    if int(cfg["cold_lanes"]) > 0:
        caveats.append(
            "Lane queue: a draw that starts in a freed lane starts at that tick, so its "
            "column and likelihood move with its batch at the convergence scale."
        )
    n = tally.get("n_eval", 0)
    if n > int(summ["n_like"]):
        caveats.append(
            f"The tally counts {n - int(summ['n_like'])} more evaluations than nautilus "
            "(n_like): a batch redrawn after a resume is counted twice, so the fractions "
            "are approximate."
        )
    return caveats


def _unmodeled_caveat(out: Path) -> str:
    """Return the caveat on network species with no opacity table, from the draws the report re-solved."""
    path = out / "nautilus_spectra.npz"
    if not path.is_file():
        return (
            "Species with no opacity table were not checked: no posterior draw was re-solved (nautilus_report "
            "--spectra N re-solves N)."
        )
    with np.load(path) as z:
        if "unmodeled_species" not in z.files:
            return f"Species with no opacity table were not checked: {path.name} was written before the check."
        n_ok, n = int(np.sum(z["certified"])), len(z["certified"])
        not_run = str(z["unmodeled_not_performed"]) if "unmodeled_not_performed" in z.files else ""
        found = dict(zip(z["unmodeled_species"].tolist(), z["unmodeled_vmr"].tolist(), strict=True))
    where = (
        f"in the transmission photosphere ({constants.PHOTOSPHERE_BAR[0]:g}-{constants.PHOTOSPHERE_BAR[1]:g} bar) "
        f"of the {n_ok} of {n} re-solved draws that certified"
    )
    if not_run:
        return f"Species with no opacity table: the check was not performed ({not_run}, of {n} re-solved draws)."
    if not found:
        return f"No species without an opacity table reaches a mean VMR of {constants.UNMODELED_VMR:g} {where}."
    return (
        "Species with no opacity table, so absent from the RT: "
        + ", ".join(f"{sp} at mean VMR up to {v:.2e}" for sp, v in found.items())
        + f" {where}. The spectrum omits their absorption, an error of unknown sign."
    )


def _identity_caveats(manifest: dict, manifest_digest: str | None, digest: str, code_now: dict) -> list[str]:
    """Return the caveats on whether the identity files and the code are the run's."""
    caveats = []
    run_code = manifest.get("code") or {}
    if manifest_digest != digest.split("|", maxsplit=1)[0]:
        caveats.append(
            "target_manifest.json does not hash to this run's target digest: the identity "
            "files in this directory belong to a different run."
        )
    dirty = sorted(r for r, s in run_code.items() if (s or {}).get("dirty"))
    if dirty:
        caveats.append(f"The run's {', '.join(dirty)} tree(s) had uncommitted changes.")
    moved = sorted(r for r, s in run_code.items() if (s or {}).get("commit") != (code_now.get(r) or {}).get("commit"))
    if moved:
        caveats.append(
            f"code_now differs from the run's code for {', '.join(moved)}: "
            "this certificate was written at another commit."
        )
    return caveats


def certificate(out_dir: str | Path) -> dict[str, Any]:
    """Return the certificate of the finished run in ``out_dir``.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        The run's output directory.

    Returns
    -------
    dict
        The certificate.
    """
    out = Path(out_dir).resolve()
    cfg = read_json(out, "config.json")
    summ = read_json(out, "nautilus_summary.json")
    tally = read_json(out, "nautilus_tally.json")
    manifest = read_json(out, provenance.MANIFEST_FILE)
    digest = (out / "target_digest.txt").read_text().strip()
    manifest_digest = provenance.archived_manifest_digest(out)
    run_code = manifest.get("code") or {}
    code_now = provenance._repo_states()

    n = tally.get("n_eval", 0)
    warm = summ["starts"] != "cold"
    rejections = _rejection_fractions(tally)
    cond = _conditioning_map(out)
    caveats = _model_caveats(cfg, summ, tally, rejections, cond)
    caveats.append(_unmodeled_caveat(out))
    caveats += _identity_caveats(manifest, manifest_digest, digest, code_now)

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
        "sampler": {
            k: summ.get(k)
            for k in (
                "nautilus",
                "n_live",
                "n_update",
                "n_batch",
                "f_live",
                "enlarge_per_dim",
                "networks",
                "discard_exploration",
                "seed",
                "spec",
            )
        },
        "result": {"logZ_box": summ["logZ_box"], "logZ_err": None, "n_eff": summ["n_eff"], "n_like": summ["n_like"]},
        "starts": {
            "mode": "warm" if warm else "cold",
            "warm_of_n_eval": tally.get("n_warm", 0) / n if n else None,
            "cold_of_n_eval": tally.get("n_cold", 0) / n if n else None,
            # Such a row stays in n_warm and warm_certified, by its cold verdict.
            "warm_frozen_resolved_cold": tally.get("warm_frozen_resolved_cold", 0),
        },
        "rejections": rejections,
        "chemistry": {
            "rtol": summ.get("chem_rtol"),
            # A presolve-chain run: its certifying rtol and cap.
            **({"chem_retry": summ["chem_retry"]} if "chem_retry" in summ else {}),
            **{k: cfg[k] for k in ("count_max", "yconv_cri", "dt_max", "nz", "cold_lanes", "cfg_overrides")},
        },
        "derived": derived_quantities(out),
        "conditioning": cond,
        "caveats": caveats,
    }


def write_certificate(out_dir: str | Path) -> Path:
    """Write ``nautilus_certificate.json`` into ``out_dir``.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        The run's output directory.

    Returns
    -------
    pathlib.Path
        The file.
    """
    path = Path(out_dir) / CERT_FILE
    path.write_text(json.dumps(certificate(out_dir), indent=2, default=str) + "\n")
    return path


def _guillot_T(
    p_bar: ArrayLike, gravity: float, kappa: float, gamma: float, Tint: float, Tirr: float, f: float
) -> np.ndarray:
    """Return Guillot (2010) Eq. 29, the numpy twin of ``exojax.atm.atmprof.atmprof_Guillot``."""
    tau = np.asarray(p_bar) * BAR_CGS * kappa / gravity
    invsq3 = 1.0 / np.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (1.0 / gamma + (gamma - 1.0 / gamma) * np.exp(-gamma * tau / invsq3))
    return (0.75 * Tint**4 * (2.0 / 3.0 + tau) + 0.75 * Tirr**4 * f * fac) ** 0.25


def config_from_json(path: Path) -> C.Config:
    """Return the run's Config from its config.json, validated.

    A key the run's config.json predates is filled from today's default and logged, and a key it records that is no
    longer a field is dropped and logged; a key with no default it lacks is refused, naming the line to add.

    Parameters
    ----------
    path : pathlib.Path
        The run's config.json.

    Returns
    -------
    config_schema.Config
        The run's configuration.

    Raises
    ------
    RuntimeError
        If the run solved with a second program this tree no longer builds.
    ValueError
        If it lacks a key with no default, or ``validate_config`` refuses it.
    """
    from retrieval_framework import case_keys

    d = json.loads(Path(path).read_text())
    if d.get("chem_presolve") or d.get("chem_retry"):
        raise RuntimeError(
            f"{path} ran the presolve chain or its retry pass, a second program this tree no longer builds: re-solve "
            "its draws at vulcan-retrieval 0a2dbf3"
        )
    names = {f.name for f in fields(C.Config)}
    kw: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names}
    missing = sorted(k for k in case_keys.required_keys(C.Config(**kw)) if kw.get(k) in (None, "", ()))
    if missing:
        raise ValueError(
            f"{path} names no {missing}, which have no default. A run written before they were Config keys ran on the "
            "engine's defaults of then: add "
            + ", ".join(_ARCHIVED_LINES.get(k, f'"{k}": <the value that run used>') for k in missing)
            + " to it"
        )
    log.info("%s: not Config fields, not read: %s", path.name, sorted(set(d) - names))
    filled = sorted(names - set(d) - {"out_dir"})
    if filled:
        log.warning("%s: written before these Config keys, filled with today's defaults: %s", path.name, filled)
    for k in ("out_dir", "obs_dir"):
        if kw.get(k) is not None:
            kw[k] = Path(kw[k])
    cfg = C.Config(**kw)
    C.validate_config(cfg)
    return cfg


def _model_spectra(
    out_dir: str | Path, cfg: C.Config, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Re-solve ``n`` weighted posterior draws cold.

    With integrated nuisances the depth carries each row's drawn offsets and the multiplier its drawn inflations;
    otherwise theta holds both and the multiplier is None.

    Parameters
    ----------
    out_dir : str or pathlib.Path
        The run's output directory.
    cfg : config_schema.Config
        The run's configuration (``config_from_json``).
    n : int
        The draws.

    Returns
    -------
    tuple
        ``(theta [n, n_dim], model depth [n, n_bin], certified [n], sigma multiplier [n, n_bin] or None)``.

    Raises
    ------
    ValueError
        If the rebuilt pipeline's parameters are not the posterior's.
    """
    out = Path(out_dir).resolve()
    C.export_network(cfg)  # before vulcan3 is imported

    from retrieval_framework import pipeline as P

    post = np.load(out / "nautilus_posterior.npz")
    rng = np.random.default_rng(int(C.need(cfg.seed, "seed")) + 1)
    theta = np.asarray(post["theta"], np.float64)
    idx = rng.choice(len(theta), size=int(n), p=weights(post["log_w"]))
    theta = theta[idx]
    log.info("solver: %s", hardware.choose_solver())  # before vulcan3 is imported
    pipe = P.build_pipeline(cfg)
    if list(pipe.names) != [str(s) for s in post["param_names"]]:
        raise ValueError(
            f"the rebuilt pipeline's parameters {pipe.names} are not the posterior's {list(post['param_names'])}"
        )
    t0 = time.perf_counter()
    Y, mu, ok = pipe.solve_cold(theta)
    log.info("model spectra: %s/%s draws certified cold in %.0f s", int(ok.sum()), len(ok), time.perf_counter() - t0)
    unmodeled, not_run = _unmodeled_absorbers(pipe, np.asarray(Y)[ok])
    mult = None
    if pipe.nuisance is not None:
        mu, mult = pipe.nuisance.apply(mu, np.asarray(post["nuisance"], np.float64)[idx])
    P.save_npz(
        out / "nautilus_spectra.npz",
        theta=theta,
        model=mu,
        certified=ok,
        unmodeled_species=np.asarray(list(unmodeled), dtype="<U16"),
        unmodeled_vmr=np.asarray(list(unmodeled.values()), np.float64),
        unmodeled_not_performed=np.asarray(not_run),
        **({} if mult is None else {"sigma_multiplier": mult}),
    )
    return theta, mu, ok, mult


def _unmodeled_absorbers(pipe: Any, Y: np.ndarray) -> tuple[dict[str, float], str]:
    """Return the network species with no opacity table that are abundant in the observable layers of columns ``Y``.

    The rule of jwst-transit-authority's ``certify.unmodeled_absorbers``: a species' mean gas VMR over
    ``constants.PHOTOSPHERE_BAR``, its largest over the columns, above ``constants.UNMODELED_VMR``; most abundant first.
    The mixing ratio is over the gas only (the RT's gas mask), so a condensed reservoir neither dilutes it nor counts.
    The second item is why the check could not run, or empty when it ran.
    """
    from vulcan_forward.constants import MOLECULES

    tabled = {v["vulcan"] for k, v in MOLECULES.items() if k not in constants.NO_OPACITY_TABLE}
    skip = tabled | set(constants.NOT_A_TABLE_ABSORBER)
    sidx = pipe.fwd.chem.sidx
    p = np.asarray(pipe.fwd.p_chem_bar)
    band = (p >= constants.PHOTOSPHERE_BAR[0]) & (p <= constants.PHOTOSPHERE_BAR[1])
    if not len(Y):
        return {}, "no re-solved draw certified"
    if not band.any():
        return {}, f"no chemistry layer lies in {constants.PHOTOSPHERE_BAR[0]:g}-{constants.PHOTOSPHERE_BAR[1]:g} bar"
    gas = np.asarray(pipe.fwd.gas_mask, np.float64)
    y_gas = Y * gas[None, None, :]
    vmr = y_gas / y_gas.sum(axis=2, keepdims=True)
    peak = vmr[:, band, :].mean(axis=1).max(axis=0)
    out = {
        sp: float(peak[i])
        for sp, i in sidx.items()
        if gas[i] > 0.0 and sp not in skip and peak[i] > constants.UNMODELED_VMR
    }
    return dict(sorted(out.items(), key=lambda kv: -kv[1])), ""


def main(argv: list[str] | None = None) -> None:
    """Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        The arguments; default ``sys.argv``.
    """
    from retrieval_framework.report_figures import plot
    from retrieval_framework.run_setup import setup_logging

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="a run_nautilus output directory")
    ap.add_argument(
        "--spectra",
        type=int,
        default=32,
        help="posterior draws to re-solve for the spectrum figure (0: no forward model, no spectrum figure; "
        "a draw frozen at dt_min stops the report, and 0 still gives the certificate and the other figures)",
    )
    args = ap.parse_args(argv)
    setup_logging()
    out = Path(args.out_dir).resolve()
    cfg = config_from_json(out / "config.json")  # refuses an archived config.json before any figure
    # the forward build comes first: the certificate's version record imports
    # exojax, which must not precede vulcan_chem
    spectra = _model_spectra(out, cfg, args.spectra) if args.spectra > 0 else None
    log.info("wrote %s", write_certificate(out))
    for p in plot(out, spectra):
        log.info("wrote %s", p)


if __name__ == "__main__":
    main()
