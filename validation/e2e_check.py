"""End-to-end check of the production forward map: the same seeded draws through a ladder of rungs.

    python validation/e2e_check.py solve <spec> --rung production [--rows a:b --part k]
    python validation/e2e_check.py solve <spec> --rung production --posterior <out_dir> --draws n
    python validation/e2e_check.py compare <a.npz> <b.npz> [<b_part2.npz> ...]
    python validation/e2e_check.py warmcold <e2e_production_posterior.npz> [<part2.npz> ...] [--results <dir>]

Each rung is one change from the rung above, compared per draw on the certificate, log L, the binned depth against the
data's sigma and the photosphere column of six species, so a disagreement names the change that made it. ``compare`` and
``warmcold`` exit 1 over a gate. Rungs, draws, gates and results are in docs/method.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifact

SPECIES = ("H2O", "CO", "CO2", "CH4", "SO2", "H2S")
P_PHOT_BAR = (1.0e-5, 1.0)
VMR_FLOOR = 1.0e-12
GOOD_FIT_NATS = 50.0
GATES = {"max_bin_sigma": 0.1, "D_sigma": 0.5, "dex_p95": 0.03, "dex_max": 0.1, "dlogL": 0.5}
WARM_COLD_SHIFT_SIGMA = 0.1
LOW_KZZ = -0.7
CORNER_Z = (0.02, 0.98)
CORNER_PARAMS = ("lnZ", "c_o", "lnKzz", "s_o", "Tirr", "T1bar", "log10kappa")
LOW_KZZ_WITH = ("lnZ", "c_o", "s_o", "T1bar")
# rung -> (environment, batched); an unlisted variable keeps the spec's value, an empty one is unset
RUNGS = {
    "production": ({}, True),
    "shuffled": ({}, True),
    "cmdbuf_off": ({"XLA_FLAGS": "cmdbuf_off"}, True),
    "no_xla_flags": ({"XLA_FLAGS": ""}, True),
    "codegen": ({"XLA_FLAGS": "", "VULCAN3_CHEM_RHS": "codegen"}, True),
    "fast": ({"XLA_FLAGS": "", "VULCAN3_CHEM_RHS": "codegen", "VULCAN3_SOLVER": "fast"}, True),
    "solo_gpu": ({"XLA_FLAGS": "", "VULCAN3_CHEM_RHS": "codegen", "VULCAN3_SOLVER": "fast"}, False),
    "solo_cpu": (
        {"XLA_FLAGS": "", "VULCAN3_CHEM_RHS": "codegen", "VULCAN3_SOLVER": "fast", "JAX_PLATFORMS": "cpu"},
        False,
    ),
    "reference_cpu": (
        {"XLA_FLAGS": "", "VULCAN3_CHEM_RHS": "codegen", "VULCAN3_SOLVER": "reference", "JAX_PLATFORMS": "cpu"},
        False,
    ),
}


def _set_environment(spec, rung):
    """The rung's environment, before any jax or vulcan3 import."""
    from retrieval_framework.run_setup import export_runtime

    env = spec.runtime.env()
    for k, v in RUNGS[rung][0].items():
        if k == "XLA_FLAGS" and v == "cmdbuf_off":
            flags = [f for f in (env["XLA_FLAGS"] or "").split() if not f.startswith("--xla_gpu_enable_command_buffer")]
            env[k] = " ".join([*flags, "--xla_gpu_enable_command_buffer="])
        else:
            env[k] = v or None
    rt = replace(
        spec.runtime, solver=env["VULCAN3_SOLVER"], chem_rhs=env["VULCAN3_CHEM_RHS"], xla_flags=env["XLA_FLAGS"]
    )
    for k in ("VULCAN3_SOLVER", "VULCAN3_CHEM_RHS", "XLA_FLAGS"):
        os.environ.pop(k, None)  # the rung decides, never the shell
    if env.get("JAX_PLATFORMS"):
        os.environ["JAX_PLATFORMS"] = env["JAX_PLATFORMS"]
    export_runtime(rt)
    return rt


def _corners(pipe, names, origin):
    """Designed corners in u-space (z = CORNER_Z on one parameter, low Kzz with each
    of LOW_KZZ_WITH), each from ``origin`` (an in-window point: the prior midpoint is
    out of the T-P window for most cases); the out-of-window ones are dropped and
    logged.
    """
    import jax.numpy as jnp

    logit = lambda z: float(np.log(z) - np.log1p(-z))  # noqa: E731  # a short local name
    rows = []
    for p in CORNER_PARAMS:
        for z in CORNER_Z:
            if p in names:
                rows.append({p: z})
    for p in LOW_KZZ_WITH:
        if p in names:
            rows += [{"lnKzz": CORNER_Z[0], p: z} for z in CORNER_Z]
    out = []
    for r in rows:
        u = np.array(origin, np.float64)
        for p, z in r.items():
            u[names.index(p)] = logit(z)
        if bool(pipe.tp_valid(pipe.theta_from_u(jnp.asarray(u, pipe.dtype)))):
            out.append(u)
        else:
            logging.info("corner %s leaves the T-P window: dropped", r)
    return np.asarray(out)


def _draw_set(args, pipe, names):
    """(U, corners, extra columns): a finished run's weighted rows, another machine's set, or seeded prior draws."""
    import jax

    if args.posterior:
        post = np.load(Path(args.posterior) / "nautilus_posterior.npz")
        if list(post["param_names"]) != names:
            raise RuntimeError(f"the posterior's parameters {list(post['param_names'])} are not the spec's {names}")
        from retrieval_framework.nautilus_report import _weights

        w = _weights(post["log_w"])
        idx = np.random.default_rng(args.seed).choice(len(w), size=args.draws, p=w)
        return np.array(post["u"][idx], np.float64), (), {"post_row": idx, "logL_run": post["log_l"][idx]}
    if args.u_from:  # another machine's draw set, row for row
        z = np.load(args.u_from)
        if not np.array_equal(z["rows"], np.arange(len(z["rows"]))):
            raise RuntimeError(f"{args.u_from} holds a slice; pass the full set")
        return np.array(z["u"], np.float64), np.zeros(int(json.loads(str(z["meta"]))["n_corners"])), {}
    U = np.array(pipe.sample_prior_u(jax.random.PRNGKey(args.seed), args.draws), np.float64)
    corners = _corners(pipe, names, U[np.argmin((U**2).sum(axis=1))])[: len(U)]
    U[: len(corners)] = corners
    return U, corners, {}


def _solve_rows(pipe, u_rows, batched):
    """(log L, columns, accepted steps, certified) of the rows ``u_rows``, in one batch or one row at a time."""
    import jax
    import jax.numpy as jnp
    from retrieval_framework import pipeline as P

    ev = jax.jit(pipe._make_batch_eval("cold"))

    def run(u):
        n = int(u.shape[0])
        Y0, R0 = P._blank_state(pipe, n)
        L, Y, _, st = ev(u, Y0, R0, jnp.ones((n,), bool))
        P._abort_if_frozen(st.reason, st.last_delta, np.asarray(u), pipe.fwd.chem_rtol)
        return L, Y, st.acc, st.conv_ok

    if batched:
        return run(u_rows)
    parts = [run(u_rows[i : i + 1]) for i in range(int(u_rows.shape[0]))]
    return tuple(jnp.concatenate([p[k] for p in parts]) for k in range(4))


def solve(args) -> int:
    from retrieval_framework.run_setup import load_spec, set_observations

    spec = load_spec(args.spec)
    rt = _set_environment(spec, args.rung)
    from retrieval_framework.config_schema import export_network

    export_network(spec.cfg)  # before vulcan3 is imported
    from vulcan_forward import vulcan_chem  # noqa: F401  (import order)
    import jax
    import jax.numpy as jnp
    from retrieval_framework import pipeline as P
    from retrieval_framework.run_nautilus import run_lanes

    lanes = run_lanes(spec)
    cfg = replace(spec.cfg, cold_lanes=lanes)
    pipe = P.build_pipeline(cfg)
    set_observations(cfg, pipe, P)
    names = list(pipe.names)
    U, C, extra = _draw_set(args, pipe, names)
    lo, hi = (int(x) if x else None for x in args.rows.split(":"))
    rows = np.arange(len(U))[lo:hi]
    order = (
        np.random.default_rng(args.seed + 1).permutation(len(rows)) if args.rung == "shuffled" else np.arange(len(rows))
    )
    Us = jnp.asarray(U[rows][order], pipe.dtype)
    L, Y, acc, ok = _solve_rows(pipe, Us, RUNGS[args.rung][1])
    inv = np.argsort(order)
    theta = pipe.theta_batch(Us)
    mu = np.asarray(pipe.observed_depth_from_y_jit(Y, theta), np.float64)[inv]
    sidx = pipe.fwd.chem.sidx
    Yn = np.asarray(Y, np.float64)[inv]
    vmr = Yn / Yn.sum(axis=2, keepdims=True)
    meta = {
        "rung": args.rung,
        "spec": str(spec.path),
        "network": spec.cfg.network,
        "seed": args.seed,
        "draws": args.draws,
        "posterior": str(args.posterior or ""),
        "rows": args.rows,
        "lanes": lanes,
        "runtime": rt.env(),
        "n_corners": len(C),
        "device": str(jax.devices()[0]),
        "provenance": _artifact.collect_provenance(),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    tag = (
        f"e2e_{args.rung}"
        + ("_posterior" if args.posterior else "")
        + (f"_part{args.part}" if args.part is not None else "")
    )
    path = args.results / f"{tag}.npz"
    np.savez(
        path,
        rows=rows,
        u=U[rows],
        theta=np.asarray(theta, np.float64)[inv],
        logL=np.asarray(L, np.float64)[inv],
        acc=np.asarray(acc)[inv],
        certified=np.asarray(ok)[inv],
        mu=mu,
        sigma=np.asarray(pipe.obs["sigma"]),
        vmr=vmr[:, :, [sidx[s] for s in SPECIES]],
        p_bar=np.asarray(pipe.fwd.chem.p_bar),
        names=np.asarray(names),
        meta=np.asarray(json.dumps(meta, default=str)),
        **{k: v[rows] for k, v in extra.items()},
    )
    logging.info("%s: %s/%s certified; wrote %s", args.rung, int(np.sum(ok)), len(rows), path)
    return 0


def _load(paths):
    zs = [np.load(p) for p in paths]
    rows = np.concatenate([z["rows"] for z in zs])
    out = {k: np.concatenate([z[k] for z in zs]) for k in ("u", "theta", "logL", "acc", "certified", "mu", "vmr")}
    for k in ("post_row", "logL_run"):
        if k in zs[0]:
            out[k] = np.concatenate([z[k] for z in zs])
    out.update(
        rows=rows,
        sigma=zs[0]["sigma"],
        p_bar=zs[0]["p_bar"],
        names=list(zs[0]["names"]),
        meta=json.loads(str(zs[0]["meta"])),
    )
    return out


def _over_the_gates(m, mism, both):
    """The names of the gates a comparison's metrics ``m`` break."""
    fails = [
        k
        for k, v in (
            ("max_bin_sigma", m["max_bin_sigma"]),
            ("D_sigma", m["D_sigma_max"]),
            ("dex_p95", m["dex_p95"]),
            ("dex_max", m["dex_max"]),
            ("dlogL", m["dlogL_good_fit_max"]),
        )
        if not v <= GATES[k]  # a NaN metric fails
    ]
    if mism.size:
        fails.append("certificate_mismatch")
    if not both.any():
        fails.append("no_jointly_certified_draw")
    return fails


def compare(args) -> int:
    from retrieval_framework import constants

    a, b = _load([args.a]), _load(args.b)
    common, ia, ib = np.intersect1d(a["rows"], b["rows"], return_indices=True)
    if not np.array_equal(a["u"][ia], b["u"][ib]):
        raise RuntimeError("the two files hold different draws for the same rows")
    ok_a = a["certified"][ia] & (a["logL"][ia] > constants.REJECT_LOGL / 2)
    ok_b = b["certified"][ib] & (b["logL"][ib] > constants.REJECT_LOGL / 2)
    both = ok_a & ok_b
    mism = common[ok_a != ok_b]
    dmu = (a["mu"][ia] - b["mu"][ib])[both] / a["sigma"][None, :]
    max_bin, D = np.max(np.abs(dmu), axis=1), np.sqrt(np.sum(dmu**2, axis=1))
    band = (a["p_bar"] >= P_PHOT_BAR[0]) & (a["p_bar"] <= P_PHOT_BAR[1])
    ya, yb = a["vmr"][ia][both][:, band], b["vmr"][ib][both][:, band]
    keep = (ya > VMR_FLOOR) & (yb > VMR_FLOOR)
    dex = np.where(keep, np.abs(np.log10(np.where(keep, ya, 1.0)) - np.log10(np.where(keep, yb, 1.0))), 0.0)
    dex_row = dex.reshape(dex.shape[0], -1).max(axis=1) if dex.size else np.zeros(0)
    crossings = int(np.sum((ya > VMR_FLOOR) != (yb > VMR_FLOOR)))
    La, Lb = a["logL"][ia][both], b["logL"][ib][both]
    good = La >= (La.max() - GOOD_FIT_NATS if La.size else 0.0)
    dL = np.abs(La - Lb)
    m = {
        "rows_compared": int(common.size),
        "certified_a": int(ok_a.sum()),
        "certified_b": int(ok_b.sum()),
        "certificate_mismatch_rows": mism.tolist(),
        "max_bin_sigma": float(max_bin.max(initial=0.0)),
        "D_sigma_max": float(D.max(initial=0.0)),
        "D_sigma_median": float(np.median(D)) if D.size else 0.0,
        "dex_p95": float(np.percentile(dex[keep], 95)) if keep.any() else 0.0,
        "dex_max": float(dex.max(initial=0.0)),
        "floor_crossings": crossings,
        "dlogL_good_fit_max": float(dL[good].max(initial=0.0)),
        "good_fit_rows": int(good.sum()),
        "dlogL_all_median": float(np.median(dL)) if dL.size else 0.0,
    }
    fails = _over_the_gates(m, mism, both)
    # the worst draws, with the parameters a disagreement usually tracks
    names, rows_both = a["names"], common[both]
    worst = np.argsort(-D)[: args.worst]
    keyp = [p for p in ("lnZ", "c_o", "lnKzz", "s_o", "T1bar", "Tirr") if p in names]
    th = a["theta"][ia][both]
    m["worst"] = [
        dict(
            row=int(rows_both[k]),
            D_sigma=float(D[k]),
            max_bin_sigma=float(max_bin[k]),
            dex=float(dex_row[k]),
            dlogL=float(dL[k]),
            **{p: float(th[k, names.index(p)]) for p in keyp},
        )
        for k in worst
    ]
    ra, rb = a["meta"]["rung"], b["meta"]["rung"]
    status = "FAIL" if fails else "PASS"
    summary = (
        f"{ra} vs {rb}: {int(both.sum())} draws certified on both, "
        f"{mism.size} certificate mismatches; worst bin {m['max_bin_sigma']:.3g} sigma, "
        f"D max {m['D_sigma_max']:.3g}; dex p95 {m['dex_p95']:.3g} max {m['dex_max']:.3g}; "
        f"|dlogL| good-fit max {m['dlogL_good_fit_max']:.3g}"
        + (f"; over the gate: {', '.join(fails)}" if fails else "")
    )
    print(summary)
    for w in m["worst"]:
        print("  ", w)
    _artifact.emit(
        f"e2e_{ra}_vs_{rb}",
        f"End-to-end forward map, {ra} vs {rb}",
        [m, {"gates": GATES, "meta_a": a["meta"], "meta_b": b["meta"]}],
        status,
        summary,
        out_dir=args.results,
    )
    return int(bool(fails))


def _wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])])


def warmcold(args) -> int:
    from retrieval_framework import constants

    z = _load(args.cold)
    if "logL_run" not in z:
        raise RuntimeError(f"{args.cold} was not solved with --posterior")
    ok = z["certified"] & (z["logL"] > constants.REJECT_LOGL / 2)
    dL = np.where(ok, z["logL"] - z["logL_run"], -np.inf)
    w = np.exp(dL - np.max(dL[ok]))
    names, th = z["names"], z["theta"]
    i_kzz = names.index("lnKzz")
    low = th[:, i_kzz] < LOW_KZZ
    shifts = {}
    for j, n in enumerate(names):
        sd = float(np.std(th[:, j]))
        shifts[n] = (_wmedian(th[:, j], w) - float(np.median(th[:, j]))) / sd if sd > 0 else 0.0
    worst = max(shifts, key=lambda n: abs(shifts[n]))
    m = {
        "rows": len(dL),
        "cold_certified": int(ok.sum()),
        "cold_certified_low_kzz": int(ok[low].sum()),
        "low_kzz_rows": int(low.sum()),
        "median_abs_dL": float(np.median(np.abs(dL[ok]))),
        "median_abs_dL_low_kzz": float(np.median(np.abs(dL[ok & low]))) if (ok & low).any() else None,
        "median_abs_dL_high_kzz": float(np.median(np.abs(dL[ok & ~low]))) if (ok & ~low).any() else None,
        "ess": float(w.sum() ** 2 / np.sum(w**2)),
        "median_shift_sigma": shifts,
        "worst": worst,
        "worst_shift_sigma": float(shifts[worst]),
    }
    status = "PASS" if abs(shifts[worst]) <= WARM_COLD_SHIFT_SIGMA else "FAIL"
    summary = (
        f"warm vs cold on {m['rows']} weighted posterior rows: {m['cold_certified']} certify "
        f"cold ({m['cold_certified_low_kzz']}/{m['low_kzz_rows']} at low Kzz); median |dL| "
        f"{m['median_abs_dL']:.3g}; worst median shift {worst} {shifts[worst]:+.3f} sigma "
        f"(gate {WARM_COLD_SHIFT_SIGMA}); ESS {m['ess']:.0f}"
    )
    print(summary)
    _artifact.emit(
        f"warm_cold_{Path(z['meta']['posterior']).name}",
        "Warm vs cold posterior rows",
        [m],
        status,
        summary,
        out_dir=args.results,
    )
    return int(status == "FAIL")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("spec")
    s.add_argument("--rung", choices=list(RUNGS), required=True)
    s.add_argument("--draws", type=int, default=1584)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--rows", default=":")
    s.add_argument("--part", type=int, default=None)
    s.add_argument("--posterior", default=None, help="a finished run's output directory")
    s.add_argument(
        "--u-from", default=None, help="take the draws from a full-set e2e npz (a GPU rung's), for another machine"
    )
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b", nargs="+")
    c.add_argument("--worst", type=int, default=10)
    w = sub.add_parser("warmcold")
    w.add_argument("cold", nargs="+", help="the production rung's --posterior solve (parts)")
    for p in (s, c, w):
        p.add_argument(
            "--results",
            type=Path,
            default=_artifact.RESULTS,
            help="the results directory (default: WASP-39b's validation directory)",
        )
    args = ap.parse_args()
    return {"solve": solve, "compare": compare, "warmcold": warmcold}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
