#!/usr/bin/env python3
"""The solver-cut test: the in-window draws a finished run rejected below a Kzz cut,
re-solved cold with a longer cap, against the run's posterior log L at the same Kzz.

    python validation/rejected_resolve.py select <out_dir> --results <dir> [--log10kzz-max 8.1] [--per-bin N]
    python validation/e2e_check.py solve <cut spec> --rung production --u-from <dir>/rejected_draws.npz --results <dir>
    python validation/rejected_resolve.py compare <out_dir> <dir>/e2e_production.npz

`select` reads the run's rejected/ log (run_nautilus: the unit-cube point z, the class
cls, the accepted steps acc, the chunk's start mode warm) and keeps the in-window
rejections (cls 1 exhausted, 2 stalled, 3 non-finite) with log10 Kzz below the cut, each
point once, at most --per-bin of them per bin (seeded), as rejected_draws.npz in
e2e_check's --u-from layout (u = logit z) with their class, steps and start mode. The
cut spec is the run's spec with only count_max raised, the run's network named
(runs/<case>/<run>_cut.yaml). The
re-solve is cold, through the production cold evaluator: the run rejected most of these
draws warm (continued from the nearest anchor within its cap), and the cold map does not
depend on the run's evaluation order.
`compare` refuses a solve that does not hold the draws of the rejected_draws.npz beside
it, whose network is not the run's, or whose theta is not the run's prior map of its u (a
spec with other priors), then bins the re-solved draws and the posterior rows in log10 Kzz
(BIN_DEX from the prior's lower edge to the cut; Kzz = KZZ0 e^lnKzz, the case's
constant profile) and reports per bin: draws re-solved and certified, the certified
draws' log L (median, max, how many reach the posterior rows' weighted median there)
and steps, the posterior rows' weight and log L (weighted median, max), and the run's
best log L. A REPORT artifact, solver_cut_<run>.json, beside the solve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifact  # noqa: E402
from e2e_check import _wmedian  # noqa: E402

KZZ0 = 1.0e9          # cm^2/s: the survey's constant Kzz (data/survey/<case>/atm_<case>_Kzz.txt)
BIN_DEX = 0.35


def _run(out: Path):
    cfg = json.loads((out / "config.json").read_text())
    names = list(cfg["inferred_param_names"])
    j = names.index("lnKzz")
    if cfg["inferred_param_prior_types"][j] != "uniform":
        raise RuntimeError(f"{out}: lnKzz's prior is not uniform")
    return cfg, names, j


def _log10kzz(lnkzz):
    return np.log10(KZZ0) + np.asarray(lnkzz) / np.log(10.0)


def _edges(cfg, cut):
    lo = float(_log10kzz(cfg["prior_lnKzz"][0]))
    return np.append(np.arange(lo, cut - 1e-9, BIN_DEX), cut)


def select(args) -> int:
    out = Path(args.out_dir).resolve()
    cfg, names, j = _run(out)
    parts = [dict(np.load(f)) for f in sorted((out / "rejected").glob("rejected_*.npz"))]
    if not parts:
        raise RuntimeError(f"{out} has no rejected/ log")
    z = np.concatenate([p["z"] for p in parts])
    cls = np.concatenate([p["cls"] for p in parts])
    acc = np.concatenate([p["acc"] for p in parts])
    warm = np.concatenate([np.full(len(p["cls"]), int(p["warm"])) for p in parts])
    lo, hi = cfg["prior_lnKzz"]
    lk = _log10kzz(lo + (hi - lo) * z[:, j])
    first = np.zeros(len(z), bool)
    first[np.unique(z, axis=0, return_index=True)[1]] = True     # a redrawn batch logs twice
    idx = np.flatnonzero(first & (cls > 0) & (lk < args.log10kzz_max))
    edges = _edges(cfg, args.log10kzz_max)
    b = np.digitize(lk[idx], edges) - 1
    n_all = np.bincount(b, minlength=len(edges) - 1)
    if args.per_bin:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(np.concatenate([rng.permutation(idx[b == k])[:args.per_bin]
                                      for k in range(len(edges) - 1)]))
    zz = z[idx]
    meta = dict(n_corners=0, run=str(out), log10kzz_max=args.log10kzz_max, kzz0=KZZ0,
                bin_edges=edges.tolist(), in_window_below_cut_per_bin=n_all.tolist(),
                per_bin=args.per_bin, seed=args.seed, selected=int(len(idx)))
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    path = res / "rejected_draws.npz"
    np.savez(path, rows=np.arange(len(idx)), u=np.log(zz) - np.log1p(-zz), z=zz, cls=cls[idx],
             acc=acc[idx], warm=warm[idx], log10kzz=lk[idx], meta=np.asarray(json.dumps(meta)))
    print(f"{out.name}: {int(n_all.sum())} in-window rejected draws below log10 Kzz "
          f"{args.log10kzz_max} (per {BIN_DEX} dex from {edges[0]:.2f}: {n_all.tolist()}); "
          f"{len(idx)} selected ({np.bincount(cls[idx], minlength=4)[1:].tolist()} exhausted/"
          f"stalled/non-finite, {int(warm[idx].sum())} rejected warm); wrote {path}")
    return 0


def compare(args) -> int:
    from retrieval_framework.pipeline import REJECT_LOGL
    out = Path(args.out_dir).resolve()
    cfg, names, j = _run(out)
    e = np.load(args.solved)
    if list(e["names"]) != names:
        raise RuntimeError(f"the solve's parameters {list(e['names'])} are not the run's {names}")
    lo, hi = np.asarray(cfg["inferred_param_prior_lo"]), np.asarray(cfg["inferred_param_prior_hi"])
    uni = np.asarray(cfg["inferred_param_prior_types"]) == "uniform"
    want = lo + (hi - lo) / (1.0 + np.exp(-np.asarray(e["u"])))
    if not np.allclose(e["theta"][:, uni], want[:, uni], rtol=0.0, atol=1e-9 * (hi - lo)[uni]):
        raise RuntimeError(f"{args.solved}: theta is not the run's prior map of u: the spec's "
                           "priors differ from the run's")
    ok = e["certified"] & (e["logL"] > REJECT_LOGL / 2)
    lk_r, L, acc = _log10kzz(e["theta"][:, j]), e["logL"], e["acc"]
    post = np.load(out / "nautilus_posterior.npz")
    w = np.exp(post["log_w"] - np.max(post["log_w"]))
    w = w / w.sum()
    Lp = post["log_l"]
    fin = np.isfinite(Lp)
    lk_p = _log10kzz(post["theta"][:, j])
    meta = json.loads(str(e["meta"]))
    if not cfg.get("network"):
        raise RuntimeError(f"{out}/config.json names no network. A run written before "
                           "`network` was a Config key solved thermo/SNCHO_photo_network.txt: add "
                           "\"network\": \"thermo/SNCHO_photo_network.txt\" to it")
    if meta.get("network") != cfg["network"]:
        raise RuntimeError(f"{args.solved} solved {meta.get('network')!r} but the run solved "
                           f"{cfg['network']!r}: the cut spec must name the run's network")
    sel = np.load(Path(args.solved).resolve().parent / "rejected_draws.npz")
    if not np.array_equal(sel["u"][e["rows"]], e["u"]):
        raise RuntimeError(f"{args.solved} does not hold the draws of rejected_draws.npz beside it")
    sel_meta = json.loads(str(sel["meta"]))
    edges = np.asarray(sel_meta["bin_edges"])
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        r = (lk_r >= a) & (lk_r < b)
        p = fin & (lk_p >= a) & (lk_p < b)
        Lc = L[r & ok]
        med = _wmedian(Lp[p], w[p]) if p.any() else None
        rows.append(dict(
            log10kzz=[round(float(a), 3), round(float(b), 3)], resolved=int(r.sum()),
            certified=int((r & ok).sum()),
            logL_median=float(np.median(Lc)) if Lc.size else None,
            logL_max=float(Lc.max()) if Lc.size else None,
            at_or_above_posterior_median=int((Lc >= med).sum()) if med is not None else None,
            steps_certified_median=float(np.median(acc[r & ok])) if Lc.size else None,
            posterior_rows=int(p.sum()), posterior_weight=float(w[p].sum()),
            posterior_logL_wmedian=med, posterior_logL_max=float(Lp[p].max()) if p.any() else None))
    best = float(Lp[fin].max())
    print(f"{out.name}: {int(ok.sum())}/{len(ok)} rejected draws certify cold "
          f"(spec {Path(meta['spec']).name}); the run's best log L {best:.2f}")
    print(f"{'log10 Kzz':>13} {'solved':>6} {'cert':>5} {'L med':>9} {'L max':>9} {'>=post':>6} "
          f"{'steps':>6} | {'post w':>6} {'post L med':>10} {'post L max':>10}")
    f = lambda x: "-" if x is None else f"{x:.2f}"      # noqa: E731
    for r in rows:
        print(f"{r['log10kzz'][0]:6.2f}-{r['log10kzz'][1]:<6.2f} {r['resolved']:6d} "
              f"{r['certified']:5d} {f(r['logL_median']):>9} {f(r['logL_max']):>9} "
              f"{'-' if r['at_or_above_posterior_median'] is None else r['at_or_above_posterior_median']:>6} "
              f"{'-' if r['steps_certified_median'] is None else int(r['steps_certified_median']):>6} | "
              f"{r['posterior_weight']:6.3f} "
              f"{f(r['posterior_logL_wmedian']):>10} {f(r['posterior_logL_max']):>10}")
    summary = (f"{int(ok.sum())}/{len(ok)} in-window rejected draws below log10 Kzz "
               f"{edges[-1]:.2f} certify cold; best recovered log L "
               f"{f(float(L[ok].max()) if ok.any() else None)} against the run's {best:.2f}")
    _artifact.emit(f"solver_cut_{out.name}", "Solver-cut test: rejected draws re-solved cold",
                   [dict(rows=rows, run_best_logL=best, selection=sel_meta, solve_meta=meta)],
                   "REPORT", summary,
                   out_dir=Path(args.solved).resolve().parent)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("out_dir", help="a finished run's output directory")
    s.add_argument("--results", required=True, help="the directory for rejected_draws.npz")
    s.add_argument("--log10kzz-max", type=float, default=8.1)
    s.add_argument("--per-bin", type=int, default=None, help="at most this many draws per bin")
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("out_dir", help="the run whose rejected draws were solved")
    c.add_argument("solved", help="e2e_check's solve of rejected_draws.npz")
    args = ap.parse_args()
    return select(args) if args.cmd == "select" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
