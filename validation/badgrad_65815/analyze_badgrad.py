"""Forensics analysis of bad_grad dumps from NAS job 65815."""
# Archived from the 2026-07-16 job-65815 investigation (docs/job65815_badgrad_investigation.md). Run from the repo root; optional argv[1] = forensics dir.
import glob
import math
import os
import re

import numpy as np

import sys
from pathlib import Path
DIR = str(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "runs" / "w39b_smc_retrieval" / "forensics_65815")
NAMES = ["lnZ", "c_o", "lnKzz", "Tirr", "log10kappa", "log10gamma",
         "lnR0", "log10kappa_cloud", "cloud_alpha", "offset_G395H"]

files = sorted(glob.glob(os.path.join(DIR, "bad_grad_stage*.npz")))
print(f"{len(files)} dump files")

records = []
for fp in files:
    m = re.search(r"stage(\d+)_sweep(\d+)", fp)
    stage, sweep = int(m.group(1)), int(m.group(2))
    d = dict(np.load(fp))
    records.append(dict(stage=stage, sweep=sweep, **d))

# sanity: keys and shapes
r0 = records[0]
for k, v in r0.items():
    if isinstance(v, np.ndarray):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

N = r0["bad_grad"].shape[0]
print(f"N = {N}")

# ---------------------------------------------------------------- per-sweep table
print("\n=== per-sweep summary ===")
tot_bad = 0
all_rates = []
for r in records:
    bad = r["bad_grad"].astype(bool)
    nb = int(bad.sum())
    tot_bad += nb
    all_rates.append(nb / N)
    chem = int(r["chem_tan_bad"][bad].sum()) if nb else 0
    convok_frac = float(r["conv_ok"].mean())
    print(f"stage {r['stage']} sweep {r['sweep']}: bad={nb:2d} chem={chem:2d} "
          f"cloud conv_ok frac={convok_frac:.3f} "
          f"offender conv_ok={r['conv_ok'][bad].astype(int).tolist() if nb else []}")
print(f"total badgrad events: {tot_bad} over {len(records)} sweeps "
      f"-> pooled rate {tot_bad/(len(records)*N):.4f}")

# ---------------------------------------------------------------- repeat offenders
print("\n=== repeat offenders within each stage (indices persist across sweeps) ===")
from collections import Counter, defaultdict
for stage in sorted(set(r["stage"] for r in records)):
    cnt = Counter()
    for r in records:
        if r["stage"] != stage:
            continue
        for i in np.flatnonzero(r["bad_grad"]):
            cnt[int(i)] += 1
    rep = {i: c for i, c in cnt.items() if c > 1}
    print(f"stage {stage}: {len(cnt)} distinct offender particles, "
          f"repeats: {dict(sorted(rep.items())) if rep else 'none'}")

# ---------------------------------------------------------------- longdy analysis
print("\n=== longdy ===")
off_longdy = np.concatenate([r["longdy"][r["bad_grad"].astype(bool)] for r in records])
all_longdy = np.concatenate([r["longdy"] for r in records])
ok_longdy = np.concatenate([r["longdy"][~r["bad_grad"].astype(bool)] for r in records])


def longdy_bins(x, label):
    edges = [0, 1e-3, 1e-2, 5e-2, 0.09, 0.101, np.inf]
    labels = ["<1e-3", "1e-3..0.01", "0.01..0.05", "0.05..0.09", "0.09..0.101", ">0.101"]
    h, _ = np.histogram(x, bins=edges)
    tot = len(x)
    parts = ", ".join(f"{l}: {c} ({100*c/tot:.0f}%)" for l, c in zip(labels, h))
    print(f"{label} (n={tot}): {parts}")


longdy_bins(off_longdy, "offenders ")
longdy_bins(ok_longdy, "non-offend")
print(f"offender longdy: min={off_longdy.min():.3g} med={np.median(off_longdy):.3g} max={off_longdy.max():.3g}")
print(f"cloud    longdy: min={ok_longdy.min():.3g} med={np.median(ok_longdy):.3g} max={ok_longdy.max():.3g}")
# fraction of proposals with longdy > yconv_cri=0.01 (weak-branch certs) overall
print(f"frac longdy>0.01: offenders {np.mean(off_longdy > 0.01):.2f}  "
      f"non-offenders {np.mean(ok_longdy > 0.01):.2f}")

# ---------------------------------------------------------------- accept counts
print("\n=== accept counts ===")
off_acc = np.concatenate([r["acc"][r["bad_grad"].astype(bool)] for r in records]).astype(float)
ok_acc = np.concatenate([r["acc"][~r["bad_grad"].astype(bool)] for r in records]).astype(float)
for lab, x in [("offenders", off_acc), ("non-offend", ok_acc)]:
    q = np.percentile(x, [5, 25, 50, 75, 95])
    print(f"{lab}: p5={q[0]:.0f} p25={q[1]:.0f} med={q[2]:.0f} p75={q[3]:.0f} p95={q[4]:.0f} max={x.max():.0f}")
print(f"frac acc>=1490 (near warm cap 1500): offenders {np.mean(off_acc >= 1490):.2f} "
      f"non-offenders {np.mean(ok_acc >= 1490):.2f}")

# ---------------------------------------------------------------- theta clustering
print("\n=== offender theta percentile-in-cloud, per dimension ===")
# for each offender, percentile of its proposal within that sweep's full cloud
pct = defaultdict(list)
off_theta = []
ok_theta = []
for r in records:
    bad = r["bad_grad"].astype(bool)
    th = r["theta_proposal"]
    off_theta.append(th[bad])
    ok_theta.append(th[~bad])
    for j in range(th.shape[1]):
        col = th[:, j]
        for v in col[bad]:
            pct[j].append(float(np.mean(col <= v)))
off_theta = np.concatenate(off_theta)
ok_theta = np.concatenate(ok_theta)
n_off = off_theta.shape[0]
print(f"pooled offenders: {n_off}")
print(f"{'param':<18}{'off_med':>10}{'cloud_med':>11}{'off_p10':>9}{'off_p90':>9}"
      f"{'pct_med':>9}{'pct_mean':>9}")
for j, name in enumerate(NAMES):
    p = np.array(pct[j])
    om = np.median(off_theta[:, j])
    cm = np.median(ok_theta[:, j])
    o10, o90 = np.percentile(off_theta[:, j], [10, 90])
    # under H0 (no clustering) pct ~ U(0,1): mean 0.5, sd of mean = 1/sqrt(12 n)
    z = (p.mean() - 0.5) / (1.0 / math.sqrt(12 * len(p)))
    print(f"{name:<18}{om:>10.3f}{cm:>11.3f}{o10:>9.3f}{o90:>9.3f}"
          f"{np.median(p):>9.2f}{p.mean():>9.2f}  z={z:+.1f}")

# ---------------------------------------------------------------- binomial check
print("\n=== gate statistics ===")
p_hat = tot_bad / (len(records) * N)
thr = math.ceil(0.05 * N)
# P(X > thr) for Binomial(N, p_hat)
from math import comb
p_exceed = sum(comb(N, k) * p_hat**k * (1 - p_hat)**(N - k) for k in range(thr + 1, N + 1))
print(f"pooled per-proposal badgrad rate p = {p_hat:.4f}")
print(f"per-sweep P(X > {thr}) under Binomial({N}, {p_hat:.4f}) = {p_exceed:.3f}")
n_sweeps_ladder = 6 * 40
print(f"P(gate trips within 14 sweeps)  = {1 - (1 - p_exceed)**14:.3f}")
print(f"P(gate trips within a full {n_sweeps_ladder}-sweep ladder) = {1 - (1 - p_exceed)**n_sweeps_ladder:.4f}")
# trend test: stage-level rates
for stage in sorted(set(r["stage"] for r in records)):
    ss = [r for r in records if r["stage"] == stage]
    nb = sum(int(r["bad_grad"].sum()) for r in ss)
    print(f"stage {stage}: {nb} events / {len(ss)*N} proposals = {nb/(len(ss)*N):.4f}")

# ---------------------------------------------------------------- conv_ok cross-check
print("\n=== conv_ok of offenders (should all be certified) ===")
allok = all(bool(r["conv_ok"][r["bad_grad"].astype(bool)].all()) for r in records if r["bad_grad"].any())
print(f"every offender canonically certified: {allok}")

# loglik of non-offender proposals for scale
ok_ll = np.concatenate([r["loglik_proposal"][~r["bad_grad"].astype(bool)] for r in records])
finite = ok_ll[ok_ll > -1e29]
print(f"non-offender loglik: {len(finite)}/{len(ok_ll)} finite, "
      f"med={np.median(finite):.1f} p5={np.percentile(finite,5):.1f} p95={np.percentile(finite,95):.1f}")
