"""Follow-up: corner localization + certified-only baselines."""
# Archived from the 2026-07-16 job-65815 investigation (docs/job65815_badgrad_investigation.md). Run from the repo root; optional argv[1] = forensics dir.
import glob
import os
import re

import numpy as np

import sys
from pathlib import Path
DIR = str(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "runs" / "w39b_smc_retrieval" / "forensics_65815")
NAMES = ["lnZ", "c_o", "lnKzz", "Tirr", "log10kappa", "log10gamma",
         "lnR0", "log10kappa_cloud", "cloud_alpha", "offset_G395H"]

records = []
for fp in sorted(glob.glob(os.path.join(DIR, "bad_grad_stage*.npz"))):
    m = re.search(r"stage(\d+)_sweep(\d+)", fp)
    d = dict(np.load(fp))
    d["stage"], d["sweep"] = int(m.group(1)), int(m.group(2))
    records.append(d)

bad = np.concatenate([r["bad_grad"] for r in records]).astype(bool)
conv = np.concatenate([r["conv_ok"] for r in records]).astype(bool)
theta = np.concatenate([r["theta_proposal"] for r in records])
longdy = np.concatenate([r["longdy"] for r in records])
acc = np.concatenate([r["acc"] for r in records]).astype(float)
stage = np.concatenate([np.full(len(r["bad_grad"]), r["stage"]) for r in records])

cert_ok = conv & ~bad          # certified, tangent fine
print(f"proposals={len(bad)}  certified={conv.sum()} ({conv.mean():.1%})  "
      f"badgrad={bad.sum()}  badgrad/certified={bad.sum()/conv.sum():.3f}")

# conditional badgrad rate among certified, per stage
print("\nbadgrad rate among CERTIFIED proposals, per stage:")
for s in sorted(set(stage)):
    m = stage == s
    print(f"  stage {s}: {bad[m].sum()}/{conv[m].sum()} = {bad[m].sum()/conv[m].sum():.3f}")

# longdy: offenders vs certified-only non-offenders
edges = [0, 1e-3, 1e-2, 5e-2, 0.09, 0.101]
labels = ["<1e-3", "1e-3..0.01", "0.01..0.05", "0.05..0.09", "0.09..0.101"]
print("\nlongdy distribution (certified populations only):")
for lab, m in [("offenders  ", bad), ("cert non-off", cert_ok)]:
    h, _ = np.histogram(longdy[m], bins=edges)
    tot = m.sum()
    print(f"  {lab} (n={tot}): " + ", ".join(f"{l}: {100*c/tot:.0f}%" for l, c in zip(labels, h)))

# ---- corner localization in (c_o, lnKzz, lnZ), certified reference population
iZ, iC, iK = 0, 1, 2
print("\nmedians (certified non-off vs offenders):")
for i, n in [(iZ, "lnZ"), (iC, "c_o"), (iK, "lnKzz")]:
    print(f"  {n:6}: cert {np.median(theta[cert_ok, i]):+.3f}   off {np.median(theta[bad, i]):+.3f}")

# local badgrad probability on a 2x2 split of certified proposals at cert medians
cm_c = np.median(theta[conv, iC])
cm_k = np.median(theta[conv, iK])
print(f"\n2x2 split of CERTIFIED proposals at medians c_o={cm_c:.2f}, lnKzz={cm_k:.2f}")
print("cell: badgrad / certified (rate)")
for clab, cmask in [("c_o<med", theta[:, iC] < cm_c), ("c_o>med", theta[:, iC] >= cm_c)]:
    row = []
    for klab, kmask in [("Kzz<med", theta[:, iK] < cm_k), ("Kzz>med", theta[:, iK] >= cm_k)]:
        m = conv & cmask & kmask
        row.append(f"{klab}: {bad[m].sum():3d}/{m.sum():4d} ({bad[m].sum()/max(m.sum(),1):.3f})")
    print(f"  {clab}  " + "   ".join(row))

# progressively deeper corner
print("\ndeep-corner badgrad rate among certified:")
for cq, kq in [(0.5, 0.5), (0.35, 0.65), (0.25, 0.75)]:
    cc = np.quantile(theta[conv, iC], cq)
    kk = np.quantile(theta[conv, iK], kq)
    m = conv & (theta[:, iC] < cc) & (theta[:, iK] > kk)
    print(f"  c_o<q{cq:.2f}({cc:+.2f}) & lnKzz>q{kq:.2f}({kk:+.2f}): "
          f"{bad[m].sum()}/{m.sum()} = {bad[m].sum()/max(m.sum(),1):.3f}")

# add lnZ to the corner
cc = np.quantile(theta[conv, iC], 0.5)
kk = np.quantile(theta[conv, iK], 0.5)
zz = np.quantile(theta[conv, iZ], 0.5)
m = conv & (theta[:, iC] < cc) & (theta[:, iK] > kk) & (theta[:, iZ] > zz)
print(f"  + lnZ>med({zz:+.2f}): {bad[m].sum()}/{m.sum()} = {bad[m].sum()/max(m.sum(),1):.3f}")
mo = conv & ~((theta[:, iC] < cc) & (theta[:, iK] > kk))
print(f"  everywhere OUTSIDE (c_o<med & Kzz>med): {bad[mo].sum()}/{mo.sum()} = {bad[mo].sum()/mo.sum():.3f}")

# physical units for the corner
co_med_off = 0.549 * np.exp(np.median(theta[bad, iC]))
z_med_off = 10.0 * np.exp(np.median(theta[bad, iZ]))
print(f"\noffender medians in physical units: C/O ~ {co_med_off:.2f}, "
      f"Z ~ {z_med_off:.0f}x solar, Kzz x{np.exp(np.median(theta[bad, iK])):.0f}")

# does the certified CLOUD itself drift toward the corner with stage?
print("\ncertified-cloud median drift by stage (is the population moving into the corner?):")
for s in sorted(set(stage)):
    m = conv & (stage == s)
    print(f"  stage {s}: lnZ={np.median(theta[m, iZ]):+.2f} c_o={np.median(theta[m, iC]):+.2f} "
          f"lnKzz={np.median(theta[m, iK]):+.2f} Tirr={np.median(theta[m, 3]):.0f}")

# certification rate inside vs outside the corner (does the corner also fail primal more?)
mc = (theta[:, iC] < cm_c) & (theta[:, iK] >= cm_k)
print(f"\ncertification rate inside corner: {conv[mc].mean():.3f}  outside: {conv[~mc].mean():.3f}")

# Tirr of offenders, absolute
print(f"offender Tirr: p10={np.percentile(theta[bad,3],10):.0f} med={np.median(theta[bad,3]):.0f} "
      f"p90={np.percentile(theta[bad,3],90):.0f}")
