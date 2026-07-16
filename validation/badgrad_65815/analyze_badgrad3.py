"""Third pass: lnZ/c_o risk gradients among certified; offender sub-classes."""
# Archived from the 2026-07-16 job-65815 investigation (docs/job65815_badgrad_investigation.md). Run from the repo root; optional argv[1] = forensics dir.
import glob
import os
import re

import numpy as np

import sys
from pathlib import Path
DIR = str(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "runs" / "w39b_smc_retrieval" / "forensics_65815")

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
stage = np.concatenate([np.full(len(r["bad_grad"]), r["stage"]) for r in records])
iZ, iC, iK, iT = 0, 1, 2, 3

# rate vs lnZ quartile (certified only)
print("badgrad rate among certified, by lnZ quartile:")
qs = np.quantile(theta[conv, iZ], [0, 0.25, 0.5, 0.75, 1.0])
for a, b in zip(qs[:-1], qs[1:]):
    m = conv & (theta[:, iZ] >= a) & (theta[:, iZ] <= b)
    zlo, zhi = 10 * np.exp(a), 10 * np.exp(b)
    print(f"  lnZ [{a:+.2f},{b:+.2f}] (Z {zlo:.0f}-{zhi:.0f}x solar): "
          f"{bad[m].sum():2d}/{m.sum():3d} = {bad[m].sum()/m.sum():.3f}")

print("\nbadgrad rate among certified, by c_o quartile:")
qs = np.quantile(theta[conv, iC], [0, 0.25, 0.5, 0.75, 1.0])
for a, b in zip(qs[:-1], qs[1:]):
    m = conv & (theta[:, iC] >= a) & (theta[:, iC] <= b)
    print(f"  c_o [{a:+.2f},{b:+.2f}] (C/O {0.549*np.exp(a):.2f}-{0.549*np.exp(b):.2f}): "
          f"{bad[m].sum():2d}/{m.sum():3d} = {bad[m].sum()/m.sum():.3f}")

print("\njoint (lnZ half x c_o half), certified:")
zm = np.median(theta[conv, iZ])
cm = np.median(theta[conv, iC])
for zl, zmask in [("lnZ<med", theta[:, iZ] < zm), ("lnZ>med", theta[:, iZ] >= zm)]:
    row = []
    for cl, cmask in [("c_o<med", theta[:, iC] < cm), ("c_o>med", theta[:, iC] >= cm)]:
        m = conv & zmask & cmask
        row.append(f"{cl}: {bad[m].sum():2d}/{m.sum():3d} ({bad[m].sum()/max(m.sum(),1):.3f})")
    print(f"  {zl}  " + "   ".join(row))

# sub-classes: marginal-cert offenders vs well-converged offenders
print("\noffender sub-classes:")
m_marg = bad & (longdy > 0.05)
m_well = bad & (longdy < 0.01)
m_mid = bad & ~m_marg & ~m_well
for lab, m in [("longdy>0.05 (marginal)", m_marg), ("0.01..0.05", m_mid),
               ("longdy<0.01 (well-conv)", m_well)]:
    if m.sum() == 0:
        continue
    print(f"  {lab}: n={m.sum()}  lnZ med={np.median(theta[m, iZ]):+.2f}  "
          f"c_o med={np.median(theta[m, iC]):+.2f}  lnKzz med={np.median(theta[m, iK]):+.2f}  "
          f"Tirr med={np.median(theta[m, iT]):.0f}")

# baseline for comparison
m = conv & ~bad
print(f"  cert non-off       : n={m.sum()}  lnZ med={np.median(theta[m, iZ]):+.2f}  "
      f"c_o med={np.median(theta[m, iC]):+.2f}  lnKzz med={np.median(theta[m, iK]):+.2f}  "
      f"Tirr med={np.median(theta[m, iT]):.0f}")

# stage-2 only: was the trip-sweep population different, or just unlucky?
print("\nstage 2 certified rates by lnZ half (small n, indicative only):")
m2 = stage == 2
zm2 = np.median(theta[conv & m2, iZ])
for zl, zmask in [("lnZ<med", theta[:, iZ] < zm2), ("lnZ>med", theta[:, iZ] >= zm2)]:
    m = conv & m2 & zmask
    print(f"  {zl}({zm2:+.2f}): {bad[m].sum()}/{m.sum()} = {bad[m].sum()/max(m.sum(),1):.3f}")

# extrapolate: if the ladder ends near the posterior (say cloud like stage2 lnZ>med),
# what per-sweep gate-trip probability does an 11% conditional rate imply?
import math
from math import comb
for p_cond, cert_frac in [(0.11, 0.57)]:
    p = p_cond * cert_frac  # per-proposal
    thr = math.ceil(0.05 * 144)
    pe = sum(comb(144, k) * p**k * (1 - p)**(144 - k) for k in range(thr + 1, 145))
    print(f"\nat stage-2 conditional rate ({p_cond:.2f}) x cert frac {cert_frac:.2f} "
          f"-> per-proposal p={p:.3f}: per-sweep P(trip)={pe:.2f}")
