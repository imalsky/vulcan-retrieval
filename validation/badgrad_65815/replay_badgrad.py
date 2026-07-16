"""Replay job-65815 badgrad offender thetas through the production chemistry
path locally (CPU), instrumenting the forward-mode tangents.

Per case:
  1. cold two-stage solve (primal, ConvDiag) -> certified y*
  2. jvp of the SMC warm re-solve (warm_cap=True) from y* at the same theta,
     6 unit tangent lanes vmapped exactly like pipeline._chem_one
  3. pointwise jvp of prep_pv (initial-state build) -- separates single-op
     overflow in the prep from recurrence divergence in the loop

Writes one JSON line per case to replay_results.jsonl and per-case npz dumps.
"""
# Archived from the 2026-07-16 job-65815 investigation (docs/job65815_badgrad_investigation.md). Run from the repo root; optional argv[1] = forensics dir. Writes to output/badgrad_65815/ (gitignored).
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
RUN_DIR = _REPO / "runs" / "w39b_smc_retrieval"
FOR_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN_DIR / "forensics_65815"
SCRATCH = _REPO / "output" / "badgrad_65815"
SCRATCH.mkdir(parents=True, exist_ok=True)
OUT_JSONL = SCRATCH / "replay_results.jsonl"

sys.path.insert(0, str(_REPO))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- select cases
import glob
import re

recs = []
for fp in sorted(glob.glob(str(FOR_DIR / "bad_grad_stage*.npz"))):
    m = re.search(r"stage(\d+)_sweep(\d+)", fp)
    d = dict(np.load(fp))
    for i in range(len(d["bad_grad"])):
        recs.append(dict(stage=int(m.group(1)), sweep=int(m.group(2)), idx=i,
                         bad=bool(d["bad_grad"][i]), conv=bool(d["conv_ok"][i]),
                         longdy=float(d["longdy"][i]), acc=int(d["acc"][i]),
                         theta=d["theta_proposal"][i]))

off = [r for r in recs if r["bad"]]
cert = [r for r in recs if r["conv"] and not r["bad"]]

off_sorted = sorted(off, key=lambda r: r["longdy"])
cases = []
for r in off_sorted[:3]:
    cases.append(("off_wellconv", r))
for r in off_sorted[-3:]:
    cases.append(("off_marginal", r))

corner = [r for r in cert if r["theta"][1] < -0.9 and r["theta"][0] > 1.5]
center = [r for r in cert if abs(r["theta"][1] + 0.5) < 0.25 and abs(r["theta"][0] - 1.2) < 0.4]
corner_s = sorted(corner, key=lambda r: r["longdy"])
center_s = sorted(center, key=lambda r: r["longdy"])
if corner_s:
    cases.append(("ctl_corner_tight", corner_s[0]))
    cases.append(("ctl_corner_marginal", corner_s[-1]))
if center_s:
    cases.append(("ctl_center_tight", center_s[0]))
    cases.append(("ctl_center_marginal", center_s[-1]))

log(f"{len(off)} offenders, {len(cert)} certified controls available; "
    f"{len(cases)} replay cases selected")
for lab, r in cases:
    log(f"  {lab}: stage{r['stage']} sweep{r['sweep']} idx{r['idx']} "
        f"longdy={r['longdy']:.3g} acc={r['acc']} "
        f"lnZ={r['theta'][0]:+.2f} c_o={r['theta'][1]:+.2f} lnKzz={r['theta'][2]:+.2f} "
        f"Tirr={r['theta'][3]:.0f}")

# ---------------------------------------------------------------- build model
log("loading case config (gpu preset)...")
from retrieval_framework.run_smc import load_case

case_mod = load_case(RUN_DIR)
cfg = case_mod.PRESETS["gpu"]()
profile = cfg.profile()

log("building chemistry model (vulcan_chem first: env + x64)...")
t0 = time.time()
from retrieval_framework.forward import vulcan_chem
from retrieval_framework import tp_profile
import jax
import jax.numpy as jnp

tpm = tp_profile.build_tp_model(cfg)
chem = vulcan_chem.build_chem_model(profile, tp_eval=tpm.eval, n_tp_params=tpm.n_params)
log(f"chem model built in {time.time() - t0:.1f}s  nz={chem.nz} ni={chem.ni} "
    f"count_max={chem.count_max} warm_count_max={chem.warm_count_max} "
    f"yconv_min={chem.yconv_min} yconv_cri={chem.yconv_cri}")

species = sorted(chem.sidx, key=chem.sidx.get)
N_LANES = 3 + tpm.n_params           # lnZ, c_o, lnKzz + T-P params
eye = jnp.eye(N_LANES, dtype=jnp.float64)
LANE_NAMES = ["lnZ", "c_o", "lnKzz", "Tirr", "log10kappa", "log10gamma"][:N_LANES]

DTYPE = jnp.float64


def cold_diag(th6):
    th = jnp.asarray(th6, dtype=DTYPE)
    th_relax = th.at[0].set(0.0).at[1].set(0.0)
    y1, d1 = chem.converged_y(th_relax, return_conv_diag=True)
    y2, d2 = chem.converged_y(th, warm_y=y1, lnZ_ref=0.0, c_o_ref=0.0,
                              return_conv_diag=True)
    return y2, d1, d2


def _pack_cd(cd):
    return jax.lax.stop_gradient(jnp.stack([
        jnp.asarray(cd.accept_count, DTYPE),
        jnp.asarray(cd.longdy, DTYPE),
        jnp.asarray(cd.longdydt, DTYPE),
        jnp.asarray(cd.count_since_new_min, DTYPE),
        jnp.asarray(cd.conv_normal, DTYPE)]))


def warm_jvp(th6, y_warm):
    """The SMC mutation gradient path: jvp through the warm-capped re-solve."""
    th = jnp.asarray(th6, dtype=DTYPE)
    lnZ_ref, c_o_ref = th[0], th[1]

    def _chain(c):
        y, cd = chem.converged_y(c, warm_y=y_warm, lnZ_ref=lnZ_ref,
                                 c_o_ref=c_o_ref, return_conv_diag=True,
                                 warm_cap=True)
        return y, _pack_cd(cd)

    (y_l, cd_l), (dy_l, _) = jax.vmap(lambda v: jax.jvp(_chain, (th,), (v,)))(eye)
    return y_l[0], cd_l[0], dy_l


def prep_jvp(th6):
    """Pointwise tangent of the initial-state build (no solver)."""
    th = jnp.asarray(th6, dtype=DTYPE)

    def _p(c):
        pv = chem.prep_pv(c)
        return jax.tree_util.tree_map(jnp.asarray, (pv.n_0, pv.Kzz, pv.atom_ini))

    bad = []
    for k in range(N_LANES):
        _, tangs = jax.jvp(_p, (th,), (eye[k],))
        finite = all(bool(jnp.all(jnp.isfinite(t))) for t in jax.tree_util.tree_leaves(tangs))
        bad.append(not finite)
    return bad


results = []
for lab, r in cases:
    th6 = np.asarray(r["theta"][:N_LANES], dtype=np.float64)
    log(f"=== {lab} (stage{r['stage']} sweep{r['sweep']} idx{r['idx']}) "
        f"theta6={np.array2string(th6, precision=3)}")
    rec = dict(label=lab, stage=r["stage"], sweep=r["sweep"], idx=r["idx"],
               nas_longdy=r["longdy"], nas_acc=r["acc"], theta6=th6.tolist())
    t0 = time.time()
    try:
        y2, d1, d2 = jax.block_until_ready(cold_diag(th6))
        rec["cold"] = dict(
            acc1=int(d1.accept_count), acc2=int(d2.accept_count),
            longdy=float(d2.longdy), conv_normal=bool(d2.conv_normal),
            t_s=round(time.time() - t0, 1))
        log(f"  cold: acc={rec['cold']['acc1']}/{rec['cold']['acc2']} "
            f"longdy={rec['cold']['longdy']:.3g} conv_normal={rec['cold']['conv_normal']} "
            f"({rec['cold']['t_s']}s)")

        pb = prep_jvp(th6)
        rec["prep_tangent_bad_lanes"] = [LANE_NAMES[i] for i, b in enumerate(pb) if b]
        log(f"  prep pointwise tangents: "
            f"{'ALL FINITE' if not any(pb) else 'NON-FINITE: ' + str(rec['prep_tangent_bad_lanes'])}")

        t1 = time.time()
        y_w, cd_w, dy_l = jax.block_until_ready(warm_jvp(th6, jnp.asarray(y2)))
        dy = np.asarray(dy_l)                          # (lanes, nz, ni)
        lane_finite = np.isfinite(dy).all(axis=(1, 2))
        lane_max = np.nanmax(np.abs(np.where(np.isfinite(dy), dy, np.nan)),
                             axis=(1, 2))
        nonfin_cells = (~np.isfinite(dy)).sum(axis=(1, 2))
        rec["warm"] = dict(
            acc=int(cd_w[0]), longdy=float(cd_w[1]),
            conv_normal=bool(cd_w[4] > 0.5),
            t_s=round(time.time() - t1, 1),
            lanes={LANE_NAMES[k]: dict(finite=bool(lane_finite[k]),
                                       max_abs=float(lane_max[k]),
                                       n_nonfinite=int(nonfin_cells[k]))
                   for k in range(N_LANES)})
        log(f"  warm re-solve: acc={rec['warm']['acc']} longdy={rec['warm']['longdy']:.3g} "
            f"conv_normal={rec['warm']['conv_normal']} ({rec['warm']['t_s']}s)")
        for k in range(N_LANES):
            L = rec["warm"]["lanes"][LANE_NAMES[k]]
            log(f"    lane {LANE_NAMES[k]:<11} finite={L['finite']} "
                f"max|dy|={L['max_abs']:.3e} nonfinite_cells={L['n_nonfinite']}")

        # non-finite pattern by species (union over lanes)
        nf = ~np.isfinite(dy)
        if nf.any():
            sp_counts = nf.any(axis=0).sum(axis=0)     # (ni,) layers hit per species
            hit = [(species[i], int(sp_counts[i])) for i in np.argsort(-sp_counts)
                   if sp_counts[i] > 0][:15]
            rec["nonfinite_species_top"] = hit
            log(f"  non-finite species (top, layers hit): {hit}")
            lay_counts = nf.any(axis=0).sum(axis=1)    # (nz,) species hit per layer
            rec["nonfinite_layers"] = np.flatnonzero(lay_counts).tolist()
            log(f"  non-finite layers: {rec['nonfinite_layers']}")
        np.savez(SCRATCH / f"replay_{lab}_s{r['stage']}w{r['sweep']}i{r['idx']}.npz",
                 theta6=th6, y_cold=np.asarray(y2), dy=dy,
                 y_warm=np.asarray(y_w))
    except Exception as e:
        rec["error"] = repr(e)
        log(f"  ERROR: {e!r}")
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    results.append(rec)

log("done.")
