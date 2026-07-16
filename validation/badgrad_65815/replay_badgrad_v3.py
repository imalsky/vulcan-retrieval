"""Replay v3: test the extrapolated-seed repair on the v2-reproduced cases.

v2 found the badgrad blowup reproduces ONLY with the warm_extrapolate seed
Y_seed = max(y_from + DY_from @ dC, 0)  -- the clip drives trace species to
exactly 0. Candidate repair keeps the extrapolation speedup but falls back
per-cell to the carried column wherever the linear prediction goes
non-positive:

    Y_seed = where(pred > 0, pred, y_from)

For every case in replay_v2_results.jsonl that reproduced (any extrap lane
non-finite), run three variants at the same (theta_prop, y_from, refs):
  A. max0    seed = max(pred, 0)        -- current production; expect NaN again
  B. fallback seed = where(pred>0, pred, y_from)
  C. plain   seed = y_from               -- control (known finite from v2)
Record per-lane finiteness, primal certification, accept counts.
"""
# Archived from the 2026-07-16 job-65815 investigation (docs/job65815_badgrad_investigation.md).
# Run from the repo root; optional argv[1] = replay_v2_results.jsonl path.
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
RUN_DIR = _REPO / "runs" / "w39b_smc_retrieval"
SCRATCH = _REPO / "output" / "badgrad_65815"
SCRATCH.mkdir(parents=True, exist_ok=True)
V2_JSONL = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRATCH / "replay_v2_results.jsonl"
OUT_JSONL = SCRATCH / "replay_v3_results.jsonl"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


cases = []
for line in open(V2_JSONL):
    r = json.loads(line)
    ex = r.get("extrap", {})
    if ex and any(not v["finite"] for v in ex.values()):
        cases.append(r)
log(f"{len(cases)} reproduced case(s) from v2")
for r in cases:
    log(f"  {r['kind']} s{r['stage']}w{r['sweep']}i{r['idx']} "
        f"prop lnZ={r['theta_prop'][0]:+.2f} c_o={r['theta_prop'][1]:+.2f}")
if not cases:
    log("nothing reproduced; nothing to test")
    sys.exit(0)

log("building chemistry model...")
from retrieval_framework.run_smc import load_case

case_mod = load_case(RUN_DIR)
cfg = case_mod.PRESETS["gpu"]()
profile = cfg.profile()

t0 = time.time()
from retrieval_framework.forward import vulcan_chem
from retrieval_framework import tp_profile
import jax
import jax.numpy as jnp

tpm = tp_profile.build_tp_model(cfg)
chem = vulcan_chem.build_chem_model(profile, tp_eval=tpm.eval, n_tp_params=tpm.n_params)
log(f"chem model built in {time.time() - t0:.1f}s")

N_LANES = 3 + tpm.n_params
eye = jnp.eye(N_LANES, dtype=jnp.float64)
LANE_NAMES = ["lnZ", "c_o", "lnKzz", "Tirr", "log10kappa", "log10gamma"][:N_LANES]
DTYPE = jnp.float64


def cold(th6):
    th = jnp.asarray(th6, dtype=DTYPE)
    th_relax = th.at[0].set(0.0).at[1].set(0.0)
    y1, _ = chem.converged_y(th_relax, return_conv_diag=True)
    y2, d2 = chem.converged_y(th, warm_y=y1, lnZ_ref=0.0, c_o_ref=0.0,
                              return_conv_diag=True)
    return y2, d2


def _pack_cd(cd):
    return jax.lax.stop_gradient(jnp.stack([
        jnp.asarray(cd.accept_count, DTYPE),
        jnp.asarray(cd.longdy, DTYPE),
        jnp.asarray(cd.longdydt, DTYPE),
        jnp.asarray(cd.count_since_new_min, DTYPE),
        jnp.asarray(cd.conv_normal, DTYPE)]))


def move_jvp(th6, y_seed, refs):
    th = jnp.asarray(th6, dtype=DTYPE)

    def _chain(c):
        y, cd = chem.converged_y(c, warm_y=y_seed, lnZ_ref=refs[0],
                                 c_o_ref=refs[1], return_conv_diag=True,
                                 warm_cap=True)
        return y, _pack_cd(cd)

    (y_l, cd_l), (dy_l, _) = jax.vmap(lambda v: jax.jvp(_chain, (th,), (v,)))(eye)
    return cd_l[0], np.asarray(dy_l)


def dy_at(th6, y_warm):
    th = jnp.asarray(th6, dtype=DTYPE)

    def _chain(c):
        y, cd = chem.converged_y(c, warm_y=y_warm, lnZ_ref=th6[0],
                                 c_o_ref=th6[1], return_conv_diag=True,
                                 warm_cap=True)
        return y, _pack_cd(cd)

    (y_l, _cd), (dy_l, _) = jax.vmap(lambda v: jax.jvp(_chain, (th,), (v,)))(eye)
    return y_l[0], dy_l


for r in cases:
    thp = np.asarray(r["theta_prop"][:N_LANES], np.float64)
    thf = np.asarray(r["theta_from"][:N_LANES], np.float64)
    tag = f"{r['kind']}_s{r['stage']}w{r['sweep']}i{r['idx']}"
    log(f"=== {tag}")
    rec = dict(case=tag, theta_prop=thp.tolist(), theta_from=thf.tolist())
    y_from, d_from = jax.block_until_ready(cold(thf))
    y_ff, DY_from = jax.block_until_ready(dy_at(thf, jnp.asarray(y_from)))
    dC = jnp.asarray(thp - thf)
    pred = jnp.asarray(y_ff) + jnp.einsum("kij,k->ij", DY_from, dC)
    n_nonpos = int(jnp.sum(pred <= 0.0))
    rec["pred_nonpos_cells"] = n_nonpos
    log(f"  extrapolated prediction has {n_nonpos} non-positive cells "
        f"(these get clipped to 0 in production)")

    seeds = dict(
        max0=jnp.maximum(pred, 0.0),
        fallback=jnp.where(pred > 0.0, pred, jnp.asarray(y_ff)),
        plain=jnp.asarray(y_from),
    )
    refs = (jnp.asarray(thf[0]), jnp.asarray(thf[1]))
    for name, seed in seeds.items():
        t1 = time.time()
        cd, dy = jax.block_until_ready(move_jvp(thp, seed, refs))
        finite = bool(np.isfinite(dy).all())
        rec[name] = dict(finite=finite, acc=int(cd[0]), longdy=float(cd[1]),
                         conv_normal=bool(cd[4] > 0.5),
                         n_nonfinite=int((~np.isfinite(dy)).sum()),
                         t_s=round(time.time() - t1, 1))
        log(f"  seed={name:<8} tangents_finite={finite} acc={int(cd[0])} "
            f"longdy={float(cd[1]):.3g} conv={bool(cd[4] > 0.5)} "
            f"nonfinite_cells={(~np.isfinite(dy)).sum()} ({rec[name]['t_s']}s)")
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")

log("v3 done.")
