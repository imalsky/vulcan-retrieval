"""Replay v2: reproduce the MUTATION-move structure for offender thetas.

The v1 zero-increment re-certification produced finite tangents everywhere,
matching the field observation that init phase 2 (same structure) rarely blows.
The NAS badgrad events all occur on real MALA moves: warm start from the
particle's carried column at theta_from, solve at theta_prop != theta_from.

Per case:
  theta_from = the SAME particle's proposal in a DIFFERENT sweep of the same
  stage when available (realistic move spacing), else theta_prop + 0.5*cloud_std
  along a fixed sign pattern.
  1. cold two-stage solve at theta_from -> y_from (certified carried column)
  2. jvp (6 lanes) of the warm-capped solve at theta_prop, warm_y=y_from,
     refs=(theta_from[0], theta_from[1])  [seed = plain carried column]
  3. same but seeded at the first-order extrapolation
     Y_seed = max(y_from + DY_from @ (c_prop - c_from), 0)  [warm_extrapolate=on
     path]; DY_from from a 6-lane jvp at theta_from (zero-increment re-cert map,
     shown finite in v1).
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
OUT_JSONL = SCRATCH / "replay_v2_results.jsonl"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


import glob
import re

dumps = {}
for fp in sorted(glob.glob(str(FOR_DIR / "bad_grad_stage*.npz"))):
    m = re.search(r"stage(\d+)_sweep(\d+)", fp)
    dumps[(int(m.group(1)), int(m.group(2)))] = dict(np.load(fp))

# offenders with a same-particle proposal in another sweep of the same stage
cases = []
for (st, sw), d in sorted(dumps.items()):
    for i in np.flatnonzero(d["bad_grad"]):
        for (st2, sw2), d2 in sorted(dumps.items()):
            if st2 == st and sw2 != sw:
                th_from = d2["theta_proposal"][i]
                th_prop = d["theta_proposal"][i]
                if np.any(th_from != th_prop):
                    cases.append(dict(kind="off", stage=st, sweep=sw, idx=int(i),
                                      longdy=float(d["longdy"][i]),
                                      theta_prop=th_prop, theta_from=th_from,
                                      from_sweep=sw2))
                    break
        else:
            continue

# dedupe by (stage, idx) keep first; cap at 8 offenders spread over stages
seen = set()
off_cases = []
for c in cases:
    k = (c["stage"], c["idx"])
    if k in seen:
        continue
    seen.add(k)
    off_cases.append(c)
by_stage = {}
sel_off = []
for c in off_cases:
    by_stage.setdefault(c["stage"], []).append(c)
for st in sorted(by_stage):
    sel_off.extend(by_stage[st][:3])
sel_off = sel_off[:8]

# controls: certified non-offenders, same move structure, spread over lnZ
ctl = []
for (st, sw), d in sorted(dumps.items()):
    conv = d["conv_ok"].astype(bool) & ~d["bad_grad"].astype(bool)
    for i in np.flatnonzero(conv):
        for (st2, sw2), d2 in sorted(dumps.items()):
            if st2 == st and sw2 != sw:
                th_from = d2["theta_proposal"][i]
                th_prop = d["theta_proposal"][i]
                if np.any(th_from != th_prop):
                    ctl.append(dict(kind="ctl", stage=st, sweep=sw, idx=int(i),
                                    longdy=float(d["longdy"][i]),
                                    theta_prop=th_prop, theta_from=th_from,
                                    from_sweep=sw2))
                    break
        break  # one control candidate scan per dump is enough
    if len(ctl) >= 12:
        break
# pick 4 controls spanning lnZ
ctl_sorted = sorted(ctl, key=lambda c: c["theta_prop"][0])
sel_ctl = [ctl_sorted[0], ctl_sorted[len(ctl_sorted) // 3],
           ctl_sorted[2 * len(ctl_sorted) // 3], ctl_sorted[-1]] if len(ctl_sorted) >= 4 else ctl_sorted

CASES = sel_off + sel_ctl
log(f"{len(sel_off)} offender + {len(sel_ctl)} control move-replays selected")
for c in CASES:
    dmove = c["theta_prop"][:6] - c["theta_from"][:6]
    log(f"  {c['kind']} s{c['stage']}w{c['sweep']}i{c['idx']} longdy={c['longdy']:.3g} "
        f"lnZ={c['theta_prop'][0]:+.2f} c_o={c['theta_prop'][1]:+.2f} "
        f"|dmove|={np.linalg.norm(dmove):.3f}")

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
species = sorted(chem.sidx, key=chem.sidx.get)


def cold_diag(th6):
    th = jnp.asarray(th6, dtype=DTYPE)
    th_relax = th.at[0].set(0.0).at[1].set(0.0)
    y1, d1 = chem.converged_y(th_relax, return_conv_diag=True)
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


def warm_move_jvp(th_prop6, y_seed, refs):
    th = jnp.asarray(th_prop6, dtype=DTYPE)

    def _chain(c):
        y, cd = chem.converged_y(c, warm_y=y_seed, lnZ_ref=refs[0],
                                 c_o_ref=refs[1], return_conv_diag=True,
                                 warm_cap=True)
        return y, _pack_cd(cd)

    (y_l, cd_l), (dy_l, _) = jax.vmap(lambda v: jax.jvp(_chain, (th,), (v,)))(eye)
    return y_l[0], cd_l[0], dy_l


def dy_at(th6, y_warm):
    """Converged-column tangents at th6 from its own column (the DY the pipeline
    carries for warm_extrapolate)."""
    th = jnp.asarray(th6, dtype=DTYPE)

    def _chain(c):
        y, cd = chem.converged_y(c, warm_y=y_warm, lnZ_ref=th6[0],
                                 c_o_ref=th6[1], return_conv_diag=True,
                                 warm_cap=True)
        return y, _pack_cd(cd)

    (y_l, _cd), (dy_l, _) = jax.vmap(lambda v: jax.jvp(_chain, (th,), (v,)))(eye)
    return y_l[0], dy_l


def lane_report(dy, tag, rec):
    dy = np.asarray(dy)
    lane_finite = np.isfinite(dy).all(axis=(1, 2))
    lane_max = np.nanmax(np.abs(np.where(np.isfinite(dy), dy, np.nan)), axis=(1, 2))
    nonfin = (~np.isfinite(dy)).sum(axis=(1, 2))
    rec[tag] = {LANE_NAMES[k]: dict(finite=bool(lane_finite[k]),
                                    max_abs=float(lane_max[k]),
                                    n_nonfinite=int(nonfin[k]))
                for k in range(N_LANES)}
    for k in range(N_LANES):
        log(f"    [{tag}] lane {LANE_NAMES[k]:<11} finite={bool(lane_finite[k])} "
            f"max|dy|={lane_max[k]:.3e} nonfinite={int(nonfin[k])}")
    nf = ~np.isfinite(dy)
    if nf.any():
        sp_counts = nf.any(axis=0).sum(axis=0)
        hit = [(species[i], int(sp_counts[i])) for i in np.argsort(-sp_counts)
               if sp_counts[i] > 0][:12]
        rec[tag + "_nonfinite_species"] = hit
        log(f"    [{tag}] non-finite species (layers hit): {hit}")
    return not bool(nf.any())


for c in CASES:
    thp = np.asarray(c["theta_prop"][:N_LANES], np.float64)
    thf = np.asarray(c["theta_from"][:N_LANES], np.float64)
    log(f"=== {c['kind']} s{c['stage']}w{c['sweep']}i{c['idx']} "
        f"prop lnZ={thp[0]:+.2f} c_o={thp[1]:+.2f} | from lnZ={thf[0]:+.2f} c_o={thf[1]:+.2f}")
    rec = dict(**{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in c.items()})
    t0 = time.time()
    try:
        y_from, d_from = jax.block_until_ready(cold_diag(thf))
        rec["from_cold"] = dict(acc=int(d_from.accept_count),
                                longdy=float(d_from.longdy),
                                conv_normal=bool(d_from.conv_normal))
        log(f"  from-state cold: acc={rec['from_cold']['acc']} "
            f"longdy={rec['from_cold']['longdy']:.3g} "
            f"conv={rec['from_cold']['conv_normal']} ({time.time() - t0:.0f}s)")
        if not rec["from_cold"]["conv_normal"]:
            log("  from-state did not certify; skipping case")
            rec["skipped"] = "from-state not certified"
            raise StopIteration

        # variant A: plain warm start (warm_extrapolate off)
        t1 = time.time()
        refs = (jnp.asarray(thf[0]), jnp.asarray(thf[1]))
        y_a, cd_a, dy_a = jax.block_until_ready(
            warm_move_jvp(thp, jnp.asarray(y_from), refs))
        rec["move_plain"] = dict(acc=int(cd_a[0]), longdy=float(cd_a[1]),
                                 conv_normal=bool(cd_a[4] > 0.5),
                                 t_s=round(time.time() - t1, 1))
        log(f"  move (plain seed): acc={rec['move_plain']['acc']} "
            f"longdy={rec['move_plain']['longdy']:.3g} "
            f"conv={rec['move_plain']['conv_normal']}")
        ok_a = lane_report(dy_a, "plain", rec)

        # variant B: extrapolated seed (warm_extrapolate on, the NAS config)
        t1 = time.time()
        y_ff, DY_from = jax.block_until_ready(dy_at(thf, jnp.asarray(y_from)))
        DY_np = np.asarray(DY_from)
        rec["DY_from_finite"] = bool(np.isfinite(DY_np).all())
        log(f"  DY at from-state finite: {rec['DY_from_finite']}")
        if rec["DY_from_finite"]:
            dC = jnp.asarray(thp - thf)
            y_seed = jnp.maximum(
                jnp.asarray(y_ff) + jnp.einsum("kij,k->ij", DY_from, dC), 0.0)
            y_b, cd_b, dy_b = jax.block_until_ready(warm_move_jvp(thp, y_seed, refs))
            rec["move_extrap"] = dict(acc=int(cd_b[0]), longdy=float(cd_b[1]),
                                      conv_normal=bool(cd_b[4] > 0.5),
                                      t_s=round(time.time() - t1, 1))
            log(f"  move (extrap seed): acc={rec['move_extrap']['acc']} "
                f"longdy={rec['move_extrap']['longdy']:.3g} "
                f"conv={rec['move_extrap']['conv_normal']}")
            ok_b = lane_report(dy_b, "extrap", rec)
            np.savez(SCRATCH / f"replayv2_{c['kind']}_s{c['stage']}w{c['sweep']}i{c['idx']}.npz",
                     theta_prop=thp, theta_from=thf, y_from=np.asarray(y_from),
                     dy_plain=np.asarray(dy_a), dy_extrap=np.asarray(dy_b))
    except StopIteration:
        pass
    except Exception as e:
        rec["error"] = repr(e)
        log(f"  ERROR: {e!r}")
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")

log("v2 done.")
