#!/usr/bin/env python3
"""probe_memory.py -- compile-only XLA memory analysis of each stage of the staged
SMC evaluator, on the current preset/overrides.

Every probed case is jit-lower()ed and compile()d but never executed (only the
pipeline build and the synthetic observation run one forward), so there is no
OOM risk from the cases themselves: XLA's buffer assignment (the same estimate behind the
"Can't reduce memory use below ..." rematerialization warnings) is printed per
case, at the widths the run actually uses. One job pinpoints which stage owns
the peak and how it scales. Exit status is nonzero if any case failed to
compile, so a wrapper cannot mistake a half-run probe for a certificate.

Run on the GH200 via:  qsub -l walltime=02:00:00 -v PROBE_MEMORY=1 run_nas_w39b.pbs
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from retrieval_framework import run_smc as R


def _gib(x) -> str:
    try:
        return f"{float(x) / 2**30:9.2f}"
    except Exception:
        return "      ???"


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    cfg, _preset = R.make_config(run_dir)
    from retrieval_framework import pipeline as P
    from retrieval_framework import config_schema as C
    import jax
    import jax.numpy as jnp

    print(C.describe_config(cfg, f"{_preset}+PROBE_MEMORY"), flush=True)
    t0 = time.time()
    pipe = P.build_pipeline(cfg)
    R.set_observations(cfg, pipe, P)
    print(f"[probe] pipeline built in {time.time()-t0:.0f}s", flush=True)

    N = int(cfg.smc_num_particles)
    # init phase 1 evaluates ceil(N * init_oversample) cold draws in ONE primal
    # batch (pipeline._init_state) -- the widest primal width in the run
    n1 = math.ceil(N * float(cfg.init_oversample))
    key = jax.random.PRNGKey(0)
    U = pipe.sample_prior_u(key, N)
    Y0, refs0 = P._blank_state(pipe, N)
    U1 = pipe.sample_prior_u(jax.random.PRNGKey(2), n1)
    Y1, refs1 = P._blank_state(pipe, n1)
    fwd = pipe.fwd
    header = f"{'case':<44s} {'temp GiB':>9s} {'args GiB':>9s} {'out GiB':>9s}"
    lines = [header, "-" * len(header)]
    print(header, flush=True)

    def report(name, fn, *args):
        t1 = time.time()
        try:
            comp = jax.jit(fn).lower(*args).compile()
            ma = comp.memory_analysis()
            line = (f"{name:<44s} {_gib(getattr(ma, 'temp_size_in_bytes', -1))} "
                    f"{_gib(getattr(ma, 'argument_size_in_bytes', -1))} "
                    f"{_gib(getattr(ma, 'output_size_in_bytes', -1))} "
                    f"[{time.time()-t1:.0f}s compile]")
        except Exception as e:
            line = f"{name:<44s} ERROR {type(e).__name__}: {e}"
        print(line, flush=True)
        lines.append(line)

    # ---- RT stage alone (abstract inputs; vjp with unit cotangent) ----
    nl = int(cfg.art_nlayer)
    mols = list(fwd.rt.molecules)

    def rt_vjp(auxb, r0b, cpb):
        def one(aux, r0, cp):
            depth, vjp_fn = jax.vjp(fwd.rt_depth, aux, r0, cp)
            bars = vjp_fn(jnp.ones_like(depth))
            return jnp.sum(depth), jax.tree_util.tree_map(jnp.sum, bars)
        return jax.vmap(one)(auxb, r0b, cpb)

    def rt_primal(auxb, r0b, cpb):
        def one(aux, r0, cp):
            return jnp.sum(fwd.rt_depth(aux, r0, cp))
        return jax.vmap(one)(auxb, r0b, cpb)

    def _aux_sds(w):
        f8 = np.float64
        return ({m: jax.ShapeDtypeStruct((w, nl), f8) for m in mols},
                jax.ShapeDtypeStruct((w, nl), f8),
                jax.ShapeDtypeStruct((w, nl), f8),
                jax.ShapeDtypeStruct((w, nl), f8),
                jax.ShapeDtypeStruct((w, nl), f8))

    # the isolated rungs must include the PRODUCTION widths -- hardcoded rungs
    # once left smc_rt_vjp_chunk=12 unprobed while the config comment claimed
    # certification
    for w in sorted({1, int(cfg.smc_rt_vjp_chunk)}):
        report(f"RT VJP x{w} particles", rt_vjp, _aux_sds(w),
               jax.ShapeDtypeStruct((w,), np.float64),
               jax.ShapeDtypeStruct((w, 2), np.float64))
    rt_pw = int(cfg.smc_rt_chunk) or n1        # 0 = unchunked = the whole phase-1 batch
    report(f"RT PRIMAL x{rt_pw} particles", rt_primal, _aux_sds(rt_pw),
           jax.ShapeDtypeStruct((rt_pw,), np.float64),
           jax.ShapeDtypeStruct((rt_pw, 2), np.float64))

    # ---- the full staged evaluators exactly as the SMC uses them ----
    report(f"FULL cold_vg (rt_vjp_chunk={cfg.smc_rt_vjp_chunk})",
           pipe.batch_eval_cold_vg, U, Y0, refs0)
    # init phase 2 runs at N + init_phase2_spare width -- the WIDEST gradient eval
    # in the run (the mutation kernel matches cold_vg at width N)
    n2 = N + int(cfg.init_phase2_spare)
    U2 = pipe.sample_prior_u(jax.random.PRNGKey(1), n2)
    Y2, refs2 = P._blank_state(pipe, n2)
    report(f"FULL init_vg x{n2} (N+{int(cfg.init_phase2_spare)} phase-2 spares)",
           pipe.batch_eval_init_vg, U2, Y2, refs2)
    report(f"FULL cold_l x{n1} (init phase-1 primal likelihood batch)",
           pipe.batch_eval_cold_l, U1, Y1, refs1)

    print("\n========== MEMORY PROBE SUMMARY ==========")
    for ln in lines:
        print(ln)
    print("(pool budget on a 96 GB GH200 at MEM_FRACTION=0.90 is ~81 GiB)", flush=True)
    failed = [ln for ln in lines if " ERROR " in ln]
    if failed:
        print(f"PROBE FAILED: {len(failed)} case(s) did not compile -- no memory "
              "certificate from this run", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
