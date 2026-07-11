#!/usr/bin/env python3
"""probe_memory.py -- compile-only XLA memory analysis of each stage of the staged
SMC evaluator, on the current preset/overrides.

Every case is jit-lower()ed and compile()d but NEVER EXECUTED, so there is no OOM
risk: XLA's buffer assignment (the same estimate behind the
"Can't reduce memory use below ..." rematerialization warnings) is printed per
case, at several batch widths. One job pinpoints which stage owns the peak and
how it scales.

Run on the GH200 via:  qsub -l walltime=02:00:00 -v PROBE_MEMORY=1 run_nas_w39b.pbs
"""
from __future__ import annotations

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
    if cfg.generate_synthetic_data:
        P.generate_observations(pipe, seed=int(cfg.seed))
    else:
        P.load_real_into_pipe(pipe)
    print(f"[probe] pipeline built in {time.time()-t0:.0f}s", flush=True)

    N = int(cfg.smc_num_particles)
    key = jax.random.PRNGKey(0)
    U = pipe.sample_prior_u(key, N)
    Y0, refs0 = P._blank_state(pipe, N)
    fwd = pipe.fwd
    n_chem_tp = int(pipe.n_chem_tp)
    dtype = pipe.dtype
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

    Theta = jax.vmap(pipe.theta_from_u)(U)
    C_full = Theta[:, :n_chem_tp]
    eye_c = jnp.eye(n_chem_tp, dtype=dtype)

    # ---- chemistry GRADIENT stage (cold two-stage solve + n_chem_tp jvp lanes) ----
    def chem_grad(Cc, Yc, Rc):
        def one(cc, yw, rf):
            def _chain(c):
                y = fwd.chem_solve_cold(c)
                return fwd.aux_from_y(y, c), y
            (aux_l, y_l), (daux_l, _dy) = jax.vmap(
                lambda v: jax.jvp(_chain, (cc,), (v,)))(eye_c)
            return jax.tree_util.tree_map(lambda x: x[0], aux_l), daux_l, y_l[0]
        return jax.vmap(one)(Cc, Yc, Rc)

    for w in (1, 2, 6):
        report(f"chem GRAD x{w} particles ({w*n_chem_tp} jvp lanes)",
               chem_grad, C_full[:w], Y0[:w], refs0[:w])

    # ---- chemistry PRIMAL, full width (the known-fits reference) ----
    def chem_primal(Cc, Yc, Rc):
        def one(cc, yw, rf):
            y = fwd.chem_solve_cold(cc)
            return fwd.aux_from_y(y, cc), y
        return jax.vmap(one)(Cc, Yc, Rc)

    report(f"chem PRIMAL x{N} particles", chem_primal, C_full, Y0, refs0)

    # ---- RT stage alone (abstract inputs; vjp with unit cotangent) ----
    nl = int(cfg.art_nlayer)
    mols = list(fwd.rt.molecules)
    use_clouds = bool(cfg.use_clouds)

    def rt_vjp(auxb, r0b, cpb):
        def one(aux, r0, cp):
            depth, vjp_fn = jax.vjp(
                lambda a, r, c: fwd.rt_depth(a, r, c if use_clouds else None),
                aux, r0, cp)
            bars = vjp_fn(jnp.ones_like(depth))
            return jnp.sum(depth), jax.tree_util.tree_map(jnp.sum, bars)
        return jax.vmap(one)(auxb, r0b, cpb)

    def rt_primal(auxb, r0b, cpb):
        def one(aux, r0, cp):
            return jnp.sum(fwd.rt_depth(aux, r0, cp if use_clouds else None))
        return jax.vmap(one)(auxb, r0b, cpb)

    def _aux_sds(w):
        f8 = np.float64
        return ({m: jax.ShapeDtypeStruct((w, nl), f8) for m in mols},
                jax.ShapeDtypeStruct((w, nl), f8),
                jax.ShapeDtypeStruct((w, nl), f8),
                jax.ShapeDtypeStruct((w, nl), f8),
                jax.ShapeDtypeStruct((w, nl), f8))

    for w in (1, 6):
        report(f"RT VJP x{w} particles", rt_vjp, _aux_sds(w),
               jax.ShapeDtypeStruct((w,), np.float64),
               jax.ShapeDtypeStruct((w, 2), np.float64))
    report("RT PRIMAL x16 particles", rt_primal, _aux_sds(16),
           jax.ShapeDtypeStruct((16,), np.float64),
           jax.ShapeDtypeStruct((16, 2), np.float64))

    # ---- the full staged evaluators exactly as the SMC uses them ----
    report(f"FULL cold_vg (chem_chunk={cfg.smc_chem_chunk}, "
           f"rt_vjp_chunk={cfg.smc_rt_vjp_chunk})",
           pipe.batch_eval_cold_vg, U, Y0, refs0)
    # init phase 2 runs at N + init_phase2_spare width -- the WIDEST gradient eval
    # in the run (the mutation kernel matches cold_vg at width N)
    n2 = N + int(cfg.init_phase2_spare)
    U2 = pipe.sample_prior_u(jax.random.PRNGKey(1), n2)
    Y2, refs2 = P._blank_state(pipe, n2)
    report(f"FULL init_vg x{n2} (N+{int(cfg.init_phase2_spare)} phase-2 spares)",
           pipe.batch_eval_init_vg, U2, Y2, refs2)
    report("FULL cold_l (primal likelihood batch)",
           pipe.batch_eval_cold_l, U, Y0, refs0)

    print("\n========== MEMORY PROBE SUMMARY ==========")
    for ln in lines:
        print(ln)
    print("(pool budget on a 96 GB GH200 at MEM_FRACTION=0.90 is ~81 GiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
