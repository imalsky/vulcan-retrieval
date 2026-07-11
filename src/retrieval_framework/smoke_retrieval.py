#!/usr/bin/env python3
"""smoke_retrieval.py -- offline end-to-end validation of the retrieval gradient.

Builds the smoke pipeline (CO-only cached opacity, nz=30, photochemistry ON), injects
synthetic observations, then:

  1. asserts the BLOCK-structured likelihood gradient == the NAIVE all-dims
     forward-mode gradient (they are algebraically identical; this catches wiring
     bugs in the block assembly),
  2. validates the gradient against a central finite difference of the re-converged
     likelihood, dimension by dimension (the same check the parent smoke_test.py
     runs for the sensitivity demo),
  3. asserts the STAGED batched evaluator (chemistry fwd-jvp lanes + ONE RT vjp,
     lax.map-chunked -- the SMC hot path) == the per-particle block gradient, and
  4. FD-checks the WARM-continuation gradient (the mutation-kernel map: re-converge
     from a carried column with incremental lnZ/C-O) against central differences of
     the same warm map.

Run it in the vulcan env before trusting any retrieval output (uses the case's
"smoke" preset unless SMC_RETRIEVAL_PRESET says otherwise):

    python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval

Exit code 0 = all checks passed. Takes ~10-30 min on a laptop CPU (each FD point
re-converges the VULCAN column).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

from retrieval_framework import run_smc as R
from retrieval_framework import pipeline as P
import jax
import jax.numpy as jnp


def main() -> int:
    t_all = time.time()
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    os.environ.setdefault("SMC_RETRIEVAL_PRESET", "smoke")
    cfg, _preset = R.make_config(run_dir)
    from retrieval_framework import config_schema as C
    print(C.describe_config(cfg, f"{_preset}+SMOKE"), flush=True)
    pipe = P.build_pipeline(cfg)
    print(f"[smoke] pipeline built | n_dim={pipe.n_dim} params={pipe.names} "
          f"bins={pipe.n_bin} gradient_mode={pipe.gradient_mode}", flush=True)

    P.generate_observations(pipe, seed=cfg.seed)
    print("[smoke] synthetic observations injected", flush=True)

    u0 = jnp.asarray(np.linspace(-0.35, 0.4, pipe.n_dim))   # generic off-center point

    # ---- likelihood primal + timing ----
    t0 = time.time()
    L0 = float(pipe.log_likelihood_u(u0))
    t_primal = time.time() - t0
    print(f"[smoke] L(u0) = {L0:.4f}   ({t_primal:.1f} s primal, includes compile)", flush=True)
    assert np.isfinite(L0), "likelihood non-finite at the prior center"

    # ---- block vs naive gradient (must agree to fp precision) ----
    t0 = time.time()
    vb, gb = pipe.value_and_grad_block(u0)
    gb = np.asarray(gb); t_block = time.time() - t0
    t0 = time.time()
    vn, gn = pipe.value_and_grad_naive(u0)
    gn = np.asarray(gn); t_naive = time.time() - t0
    print(f"[smoke] value block={float(vb):.6f} naive={float(vn):.6f} "
          f"| t_block={t_block:.1f}s t_naive={t_naive:.1f}s", flush=True)
    ok_val = abs(float(vb) - float(vn)) <= 1e-8 * max(1.0, abs(float(vn)))
    denom = np.maximum(np.abs(gn), 1e-12 * np.max(np.abs(gn)) + 1e-30)
    rel_bn = np.max(np.abs(gb - gn) / denom)
    ok_bn = bool(rel_bn < 1e-6) and ok_val
    print(f"[smoke] block-vs-naive max rel diff = {rel_bn:.2e}  -> {'OK' if ok_bn else 'FAIL'}",
          flush=True)
    for i, nm in enumerate(pipe.names):
        print(f"    {nm:12s} block={gb[i]:+12.5e}  naive={gn[i]:+12.5e}", flush=True)

    # ---- FD validation of the gradient (re-converged central differences) ----
    h = 1e-3
    print(f"[smoke] central FD check (h={h:g}, 2 re-converged solves per dim)...", flush=True)
    ok_fd = True
    gmax = np.max(np.abs(gb))
    for i in range(pipe.n_dim):
        e = np.zeros(pipe.n_dim); e[i] = h
        t0 = time.time()
        Lp = float(pipe.log_likelihood_u(u0 + jnp.asarray(e)))
        Lm = float(pipe.log_likelihood_u(u0 - jnp.asarray(e)))
        fd = (Lp - Lm) / (2 * h)
        ad = gb[i]
        rel = abs(ad - fd) / max(abs(fd), 1e-12)
        # weak directions: absolute agreement relative to the dominant gradient scale
        ok_i = (rel < 5e-2) or (abs(ad - fd) < 1e-4 * gmax)
        ok_fd &= ok_i
        print(f"    {pipe.names[i]:12s} ad={ad:+12.5e}  fd={fd:+12.5e}  rel={rel:.2e} "
              f"[{time.time()-t0:.0f}s]  {'OK' if ok_i else 'FAIL'}", flush=True)

    # ---- staged batched evaluator (SMC hot path) vs per-particle block gradient ----
    # Same chain rule, regrouped (fwd-jvp chemistry lanes contracted against ONE
    # reverse-mode RT vjp, RT lax.map-chunked); must agree to fp precision.
    t0 = time.time()
    du = jnp.asarray(np.linspace(-0.06, 0.09, pipe.n_dim))
    U_test = jnp.stack([u0, u0 + du, u0 - du])
    Y0, refs0 = P._blank_state(pipe, int(U_test.shape[0]))
    Lb, Gb2, Yb, refsb, nbad_b, _dy, _ncap = jax.jit(pipe.batch_eval_cold_vg)(U_test, Y0, refs0)
    assert int(nbad_b) == 0, "staged cold eval flagged gradient pathologies"
    Lb = np.asarray(Lb); Gb2 = np.asarray(Gb2)
    ok_staged = True
    for r in range(int(U_test.shape[0])):
        vr, gr = pipe.value_and_grad_block(U_test[r])
        vr = float(vr); gr = np.asarray(gr)
        dv = abs(Lb[r] - vr) / max(1.0, abs(vr))
        dg = np.max(np.abs(Gb2[r] - gr) / np.maximum(np.abs(gr), 1e-12 * np.max(np.abs(gr)) + 1e-30))
        ok_r = (dv < 1e-8) and (dg < 1e-6)
        ok_staged &= ok_r
        print(f"[smoke] staged-vs-block row {r}: dval={dv:.2e} dgrad={dg:.2e} "
              f"{'OK' if ok_r else 'FAIL'}", flush=True)
    print(f"[smoke] staged batched evaluator check done [{time.time()-t0:.0f}s] "
          f"-> {'OK' if ok_staged else 'FAIL'}", flush=True)
    assert np.all(np.isfinite(Yb)) and np.asarray(refsb).shape == (int(U_test.shape[0]), 2)

    # ---- warm-continuation gradient (the mutation-kernel map) vs FD of the same map ----
    # State = the converged columns from the cold batch above; evaluate the move
    # gradient at a DIFFERENT point (a realistic MCMC proposal) and FD the identical
    # warm map (fixed carried state) -- validates that the tangent relaxes through
    # the warm-started while_loop (the continuation-jvp pattern).
    t0 = time.time()
    u1 = U_test[0] + 0.5 * du
    U1 = u1[None, :]
    Y_w, refs_w = Yb[:1], refsb[:1]
    move_vg = jax.jit(pipe.batch_eval_move_vg)
    move_l = jax.jit(pipe.batch_eval_move_l)
    L1, G1, _, _, nbad_w, _dyw, _ncapw = move_vg(U1, Y_w, refs_w)
    assert int(nbad_w) == 0, "warm move eval flagged gradient pathologies"
    g_warm = np.asarray(G1[0])
    ok_warm = True
    gmax_w = np.max(np.abs(g_warm))
    for i in range(pipe.n_dim):
        e = np.zeros(pipe.n_dim); e[i] = h
        Lp = float(move_l(U1 + jnp.asarray(e)[None, :], Y_w, refs_w)[0][0])
        Lm = float(move_l(U1 - jnp.asarray(e)[None, :], Y_w, refs_w)[0][0])
        fd = (Lp - Lm) / (2 * h)
        ad = g_warm[i]
        rel = abs(ad - fd) / max(abs(fd), 1e-12)
        ok_i = (rel < 5e-2) or (abs(ad - fd) < 1e-4 * gmax_w)
        ok_warm &= ok_i
        print(f"    warm {pipe.names[i]:12s} ad={ad:+12.5e}  fd={fd:+12.5e}  rel={rel:.2e} "
              f"{'OK' if ok_i else 'FAIL'}", flush=True)
    print(f"[smoke] warm-continuation gradient FD check [{time.time()-t0:.0f}s] "
          f"-> {'OK' if ok_warm else 'FAIL'}", flush=True)

    # ---- inventory-response liveness (regression guard for the 2026-07-05 finding:
    # perturbing the cold EQ init under a retrieved T-P erased the lnZ/c_o response;
    # the two-stage solve restores it -- these gradients must be alive, not ~1e-20) ----
    ok_live = True
    for nm in ("lnZ", "c_o"):
        if nm in pipe.names:
            gi = abs(gb[pipe.names.index(nm)])
            alive = gi > 1e-3
            ok_live &= alive
            print(f"[smoke] liveness {nm:4s}: |dL/d{nm}|={gi:.3e}  "
                  f"{'OK' if alive else 'FAIL (inventory response dead -- check two_stage_z)'}",
                  flush=True)

    ok = ok_bn and ok_fd and ok_staged and ok_warm and ok_live
    print(f"[smoke] TOTAL {time.time()-t_all:.0f}s  ->  {'ALL CHECKS PASSED' if ok else 'FAILURES'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
