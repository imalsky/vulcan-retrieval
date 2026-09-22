#!/usr/bin/env python3
"""smoke_retrieval.py -- offline end-to-end validation of the retrieval gradient.

Builds the smoke pipeline (CO-only cached opacity, nz=30, photochemistry ON), injects
synthetic observations, then:

  1. asserts the BLOCK-structured likelihood gradient == the NAIVE all-dims
     forward-mode gradient (they are algebraically identical; this catches wiring
     bugs in the block assembly),
  2. validates the gradient against a central finite difference of the re-converged
     likelihood, dimension by dimension (the same check jax_paper's
     scripts/retrieval/smoke_test.py runs for the sensitivity demo),
  3. asserts the STAGED batched evaluator (chemistry fwd-jvp lanes + ONE RT vjp,
     lax.map-chunked -- the SMC hot path) == the per-particle block gradient, to
     fp precision at cold_lanes=0 and to the convergence-scale standard
     (validate_warm.DLOGL_MAX_PASS) when cold_lanes>0 puts the staged side on the
     lane queue and the two become different maps; the regime is printed, and
  4. FD-checks the WARM-continuation gradient (the mutation-kernel map: re-converge
     from a carried column with incremental lnZ/C-O) against central differences of
     the same warm map. That pair is built explicitly for chem_mode "warm":
     `smc_chem_mode` defaults to "cold", so the pipeline's own move evaluators
     would be the cold ones and this check would never touch the warm map.

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
          f"bins={pipe.n_bin}", flush=True)

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
    # Block and naive are the SAME chain rule regrouped, so any difference is
    # floating-point accumulation and the right yardstick is the gradient's own
    # scale, not each component's. A componentwise ratio is ill-posed in a weak
    # direction: measured on the correlated-k path, every component agrees to
    # <= 2.3e-6 ABSOLUTE while the dominant ones are ~1e3, yet dL/dc_o (|g| =
    # 0.38, 2500x smaller) turns its 1.3e-6 into a 3.3e-6 "relative error" that
    # says nothing about the wiring. Gate on the norm-relative figure -- 1e-8
    # there is a TIGHTER absolute requirement on the components that matter than
    # the old componentwise 1e-6 was -- and report both.
    scale_g = float(np.max(np.abs(gn)))
    rel_bn = float(np.max(np.abs(gb - gn)) / max(scale_g, 1e-300))
    rel_cw = float(np.max(np.abs(gb - gn)
                          / np.maximum(np.abs(gn), 1e-12 * scale_g + 1e-30)))
    ok_bn = bool(rel_bn < 1e-8) and ok_val
    print(f"[smoke] block-vs-naive max|d| / max|g| = {rel_bn:.2e} "
          f"(componentwise {rel_cw:.2e})  -> {'OK' if ok_bn else 'FAIL'}", flush=True)
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
        # weak directions: absolute agreement relative to the dominant gradient scale.
        # 5e-2 is also what the correlated-k path needs: ckd.overlap's resort-rebin
        # is continuous but its derivative has dense kinks, so AD returns the
        # almost-everywhere derivative while a central difference averages across
        # them. Measured ~1.5e-2 in temperature (h-independent from 10 K to 0.1 K)
        # and 1e-9 in an order-preserving all-species scaling. Not a bug, and not a
        # gate to tighten -- see CLAUDE.md, "Opacity: correlated-k".
        ok_i = (rel < 5e-2) or (abs(ad - fd) < 1e-4 * gmax)
        ok_fd &= ok_i
        print(f"    {pipe.names[i]:12s} ad={ad:+12.5e}  fd={fd:+12.5e}  rel={rel:.2e} "
              f"[{time.time()-t0:.0f}s]  {'OK' if ok_i else 'FAIL'}", flush=True)

    # ---- staged batched evaluator (SMC hot path) vs per-particle block gradient ----
    # Same chain rule, regrouped (fwd-jvp chemistry lanes contracted against ONE
    # reverse-mode RT vjp, RT lax.map-chunked). TWO regimes, and which one is in
    # force is printed:
    #
    #   cold_lanes == 0 (production default): the two routes are the same runner
    #     CADENCE CLASS, but the block reference is the scalar runner, so the tight
    #     gate -- dval < 1e-6 relative (floor 1), dgrad < 1e-5 norm-relative -- is
    #     EMPIRICAL on these probe draws, not an identity: a lockstep lane keys
    #     photolysis and the geometry refresh to the loop tick instead of its own
    #     accept count, which on a harder case can move the column at the
    #     convergence scale. Tolerances CALIBRATED, not assumed: the two
    #     routes batch and fuse differently, so the correlated-k RT's resort-rebin
    #     (a 256-wide cumsum + interp per fold) accumulates visibly more floating
    #     point than the sampled path did. Measured over the three probe points:
    #     dval 4.8e-12 / 8.2e-11 / 3.8e-08, dgrad 1.7e-09 / 1.9e-08 / 5.4e-07, so
    #     the gates sit ~20x above the worst and still leave five orders of margin
    #     against the wiring bug this check exists to catch.
    #
    #   cold_lanes > 0: the staged evaluator runs the lane QUEUE while
    #     value_and_grad_block is the per-particle SOLO solve, so a refilled draw
    #     enters the loop at the tick its lane was freed at and the two are
    #     DIFFERENT MAPS by design -- they can only agree at the convergence
    #     scale. The gate is then the repo's own convergence-scale standard:
    #     |logL_staged - logL_block| < DLOGL_MAX_PASS (0.1 ABSOLUTE, validate_warm's
    #     warm-vs-cold likelihood gate) and max|dG| / max(max|G_block|, 1) <
    #     DLOGL_MAX_PASS (the same 0.1, norm-relative with an absolute-1 scale
    #     floor -- what validate_warm holds the re-solved u-space gradient to).
    #     The measured sizes at cold_lanes = 2 are in notes 2.13; a lane count is
    #     a config change, so this regime is the bench's, never production's.
    from retrieval_framework.validate_warm import DLOGL_MAX_PASS
    t0 = time.time()
    lanes = int(getattr(cfg, "cold_lanes", 0) or 0)
    queued = lanes > 0
    rule = (f"cold_lanes={lanes}: the staged evaluator QUEUES and the block "
            f"reference is the per-particle solo solve, so the gate is the "
            f"convergence-scale standard |dlogL| < {DLOGL_MAX_PASS} absolute and "
            f"max|dG|/max(max|G|,1) < {DLOGL_MAX_PASS}" if queued else
            "cold_lanes=0: same runner cadence class, but the block reference is "
            "the SCALAR runner, so the tight pair is empirical on these probe "
            "draws: gate dval < 1e-6 (relative, floor 1) and dgrad < 1e-5 "
            "(norm-relative)")
    print(f"[smoke] staged-vs-block regime -- {rule}", flush=True)
    du = jnp.asarray(np.linspace(-0.06, 0.09, pipe.n_dim))
    U_test = jnp.stack([u0, u0 + du, u0 - du])
    Y0, refs0 = P._blank_state(pipe, int(U_test.shape[0]))
    Lb, Gb2, Yb, refsb, nbad_b, _stats = jax.jit(pipe.batch_eval_cold_vg)(
        U_test, Y0, refs0)
    assert int(nbad_b) == 0, "staged cold eval flagged gradient pathologies"
    Lb = np.asarray(Lb); Gb2 = np.asarray(Gb2)
    ok_staged = True
    for r in range(int(U_test.shape[0])):
        vr, gr = pipe.value_and_grad_block(U_test[r])
        vr = float(vr); gr = np.asarray(gr)
        dv_abs = abs(Lb[r] - vr)
        dv = dv_abs / max(1.0, abs(vr))
        # same norm-relative yardstick as the block-vs-naive check above
        gmax = float(np.max(np.abs(gr)))
        dgmax = float(np.max(np.abs(Gb2[r] - gr)))
        dg = dgmax / max(gmax, 1e-300)
        if queued:
            ok_r = ((dv_abs < DLOGL_MAX_PASS)
                    and (dgmax / max(gmax, 1.0) < DLOGL_MAX_PASS))
        else:
            ok_r = (dv < 1e-6) and (dg < 1e-5)
        ok_staged &= ok_r
        print(f"[smoke] staged-vs-block row {r}: dval={dv:.2e} dgrad={dg:.2e} "
              f"(|dlogL|={dv_abs:.2e}) {'OK' if ok_r else 'FAIL'}", flush=True)
    print(f"[smoke] staged batched evaluator check done [{time.time()-t0:.0f}s] "
          f"-> {'OK' if ok_staged else 'FAIL'}", flush=True)
    assert np.all(np.isfinite(Yb)) and np.asarray(refsb).shape == (int(U_test.shape[0]), 2)

    # ---- stage-split vs legacy single-chain cold jvp ----
    # The cold two-stage gradient path runs its two stages as two explicit
    # jvps. That is the SAME program regrouped, so the primal must be
    # bit-identical and the gradient may differ only by XLA fusion: the two
    # routes batch the stage-1 tangent at different widths, measured 1.1e-10
    # norm-relative on these three points, and the gate sits ~90x above that.
    t0 = time.time()
    legacy = jax.jit(pipe._make_batch_eval("cold", True, split_stage1=False))
    Ll, Gl, Yl, _rl, nbad_l, _sl = legacy(U_test, Y0, refs0)
    assert int(nbad_l) == 0
    Ll = np.asarray(Ll); Gl = np.asarray(Gl)
    same_y = bool(np.array_equal(np.asarray(Yb), np.asarray(Yl)))
    same_l = bool(np.array_equal(Lb, Ll))
    dg_split = float(np.max(np.abs(Gb2 - Gl)) / max(float(np.max(np.abs(Gl))), 1e-300))
    ok_split = same_y and same_l and (dg_split < 1e-8)
    print(f"[smoke] split-vs-legacy: Y bit-equal={same_y} L bit-equal={same_l} "
          f"dgrad={dg_split:.2e} [{time.time()-t0:.0f}s] "
          f"-> {'OK' if ok_split else 'FAIL'}", flush=True)

    # ---- warm-continuation gradient (the mutation-kernel map) vs FD of the same map ----
    # State = the converged columns from the cold batch above; evaluate the move
    # gradient at a DIFFERENT point (a realistic MCMC proposal) and FD the identical
    # warm map (fixed carried state) -- validates that the tangent relaxes through
    # the warm-started while_loop (the continuation-jvp pattern).
    #
    # The WHOLE 3-particle cloud goes in, and each particle carries its own
    # column AND its own reference composition (the refsb rows differ), so this
    # also exercises the per-lane reference threading of the batched warm solve;
    # with cold_lanes = 2 the three particles on two lanes make it refill. The
    # evaluators are built for chem_mode "warm" explicitly: the pipeline's
    # batch_eval_move_* follow cfg.smc_chem_mode, which defaults to "cold".
    # Only row 0 is perturbed, so its own solve history is the same in both FD
    # arms.
    t0 = time.time()
    U1 = U_test + 0.5 * du
    Y_w, refs_w = Yb, refsb
    move_vg = jax.jit(pipe._make_batch_eval("warm", True))
    move_l = jax.jit(pipe._make_batch_eval("warm", False))
    L1, G1, _, _, nbad_w, statsw = move_vg(U1, Y_w, refs_w)
    assert int(nbad_w) == 0, "warm move eval flagged gradient pathologies"
    print(f"[smoke] warm move eval: L={np.asarray(L1).tolist()} accept="
          f"{np.asarray(statsw.acc).tolist()} conv_ok="
          f"{np.asarray(statsw.conv_ok).astype(int).tolist()} refs="
          f"{np.asarray(refs_w).round(4).tolist()}", flush=True)
    assert float(np.asarray(L1)[0]) > P.REJECT_BELOW, (
        "the warm proposal was REJECTED (capped / uncertified): the FD check "
        "below would compare two rejections")
    g_warm = np.asarray(G1[0])
    ok_warm = True
    gmax_w = np.max(np.abs(g_warm))
    for i in range(pipe.n_dim):
        E = np.zeros_like(np.asarray(U1)); E[0, i] = h      # row 0 only
        Lp = float(move_l(U1 + jnp.asarray(E), Y_w, refs_w)[0][0])
        Lm = float(move_l(U1 - jnp.asarray(E), Y_w, refs_w)[0][0])
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

    ok = ok_bn and ok_fd and ok_staged and ok_split and ok_warm and ok_live
    print(f"[smoke] TOTAL {time.time()-t_all:.0f}s  ->  {'ALL CHECKS PASSED' if ok else 'FAILURES'}",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
