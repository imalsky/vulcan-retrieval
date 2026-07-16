# Negative-results register: every failed approach in the W39b retrieval

Purpose: the citable "we tried X and it did not work" record for the WASP-39b
retrieval effort (vulcan-retrieval + the retrieval-relevant parts of
VULCAN-JAX). Every failed, retracted, measured-worse, or dead-end approach,
with the date, the measured number where one exists, and an evidence pointer
(notes.md section, commit/tag, doc, or NAS job number).

Maintenance rules: APPEND, never delete (a retracted retraction gets its own
entry). Every entry needs (a) what was tried, (b) why it failed / the number,
(c) date, (d) evidence pointer. The prose "why" behind each lives in
`../notes.md` (the dev diary); this file is the scannable index of negatives.
Sources mined at creation (2026-07-15): notes.md, both repos' full git logs +
tags/branches, ../docs/* investigation records, runs/w39b_smc_retrieval/notes,
and the session-memory incident log.

---

## A. Samplers + inference architecture

1. **NUTS / nested sampling / parallel tempering.** Ruled out by the batched
   architecture: the chemistry runs as ONE lockstep vmapped while_loop over the
   whole particle cloud, so samplers needing per-chain adaptive path lengths or
   asynchronous evaluation waste the batch. Verdict: keep SMC+MALA.
   -- 2026-07-09 (24 h feasibility audit); notes.md.
2. **RWMH mutation kernel.** Removed in the GH200 saturation rework; MALA-only
   (per the loud-errors pass -- a silent gradient-free fallback kernel is a
   different sampler). -- 2026-07-06; notes.md SS B/C.
3. **Blocked-LIS sampler (pre-framework `33.py`).** Abandoned with three
   unfixed issues and a test gap; superseded by the SMC framework entirely.
   -- 2026-05-19; session memory.
4. **Shape-only MALA preconditioner (unit-geometric-mean diagonal, the SWAMPE
   kernel).** Left overall proposal width to a scalar Robbins-Monro that lagged
   the beta ladder: final-stage acceptance 0.019 (same signature as SWAMPE's
   WASP-43b pilot, accept=0.001). Replaced by an ABSOLUTE per-dim-std
   preconditioner (accept 0.62-0.91). -- notes.md SS B.
5. **12 MALA sweeps/stage.** Halved to 6 (published MALA-within-SMC practice
   is 3-10 with a good preconditioner; each sweep is one full batched
   gradient). -- 2026-07-09; notes.md.
6. **Raise-on-nonconverged init gate (raise if >10% of cold draws fail).**
   Wrong response to a real ~27% minority; carrying an unconverged state as
   finite L was a silent bias. Replaced by reject-with-`-inf` + oversample.
   -- 2026-07-08; notes.md "Cold-init reject-and-cull".
7. **Init phase 2 capped at warm_count_max=1500.** A marginal survivor can
   need >1500 accepted steps to RE-certify warm: job 64854 gated 5/96 healthy
   survivors into a spurious "RT/AD problem" RuntimeError. Phase 2 now runs
   UNCAPPED. -- 2026-07-10; notes.md.
8. **Uncapped phase 2 with no spares.** Still insufficient: job 64897, 3/96
   survivors certify cold but cannot re-certify warm in 5000. Fixed by
   cull-and-backfill from `init_phase2_spare`. -- 2026-07-10; notes.md.
9. **init_phase2_spare=8 at N=144 on real data.** First real-data N=144 run
   died in init phase 2: 27/192 (14%) warm-recert culls exhausted the 8
   spares. Raised to 48 + init_oversample 2.0 -> 2.5 (width is ~free in the
   lockstep batch). -- 2026-07-12/13; commit a4c15c2; session memory.
10. **calibrate() at hard-coded (beta=0.5, step=0.2, scale=1).** Prior-like
    clouds carry |G|~1e6, so the drift landed proposals far off the converged
    map: job 64961, 8/144 finite-spectrum/non-finite-gradient per sweep,
    spurious abort at accept=0.00. calibrate() now reproduces the ladder's own
    stage-0 conditions. -- 2026-07-11; commit 6edb289; notes.md.
11. **Warm MALA mutation with no cap of its own.** Job 64745: early-ladder
    sweeps gated at the full count_max=5000 (~30% of prior mass
    non-convergent), >3 h in stage 1, projected 3-6 h/stage. Fixed:
    warm_count_max=1500 twin runner. -- 2026-07-09; notes.md.
12. **warm_count_max=800 (first guess).** The conv_step=500 certification
    window dominates the warm floor (~780 accepted steps for a GOOD MALA-small
    move), so 800 would reject typical healthy proposals. Set 1500.
    -- 2026-07-09; notes.md.
13. **Separate primal-only diag solve to read the warm accept count.** Doubled
    the chemistry wall time per sweep; the count (now the full ConvDiag) rides
    the jvp'd primal carry for free. -- 2026-07-09; notes.md.
14. **Accept-count-only warm "usable" gate (`ACC < warm_count_max` alone).**
    Certif-by-stall / loose-branch exits passed the gate with unsettled
    tangents: job 65200, 16/864 warm proposals finite-L/non-finite-gradient,
    died at stage 0 after 4.2 h with the 2.1 h init lost and zero forensics.
    Fixed: conv_normal certification gate + `stalled` rejection class +
    init-level checkpoint + per-particle forensics dump. -- 2026-07-13/15;
    notes.md 65200 post-mortem; commit f05e537.
15. **Predicate P0 (`longdy < yconv_min` at exit) as the proposal gate.**
    Provably a no-op for the failing class: ALL certified exits (including
    stall-fallback) have longdy < yconv_min by construction under central
    diffusion. -- 2026-07-15; notes.md corner-probe; validation/
    diag_warm_stall_tangent.py.
16. **Predicate P2 (tight branch only: longdy < yconv_cri AND longdydt <
    slope_cri).** Ruled out permanently: every HEALTHY warm proposal certifies
    on the LOOSE branch (longdy 0.04-0.09 >> yconv_cri=0.01) -- P2 rejects the
    entire warm kernel. Measured under both hybrid and central schemes.
    -- 2026-07-15; notes.md corner-probe.
17. **Zero-tolerance raise on any non-finite mutation gradient.** After the
    conv_normal gate, job 65789 still died at stage-0 sweep 1 on ONE event:
    particle 12, acc 470 (under cap), longdy 0.087 (genuine loose-branch
    certification, stalled=0), chemistry-tangent side. The residual class is
    forward-mode tangent divergence at marginally-stable CERTIFIED fixed
    points -- unflaggable by any primal-side predicate, ~1%/sweep at
    prior-like beta, so zero tolerance means no ladder can pass stage 0.
    Replaced by MH-reject-with-floored-L + `smc_tangent_reject_max_frac`
    systematic-abort threshold (default 5%). -- 2026-07-15; notes.md 65789;
    commit 273b6d5.
18. **Eval-level "zero the bad gradient for hygiene" as the handling.** A
    zeroed-gradient proposal can be ACCEPTED with a corrupted MH ratio -- the
    exact silent random-walk degradation the loud rule guarded against.
    Handling must be an L-floor (clean MH rejection). -- 2026-07-06/15;
    notes.md SS C + 65789.
19. **`logZ_box_physical` evidence field (logZ + ln f_tp).** RETRACTED the
    same day it shipped: P(A)*E[L|A and C] restores the T-P prior mass while
    silently keeping the convergence conditioning renormalized -- reconstructs
    no integral (toy counterexample pinned in tests/test_evidence_semantics).
    -- 2026-07-12; commits bc96cb3/b85ac3f; notes.md.
20. **BlackJAX SMC.** Unavailable in the vulcan env and NAS pyt2_8_gh; a
    fragile HPC pip-install was rejected -- forced the ~200-line pure-JAX
    Del Moral SMC core. (Environment constraint that foreclosed the
    off-the-shelf path.) -- notes.md SS B.

## B. Gradients / automatic differentiation

21. **All-in-one all-particles-vmap merged gradient.** Jobs 63886/63972/63995/
    63997: jit_mutate requested 1.52 TiB; remat bottomed at 1.06 TiB; the
    executable exceeded the 2 GB protobuf cap; 4-particle chunks still peaked
    120.9 GiB vs the ~87 GiB pool and went launch-latency-bound (~200/700 W).
    Redesigned into the staged evaluator (full-width chem jvp lanes x one RT
    vjp per particle). -- 2026-07-06; notes.md SS C.
22. **MEM_FRACTION=0.98.** Broke executable-constant allocation on GH200;
    reverted to 0.90. -- notes.md SS C.
23. **Two-call chem-gradient form (dir-0 + (n-1)-lane call).** ~2x wall when
    latency-bound; merged into one vmapped jvp over all lanes. -- 2026-07-05;
    notes.md SS H.
24. **nu_pts=16500 native RT (and the earlier 6000 default).** Job 64601: the
    init-phase-2 RT-vjp tried to allocate 343 GiB on a 96 GB GH200. Production
    nu_pts=1652 (~R1000); validate_config warns above 2500; PROBE_MEMORY=1
    before any raise. Probe 64144: full-width cold_vg at the old settings =
    195.25 GiB (would OOM). -- 2026-07-09; notes.md; CLAUDE.md.
25. **fp32 anywhere in the chemistry.** Rejected: rate constants span ~50 dex;
    float32 silently fails. fp32-RT alone is <2x on a non-dominant term.
    -- 2026-07-09; notes.md; VULCAN-JAX CLAUDE.md.
26. **The ~1.3 GB/lane chem-tangent memory estimate.** Wrong by ~60x (probe:
    ~20 MB/lane-pair); it was the all-in-one architecture's PreMODIT tangents
    misattributed. Chemistry gradient runs unchunked. -- 2026-07-07; notes.md.
27. **Differentiating Tirr through Heng+14 `expn` in forward mode.**
    Pathologically slow over a deep column (huge argument range);
    differentiate the Tco leaf directly. -- 2026-06-17; VULCAN-JAX CLAUDE.md.
67. **MH-rejecting the tangent-blown (badgrad) proposal class
    (`smc_tangent_reject_max_frac`, commit 273b6d5).** Designed for a ~1%
    theta-INDEPENDENT stochastic tail; job 65815 forensics (13 sweeps, 64
    events) measured the class theta-DEPENDENT and posterior-concentrated:
    6.5% of certified proposals (5.1/5.9/11.0% by stage), 1.2% at Z=1-12x
    solar vs 11.7% at 67-99x, 11.3% at C/O 0.10-0.18 vs 4.4% above 0.32,
    all 64 chemistry-side, 44% at longdy<0.01 (not only marginal certs),
    with the cloud median drifting lnZ +0.77->+1.71 INTO the dense region.
    Forced rejection = theta-correlated suppression of the posterior bulk
    (metallicity bias), and the 5% per-sweep abort trips with P~6%/sweep at
    the measured base rate -- certain over a 240-sweep ladder (65815 died at
    stage 2 sweep 2, 11/144 > 8). Local replays could NOT reproduce the
    blowup at zero-increment warm re-certification from each theta's own
    cold column (offenders and controls all tangent-finite, prep tangents
    finite) -- the divergence needs the actual mutation-move warm trajectory,
    so no pointwise clip or certification tightening can remove it (38% of
    offenders are loose-branch certs, but 44% are well-converged). Replaced
    by ZERO-DRIFT MALA handling (valid MH: eval-zeroed drift used
    consistently in both proposal densities; certified likelihood decides
    acceptance) + `smc_tangent_bad_max_frac` systematic-breakage backstop
    (0.25). -- 2026-07-16; job 65815; forensics
    runs/w39b_smc_retrieval/forensics_65815/; notes.md 2026-07-16.

68. **Unconditional warm_extrapolate seed max(Y + DY.dC, 0), and the
    per-cell repair where(pred>0, pred, Y).** The clipped extrapolated seed
    is the dominant manufacturer of the badgrad tangent class: move-replays
    of job 65815 states gave 3/11 non-finite tangent sets with the
    extrapolated seed vs 0/11 with the plain carried column (0 in 24+ plain
    warm jvps overall), each reproduction preceded by 872-1509 cells driven
    non-positive by the linear prediction. The per-cell fallback repair
    still blew 2 of 3 reproduced cases -- a prediction that far out is
    toxic even in its positive cells -- and the poisoned solves also
    burned to the warm cap while plain seeds converged in 376-1141 steps.
    Replaced by the PER-PARTICLE extrapolation gate (extrapolate only when
    no cell needs clipping; fall back to the plain column + carried refs).
    -- 2026-07-16; validation/badgrad_65815/replay_badgrad{,_v2,_v3}.py;
    docs/job65815_badgrad_investigation.md SS10.


Reverse-mode steady-state adjoint dead-ends (VULCAN-JAX; the reason the SMC
uses forward-mode MALA and reverse-mode stays off the hot path):

28. **Residual-IFT custom_vjp (defect-correction block-Thomas).** df/dy is
    singular (conserved-mass null space) AND ill-conditioned (residual ~1e21)
    on real closed columns: gradient +876 (wrong sign, ~1500x off). DELETED.
    Same-fate variants: matrix-free LSQR pseudoinverse (istop=7, gradient ~0)
    and raw-Neumann fixed-point adjoint (diverged to 1e57). -- 2026-06-16;
    ../docs/vulcan_jax_notes.md.
29. **Left-preconditioning the adjoint solve.** Minimizes the wrong metric
    (small preconditioned resid, garbage gradient). Right-precondition.
    -- vulcan_jax_notes.md.
30. **`reg*I - J` preconditioner shift.** No usable reg exists: resolving the
    slow mode needs reg~mu_slow -> cond(M)~3e20 -> float64 garbage.
    -- vulcan_jax_notes.md.
31. **Orthogonal deflation of slow modes.** Biases the gradient (drops their
    real contribution). -- vulcan_jax_notes.md.
32. **Augmented GMRES w/ real-block power iteration (n_slow=24, m=300).**
    Residual stuck ~0.6; the slow subspace is high-dimensional with complex
    pairs the real-block iteration misses. -- vulcan_jax_notes.md.
33. **Restarted GMRES(300).** Oscillates/stagnates on the indefinite operator
    (resid bounces 0.1-1.6); the bordered/saddle fix diverged. LGMRES is the
    only bounded solver of five tried. -- vulcan_jax_notes.md.
34. **Other Krylov variants.** Un-restarted GMRES(2500) breaks down (resid
    60); GCROT blows up (resid 191); LSMR FP-dead (transpose-pair fails by
    1e17); one-sided eigendeflation makes it WORSE (non-normal operator).
    -- 2026-07-01; vulcan_jax_notes.md.
35. **body_dt outside the measured window.** 1e6 diverges (wrong sign); 3e6 is
    a 27.6% deterministic bias; 1e8 stalls with a +/-20% lottery; >=3e8
    garbage; ~1e11 hits the singular-step pole. Optimum 1e7 (0.3-6%).
    -- 2026-07-01; vulcan_jax_notes.md.
36. **solver_map="bare" (linearize the raw Ros2 step).** Few-% floor (HD189
    CH4 6.6%, W39b OH+H2 11%) is a linearized-map mismatch, not convergence;
    renorm default reaches 0.7%. -- 2026-07-03/04; vulcan_jax_notes.md.
37. **renorm_td deflation (extra per-layer total-density deflation).**
    Over-corrects (0.7% -> 2.5%). Not adopted. -- vulcan_jax_notes.md.
38. **Frozen-photolysis adjoint (omit dJ/dy).** Photo-coupled rows stuck at
    11-13%; requires photo_recompute_k (an RT solve per Krylov matvec).
    -- 2026-07-03; vulcan_jax_notes.md.
39. **jax_compilation_cache_dir for the adjoint's ~20-min step-vjp compile.**
    Did not help and appeared to break the warm in-process path; left unset.
    -- vulcan_jax_notes.md.
40. **Sub-1% reverse-mode via the f=0 IFT at production tolerance.**
    Unreachable in principle: a steady-state-DEFINITION mismatch (slowest
    chemical mode tau~3e10 s frozen at ~2e7 s "convergence"; FD and IFT
    differentiate different states). Forward-mode is the exact route.
    -- 2026-06-16; vulcan_jax_notes.md.
41. **Host-LGMRES adjoint on the SMC hot path.** Not hot-path-usable
    (host-side scipy, one-shot post-convergence); the dimension-independent
    reverse-mode gradient stays deferred. -- 2026-07-07; notes.md init-stall
    diagnosis.

## C. Condensation / Fisher (Route B and the pin)

42. **Isothermal escape hatch (`_condense_validated_isothermal` +
    NotImplementedError).** Unsafe for W39b (saturation tables at GCM T,
    chemistry at T_iso). REMOVED; on-graph per-proposal conden rebuild.
    -- 2026-07-13; commits 20c5e89 / VULCAN-JAX 036a813.
43. **Naive condensation-on solves (no window+pin).** Do not converge: 1 um S8
    pins dt~0.4 s (longdy 10.4); 50 um is transport-limited (reservoir drains
    on the Kzz timescale, longdy 1.55). -- 2026-07-13; notes.md; CLAUDE.md.
44. **Cold-trap-level pin on isothermal columns.** The argmin degenerates to
    layer 0, the gas pin is EMPTY, post-pin chemistry re-supersaturated S8 by
    2560x. Whole-column pin required. -- 2026-07-13; notes.md.
45. **conver_ignore alone / mtol_conv alone for photo-off convergence.** No
    effect / actively worse (the blow-up is N-driven, not S; mtol alone blows
    up the majors, longdy 46). Only the full recipe works (central diffusion +
    dt_max cap + allotrope ignore + mtol). -- 2026-07-15;
    ../docs/photo_off_convergence_investigation.md.
46. **Any total derivative through the pinned condensation state (Fisher /
    gradient-MALA / input sensitivity).** The pinned S8 tangent is ~91% WRONG
    (jvp-vs-FD rel err 0.91; path-sensitive transient + discrete switches).
    REFUSED at the AD entry points and by the resolved-config retrieval gate;
    condensation is a forward-model capability only. -- 2026-07-13/15;
    ../docs/condensation_differentiation.md; commits 3818c39 / 7eda1af.
47. **Route B smooth open-system rainout + deep H2S reservoir (B0C).** NO-GO:
    G1 fails (no full-network cold steady state within 15000 steps;
    quench-frozen N radicals at ~1e-14 pin longdy) and G3 shows honest
    non-closure (sulfur-flux residual 26.4% on the settling arc). ~1250 lines
    across 15 files shelved to branch research/smooth-rainout-fisher, tag
    smooth-rainout-b0c-no-go-2026-07-14 (both repos). B1 never authorized.
    -- 2026-07-14; ../docs/route_b_smooth_condensation_plan.txt SS 6.
48. **Route B H2S boundary as built (trilinear-in-ln-x lookup).** C0 only:
    first derivatives piecewise-constant, 21% relative worst-case on near-zero
    slopes; would need a C1 upgrade before any Fisher consumes it.
    -- 2026-07-13; ../docs/route_b_b0a_decision_record.txt.
49. **Anchor-scale prototype BC (x_H2S = x_base*exp(lnZ)).** Disqualified as
    final: T and c_o derivatives identically zero, ~8% value error.
    -- 2026-07-13; route_b_b0a_decision_record.txt.
50. **Middle paths between the guards and full Route B.** Criterion-gated pin
    (doesn't fix phase-boundary nonsmoothness or closed/open physics),
    differentiate-only-the-frozen-branch (breaks exactly at the switches),
    smooth surrogate/emulator (a different project). All rejected.
    -- ../docs/condensation_differentiation.md SS 8.
51. **Hessian through condensation.** Strictly harder than Route B (needs a C2
    hinge + C1 boundary); no path. -- condensation_differentiation.md SS 6.
52. **Wrapping `_runner` in custom_jvp to auto-block conden jvp.** Too
    intrusive (blocks valid low-level uses); guards live at the reverse-mode
    entry points and consumers. -- condensation_differentiation.md SS 10.
53. **Photo-off Fisher forecasts on the default WIDE profile.** The photo-off
    forward model does not reach steady state on defaults (uncapped
    dt_max=1e17 blow-up + stiff sulfur allotropes + hybrid upwind diffusion);
    the gate's original "warm-jvp under-relaxed" reason was a mislabel. Even
    on the recipe-converged state the through-loop tangent is quantitatively
    unreliable (magnitudes off 25% to ~2x; warm lnKzz correlation 0.69).
    -- 2026-07-15; ../docs/photo_off_convergence_investigation.md.

## D. Solver numerics / convergence

54. **Init-scaling lnZ/C-O with a moving T-P (the inventory-erasure
    channel).** dL/dlnZ ~ 1e-20 and FD AGREED: the hydrostatic renorm under a
    displaced T makes the column forget its initial inventory entirely. Fixed
    by the two-stage solve + exact-elemental abundance mode. -- 2026-07-04/05;
    notes.md SS A (the big one).
55. **Snap-to-baseline continuation (SO2-Hessian-campaign pattern).** Same
    init-forgetting class; superseded by warm-started two-stage jvp.
    -- notes.md SS A.
56. **dt_max at the master default (runtime*1e-5 = 1e17 s).** Adaptive dt
    balloons to ~1e16-1e18 s and the solver spins in a large-dt oscillation
    (longdy stuck 2-4) -- the whole >10k-step tail. Capped 1e11 (job 64523
    d19: >11000 -> 986 steps, steady state bit-identical). -- 2026-07-08;
    notes.md; CLAUDE.md.
57. **yconv_cri=1e-3 (inherited from the sensitivity-demo need).** Thousands
    of extra steps for no gradient-quality gain; reverted to canonical 0.01.
    -- 2026-07-08; notes.md.
58. **conv_step=300.** Certifies a LESS-converged state (up to 0.072 dex from
    the 500-certified one) with zero hot-path saving. Keep 500. -- 2026-07-10;
    notes.md.
59. **count_max 3e4 / 1e4.** Open-ended phase-1 tail (jobs 64073/64163 sat for
    hours); 5000 with reject+oversample instead. Probe evidence: job 64575
    (27.1% non-convergent at 5000, 12.5% at 12000) and probe job 64437
    (2026-07-08: 21% right-censored at a 20000-step cap; recorded in session
    memory only, not in notes.md). Raising count_max is explicitly forbidden.
    -- 2026-07-07/08; notes.md SS K; CLAUDE.md.
60. **T-P clipping into [300,3000] K.** Clipping produces silently-wrong
    finite-likelihood states; replaced by reject-and-redraw at the prior and
    -inf proposals. -- 2026-07-08; notes.md; CLAUDE.md.
61. **prior_c_o upper edge 0.6.** The fixed-O b_z positivity bound on the real
    column is +0.566 (INSIDE the box): draws beyond it were clip-mangled into
    negative O-carriers with finite likelihoods (16 h job 64144-adjacent run).
    Capped at 0.45; C/O ~ 1.0 is structurally unreachable by the fixed-O knob.
    -- 2026-07-07; notes.md SS K.
62. **Gustafsson PI step-size controller (from neoVULCAN).** Measured net
    11-17% SLOWER (cuts delta-rejections 13-16% but slows dt growth: 14-20%
    more accepted steps; HD189 48.6->53.9 s, W39b 63.7->74.3 s). Default off;
    port closed, never enable by default. -- 2026-07-13; vulcan_jax_notes.md;
    VULCAN-JAX e5586c4.
63. **Frozen setup-time vm (upwind molecular diffusion).** Biased a
    moldiff-dominated upper atmosphere by up to ~1.7 dex on depleted traces;
    vm must be refreshed in-loop. -- 2026-06-29; VULCAN-JAX CLAUDE.md.
64. **Hybrid vm_mol defaults consumed unexamined by the retrieval.** Two
    measured issues: (a) destabilizes photo-off convergence; (b) the 07-14
    port moved termination budgets to CARRY fields seeded at pack time, which
    silently UNBOUND the warm twin's 1500 cap and would restart every solve in
    upwind phase 0. Fixed by per-solve carry re-seeding + pinning warm
    CONTINUATIONS to the central operator (cold solves keep hybrid; certified
    fixed points agree <= 0.182 dex on RT cells). Hybrid also extends a cold
    solve past the static count_max (6002 observed), raising phase-1
    attrition. -- 2026-07-14/15; notes.md 65200 fix 4 + corner-probe;
    commit f05e537.
65. **Naive two-stream particular-solution pole fix (denominator floor /
    clip).** Changes near-pole physics and breaks master parity; deferred to a
    proper analytic resonant-limit solve (W39b's 83 deg config is safe).
    -- ../docs/corrections_to_original_code.md.
66. **"Fixing" inherited master defects unilaterally.** Rejected for parity:
    unweighted atom-conservation diagnostic, condensate mass in gas mu,
    atm_type='table' stale-pico. Documented, not patched. --
    corrections_to_original_code.md.

## E. HPC / ops incidents (chronological)

- **63886/63972/63995/63997** (07-06): all-in-one gradient OOM (1.52 TiB
  request; 2 GB protobuf cap; 120.9 GiB at 4-particle chunks).
- **64073** (07-07): 16.7 h silent init (no log, no checkpoint) inside the
  old cold-map _init_state; count_max=3e4 bounded only accepted steps.
- **64144** (07-07): b_z guard tripped on first contact; memory table measured
  (full cold_vg 195.25 GiB would OOM; RT vjp 18.4 GiB/lane).
- **64163** (07-07): two-phase init still sat hours in phase 1 at count_max
  3e4 -> lowered to 5000.
- **64437** (07-08, session memory only): count_max probe, 21% of prior draws
  right-censored at a 20000-step cap.
- **64523** (07-08): dt ballooning calibration (draw d19 >11000 -> 986 steps
  once dt_max=1e11).
- **64575** (07-08): 27.1% of cold draws non-convergent at count_max=5000
  (12.5% at 12000) -> reject-and-cull + oversample.
- **64601** (07-09): nu_pts=16500 RT vjp tried 343 GiB on a 96 GB GH200.
- **64604** (07-09): nsys masked the exit code (rc=0 on a dead run) -> never
  profile a first/debug run.
- **64745** (07-09): uncapped warm mutation, >3 h in stage 1, 3-6 h/stage
  projection -> warm_count_max twin runner.
- **64854** (07-10): warm cap wrongly applied to init phase 2 (5/96 healthy
  survivors mislabeled).
- **64897** (07-10): 3/96 survivors die even uncapped -> spares + backfill.
- **64944** (07-10): probe: peak memory width-independent (73.25 GiB at
  N=96/144/152) -> enabled N=144.
- **64961** (07-11): calibrate() stage-0 mismatch (8/144 bad grads); PLUS the
  concurrent per-job `pip install --user -e` race that crashed mid-uninstall
  on the shared userbase -> read-only jobs + bootstrap-only installs +
  per-interpreter userbase.
- **65200** (07-13): stage-0 death, 16/864 finite-L/non-finite-grad, init
  lost, no forensics -> certification gate + init checkpoint + forensics.
- **65789** (07-15): one bad grad at stage-0 sweep 1 on a CERTIFIED proposal
  (chem-tangent side) -> zero-tolerance raise replaced by reject-with-
  tolerance (`smc_tangent_reject_max_frac`).
- **65815** (07-15/16): real-data resume died at stage 2 sweep 2 on the 5%
  badgrad gate (11/144 > 8); forensics: class is theta-dependent
  (high-Z/low-C-O), 6.5% of certified proposals, rising along the ladder ->
  zero-drift MALA handling + 0.25 backstop (see #67). Also measured: 68% of
  warm proposals burn to the 1500 warmcap; ~2 h/stage x 40 stages exceeds the
  24 h walltime (checkpoint+resume across submissions is the operating mode).
- **NAS proxy**: the proxy hostname does not resolve from the front end;
  `unset https_proxy http_proxy` for git. rsync / tarballs / wrapped remote
  commands prohibited (scp -r only).

## F. Retired designs / removed subsystems

- **fisher_forecast/ module** -- superseded by scripts/zco_information +
  vulcan-jwst-tool's live Fisher; removed with its stale caches (07-11,
  commit 732aa9c).
- **sys.path bundle-import contract + import-time os.chdir(JAXROOT)** --
  killed in the two-package split; editable installs + loud import-order
  guards (07-11, commits 09bb9bb/22ecb30).
- **vulcan_exojax_run monorepo** -- retired; standalone sibling repos
  (07-11).
- **`gs` gravity knob** -- removed ("too much legacy"); gravity is G*Mp/Rp^2,
  missing Mp/Rp raises (07-14).
- **Hand-written vulcan_cfg.py + cfg_examples/*.py** -- deleted; YAML-only
  config (07-14).
- **BlackJAX dependency** -- never adopted (see A.20).
- **examples/quickstart.ipynb** -- removed, .py demos kept (07-12).
