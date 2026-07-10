# CLAUDE.md — vulcan_exojax_run operational notes

Critical rules and decisions for this bundle. Read before touching the retrieval or
running anything on the supercomputer.

## Supercomputer transfer — ALWAYS scp -r, NEVER rsync

Transfer with **exactly** this style (one command per top-level dir, no backslashes):

```
scp -r -oProxyCommand='ssh imalsky@sfe6.nas.nasa.gov ssh-proxy %h' [local dir] imalsky@pfe.nas.nasa.gov:[hpc location]
```

- **Never use rsync.** Never make tarballs.
- **Never** wrap remote commands in `ssh nas '...'`. Give the `scp -r` transfer, then the
  commands to run **while logged in on the node** (`cd …`, `qsub …`) as plain commands.
- The two trees to send for a run: `VULCAN-JAX` and `vulcan_exojax_run`, into the same
  `PROJECT_ROOT` on `/nobackup` (currently `/nobackup/imalsky/VULCAN_W39b_HPC`). The PBS
  preflight requires both.

## nsys masks the exit code — never profile a first/debug run (learned 2026-07-09)

`nsys profile ... python ...` returns **0 even when the profiled process is killed or
crashes**. Job 64604 ran under `NSYS=1`, so when SMC stage 0 died the wrapper saw `rc=0`,
ran the plot step, and printed `job finished rc=0` — a green result on a dead run (no
posterior, no checkpoint), and the real error was swallowed. That masking hid the stage-0
mutation bug for ~15 calibration retries. **Rule: run calibrations and first/debug runs
WITHOUT `NSYS`** so a real failure surfaces (a `RuntimeError` traceback, a CUDA error, or a
`Killed`/OOM line). Add `NSYS=1` only once a run is known-good and you specifically want a
kernel timeline. The always-on `nvidia-smi` monitor (`logs/gpu_monitor_*.log`) already
gives util/power/clocks without it. Also: `NSYS_DELAY` is in **seconds** (64604 used
6000 = 100 min, not 6 s).

## Retrieval — critical decisions (2026-07-08)

- **count_max = 5000, always** (lowered from 10000, Isaac 2026-07-08). Set in `runs/w39b_smc_retrieval/case.py::gpu_config`;
  every override file falls back to it. Do NOT raise it. A solve that doesn't converge in
  5k accepted steps is a **failed draw** — acceptable, not a bug to chase: it is REJECTED
  at init and the cloud is oversampled to compensate (see "Cold-init reject-and-cull").
- **Convergence = VULCAN-master canonical criteria.** `yconv_cri = 0.01` (schema default),
  and `slope_cri` / `yconv_min` / `flux_cri` are NOT overridden (inherit
  `vulcan_cfg_W39b`). The old 1e-3 (from the sensitivity demo's tight-jvp needs) is gone —
  it barely changed gradient quality but ground out extra thousands of steps.
- **No T-P clipping — reject and redraw.** `tp_profile` returns the raw Guillot profile.
  A draw whose T-P leaves the modelable window **[300, 3000] K** (premodit table range,
  inset 20 K) on the ART grid is REJECTED, never clipped: rejection-sampled away at the
  prior (`pipeline.sample_prior_u` redraws) and given `-inf` likelihood as a MALA proposal
  (`pipeline.tp_valid` gates every likelihood path). The prior rejection sampler raises
  loudly if the T-P prior is mostly out-of-window (a mis-specified prior fails fast).
- **Realistic priors** live in `case.py::_W39B`, literature-anchored (Tsai et al. 2023
  VULCAN grid + Rustamkulov et al. 2023 PRISM ERS): C/O ∈ [0.10, 0.70], Kzz ±2 dex,
  Tirr ∈ [1100, 2200] K, γ ≤ ~2 (weak inversion allowed, Isaac 2026-07-08). Metallicity kept wide (1–100×
  solar) so the data localizes it.
- **Fail-fast everywhere** (standing rule): no silent fallbacks. Missing H2-He CIA raises;
  planet identity (rp_cm/rstar_cm/vulcan_cfg_module) must be set or `validate_config`
  raises; a real-data run with no overlapping bins raises; RESUME with no checkpoint
  raises; non-finite gradients surface as `n_bad` and the host raises.

## Why the >10k-step tail happened — dt_max ballooning (diagnosed 2026-07-08, local)

Local chemistry-only diagnostics (`vulcan` env, no RT; scratchpad `diag_*.py`,
`sweep_dtmax.py`): the tail is MOSTLY a numerical artifact, not slow physics. The
VULCAN-master default `dt_max = runtime*1e-5 = 1e17 s` lets the adaptive Ros2 step
balloon to ~1e16 s on high-Kzz columns (per-step local error stays tiny — the published
Tsai+2017 adaptive-stepping behavior), so the solver SPINS in a large-dt oscillation
(`longdy` stuck ~2-4, marched to t~1e17 s) instead of settling. Capping `dt_max`
converges those ballooning draws in ~1000 steps (d10/d19/d59 from calib job 64523; d19:
>11000 → 986). The cap VALUE in [1e9, 1e12] doesn't change WHICH draws converge, only
convergence tightness. **Fix applied: `Config.dt_max` (first-class, banner-shown) = 1e11
s in `case.py::_W39B`** (catches ballooning, reaches t~5e14 in 5000 steps >> physical
settling ~1e13, leaves the truth at 4275 steps identically). This is a STEP-SIZE control,
NOT a convergence criterion (yconv_cri/slope_cri stay master).

**A residual fraction still fails and `dt_max` CANNOT fix it** — two distinct modes:
(1) `longdy` stuck just above the 0.1 loose gate (~0.13, moderate t, aflux fine — a
marginal oscillation, e.g. d5 +Kzz2.0); (2) hot + low-Kzz photolysis limit cycles
(`aflux_change` stuck ~0.2-0.36, e.g. d40). These are genuine non-converging columns at
extreme prior corners (hot + extreme-Kzz), inherent to a full-kinetics forward.

## Cold-init reject-and-cull + oversample (measured + WIRED 2026-07-08)

The R=100 calibration (job 64575, dt_max=1e11 live, N=192 draws, probe 12000) MEASURED
the residual: **27.1% of cold draws don't converge at count_max=5000** (12.5% even at
12000; heavy tail — p50=2525 fine, but p75=5420 busts 5000). That is a real minority, not
a bug. **Fix (best practice, WIRED): `pipeline._init_state` now REJECTS non-converged
draws (-inf) and OVERSAMPLES so the culled cloud still holds exactly N healthy
particles** — the same handling every retrieval code uses for a failed forward
(petitRADTRANS / nested sampling `-inf` for invalid outputs) plus the Herbst-Schorfheide
SMC oversample-for-ESS rule. It draws `ceil(N * init_oversample)` (default
`init_oversample=2.0`, tolerates up to 50% non-convergence), pays the expensive gradient
pass on the N survivors only, and RAISES only if fewer than N survive (systemic). The old
behavior (raise at >10%, or carry unconverged states as finite L) is gone — carrying a
non-converged spectrum is a silent bias, rejecting it is correct. `init_max_nonconverged_frac`
is now just a WARNING threshold on the observed reject fraction. Unit-tested in
`retrieval_framework/tests/test_init_state.py` (5 tests: reject, cull, raise-if-too-few,
oversample count). Full suite 14/14 green.

**FIXED 2026-07-09 (was the deferred residual, and it was NOT low-risk):** a warm MALA
*mutation* proposal that hits a non-convergent corner is now count_max-REJECTED before its
gradient is trusted — the warm-side analogue of the cold-init reject-and-cull. Previously
the warm continuation returned a finite-but-off L **and had its jvp/RT-vjp computed
anyway**; that garbage gradient tripped `n_bad` (a spurious `RuntimeError`) or NaN'd at SMC
**stage 0** — which is what killed the tempering ladder in job 64604 and made every timing
calibration fail (~15 retries). It was invisible because the failure was masked by `nsys`
(see below) and because the memory probe never covered the mutation kernel. Root-caused by
ruling OUT memory (the mutation kernel's compiled footprint is byte-identical to the
cold-init gradient that already fit) and the sampler logic (14/14 unit tests), then
reproduced on the smoke pipeline. **Fix:** `retrieval_forward.chem_solve_warm_diag` (warm
twin of `chem_solve_cold_diag`) reports the warm accept_count, and
`pipeline._make_batch_eval` (warm+want_grad) gates `L→-inf`, drops it from `n_bad`, and
pins its carried state when `accept_count >= count_max`. Cold init phase 2 shares the same
evaluator but is unaffected (its survivors re-converge at ~zero increment, so
`accept_count ≪ count_max`). Regression-tested in `tests/test_warm_reject.py` (3 tests on
the real smoke pipeline); full suite 17/17 green.

## Mutation sweep cost — the <24 h rework (2026-07-09, after job 64745)

Job 64745 (N=48, 12 sweeps/stage, 44 h governor) sat >3 h in SMC stage 1 at 100%
util / ~300 W: every early-ladder sweep step was gated at the full `count_max=5000`,
because a warm MALA proposal had NO cap of its own and ~30% of prior mass is
non-convergent — with 48 proposals per sweep, P(≥1 bad) ≈ 1 while the cloud is
prior-like, and the full-width lockstep while_loop runs at the slowest lane. On top of
that the warm gradient ran the chemistry TWICE (a separate primal-only diag solve just
to read accept_count). Projected ~3-6 h/stage × 15-40 stages ≫ any wall. Fixes (all
wired, suite 18/18):

- **`warm_count_max` = 1500 (schema default, banner-shown).** Warm mutation solves run a
  TWIN runner whose `_Statics` snapshot the smaller cap (`vulcan_chem.build_chem_model`
  builds it by temporarily setting `cfg.count_max`; `converged_y(..., warm_cap=True)`);
  the gate in `pipeline._make_batch_eval` rejects at `ACC >= warm_count_max`.
  MEASURED (smoke chain, same day): a MALA-small warm move needs ~780 accepted steps —
  the **conv_step=500 longdy certification window dominates the warm floor**, not
  count_min — so the first-guess cap of 800 would have rejected typical GOOD proposals;
  1500 gives ~2x margin and still cuts the gated worst case 3.3x vs 5000. A
  tangent-extrapolated warm start (seed the proposal solve from `Y + dy·Δθ` using the
  jvp tangents the gradient step already computes and currently discards) converged the
  same move in ~470 steps with the same column to 9e-3 dex — wiring that into the
  mutation carry would allow ~800 again (a further ~1.65x; not yet implemented).
  Proposals that would converge in (1500, 5000] become extra MH rejections — a valid
  kernel either way. Cold/two-stage solves keep `count_max` (init phase 1 and the
  two_stage_z stage-2 increment genuinely need it). `warm_count_max > count_max` raises
  (schema + build).
- **accept_count rides the jvp chain** (`chem_solve_warm_diag` IS the warm gradient
  solve now): it is part of the runner's primal carry, integer-valued (tangent-free;
  stop_gradient + cast inside `_chain`). The duplicate diag while_loop is gone — ~2× on
  the chemistry per sweep step.
- **6 sweeps/stage** (was 12; schema + gpu preset). Published MALA-within-SMC practice
  is 3-10 with a good preconditioner; each sweep costs one full batched gradient.
- **N=96, `smc_rt_vjp_chunk=12`** at nu_pts=1652 (96/12 = 8 serialized RT chunks, same
  count as the old 48/6; chemistry is full-width so N is nearly free; 96 final
  particles instead of 48 for a 10-D posterior). Run `PROBE_MEMORY=1` once before the
  first production submit after ANY nu_pts / chunk / N change.
- **Per-sweep heartbeat**: `_make_mutation` logs `sweep k/n: accept= rejected= n_bad_grad=`
  via `jax.debug.callback` — a slow stage is visible as it happens, never hours of silence.
- **Walltime: 24 h PBS / 20 h governor** (gpu preset). Projected ~15-25 min/stage after
  the fixes; `CALIBRATE_ONLY=1` (~1 h) gives timing.json before committing a run.
- **fp32 considered and REJECTED** (Isaac: only if much faster — it isn't): chemistry
  must stay f64 (VULCAN-JAX numerical-hygiene rule; rate constants span ~50 dex), and
  the RT is not the dominant cost, so fp32-RT is <2× on a minority term. Precedent
  exists (ExoJAX Gl229B ran fp32) if the RT ever dominates.

## Warm-vs-cold validation + exactness hygiene (2026-07-09, external-review response)

The warm mutation kernel's likelihood is history-dependent at the convergence
tolerance, so it is only approximately pi_beta-invariant — the one substantive point
from the external review. The measurement tool is `validate_warm`:

```
SMC_RETRIEVAL_PRESET=gpu python -m retrieval_framework.validate_warm runs/w39b_smc_retrieval
```

It cold re-solves the checkpointed cloud (init phase-1-equivalent, ~minutes) and
compares against the warm-carried logL. PASS gate: max|dlogL| < 0.1 over the cloud
(tolerance predicts ~1e-2; the cloud's logL spread is ~n_dim/2 ≈ 5). Run once per
production run; quote the result in the paper together with the init reject fraction
(the operational prior is p(theta | chemistry converges)). FAIL exits nonzero and
says what to do (tighten yconv_cri or rerun `smc_chem_mode="cold"`).

Same pass fixed: pilot-tuner PRNG-key reuse (dormant — only the non-default
`mcmc_auto_tune and not mcmc_stage_adapt` path; now `fold_in`-decorrelated), the
silent nonfinite-L floor + logZ-increment skip in the SMC loop are now raises (both
unreachable in a healthy run — invariant checks, not normalizations), and plot_smc
stamps `[TEMPERED beta=...]` on corner + spectrum figures when a governor-stopped
run hasn't reached beta=1. Rejected from the same review, with measured reasons (see
memory/README): steady-state-adjoint gradient swap, stiffness bucketing, delayed
acceptance (all void under the lockstep per-step cost model), fp32 RT, exojax
unpinning, remat retuning.

## RT resolution — R~1000 (nu_pts~1652) is the MEMORY-SAFE DEFAULT (this keeps biting)

**RULE: keep R~1000, i.e. `nu_pts`~1652 for the production band. The RT-vjp gradient
memory scales with `nu_pts` (the absolute point count — NOT R; a narrow-band smoke can be
high-R at tiny nu_pts). NEVER raise `nu_pts` without `PROBE_MEMORY=1` first.** This has
OOM'd the run more than once, so as of 2026-07-09 it is enforced, not just documented:
- `config_schema.Config.nu_pts` DEFAULT is now **1652** (was 6000, itself a ~R10000 memory
  bomb). Any preset that forgets to set it gets the safe value.
- `validate_config` WARNS loudly when `nu_pts > 2500`, pointing at `PROBE_MEMORY`. The
  `runs/*/overrides/r3000_*.json` files (`nu_pts=5000`) trip this on purpose — they are
  experimental high-res configs and need a lowered `smc_rt_vjp_chunk` to fit.

History: the first SYNTH run past the reject-cull fix (job 64601) died in init phase 2 (the
RT-vjp gradient), allocating **343 GiB on the 96 GB GH200**, because `gpu_config` used
`nu_pts=16500` (native ~R10000). The OLD init masked it (it raised at phase 1 before
reaching the phase-2 gradient); the reject-cull fix exposed it. `nu_pts=1652` drops the
RT-vjp to ~34 GiB → fits the ~81 GiB pool (0.90 × 96 GB) with wide margin at the default
chunk. The data is ~150 binned points, so R~1000 (~11 model pts/bin) is ample; 16500 was
overkill. `overwrite=True` regenerates synthetic obs at the new resolution automatically.
`case.py::gpu_config` still sets `nu_pts=1652` explicitly (belt and suspenders).

**VULCAN-publication check (2026-07-08):** yconv_cri=0.01 + slope_cri=1e-4 are the
published Tsai+2017 steady-state values; the ballooning is the published adaptive-step
behavior; priors trace to Tsai 2023; baseline Kzz = GCM file (5e7 deep, Tsai-consistent);
metallicity/photochem match Tsai 2023. The `dt_max` cap is a documented deviation from
upstream's default that PRESERVES the longdy-defined steady state (truth bit-identical).

## Which parameter space fails (determined 2026-07-08)

- **T-P window failures = the HOT DEEP atmosphere**, not cold corners (cheap numpy Guillot
  sweep). High Tirr + low γ (strong greenhouse) push the deep (7 bar) layer above 3000 K.
  Under the new priors ~5.6% of draws are rejected (fine for the redraw sampler);
  physically those are implausible for W39b anyway. `too_cold` is ~0%.
- **Convergence tail (>10k steps)** is a *separate* axis (stiff transients relaxing from
  the ~1100 K baked baseline to a hot in-window T). Determine it empirically with the
  count_max calibration (`CALIBRATE_COUNT_MAX=1`, `CALIBRATE_COUNT_MAX_PROBE=5000`), which
  logs the per-draw θ of every censored draw. Expect it to be much smaller now that the
  hottest (out-of-window) draws are rejected before the chemistry and yconv_cri is 0.01.

## Layout / entry points

- Framework: `retrieval_framework/` (planet-agnostic). Cases: `runs/<case>/case.py`.
- Run from the bundle dir: `python -m retrieval_framework.run_smc runs/w39b_smc_retrieval`
  (also `calibrate_count_max`, `probe_memory`, `smoke_retrieval`, `plot_smc`,
  `validate_warm`).
- `config.py` / `zco_lib.py` roots are portable via `$VULCAN_PROJECT_ROOT`.
- Figures still go to `../jax_paper/figures/`; never modify `../VULCAN-JAX`.
