# CLAUDE.md — vulcan-retrieval operational notes

Critical rules and decisions for this repo. Read before touching the retrieval or
running anything on the supercomputer.

## 2026-07-11 scientific-correctness pass — EVERY cache/checkpoint is stale

Audit-response changes (README "Scientific-correctness pass" + module docstrings):
exact-elemental abundance map (`abundance_mode="elemental"`, schema default — lnZ/c_o
are now exact conserved-ratio directions, atom_ini exact, Σn=P/kT at init), per-proposal
atmosphere rebuild (Dzz(T,M)/vm/vs + pv.Kzz + initial carry geometry; conden refuses
T-varying builds), H2-He CIA REQUIRED in every RT call (vmr_he is a required arg),
broadening knob ("air"/"h2he"), evidence reported as OPERATIONAL-prior conditioned with
measured support fraction (`logZ_box = logZ + ln f` in results/npz; init cull counts
persisted through checkpoints), per-sweep/stage `warmcap=` counters, tempered-draw
labels on every output path, validate_warm gates on logL + spectrum-ppm + inventories,
jwst_tool v5 (floor-aware transits, R=100-anchored floors, offset-marginalized detect,
saturation-consistent Fisher; `_VERSION=5` -- the tool now lives in the SIBLING
vulcan-jwst-tool repo). This repo is the standalone vulcan-retrieval package since
2026-07-11 (`pip install --no-deps -e .`); see "Layout / entry points" below.

Operational consequences:
- **The chemistry map changed** (elemental + atm rebuild): synthetic obs, demo npz
  caches (`data/*.npz`), zco/Fisher caches, jwst_tool model_cache, and ALL SMC
  checkpoints are STALE. Regenerate; do NOT resume a pre-pass checkpoint into the new
  map (likelihoods re-anchor mid-run). `overwrite=True` handles synthetic obs.
- **Before the next production run**: `PROBE_MEMORY=1` (the evaluator gained small
  per-proposal structure rebuilds), then the smoke chain + suite, then on the GPU node
  the new validation set: `validation/elemental_audit.py`,
  `resolution_ladder.py`, `top_pressure_ladder.py --extend-chem`,
  `broadening_ab.py`, and post-run `validate_warm` + `mala_reversibility.py`.
- **`h2he` broadening** downloads separate `<db>_h2he` line-list caches on first use
  (network / NAS proxy); default stays "air" until broadening_ab.py is run and judged.

## Supercomputer sync — git pull for CODE (preferred, 2026-07-10), scp for DATA

**Code updates: `git pull` on the NAS front end** (all repos are public on GitHub and
every local change is committed + pushed, so GitHub is always current). Since the
2026-07-11 sibling-repo split there are TWO repos to pull for the retrieval (the
jwst tool is local-only and never deployed to NAS):

```
cd /nobackup/imalsky/VULCAN_W39b_HPC/vulcan-retrieval
git pull --ff-only
cd ../VULCAN-JAX
git pull --ff-only
```

No manual install step after a pull: the PBS preflight `pip install --user --no-deps
-e`'s both VULCAN-JAX and vulcan-retrieval from the synced trees (idempotent) and
HARD-fails if a stale user-site vulcan-jax shadows the clone.

One-time setup (front end). MEASURED 2026-07-10 on cghfe02: **direct https to
github.com works and the proxy hostname does NOT resolve** -- make sure
`https_proxy`/`http_proxy` are UNSET for git (the "Could not resolve proxy" failure
mode), no proxy exports needed:

```
cd /nobackup/imalsky/VULCAN_W39b_HPC
unset https_proxy http_proxy
git clone https://github.com/imalsky/vulcan-retrieval.git
git clone https://github.com/imalsky/jax-vulcan.git VULCAN-JAX
```

**2026-07-11 sibling-split migration from an existing vulcan_exojax_run tree**: clone
vulcan-retrieval as above, then MOVE the seeded data across (no re-transfer needed):

```
mv vulcan_exojax_run/data/opacity_cache vulcan-retrieval/data/
mv vulcan_exojax_run/data/exojax_linelists vulcan-retrieval/data/
```

The old vulcan_exojax_run clone can then be parked/removed.

- The **`VULCAN-JAX` clone target name is load-bearing** (the PBS preflight hard-codes
  it; the GitHub repo is named `jax-vulcan`). Same for `vulcan-retrieval` (repo and
  clone name match).
- The NAS clones are **read-only deploys**: never edit there; `--ff-only` guarantees a
  pull can never merge; run outputs / PBS `.o` files / caches are all gitignored so the
  tree stays clean.
- **Data is NOT in git** and needs a ONE-TIME seed into a fresh clone —
  `data/opacity_cache/` (preflight ERRORS without it) and `data/exojax_linelists/`
  (else re-downloaded via the proxy). Move from the old tree as above, or scp once
  from local.

**scp fallback / data transfers** (also if git https is ever blocked), **exactly** this
style (one command per dir, no backslashes):

```
scp -r -oProxyCommand='ssh imalsky@sfe6.nas.nasa.gov ssh-proxy %h' [local dir] imalsky@pfe.nas.nasa.gov:[hpc location]
```

- **Never use rsync.** Never make tarballs.
- **Never** wrap remote commands in `ssh nas '...'`. Give the transfer, then the
  commands to run **while logged in on the node** (`cd …`, `qsub …`) as plain commands.
- `PROJECT_ROOT` is currently `/nobackup/imalsky/VULCAN_W39b_HPC`; the PBS preflight
  requires both trees under it.

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

- **count_max = 5000, always** (lowered from 10000, Isaac 2026-07-08). Set in `vulcan-retrieval/runs/w39b_smc_retrieval/case.py::gpu_config`;
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
`vulcan-retrieval/tests/test_init_state.py` (5 tests: reject, cull, raise-if-too-few,
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
pins its carried state when `accept_count >= count_max`. Regression-tested in
`tests/test_warm_reject.py` on the real smoke pipeline.

**Init phase 2 must run UNCAPPED (learned from NAS job 64854, 2026-07-10).** The claim
that phase 2 "is unaffected because survivors re-converge at ~zero increment" was WRONG
once the cap tightened to `warm_count_max=1500`: a marginal survivor (slow phase-1
converger / stall-fallback certification) can need >1500 accepted steps just to
RE-CERTIFY convergence from its own converged column (the criterion is time-based --
unchanged vs the run at half its integrated time -- so a fresh warm restart of a wobbly
column re-pays the certification window). Job 64854: 5/96 healthy survivors gated at the
cap -> mislabeled "non-finite forward" -> spurious "RT/AD problem" RuntimeError; the tell
was phase-2 wall time sitting exactly at the cap (~780 s ≈ 1500 steps). Fix: phase 2 uses
`batch_eval_init_vg` (`_make_batch_eval(..., mutation_cap=False)` ->
`chem_solve_warm_diag_full`, gated at the cold `count_max`) -- survivors are
proven-convergent particles, not disposable proposals. Mutation proposals keep the
`warm_count_max` cap unchanged. Cost: phase 2 can run to ~5000 steps (~40 min) instead of
~13 min when a marginal survivor is present -- once per run, and correctness is not
negotiable. Regression: `test_warm_reject.py::test_init_eval_is_uncapped`.

**...and uncapped is still not enough: cull-and-backfill (job 64897, same day).** With
phase 2 uncapped, 3/96 survivors STILL died -- these columns certify cold from baseline
within 5000 steps but cannot RE-certify warm from their own converged column even in
5000 (marginal oscillators / stall-fallback certifications re-pay the time-based window
on restart and lose). A repeatable class (5/96, then 3/96 on a different seed), so it is
handled like phase 1 handles non-convergence: **phase 2 now evaluates
`N + init_phase2_spare` survivors (default spare 8; width ~free in lockstep), culls the
re-certification failures, and backfills from the spares** -- part of the operational
prior, logged loudly, reported alongside the phase-1 reject fraction. The init eval
threads the accept counts out so a TRUE RT/AD death (non-finite forward with a
NON-exhausted count) still raises. PROBE_MEMORY now probes the widened init eval (the
widest gradient batch in the run: N+8 = 152 at the gpu preset, projected ~80.5 GiB).
Raise `init_phase2_spare` only with a probe. Tests: `test_init_state.py` (cull/backfill,
RT-death raise, spares-exhausted raise).

**warm_extrapolate is ON in the gpu preset (Isaac, 2026-07-10).** The schema default
stays False; the gpu preset enables it, so the staged CALIBRATE -> SYNTH -> production
sequence exercises it end-to-end (and validate_warm gates the result) before real data.
`warm_count_max` stays 1500 -- drop toward ~800 only after the per-sweep heartbeat's
rejected-counts confirm typical warm solves sit well under it.

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
  1500 gives ~2x margin and still cuts the gated worst case 3.3x vs 5000.
  Proposals that would converge in (1500, 5000] become extra MH rejections — a valid
  kernel either way. Cold/two-stage solves keep `count_max` (init phase 1 and the
  two_stage_z stage-2 increment genuinely need it), and so does the INIT phase-2
  gradient pass (see "Init phase 2 must run UNCAPPED" above — job 64854).
  `warm_count_max > count_max` raises (schema + build).
- **`warm_extrapolate` (opt-in, WIRED 2026-07-10, default off).** Seeds each proposal's
  warm solve at the first-order prediction `Y + (dy/dθ)·Δθ`, where dy/dθ = the converged
  column's parameter tangents read off the SAME jvp lanes that produce the gradient
  (zero extra compute; ~14 MB carried at N=96; `y_tangents` added to the checkpoint —
  resuming an extrapolated run from a tangent-less checkpoint raises). The seed's refs
  are set to the PROPOSAL's (lnZ, c_o) so the solver's refs-rescale is a no-op — the
  no-double-scaling recipe; getting this wrong silently double-applies the composition
  shift, which is why `tests/test_warm_extrap.py` pins seed-vs-plain likelihood parity
  on the real smoke chain. Measured: ~780 → ~470 warm steps (1.65x) on a MALA-small
  move. Flag off compiles today's exact kernel (trace-time gating). VALIDATION before
  production use: one `SYNTH=1` A/B (same seed, flag on vs off — same posterior, faster
  sweeps), then optionally `warm_count_max` → ~800 for the second half of the win.
- **accept_count rides the jvp chain** (`chem_solve_warm_diag` IS the warm gradient
  solve now): it is part of the runner's primal carry, integer-valued (tangent-free;
  stop_gradient + cast inside `_chain`). The duplicate diag while_loop is gone — ~2× on
  the chemistry per sweep step.
- **6 sweeps/stage** (was 12; schema + gpu preset). Published MALA-within-SMC practice
  is 3-10 with a good preconditioner; each sweep costs one full batched gradient.
- **N=96 (2026-07-09) → 144 (2026-07-10), `smc_rt_vjp_chunk=12`** at nu_pts=1652
  (8 serialized RT chunks at 96, 12 at 144; chemistry is full-width so N is nearly
  free — the wattage evidence and memory projection live in "GPU power headroom"
  below). Run `PROBE_MEMORY=1` once before the
  first production submit after ANY nu_pts / chunk / N change.
- **Per-sweep heartbeat**: `_make_mutation` logs `sweep k/n: accept= rejected= n_bad_grad=`
  via `jax.debug.callback` — a slow stage is visible as it happens, never hours of silence.
- **Walltime: 24 h PBS / 20 h governor** (gpu preset). Projected ~15-25 min/stage after
  the fixes; `CALIBRATE_ONLY=1` (~1 h) gives timing.json before committing a run.
- **conv_step 500 → 300 probed and REJECTED (2026-07-10, measured).** conv_step is the
  convergence ring depth; the criterion itself is time-based (y unchanged vs the run at
  t·st_factor=0.5, lookback clamped to the ring). Smoke-chain probe (same small MALA
  move): extrapolated warm 472 → 472 steps (ZERO saving — the ring never binds once the
  seed starts at the answer), plain warm 779 → 722 (7%), cold 4484 → 2885 — and that
  cold "saving" is the tell: the 300-window certifies a state that differs from the
  500-certified one by up to **0.072 dex**, 7x the yconv tolerance. It is not the same
  answer faster; it is a less-converged answer, which inflates exactly the warm-vs-cold
  path dependence validate_warm gates. Unlike dt_max (validated state-preserving), this
  changes the certified state — keep the master default 500. The step-count lever is
  warm_extrapolate (+ warm_count_max→800 after its A/B), not the certification window.
- **fp32 considered and REJECTED** (Isaac: only if much faster — it isn't): chemistry
  must stay f64 (VULCAN-JAX numerical-hygiene rule; rate constants span ~50 dex), and
  the RT is not the dominant cost, so fp32-RT is <2× on a minority term. Precedent
  exists (ExoJAX Gl229B ran fp32) if the RT ever dominates.

## Warm-vs-cold validation + exactness hygiene (2026-07-09, external-review response)

The warm mutation kernel's likelihood is history-dependent at the convergence
tolerance, so it is only approximately pi_beta-invariant — the one substantive point
from the external review. The measurement tool is `validate_warm`:

```
SMC_RETRIEVAL_PRESET=gpu python -m retrieval_framework.validate_warm vulcan-retrieval/runs/w39b_smc_retrieval
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

## GPU power headroom → N=144 + XLA A/B candidates (2026-07-10)

Reading the GH200 monitor correctly: nvidia-smi "100% util" only means SOME kernel was
resident each sample — the WATTAGE is the honest saturation signal (700 W cap). Job
64854's trace: ~290-300 W during the 192-lane init-1 primal, ~360-390 W during the
672-lane gradient pass. That jump is the load-bearing observation: **batch width fills
the GPU**; the sequential solver chain itself cannot be shortened by idle silicon (a
step can't start before the previous finishes), so headroom converts to WIDTH
(statistics) or to per-step LAUNCH-OVERHEAD reduction (speed) — nothing else.

- **Width: gpu preset raised N 96 → 144 (2026-07-10, "more aggressive" per Isaac).**
  Chemistry rides ~free; RT-vjp goes 8 → 12 serialized chunks of 12 (tail ×1.5).
  **MEASURED (probe job 64944): peak memory is WIDTH-INDEPENDENT** — FULL cold_vg at
  N=144 and FULL init_vg at 152 are both **73.25 GiB, byte-identical to N=96**. The
  peak lives inside the fixed-width RT-vjp chunk stage; the chemistry tangent buffers
  (~0.13 GiB/particle) are freed before it and only become the peak owner near
  N≈500. So N buys memory-free width; its only real cost is the serialized RT chunk
  count (N/12). **N=192 is memory-viable** (16 chunks, RT tail ×2 vs 96) if more
  particles are ever wanted. PROBE_MEMORY=1 stays REQUIRED after any N / chunk /
  nu_pts change — nu_pts and rt_vjp_chunk DO move the peak.
- **Speed: two untested XLA A/B experiments** (launch-overhead reduction; judged purely
  by `t_mutation_sweep_s` vs a baseline calibration — they change scheduling, not math;
  the PBS XLA_FLAGS line is `${XLA_FLAGS:-...}` so qsub -v overrides it cleanly):
  `qsub -v CALIBRATE_ONLY=1,XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=4' run_nas_w39b.pbs`
  `qsub -v CALIBRATE_ONLY=1,XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL,WHILE' run_nas_w39b.pbs`
  If a combo crashes or shows nothing, discard it; if command buffers capture the
  while_loop body as a CUDA graph, every stage gets faster at identical physics.
  Adopt a winner by editing the PBS default, with a note here.

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

## Layout / entry points (standalone sibling repo since 2026-07-11)

- THIS repo (dist vulcan-retrieval, import `retrieval_framework`, src layout) is a
  sibling of `VULCAN-JAX/` and `vulcan-jwst-tool/` under the project root: the SMC
  framework at `src/retrieval_framework/`, the shared forward-model engine at
  `src/retrieval_framework/forward/` (config, vulcan_chem, exojax_rt, interp_map,
  sensitivity -- import as `from retrieval_framework.forward import config`).
  Cases: `runs/<case>/case.py`. Also `tests/`, `examples/` (ex sensitivity_demo),
  `validation/`, `scripts/zco_information/`.
- The jwst tool lives in the SIBLING `vulcan-jwst-tool/` repo (dist vulcan-jwst-tool,
  import `jwst_tool`, console script `jwst-tool`); it depends on this dist for the
  forward engine and runs locally only (never deployed to NAS).
- Install editable (`pip install --no-deps -e .`; --no-deps because vulcan-jax is
  TestPyPI-only). The old sys.path "bundle import contract" is GONE: vulcan_chem no
  longer inserts VULCAN-JAX/src, so vulcan_jax resolves via its own (editable)
  install -- the PBS preflight installs and hard-checks this on NAS.
- Import order is guard-enforced: `retrieval_framework.forward.vulcan_chem` raises if
  exojax was imported first.
- Run from the repo root:
  `python -m retrieval_framework.run_smc runs/w39b_smc_retrieval`
  (also `calibrate_count_max`, `probe_memory`, `smoke_retrieval`, `plot_smc`,
  `validate_warm`). Suite: `python -m pytest tests -q`.
- `data/` = INPUTS at the repo root (cm24 obs + NAS-seeded opacity caches);
  `output/` = GENERATED npz caches (config.OUTPUTS, gitignored). Roots are portable
  via `$VULCAN_PROJECT_ROOT` = the directory CONTAINING this repo (forward/config.py
  raises loudly if the data tree is missing).
- Historical/design-log content lives in per-directory `notes.md` files (user
  convention, 2026-07-11); READMEs carry current usage only, one per repo.
- Figures still go to `../jax_paper/figures/`; never modify `../VULCAN-JAX`.
