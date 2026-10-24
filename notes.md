# vulcan-retrieval notes

What is true in this repo and why, written for Claude: the results of the WASP-39b
runs and studies, each keyed by its name with its settings and numbers, the measured
evidence behind the CLAUDE.md rules, design records, superseded designs, ideas not
adopted and a do-not-retry register; git history is the log. Open items are in
`todo.md`. The development studies' outputs lived in `analysis/` on NAS (deleted); a
reference to `analysis/<name>`, or to `<name>` in notes.md, in code comments or run
specs means the `<name>` entry here, or in VULCAN3's / vulcan-forward's `notes.md`
(solver and RT findings). [M] measured, [E] estimated or modelled, [I] inferred.
sigma = posterior standard deviation, L = log-likelihood.

## Current state

- Version 0.28.0: the v5 model options (T at a deep level in place of Tint, the window
  on the chemistry grid, `s_o`, a Gaussian `lnM`, the Kzz power law, the grey deck and
  the adiabat; "Design records"), the rejected-draw log and the certificate's derived
  quantities. 0.27.0 removed the SMC path and added the nautilus report (the last commit
  with SMC is `397d8bb`; "Superseded designs"). pyproject floors: vulcan-forward>=0.30.0,
  vulcan3>=0.18.0. `requirements-hpc.txt` pins vulcan3 `490e822`, vulcan-forward
  `c5770a4` (the v5 code; v4 ran on `a420570` / `acdf63e`), jax 0.6.2, exojax 2.6.0 and the stock nautilus-sampler 1.0.6; `validate_env`
  refuses the former fork (same version string) and fails on stale metadata after a
  version bump until all three trees are reinstalled editable. TestPyPI releases, tags
  confirmed by the maintainer: vulcan3 0.18.0, vulcan-forward 0.29.0, this repo 0.27.0,
  jwst-transit-authority 0.72.0.
- Production: `run_nautilus`, the only sampler, warm starts from a growing anchor set,
  solver `ffi` (the spec's `runtime` block), one chemistry program per draw (v5: rtol
  0.05 within 5000 accepted steps; v4 ran the presolve chain, "Superseded designs"),
  per-instrument noise inflation and log10gamma down to -3: 15 parameters in v4 (13
  before its override file's additions); v5 samples 12 (s_o and lnM added, T1bar in
  place of Tint, the 2 offsets and 3 inflations integrated out: `integrate_nuisance`). Last finished run: baseline_v4 (job 94946, 23.82 h, exit 0; "Runs"), never
  quotable: its outputs were deleted before a warm-vs-cold re-solve (the maintainer
  decided). The headline run is v5,
  `runs/w39b_smc_retrieval/wasp39_test1.yaml` (run name `wasp39_test1`; the
  external-review configuration; "Design records"), then four sensitivity variants
  beside it; launch: `PBS_GROUP=<allocation>
  python tools/submit.py runs/w39b_smc_retrieval/wasp39_test1.yaml` ("Design records",
  run specification). Forward checks at a run's draws:
  `validation/posterior_forward_check.py`. A second case, WASP-107b (`runs/w107b/`, three
  specs; "Design records"), runs the same model on the 2024 JWST data.
- Run report: `run_nautilus` writes `nautilus_certificate.json` when a run finishes
  (`nautilus_report.write_certificate`); `python -m retrieval_framework.nautilus_report
  <out_dir> [--spectra N]` rewrites it and draws the run figures (the spectrum fit from
  N weighted draws re-solved cold, default 32; "Design records", run certificate and
  run figures). First used on a real run: baseline_v4.
- Tests: the normal tier (plain `pytest tests`; CI runs it serially on Python
  3.10-3.12) deselects `slow` through pyproject's addopts. Full tier
  (`-m "slow or not slow"`; slow: `test_cold_reject`, `test_condensation_live_tp`,
  `test_set_observations`' production preflight and `test_v5_options`, every v5 option
  on the real smoke pipeline). Counts [M]: the v4 tree, daw CPU `-n 16`, normal 79
  passed / 1 skipped in 18 s, full 88 passed / 2 skipped in 4.5 min; the v5 tree
  without the presolve chain, laptop CPU `-n 8`, normal 82 passed / 1 skipped in 15 s.
  With the chain still in, `test_v5_options` and `test_cold_reject` passed (2 in 15.6
  min of CPU). The full tier has not run on the tree without the chain (owed before
  the v5 launch, `todo.md`).

## Runs

One entry per run: job id, settings, numbers as run; posterior medians with [16%, 84%]
unless stated.

### fixedmass (job 91942) and wide
- fixedmass: logZ_box 1159.599, n_eff 10,098; max-L 1194.345, chi2 196.6 over 150 bins.
  Wide run: logZ 1160.98, metallicity 1.09 [1.01, 1.73], C/O 0.31.
- In literature units (comparison), median [68%]: log10 O/H 1.06 [0.98, 1.16] x solar
  (Lodders 2020; +0.04 dex for Asplund 2009), C/O 0.33 [0.23, 0.48], log10 Kzz(1 mbar)
  9.39 [8.75, 10.21] cgs, limb T (0.1-10 mbar) 888 [817, 966] K, SO2 (0.01-10 mbar mean)
  log10 VMR -5.41 [-5.60, -5.21], peak -4.40 at ~1.8 mbar; H2O -2.20, CO2 -3.49, CO
  -2.52, CH4 -9.2; offsets NRS1 118 [94, 140], NRS2 91 [48, 132] ppm.
- Literature tensions (agent-extracted values; "Savel et al. 2026" authors unchecked):
  metallicity within 1 sigma of Tsai, Carter & May, Constantinou 2023, Powell 2024
  +2.3 sigma, Chen 2025 -7.2 sigma; C/O below VIRA 0.83 (-3.1 sigma), Rustamkulov and
  Savel 0.68 (-1.7 sigma); SO2 consistent with Tsai, Alderson, Powell (Gruebel -1.9 sigma).
- failure_diag (jobs 92255, 92259, 92278): the low-Kzz ridge of rejections came from
  upper-atmosphere trace species (O(1D), SO2, C, NS, CN at layers 45-52, ~4e-7..3e-8
  bar, VMR 1e-13..1e-8) that never meet VULCAN's longdy/aflux test within the cap. Not
  the element budget (relaxing it rescues nothing), not too few steps (40000 steps: 0/4),
  not use_vm_mol. Stuck columns are spectrally stationary (L spread median 0.06-0.08).
- figdata: 2 of 24 rejected draws near the posterior converge at count_max 20000, best
  recovered L 1192.57 < max-L 1194.35: no hidden better fit.

### baseline_v2 (job 93269, tag w39b-baseline-v2)
- Override file `overrides/w39b_baseline_v2.json`: removed from main with the
  retry-only mode ("Superseded designs"); it is at tag `w39b-baseline-v2`.
- Code: VULCAN-JAX d7021cb, vulcan-forward d26e9d3, vulcan-retrieval 29aa87e. Retry C
  two-pass chemistry (rtol 0.1 to 3000 steps, then rtol 0.2 to 5000), per-instrument
  noise, log10gamma down to -3, 792 lanes x n_batch 2376, compaction [264, 132], the
  nautilus fork (SiLU networks + spreading, see "Sampler").
- 17.4 h, 63 batches, 147,680 calls, n_eff 5111, logZ_box 1159.120 [M].
- Posterior as run: lnZ 0.293 [0.059, 0.607] (1.13 [1.03, 1.26] x solar); c_o -0.370
  [-0.761, 0.004] (C/O 0.38 [0.26, 0.55]); lnKzz -0.069 [-1.566, 2.200]; Tirr 970
  [917, 1045] K; log10kappa -0.042 [-0.362, 0.205]; log10gamma -2.173 [-2.612, -1.728];
  Tint 181 [125, 251] K; lnR0 0.0302 [0.0286, 0.0315]; log10kappa_cloud -3.042
  [-3.405, -2.782]; cloud_alpha 1.569 [1.164, 2.163]; offsets NRS1 111.9 [88.5, 134.1],
  NRS2 70.8 [24.3, 116.4] ppm; noise inflation NIRISS 1.242, NRS1 0.920, NRS2 1.273.
- logZ error: statistical ~+/-0.015 [I]; known offsets spreading ~-0.02, tolerance
  correction -0.028, cold start -0.040.
- Problems found (postrun_93269) [M]:
  1. rtol-0.2 retry columns are not converged at 0.1 nat: vs rtol 0.05 the median |dL|
     is 0.28-0.35 (pass-1 rows: 0.07 at low Kzz, 0.012 at high Kzz).
  2. Warm-start path dependence: 97 of 1320 posterior rows certify warm but fail cold
     (lnKzz -1.9..-1.1); net dlogZ -0.040.
- Correction (the postrun_93269 method): all 1623 retry rows and 6265 pass-1 rows at
  lnKzz < -0.7 re-solved at rtol 0.05 / cap 15000, reweighted by exp(L_0.05 - L_run),
  uncertified rows w = 0. Every median moves <= 0.035 sigma (lnKzz -0.069 -> -0.002,
  c_o -> -0.357, Tirr -> 972); 413/7888 rows rejected; mass at lnKzz < -0.7 0.353 ->
  0.335; ESS 5324; logZ 1159.092. Saved next to the run:
  `data/w39b_baseline_v2_nautilus/corrected_posterior_rtol005.npz` (theta, w_as_run,
  w_corrected, L_rtol005, cert_rtol005) and `.json`.
- vs fixedmass: log10gamma -2.17 vs -1.73 (64% of the mass below -2 once the prior
  reaches -3); lnZ +0.73 old sigma; logZ -0.48 (Occam penalty of the wider prior ~-0.36);
  the fixedmass lnKzz dip at -1.3..-1.1 is gone.

### baseline_v4 (job 94946)
- Code: VULCAN-JAX a420570, vulcan-forward acdf63e, vulcan-retrieval d850ce4 (the
  `requirements-hpc.txt` pins). Uniform 100 layers, from
  `overrides/w39b_baseline_v4_nz100.json` (in the tree at 8e3a21c; the run's config.json
  holds the resolved values; presolve chain, 528 lanes, per-instrument noise,
  log10gamma down to -3; case.py's Tint prior 50-600 K; stock nautilus 1.0.6),
  submitted from `runs/w39b_smc_retrieval` on cghfe01 with
  `qsub -l walltime=48:00:00 -v SMC_RETRIEVAL_OVERRIDES_FILE=overrides/w39b_baseline_v4_nz100.json,SMC_RETRIEVAL_OUT_DIR=data/w39b_baseline_v4_nz100,NAUTILUS_N_EFF=5000,VULCAN3_CHEM_RHS=grouped,PROFILE_BATCH=3 run_nas_w39b.pbs`.
- Gates on the same tree [M]: bootstrap 94874, `validate_env` PASS;
  `tools/gpu_kernel_test.pbs` 94875, `tests/test_solver_fast.py` 53 passed / 21 skipped
  plain and again under XLA_CMDBUF=2's flags (the CUDA cases ran), vulcan-forward
  `tests/test_ckd.py` 7 passed on the GPU. The job's start-up build found the CUDA
  library current (same stamp; nvcc 12.6 V12.6.85, sm_90), so it ran the tested build.
- Batches 1-7 [M]: cold 1568 s, then warm 729-872 s. Batch 3 profiled: "Pipeline
  speed", v4 batch profile.
- nautilus 23.82 h (job wall 24.1 h), exit 0: 131 batches, 207,504 calls, n_eff 6873,
  logZ_box 1159.098, max log L 1195.95 [M]. Certificate [M]: T-P out of window 3.5% of
  calls; of in-window draws 1.60% rejected (1.597% exhausted, 5 stalled, 0 non-finite);
  99.2% of calls warm; 63.7% of the 1584 cold draws certified; the retry pass certified
  98.4% of the rows it re-solved.
- Posterior as run: lnZ 0.306 [0.073, 0.660] (13.6 [10.8, 19.4] x solar); c_o -0.350
  [-0.736, 0.009] (C/O 0.39 [0.26, 0.55]); lnKzz -0.233 [-1.631, 2.354]; Tirr 974
  [923, 1053] K; log10kappa -0.022 [-0.363, 0.228]; log10gamma -2.187 [-2.621, -1.704];
  Tint 159 [86, 241] K; lnR0 0.0300 [0.0283, 0.0312]; log10kappa_cloud -3.031
  [-3.412, -2.768]; cloud_alpha 1.583 [1.164, 2.212]; offsets NRS1 110.8 [88.1, 133.7],
  NRS2 68.6 [23.9, 114.5] ppm; noise inflation NIRISS 1.245, NRS1 0.923, NRS2 1.266.
- Tails and prior edges [M]: the 99% quantile sits within 1% of the prior range of the
  upper edge for c_o (0.227 vs 0.24) and log10kappa (0.460 vs 0.5), the 1% quantile
  within 1% of the lower edge for lnKzz (-2.261, the x0.1 floor) and Tint (52 vs 50 K).
  Kzz factor: 16.7% of the mass below x0.2, 9.0% below x0.15, 16.2% above x10. Tint:
  5.2% above 297 K, 2.1% above 358 K. 3.1% of the mass above 54.5x solar.
- vs baseline_v2 as run: every median within 0.16 v2 sigma except Tint (-0.35 sigma,
  181 -> 159 K); logZ_box -0.022 [M].
- Quotable only after the warm-vs-cold re-solve of its posterior rows (the
  postrun_93269 method; "Physics assumptions and limitations"): the posterior and
  logZ_box above are as run.
- The profile's nsys-jax post-processing failed after the run (FileNotFoundError on a
  `protos` file under `$VULCAN_SCRATCH/tmp`); `logs/nsys_94946/pipeline_gh200_jax_profile.zip`
  was written, the job exited 0 and the retrieval output is unaffected [M].
- Report [M]: `nautilus_report --spectra 128` (125 of 128 weighted draws certified cold)
  rewrote the certificate and drew its three figures in `plots/` next to the run. A copy
  of the run without `anchors/` (20 GB) is at `~/Desktop/w39b/_build/runs/` on the laptop.

### v4_nopresolve_test (jobs 95161 + 95569, resumed)
- The v4 target with the presolve chain OFF: one program at rtol 0.05, cap 5000
  (`chem_retry` and `chem_presolve` false, `cfg_overrides.rtol` 0.05, `count_max` 5000);
  everything else v4's (100 layers, 528 lanes, warm starts, stock nautilus, n_eff 5000).
  Stopped at a 4 h wall budget at 31,680 calls and resumed: nautilus 4.06 + 15.71 =
  19.8 h against v4's 23.8 h [M, run.log].
- 205,920 calls (v4 207,504), n_eff 6511 (6873), logZ_box 1159.034 (1159.098: -0.064)
  [M]. Certificate: T-P out 4.0% of calls (v4 3.5%); of in-window draws 2.56% rejected
  (v4 1.60%), 7 stalled, 0 non-finite; 99.2% warm.
- Posterior as run, against v4: lnZ 0.305 [0.076, 0.644] (v4 0.306); c_o -0.340
  [-0.724, 0.018] (-0.350); lnKzz -0.190 [-1.634, 2.421] (-0.233); Tirr 975 (974) K;
  log10gamma -2.232 (-2.187); Tint 158 [85, 240] (159) K; lnR0 0.0299 (0.0300); clouds
  and offsets within 0.01 of v4; Kzz weight below x0.2 0.167 (0.167), below x0.15 0.097
  (0.090), above x10 0.168 (0.162) [M, nautilus_posterior.npz]. Every median within
  0.1 v4 sigma: the presolve chain moves the posterior by less than the sampling noise
  and buys 1.0 point of in-window rejection and ~0.06 in logZ_box at ~4 h of wall.
  Launched outside the documented recipe (no notes entry on NAS). On this result the
  maintainer decided v5 runs the plain program ("Design records", the chemistry program).

## Low-Kzz hole

- conv_gate: the notch came from capped solves failing only in layers 45-50 (4e-7..6e-8
  bar), far above the photosphere (> 5e-5 bar). Every gate relaxation (pressure mask,
  cap-exit acceptance, error-norm mask) DROPPED: VULCAN's own criteria stay.
- retry_policy: two-pass Retry C (vulcan-retrieval 82a0130) ~1.33x production per batch
  vs in-queue 1.45x (measured 1.39x); cap1 stays 3000. Used by baseline_v2.
- lowkzz_fix (8.3 GPU-h). Root causes [M]: (A) photolysis every 5 ticks triggers stiff
  transients (100% of reject cascades start on a photolysis tick, dt collapses a median
  6.9 dex); (B) at lnKzz < -2 and rtol <= 0.05 one near-atol cell (atomic S at 1e-9
  bar) sets the step while a C2H4/C2H2 drift at ~5e-3 bar needs 19-23k steps; (C) at 62
  layers two states really exist (see bistable_policy).
- PRESOLVE CHAIN (vulcan-retrieval cdcffd1 to `0a2dbf3`, the last commit with it;
  `Config.chem_presolve` with `chem_retry`; the 62-layer study override
  `w39b_baseline_v3_presolve.json` is in git at `397d8bb`, v4's
  `w39b_baseline_v4_nz100.json` at `8e3a21c`; not in v5, "Superseded designs"): stage A
  chemistry-only presolve at rtol 0.2, photolysis every step, cap 3000, never kept;
  stage B re-solves every in-window draw at rtol 0.05, photolysis every step, cap 5000,
  warm from A's exit column where A certified, otherwise from the row's original start.
  Only stage B's VULCAN certificate keeps a draw. On a cold chunk a draw whose presolve
  fails still starts stage B from the presolve exit column (the baseline column if that
  is non-finite).
  - Hard set: 263/264 certified vs 99 (production pass 1), 183 (retry), 203 (direct
    rtol 0.05, cap 15000); error vs reference median 0.027, p90 0.25 (retry 0.26 / 1.24).
  - Representative set: 264/264; median 0.008, p90 0.072 (production 0.022 / 0.126).
  - Production path (job 93676, 792 lanes, 2 real batches): in-window rejection 2.5% vs
    6.1% (Retry C); batch 816 vs 1082 s compile-free.
  - Dropped: direct rtol 0.05 (2.8x per batch on this hard set, which over-weights low
    Kzz: 4 h faster over a full run, "Runs", v4_nopresolve_test), master's adaptive rtol
    (4.1x), master's photolysis cadence, Ros2 polish start, stage A at rtol 0.1
    (1.7-2.2x). Flagged: nz 150 certifies 65/66 vs 48/66 at 2.36x per tick; the 2025
    network changes no certification.
- bistable_policy (3.7 GPU-h): the two low-Kzz states differ by where the photochemical
  front (1e-7..1e-6 bar) sits, by whole grid layers; they merge at 100 (35/40) and 150
  (40/40) layers: a 62-layer artifact. 11% of posterior mass has >= 2 states (none above
  lnKzz -0.7). Largest |dmu|/sigma median 0.35, p90 1.08, in the SO2 band at 4.0-4.1 um.
  Start policies: warm (status quo, 581 s/batch), cold2 chain 1.77x, eq chain 1.47x, eqd
  4.10x; every parameter shift <= 0.06 sigma, |dlnZ| <= 0.01. Kept: warm start + presolve
  (v5 keeps the warm start and drops the presolve, "Design records").

## Grid resolution (resolution_study, 10.4 GPU-h)

- Only SO2/SO change with resolution (VULCAN3 notes); in the spectrum the 4.05 um band
  moves a median 1.5 sigma per draw (62 vs 300 layers); L(300) - L(62) median -2.8.
  The 67-layer RT grid and the 1e-9 bar top are converged.
- Worst science-parameter error vs the 284-layer reference, in sigma (SE ~0.12-0.20 at
  nz <= 100): 62: 0.45 (lnZ, c_o); 75: 0.45; 100: 0.43 (c_o median; 0.26 by mean);
  125: 0.28; 150: 0.24; 200: 0.13; 300: 0.07.
- Run wall [E]: 62 layers 10.4 h, 100 16.0 h, 125 21.9 h, 150 26.1 h, 200 37.5 h,
  300 56.3 h (fork sampler with spreading; stock needs more calls, see "Sampler").
- 93269 reweighted to 284 layers (ESS 187): lnR0 -0.58 sigma, Tirr +0.29, c_o +0.19,
  log10kappa +0.18.
- Moving the bottom to 47 bar changes ~10% of Tint >= 180 K draws, 8% uncertified: not
  worth it as is.
- v4 and v5 run uniform 100 layers (the maintainer's choice; `nz: 100` in the spec).
  Lane probe (job 94050): 528 lanes peak 64.4 GiB (cold presolve) of 85.6, 4.78 draws/s;
  396 lanes 50.9 GiB, 4.59 draws/s; 660 compiled at ~79 GiB, never timed.
- 125-layer rehearsal (v4_rehearsal, 396 lanes): cold batch 1667 s, then 709-951 s;
  in-window rejection 7.0%; nsys hook overhead 1.9% in the captured batch, 0.6% elsewhere.

## Sampler

- Production: the stock nautilus-sampler 1.0.6 package (83b2980), 4 default ReLU
  networks, no spreading; settings n_live 1000, n_update n_live/2, f_live 0.03,
  enlarge_per_dim 1.3, n_batch 3 x lanes, discard_exploration True.
- nautilus_relu_test (surrogates of the real posteriors, n_eff 5000, 8-24 seeds):
  - fork (2 ReLU + 2 SiLU, spread_batch): 124.9k calls, logZ -0.023 +/- 0.016 vs truth.
  - stock 1.0.6, no spreading: 202.0k calls (+62%), logZ -0.001 +/- 0.011.
  - keeping the exploration phase: logZ -0.206: biased, do not use.
  - SiLU vs ReLU: same calls (-0.6 +/- 1.2%), logZ, marginals and mode weights.
  - Spreading biases logZ ~0.02 low (SE 0.003): optimistic adaptive-allocation bias (Shin,
    Ramdas & Rinaldo 2019); it also overstates n_eff (reported 5274, true ~1509 in one
    replay). So 93269's logZ may be ~0.02 low.
- Why nautilus (stand-in bake-offs on daw CPU, 11-D, exact truths, 5-8 seeds): nautilus
  |lnZ error| <= 0.03 and posterior bias <= 0.01 sd on every stand-in; the production
  MALA-SMC failed the funnel (|lnZ error| 2.02) and was off +0.31..+0.85 on the spectrum
  stand-ins; pocoMC missed the funnel's width; NUTS found one mode of the bimodal and
  gives no evidence. The settings above came from a real-posterior stand-in (an MLP of
  log L on 81542's 83,160 points): enlarge 1.3 cut the mean error from 0.10-0.14 sd to
  0.02-0.08 sd; 1000 live points were the most accurate row.
- smc_mala_ab: SMC + MALA DROPPED. MALA costs 2.9x an RWM sweep for 3.1x the ESJD
  (break-even per GPU-second); gradient memory caps it at ~300 lanes; projections SMC +
  MALA ~29 h, RWM > 100 h vs nautilus ~16 h. Chemistry convergence, not the kernel,
  drives the cost (36-45% of warm mutation solves hit the cap).
- sampler_options (17-D stand-in of v5 on daw CPU: 8 chemistry + 9 RT-only dims, a 9%
  narrow second mode, mass piled on the Kzz-like floor, a -inf hole, a banana; exact
  truth; 4 seeds; cost in chemistry calls, an RT-only move 0.05, a gradient 2.9) [M]:
  - stock nautilus (production settings): lnZ -0.093 +/- 0.003, the missed mode's
    share; the mode found in 0 of 4 seeds, so the seed spread hid it; 158k calls.
  - the 2 offset and 3 noise dims integrated out (12-D nautilus): lnZ +0.003 +/- 0.004,
    the mode in 4 of 4, posterior means within 0.03 sd; 121k calls.
  - adaptive-tempered SMC (RWM, MALA every step or every 5th, RT-only moves,
    waste-free) at 10 moves per stage: none within +-0.1 in lnZ at 1.5-6x the calls;
    MALA every step found the mode (lnZ +0.32). With more moves the bias goes: MALA
    every step, 60 moves, lnZ -0.033 +/- 0.072, the mode at 0.097, means within 0.12
    sd, 3.84M calls (~30x 12-D nautilus); MALA every 5th + 5 RT-only moves, 30 moves,
    +0.013 +/- 0.115, mode 0.087, 1.29M; RWM 60 moves +0.019 +/- 0.121, mode 0.048,
    1.12M. A MALA kernel started on exact draws kept them (moments within 0.04 sd).
    The SMC was untuned (fixed move counts, ESS 0.5); an outside review calls the
    family untested, not beaten.
  - 4 independent populations (the 4-GPU recipe: Z averaged), N 500 each, MALA every
    5th + RT-only moves, 30 moves/stage, 16 populations in fours [M]: 17-D combined
    lnZ +0.023 / +0.061 / -0.015 / +0.004; 12-D +0.000 / +0.183 / +0.050 / +0.157;
    268-319k call-equivalents and 390-470 sequential sweeps per population.
  Gradient moves on a schedule (every k-th step, or late stages only) are exact; a
  warm start inside an MCMC move is not (the likelihood depends on the path).
- Sampler choice (the maintainer decided, with two independent outside surveys):
  nautilus stays; no gradient sampler. At 12-D one SMC population already costs ~240k
  calls before gradients, twice 12-D nautilus's whole run, and an evidence-grade SMC
  needs ~400-500 sequential sweeps, while a GH200 runs ~330 primal or ~100 gradient
  sweeps a day whatever the population size (a sweep is one lane wave). Width (4 GPUs)
  does not buy depth. A gradient sampler would pay above ~25-30 dimensions, on a
  posterior nautilus fails, or without the evidence.

## Priors and the T-P profile

- evidence_noise_priors (job 92980): evidence stable (cold target 1159.53 +/- 0.02 vs warm
  1159.60). NIRISS drove the noise misfit -> per-instrument noise (4581e78). log10gamma
  lower bound -3 adopted; lnKzz kept at [ln 0.1, ln 1000] (no validation solve below).
- Tint prior (tint_prior), v4: uniform 50-600 K (8bc1425; was 100-500). v5 retrieves T
  at 1 bar instead ("Design records").
  - Literature: heated-interior estimates 300-360 K (Thorngren+2019 1907.07777, 316-356 K
    but calibrated above 0.5 MJ; Tsai+2023 adopt 358 K; Wilkinson+2024 2410.04470
    150-320 K, 297 +9/-17 K with G395H; Komacek & Youdin 2017 ~370 K). Cold floor
    ~55-75 K for 7-9 Gyr (Fortney+2020 2010.00146, extrapolated). Alderson+2023: 100-400 K
    grids fit equally; Kawashima & Min 2021: ~110 K, weak.
  - Why the posterior sits near the floor: the pipeline rejects profiles outside
    [320, 2980] K down to 7 bar (`tp_profile.T_EDGE_INSET_K`, `pipeline.tp_valid`).
    Guillot is radiative with one constant kappa: deep T^4 ~ 3/4 Tint^4 kappa P/g. At the
    preferred kappa (~1 cm2/g) tau(7 bar) ~ 1.6e4, so Tint 358 K gives ~4400 K and Tint
    ~280 K hits the 2980 K edge. Tsai's GCM with Tint 358 K is ~1600 K at 1 bar. Tint
    here is a knob for the 1-7 bar temperature, degenerate with kappa (Tint ~
    kappa^-1/4), and the pile-up is a window / prior-volume effect (v4's 1% quantile sits
    at 52 K against the 50 K edge, "Runs").
  - Induced prior [M, outside review, 1e6 Guillot draws over the v3/v4 prior box]:
    67.58% of draws pass the window; Tint's median moves 325 -> 278 K, 16-84%
    121-481 K. At 7 bar and kappa 1 cm2/g the deep term alone caps Tint at ~282 K.
    Read the Tint posterior against this effective prior, not the uniform one.
  - Fix options: (A) apply the window only above ~1 bar (lets 4000 K deep layers into
    the chemistry; not alone); (B) retrieve T at 1 bar instead of Tint: adopted for v5
    (`tp_deep_param="T1bar"`); (C) a physical profile: pressure-dependent opacity plus
    an adiabat below the convective boundary, e.g. Parmentier+2015 (with constant kappa
    the deep gradient is 0.25 < adiabatic ~0.29, so a convective switch alone never
    triggers). The external review's version of (C), an imposed adiabat below a free
    boundary pressure, is the `tp_adiabat` option ("Design records").
- Earlier prior edges and why they moved: "Superseded designs".

## Rate sensitivities (adjoint_w39b, 5.4 GPU-h)

- Method: VULCAN3 adjoint of log L through the RT for every rate row (~1150) on 21
  posterior columns of 93269, tight rtol-0.05 solves (`vulcan3.steady_state_grad`:
  `_adjoint_solve_core`, `steady_state_reaction_sensitivity`). Validated against
  forward-mode jvp on 212 directions: median error 0.57%, signs 100%; directions with
  > 10% of their sensitivity above 1e-6 bar are not trustworthy. The driver scripts were
  deleted with analysis/.
- Ranking, median |dlogL/dln k|: R607 OH+S->H+SO 1.66; J(SO) 1.14; J(SH) 1.03; J(S2)
  1.03; R1011 H+S2+M->HS2+M 0.94; R875 H+H+M->H2+M 0.90; R37 OH+CO->H+CO2 0.88; R1025
  HSO+M->SO+H+M 0.50; then R691 SO+OH->SO2+H 0.18. The spectrum tracks the sulfur
  partitioning feeding SO and H recombination more than the SO2-forming step.
- The data prefer a faster H+S2+M (dlnZ/dln k +0.99 +/- 0.28) and slower SH photolysis
  (-0.99 +/- 0.30).
- A factor-2 rate error biases mostly the lnR0-lnZ direction (R607: lnR0 +0.92, lnZ
  -0.52 sigma); c_o moves <= 0.17 sigma. With factor-2 thermal / 1.5 photolysis
  uncertainties, sigma_rate/sigma_post: lnZ 0.69, lnR0 1.24, lnKzz 0.42, Tirr 0.26,
  c_o 0.25: rate uncertainty is comparable to the statistical error on metallicity.
- JWST constrains at most two rates (J(SO), R607) to a factor of ~4-5.
- Caveats: 11 random draws only; linear factor-2 numbers off by up to 0.24 sigma;
  photolysis sensitivities exclude shielding changes.

## Pipeline speed

- 93269 time accounting (56,043 s through batch 55): pass-1 chemistry 38.7%, pass-1 RT
  24.6% [I], retry chemistry 24.1%, retry compiles 3.5%, host/nautilus 5.6%; chemistry
  lane time useful 35%, capped or uncertified 38%, idle 27% [I]. speed2 (all bitwise:
  masked padding rows, RT only on kept rows, VULCAN3 a5b1215 / eeb4873, vulcan-forward
  5e8d43a / 39c2c45; the retry width buckets and the host-thread warm precompile were
  later removed, "Superseded designs"): pass-1 batch 344.0 -> 287.3 s, full batch ~892
  -> 628 s excluding compiles, RT pass 256 -> 60 s; whole run ~11 h at 62 layers (~3.4x
  faster than fixedmass's 37.2 h) with the fork sampler's call count [E].
- Warm starts on run 90302 (growing anchors): 507 steps per certified warm column
  against 1,119 cold (the first batch's prior draws), 2.21x [M]; the CPU screen predicted
  2.13x. The batch time follows the capped columns: a batch with any column at the cap
  runs the full cap, so a tick costs the same whether its lanes are busy or done.
- v4 batch profile (job 94946, batch 3, warm, nz 100, 528 lanes, nsys-jax) [M]: 843 s =
  presolve program 397 + certify program 438 (its last 37 s the RT on kept rows) +
  nautilus update 8. Over the chemistry loops the GPU is active 95.7% of the wall and
  the host spends 813 s in `cuStreamSynchronize`: launch cost is not the limit. The lane
  queue compacts 528 -> 396 -> 264 -> 132 (the kernels' gridX); the 132-lane tails take
  370 s (115 presolve, 255 certify) at ~3,950 lane-steps/s against ~5,050 at 528. GPU
  time in the chemistry: block-Thomas factor 216 s and solve 121 s (44% of all GPU
  time), device-to-device copies 74 s (1.12 M), other XLA kernels 351 s, idle 34 s.
  Kernel rates: VULCAN3 notes "Efficiency program". Trace caveats: "HPC (NAS GH200) and
  daw". Speed ideas not built: "Ideas not adopted".
- v5 batch profile (1 h v5 run, batch 2, 528 lanes, kernels matched to HLO source lines)
  [M]: of 899 s chemistry GPU span, block-Thomas 41%, device copies 10% (the y ring, gone
  since VULCAN3's deferred ring write: v5 warm batch 848 -> 768 s, results byte-identical),
  photolysis ~20%, Jacobian 6%, RHS 4%. Factor waves step at 264 lanes, solve at 396.
- v5 queue replay (that run's four batches, per-draw steps, GH200 ticks) [E]: perfect
  packing would make the chemistry 1.22x faster, known-length longest-first 1.14x; capped
  draws hold 20-37% of a warm batch's accepted steps (64% cold); rungs 66/33 1.03x;
  n_batch 3168 1.10x per draw before its sampler cost; 792 lanes ~5% (memory unmeasured);
  each batch split over 2 / 3 GPUs (stock nautilus `pool` splits a vectorized batch)
  1.68x / 2.24x wall for +19% / +34% GPU-hours.
- T-P-invalid draws take no chemistry step and no RT from baseline v4 (the maintainer's
  choice; manifest_version 4): `loglike` ANDs `tp_ok` into the `active` mask before
  the first pass. 32.4% of the box prior is out of window [M, outside review]; the
  saving sits in the exploration phase and the first batches (the shells hold far
  fewer [I]). Not bitwise for the kept draws: fewer jobs on the lane queue shift the
  shared tick cadence. `batches.jsonl` records acc 0 for those rows.

## Provenance

- Commits before 2026-10-13 were made on other machines; their decision records are not
  all traceable (256 vulcan-retrieval commits, ~70 change science).
- The vulcan3 rename (882b9e5) is on main and pinned (41aacb4, then 5eb5df9);
  baseline_v2 (29aa87e) predates it, while the speed2 and presolve code (764ccb8,
  cdcffd1) that the later NAS studies ran contains it [I].
- The Kzz bounds are confirmed by the maintainer; the paper states that the x1000
  ceiling was set after seeing the data. The rest stay live with provenance not
  recorded: 3eafb1b discard_exploration and the ffi solver default (`VULCAN3_SOLVER` in
  run_nas_w39b.pbs); 3e40f03 + 4c23465 warm-start anchors and warm nautilus by default
  (results depend on evaluation order); 8e633bf + 500b340 lane queue default; a16188e
  p_ref 1 mbar, fixed-mass lnR0, chemistry Rp 1.21 RJ, free Tint, split NRS offsets,
  Tirr 700-2200; 18da2a0 Kzz ceiling x1000 chosen after seeing the data; 52ad7be/611f62b
  Kzz floor x0.1; cc6a1e9 "Snapshot" (noise inflation, correlated-k default); 24bd41b
  12 molecules, 2 bins dropped; 5792482 and d549f34 cold gate, in-graph equilibrium
  seed, XLA command buffers; f05e537 + 88f995e certification gate on stall/budget exits;
  93ed4fd elemental abundance map (redefines lnZ and C/O); c7899ee hybrid vm_mol
  2026-07-14..07-26 (reverted: check no reported result comes from that window).

## HPC (NAS GH200) and daw

- `PROJECT_ROOT` (the PBS header uses `/nobackup/$USER/VULCAN_W39b_HPC`) holds all three
  trees: `vulcan-retrieval`, `VULCAN-JAX` (the clone name is load-bearing for the
  bootstrap and `validate_env`) and `vulcan-forward`. `git pull --ff-only` is live
  (editable installs); a version, dependency or entry-point change needs the bootstrap
  (`tools/bootstrap_nas_env.pbs`).
  A new run writes to `vulcan-retrieval/output/w39b_smc_retrieval/<name>_nautilus/` with
  its logs in `output/w39b_smc_retrieval/logs/`; the v4-era outputs (the old
  convention, `runs/w39b_smc_retrieval/data/` and `logs/`) are deleted. Copy a run home with rsync, excluding
  `anchors/` (gigabytes) and, unless the conditioning map needs it, the checkpoint.
- Queues: the GH200 queues (`gpu_long` 120 h, `gpu_normal` 24 h, `gpu_devel` 2 h) are
  served by `pbs05a` and take jobs only from `cghfe01`/`cghfe02`; `cfe` answers `Unknown
  queue`; `qsub -q gpu_long@pbs05a` is rejected. Login: sfe -> `ssh -o
  PreferredAuthentications=keyboard-interactive cfe02` -> `ssh cghfe01`.
- Data is not in git: seed `data/opacity_cache/` (CIA) and `data/exomolop/`. Fetch the
  k-tables on the front end with `python3 -m vulcan_forward.fetch_exomolop --molecules
  <the spec's molecules>` (`module load python3`, `VULCAN_FORWARD_DATA` and `PYTHONPATH` set
  to the repo data and vulcan-forward/src); scp from home ran at 140 KB/s. Other files:
  `scp -r` through the sfe proxy. `unset https_proxy http_proxy` for git on the front end.
- The v5 code on NAS (vulcan-retrieval `2606bb7`, VULCAN3 `da54541`, vulcan-forward
  `c5770a4`) [M]: bootstrap 96220 and `validate_env` PASS (front end and node);
  `tools/gpu_kernel_test.pbs` 96221, `tests/test_solver_fast.py` 33 passed / 13 skipped,
  vulcan-forward `tests/test_ckd.py` 7 passed. The CUDA solve-guard A/B is in VULCAN3
  notes "Efficiency program".
- Pinning a node (`-l select=...:host=<node>`): take one with no `queue` attribute in
  `pbsnodes`; a node assigned to another queue (`queue = app_test`) leaves the job
  "Can Never Run". A node that fails every start (exit -16, "job held, too many failed
  attempts to run" after 21 tries; cgh11n04) holds the job: `qdel` and resubmit pinned.
- cghfe01 can lose its route to the internet (no route to api.anthropic.com or
  downloads.claude.ai) while cghfe02 keeps it; both share the home folder, so tools
  installed in `~/.local` from cghfe02 serve both.
- Env `pyt2_8_gh`, caches under `/nobackup/$USER/.vulcan`; jobs are read-only on the env
  (a per-job `pip install --user -e` raced across concurrent jobs, 64961). The bootstrap
  installs `exogibbs==0.6.0` with `--no-deps` (its `pyfastchem` does not build on aarch64).
- The nsys-jax used by PROFILE_BATCH lives in `$VULCAN_SCRATCH/tools/nsysjax_venv`
  (rebuild recipe in tools/run_nas.pbs next to NSYS_JAX). Profiling lessons: nsys masks
  the exit code (64604), a full trace hits the 100 GiB cap after ~340 s (66476), a
  windowed nsys can orphan the app with its 90% preallocation (78814), ncu's lock file
  cannot live on the /nobackup TMPDIR (80076). PROFILE_BATCH's `.nsys-rep` is written
  to nsys-jax's `${TMPDIR}/tmp*/` when the window stops; the zip in
  `logs/nsys_<jobid>` only at exit (94946). The front end reads it:
  `/opt/nvidia/nsight-systems/2026.3.2/target-linux-sbsa-armv8/nsys export --type
  sqlite` took 43 s for 14.6 M events. In that trace the RT pass shows 0.5 s of kernels
  in 37 s while the host waits on the stream (kernels inside CUDA-graph conditional
  bodies not traced per node, probably [I]), and CUPTI's device-to-device byte counts
  are unphysical (238 MB per 66 us copy): use copy durations, not bytes. `PROFILE_HLO=1`
  fails at start: nsys-jax adds its dump flags to `XLA_FLAGS`, which
  `run_setup.export_runtime` refuses (the spec decides); put `--xla_dump_to=<dir> --xla_dump_hlo_as_text=true
  --xla_dump_hlo_as_proto=false` in a throwaway spec's `runtime.xla_flags`, with a fresh
  `JAX_COMPILATION_CACHE_DIR`. Match kernels to HLO by the sqlite's `demangledName`.
- JAX compile cache: see VULCAN3 notes (private cache for any memory fraction != 0.90).
- daw lane benchmark [M, GPU 0 RTX 6000 Ada; `fast` solver; the v4 config: nz 100, the
  presolve program at rtol 0.2, photolysis every step, grouped RHS; 150 accepted steps
  per lane, no refill, compaction off, compile excluded; memory is the presolve
  program's]:

  | lanes | ms per tick | accepted lane-steps/s | peak GiB |
  |---|---|---|---|
  | 142 | 274.6 | 459 | 11.4 |
  | 284 | 477.3 | 522 | 23.4 |
  | 426 | 787.8 | 464 | 33.4 |

  568 lanes were projected at 44.6 GiB against the 42.8 GiB pool and not run. A fixed
  per-call cost of ~58 ms per lane (seed and loop, every lane masked) is unprofiled.
  CPU, one lane single-threaded: 57.3 ms per accepted step, so 32 such processes would
  give ~559 lane-steps/s [E], about the GPU's best.
- daw batched run: `run_nautilus` multiplies `cold_lanes` by `FFI_BLOCKS_PER_SM` only
  under `ffi`, and daw runs `fast` (no nvcc), so the lanes are `cold_lanes`. Use a
  variant spec (`base: wasp39_test1.yaml`; `runtime: {solver: fast, xla_flags: null}`; `config:
  {cold_lanes: 284}`) with `XLA_PYTHON_CLIENT_PREALLOCATE=true` in the shell (the lane
  count is in the target, so it is its own run). At 284 lanes the v4 ladder's 264 and
  132 rungs stay active; their effect at that width is unmeasured.

## Evidence behind the rules

One entry per CLAUDE.md rule, by topic. "No measurement" marks a rule that rests on a
maintainer decision or on design alone.

Sampler and policy
- Stock nautilus, warm by default, evidence claims cold: "Sampler"; warm vs cold on
  93269's posterior rows ("Runs", postrun_93269) and the bistable_policy start policies
  ("Low-Kzz hole"). The warm default is the maintainer's decision.
- One chemistry program (rtol 0.05, cap 5000): the full-run A/B ("Runs",
  v4_nopresolve_test) and the maintainer's decision ("Design records", the chemistry
  program); the presolve chain's hard-set numbers: "Low-Kzz hole".
- Certifying cap 5000: job 64575 (192 draws) 27.1% uncertified at 5000, 12.5% at 12000
  (p50 2525, p75 5420); 40000 steps rescued 0/4 (failure_diag); stage B at 8000 rescues
  1 of 9 prior-edge hard lanes for ~+11% chemistry; of 384 warm nautilus solves replayed
  on CPU, 1 certified in (3000, 5000], 2 in [5000, 10000), 15 never by 10,000. Caps of
  3e4 / 1e4 gave open-ended init tails (64073, 64163); 29% stayed censored at 20000.
- Override keys enter the target digest (`provenance.TARGET_FREE_KEYS` lists the
  exceptions; `run_nautilus` refuses a resume across a digest or setting change): no
  measurement. The manifest (`manifest_version` 4; the version also moves when the
  evaluators' keep/reject logic changes at a fixed config) also binds the solver choices
  `VULCAN3_SOLVER`, `VULCAN3_CHEM_RHS` and `VULCAN3_BT_BUFFERS`, so a resume cannot
  switch them (design; not every one moves results).
- Quote from the weighted points: `nautilus_posterior.npz`'s `samples` come from
  `sampler.posterior(equal_weight=True)` (`run_nautilus.main`), which draws about
  sum(w)/max(w) samples (256 for 90302) [M].
- Report from the certificate and quote the seed spread: nautilus 1.0.6 gives no logZ
  error; the stock sampler's seed spread on the surrogates is in "Sampler"
  (nautilus_relu_test).

Convergence and the chemistry map
- VULCAN's own criteria: every gate relaxation dropped (conv_gate). `yconv_cri` 1e-3
  costs thousands of steps for no gain. `conv_step` 300 cut cold steps 4484 -> 2885 but
  certified a state 0.072 dex away (7x the tolerance). yconv_min 0.1 / 0.03 / 0.01
  certify 8 / 7 / 4 of 16 draws; worst depth slop at 0.1 13.4 ppm (0.19 sigma). Every
  shipped config exits on the LOOSE branch (W39b longdy 0.099 against yconv_min 0.1):
  quote the realised longdy, not the criterion.
- `dt_max = 1e11 s`: the master default (1e17) lets Ros2 dt balloon to ~1e16 on
  high-Kzz columns; the cap converges them (64523, draw d19: > 11000 -> 986 steps); any
  cap in [1e9, 1e12] changes only tightness (truth solve bit-identical at 4275 steps).
- Final-stage gate and equilibrium seed: the old max(stage 1, stage 2) count rejected 5
  of 9 certified columns; from the seed, a draw censored from the baseline start
  certifies (stage 1 8012, stage 2 3126; upstream 8103).
- One certificate for every likelihood entry point: job 65200 died with 16/864 proposals
  finite in L and non-finite in the gradient under an accept-count-only gate; a constant
  "healthy" ConvDiag once left the cold gradient path ungated for two weeks.
- Chemistry f64: rate constants span ~50 dex.
- T-P reject, never clip: clipping into [300, 3000] K gave silently wrong
  finite-likelihood states.
- Elemental map only: the retired masks map leaked ~0.6% of H per e-fold of Z; the
  elemental map gives d ln(X/H)/d lnZ = +1.0000 for O, C, N, S, 0.0000 for He.
- `prior_c_o` upper edge 0.24: the fixed-O knob's O-only compensation b_z must stay
  positive (bound +0.566 on the 10x-solar column); beyond it draws became negative
  O-carriers with finite likelihoods. `build_retrieval_forward` raises at the bound.
- The positional chemistry block `[lnZ, c_o, lnKzz]`: no measurement (a dropped entry
  would silently mislabel the posterior).

Opacity and the RT
- Correlated-k only: sampled line-by-line at nu_pts 1652-13216 stayed 667-857 ppm rms
  from the reference with a 1.30x feature contrast while adjacent rungs agreed to
  719-798 ppm.
- FD vs AD: central FD disagrees with the jvp at ~1e-2 in directions that reshuffle the
  g-ordering in `ckd.overlap` (lnKzz 1.7e-3, Tirr 2.8e-3 end-to-end; lnR0 / cloud 9e-8);
  `jax.jvp` and `jax.vjp` agree to 1.6e-16, `_interp_logk` FD-checks to 1e-10; an order
  exchange leaves a finite jump at any perturbation size. Compare two AD routes by
  max|dg| / max|g|: dL/dc_o is ~2500x smaller than the dominant components.
- `art_nlayer = 67`: at 60 the Jacobian gate failed (1.345% > 1%); at 67 1.02 ppm /
  0.933%. Model top 1e-9 bar: 1e-8 vs 1e-9 14.65 ppm (FAIL), 1e-9 vs 1e-10 1.14 ppm.

Lanes and batching
- Lane width: a one-block-per-SM kernel runs the primal 1.69x faster at 132 lanes than
  at 144 (one wave against two, 79666); the ffi one-buffer kernel keeps 3 blocks per SM
  (`run_nautilus.FFI_BLOCKS_PER_SM`); the 100-layer probe fits 528 lanes (94050).
- One route per config, regenerate across a lane change: a refilled draw enters at the
  tick its lane was freed at, so its column moves at the convergence scale (CPU A/B: one
  of 4 draws flipped certification; the refilled row moved |dlogL| 1.02e-3). Every batch
  through `_chem_batch_route`, no per-row vmap of solves: the photolysis and geometry
  cadences follow the loop tick the batch shares (design, no measurement).

Evidence semantics and condensation
- `logZ_box` depends on the solver settings: rejected draws are zero-filled, and the
  rejected set moves with `count_max` and the rtol (design). No T-P-window-corrected
  evidence field: "Do not retry".
- Inference with condensation is refused by default because no retrieval with
  condensation has been validated: whether condensing columns certify, and how they
  depend on their start, is untested (`allow_condense_inference`
  is the opt-in). The gates read `use_condense` alone: a `run_inference` flag they
  also read let `run_inference=False` pass them while nautilus still sampled, so the
  flag was removed; a condensation forward model goes through the engine. Gradients are
  a separate matter, relevant only to a gradient sampler: through the pinned state the
  pinned-species tangent is order-unity wrong (0.91 relative, FD ~ 1/dT); the gas +
  condensate reservoir matches to 18-22% on aarch64 and misses the 35% gate on x86
  (rel. 0.407, xfail there).

HPC
- `module load cuda` in a JAX process breaks the compile (78981, 80074) and hides the pip
  CUDA plugin.
- Command buffers on: the 1.9x primal gain (~255 -> 132 ms per step) compared XLA's
  default set with the WHILE set (78991 vs 78995), not off with on; with no flag jaxlib
  0.6.2 keeps the default set, so off needs the empty value. Truly-off A/B not run.
- ugrep honours `.gitignore`: it once hid 16 of 86 engine imports.
- Jobs install nothing: per-job installs raced on the shared userbase (64961).
- The GPU validation set at the maintainer's call, the full tier before production:
  maintainer decisions, no measurement.
- Kept names (`runs/w39b_smc_retrieval`, `smc_rt_chunk`): the NAS tree and archived
  runs' `config.json` use them (`nautilus_report` rebuilds a run's Config from its
  `config.json`); no measurement. `SMC_RETRIEVAL_*` and `SMC_RESUME` went with the
  environment-driven setup ("Design records", run specification).

## Design records

- WASP-39b v5, the maintainer's decisions after the external science review. Each
  model option is a Config switch, so the v4 target is unchanged at the defaults; v5 is
  the headline run and uses T1bar, `s_o` and `lnM` ("Current state").
  - The deep T-P parameter is T at 1 bar, not Tint. Guillot's constant-kappa deep
    gradient is 0.25 against the H2-He adiabat 0.28-0.29, so the deep column is set by
    Tint and kappa alone and the T-P window cut the Tint prior where it controls
    quenching (CH4, NH3, CO). T^4(P) is linear in Tint^4, so Tint is reconstructed in
    closed form and a draw with negative Tint^4 is rejected. The window applies on the
    chemistry grid to 7.6 bar as well as the RT grid (through v4 it ended at the RT
    bottom, 7 bar: 0.7% of prior draws passed it but exceeded 2980 K at the chemistry
    bottom [M, outside review]). Through v4 Tint was free (uniform 50-600 K; Khorshid et
    al. 2024, arXiv 2405.12061, retrieve it log-flat) and T(1 bar) was read off the
    posterior; v4's Tint is a profile parameter, never an interior-flux measurement.
  - The planet mass is a nuisance ln(M/M0) ~ N(0, 0.11) about M0 = 0.2785 MJ (g = 422
    cm/s^2 at 1.279 RJ; W39b.yaml `Mp`), applied in the RT at the fixed reference
    radius, so gravity trades with metallicity through the scale height. The
    chemistry's hydrostatic column keeps M0; the few-% gravity inconsistency in H is a
    stated limitation. Through v4 the mass was fixed and the results conditional on it.
  - Sulfur is a free ln(S/O) on top of lnZ (the engine side is vulcan-forward's).
  - The certifying cap stays at 5000; the Kzz headline prior
    is x0.1 to x30 in `wasp39_test1.yaml` (x10 is the full H w_rms of the Tsai+2023 GCM; v4 ran
    x0.1 to x1000, its override at `8e3a21c`). A longer step budget at low Kzz was
    declined ("Ideas not adopted"). Kzz is reported truncated at x10 with the weight at
    the x0.1 floor, and as an upper limit if that weight grows.
  - Four sensitivity runs the external review asked for, each a `base:
    wasp39_test1.yaml` variant at the base's solver settings, in priority order (an
    outside review's ranking by referee risk): `wasp39_test1_kzz1000` (the Kzz prior to
    x1000, 12 sampled parameters), `wasp39_test1_deck` (a grey deck at a free pressure in place
    of the power-law cloud, 16), `wasp39_test1_kzz_powerlaw` (Kzz as a power law in
    pressure, 18; compare Kzz at stated pressures, not the amplitude) and
    `wasp39_test1_adiabat` (an adiabat below a free boundary, 18). Each is compared
    with v5 on O/H and total metals, C/O, S/O, Kzz and SO2; their warm evidences are
    approximate and compared only at matched programs. They wait on the paper plan
    (the maintainer decided).
  - The chemistry program: every draw is solved once, at rtol 0.05 within 5000
    accepted steps with photolysis every step, and only its VULCAN certificate keeps
    it; the presolve chain is removed (the maintainer decided). On the v4 target the
    plain program ("Runs", v4_nopresolve_test) moved every median by under 0.1 sigma,
    ran 19.8 h against 23.8 h and rejected 2.56% of in-window draws against 1.60%,
    with logZ_box 0.064 lower. It needs no second compile. logZ_box is comparable only
    at matched programs, so v5's is not set against v4's.
- Integrated nuisances (`integrate_nuisance`, v5; the maintainer decided, with an
  outside design review): per instrument group the offset integral is closed form and
  the inflation one a trapezoid over 2049 nodes in log10 b (`pipeline.make_nuisance_marginal`),
  at the sampled model's priors, so the evidence and the physical posterior are the
  sampled model's and nautilus samples 12 dimensions. Go/no-go on the real v5 bins
  (NIRISS 91, NRS1 30, NRS2 29) against brute force [M]: <= 5e-5 nats on realistic
  spectra at 257 nodes; the worst case (b on its floor) 3.6e-3 at 513, falling as h^2;
  an offset past its prior edge exact (4.6e-7 once the brute-force grid is refined);
  the jax version 1.4e-7 at 2049. The run logs each certified row's group statistics
  (R, S) with the log L nautilus sees (`nuisance/`), and the posterior file gets one
  conditional draw per row (`nuisance`, `samples_nuisance`; b on the node grid, then
  the offset's truncated normal), joined by (z, log L) so a replayed row matches; the
  report's spectrum and predictive band use those draws. Chosen over drawing them from
  the report's re-solved rows (32 cold rows of a warm run: noisy and not the run's
  conditional). `tests/test_nuisance_integral.py` pins it.
- Work order (the maintainer decided): checks and changes that make the model right in
  general come first; everything about WASP-39b's results (the v5 run's checks,
  sensitivity runs, how a number is quoted) waits for the paper plan, where those tests
  are designed together. todo.md keeps the two parts. v5 launches on 12-D nautilus
  once the end-to-end check passes (the maintainer decided).
- v5 model options (0.28.0): the chemistry block is `[lnZ, c_o, lnKzz, <chem_extras>]`
  with `s_o` (`infer_s_o`) and `kzz_slope` (`kzz_mode="powerlaw"`) in the engine's
  `CHEM_EXTRAS` order, read through `pipe.n_chem`; the warm references and the anchors
  are three wide (lnZ, c_o, s_o; an old two-wide anchor file reads as s_o = 0) and
  stage 1 zeroes all three composition knobs. The T-P block's deep entry is `Tint` or,
  with `tp_deep_param="T1bar"`, T at `tp_deep_p_bar` (Tint in closed form, NaN rejected
  by `tp_valid`), then `log10P_rcb` with `tp_adiabat`; with both, the Guillot part is
  inverted at min(P_rcb, p_deep) for T_deep (P_rcb / p_deep)^nabla so T at p_deep stays
  the retrieved value, and a T_deep below the deep irradiation floor times
  (p_deep / P_rcb)^nabla has no profile (at gamma 0.03 and P_rcb 0.1 bar, T1bar under
  ~2700 K [E]): a run with `tp_adiabat` reports the (T1bar, P_rcb) pairs with no
  Guillot solution as its accepted joint prior. `tp_valid` evaluates the window on the
  RT grid and the chemistry grid. `lnM` follows `lnR0` with the first Gaussian prior
  type (`make_uspace`: mean + sigma Phi^-1(z)); `cloud_model="deck"` replaces the two
  power-law entries by `log10P_deck` (an opaque grey deck below a free pressure in place
  of the uniformly mixed power-law cloud). `make_loglike(rejects=)` writes every
  rejected draw with its class (`rejected/*.npz`: the unit-cube point, the class, the
  accepted steps and the start mode per chunk), and the certificate's `derived` block
  holds the weighted 16/50/84% quantiles the paper quotes (O/H and total-metals
  metallicity, C/O, S/O, the Kzz factor with the weights below x0.2 / x0.15 and above
  x10 and the x10-truncated posterior, T at the deep level and Tint).
- Storage convention (the maintainer's, with an outside design review): code, tests,
  the inputs a run needs and the validation scripts are tracked; everything generated is
  not. `runs/<case>/` holds inputs only (the run specification files);
  `output/<case>/` (ignored) holds the run directories (each with its `jobs/` logs) and
  `validation/` (the artifacts `_artifact.emit` writes); `scratch/` (ignored, every
  repo) holds one-off scripts, tests and figures. `run_setup.output_base` gives a
  spec's out dir, `output/<case>/<name>_nautilus`. Rejected: renaming `runs/` to
  `cases/` (the kept-names rule: the NAS launch commands and archived config.json
  files name `runs/w39b_smc_retrieval`) and moving the observed spectra out of the
  engine's data root (no separation gained; `forward/config.py` uses them as the tracked
  marker). The old convention put outputs in `runs/<case>/data/` and `logs/`; those
  outputs are deleted (the maintainer decided) except a copy of the wide run's results
  on daw, so the legacy ignore rules stay.
- Two-stage solve. With a moving T-P the hydrostatic density rebuild makes the fixed
  point forget the initial inventory (dL/dlnZ ~ 1e-20, FD agreeing). Stage 1 converges
  at (T(theta), Kzz(theta)) with baseline composition; stage 2 scales metals and C/O on
  that column and re-converges warm (+5% lnZ -> +5.29% CO; ~1.2-1.4x one solve). Any
  code that moves T and uses the composition knobs perturbs a T-converged column. The
  starting column is repaired on He/H2O/CO/N2/H2S so column ratios hit their targets to
  ~1e-15; atom_ini is rebuilt from it.
- Certification: `ConvDiag` rides the runner's primal carry; usable = valid &
  conv_normal; stalled exits are a rejection class. Healthy warm proposals certify on
  the LOOSE branch, so a tight-only predicate (P2) rejects everything, and P0 (longdy <
  yconv_min at exit) is a no-op for the failing class. `native_depth_aux` returns the
  certificate bit; the synthetic injection refuses an uncertified truth spectrum on it
  (before, the truth was checked for finiteness only). `observed_depth_from_y_jit` is
  the only ungated spectrum (plots, diagnostics). No accept-count clause, from baseline
  v4 (the maintainer's choice; manifest_version 4): the runner stops on
  `accept_count > count_max` with convergence first, so a lane can certify at cap or
  cap + 1, and `conv_normal` is False on every cap exit. The former `accept_count < cap`
  clause rejected those certified draws (~3e-6 of draws [I, from one count in 384
  replayed solves]); the tally's exhausted class is `~conv & (acc >= cap)`.
- Cold start: the in-graph Gibbs equilibrium (`vulcan3.ini_abun.eq_seed`, ExoGibbs) at
  the proposal's own T(P) and elemental ratios, the engine's only cold start; zero
  tangent, lnZ / c_o tangents enter through the elemental projection.
  `chem_solve_cold_diag` returns stage 2's ConvDiag.
- Lane queue: every chemistry batch runs on `vulcan_forward.converged_y_queue` at
  `min(cold_lanes, draws)` lanes, refilling a certified lane in the same while loop
  (`retrieval_forward._chem_batch_route`); `validate_config` refuses `cold_lanes` 0. The
  spec sets `cold_lanes` (264 for v2, 176 for v4 and v5; one GH200 kernel wave is 132
  SMs), times FFI_BLOCKS_PER_SM 3 = 792 / 528 lanes. One chemistry call per batch; only
  the RT is chunked.
- Growing anchors (`run_nautilus.Anchors`): each certified column is an anchor keyed by
  its unit-cube chemistry + T-P coordinates; a later solve continues from the nearest
  anchor at the cold cap. Never pruned (~44 KB per column; write order: "Known and
  deliberately untracked"). The maintainer accepted an evaluation-order dependent
  likelihood for the speed.
- Fixed-mass geometry (vulcan-forward 0.28.0, a16188e): `transmission_depth_r` scales
  (rp_cm, g) at p_ref to (rp_cm e^lnR0, g e^(-2 lnR0)), so GM = 422 rp_cm^2 (0.2785 MJ)
  holds at every level (`test_e2e_rt_reference.py`). p_ref = 1 mbar
  (continuum depths give lnR0 ~ +0.028). The chemistry Rp is overridden to 1.21 RJ (the
  fixed-GM 1-bar radius on the 90302 posterior).
- Split G395H offsets: Sarkar+2024 (arXiv:2405.06737 §5.2) find 40-50 ppm between the
  detectors of this visit; a step between the H2O (NRS1) and CO2/SO2 (NRS2) bands lands
  on Z. The offset reference group is the wavelength-first group (NIRISS).
- Run specification (the maintainer asked for one clear file; with an outside design
  review): a run is one YAML file, `runs/<case>/<name>.yaml`, in three blocks; the PBS
  job is not one of them. `runtime`: the import-frozen engine environment (solver,
  chemistry RHS, kernel buffers, XLA memory fraction and flags); every variable is
  stated or required unset, and an environment value that disagrees is refused, so a
  shell export cannot move the target. `sampler`: n_live, n_eff, n_batch, warm_starts.
  `config`: flat `Config` fields, the dataclass as the schema (unknown keys and
  `out_dir` refused; text where a number belongs refused with the `1.0e+11` hint, since
  YAML reads `1e11` as text). The run name is the file stem and the out dir follows from
  it. A variant names one `base` and replaces values one level deep, whole (a deep merge
  cannot remove a key); a base has no base; the merge stays for the daw variant ("HPC
  (NAS GH200) and daw"). The sampler block stays out of `Config` (nautilus's settings;
  n_eff is a stopping rule a resume may change) and the resume identity stays the target
  digest plus the sampler string, never a byte compare of the file. `tools/submit.py`
  validates the spec, builds the qsub line (`gpu_long`, one GH200 node, the allocation
  in `$PBS_GROUP`, the wall 2 h past the governor, so the two cannot disagree), passes
  three comma-free values (spec, out dir, resume flag) to the generic
  `tools/run_nas.pbs`, records each submission under `<out dir>/jobs/` and chains a
  `--resume` after a still-active job. Replaced: case.py presets, JSON override files
  with `_comment` keys, the `SMC_RETRIEVAL_*` / `NAUTILUS_*` / `SMC_RESUME` environment,
  the `XLA_CMDBUF` switch and the per-run `qsub -v` lists (last tree with them:
  `8e3a21c`). Rejected: job settings in the `#PBS` header (walltime and the governor in
  two files), a `job` block in the spec (it named one allocation, so another user edited
  the run file to submit it), full explicit files per variant (the six JSON files had
  drifted), YAML anchors (one file, one run name), renaming the case directory (the
  archived paths).
- End-to-end check of the production stack (`validation/e2e_check.py`, GPU rungs by
  `tools/check_nas.pbs`; the maintainer asked for whole-pipeline checks against another
  route's final results, with an outside review): the same 1584 seeded draws (in-window
  prior draws, the first rows designed corners; or weighted posterior rows) through a
  ladder one change apart: production (ffi, grouped RHS, the spec's XLA flags, 528
  lanes, the production cold evaluator), shuffled (the noise floor), command buffers
  off, no XLA flags, codegen RHS, fast solver, one draw at a time on the GPU, the same on
  the CPU, the reference solver on the CPU. Compared per draw: the certificate, the
  binned depth in data sigma (worst bin, D = the quadrature sum), photosphere
  (1e-5..1 bar) log-VMR of H2O, CO, CO2, CH4, SO2, H2S and log L. Provisional gates:
  worst bin 0.1 sigma, D 0.5, 0.03 dex p95 and 0.1 dex worst, |d log L| 0.5 within 50
  nats of the best draw, no certificate mismatch; an exceedance is a disagreement to
  explain, not a verdict. Against VULCAN 2: the W39b and HD189 configs through the
  CLI under ffi + grouped, compared by `VULCAN-JAX/tools/compare_vul.py`; the CPU
  port's W39b column is within 0.003 dex of VULCAN 2 on those six species (CH4 worst;
  VULCAN-JAX `output/W39b.vul` against the stored VULCAN 2 columns) [M]. The same
  script's `warmcold` is the warm-vs-cold check (todo).
  Results on v5 (NAS GH200, 1584 in-window prior draws, cold) [M]: command buffers off,
  no XLA flags and the codegen RHS are bitwise the production rung (908 certified,
  every metric 0). The noise floor (production vs shuffled, 17-D): worst bin 0.050
  sigma, 4 certificate flips, good-fit |dlogL| <= 0.118. ffi vs fast: p95 9e-6 dex,
  good-fit |dlogL| <= 3e-4, 3 flips; one poor fit (row 1352, 171 nats below the best;
  lnZ 1.25, lnKzz 3.13, T1bar 979 K) differs by 1.24 sigma in one bin and 0.14 dex.
  The CLI under ffi + grouped against the CPU port: W39b 4.6e-8 dex, HD189 1.6e-7 dex,
  both at their gate step counts (1202, 1495).
- Cold certification of in-window v5 prior draws at count_max 5000 [M]: 57% (885-908
  of 1584); every failure is a cap exit, worst where the column is cool and metal-rich
  and weakly mixed (lowest T1bar quintile 21%, highest lnZ quintile 24%, lowest lnKzz
  quintile 36%, against 81-86% at the other ends). 81% of uniform v5 box draws leave
  the T-P window (v5's first nautilus batch: 1277 of 1584), against ~4% for v4.
- Run certificate (`nautilus_report.certificate`, version 1): provenance (the run's
  target digest and manifest, code commits, dirty flags and `src_diff`, versions,
  per-file k-table + CIA sha256, and the same for the code now), the sampler settings,
  `logZ_box`, `n_eff`, `n_like`, `logZ_err` null, the warm/cold start mode, the
  rejection tally as fractions of the in-window draws, the chemistry settings (`rtol`
  from the summary's `chem_rtol`, `count_max`, ...; a presolve-chain run's `chem_retry`
  block is copied as its summary recorded it) and plain-English
  caveats. No pass/fail verdict and no gates: a dirty tree, code moved since the run
  and a manifest that does not hash to the digest are caveats, not refusals; there is
  no prior-rail check (an 84% / 99% quantile against a prior edge was planned and not
  built) and no attrition threshold. `src_diff` hashes untracked files under `src/`.
  The resume refusal lives in `run_nautilus.main` (target digest, start mode and
  sampler settings) and runs before the identity files and the digest are written
  (only `run.log` comes earlier).
- Engine split: the chemistry and RT engine moved to vulcan-forward, bit-identical on
  the planner's W39b spectrum (the gate for engine surgery).
- Condensation with a live T(P): static `make_conden_spec` + per-proposal
  `build_conden_profile`. The certified recipe (whole-column pin, `mtol_conv` 1e-15,
  `conver_ignore` + S/S2/S3/S4, `trun_min = stop_conden_time`) is pinned by
  `tests/test_condensation_live_tp.py`, which starts from the cfg's const_mix column.
- Performance scope (the maintainer's): a full retrieval in about 48 h on one GH200 at
  >= 50% GPU utilisation. The metric is unnamed: the ffi kernels sit at 9.4% achieved
  occupancy and ~2% of FP64 peak while `nvidia-smi` reads ~100%; name the metric before
  claiming progress. Production solver `ffi` (79954: gradient 151.8 ms per step at 132
  lanes; primal 1.24x `fast`). Custom CUDA kernels are in scope and live in VULCAN-JAX.
  Rejected: cyclic reduction (cond up to 6.7e23), a Pallas LU, Tridigpu / TBD-GPU,
  multi-GPU nodes. No minichem and no self-made or pruned networks; any published
  upstream network may be used (every shipped sulfur network has >= 89 species).
- Validation artifacts: none in the tree, and nothing reads them automatically; neither
  ladder has run at the current config. Keep `art_nlayer` only if the depth (5 ppm) and
  Jacobian (1%) gates pass; keep the model top only if the one-decade extension passes.
- Run figures (the maintainer decided): a run's figures are the three
  `nautilus_report` writes (`plots/corner.png`, the weighted corner;
  `plots/tp_posterior.png`, the T-P 5-95% band; `plots/spectrum_fit.png`, the spectrum
  fit with the posterior-predictive band), installed by the `plot` extra and pinned by
  `tests/test_nautilus_report.py`. Nothing else is drawn beside a run by default; the
  physical-units corner, grouped marginals, VMR profiles, rejection map and batch
  diagnostics once drawn beside baseline_v4 were retired with their scripts. A figure
  the maintainer asks for is one `scratch/` script (root CLAUDE.md "Figures").
- WASP-107b case (`runs/w107b/`, inputs in `data/w107b/`): v5's model and solver
  settings on WASP-107b, to test Welbanks+2024 (arXiv:2405.11018; a 1D-RCPE grid with
  constant log10 Kzz 7-10 fit to WFC3 + NIRCam + MIRI: 8.6 +0.4/-0.2, ~12x solar, C/O
  0.33, Tint > 345 K at 3 sigma) against Sing+2024 (arXiv:2405.11027; an ATMO grid
  with log10 Kzz 11-13 fit to five abundances retrieved from G395H: 11.6 +- 0.1, 43x,
  Tint 460 K). The grids do not overlap; the prior spans both. Three runs, one model:
  the joint fit, G395H alone (`w107b_g395h`) and Welbanks's bands alone
  (`w107b_welbanks`); a statement about either paper needs the three to agree.
  - Data. Sing: Zenodo 10891400, `Sing_2024_Fig2_WASP107b_transit_spec_data.csv` (Rp/Rs,
    R ~1000), binned per detector (split at 3.77 um) into equal ln-lambda bins at
    R 100 with inverse-variance depth means (R 93-110). Welbanks: Zenodo 10780449,
    `FinalResults/FinalSpectra` (wavelength, half-width, depth, error): the Eureka!
    NIRCam spectra merged 2 (F322W2, R 82-131) or 3 (F444W, R 67-110) native 0.015 um
    bins at a time (native R up to 336 is refused by the binning operator), the
    Pegasus WFC3 G141 and G102 as published. The files keep the reader's Rp/Rs layout.
    Six groups, G102 the offset reference; offsets (+-800 ppm) and inflations
    integrated out. Level differences between the products [M]: NRS1 - F322W2 +71 ppm,
    NRS2 - F444W +267 ppm, F444W - F322W2 -83 ppm (Welbanks applied -81 ppm). The
    G102 bins skip 1.078-1.088 um (the He 1.083 um line); `obs_wl_lo` 1.02 keeps four.
    MIRI is past the model band (5.26 um).
  - System: Piaulet+2021 (arXiv:2011.13444): Mp 30.5 +- 1.7 M_E (lnM sigma 0.0557),
    R* 0.67 R_Sun, Teff 4425 K, a 0.0566 AU, so Teq 734 K and Tirr = sqrt 2 Teq =
    1038 K. `rp_cm` 6.6748e9 cm (Rp/Rs 0.1432 at 1 mbar), g there 272.9 cm s^-2; the
    chemistry's Rp 6.037e9 cm at 1 bar, hydrostatic on the nominal profile.
  - UV: the MUSCLES v22 HD 85512 `adapt-const-res-sed` (Welbanks's proxy; sha256
    7c7a683a...407c), its BOLOFLUX shape times sigma (4425 K)^4 at the stellar surface;
    the file's integral is 1% below sigma Teff^4 (the SED ends at 5.5 um).
  - Chemistry bottom 10 bar. The T window (<= 2980 K) applies on the chemistry grid,
    and a deep Guillot column warms as ~P^0.25: a 100 bar bottom would cap T(1 bar)
    near 940 K and reject Welbanks's profiles (1100-1300 K at 0.25-0.65 bar); 10 bar
    caps it near 1680 K. A quench below 10 bar (Kzz >~ 1e12 on cool columns) is
    truncated at the bottom, so Kzz above that reads as one value.
  - Priors: 1-300x solar about the 10x baseline, C/O 0.1-0.7 (log-uniform), log10 Kzz
    6-13 constant (Sing's sound-speed bound 13.5), S/O x0.1-10, T1bar 400-2000 K,
    Tirr 600-1500 K, the v5 cloud power law.
  - Not modelled: the resolution-linked bias Sing corrected CO for (-0.2 dex; stellar
    CO dilutes the planet's lines, which needs line-resolved transits); the ~29x NH3
    enhancement and patchy clouds of Welbanks's fit; stellar contamination differing
    between epochs beyond a grey offset per group.

## Physics assumptions and limitations (read before publishing a number)

- Beside the v4 posterior, report: Tint against the T-P window's induced prior
  ("Priors and the T-P profile"; on a Tint ~ kappa^-1/4 ridge report T at 1 bar or at
  the quench level instead); the mass above Thorngren & Fortney's 54.5x
  (arXiv:1811.11859 Table 1); the posterior truncated at Kzz <= x10 and the weight at the
  x0.1 floor (a solver limit); Tint's marginal against 297 K (arXiv:2410.04470) and
  358 K; the 84% and 99% quantiles against each prior edge (the certificate has no
  prior-rail check). v4 is not quotable (its outputs are deleted, never re-solved cold).
- Beside the v5 posterior, report: the accepted joint prior of T1bar (the irradiation
  term and the window reshape the uniform prior); the certificate's `derived` block
  (metallicity as O/H and as total metals, C/O, S/O, the Kzz factor with the weight at
  the floor and above x10 and the x10-truncated posterior, T at 1 bar and Tint); that
  lnM acts in the RT only while the chemistry column keeps M0; and the warm-vs-cold
  check: a weighted subsample of 500 to 1000 posterior draws re-solved cold through the
  production chain at unchanged tolerances. logZ_box is comparable only between runs
  with the same chemistry and rejection rule.
- Guillot T-P, f = 1/4, no convective adjustment: (Tirr, kappa, gamma, the deep
  parameter) shape one 1D limb; the deep parameter is a deep-temperature knob (Tint
  through v4, T at 1 bar from v5). One profile drives both grids.
- Closed column; lnZ / c_o (and s_o from v5) are exact column-ratio directions; fixed-O
  C/O keeps each layer's oxygen while b_z > 0. Everything is rebuilt per proposal except
  the photolysis cross-section T-interpolation (host side, second order).
- T-P window [320, 2980] K inside the ExoMolOP tables' 100-3400 K.
- He/H fixed at 0.0838 (present-day photospheric; the vulcan3 .dat header cites
  Lodders 2019 Table 8), not retrieved and not scaled with Z (d ln(He/H)/d lnZ = 0).
- RT: 12 molecules, H2-H2 / H2-He CIA, power-law clouds uniformly mixed, H2/He Rayleigh,
  inverse-square gravity (+1.5 ppm against -102 ppm for constant g on a test slab).
  13 produced species with installed tables are unscreened and out of the RT (CS N2O NO
  NS NH CN OH CH3 H2CO C2H4 CH C2 H2O2); drop one only with a recorded
  `opacity_leave_one_out.py --states` bound below the gate.
- Clouds parametric; no stellar contamination; diagonal Gaussian likelihood with
  per-group inflation (no covariance in the products).
- Data: Carter & May 2024 fixed-LD NIRISS SOSS + G395H, 150 bins in 1.02-5.24 um after
  `obs_max_bin_R = 200` drops two truncation remnants (0.069% of the weight); sigma
  37-436 ppm (median 70); trapezoidal binning, no LSF (bins near the model R refused).
- 194 of 513 thermal rates run outside their documented T range in >= 1 layer (SO + OH
  is a smooth power law): disclose. Rate uncertainty matches the statistical error on
  metallicity ("Rate sensitivities").
- k-table floor 1e-5 bar: <= 1.53 ppm per node. T-P table constant fill above 5.35e-6
  bar: a +-200 K ramp moves the depth 17.5 / 68.0 ppm (a bracket).
- Known and deliberately untracked:
  - Anchors and tallies are written before nautilus checkpoints the batch: a crash
    between them redraws that batch, which then starts from its own columns, and the
    tally counts it twice (`run_nautilus` module docstring).
  - `opacity_leave_one_out.py` reads an archived run through the current names and
    priors: use it on the run it came from.
  - The ladder scripts build the RT inputs by hand (no `gas_mask`, `_artifact.bin_trapz`).
  - `validate_env` lets exogibbs 0.6.0rc1 pass the 0.6.0 floor; `tools/gpu_kernel_test.pbs`
    installs on the shared userbase with no lock; the likelihood's -sum(log sigma) term is
    pinned by slow tests only.
  - `provenance` reads the named config YAML for `network`; the named configs now set
    only what differs from `default.yaml` (VULCAN3 `2216219`), so only W39b carries the
    key and an HD189 or HD209 run would record no network-file hash. Resolve it through
    `vulcan3.config.load_config` when such a run exists.

## Literature (WASP-39b comparison set)

arXiv ids for the "Runs" comparison: ERS 2208.11692; Rustamkulov 2211.10487; Alderson
2211.10488; Feinstein 2211.10493; Tsai 2211.10490 (Kzz only x0.1 and x10 tested);
Espinoza 2407.10294; Carter & May 2407.13893; Thorngren & Fortney 1811.11859 (interior
ceiling 54.5x); Fu 2507.15854 (limb averaging biases T low, mu high). Unverified: Powell
2024 MIRI 7.1-8.0x (a search snippet). Full kinetics inside the sampler on real JWST
data with AD gradients: no precedent found (searches through 2026-09, not exhaustive).

## Superseded designs

- Adaptive-tempered SMC with MALA or RWM moves; `397d8bb` is the last commit with it.
  Removed: `run_smc`, the SMC loop and gradient evaluators in `pipeline`, the capped
  warm solves in `retrieval_forward`, `plot_smc`, `certificate` (PASS/FAIL gates),
  `validate_warm`, `smoke_retrieval`, `probe_memory`, `calibrate_count_max`,
  `validation/mala_reversibility.py`, the SMC tests (BlackJAX oracle included), the
  `smc_*` / `init_*` / `ppc_*` Config fields and the PBS `SAMPLER=smc`, `PROBE_MEMORY`
  and `CALIBRATE_*` branches. Design: cold two-stage target, oversampled cold init,
  full-covariance Cholesky preconditioner fixed per stage, badgrad as a zero-drift
  move; the warm mutation (`smc_chem_mode="warm"`, cap 1500) was history-dependent.
  Matched BlackJAX over 24 seeds (bias +0.009 +/- 0.015). Dropped for nautilus:
  smc_mala_ab and the stand-in bake-offs ("Sampler"). Lost: the cold accept-count
  distribution above `count_max` and the fixed-step step-cost bench
  (`calibrate_count_max`), and the compile-only XLA memory probe (`probe_memory`).
- Gradient samplers scored on CPU stand-ins, never built into the pipeline:
  Fisher-metric MALA-SMC (lnZ error -0.03..-0.16), best fit + Laplace + importance
  sampling (wrong on the funnel and banana), multi-path Pathfinder (30-39 GH200-h, less
  accurate than nautilus). Anchored warm starts for the SMC moves: designed, not built.
- The nautilus fork `imalsky/nautilus` (2 ReLU + 2 SiLU networks, spread_batch; runs
  90302 and 93269): retired at 83b2980 (spreading biases logZ ~0.02 low and overstates
  n_eff; SiLU = ReLU). Its branches are tags `archive/vulcan-multishell-2026-10-19` and
  `archive/per-network-kwargs-silu-2026-10-19`; the GitHub repo is archived.
- Unused paths removed at `76148c2` (no preset, override or production run set them;
  a CPU smoke A/B of the old and new code, presolve chain and plain path, cold then
  warm chunk, gave bitwise-equal L, anchors and tally [M]; not compared on the GPU):
  the lockstep chemistry batch (`cold_lanes = 0`),
  the host-thread warm-program precompile (`NAUTILUS_PRECOMPILE`; the warm programs
  now compile at their first batch), the profiling hook's NVTX phase ranges
  (`VULCAN_NVTX_LIB` and the PBS's gcc shim; the cudaProfilerApi window stays), the
  `infer_lnZ` / `infer_c_o` / `infer_lnKzz` switches (one legal value; an old
  config.json still loads, `nautilus_report._config_from_json` drops unknown keys), the
  cold evaluator's `diag` return convention, the equal-weight `medians` in
  `nautilus_summary.json`, `observed_depth_model_jit`, `config_schema.hardware_warnings`,
  the `XLA_CMDBUF=2` arm and the second command-buffer pass of
  `tools/gpu_kernel_test.pbs`, and the manifest's `VULCAN3_BT_FACTOR` binding (the
  factor variant is leaving VULCAN3).
- Retry C two-pass chemistry (82a0130, baseline_v2): retry columns not converged at 0.1
  nat; replaced in production by the presolve chain (cdcffd1). The retry-only mode
  (`chem_retry` without `chem_presolve`: its second pass, the retry width buckets,
  the cold retry evaluator) was then removed. baseline_v2's override file is at tag
  `w39b-baseline-v2`.
- The presolve chain (cdcffd1 to `0a2dbf3`; baseline_v3_presolve and baseline_v4): a
  chemistry-only presolve at rtol 0.2 within 3000 steps as the start, then every
  in-window draw solved and certified at rtol 0.05 within 5000 ("Low-Kzz hole").
  Replaced in v5 by one program at rtol 0.05 ("Design records", the chemistry
  program). Removed: the Config fields `chem_presolve`, `chem_retry`,
  `chem_retry_rtol` and `chem_retry_count_max`, `config_schema.retry_profile`, the
  retry program and its `_Statics` check (`statics_diff`) in `retrieval_forward`,
  `pipeline._make_presolve_eval` and the retry evaluator, the two-pass branch of
  `run_nautilus.make_loglike` (the `retry_*` / `presolve_*` tally keys, the batch
  log's `presolve` / `retry` entries, the anchors' `rtol`), the manifest's switched
  keys, the certificate's retry fractions, the forward check's retry pass and
  `tests/test_chem_retry.py`. A chain-off manifest's config block is byte-identical
  before and after [M]. A v4 config.json still loads (unknown keys are dropped), so
  `validation/posterior_forward_check.py` refuses a chain run.
- 62 chemistry layers in production: bistable at low Kzz, up to 0.45 sigma resolution
  error; v4 and v5 use 100 (the schema default is still 62).
- RT anchored at 7 bar with fixed g (90302 and earlier): the model's GM was e^(2 lnR0)
  of the planet's (g ratio 0.83 / 0.87), every scale height 15-21% too large; replaced
  by fixed GM at 1 mbar. 90302's lnR0 is not comparable with later runs.
- Prior edges: Tint fixed 150 K -> free 100-500 K (a16188e) -> 50-600 K (8bc1425) ->
  T(1 bar) in place of Tint (v5); Kzz x0.01..x100 -> x0.1..x100 (52ad7be; 12 of 15
  draws below x0.03 rejected at 5000) -> x0.1..x1000 (18da2a0; the 81542 posterior
  piled against x100) -> x0.1..x30 (v5); Tirr floor 1100 -> 800 (18da2a0) -> 700 K
  (a16188e); one G395H offset -> NRS1/NRS2 (a16188e); one noise inflation -> per
  instrument (4581e78, opt-in).
- FastChem cold seed -> in-graph ExoGibbs (0.18.0, d549f34); the baseline-composition
  cold start and `cold_seed` went in 0.26.0. Abundance `masks` mode -> elemental only
  (0.26.0, 75cceff). Block-MALA schedule (0d56afa, deleted d549f34); `Stage1Cache`
  (removed cd96136); `warm_extrapolate`; single-stage cold solve (removed 7a1c2e1).
- Sampled line-by-line opacity and the Mie deck: retired with vulcan-forward 0.11.0.
- Route B smooth open-system rainout: NO-GO; the code is in
  `~/VULCAN_Project/.backup/git_tags_2026-09-26/vulcan-retrieval-smooth-rainout-b0c-no-go-2026-07-14.bundle`.

## Ideas not adopted

- Abort doomed lanes early (dt < 1e4 s and reject fraction > 0.3 at step 801: 64/75
  caught, 1 false positive). A new rejection rule: the maintainer's call.
- RT overlapped with the retry pass; longest-first ordering (no predictor, "Do not
  retry"); halving the host update.
- Refill across batches (a nautilus subclass plus a re-entrant queue, 300-1000 lines
  [E], up to 1.22x) and a multi-GPU batch split (MPI across nodes, 150-200 lines [E]):
  declined by the maintainer as not simple enough ("Pipeline speed", v5 queue replay).
- AD warm-start polish (-8 to -16% wall [E]).
- Gradient samplers for the 12-D v5 problem (the outside surveys' top picks, none
  tested): flow-assisted MCMC (flowMC) with bridge-sampling evidence, persistent
  sampling with scheduled MALA, gradient-guided nested sampling (its own scaling puts
  the crossover with region nested sampling near 32-D), parallel tempering + stepping
  stones. A learned-harmonic-mean evidence (arXiv:2405.05969) from nautilus's
  posterior as a free second lnZ, with the flow's mass in the rejected regions
  measured and removed. A batched cold gradient evaluator (one jvp of the lane-queue
  chemistry over the chemistry directions + an RT vjp per row; smoke CPU: log L equal
  to the production evaluator to 1e-13, RT-only directions 1e-7 against central FD,
  chemistry 2e-4..4e-2 where above the FD noise) was built and removed with the study;
  a kept row with a non-finite gradient keeps its log L (rejecting it changes the
  support).
- nautilus (nautilus_improve, upside-only): bootstrap logZ error within shells plus
  per-shell diagnostics; parallel network fits (`pool=(None, Pool(4))`, -10 to -18 min
  per run); n_live 700-800 / f_live 0.05-0.1 after a surrogate test; allocation from
  exploration-phase shell statistics if spreading ever returns (unbiased, 25-32% fewer
  calls than stock).
- RT: unrolled searchsorted interp (bitwise, ~4% faster); CUDA fold.
- Non-uniform pressure grid support in VULCAN3 (a 284-layer non-uniform grid matched
  uniform 600 to <= 0.06 sigma at ~55 h).
- A fixed anchor set hashed into the manifest (Alquier et al. arXiv:1403.5496): warm
  starts that stay a deterministic function of theta, ~2.7x [I].
- A per-draw chemistry geometry shared with the RT (the 1.21 RJ override leaves a few %
  in H).
- Krylov on the stage system: untested; needs matched GPU time against the pivoted
  Thomas solve on large-dt states (cond ~6.7e23) at equal accuracy.
- A Kzz-dependent or longer certifying cap in place of the x0.1 floor (proposed by the
  external science review): the cap measurements ("Evidence behind the rules",
  certifying cap; "Low-Kzz hole", lowkzz_fix and the presolve chain's hard set) show a
  longer budget cutting the uncertified fraction but rescuing no hard lane, with the
  low-Kzz failures needing 19-23k steps for one ultratrace cell: it buys numerics, not
  physics. The floor stays a stated prior cut, reported with the weight at it.

## Do not retry

One line each: what was tried, the measured reason, what stands instead. Not repeated
here because "Evidence behind the rules" holds them: raising the cap, relaxing
VULCAN's gate, `yconv_cri` 1e-3, `conv_step` 300, `dt_max` 1e17, the accept-count-only
gate and P0/P2 predicates, the constant ConvDiag, T-P clipping, the c_o edge 0.6, fp32,
`XLA_CMDBUF=0` without a flag.

Convergence and solver
- Raise if > 10% of cold draws fail: wrong for a real ~27% minority; reject with log L = -inf.
- A separate primal diag solve for the accept count: doubled chemistry wall; it rides the jvp carry.
- Single-stage cold solve with the eq seed: 2/4 production draws certify against 3/4 for two stages.
- Gustafsson PI step controller: 11-17% slower net.
- Frozen setup-time vm (upwind moldiff): up to 1.7 dex bias on depleted traces; in-loop refresh.
- Hybrid vm_mol defaults consumed unexamined: unbound warm cap and upwind phase-0 restarts; per-solve carry re-seeding, warm pinned to central (f05e537).
- Naive two-stream particular-solution pole fix: breaks master parity (W39b's 83 deg is safe).
- Fixing inherited master defects unilaterally (unweighted atom diagnostic, condensate mass in mu, stale table entries): rejected for parity; documented.
- Init-scaling lnZ / C-O under a moving T-P, or a snap-to-baseline continuation: inventory erasure (dL/dlnZ ~ 1e-20); the two-stage solve.
- `warm_extrapolate` seed max(Y + DY dC, 0) and its per-cell repair: manufactured the badgrad class (3/11 vs 0/11 plain; repair fails 2/3).
- A chemistry gradient through vulcan-forward's stacked `converged_y_jvp` or a per-particle `vmap` of jvps: a tangent-certified stop is a different map and cannot carry the single-chain `warm_y` tangent, and under `vmap(jvp)` the primal carry stays unbatched (jaxpr); the SMC path used one `jax.jvp` per direction through the batched runner (code at `397d8bb`).
- A carried cache with no consumer (`Stage1Cache`, ~32 MB per checkpoint): re-add a cache only with its consumer.
- Picking the chemistry program on a hard set alone: the 264-draw low-Kzz hard set made the plain rtol-0.05 program look 2.8x slower per batch; the full-run A/B ran it 4 h faster ("Runs", v4_nopresolve_test). Decide a program on a full run or a representative set.

Gradients, adjoints and precision
- A direct steady-state (IFT) derivative or a tangent tail as the gradient: 50-500x off the re-converged FD even at yconv 1e-3; at production tolerance the slowest mode (tau ~3e10 s) is frozen at ~2e7 s, so FD and IFT differentiate different states.
- Adjoint-solver variants (residual-IFT custom_vjp, the Krylov families, `body_dt` outside [3e6, 1e8], `solver_map` "bare", `renorm_td`, frozen photolysis): recorded in VULCAN3's notes, which owns the adjoint; deflated LGMRES is the only bounded solver, host-side, ~27 min per gradient.
- JAX compilation cache for the ~20-min adjoint step-vjp compile: no help, broke the warm in-process path.
- All-particles vmap gradient in one program: 1.52 TiB request, then 120.9 GiB at 4-particle chunks and launch-bound (63886-63997); the staged evaluator (at `397d8bb`).
- Two-call chemistry gradient (dir-0 + n-1 lanes): ~2x wall when latency-bound; one vmapped jvp.
- ~1.3 GB/lane chemistry-tangent estimate: wrong by ~60x (~20 MB per lane-pair, 64144).
- Tirr through Heng+14 `expn` in forward mode: pathologically slow; differentiate the Tco leaf.

Opacity and RT
- Sampled line-by-line opacity at any affordable nu_pts (nu_pts 16500: 343 GiB RT vjp, 64601) and the Mie deck: non-convergent, 857 ppm rms / 1.30x contrast; correlated-k only.
- A constant-VMR clamp above the chemistry top (73.47 ppm) or truncating the RT grid there (80.82 ppm): chemistry extended to the RT top.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` 0.98: broke executable-constant allocation; 0.90 (pool ~81 GiB). A compiled program can stack stage peaks beyond the component sum (195 vs ~123 GiB at a failed setting); RT vjp memory was 4.32 GiB per lane after `ckd.fold`.

Evidence
- An f_tp-only evidence field (`logZ_box_physical = logZ + ln f_tp`): retracted the day it shipped (toy counterexample, `tests/test_evidence_semantics.py` at `397d8bb`); `logZ_box` stays the zero-filled box evidence.

Condensation
- Naive condensation-on solves (1 um: dt ~0.4 s, longdy 10.4; 50 um: longdy 1.55), a cold-trap-level pin on isothermal columns (S8 re-supersaturated 2560x), `conver_ignore` or `mtol_conv` alone: only the full recipe converges.
- Any total derivative through the pinned condensation state (Fisher, MALA, input sensitivity, Hessian), middle paths (criterion-gated pin, frozen-branch differentiation, smooth surrogate), a `custom_jvp` wrap of `_runner`: refused at the AD entry points.
- Route B smooth open-system rainout + deep H2S reservoir: no cold steady state in 15000 steps, sulfur-flux non-closure 26.4%; its trilinear H2S boundary C0 only (21% derivative error); the anchor-scale BC zeroes the T and c_o derivatives.
- Photo-off Fisher forecasts on the WIDE profile: no steady state on defaults, tangent 25% to 2x off.
- Isothermal condensation escape hatch: unsafe for W39b; the on-graph per-proposal rebuild.
- sign(jvp) == sign(FD) per pinned species: measures the FD step (FD ~ 1/dT); the conserved-reservoir check.
- jvp == FD through the pinned state on x86 CI: platform-dependent since VULCAN-JAX 5427cd1; strict on aarch64, xfail on x86 (open in VULCAN-JAX).
- The engine's cold equilibrium seed as the condensation test's start: S8 ~1e-66 VMR, nothing condenses; start from the const_mix column.

Sampling (nautilus and stand-ins)
- NUTS: the lockstep batch pays the maximum tree depth, ~50x; one mode on the bimodal stand-in, no evidence.
- Keeping nautilus's exploration phase: logZ -0.206 on the surrogates (81542: n_eff 2,843 against 1,557, log Z 1151.60 against 1151.82); discard it.
- nautilus settings that lost on the real-posterior stand-in: 1500 live points, tanh, 8 networks, (64, 32, 16) networks, enlarge_per_dim 1.05, n_update = 2 n_live, and 500-600 live points without enlarge 1.3 (0.5-2.7 million calls).
- A nautilus warm cap below 5000: 1 certified column in (3000, 5000], 2 in [5000, 10000), 15 never by 10,000; the refilling queue removes the wait instead.
- Longest-first batch order by a pre-solve predictor: the anchor distance (rank correlation with the steps 0.13-0.23) and the mean steps of the 10 nearest earlier draws (0.57-0.66) both replay at 0.99x on v5 (known lengths 1.16x): a batch ends with its worst-placed long draw.
- Seeding a new run's anchors from 90302's 213k columns: no measurable gain (0.79 s per call in exploration, 0.77 in sampling) and a different box and geometry.
- A learned initializer (an emulator of y*(theta) as the warm start): the draw's own converged column still costs 191-1321 steps, so no start beats cold by more than 4.4x; the nearest anchor gives 2.13x, PCA+GP 2.05x; no start rescues a censored draw (record: `~/VULCAN_Project/.backup/speed_studies_2026-09/emulator_ideas/results.txt`).
- A spectrum emulator with nested sampling and an importance correction: not pursued, the maintainer's decision (the forward model is too complicated to emulate).
- Importance sampling from a log L emulator (off by 0.79 nats out of sample, efficiency 0.000) or from Gaussian, Student-t or mixture fits to the 81542 posterior (~0): the posterior is far from Gaussian and piles against prior walls.
- MCMC on a fitted stand-in without bounding its inputs: walkers ran to |logit z| ~ 37 where the MLP extrapolates to log L 1304; truncate any sampler on a stand-in to its training range.
- Tint fixed at 358 K in the Guillot profile: the cool lobe's 7 bar level goes to 3847-4786 K (99% of the weight past 2980 K); Tint is retrieved.

SMC path (removed; code at `397d8bb`; the lessons apply to any rebuilt gradient sampler)
- MALA preconditioner shape-only (accept 0.019) or diagonal-only (blind to the Z / C-O / cloud / lnR0 / offset degeneracies), or re-estimated before every sweep (lnZ +1.1 at N = 132): the full-covariance Cholesky, fixed per stage.
- Split-chain RNG across stages: replayed the killed job's randomness on resume; `fold_in(seed, absolute stage)`.
- Init phase 2 capped at `warm_count_max` (5/96 healthy survivors gated, 64854) or with no spares (3/96 died, 64897); `calibrate()` at hard-coded beta / step / scale (8/144 bad gradients, 64961).
- Warm mutation with no cap (> 3 h in stage 1, 64745) or a cap of 800 (the conv_step window needs ~780); 1500.
- MH-rejecting the badgrad class or a zero-tolerance raise (~1%/sweep, 65789): theta-dependent, biases metallicity; zero-drift MALA with the same drift in both proposal densities, backstop 0.25.
- More particles than lanes at the same move budget (lnZ spread 0.38 -> 0.67); Fisher-metric MALA with an unfloored prior curvature (lnZ errors 1-3); BlackJAX as the core (not in the NAS env).

Operations
- `gpu_devel` for short jobs: longer waits than `gpu_long` with a short walltime.
- Comma lists in `qsub -v` (`XLA_FLAGS`): the commas split the variable list; the flags live in the spec's `runtime.xla_flags`, read inside the job.
- nsys `--cuda-graph-trace=node` under command buffers: `CUDA_ERROR_ILLEGAL_ADDRESS` mid-capture (79666); profile with command buffers off or graph-level tracing.
- Chaining jobs on `afterok` of a kernel test: one missing module deleted five benches; chain on `afterany` and let `validate_env` refuse.
- Merging a file without `ruff check`: an undefined `log` killed five GH200 jobs after their compile; ruff every merged file and run a new job path end-to-end on the CPU first.
- A 24 h multi-arm calibration chain as a profiling vehicle: a MALA calibrate arm costs ~11 h, so the profiling arms never ran; profile in short bounded jobs.
- Expecting `LuPivotsToPermutation` count 0 as the perm-cache check: `lax.linalg.lu` reports its permutation through that kernel; 1:1 with getrf is correct.
- Shrinking the slow tests' memory (2 probe particles, `jax.clear_caches()` between compiles): wall -13% or peak 28.6 -> 23.9 GB in one file only; the peaks are single XLA CPU compiles.
