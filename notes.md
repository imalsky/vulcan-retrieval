# vulcan-retrieval notes

What is true in this repo and why, written for Claude: the WASP-39b campaign findings,
the measured evidence behind the CLAUDE.md rules, design records, superseded designs and
a do-not-retry register. Open items are in `todo.md`. The development studies' outputs
lived in `analysis/` on NAS (deleted 2026-10-19); a reference to `analysis/<name>`, or to
`<name>` in notes.md, in code comments or override files means the `<name>` entry here,
or in VULCAN3's / vulcan-forward's `notes.md` (solver and RT findings). [M] measured,
[E] estimated or modelled, [I] inferred. sigma = posterior standard deviation,
L = log-likelihood.

## Current state

- Version 0.27.0, main `397d8bb`. pyproject floors: vulcan-forward>=0.28.0,
  vulcan3>=0.18.0. `requirements-hpc.txt` pins vulcan3 `eeb4873`, vulcan-forward
  `39c2c45`, jax 0.6.2, exojax 2.6.0 and the stock nautilus-sampler 1.0.6; `validate_env`
  refuses the former fork (same version string). After any version bump reinstall all
  three trees editable, or `validate_env` fails on stale metadata; on NAS re-run
  `tools/bootstrap_nas_env.pbs`. The vulcan3 rename (882b9e5) is on main and pinned
  (41aacb4, then 5eb5df9); baseline_v2 (29aa87e) predates it, while the speed2 and
  presolve code (764ccb8, cdcffd1) that the later NAS studies ran contains it [I].
- Production: `run_nautilus`, the only sampler (the SMC/MALA/RWM path is removed after
  `397d8bb`, "Superseded designs"), warm starts from a growing anchor set, solver `ffi`
  pinned by `run_nas_w39b.pbs`, and from the override file the chemistry presolve
  chain, per-instrument noise inflation and log10gamma down to -3: 15 parameters (the
  gpu preset alone is 13-D). Last finished run: baseline_v2 (job 93269). Next:
  baseline v4 at uniform 100 layers (todo).
- Run report: `run_nautilus` writes `nautilus_certificate.json` when a run finishes
  (`nautilus_report.write_certificate`); `python -m retrieval_framework.nautilus_report
  <out_dir> [--spectra N]` rewrites it and draws `plots/corner.png`,
  `plots/tp_posterior.png` and, for N > 0 (default 32), `plots/spectrum_fit.png` from N
  weighted draws re-solved cold. Not yet run on a real run's output [I]. Design:
  "Design records", run certificate.
- Tests: the normal tier (plain `pytest tests`; CI runs it serially on Python
  3.10-3.12) deselects `slow` through pyproject's addopts: 73 passed / 1 skipped in
  30.6 s on daw with `-n 8`, CPU, at the SMC removal [M]. Full tier
  (`-m "slow or not slow"`; slow: `test_cold_reject`, `test_condensation_live_tp`,
  `test_set_observations`' production preflight, two tests in `test_chem_retry`): not
  re-timed since the SMC removal (the 17.9 min before `a16188e` was set by the deleted
  `test_stage_split`).

## Runs

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
- Code: VULCAN-JAX d7021cb, vulcan-forward d26e9d3, vulcan-retrieval 29aa87e. Retry C
  two-pass chemistry (rtol 0.1 to 3000 steps, then rtol 0.2 to 5000), per-instrument
  noise, log10gamma down to -3, 792 lanes x n_batch 2376, compaction [264, 132], the
  nautilus fork (SiLU networks + spreading, see Sampler).
- 17.4 h, 63 batches, 147,680 calls, n_eff 5111, logZ_box 1159.120 [M].
- Posterior as run, median [16%, 84%]: lnZ 0.293 [0.059, 0.607] (1.13 [1.03, 1.26] x
  solar); c_o -0.370 [-0.761, 0.004] (C/O 0.38 [0.26, 0.55]); lnKzz -0.069
  [-1.566, 2.200]; Tirr 970 [917, 1045] K; log10kappa -0.042 [-0.362, 0.205];
  log10gamma -2.173 [-2.612, -1.728]; Tint 181 [125, 251] K; lnR0 0.0302
  [0.0286, 0.0315]; log10kappa_cloud -3.042 [-3.405, -2.782]; cloud_alpha 1.569
  [1.164, 2.163]; offsets NRS1 111.9 [88.5, 134.1], NRS2 70.8 [24.3, 116.4] ppm; noise
  inflation NIRISS 1.242, NRS1 0.920, NRS2 1.273.
- logZ error: statistical ~+/-0.015 [I]; known offsets spreading ~-0.02, tolerance
  correction -0.028, cold start -0.040.
- Problems found (postrun_93269) [M]:
  1. rtol-0.2 retry columns are not converged at 0.1 nat: vs rtol 0.05 the median |dL|
     is 0.28-0.35 (pass-1 rows: 0.07 at low Kzz, 0.012 at high Kzz).
  2. Warm-start path dependence: 97 of 1320 posterior rows certify warm but fail cold
     (lnKzz -1.9..-1.1); net dlogZ -0.040.
- Correction: all 1623 retry rows and 6265 pass-1 rows at lnKzz < -0.7 re-solved at
  rtol 0.05 / cap 15000, reweighted by exp(L_0.05 - L_run), uncertified rows w = 0.
  Every median moves <= 0.035 sigma (lnKzz -0.069 -> -0.002, c_o -> -0.357, Tirr ->
  972); 413/7888 rows rejected; mass at lnKzz < -0.7 0.353 -> 0.335; ESS 5324; logZ
  1159.092. Saved next to the run: `data/w39b_baseline_v2_nautilus/corrected_posterior_rtol005.npz`
  (theta, w_as_run, w_corrected, L_rtol005, cert_rtol005) and `.json`.
- vs fixedmass: log10gamma -2.17 vs -1.73 (64% of the mass below -2 once the prior
  reaches -3); lnZ +0.73 old sigma; logZ -0.48 (Occam penalty of the wider prior ~-0.36);
  the fixedmass lnKzz dip at -1.3..-1.1 is gone.

## Low-Kzz hole

- conv_gate: the notch came from capped solves failing only in layers 45-50 (4e-7..6e-8
  bar), far above the photosphere (> 5e-5 bar). Every gate relaxation (pressure mask,
  cap-exit acceptance, error-norm mask) DROPPED: VULCAN's own criteria stay.
- retry_policy: two-pass Retry C (vulcan-retrieval 82a0130) ~1.33x production per batch
  vs in-queue 1.45x (measured 1.39x); cap1 stays 3000. Used by baseline_v2.
- lowkzz_fix (2026-10-17..18, 8.3 GPU-h). Root causes [M]: (A) photolysis every 5
  ticks triggers stiff transients (100% of reject cascades start on a photolysis tick,
  dt collapses a median 6.9 dex); (B) at lnKzz < -2 and rtol <= 0.05 one near-atol cell
  (atomic S at 1e-9 bar) sets the step while a C2H4/C2H2 drift at ~5e-3 bar needs
  19-23k steps; (C) at 62 layers two states really exist (see bistable_policy).
- PRESOLVE CHAIN (vulcan-retrieval cdcffd1, opt-in `Config.chem_presolve`;
  `pipeline._make_presolve_eval` and the presolve branch of `run_nautilus.make_loglike`;
  the 62-layer study override `w39b_baseline_v3_presolve.json` is in git at `397d8bb`,
  `w39b_baseline_v4_nz100.json` carries the chain now): stage A chemistry-only presolve
  at rtol 0.2, photolysis every step, cap 3000, never kept; stage B re-solves every in-window
  draw at rtol 0.05, photolysis every step, cap 5000, warm from A's exit column where A
  certified, otherwise from the row's original start. Only stage B's VULCAN certificate
  keeps a draw. On a cold chunk a draw whose presolve fails still starts stage B from
  the presolve exit column (the baseline column if that is non-finite).
  - Hard set: 263/264 certified vs 99 (production pass 1), 183 (retry), 203 (direct
    rtol 0.05, cap 15000); error vs reference median 0.027, p90 0.25 (retry 0.26 / 1.24).
  - Representative set: 264/264; median 0.008, p90 0.072 (production 0.022 / 0.126).
  - Production path (job 93676, 792 lanes, 2 real batches): in-window rejection 2.5% vs
    6.1% (Retry C); batch 816 vs 1082 s compile-free.
  - Dropped: direct rtol 0.05 (2.8x), master's adaptive rtol (4.1x), master's photolysis
    cadence, Ros2 polish start, stage A at rtol 0.1 (1.7-2.2x). Flagged: nz 150 certifies
    65/66 vs 48/66 at 2.36x per tick; the 2025 network changes no certification.
- bistable_policy (3.7 GPU-h): the two low-Kzz states differ by where the photochemical
  front (1e-7..1e-6 bar) sits, by whole grid layers; they merge at 100 (35/40) and 150
  (40/40) layers: a 62-layer artifact. 11% of posterior mass has >= 2 states (none above
  lnKzz -0.7). Largest |dmu|/sigma median 0.35, p90 1.08, in the SO2 band at 4.0-4.1 um.
  Start policies: warm (status quo, 581 s/batch), cold2 chain 1.77x, eq chain 1.47x, eqd
  4.10x; every parameter shift <= 0.06 sigma, |dlnZ| <= 0.01. Kept: warm start + presolve.

## Grid resolution (resolution_study, 10.4 GPU-h)

- Only SO2/SO change with resolution (VULCAN3 notes); in the spectrum the 4.05 um band
  moves a median 1.5 sigma per draw (62 vs 300 layers); L(300) - L(62) median -2.8.
  The 67-layer RT grid and the 1e-9 bar top are converged.
- Worst science-parameter error vs the 284-layer reference, in sigma (SE ~0.12-0.20 at
  nz <= 100): 62: 0.45 (lnZ, c_o); 75: 0.45; 100: 0.43 (c_o median; 0.26 by mean);
  125: 0.28; 150: 0.24; 200: 0.13; 300: 0.07.
- Run wall [E]: 62 layers 10.4 h, 100 16.0 h, 125 21.9 h, 150 26.1 h, 200 37.5 h,
  300 56.3 h (fork sampler with spreading; stock needs more calls, see Sampler).
- 93269 reweighted to 284 layers (ESS 187): lnR0 -0.58 sigma, Tirr +0.29, c_o +0.19,
  log10kappa +0.18.
- Moving the bottom to 47 bar changes ~10% of Tint >= 180 K draws, 8% uncertified: not
  worth it as is.
- Next run: uniform 100 layers (the maintainer's choice), overrides/w39b_baseline_v4_nz100.json.
  Lane probe (job 94050): 528 lanes peak 64.4 GiB (cold presolve) of 85.6, 4.78 draws/s;
  396 lanes 50.9 GiB, 4.59 draws/s; 660 compiled at ~79 GiB, never timed.
- 125-layer rehearsal (v4_rehearsal, 396 lanes): cold batch 1667 s, then 709-951 s;
  in-window rejection 7.0%; nsys hook overhead 1.9% in the captured batch, 0.6% elsewhere.

## Sampler

- nautilus_relu_test (surrogates of the real posteriors, n_eff 5000, 8-24 seeds):
  - fork (2 ReLU + 2 SiLU, spread_batch): 124.9k calls, logZ -0.023 +/- 0.016 vs truth.
  - stock 1.0.6, no spreading: 202.0k calls (+62%), logZ -0.001 +/- 0.011.
  - keeping the exploration phase: logZ -0.206: biased, do not use.
  - SiLU vs ReLU: same calls (-0.6 +/- 1.2%), logZ, marginals and mode weights.
  - Spreading biases logZ ~0.02 low (SE 0.003): optimistic adaptive-allocation bias (Shin,
    Ramdas & Rinaldo 2019); it also overstates n_eff (reported 5274, true ~1509 in one
    replay). So 93269's logZ may be ~0.02 low.
- Switched to the stock nautilus-sampler 1.0.6 package, 4 default ReLU networks, no
  spreading (2026-10-19, 83b2980). Kept settings: n_live 1000, n_update n_live/2, f_live
  0.03, enlarge_per_dim 1.3, n_batch 3 x lanes, discard_exploration True.
- Why nautilus (stand-in bake-offs on daw CPU, 11-D, exact truths, 5-8 seeds): nautilus
  |lnZ error| <= 0.03 and posterior bias <= 0.01 sd on every stand-in; the production
  MALA-SMC failed the funnel (|lnZ error| 2.02) and was off +0.31..+0.85 on the spectrum
  stand-ins; pocoMC missed the funnel's width; NUTS found one mode of the bimodal and
  gives no evidence. The settings above came from a real-posterior stand-in (an MLP of
  log L on 81542's 83,160 points): enlarge 1.3 cut the mean error from 0.10-0.14 sd to
  0.02-0.08 sd; 1000 live points were the most accurate row.
- nautilus_improve, upside-only ideas, none adopted: see "Ideas not adopted".
- smc_mala_ab: SMC + MALA DROPPED. MALA costs 2.9x an RWM sweep for 3.1x the ESJD
  (break-even per GPU-second); gradient memory caps it at ~300 lanes; projections SMC +
  MALA ~29 h, RWM > 100 h vs nautilus ~16 h. Chemistry convergence, not the kernel,
  drives the cost (36-45% of warm mutation solves hit the cap).

## Priors and the T-P profile

- evidence_noise_priors (job 92980): evidence stable (cold target 1159.53 +/- 0.02 vs warm
  1159.60). NIRISS drove the noise misfit -> per-instrument noise (4581e78). log10gamma
  lower bound -3 adopted; lnKzz kept at [ln 0.1, ln 1000] (no validation solve below).
- Tint prior (tint_prior): now uniform 50-600 K (8bc1425; was 100-500).
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
    kappa^-1/4), and the pile-up is a window / prior-volume effect. Expect it to move
    to 50 K.
  - Fix options: (A) apply the window only above ~1 bar (lets 4000 K deep layers into
    the chemistry; not alone); (B) retrieve T at 1 bar instead of Tint (~1/2-1 day);
    (C) a physical profile in our code: pressure-dependent opacity plus an adiabat below
    the convective boundary, e.g. Parmentier+2015 (several days with validation; with
    constant kappa the deep gradient is 0.25 < adiabatic ~0.29, so a convective switch
    alone never triggers).
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

- Time accounting of 93269 (56,043 s through batch 55): pass-1 chemistry 38.7%, pass-1
  RT 24.6% [I], retry chemistry 24.1%, retry compiles 3.5%, host/nautilus 5.6%.
  Chemistry lane time: useful 35%, capped or uncertified 38%, idle lanes 27% [I].
- speed2 (all bitwise, on main): retry width buckets compiled ahead, masked padding
  rows, RT only on kept rows (764ccb8; `run_nautilus.RETRY_BUCKET` 132,
  NAUTILUS_PRECOMPILE), plus VULCAN3 a5b1215 / eeb4873 and vulcan-forward 5e8d43a /
  39c2c45. Pass-1 batch 344.0 -> 287.3 s; full batch ~892 -> 628 s excluding compiles;
  RT pass 256 -> 60 s.
- Whole run [E]: ~11 h at 62 layers (~3.4x the fixedmass run's 37.2 h), with the fork
  sampler's call count.
- Warm starts on run 90302 (growing anchors): 507 steps per certified warm column
  against 1,119 cold (the first batch's prior draws), 2.21x [M]; the CPU screen predicted
  2.13x. The batch time follows the capped columns: a batch with any column at the cap
  runs the full cap, so a tick costs the same whether its lanes are busy or done.
- Not done: RT overlapped with the retry pass, aborting doomed lanes (a new rejection
  rule: the maintainer's call), longest-first ordering, halving the host update.

## Provenance

- Commits before 2026-10-13 were made on other machines; their decision records are not
  all traceable (256 vulcan-retrieval commits, ~70 change science).
- Live on main, provenance not recorded; each needs the maintainer's confirmation:
  3eafb1b discard_exploration and the ffi solver default (`VULCAN3_SOLVER` in
  run_nas_w39b.pbs); 3e40f03 + 4c23465 warm-start anchors and warm nautilus by default
  (results depend on evaluation order); 8e633bf + 500b340 lane queue default; a16188e
  p_ref 1 mbar, fixed-mass lnR0, chemistry Rp 1.21 RJ, free Tint, split NRS offsets,
  Tirr 700-2200; 18da2a0 Kzz ceiling x1000 chosen after seeing the data; 52ad7be/611f62b
  Kzz floor x0.1; cc6a1e9 "Snapshot" (noise inflation, correlated-k default); 24bd41b
  12 molecules, 2 bins dropped; 5792482 and d549f34 cold gate, in-graph equilibrium
  seed, XLA command buffers; f05e537 + 88f995e certification gate on stall/budget exits;
  93ed4fd elemental abundance map (redefines lnZ and C/O); c7899ee hybrid vm_mol
  2026-07-14..07-26 (reverted: check no reported result comes from that window).

## HPC (NAS GH200)

- `PROJECT_ROOT` (the PBS header uses `/nobackup/$USER/VULCAN_W39b_HPC`) holds all three
  trees: `vulcan-retrieval`, `VULCAN-JAX` (the clone name is load-bearing for the
  bootstrap and `validate_env`) and `vulcan-forward`. `git pull --ff-only` is live
  (editable installs); a version, dependency or entry-point change needs the bootstrap.
- Queues: the GH200 queues (`gpu_long` 120 h, `gpu_normal` 24 h, `gpu_devel` 2 h) are
  served by `pbs05a` and take jobs only from `cghfe01`/`cghfe02`; `cfe` answers `Unknown
  queue`; `qsub -q gpu_long@pbs05a` is rejected. Login: sfe -> `ssh -o
  PreferredAuthentications=keyboard-interactive cfe02` -> `ssh cghfe01`.
- Data is not in git: seed `data/opacity_cache/` (CIA) and `data/exomolop/`. Fetch the
  k-tables on the front end with `python3 -m vulcan_forward.fetch_exomolop --molecules
  <case.py molecules>` (`module load python3`, `VULCAN_FORWARD_DATA` and `PYTHONPATH` set
  to the repo data and vulcan-forward/src); scp from home ran at 140 KB/s. Other files:
  `scp -r` through the sfe proxy. `unset https_proxy http_proxy` for git on the front end.
- Env `pyt2_8_gh`, caches under `/nobackup/$USER/.vulcan`; jobs are read-only on the env
  (a per-job `pip install --user -e` raced across concurrent jobs, 64961). The bootstrap
  installs `exogibbs==0.6.0` with `--no-deps` (its `pyfastchem` does not build on aarch64).
- The nsys-jax used by PROFILE_BATCH lives in `$VULCAN_SCRATCH/tools/nsysjax_venv`
  (rebuild recipe in run_nas_w39b.pbs next to NSYS_JAX). Profiling lessons: nsys masks
  the exit code (64604), a full trace hits the 100 GiB cap after ~340 s (66476), a
  windowed nsys can orphan the app with its 90% preallocation (78814), ncu's lock file
  cannot live on the /nobackup TMPDIR (80076).
- JAX compile cache: see VULCAN3 notes (private cache for any memory fraction != 0.90).

## Evidence behind the rules

One entry per CLAUDE.md rule, by topic. "No measurement" marks a rule that rests on a
maintainer decision or on design alone.

Sampler and policy
- Stock nautilus, warm by default, evidence claims cold: "Sampler"; warm vs cold on
  93269's posterior rows dlogZ -0.040 [M]; bistable_policy start policies move every
  parameter <= 0.06 sigma [M]. The warm default is the maintainer's decision.
- Presolve chain: "Low-Kzz hole".
- Certifying cap 5000: job 64575 (192 draws) 27.1% uncertified at 5000, 12.5% at 12000
  (p50 2525, p75 5420); 40000 steps rescued 0/4 (failure_diag); stage B at 8000 rescues
  1 of 9 prior-edge hard lanes for ~+11% chemistry; of 384 warm nautilus solves replayed
  on CPU, 1 certified in (3000, 5000], 2 in [5000, 10000), 15 never by 10,000. Caps of
  3e4 / 1e4 gave open-ended init tails (64073, 64163); 29% stayed censored at 20000.
- Override keys enter the target digest (`provenance.TARGET_FREE_KEYS` lists the
  exceptions; `run_nautilus` refuses a resume across a digest or setting change): no
  measurement.
- Quote from the weighted points: `nautilus_summary.json`'s medians come from
  `sampler.posterior(equal_weight=True)` (`run_nautilus.main`), which draws about
  sum(w)/max(w) samples (256 for 90302) [M].
- Report from the certificate and quote the seed spread: nautilus 1.0.6 gives no logZ
  error; on the surrogates stock nautilus's logZ was -0.001 +/- 0.011 against truth
  over 8-24 seeds ("Sampler").

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
- Inference with condensation is refused: through the pinned state the pinned-species
  tangent is order-unity wrong (0.91 relative, FD ~ 1/dT); the gas + condensate reservoir
  matches to 18-22% on aarch64 and misses the 35% gate on x86 (rel. 0.407, xfail there).

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
- Kept names (`SMC_RETRIEVAL_*`, `SMC_RESUME`, `runs/w39b_smc_retrieval`,
  `smc_rt_chunk`): the NAS launch command and archived runs' `config.json` use them
  (`nautilus_report` rebuilds a run's Config from its `config.json`); no measurement.

## Design records

- Two-stage solve. With a moving T-P the hydrostatic density rebuild makes the fixed
  point forget the initial inventory (dL/dlnZ ~ 1e-20, FD agreeing). Stage 1 converges
  at (T(theta), Kzz(theta)) with baseline composition; stage 2 scales metals and C/O on
  that column and re-converges warm (+5% lnZ -> +5.29% CO; ~1.2-1.4x one solve). Any
  code that moves T and uses the composition knobs perturbs a T-converged column. The
  starting column is repaired on He/H2O/CO/N2/H2S so column ratios hit their targets to
  ~1e-15; atom_ini is rebuilt from it.
- Certification: `ConvDiag` rides the runner's primal carry; usable = valid &
  (accept_count < cap) & conv_normal; stalled exits are a rejection class. Healthy warm
  proposals certify on the LOOSE branch, so a tight-only predicate (P2) rejects
  everything, and P0 (longdy < yconv_min at exit) is a no-op for the failing class.
  `native_depth_aux` returns the certificate bit; `observed_depth_model_jit` and
  `observed_depth_from_y_jit` are the only ungated spectra (plots, diagnostics).
- Cold start: the in-graph Gibbs equilibrium (`vulcan3.ini_abun.eq_seed`, ExoGibbs) at
  the proposal's own T(P) and elemental ratios, the engine's only cold start; zero
  tangent, lnZ / c_o tangents enter through the elemental projection.
  `chem_solve_cold_diag` returns stage 2's ConvDiag.
- Lane queue: `cold_lanes > 0` runs every chemistry batch on
  `vulcan_forward.converged_y_queue` at `min(lanes, draws)` lanes, refilling a certified
  lane in the same while loop (`retrieval_forward._chem_batch_route`); 0 is the lockstep
  reference. The gpu preset sets `device_lane_count()` (132 on the GH200); the
  production overrides set 264 (v2) and 176 (v4), times FFI_BLOCKS_PER_SM 3 = 792 /
  528 lanes. One chemistry call per batch; only the RT is chunked.
- Growing anchors (`run_nautilus.Anchors`): each certified column is an anchor keyed by
  its unit-cube chemistry + T-P coordinates; a later solve continues from the nearest
  anchor at the cold cap. Never pruned (~44 KB per column);
  written before nautilus checkpoints the batch. The maintainer accepted an
  evaluation-order dependent likelihood for the speed.
- Fixed-mass geometry (vulcan-forward 0.28.0, a16188e): `transmission_depth_r` scales
  (rp_cm, g) at p_ref to (rp_cm e^lnR0, g e^(-2 lnR0)), so GM = 422 rp_cm^2 (Mancini+2018
  Table 15) holds at every level (`test_e2e_rt_reference.py`). p_ref = 1 mbar
  (continuum depths give lnR0 ~ +0.028). The chemistry Rp is overridden to 1.21 RJ (the
  fixed-GM 1-bar radius on the 90302 posterior), worth about x1.2 in equivalent Kzz.
- Split G395H offsets: Sarkar+2024 (arXiv:2405.06737 §5.2) find 40-50 ppm between the
  detectors of this visit; a step between the H2O (NRS1) and CO2/SO2 (NRS2) bands lands
  on Z. The offset reference group is the wavelength-first group (NIRISS).
- Run certificate (`nautilus_report.certificate`, version 1): provenance (the run's
  target digest and manifest, code commits, dirty flags and `src_diff`, versions,
  per-file k-table + CIA sha256, and the same for the code now), the sampler settings,
  `logZ_box`, `n_eff`, `n_like`, `logZ_err` null, the warm/cold start mode, the
  rejection tally as fractions of the in-window draws, the chemistry settings (`rtol`
  from the summary's `chem_rtol`, `chem_retry`, `count_max`, ...) and plain-English
  caveats. No pass/fail
  verdict and no gates: a dirty tree, code moved since the run and a manifest that does
  not hash to the digest are caveats, not refusals; there is no prior-rail check (an
  84% / 99% quantile against a prior edge was planned and not built) and no attrition
  threshold. `src_diff` hashes untracked files under `src/`. The resume refusal lives in
  `run_nautilus.main` (target digest, start mode and sampler settings) and runs before
  the identity files and the digest are written (only `run.log` comes earlier).
- Engine split (2026-07-29): the chemistry and RT engine moved to vulcan-forward,
  bit-identical on the planner's W39b spectrum (the gate for engine surgery).
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
- Validation artifacts: none in the tree, and nothing reads them automatically; both
  ladders are owed at the current config.
  Keep `art_nlayer` only if the depth (5 ppm) and Jacobian (1%) gates pass; keep the
  model top only if the one-decade extension passes.

## Physics assumptions and limitations (read before publishing a number)

- Planet mass fixed at 0.281 MJ: its 11% uncertainty is an 11% systematic on the
  photospheric T/mu and ~20% on the equivalent Kzz, in no posterior width (todo).
- Guillot T-P, f = 1/4, no convective adjustment: (Tirr, kappa, gamma, Tint) shape one
  1D limb; Tint is a deep-temperature knob. One profile drives both grids.
- Closed column; lnZ / c_o are exact column-ratio directions; fixed-O C/O keeps each
  layer's oxygen while b_z > 0. Everything is rebuilt per proposal except the photolysis
  cross-section T-interpolation (host side, second order).
- T-P window [320, 2980] K inside the ExoMolOP tables' 100-3400 K.
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
  - Cap off by one: the runner stops on `count > count_max`, the rejection rule is
    `accept_count >= count_max`.
  - `run_nautilus` solves T-P out-of-window draws before masking them.
  - `opacity_leave_one_out.py` reads an archived run through the current names and
    priors: use it on the run it came from.
  - The ladder scripts build the RT inputs by hand (no `gas_mask`, `_artifact.bin_trapz`).
  - `validate_env` lets exogibbs 0.6.0rc1 pass the 0.6.0 floor; `XLA_CMDBUF` appends to
    an inherited `XLA_FLAGS`; `tools/gpu_kernel_test.pbs` installs on the shared userbase
    with no lock; the likelihood's -sum(log sigma) term is pinned by slow tests only.

## Literature (WASP-39b comparison set)

arXiv ids for the "Runs" comparison: ERS 2208.11692; Rustamkulov 2211.10487; Alderson
2211.10488; Feinstein 2211.10493; Tsai 2211.10490 (Kzz only x0.1 and x10 tested);
Espinoza 2407.10294; Carter & May 2407.13893; Thorngren & Fortney 1811.11859 (interior
ceiling 54.5x); Fu 2507.15854 (limb averaging biases T low, mu high). Unverified: Powell
2024 MIRI 7.1-8.0x (a search snippet). Full kinetics inside the sampler on real JWST
data with AD gradients: no precedent found (searches 2026-07 and 2026-09, not exhaustive).

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
  `archive/per-network-kwargs-silu-2026-10-19`.
- Retry C two-pass chemistry (82a0130, baseline_v2): retry columns not converged at 0.1
  nat; replaced in production by the presolve chain (cdcffd1). `chem_retry` alone stays
  in the code.
- 62 chemistry layers in production: bistable at low Kzz, up to 0.45 sigma resolution
  error; the next run uses 100 by override (the gpu preset is still 62).
- RT anchored at 7 bar with fixed g (90302 and earlier): the model's GM was e^(2 lnR0)
  of the planet's (g ratio 0.83 / 0.87), every scale height 15-21% too large; replaced
  by fixed GM at 1 mbar. 90302's lnR0 is not comparable with later runs.
- Prior edges: Tint fixed 150 K -> free 100-500 K (a16188e) -> 50-600 K (8bc1425); Kzz
  x0.01..x100 -> x0.1..x100 (52ad7be; 12 of 15 draws below x0.03 rejected at 5000) ->
  x0.1..x1000 (18da2a0; the 81542 posterior piled against x100); Tirr floor 1100 -> 800
  (18da2a0) -> 700 K (a16188e); one G395H offset -> NRS1/NRS2 (a16188e); one noise
  inflation -> per instrument (4581e78, opt-in).
- FastChem cold seed -> in-graph ExoGibbs (0.18.0, d549f34); the baseline-composition
  cold start and `cold_seed` went in 0.26.0. Abundance `masks` mode -> elemental only
  (0.26.0, 75cceff). Block-MALA schedule (0d56afa, deleted d549f34); `Stage1Cache`
  (removed cd96136); `warm_extrapolate`; single-stage cold solve (removed 7a1c2e1).
- Sampled line-by-line opacity and the Mie deck: retired with vulcan-forward 0.11.0.
- Route B smooth open-system rainout: NO-GO; the code is in
  `/home/imalsky/VULCAN_Project/.backup/git_tags_2026-09-26/vulcan-retrieval-smooth-rainout-b0c-no-go-2026-07-14.bundle`.

## Ideas not adopted

- Abort doomed lanes early (dt < 1e4 s and reject fraction > 0.3 at step 801: 64/75
  caught, 1 false positive). A new rejection rule: the maintainer's call.
- RT overlapped with the retry pass; longest-first ordering; halving the host update.
- AD warm-start polish (-8 to -16% wall [E]).
- nautilus: bootstrap logZ error within shells plus per-shell diagnostics; parallel
  network fits (`pool=(None, Pool(4))`, -10 to -18 min per run); n_live 700-800 /
  f_live 0.05-0.1 after a surrogate test; allocation from exploration-phase shell
  statistics if spreading ever returns (unbiased, 25-32% fewer calls than stock).
- RT: unrolled searchsorted interp (bitwise, ~4% faster); CUDA fold.
- Non-uniform pressure grid support in VULCAN3 (a 284-layer non-uniform grid matched
  uniform 600 to <= 0.06 sigma at ~55 h).
- A fixed anchor set hashed into the manifest (Alquier et al. arXiv:1403.5496): warm
  starts that stay a deterministic function of theta, ~2.7x [I].
- A per-draw chemistry geometry shared with the RT (the 1.21 RJ override leaves a few %
  in H).
- Krylov on the stage system: untested; needs matched GPU time against the pivoted
  Thomas solve on large-dt states (cond ~6.7e23) at equal accuracy.

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
- Seeding a new run's anchors from 90302's 213k columns: no measurable gain (0.79 s per call in exploration, 0.77 in sampling) and a different box and geometry.
- A learned initializer (an emulator of y*(theta) as the warm start): the draw's own converged column still costs 191-1321 steps, so no start beats cold by more than 4.4x; the nearest anchor gives 2.13x, PCA+GP 2.05x; no start rescues a censored draw (record: `/home/imalsky/VULCAN_Project/.backup/speed_studies_2026-09/emulator_ideas/results.txt`).
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
- Comma lists in `qsub -v` (`XLA_FLAGS`): the commas split the variable list; use `XLA_CMDBUF`.
- nsys `--cuda-graph-trace=node` under command buffers: `CUDA_ERROR_ILLEGAL_ADDRESS` mid-capture (79666); profile with command buffers off or graph-level tracing.
- Chaining jobs on `afterok` of a kernel test: one missing module deleted five benches; chain on `afterany` and let `validate_env` refuse.
- Merging a file without `ruff check`: an undefined `log` killed five GH200 jobs after their compile; ruff every merged file and run a new job path end-to-end on the CPU first.
- A 24 h multi-arm calibration chain as a profiling vehicle: a MALA calibrate arm costs ~11 h, so the profiling arms never ran; profile in short bounded jobs.
- Expecting `LuPivotsToPermutation` count 0 as the perm-cache check: `lax.linalg.lu` reports its permutation through that kernel; 1:1 with getrf is correct.
- Shrinking the slow tests' memory (2 probe particles, `jax.clear_caches()` between compiles): wall -13% or peak 28.6 -> 23.9 GB in one file only; the peaks are single XLA CPU compiles.
