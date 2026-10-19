# vulcan-retrieval notes: WASP-39b campaign (Oct 2026)

What the development studies found. Their outputs lived in `analysis/` on NAS, deleted
2026-10-19; a reference to `analysis/<name>` in code comments or override files means
the `<name>` entry here, or in VULCAN3's / vulcan-forward's `notes.md` (solver and RT
findings). Open items are in `todo.md`. [M] measured, [E] estimated or modelled,
[I] inferred. sigma = posterior standard deviation, L = log-likelihood.

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
  noise, log10gamma down to -3, 792 lanes x n_batch 2376, compaction [264, 132], fork
  nautilus (SiLU + spreading: not approved, see Sampler).
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
- PRESOLVE CHAIN (vulcan-retrieval cdcffd1, opt-in `chem_presolve`, config_schema.py:
  289-303; run_nautilus.py:428-434, :545-560; overrides/w39b_baseline_v3_presolve.json):
  stage A chemistry-only presolve at rtol 0.2, photolysis every step, cap 3000, never
  kept; stage B re-solves every in-window draw at rtol 0.05, photolysis every step, cap
  5000, warm from A's exit column where A certified, otherwise from the row's original
  start. Only stage B's VULCAN certificate keeps a draw. Detail: on a cold chunk a draw
  whose presolve fails still starts stage B from the presolve exit column
  (run_nautilus.py:556-558).
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
- Next run: uniform 100 layers (user's choice), overrides/w39b_baseline_v4_nz100.json.
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
  spreading (2026-10-19). Kept settings: n_live 1000, n_update n_live/2, f_live 0.03,
  enlarge_per_dim 1.3, n_batch 3 x lanes, discard_exploration True.
- nautilus_improve, upside-only ideas, none adopted: (a) bootstrap logZ error within
  shells plus per-shell diagnostics; (b) `pool=(None, Pool(4))` to fit the 4 networks in
  parallel (-10 to -18 min per run); (c) allocate from exploration-phase shell
  statistics if spreading ever returns (unbiased, 25-32% fewer calls than stock).
  Need a surrogate test first: n_live 700-800, f_live 0.05-0.1.
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
    [320, 2980] K down to 7 bar (tp_profile.py:44-50, pipeline.tp_valid). Guillot is
    radiative with one constant kappa: deep T^4 ~ 3/4 Tint^4 kappa P/g. At the preferred
    kappa (~1 cm2/g) tau(7 bar) ~ 1.6e4, so Tint 358 K gives ~4400 K and Tint ~280 K hits
    the 2980 K edge. Tsai's GCM with Tint 358 K is ~1600 K at 1 bar. Tint here is a knob
    for the 1-7 bar temperature, degenerate with kappa (Tint ~ kappa^-1/4), and the
    pile-up is a window / prior-volume effect. Expect it to move to 50 K.
  - Fix options: (A) apply the window only above ~1 bar (lets 4000 K deep layers into
    the chemistry; not alone); (B) retrieve T at 1 bar instead of Tint (~1/2-1 day);
    (C) a physical profile in our code: pressure-dependent opacity plus an adiabat below
    the convective boundary, e.g. Parmentier+2015 (several days with validation; with
    constant kappa the deep gradient is 0.25 < adiabatic ~0.29, so a convective switch
    alone never triggers).

## Rate sensitivities (adjoint_w39b, 5.4 GPU-h)

- Method: VULCAN3 adjoint of log L through the RT for every rate row (~1150) on 21
  posterior columns of 93269, tight rtol-0.05 solves (steady_state_grad.py:
  _adjoint_solve_core :581, steady_state_reaction_sensitivity :703). Validated against
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
  rows, RT only on kept rows (764ccb8; RETRY_BUCKET=132 run_nautilus.py:136,
  NAUTILUS_PRECOMPILE), plus VULCAN3 a5b1215 / eeb4873 and vulcan-forward 5e8d43a /
  39c2c45. Pass-1 batch 344.0 -> 287.3 s; full batch ~892 -> 628 s excluding compiles;
  RT pass 256 -> 60 s.
- Whole run [E]: ~11 h at 62 layers (~3.4x the fixedmass run's 37.2 h), with the fork
  sampler's call count.
- Not done: RT overlapped with the retry pass, aborting doomed lanes (a new rejection
  rule: user's call), longest-first ordering, halving the host update.

## Provenance audit (2026-10-19)

- Nothing made on NAS changed science without a recorded request, apart from the items
  below. Everything before 2026-10-13 came from other machines and cannot be traced
  here (256 vulcan-retrieval commits, ~70 change science).
- NAS-session lapses: the fork sampler was carried into the approved v2 run without
  telling the user; the presolve fallback above was never described; 764ccb8 and the
  10-17 mains were pushed without an explicit OK (user: keep them).
- Live on main, provenance unknown (each needs confirmation): 3eafb1b discard_exploration
  and ffi solver default (run_nas_w39b.pbs:213); 3e40f03 + 4c23465 warm-start anchors and
  warm nautilus by default (results depend on evaluation order); 8e633bf + 500b340 lane
  queue default; a16188e p_ref 1 mbar, fixed-mass lnR0, chemistry Rp 1.21 RJ, free Tint,
  split NRS offsets, Tirr 700-2200; 18da2a0 Kzz ceiling x1000 chosen after seeing the
  data; 52ad7be/611f62b Kzz floor x0.1; cc6a1e9 "Snapshot" (noise inflation, correlated-k
  default, full-covariance MALA); 24bd41b 12 molecules, 2 bins dropped; 5792482 and
  d549f34 cold gate, in-graph equilibrium seed, XLA command buffers; f05e537 + 88f995e
  certification gate on stall/budget exits; 93ed4fd elemental abundance map (redefines
  lnZ and C/O); 111ee8c + da7418e certificate failures turned into warnings; c7899ee
  hybrid vm_mol 07-14..07-26 (reverted: check no reported result comes from that window).
  Five commit messages claim "the maintainer's" decision: 52ad7be, 111ee8c, 500b340,
  d549f34, fc52136.
- The SMC path still defaults to MALA (config_schema.py:244); production uses nautilus.

## Environment

- The nsys-jax used by PROFILE_BATCH lives in `$VULCAN_SCRATCH/tools/nsysjax_venv`
  (rebuild recipe in run_nas_w39b.pbs next to NSYS_JAX).
- JAX compile cache: see VULCAN3 notes (private cache for any memory fraction != 0.90).
