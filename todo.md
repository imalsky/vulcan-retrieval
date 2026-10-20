# todo

The evidence behind each item is in `notes.md` (sections named below); solver and engine
items live in VULCAN3's and vulcan-forward's own `todo.md`.

1. [decision] Launch the next run, baseline v4 at uniform 100 layers (needs only the maintainer's NAS pull of the three trees, the bootstrap and the qsub).
2. [science] After the v4 run, check the Tint posterior against the T-P window's effective prior, settle what is reported beside the posterior, and run the solver measurements owed to VULCAN3.
3. [decision] T-P profile: retrieve T at 1 bar instead of Tint (fix B), or build a pressure-dependent opacity plus adiabat profile (fix C).
4. [decision] Planet mass: keep it fixed as a stated 11% systematic on T/mu, or add a Gaussian ln M nuisance.
5. [unverified] Confirm the live retrieval choices whose provenance is not recorded, the Kzz bounds first.
6. [unverified] Confirm the 2026-10-17 TestPyPI releases and their tags.
7. [low] Archive the GitHub fork imalsky/nautilus.

## 1. Next run: baseline v4 at 100 layers

- Evidence: notes "Grid resolution", "Low-Kzz hole", "Sampler".
- Settings: `runs/w39b_smc_retrieval/overrides/w39b_baseline_v4_nz100.json` (presolve
  chain, uniform 100 layers, 528 lanes, per-instrument noise, log10gamma down to -3) with
  case.py's Tint prior uniform 50-600 K and stock nautilus 1.0.6.
- Wall [E]: ~16 h at 100 layers with the old fork's call count; stock without spreading
  took +62% calls on the surrogate, so expect ~20-26 h. The governor stops at 46 h of a
  48 h PBS wall; RESUME=1 continues.
- Before launch: the v4 target (the certificate without a cap clause and T-P-invalid
  draws masked before the chemistry, notes "Design records" and "Pipeline speed") is on
  this repo's main, to be pushed; `requirements-hpc.txt` pins vulcan3 `92c3b1e` and
  vulcan-forward `32adc85`. `git pull --ff-only` the three trees on NAS and re-run
  `tools/bootstrap_nas_env.pbs` (`validate_env` must pass). The full tier passed on daw
  on this tree (notes "Current state").
- From `runs/w39b_smc_retrieval` on cghfe01/02:
  `qsub -l walltime=48:00:00 -v SMC_RETRIEVAL_OVERRIDES_FILE=overrides/w39b_baseline_v4_nz100.json,SMC_RETRIEVAL_OUT_DIR=data/w39b_baseline_v4_nz100,NAUTILUS_N_EFF=5000,VULCAN3_CHEM_RHS=grouped,PROFILE_BATCH=3 run_nas_w39b.pbs`
- Optional: 660 lanes compiled at ~79 GiB but were never timed (528 is measured).
- Done means: the run finished, its numbers in notes "Runs", and
  `python -m retrieval_framework.nautilus_report <out_dir>` run on its output (the
  report's first real use; notes "Design records", run certificate) before any number is
  reported.

## 2. After the run: Tint, what to report, owed solver measurements

- Tint (notes "Priors and the T-P profile", Induced prior): the effective prior the T-P
  window induces is measured. Compare the v4 posterior with it and plot Tint against
  log10kappa. On a Tint ~ kappa^-1/4 ridge, report T at 1 bar (or at the quench level)
  instead of Tint.
- Report beside the posterior: the mass above Thorngren & Fortney's 54.5x
  (arXiv:1811.11859 Table 1), the posterior truncated at Kzz <= x10 (Tsai's w_rms H), the
  weight at the x0.1 floor (a solver limit), Tint's marginal against 297 K
  (arXiv:2410.04470) and 358 K. The run certificate has no prior-rail check: read the
  84% and 99% quantiles against each prior edge by hand (the 81542 Kzz posterior piled
  against x100 with its median inside).
- Already settled by the NAS studies (notes "Runs", "Low-Kzz hole"): rejected draws near
  the posterior hide no better fit (figdata); the noise misfit is NIRISS's (per-instrument
  inflation); warm vs cold moved 93269's logZ by -0.040 and parameters by <= 0.06 sigma
  (bistable_policy). Still open for v4: the same warm-vs-cold re-solve of posterior rows
  (the postrun_93269 method) before its logZ is quoted.
- Solver measurements owed after v4, on v2 posterior draws, tracked in VULCAN3's todo
  ("Confirm the departures from VULCAN master"): the zero-code A/B of the geometry and
  element-budget certificate terms, and the Ros2 stage repair on/off.
- Done means: the numbers in notes "Runs" and the maintainer's call on what is reported.

## 3. T-P profile

- Decide between fix B (retrieve T at 1 bar instead of Tint, ~1/2-1 day) and C
  (pressure-dependent opacity + adiabat, e.g. Parmentier+2015; several days, plan first).
  Evidence: notes "Priors and the T-P profile".
- Done means: the maintainer's choice recorded in notes; a chosen fix opens its own item.

## 4. Planet mass: systematic or nuisance

- Mancini+2018 (arXiv:1802.03859 Table 15): 0.281 +- 0.031 +- 0.006 MJ, R* 0.939 +-
  0.019 R_sun (the case uses 0.932). At fixed observed depths T/mu scales with GM: 11% on
  the photospheric T/mu, about 20% on the equivalent Kzz (notes "Physics assumptions and
  limitations"). One review rated it acceptable fixed for this run; another proposed a
  nuisance ln(M/M0) ~ N(0, 0.11). No mass parameter exists in the code.
- A nuisance needs: an RT argument scaling GM at fixed R(p_ref) in vulcan-forward
  `transmission_depth_r` (RT-only, like lnR0) and a Gaussian prior type in
  `config_schema.ParamSpec` / `pipeline.make_uspace` (today uniform and log10_uniform).
- Done means: the maintainer's choice recorded in notes.

## 5. Confirm provenance (retrieval side)

- The list is in notes "Provenance". First: the Kzz bounds (ceiling x1000 chosen after
  seeing the data, 18da2a0; floor x0.1, 52ad7be / 611f62b).
- The solver and engine items (VULCAN3's element-conservation projection, `_repair_stage`,
  the extra certificate terms, the FastChem abundance file swap and its unaudited commit
  batches; vulcan-forward's conv_normal geometry / element-budget terms and per-theta
  re-anchoring) are in VULCAN3's and vulcan-forward's `todo.md`; the two referees'
  verdicts are in VULCAN3's notes "Differences from VULCAN master".
- Done means: each item confirmed or reverted by the maintainer, the result in notes
  "Provenance".

## 6. TestPyPI releases of 2026-10-17

- vulcan3 0.18.0, vulcan-forward 0.29.0, vulcan-retrieval 0.27.0 and
  jwst-transit-authority 0.72.0 and their tags; provenance not recorded. All four
  versions are on TestPyPI, and this repo's `v0.27.0` tag is on 275723c (checked
  2026-10-19).
- Done means: the maintainer confirms them or yanks them.

## 7. Archive the nautilus fork

- GitHub `imalsky/nautilus`, Settings > Archive. Its branches are kept as tags
  `archive/vulcan-multishell-2026-10-19` and `archive/per-network-kwargs-silu-2026-10-19`;
  only `main` is left and the repo is not archived (checked 2026-10-19).
- Done means: archived.
