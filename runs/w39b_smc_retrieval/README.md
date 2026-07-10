# w39b_smc_retrieval — WASP-39b retrieval case

Fits the **real Carter & May (2024) combined JWST transmission spectrum of
WASP-39b** (NIRISS SOSS + NIRSpec G395H, `../../data/cm24_wasp39b/`) with the
reusable differentiable VULCAN-JAX → ExoJax SMC retrieval framework in
`../../retrieval_framework/` (see its README for the algorithm, the staged GPU
architecture, memory/count_max engineering notes, and validation history).

This directory holds ONLY what is specific to this run:

- `case.py` — planet identity (gravity, radii, VULCAN cfg module, C&M product
  table) + the `smoke` / `gpu` / `prod` presets (`PRESETS` dict).
- `run_nas_w39b.pbs` — the NAS GH200 submit script (all modes: run, SYNTH,
  CALIBRATE_ONLY, CALIBRATE_COUNT_MAX, PROBE_MEMORY, NSYS profiling).
- `overrides/*.json` — optional Config-override files
  (`SMC_RETRIEVAL_OVERRIDES_FILE=overrides/<f>.json`, resolved against this dir).
- `data/<preset>/` — run outputs (posterior npz, config.json, run.log, plots/).
- `logs/` — PBS live logs + GPU monitor + nsys reports.

## Run

Local smoke (offline, CPU, ~minutes; always do this after framework changes):

```
cd ../..    # vulcan_exojax_run/
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval   # gradient FD checks
```

NAS GH200 (from this directory), in the staged order for a fresh campaign:

```
qsub -v PROBE_MEMORY=1 run_nas_w39b.pbs        # compile-only buffer report (REQUIRED after any N/chunk/nu_pts change)
qsub -v CALIBRATE_ONLY=1 run_nas_w39b.pbs      # ~1.5 h; check timing.json t_mutation_sweep_s
qsub -v SYNTH=1 run_nas_w39b.pbs               # synthetic recovery test at gpu fidelity
qsub run_nas_w39b.pbs                          # real-data production (gpu preset)
qsub -v RESUME=1 run_nas_w39b.pbs              # continue a governor-stopped ladder
qsub -v CALIBRATE_COUNT_MAX=1,CALIBRATE_COUNT_MAX_PROBE=60000,CALIBRATE_N_DRAWS=96 run_nas_w39b.pbs
```

On success, plots + the warm-vs-cold validation (`validate_warm`, PASS gate
max|dlogL| < 0.1) run automatically; quote the verdict and the init reject
fraction in the paper.

## Status / open items (2026-07-10)

- **Resolved history** (full details in `../../CLAUDE.md`): the >10k-step tail was
  `dt_max` ballooning (fixed, `dt_max=1e11`, `count_max=5000` — do NOT raise);
  measured residual non-convergence at the prior is ~27-30%, absorbed by
  init reject-and-oversample; the job-64745 sweep pathology is fixed by the
  `warm_count_max=1500` mutation cap + merged diag (+ N=96, 6 sweeps, 20 h
  governor); the job-64854 init failure is fixed by running init phase 2
  UNCAPPED (the mutation cap must not gate proven survivors).
- **`warm_extrapolate=false` by default** (the measured-1.65x tangent-extrapolated
  warm start): flip it only after the same-seed `SYNTH=1` A/B validates it
  (`SMC_RETRIEVAL_OVERRIDES='{"warm_extrapolate": true}'`), then consider
  `warm_count_max` 1500 → ~800.
- **N=144 as of 2026-07-10** (raised from 96 to spend the measured GPU power
  headroom on particles — ~300 of 700 W drawn during primal phases; width is
  nearly free in the lockstep chemistry, RT tail goes 8 → 12 chunks). Projected
  gradient-eval peak ~79.5 GiB vs the 73.25 GiB probed at N=96: **run
  `PROBE_MEMORY=1` before the first N=144 submit.** Two XLA launch-overhead A/B
  candidates (autotune=4, CUDA-graph command buffers) are documented in the PBS
  header; judge by `t_mutation_sweep_s`.
- **Before trusting the real-data posterior:** one clean `SYNTH=1` recovery at gpu
  fidelity, and `VERDICT: PASS` from the automatic warm-vs-cold validation.
