# w39b_smc_retrieval: decision history and open items

Moved verbatim from this case directory's pre-0.6 README (last updated
2026-07-10). Path references are historical. Current usage lives in the package
README's "The WASP-39b case" section (`../../README.md`); the operational rules
and full incident history are in the repo-root `CLAUDE.md`.

## Status / open items (2026-07-10)

- **Resolved history** (full details in `../../CLAUDE.md`): the >10k-step tail was
  `dt_max` ballooning (fixed, `dt_max=1e11`, `count_max=5000` — do NOT raise);
  measured residual non-convergence at the prior is ~27-30%, absorbed by
  init reject-and-oversample; the job-64745 sweep pathology is fixed by the
  `warm_count_max=1500` mutation cap + merged diag (+ N=96, 6 sweeps, 20 h
  governor); the job-64854 init failure is fixed by running init phase 2
  UNCAPPED (the mutation cap must not gate proven survivors).
- **`warm_extrapolate=true` in the gpu preset** (Isaac, 2026-07-10; schema default
  stays false): the measured-1.65x tangent-extrapolated warm start rides the whole
  staged sequence (calibrate → SYNTH → production), so SYNTH + the automatic
  warm-vs-cold validation gate it before real data. `warm_count_max` stays 1500;
  drop toward ~800 only after heartbeat rejected-counts confirm the margin.
- **Init phase 2 = uncapped + cull-and-backfill** (jobs 64854/64897): marginal
  survivors that cannot re-certify warm are culled and backfilled from
  `init_phase2_spare=8` extras — expect a loud but benign warning; a genuine
  RT/AD failure still raises. Report the phase-2 cull count with the phase-1
  reject fraction (operational prior).
- **N=144 as of 2026-07-10** (raised from 96 to spend the measured GPU power
  headroom on particles — ~300 of 700 W drawn during primal phases; width is
  nearly free in the lockstep chemistry, RT tail goes 8 → 12 chunks). Probe job
  64944 PASSED: peak memory is width-independent (73.25 GiB at N=96/144 and the
  152-wide init eval alike — the peak is the fixed-width RT-vjp chunk stage), so
  N=192 is memory-viable if ever wanted. Two XLA launch-overhead A/B candidates
  (autotune=4, CUDA-graph command buffers) are documented in the PBS header;
  judge by `t_mutation_sweep_s`.
- **Before trusting the real-data posterior:** one clean `SYNTH=1` recovery at gpu
  fidelity, and `VERDICT: PASS` from the automatic warm-vs-cold validation.
