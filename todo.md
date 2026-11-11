# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [blocker] The sulfur survey: nothing is queued or running; two of its six runs hold checkpoints that resume only on their first submission's trees (the current survey spec is a new target), and the other four froze in a warm start and restart fresh on the cold re-solve of a frozen warm draw.
2. [stale] `requirements-hpc.txt` pins vulcan-forward `9ae2fa0`, below this package's floor 0.31.0.
3. [decision] The warm-against-cold check passes on the median shift alone, with no effective-sample threshold.
4. [decision] `art_nlayer` 67 sits at the 5 ppm ladder gate in the CO2 band of the survey's WASP-39b column, before the survey runs.
5. [unverified] The job-path test has not run end to end on the current tree (the run-specification and engine interface change).

## 1. Sulfur survey runs

Evidence: notes "Current state" (Next), "Design records" (the step floor and the frozen draws, the sulfur survey case)
and "HPC (NAS GH200) and daw". State at 2026-11-10 13:28 PST, from qstat and the run folders.

- Layout (the maintainer decided): six planets on two `gpu_long` nodes, three per node, one after another. HAT-P-26b
  is dropped (104992 never started).
- Stopped with checkpoints: hd189733b (20 batches) and hd209458b (17 batches), stopped by qdel with no freeze (104990
  and 104991, then the resumes 105519 and 105520, stopped at the maintainer's request). The target digest binds each
  tree's HEAD commit, so a resume runs only with the three trees at the first submission's commits (VULCAN3 `05db2db`,
  vulcan-forward `9ae2fa0`, this repo `49badf9`): the resumes submitted from later commits (105516, 105517) were
  refused at startup, though no `src` file differs but one error message's text. The trees are back on `main`.
- Waiting: wasp39b 104985 (batch 4), wasp107b 104987, wasp96b 104988 and wasp166b 104989 (each in batch 2, the first
  warm batch) ended on `DtMinFrozenError: 1 of 1584 columns froze at dt_min`. The traceback and the frozen draw's
  unit-cube point are in `output/survey/<planet>_nautilus/jobs/<job>.pbs05a.gh.nas.nasa.gov.OU`, not in `run.log`.
- The four frozen draws, re-solved alone on a GH200 (`ffi`, one lane; jobs 105008, 105010 and 105011; scripts and logs
  in the git-ignored `scratch/frozen_repro*`):
  - Warm from its nearest anchor (unit-cube distance 0.29 to 0.51) each freezes again after 3 or 4 accepted steps, at
    t ~ 3e-15 s: the start state freezes, not the path.
  - The failing test at `dt_min`: wasp39b O at layer 69 to -9.5e8 cm^-3 and wasp96b H2O at layer 68 to -2.8e8 cm^-3,
    below `nega_cut`; wasp107b CH and wasp166b C3H2 at layer 0, error ratios 0.45 and 0.378 against rtol 0.2.
  - Cold, on the two-stage route, none freezes: wasp96b, wasp107b and wasp166b certify (905 + 717, 430 + 1643 and
    2585 + 1046 steps), and wasp39b exits both stages at the cap.
  - The warm move taken in two does not cure all four. T-P and Kzz first: wasp39b freezes in the composition move,
    wasp107b and wasp166b in the first move. Composition first: wasp39b and wasp96b freeze in the first move, wasp107b
    in the T-P move.

Next: the maintainer decided that a frozen warm draw is re-solved cold (notes "Design records"; not yet run on the real
solver). Before the restarts, re-solve the four frozen points through `run_nautilus`'s warm-then-cold route on a GH200,
and move the reproduction scripts from the NAS `scratch/frozen_repro*` into the tracked tree. The change is a new
target, so the four restart fresh, each after a running planet on its node; hd189733b and hd209458b resume only on
their first submission's trees, which still abort on a freeze.

The survey base now adds H2CO and CS to `molecules` (notes "Design records", sulfur survey case, absorber screen), and
the target manifest is version 8: every survey run is a new target, and no checkpoint resumes on the current specs. The
NAS data tree needs `H2CO.ktable.h5` and `CS.ktable.h5` (not checked there).

After each run: `nautilus_report`, the warm-vs-cold check, and the rtol 0.05 re-solve of the posterior rows within 15000
steps (notes "Design records", sulfur survey case); its numbers go to notes "Runs". Done when the six runs are in
"Runs".

## 2. The HPC engine pin

`pyproject.toml` floors vulcan-forward at 0.31.0 (the caller names the network, its atoms and the RT bottom);
`requirements-hpc.txt` still pins `9ae2fa0`, a 0.30 commit, so a standalone reproduction env built from it fails
`validate_env`'s pin check. Next: replace the pin with the engine's released 0.31.0 commit once it is pushed. The daw
root env's installed vulcan-forward metadata also reads 0.30.0 until `uv sync` reinstalls it. Done when both read 0.31.

## 3. The warm-against-cold threshold

Evidence: notes "Runs" (wasp39_test1, warm vs cold) and docs/method.md "e2e_check". `validation/e2e_check.py warmcold`
reweights the cold re-solved rows by exp(L_cold - L_run) and passes when every parameter's median moves under 0.1
sigma; it reports no gate on the effective sample size of those weights. On the finished WASP-39b run (v5): 919 of
1000 rows certified cold, ESS 914, worst median shift 0.075 sigma (c_o) [M]; the earlier WASP-39b correction
(postrun_93269) had ESS 5324 of 7888 rows. A run whose weights collapse onto a few rows could pass on medians alone.
Decision for the maintainer: whether to add an ESS threshold (and its value), or keep the median gate alone.

## 4. The survey's RT layers

Evidence: notes "Physics assumptions and limitations" (RT bottom). On the survey WASP-39b planet at its nominal state,
101 against 67 RT layers moves the binned depth by 5.03 ppm at 4.346 um (gate 5 ppm, 0.055 sigma), from a scratch
script's own binning on one certified column; the WASP-39b case's ladder measured 1.02 ppm. Next, the maintainer's
call: `validation/resolution_ladder.py --spec runs/survey/<planet>.yaml` on a GPU, then keep 67 or raise `art_nlayer` in
`survey_base.yaml` (a new target). Done when the ladder is recorded for the survey case.

## 5. The job-path test on the current tree

Evidence: notes "Current state" (tests). `tests/test_run_nautilus.py::test_run_nautilus_runs_the_smoke_case_end_to_end_and_resumes`
last passed whole at retrieval `33ba761` (95 min). On the current tree its first leg ran twice (cold startup check and
a warm batch, as the test asserts), and both resumes were refused because VULCAN3 or vulcan-forward was committed
between the legs, so the resume and finish legs are not yet shown. Run it alone, with the three trees unchanged for
its 1.5 h: `python -m pytest tests/test_run_nautilus.py::test_run_nautilus_runs_the_smoke_case_end_to_end_and_resumes
-m slow`. Done when it passes and its time is in notes.
