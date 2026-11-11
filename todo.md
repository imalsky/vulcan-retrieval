# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [blocker] The sulfur survey: nothing is queued or running; two of its six runs hold resumable checkpoints, and the other four froze in a warm start and restart fresh on the cold re-solve of a frozen warm draw.
2. [speed] The job-path test fills the 95-minute full tier; `oracle.md` caps a test at 90 minutes on a CPU and aims under an hour.
3. [owed] Checks `oracle.md` asks for that are not there yet: the absorber screen per survey planet, and a truth test with no sampler.

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

After each run: `nautilus_report`, the warm-vs-cold check, and the rtol 0.05 re-solve of the posterior rows within 15000
steps (notes "Design records", sulfur survey case); its numbers go to notes "Runs". Done when the six runs are in
"Runs".

## 2. The job-path test and the 90-minute cap

Evidence: notes "Current state" (the full tier's wall time, the job-path test all of it). The test is
`tests/test_run_nautilus.py::test_run_nautilus_runs_the_smoke_case_end_to_end_and_resumes`: the smoke case through
`run_nautilus` in separate processes, with a resume. `oracle.md` "Success: the repository's tests" caps a test at 90
minutes on a CPU and aims under an hour; the test stays, as the small synthetic case the job path is tested on. Where
its time goes is not measured (the processes each compile the chemistry program). Next: one timed run with
`--durations`, then cut what the breakdown allows (live points, the column, the number of processes) while the test
still fails when the job path breaks. The other slow tests' times are not recorded. Done when the test is under the
cap and its time is in notes.

## 3. Checks the oracle asks for

Each is in `oracle.md` "Success: the repository's tests" or "Success: a run that is reported".

- Absorber screen: the bound on the tabled species left out of the RT was measured on the WASP-39b case only (notes
  "Physics assumptions and limitations"); the survey planets have none. `validation/opacity_leave_one_out.py --states`
  per planet.
- Truth test: no test scores synthetic data at its own parameters against perturbed ones without a sampler.
- No clear-atmosphere option: `cloud_model` is `powerlaw` or `deck`, and a spec must state one; a retrieval with no
  cloud term needs a third value.
- Nothing reports a species that is abundant in the observable layers and has no opacity table
  (jwst-transit-authority's `certify.unmodeled_absorbers` does).

Done when each is run, tested or dropped from `oracle.md` by the maintainer.
