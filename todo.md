# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [blocker] The sulfur survey: two of its six runs resume from their checkpoints, with the three trees held at their commits; the other four froze in a warm start and wait for the maintainer's choice of how such a draw is handled.

## 1. Sulfur survey runs

Evidence: notes "Current state" (Next), "Design records" (the step floor and the frozen draws, the sulfur survey case)
and "HPC (NAS GH200) and daw". State at 2026-11-10 12:45 PST, from qstat and the run folders.

- Layout (the maintainer decided): six planets on two `gpu_long` nodes, three per node, one after another. HAT-P-26b
  is dropped (104992 never started).
- Resuming: hd189733b and hd209458b, from the checkpoints of 104990 and 104991 (17 and 15 batches, stopped by qdel with
  no freeze); the job ids are in each run's `jobs/jobs.txt`. The target digest binds each tree's HEAD commit, so the
  resumes submitted from later commits (105516, 105517) were refused at startup, though no `src` file differs but one
  error message's text. The three trees are therefore detached at the first submission's commits (VULCAN3 `05db2db`,
  vulcan-forward `9ae2fa0`, this repo `49badf9`) while the two runs go: no commit, pull or checkout in any of them
  until both end, then `git switch main` in each. The detached tree shows this file as it was; `git show main:todo.md`
  is the current one.
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

Next: the maintainer picks how a frozen warm draw is handled; the options went to the maintainer with these results.
It changes the start policy or the abort rule, so the four restart fresh on a new target, each after a running planet
on its node. A plain `--resume` of the four likely redraws the same batch, since the nautilus checkpoint restores its
random state. After the decision, move the reproduction into the tracked tree and record it in notes "Design records".

After each run: `nautilus_report`, the warm-vs-cold check, and the rtol 0.05 re-solve of the posterior rows within 15000
steps (notes "Design records", sulfur survey case); its numbers go to notes "Runs". Done when the six runs are in
"Runs".
