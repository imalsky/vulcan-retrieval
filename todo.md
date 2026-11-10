# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [blocker] The sulfur survey: four of its seven runs died on a column frozen at dt_min and the other three were stopped; find why each column froze before anything is resubmitted or resumed.

## 1. Sulfur survey runs

Evidence: notes "Current state" (Next), "Design records" (the step floor and the frozen draws, the sulfur survey case)
and "HPC (NAS GH200) and daw". State at 2026-11-10 00:54 PST, from qstat and the run folders: nothing is queued or
running, and all seven runs under `output/survey/` ended unfinished. All seven were submitted from the checkout at
`49badf9`.

- Four ended with exit 1 on `DtMinFrozenError: 1 of 1584 columns froze at dt_min`: wasp39b 104985 (batch 4), and
  wasp107b 104987, wasp96b 104988 and wasp166b 104989 (each in batch 2, the first warm batch). The traceback and the
  frozen draw's unit-cube point are in `output/survey/<planet>_nautilus/jobs/<job>.pbs05a.gh.nas.nasa.gov.OU`, not in
  `run.log`.
- wasp107b and wasp166b froze with last error ratios 0.45 and 0.38 against rtol 0.2. wasp39b and wasp96b froze with
  ratios 5.5e-10 and 1.3e-10, so the error test did not stop those steps; both sit at unit-cube Tirr 0.004 and 0.021,
  with C/O at 0.95 and 0.99.
- hd189733b 104990 and hd209458b 104991 were stopped by qdel after 17 and 15 batches, with no freeze; hatp26b 104992
  never started. Their checkpoints are kept.

Next: reproduce the four frozen columns from their points and their runs' `anchors/`, find why each froze, and take the
cause and the options to the maintainer before any change to the solver, the abort rule or a run setting. Nothing is
resubmitted or resumed until the maintainer says so: a plain `--resume` likely redraws the same batch, since the nautilus
checkpoint restores its random state.

After each run: `nautilus_report`, the warm-vs-cold check, and the rtol 0.05 re-solve of the posterior rows within 15000
steps (notes "Design records", sulfur survey case); its numbers go to notes "Runs". Done when the seven runs are in
"Runs".
