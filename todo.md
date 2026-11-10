# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [owed] The sulfur survey runs on the NAS (jobs 104985 and 104987-104992): read each run's startup check, then do its after-run checks.

## 1. Sulfur survey runs

Evidence: notes "Current state" (Next), "Design records" (the end-to-end check on the survey's spec, the sulfur survey
case) and "HPC (NAS GH200) and daw". All seven specs of `runs/survey/` are submitted from the checkout at `49badf9`:
wasp39b 104985, wasp107b 104987, wasp96b 104988, wasp166b 104989, hd189733b 104990, hd209458b 104991, hatp26b 104992.
The other six were submitted before WASP-39b's startup check had printed (the maintainer decided), so read each
`output/survey/<planet>_nautilus/run.log` for its `startup:` lines and for a `startup check:` error. Keep the checkout
unchanged until all seven end.

wasp96b and wasp166b were moved to `gpu_normal` with a 24 h wall (the maintainer asked for two runs there beside the
two `gpu_long` allows at once). Their governor is still 46 h, so PBS ends such a run at 24 h if it is not done: then
`python tools/submit.py runs/survey/<planet>.yaml --resume` continues it from its last checkpointed batch.

After each run: `nautilus_report`, the warm-vs-cold check, and the rtol 0.05 re-solve of the posterior rows within 15000
steps (notes "Design records", sulfur survey case); its numbers go to notes "Runs". Done when the seven runs are in
"Runs".
