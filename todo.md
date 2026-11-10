# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [blocker] The sulfur survey: WASP-39b runs as NAS job 104985; submit the other six when its startup check has passed, then do each run's after-run checks.

## 1. Survey launch

Evidence: notes "Current state" (Next), "Design records" (the end-to-end check on the survey's spec, the sulfur survey
case) and "HPC (NAS GH200) and daw". When `output/survey/wasp39b_nautilus/run.log` prints its `startup:` lines with no
`startup check:` error, submit the other six from the activated environment on a front end, repo root, the checkout
unchanged at `49badf9`:

    for p in wasp107b wasp96b wasp166b hd189733b hd209458b hatp26b; do
      PBS_GROUP=<allocation> python tools/submit.py runs/survey/$p.yaml
    done

After each run: `nautilus_report`, the warm-vs-cold check, and the rtol 0.05 re-solve of the posterior rows within 15000
steps (notes "Design records", sulfur survey case); its numbers go to notes "Runs". Done when the seven runs are in
"Runs".
