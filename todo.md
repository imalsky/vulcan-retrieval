# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [science] Build an injection-recovery test that runs in under 10 minutes on daw, or show that none fits.

## 1. Injection-recovery test

A success gate of the inverse model. Evidence: notes "Design records" (the synthetic injection refuses an uncertified
truth). Start from `runs/w39b_smc_retrieval/smoke.yaml` (nz 30, CO only, a synthetic injection on the G395H bin grid, 8
parameters); its run time is not recorded [unverified]. The test injects at a known theta with
`pipeline.generate_observations`, runs `run_nautilus`, and checks that the truth lies inside each sampled parameter's
95 % credible interval from the weighted points. Done when that test passes in under 10 minutes of wall time on daw CPU
(marked `slow` if it does not fit the normal tier), or, if no configuration fits, when the measured time is in notes and
the item goes back to the maintainer.
