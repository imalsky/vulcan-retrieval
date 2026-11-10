# todo

The evidence behind each item is in its section or in `notes.md`; solver and engine items live in VULCAN3's and
vulcan-forward's own `todo.md`.

1. [blocker] The sulfur survey waits on the comparison of the end-to-end check at its WASP-39b spec; then launch the seven runs.

## 1. Survey launch

Evidence: notes "Current state" (Next) and "HPC (NAS GH200) and daw". NAS job 104972 (`tools/check_nas.pbs` with
`VR_SPEC=runs/survey/wasp39b.yaml`, rungs `production:shuffled:fast`) solves the draws and writes
`output/w39b_smc_retrieval/validation/e2e_<rung>.npz` (the default results folder whatever the spec; v5's files of the
same names are kept in `v5_e2e_before_survey_check/` there). The job only solves. On the front end, in the activated
env, from the repo root:

    V=output/w39b_smc_retrieval/validation
    python validation/e2e_check.py compare $V/e2e_production.npz $V/e2e_shuffled.npz   # the noise floor
    python validation/e2e_check.py compare $V/e2e_production.npz $V/e2e_fast.npz       # ffi against fast

An exit of 1 is a gate exceeded: explain it before any launch (docs/method.md, e2e_check). Also read the production
rung's log for the peak memory and tick time at 528 lanes on the 93-species network, not yet measured on a GH200.
Then `PBS_GROUP=<allocation> python tools/submit.py runs/survey/wasp39b.yaml`; when its startup check has passed
(`run.log`: kept draws per Kzz decade), the other six (the maintainer allowed all seven queued at once). After each
run: `nautilus_report`, the warm-vs-cold check, and the rtol 0.05 re-solve of the posterior rows within 15000 steps
(notes "Design records", sulfur survey case); its numbers go to notes "Runs". Done when the seven runs are in "Runs".
