# todo

The evidence behind each item is in `notes.md` (sections named below); solver and engine
items live in VULCAN3's and vulcan-forward's own `todo.md`.

1. [owed] Rewrite the three WASP-107b certificates on NAS with the report's fixed Kzz block (absolute Kzz, floor weights from the prior, no x10 ceiling off W39b's profile); their notes "Runs" numbers were derived by hand.
2. [robustness] Make the manual pre-launch checks unnecessary: a user who writes a spec and runs `tools/submit.py` should get the protection the opt-in production preflight test, the per-case cold prior batch (the survey's pre-launch check) and the "run a new job path end-to-end on the CPU first" rule give today; the refusal rule in step 3 below is the maintainer's decision.

## 1. WASP-107b certificates

Evidence: notes "Design records" (the run certificate's `derived` block) and "Runs" (the
w107b entries, whose log10 Kzz and "above 10^11.6" weights were computed by hand). On
NAS, after `git pull --ff-only` and the bootstrap: add `"network":
"thermo/SNCHO_photo_network.txt"` to each run's `config.json` (the report refuses one
without it), then from the repo root `python -m retrieval_framework.nautilus_report
output/w107b/<run>_nautilus --spectra 0` for `w107b_joint`, `w107b_welbanks` and
`w107b_g395h`. Done when the three rewritten certificates' Kzz blocks are in "Runs".
## 2. Pre-launch checks folded into the launch

Evidence: the maintainer asked for it (users will not know to run separate checks); a
design review (Codex) checked against the code, ranked by value / cost:
(1) CI: add `ruff check .` (no workflow lints today [M]; the undefined `log` that killed
five GH200 jobs, notes "Do not retry", is a ruff F error) and one driver-level
`run_nautilus.main` test on the stub pipeline (startup, likelihood callback, governor
return, resume). (2) Spec-time refusals: move configuration-only checks now deferred to
the pipeline build (the `prior_c_o` b_z bound in `build_retrieval_forward`) into
`validate_config`; `submit.py` and `validate_env --spec` check the spec's own
observation products and case files (`provenance.case_inputs`), where
`validate_env._check_data_tree` checks only `data/cm24_wasp39b` today [M]. (3) The first
production batch as the startup check, in `run_nautilus` (`make_loglike`): per log10 Kzz
decade at 1 mbar (`nautilus_report._kzz_at_ref`) record in-window proposals, certified,
finite log L and the exhausted / stalled / non-finite counts; refuse the run, never a
draw, when an adequately sampled region has zero certificates (a candidate rule, to
calibrate first: 0 of >= 135 in-window draws over ten bins; the notes' "raise if > 10% of
cold draws fail" stays rejected), and persist the status with the target so a resume
cannot skip it. Limits: the batch samples the box prior (v5's first batch had 307 of 1584
in window), so sparse bins report "insufficient coverage", never "passed"; it measures
cold support only. This also replaces the opt-in production preflight test. (4)
Full-stack CI: `run_nautilus.main` on `smoke.yaml` with a test-bounded `n_like_max`
(stock nautilus has it), cold, warm and resume. (5) Designed corners only as an explicit
contract, after `e2e_check._corners` starts from an in-window origin. Done when a new case
needs only a spec and `submit.py`, and the survey's separate pass, the opt-in preflight test and
CLAUDE.md's CPU end-to-end rule are retired.
