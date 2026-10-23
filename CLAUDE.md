# CLAUDE.md — vulcan-retrieval (dist `vulcan-retrieval`, import `retrieval_framework`)

The WASP-39b retrieval: nautilus nested sampling over a forward model that solves VULCAN
3 photochemistry per draw and computes the transmission spectrum with ExoJAX; production
on the NAS GH200s, tests and small runs on daw. A leaf of vulcan3 <- vulcan-forward <-
{vulcan-retrieval, jwst-transit-authority}: solver and engine live upstream; this repo
holds the likelihood, the sampler driver, the cases and the job scripts.

## Maintainer rules (all repos)

These four rules are identical in the project-root CLAUDE.md and every repo's; change
them together.

- **Layout.** Keep code, data and generated files apart. Shipped code, tests, inputs,
  fixtures and validation assets stay where the repo's CLAUDE.md puts them. Work asked
  for outside the product (analysis, one-off scripts and checks, their figures and
  results) goes to `scratch/<topic>/`, one folder per question; run results go to
  `output/`, and vulcan3's own figures to `plot/`. Keep these three top-level folders
  gitignored and never commit their contents; never put such work in `src/`, `tests/`,
  `tools/`, `validation/`, `data/` or the repo root. A durable conclusion goes to
  notes.md; a reusable check becomes a tracked test or validation script.
- **Figures.** An ad hoc plot request gets 1 to 3 figures, more only when asked, from one
  script in `scratch/<topic>/`: no helper module, style layer or extra variants.
- **Branches.** Work in the existing checkout on `main`. Create no branch, worktree or
  copy of the repo, and switch no branch, unless the maintainer asks.
- **Behind main.** At the start of work in a checkout, `git fetch origin`, then
  `git rev-list --count HEAD..origin/main`. If the count is above 0, tell the
  maintainer at once; if the fetch fails, say the state is unknown. This alerts and does
  not block; never pull, rebase or reset unasked. Check again before a commit, a push or
  a job on daw or NAS.

## Docs

`README.md` is the maintainer's: never edit it unasked. `notes.md` is the results of past
experiments and the ideas tried, by section, never a dated history; the measured reason
behind each rule below is in notes "Evidence behind the rules" unless another section is
named. `todo.md` is open work: read it first and
open or close items in the same pass. This file holds directions only. The project-root
`CLAUDE.md` doc, no-bloat and test-tier policy applies: the three are tracked and public
(no personal names except cited authors, no emails, IDs, accounts or keys, no chat
quotes; a decision is "the maintainer decided"); no other `.md` file; point at a notes
section by its heading name. Format and placement: the `doc-upkeep` skill. Before
proposing a solver, sampler or speed change, read notes "Do not retry" and "Ideas not
adopted".

## Setup and commands

- Install: a member of the root uv workspace (`uv sync` at the project root). Standalone:
  `pip install --no-deps -e .` (vulcan3 and vulcan-forward are TestPyPI-only), plus
  `'.[plot]'` for figures. Keep it editable (`tests/conftest.py` refuses otherwise);
  after a version or dependency bump reinstall all three trees editable, or
  `validate_env` fails on stale metadata.
- Import order is guard-enforced: `vulcan_forward.vulcan_chem` raises if exojax was
  imported first.
- Inputs live in the repo `data/` (`$VULCAN_PROJECT_ROOT` = the directory holding this
  repo); only the observed spectra (`data/cm24_wasp39b/`) are tracked.
- `runs/<case>/` holds a case's inputs only: one YAML run specification per run
  (`wasp39_test1.yaml`, the v5 configuration; `smoke.yaml`; any `base:` variant) and
  nothing else. A spec is the whole run: `runtime` (the engine's environment),
  `sampler` (nautilus) and `config` (every `Config` field it sets), never the PBS job;
  its stem is the run name and every output goes under `output/<case>/<name>_nautilus/`
  (`run_setup.load_spec`, `output_base`). A new run is a new file; a variant restates
  only the keys it changes (one level, whole values). Never set a run through the
  environment or the command line.
- Run from the repo root: `python -m retrieval_framework.run_nautilus
  runs/w39b_smc_retrieval/<name>.yaml [--resume] [--dry-run]` (`--dry-run` prints the
  resolved settings without jax), then `python -m retrieval_framework.nautilus_report
  <out dir> [--spectra N]` for the certificate and figures (`--spectra 0` skips the
  forward model). The PBS does not run the report. The report's three figures are a run's only figures; draw nothing else
  beside a run unless asked (notes "Design records", run figures).
- Tests, two tiers. Normal (everyday and CI): `python -m pytest tests -n 8` (`-n` needs
  pytest-xdist, in the root env; conftest defaults `JAX_PLATFORMS=cpu`; pyproject's
  addopts deselect `slow`). Full, before a production run or after a change to the
  chemistry/RT path or a likelihood gate: add `-m "slow or not slow"`; a single slow
  file or node id needs `-m slow`. Production forward build, opt-in:
  `RUN_PRODUCTION_PREFLIGHT=1 python -m pytest tests/test_set_observations.py
  tests/test_group_noise.py -m "slow or not slow"`.
- Mark a test `slow` if it builds a real pipeline (`build_pipeline` on a real config);
  claims that share one heavy build stay in one test (xdist builds a fixture per worker).
- Lint every changed file with `ruff check .` before a merge. After a push, check CI:
  `gh run list --repo imalsky/vulcan-retrieval --limit 1`.
- On daw: tests and single chemistry runs on CPU. A batched run uses the one GPU with
  the `fast` solver, so the lanes are `cold_lanes` as given: a variant spec with
  `runtime: {solver: fast, xla_flags: null}` and `config: {cold_lanes: 284}`, with
  `XLA_PYTHON_CLIENT_PREALLOCATE=true` in the shell (notes "HPC (NAS GH200) and daw").

### On NAS

- Submit from `cghfe01`/`cghfe02` to `gpu_long`, short jobs too (short walltime);
  never `-q queue@server`.
- `PROJECT_ROOT` holds `vulcan-retrieval`, `VULCAN-JAX` (the clone name is
  load-bearing) and `vulcan-forward`. Update with `git pull --ff-only` in all three; a
  version or dependency change needs `tools/bootstrap_nas_env.pbs` again.
- Production jobs never install anything; `validate_env` runs in every preflight.
  Chain dependent jobs on `afterany` and let `validate_env` refuse. Run a new job path
  end-to-end on the CPU before submitting it (notes "Do not retry").
- Submit with `python tools/submit.py runs/<case>/<name>.yaml [--resume] [--dry-run]`
  from the activated env: it validates the spec, builds the qsub line (`gpu_long`, one
  GH200 node, the allocation in `$PBS_GROUP`, a walltime 2 h past the spec's governor)
  and records the submission under `<out dir>/jobs/`. The bootstrap, the kernel test and
  the GPU checks (`tools/check_nas.pbs -v CHECK=e2e`, the end-to-end ladder) go through
  `qsub -W "group_list=${PBS_GROUP:?}" [-v ...] tools/<script>.pbs`. No allocation,
  account or user path in a spec or a tracked script. `tools/run_nas.pbs` is
  generic: it exports the spec's `runtime` block through `run_setup`, builds a stale
  ffi kernel (failing loudly), runs the preflight, then the run. Never `qsub` the job
  script by hand and never pass a run setting through `qsub -v`.
- Command buffers and every other XLA flag are the spec's `runtime.xla_flags` (off is
  the empty `--xla_gpu_enable_command_buffer=`, not an absent flag).
- Never `module load cuda` in a JAX process on the node.
- Never resume a checkpoint on changed physics or a changed target.
- Recipes (queues, login, k-table fetch, env, profiling): notes "HPC (NAS GH200) and
  daw". The next run's launch command: notes "Current state".

## Rules

### Fail fast

- No behavior-changing fallback paths. Keep the refusals: missing H2-He CIA, an unset
  planet identity (`rp_cm`, `rstar_cm`, `vulcan_cfg_name`; `validate_config`), a
  real-data run with no overlapping bins, RESUME with no checkpoint.
- The chemistry block `[lnZ, c_o, lnKzz, <extras>]` is positional (`theta[0:n_chem]`,
  then the T-P entries; the extras are `config_schema.chem_extras`: `s_o` with
  `infer_s_o`, `kzz_slope` with `kzz_mode="powerlaw"`, in that order) and the first
  three are always inferred, by construction (`specs_from_config` has no switch for
  them): never add a drop path. Read the block through `pipe.n_chem` / `pipe.n_chem_tp`,
  never a literal 3.
- `set_observations` validates at the API boundary (`pipeline.validate_observations`):
  mask invalid bins before injection.

### Sampler and policy

- `run_nautilus` on the stock nautilus-sampler pinned in pyproject is the only sampler.
  `validate_env` refuses the former fork.
- Warm starts are the default (`sampler.warm_starts`): each solve continues from the
  nearest certified column under the cold `count_max`, so the likelihood and the
  rejected set depend on evaluation order and a warm run's evidence is approximate. An
  evidence claim uses `warm_starts: false`.
- The chemistry policy is the spec's: one program per draw, at the spec's `rtol`
  (`cfg_overrides`) within `count_max` accepted steps, and only its VULCAN certificate
  keeps a draw. Do not bring back a second program (the presolve chain) without a
  full-run A/B (notes "Design records", the chemistry program).
- The certifying cap is `count_max`. A draw that does not certify within it is rejected
  (log L = -inf), never extended. Do not raise the cap without a measured reason.
- The run target is every Config key outside `provenance.TARGET_FREE_KEYS`, the solver
  env choices (`VULCAN3_SOLVER`, `VULCAN3_CHEM_RHS`, `VULCAN3_BT_BUFFERS`), the start
  policy and the nautilus settings. Changing any needs a new spec file (a new run
  name); `run_nautilus` refuses a resume across it.
- After a config or physics change, synthetic observations, `data/*.npz` caches and
  every checkpoint are stale: regenerate, never resume.
- Kept names, do not rename: the `runs/w39b_smc_retrieval` directory and
  `smc_rt_chunk` (the NAS tree and archived runs' `config.json` use them).

### Convergence and the chemistry map

- Convergence is VULCAN's own: `yconv_cri` at the schema default; `slope_cri`,
  `yconv_min`, `flux_cri` and `conv_step` from vulcan3's `W39b.yaml`. Never relax the
  gate (pressure masks, cap-exit acceptance, error-norm masks). Quote the realised
  longdy, not the criterion.
- Keep `dt_max` as the spec sets it. Ros2 conditioning grows with dt, so measure any
  "accuracy-neutral" solver claim near the cap, not at small dt.
- One certificate for every likelihood entry point: the solver's `conv_normal`, with no
  accept-count clause (a solve can certify at cap or cap + 1; every cap exit is
  uncertified). Every batched evaluator rejects on it; a new sampler adapter goes
  through one of them; `generate_observations` refuses an uncertified synthetic truth
  (`native_depth_aux`'s bit, `pipe.binned_ok_jit`). `observed_depth_from_y_jit` is the
  only ungated spectrum (plots, diagnostics). Pinned by `tests/test_cold_reject.py` and
  `tests/test_run_nautilus.py`.
- A new rejection rule (e.g. aborting doomed lanes) is the maintainer's decision (notes
  "Ideas not adopted").
- Cold solves start from the network's Gibbs equilibrium at the proposal's own T-P and
  composition, the engine's only cold start. Stage 2's ConvDiag is the certificate;
  stage 1 (baseline composition) is not gated.
- Composition knobs act on a T-converged column (the two-stage solve), never on the
  cold equilibrium start (notes "Design records").
- Chemistry stays f64.
- Never clip the T-P profile: a draw whose profile leaves the window on the ART grid is
  rejected (`pipeline.tp_valid` / `sample_prior_u`; window: `tp_profile.T_EDGE_INSET_K`
  inside the opacity tables' range).
- The elemental map is the only abundance map; do not add another. Keep `prior_c_o`'s
  upper edge below the b_z bound (`build_retrieval_forward` raises).
- No minichem and no self-made or pruned network; any published upstream network may
  be used (notes "Design records").

### Lanes and batching

- Every chemistry batch runs on the lane queue at `cold_lanes` lanes (`validate_config`
  refuses 0); under `ffi`, `run_nautilus` runs `cold_lanes` x `FFI_BLOCKS_PER_SM` lanes.
- One route: every batch, cold or warm, goes through
  `retrieval_forward._chem_batch_route` (never a per-row `vmap` of solves).
- A refilled draw moves at the convergence scale, so a lane-count change is a target
  change: regenerate, never resume.
- A run's solver is its spec's `runtime.solver` (`run_setup.export_runtime`, before
  the first jax import; an environment value that disagrees is refused). The analysis
  tools (`nautilus_report`, the forward checks) call `config_schema.choose_solver`
  instead: an explicit `VULCAN3_SOLVER` wins, else `ffi` on compute capability 9.0 with
  the CUDA library built, else `fast`. Name the metric before claiming a GPU speed-up
  (notes "Design records").

### Opacity and the RT

- `opacity_mode="exomolop"` (correlated-k) is the only value the engine accepts. Never
  re-open sampled line-by-line opacity or the Mie deck.
- Drop an RT molecule only with a recorded `validation/opacity_leave_one_out.py
  --states` bound below the gate (notes "Physics assumptions and limitations").
- Change `art_nlayer` or the model top only when the ladder gates pass (notes "Design
  records").
- A finite-difference check disagrees with AD in directions that reshuffle the
  g-ordering in `ckd.overlap`: not a bug; do not "fix" it. Compare two AD routes by
  `max|dg| / max|g|`, never componentwise.

### Evidence semantics and condensation

- `logZ_box` is the zero-filled box evidence (every rejected draw has log L = -inf), so
  it depends on `count_max` and the rtol through what is rejected: cross-model Bayes
  factors only at matched solver settings, with the rejection fractions shown. No
  evidence field "corrected" for the T-P window (notes "Do not retry").
- nautilus reports no logZ error (the certificate records null): quote logZ with the
  spread over >= 3 independent seeds (`SEED=<n>` in the PBS).
- `use_condense=True` works for T-varying forward models through the engine
  (`vulcan_forward.vulcan_chem.build_chem_model`). A retrieval with condensation is
  refused twice (`validate_config`, `retrieval_forward._refuse_condense_inference`)
  because none has been validated; the only opt-in is `allow_condense_inference=True`
  after validating it on the case. Never remove either gate.
- Conden-on does not reduce to conden-off where nothing condenses: enable it only where
  a species condenses; production configs keep it off. The recipe is pinned by
  `tests/test_condensation_live_tp.py`, which starts from the cfg's const_mix column.

### Reporting

- Report a finished run from `nautilus_certificate.json` (written when a run finishes;
  a run stopped by the governor has none yet) and `nautilus_report`'s figures. The
  certificate has no pass/fail verdict: read its caveats. Quote the paper's quantities
  from its `derived` block (metallicity as O/H and as total metals, C/O, S/O, the Kzz
  factor with the floor weight and the x10 truncation, T at the deep level), never
  re-derived by hand; a T1bar run is reported against its accepted joint prior (notes
  "Physics assumptions and limitations"). Forward checks at a run's draws:
  `validation/posterior_forward_check.py`.
- Quote posterior numbers from the weighted points (`theta`, `log_w`) in
  `nautilus_posterior.npz`, never from its equal-weight `samples`.
- Before publishing a number, read notes "Physics assumptions and limitations" and
  report beside the posterior what it lists. A finished GPU run's numbers go into
  notes "Runs" before any of them is reported.
- The GPU validation set (`validation/elemental_audit.py`, `resolution_ladder.py`,
  `top_pressure_ladder.py`) runs at the maintainer's call; nothing reads its artifacts
  automatically.

### Cross-repo and search

- Do not edit the sibling repos from a task here: solver changes go through
  VULCAN-JAX, engine changes through vulcan-forward. Cross-repo version floors: this
  pyproject (checked by `validate_env`); HPC pins: `requirements-hpc.txt`.
- This repo's `grep` is ugrep honoring `.gitignore`: sweep with `--no-ignore-files`.

## Where things live

- A run is `runs/<case>/<name>.yaml` (`run_setup.load_spec`; the format is that
  module's docstring and `wasp39_test1.yaml`'s comments). `runs/w39b_smc_retrieval/`
  holds the WASP-39b specs; `tools/submit.py` and `tools/run_nas.pbs` are the NAS launch.
- `src/retrieval_framework/`: `config_schema` (Config, `validate_config`), `run_setup`
  (`load_spec`, `export_runtime`), `pipeline`
  (evaluators, T-P window), `retrieval_forward` (chemistry + RT, lane route),
  `run_nautilus` (sampler, anchors, resume refusal), `provenance` (target digest),
  `nautilus_report` (certificate, figures), `validate_env`.
- notes: "Current state" (versions, pins, production setup, next run), "Runs",
  "Design records", "HPC (NAS GH200) and daw", "Superseded designs".
