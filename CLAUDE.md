# CLAUDE.md — vulcan-retrieval (dist `vulcan-retrieval`, import `retrieval_framework`)

The WASP-39b retrieval and the sulfur survey: nautilus nested sampling over a forward model that solves VULCAN 3
photochemistry per draw and computes the transmission spectrum with ExoJAX; production on the NAS GH200s, tests and
small runs on daw. A leaf of `vulcan3 <- vulcan-forward <- {vulcan-retrieval, jwst-transit-authority}`: solver and
engine live upstream; this repo holds the likelihood, the sampler driver, the cases and the job scripts.

Generic rules (code, docs, git, reviews, jobs) are in the maintainer's global rules, and the shared policy is in the
project-root `CLAUDE.md`. This file holds only what is unique to this repo.

## Maintainer rules

Layout, Figures, Branches, Behind main and Tests are in the project-root `CLAUDE.md`, under "Maintainer rules".

## Docs

Five root docs: `oracle.md` (the maintainer's goal for this repo and what success and failure are: read it before
planning work and compare a result with it before calling work done; never edit it unasked), `README.md` (the
maintainer's; never edit it unasked), `notes.md` (the results and reasons behind the
rules below, by topic section, never a dated history; unless another section is named, the measured reason for a rule is
in notes "Evidence behind the rules"), `todo.md` (open work: read it first; open or close items in the same pass) and
this file (directions only). A decision is "the maintainer decided". Any other hand-written `.md` file goes in `docs/`.
Name a notes section by its heading. Before proposing a solver, sampler or speed change, read notes "Do not retry" and
"Ideas not adopted". Format: the `doc-upkeep` skill; policy: the project-root `CLAUDE.md`.

## Setup and commands

- Install: a member of the root uv workspace (`uv sync` at the project root). Standalone: `pip install --no-deps -e .`
  (vulcan3 and vulcan-forward are TestPyPI-only), plus `'.[plot]'` for figures. Keep it editable (`tests/conftest.py`
  refuses otherwise); after a version or dependency bump reinstall all three trees editable, or `validate_env` fails on
  stale metadata.
- Import order is guard-enforced: `vulcan_forward.vulcan_chem` raises if exojax was imported first.
- Inputs live in the repo `data/` (`$VULCAN_PROJECT_ROOT` = the directory holding this repo); only the observed spectra
  (`data/cm24_wasp39b/`) and the survey's inputs (`data/survey/<case>/`: spectra, each planet's vulcan3 config,
  atmosphere and UV files) are tracked. A case config given as a path (`vulcan_cfg_name: data/survey/w39b/W39b.yaml`)
  and its file paths resolve against the working directory: run from the repo root.
- A run is one YAML spec, `runs/<case>/<name>.yaml` (`wasp39_test1.yaml`, `smoke.yaml`, any `base:` variant);
  `runs/<case>/` holds nothing else. A spec is the whole run: `runtime` (the engine's environment), `sampler` (nautilus)
  and `config` (every `Config` field it sets), never the PBS job. Its stem is the run name, and every output goes under
  `output/<case>/<name>_nautilus/`. A new run is a new file; a variant restates only the keys it changes (one level,
  whole values). Never set a run through the environment or the command line.
- Run from the repo root: `python -m retrieval_framework.run_nautilus runs/w39b_smc_retrieval/<name>.yaml [--resume]
  [--dry-run]` (`--dry-run` prints the resolved settings without jax), then `python -m
  retrieval_framework.nautilus_report <out dir> [--spectra N]` for the certificate and figures (`--spectra 0` skips the
  forward model; the PBS does not run the report). The report's three figures are a run's only figures; draw nothing
  else beside a run unless asked.
- Tests: normal tier `python -m pytest tests -n 8` (pyproject's addopts deselect `slow`; conftest defaults
  `JAX_PLATFORMS=cpu`). Full tier: add `-m "slow or not slow"`; a single slow file or node id needs `-m slow`. The full
  tier holds the job path end to end (`run_nautilus` on the smoke case) and the production case's build: run it when the
  maintainer asks. Run the job-path test by itself on CPU cores nothing else uses (`taskset`), not inside `-n 8`: its
  job runs threaded BLAS and collapses on shared cores (notes "HPC (NAS GH200) and daw", CPU cores).
- Mark a test `slow` if it builds a real pipeline (`build_pipeline` on a real config); claims that share one heavy build
  stay in one test (xdist builds a fixture per worker).
- Lint and types: `uvx pre-commit run --all-files`; size limits: `python3 tools/check_size_limits.py`. CI runs both and
  the normal tier, with the 80 % coverage floor; lint before a push.
- On daw: tests and single chemistry runs on CPU. A batched run uses one GPU with the `fast` solver, so the lanes are
  `cold_lanes` as given: use the daw variant spec in notes "HPC (NAS GH200) and daw".

### On NAS

Run these only when the maintainer explicitly asks for NAS work; otherwise give the exact commands to paste.

- Submit from `cghfe01`/`cghfe02` to `gpu_long`, short jobs too (short walltime); never `-q queue@server`.
- `PROJECT_ROOT` holds `vulcan-retrieval`, `VULCAN-JAX` (the clone name is load-bearing) and `vulcan-forward`. Update
  with `git pull --ff-only` in all three; a version or dependency change needs `tools/bootstrap_nas_env.pbs` again.
- Submit with `python tools/submit.py runs/<case>/<name>.yaml [--resume] [--dry-run]` from the activated env: it
  validates the spec, builds the qsub line and records the submission under `<out dir>/jobs/`; the job refuses a spec
  edited after submission. The bootstrap, the kernel test and the GPU checks (`tools/check_nas.pbs -v CHECK=e2e`, the
  end-to-end ladder) go through `qsub -W "group_list=${PBS_GROUP:?}" [-v ...] tools/<script>.pbs`. Never `qsub` the job
  script by hand and never pass a run setting through `qsub -v`.
- Production jobs never install anything; `validate_env` runs in every preflight. Chain dependent jobs on `afterany` and
  let `validate_env` refuse. A new guard goes into `submit.py`, `validate_env`, CI or the run's first chunk, never into
  a step a user must remember.
- No allocation, account or user path in a spec or a tracked script. Command buffers and every other XLA flag are the
  spec's `runtime.xla_flags` (off is the empty `--xla_gpu_enable_command_buffer=`, not an absent flag). Never `module
  load cuda` in a JAX process on the node. Never resume a checkpoint on changed physics or a changed target.
- Recipes (queues, login, k-table fetch, env, profiling): notes "HPC (NAS GH200) and daw". The next run's launch
  command: notes "Current state".

## Rules

### Fail fast

- The refusal points are spec load, `submit.py`, `validate_env` and the run's first chunk. Keep the refusals: missing
  H2-He CIA, an unset planet identity (`rp_cm`, `rstar_cm`, `vulcan_cfg_name`; `validate_config`), a named data source
  with too few bins in any mode, RESUME with no checkpoint.
- A spec states every key its case reads with no default (`case_keys.required_keys`: `CASE_KEYS`, the priors and
  truths of the parameters `config_schema.active_parameters` samples, and `MODE_KEYS`), and a stated key its mode does
  not read is refused (`case_keys.refused_keys`); a variant removes a key its base states with `unset:`. A required
  Config field has no default (None), refused unset by `validate_config`; which parameters a mode samples is decided
  only in `active_parameters`. A number that belongs to a case is a spec key, never a value in the code.
  The synthetic bin grid is only for a spec that names no data source.
- The chemistry block `[lnZ, c_o, lnKzz, <extras>]` is positional (`theta[0:n_chem]`, then the T-P entries; the extras
  are `config_schema.chem_extras`: `s_o` with `infer_s_o`, `kzz_slope` with `kzz_mode="powerlaw"`, in that order), and
  the first three are always inferred: never add a drop path. Read the block through `pipe.n_chem` / `pipe.n_chem_tp`,
  never a literal 3.
- `set_observations` validates at the API boundary (`pipeline.validate_observations`): mask invalid bins before
  injection.

### Sampler and policy

- `run_nautilus` on the stock nautilus-sampler pinned in pyproject is the only sampler.
- Warm starts are the default (`sampler.warm_starts`): each solve continues from the nearest certified column, so a warm
  run's evidence is approximate. An evidence claim uses `warm_starts: false`.
- The chemistry policy is the spec's: one program per draw, at the spec's `rtol` (`cfg_overrides`) within `count_max`
  accepted steps, and only its VULCAN certificate keeps a draw. A second program (a presolve chain) needs a full-run A/B
  first (notes "Design records"); the cold re-solve of a frozen warm start is not one (the maintainer decided).
- `count_max` is the certifying cap: a draw that does not certify within it is rejected (log L = -inf), never extended.
  Do not raise the cap without a measured reason. A column frozen at `dt_min` is not a rejection: `run_nautilus` re-solves a frozen warm row cold, then checks
  each chunk's `termination_reason` and aborts on a cold freeze (notes "Design records"); `nautilus_report` and the forward
  checks abort on any freeze in the draws they re-solve.
- The run target is every Config key outside `provenance.TARGET_FREE_KEYS`, the solver env choices (`VULCAN3_SOLVER`,
  `VULCAN3_CHEM_RHS`, `VULCAN3_BT_BUFFERS`) and the rest of the runtime block, the composition table, the network and
  its thermo data and abundance preset when they resolve outside vulcan3's package, the JAX backend, both packages'
  fixed numbers (`constants.fixed_values`), the start policy and the nautilus settings. Changing any needs a new spec
  file; `run_nautilus` refuses a resume across it.
- After a config or physics change, synthetic observations, `data/*.npz` caches and every checkpoint are stale:
  regenerate, never resume.
- Kept names, do not rename: the `runs/w39b_smc_retrieval` directory and `smc_rt_chunk` (the NAS tree and archived runs'
  `config.json` use them).

### Convergence and the chemistry map

- Convergence is VULCAN's own: `yconv_cri` at the schema default; `slope_cri`, `yconv_min`, `flux_cri` and `conv_step`
  from vulcan3's `W39b.yaml`. Never relax the gate (pressure masks, cap-exit acceptance, error-norm masks). Quote the
  realised longdy, not the criterion.
- Keep `dt_max` as the spec sets it. Ros2 conditioning grows with dt, so measure any "accuracy-neutral" solver claim
  near the cap, not at small dt.
- One certificate for every likelihood entry point: the solver's `conv_normal`, with no accept-count clause (every cap
  exit is uncertified). Every batched evaluator rejects on it; a new sampler adapter goes through one of them;
  `generate_observations` refuses an uncertified synthetic truth. `observed_depth_from_y_jit` is the only ungated
  spectrum (plots, diagnostics). Pinned by `tests/test_cold_reject.py` and `tests/test_run_nautilus.py`.
- A new rejection rule (e.g. aborting doomed lanes) is the maintainer's decision.
- Cold solves start from the network's Gibbs equilibrium at the proposal's own T-P and composition, the engine's only
  cold start. Stage 2's ConvDiag is the certificate; stage 1 (baseline composition) is not gated. Composition knobs act
  on a T-converged column (the two-stage solve), never on the cold equilibrium start.
- Chemistry stays f64.
- Never clip the T-P profile: a draw whose profile leaves the window on the ART or chemistry grid is rejected
  (`pipeline.tp_valid` / `sample_prior_u`; window: `constants.T_EDGE_INSET_K` inside the opacity tables' range).
- The elemental map is the only abundance map; do not add another. Keep `prior_c_o`'s upper edge below the b_z bound
  (`build_retrieval_forward` raises).
- No minichem and no self-made or pruned network; any published upstream network may be used (notes "Design records").
- The network is `config.network` with its `atom_list`, and neither has a default: every base spec names both
  (`validate_config` refuses an unset one). They are import-frozen: every entry point that rebuilds a run calls
  `config_schema.export_network(cfg)` (the engine's `select_network`, which imports jax: after `export_runtime`) before
  the first vulcan3 import; `pipeline.build_pipeline` calls it itself, and the engine refuses a second pair in one
  process. `tests/conftest.py` sets the session's one network. A re-solve of a finished run names that run's network.

### Lanes and batching

- Every chemistry batch runs on the lane queue at `cold_lanes` lanes, the number that runs under every solver
  (`validate_config` refuses 0); `run_nautilus --dry-run` prints the lanes and the batch width.
- One route: every batch, cold or warm, goes through `RetrievalForward._chem_batch_route`, never a per-row `vmap` of
  solves.
- A refilled draw moves at the convergence scale, so a lane-count change is a target change: regenerate, never resume.
- A run's solver is its spec's `runtime.solver` (`run_setup.export_runtime`, before the first jax import; a disagreeing
  environment value is refused). The analysis tools (`nautilus_report`, the forward checks) call
  `hardware.choose_solver` instead. Name the metric before claiming a GPU speed-up.

### Opacity and the RT

- Correlated-k over the ExoMolOP tables is the only opacity path, with no key to choose it. Never re-open sampled
  line-by-line opacity or the Mie deck. Rayleigh is in every retrieval (`validate_config` refuses `use_rayleigh:
  false`); `cloud_model` is `powerlaw`, `deck` or `none` (no cloud term), and a spec states it.
- `art_pbtm_bar`, the RT bottom, has no default: every base states it above the case's chemistry bottom.
- Drop an RT molecule only with a recorded `validation/opacity_leave_one_out.py --states` bound below the gate (notes
  "Physics assumptions and limitations"). The engine refuses a band that lacks H2O, CO2, CO, CH4 or NH3 and warns for
  HCN, H2S, SO2, C2H2, OCS or SO (vulcan-forward `constants.CORE_MOLECULE_BANDS_UM`) unless the case names it in
  `omit_core_molecules`: only a synthetic or single-molecule case does.
- Change `art_nlayer` or the model top only when the ladder gates pass (`oracle.md` "The RT grid is converged"; notes
  "Design records").
- A finite-difference check disagrees with AD in directions that reshuffle the g-ordering in `ckd.overlap`: not a bug;
  do not "fix" it. Compare two AD routes by `max|dg| / max|g|`, never componentwise.

### Evidence and condensation

- `logZ_box` is the zero-filled box evidence (every rejected draw has log L = -inf), so it depends on `count_max` and
  the rtol: cross-model Bayes factors only at matched solver settings, with the rejection fractions shown. No evidence
  field "corrected" for the T-P window (notes "Do not retry").
- nautilus reports no logZ error: quote logZ with the spread over >= 3 independent seeds, one spec file per seed
  (`config.seed`).
- Condensation works in T-varying forward models through the engine (`vulcan_forward.vulcan_chem.build_chem_model`), but
  a retrieval with condensation is refused twice (`validate_config`, `retrieval_forward._refuse_condense_inference`)
  because none has been validated; the only opt-in is `allow_condense_inference=True` after validating it on the case.
  Never remove either gate. Conden-on does not reduce to conden-off where nothing condenses: enable it only where a
  species condenses; production configs keep it off (`tests/test_condensation_live_tp.py` pins the recipe).

### Reporting

- Name the case: a setting belongs to a case's spec (the WASP-39b case, a survey planet), never to "the retrieval"; say
  where it is set.
- Report a finished run from `nautilus_certificate.json` (a run stopped by the governor has none yet) and
  `nautilus_report`'s figures. The certificate has no pass/fail verdict: read its caveats. Quote the paper's quantities
  from its `derived` block, never re-derived by hand; a T1bar run is reported against its accepted joint prior (notes
  "Physics assumptions and limitations"). Forward checks at a run's draws: `validation/posterior_forward_check.py`.
- Quote posterior numbers from the weighted points (`theta`, `log_w`) in `nautilus_posterior.npz`, never its
  equal-weight `samples` (`oracle.md` "Its numbers are the posterior's own"). With `integrate_nuisance` the offsets and
  inflations are the file's `nuisance` arrays, not `theta`.
- Before publishing a number, read notes "Physics assumptions and limitations" and report beside the posterior what it
  lists. A finished GPU run's numbers go into notes "Runs" before any of them is reported.
- The GPU validation set (`validation/elemental_audit.py`, `resolution_ladder.py`, `top_pressure_ladder.py`) runs at the
  maintainer's call; nothing reads its artifacts automatically.

### Cross-repo and search

- Do not edit the sibling repos from a task here: solver changes go through VULCAN-JAX, engine changes through
  vulcan-forward. Cross-repo version floors: this pyproject (checked by `validate_env`); HPC pins:
  `requirements-hpc.txt`.
- This repo's `grep` is ugrep honoring `.gitignore`: sweep with `--no-ignore-files`.

## Where things live

- Runs: `runs/w39b_smc_retrieval/` (the WASP-39b specs) and `runs/survey/` (`survey_base.yaml` and one file per planet);
  the spec format is `run_setup`'s docstring and `wasp39_test1.yaml`'s comments. NAS launch: `tools/submit.py`,
  `tools/run_nas.pbs`.
- `docs/method.md`: the module map ("Modules"), the methods, their sources, the validation scripts' rungs and variants,
  the glossary.
- notes: "Current state" (versions, pins, production setup, next run), "Runs", "Design records", "HPC (NAS GH200) and
  daw", "Superseded designs".
