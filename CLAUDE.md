# CLAUDE.md — vulcan-retrieval (dist `vulcan-retrieval`, import `retrieval_framework`)

The WASP-39b retrieval: nautilus nested sampling over a forward model that solves VULCAN
3 photochemistry per draw and computes the transmission spectrum with ExoJAX; production
on the NAS GH200s, tests and small runs on daw. A leaf of vulcan3 <- vulcan-forward <-
{vulcan-retrieval, jwst-transit-authority}: solver and engine live upstream; this repo
holds the likelihood, the sampler driver, the cases and the job scripts.

## Docs

`README.md` is the maintainer's: never edit it unasked. `notes.md` is what is true and
why, by section; the measured reason behind each rule below is in notes "Evidence behind
the rules" unless another section is named. `todo.md` is open work: read it first and
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
- Data is not in git: inputs live in the repo `data/` (`$VULCAN_PROJECT_ROOT` = the
  directory holding this repo). An `overrides/prod_*.json` file is local-only.
- Run from the repo root: `python -m retrieval_framework.run_nautilus
  runs/w39b_smc_retrieval` (output in `<out_dir>_nautilus`), then `python -m
  retrieval_framework.nautilus_report <that dir> [--spectra N]` for the certificate and
  figures (`--spectra 0` skips the forward model). The PBS does not run the report.
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
  the `fast` solver, so the lanes are `cold_lanes` as given: the production override
  file plus `SMC_RETRIEVAL_OVERRIDES='{"cold_lanes": 284}'`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` and a fresh
  `SMC_RETRIEVAL_OUT_DIR` (notes "HPC (NAS GH200) and daw").

### On NAS

- Submit from `cghfe01`/`cghfe02` to `gpu_long`, short jobs too (short walltime);
  never `-q queue@server`.
- `PROJECT_ROOT` holds `vulcan-retrieval`, `VULCAN-JAX` (the clone name is
  load-bearing) and `vulcan-forward`. Update with `git pull --ff-only` in all three; a
  version or dependency change needs `tools/bootstrap_nas_env.pbs` again.
- Production jobs never install anything; `validate_env` runs in every preflight.
  Chain dependent jobs on `afterany` and let `validate_env` refuse. Run a new job path
  end-to-end on the CPU before submitting it (notes "Do not retry").
- `run_nas_w39b.pbs` pins `VULCAN3_SOLVER=ffi` and builds a stale kernel at job start,
  failing loudly if the build fails.
- Command buffers are on by default (`XLA_CMDBUF=1`); off is `XLA_CMDBUF=0` (the empty
  flag). Never pass a comma list through `qsub -v`.
- Never `module load cuda` in a JAX process on the node.
- Never resume a checkpoint on changed physics or a changed target.
- Recipes (queues, login, k-table fetch, env, profiling): notes "HPC (NAS GH200) and
  daw". The next run's launch command: notes "Current state".

## Rules

### Fail fast

- No behavior-changing fallback paths. Keep the refusals: missing H2-He CIA, an unset
  planet identity (`rp_cm`, `rstar_cm`, `vulcan_cfg_name`; `validate_config`), a
  real-data run with no overlapping bins, RESUME with no checkpoint.
- The chemistry block `[lnZ, c_o, lnKzz]` is positional (`theta[0:3]`, then the T-P
  entries): keep all three inferred, never add a drop path (`validate_config` and
  `pipeline` both refuse).
- `set_observations` validates at the API boundary (`pipeline.validate_observations`):
  mask invalid bins before injection.

### Sampler and policy

- `run_nautilus` on the stock nautilus-sampler pinned in pyproject is the only sampler.
  `validate_env` refuses the former fork; `run_nas_w39b.pbs` refuses any other
  `SAMPLER` and the removed `PROBE_MEMORY` / `CALIBRATE_*` knobs.
- Warm starts are the default (`NAUTILUS_WARM=1`): each solve continues from the nearest
  certified column under the cold `count_max`, so the likelihood and the rejected set
  depend on evaluation order and a warm run's evidence is approximate. An evidence
  claim uses `NAUTILUS_WARM=0`.
- The chemistry policy comes from the override file, not the gpu preset. Set
  `chem_retry` and `chem_presolve` together (the only two-program mode): the presolve,
  at the base `rtol` and `count_max`, is only a start; every in-window draw is then
  solved at `chem_retry_rtol` within `chem_retry_count_max`, and only that solve's
  VULCAN certificate keeps it (notes "Low-Kzz hole").
- The certifying cap is `chem_retry_count_max` under the presolve chain, else
  `gpu_config`'s `count_max`. A draw that does not certify within it is rejected
  (log L = -inf), never extended. Do not raise the cap without a measured reason.
- The run target is every Config key outside `provenance.TARGET_FREE_KEYS`, the solver
  env choices (`VULCAN3_SOLVER`, `VULCAN3_CHEM_RHS`, `VULCAN3_BT_FACTOR`,
  `VULCAN3_BT_BUFFERS`), the start policy and the nautilus settings. Changing any needs
  a fresh `SMC_RETRIEVAL_OUT_DIR`; `run_nautilus` refuses a resume across it.
- After a config or physics change, synthetic observations, `data/*.npz` caches and
  every checkpoint are stale: regenerate, never resume.
- Kept names, do not rename: `SMC_RETRIEVAL_*` / `SMC_RESUME`, the
  `runs/w39b_smc_retrieval` directory and `smc_rt_chunk` (the NAS launch command and
  archived runs' `config.json` use them).

### Convergence and the chemistry map

- Convergence is VULCAN's own: `yconv_cri` at the schema default; `slope_cri`,
  `yconv_min`, `flux_cri` and `conv_step` from vulcan3's `W39b.yaml`. Never relax the
  gate (pressure masks, cap-exit acceptance, error-norm masks). Quote the realised
  longdy, not the criterion.
- Keep `dt_max` as `case.py` sets it. Ros2 conditioning grows with dt, so measure any
  "accuracy-neutral" solver claim near the cap, not at small dt.
- One certificate for every likelihood entry point: the solver's `conv_normal`, with no
  accept-count clause (a solve can certify at cap or cap + 1; every cap exit is
  uncertified). Every batched evaluator rejects on it; a new sampler adapter goes
  through one of them; `generate_observations` refuses an uncertified synthetic truth
  (`native_depth_aux`'s bit, `pipe.binned_ok_jit`). `observed_depth_model_jit` and
  `observed_depth_from_y_jit` are the only ungated spectra (plots, diagnostics). Pinned
  by `tests/test_cold_reject.py` and `tests/test_run_nautilus.py`.
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

- `cold_lanes > 0` runs every chemistry batch on the lane queue; under `ffi`,
  `run_nautilus` runs `cold_lanes` x `FFI_BLOCKS_PER_SM` lanes.
- One route per config: every batch, cold or warm, goes through
  `retrieval_forward._chem_batch_route` (never a per-row `vmap` of solves), and a narrow
  call (a padded last chunk, a few model spectra) must not switch to the lockstep batch.
- A refilled draw moves at the convergence scale, so a lane-count change is a target
  change: regenerate, never resume.
- Off the PBS, `make_config` calls `config_schema.choose_solver`: an explicit
  `VULCAN3_SOLVER` wins, else `ffi` on compute capability 9.0 with the CUDA library
  built, else `fast`. Name the metric before claiming a GPU speed-up (notes "Design
  records").

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
  certificate has no pass/fail verdict: read its caveats.
- Quote posterior numbers from the weighted points in `nautilus_posterior.npz`, never
  `nautilus_summary.json`'s medians (equal-weight draws).
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

- A case is `runs/<case>/case.py` (presets, priors). `runs/w39b_smc_retrieval/` holds
  the WASP-39b case, its `overrides/` and the NAS job. Override resolution:
  `run_setup.make_config`.
- `src/retrieval_framework/`: `config_schema` (Config, `validate_config`), `pipeline`
  (evaluators, T-P window), `retrieval_forward` (chemistry + RT, lane route),
  `run_nautilus` (sampler, anchors, resume refusal), `provenance` (target digest),
  `nautilus_report` (certificate, figures), `validate_env`.
- notes: "Current state" (versions, pins, production setup, next run), "Runs",
  "Design records", "HPC (NAS GH200) and daw", "Superseded designs".
