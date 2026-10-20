# CLAUDE.md — vulcan-retrieval

## Docs

Four root docs: `README.md` (hand-written by the maintainer; never edit it unasked),
`notes.md`, `todo.md` and this file. The last three are tracked and public: write them
for public reading (no personal names except cited authors, no account details, no chat
quotes; a recorded decision is "the maintainer decided"). No other `.md` file in this
tree. Doc policy, no-bloat and test-tier rules follow the project-root `CLAUDE.md`.
Read `todo.md` before starting work and open or close items in the same pass. The
measured reason behind each rule here is in notes.md, "Evidence behind the rules";
refer to notes sections by name, never by number.

## Fail fast and loud

No behavior-changing fallback paths. Missing H2-He CIA raises; planet identity
(`rp_cm`, `rstar_cm`, `vulcan_cfg_name`) must be set or `validate_config` raises; a
real-data run with no overlapping bins raises; RESUME with no checkpoint raises. The
chemistry block `[lnZ, c_o, lnKzz]` is positional (`theta[0:3]`, then the T-P entries):
`validate_config` raises on any `infer_lnZ/infer_c_o/infer_lnKzz=False` and `pipeline`
re-checks the prefix; keep all three inferred and never add a drop path.
`set_observations` validates at the API boundary (`pipeline.validate_observations`):
mask invalid bins before injection.

## Production path

- The only sampler is `run_nautilus` on stock nautilus-sampler 1.0.6; `validate_env`
  refuses the former fork. `run_nas_w39b.pbs` refuses any other `SAMPLER` and the
  removed `PROBE_MEMORY` / `CALIBRATE_*` knobs.
- Warm starts are the default (`NAUTILUS_WARM=1`): each solve continues from the
  nearest certified column, so the likelihood and which draws are rejected depend on
  evaluation order; a warm run's evidence is approximate. An evidence claim uses
  `NAUTILUS_WARM=0`.
- The chemistry policy comes from the override file, not the gpu preset:
  `chem_retry` + `chem_presolve` (`overrides/w39b_baseline_v4_nz100.json`). The
  presolve runs at `count_max` (3000) and rtol 0.2 and is only a start; every in-window
  draw is then solved at `chem_retry_rtol` (0.05) within `chem_retry_count_max`, and
  only that solve's VULCAN certificate keeps it.
- The certifying solve's cap is 5000 accepted steps: `gpu_config`'s `count_max`, or
  `chem_retry_count_max` under the retry/presolve policy. A draw that does not certify
  within it is rejected (log L = -inf), never extended. Do not raise it without a
  measured reason.
- Every Config key outside `provenance.TARGET_FREE_KEYS` (e.g. `walltime_seconds`), the
  solver env choices (`VULCAN3_SOLVER`, `VULCAN3_CHEM_RHS`, `VULCAN3_BT_FACTOR`,
  `VULCAN3_BT_BUFFERS`), the start policy and the nautilus settings define the target: a
  change needs a fresh `SMC_RETRIEVAL_OUT_DIR`; `run_nautilus` refuses a resume across it.
- Report a finished run from `nautilus_certificate.json` (written when a run finishes;
  a run stopped by the governor has none until it finishes) and the figures of
  `python -m retrieval_framework.nautilus_report <out_dir> [--spectra N]` (the PBS does
  not run it). The certificate has no pass/fail verdict: read its caveats. Quote posterior numbers from the weighted points in
  `nautilus_posterior.npz`, not `nautilus_summary.json`'s medians (equal-weight draws).
- Kept names, do not rename: the env vars `SMC_RETRIEVAL_*` / `SMC_RESUME`, the
  `runs/w39b_smc_retrieval` directory and `smc_rt_chunk` (the NAS launch command and
  archived runs' `config.json` use them). `warm_count_max` is an engine key that
  `validate_config` keeps <= `count_max`; run_nautilus's warm starts run under
  `count_max`.

## Solver and numerics (do not change without a measured reason)

- Convergence is VULCAN's own: `yconv_cri = 0.01` (schema default),
  `slope_cri`/`yconv_min`/`flux_cri` inherit `vulcan_cfg_W39b`, `conv_step` 500. Never
  relax the gate (pressure masks, cap-exit acceptance, error-norm masks). Quote the
  realised longdy, not the criterion.
- `dt_max = 1e11 s` (`case.py`). The Ros2 matrix conditioning grows with dt, so measure
  any "accuracy-neutral" solver claim near the cap, not at small dt.
- Cold solves start from the network's Gibbs equilibrium at the proposal's own T-P and
  composition (the engine's only cold start; a `cold_seed` override raises). Stage 2's
  ConvDiag is the certificate; stage 1 (baseline composition) is not gated.
- Composition knobs act on a T-converged column (the two-stage solve), never on the cold
  equilibrium init.
- One certificate for every likelihood entry point: `retrieval_forward.native_depth_aux`
  returns canonical convergence AND `accept_count < cap`; every evaluator and the scalar
  `log_likelihood_u` reject on it, and a new sampler adapter goes through one of them.
  `observed_depth_model_jit` and `observed_depth_from_y_jit` are the only ungated
  spectra (plots and diagnostics). Pinned by `tests/test_cold_reject.py`.
- Chemistry stays f64.
- No T-P clipping: a draw whose profile leaves [320, 2980] K on the ART grid is rejected
  (`pipeline.tp_valid` / `sample_prior_u`).
- Elemental is the one abundance map (no `abundance_mode` or `reanchor_atom_ini`; an
  override naming one raises). Keep `prior_c_o`'s upper edge below the b_z bound
  (`build_retrieval_forward` raises).
- Solver: `run_nas_w39b.pbs` pins `VULCAN3_SOLVER=ffi` and builds a stale kernel at job
  start, failing loudly if the build fails. Off the PBS, `make_config` calls
  `config_schema.choose_solver`: an explicit `VULCAN3_SOLVER` wins, else `ffi` on compute
  capability 9.0 with the CUDA library built, else `fast`.
- Lanes: `cold_lanes > 0` runs every chemistry batch on the lane queue
  (`retrieval_forward._chem_batch_route`); `run_nautilus` runs `cold_lanes` x
  `FFI_BLOCKS_PER_SM` (3) lanes under `ffi` (the overrides set 264 or 176: 792 or 528
  lanes). One route per config: every batch, cold or warm, goes through
  `_chem_batch_route` (never a per-row `vmap` of solves), and a narrow call (a padded
  last chunk, a few model spectra) must not switch to the lockstep batch. A refilled
  draw moves at the convergence scale, so a lane-count change is a config change:
  regenerate, never resume. On daw (`fast`, so the lanes are `cold_lanes` as given) a
  batched W39b run uses `cold_lanes` 284 via `SMC_RETRIEVAL_OVERRIDES`, with
  `XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` and a fresh
  out dir (notes "HPC (NAS GH200) and daw").

## Opacity and the RT

- `opacity_mode="exomolop"` (correlated-k) is the only value the engine accepts. Never
  re-open sampled line-by-line opacity or the Mie deck.
- A finite-difference check disagrees with AD at ~1e-2 in directions that reshuffle the
  g-ordering in `ckd.overlap`: not a bug; do not "fix" it. Compare two AD routes by
  `max|dg| / max|g|`, never componentwise.

## Evidence semantics

- nautilus's `logZ_box` is the zero-filled box evidence (every rejected draw has
  log L = -inf), so it depends on `count_max` and the rtol through what is rejected:
  cross-model Bayes factors only at matched solver settings, with the rejection
  fractions shown. Do not add an evidence field "corrected" for the T-P window.
- nautilus 1.0.6 reports no logZ error (the certificate records null): quote logZ with
  the spread over >= 3 independent seeds (`SEED=<n>` in the PBS).

## Condensation

`use_condense=True` works for T-varying forward models (`make_conden_spec` once,
per-proposal rebuild). Inference with condensation is refused twice (`validate_config`
and `retrieval_forward._refuse_condense_inference`, the resolved config) because no
retrieval with condensation has been validated; the only opt-in is
`allow_condense_inference=True` after validating it on the case (notes "Evidence behind
the rules"). Never remove either gate. Conden-on does
not reduce to conden-off where nothing condenses; enable it only where a species
condenses. Production configs keep it off. The recipe is pinned by
`tests/test_condensation_live_tp.py`, which starts from the cfg's const_mix column.

## After a config or physics change

Synthetic observations, `data/*.npz` caches and every checkpoint go stale: regenerate,
never resume across the change. Before a production run: the full test tier. The GPU
validation set (`validation/elemental_audit.py`, `resolution_ladder.py`,
`top_pressure_ladder.py`) is the maintainer's call; nothing reads its artifacts
automatically.

## Running on NAS

- Submit from `cghfe01`/`cghfe02`, queue `gpu_long`; never `-q queue@server`. Short jobs
  go to `gpu_long` with a short walltime.
- `PROJECT_ROOT` holds three trees: `vulcan-retrieval`, `VULCAN-JAX` (the clone name is
  load-bearing) and `vulcan-forward`. Update with `git pull --ff-only` in all three; a
  version or dependency change needs `tools/bootstrap_nas_env.pbs` again. Jobs never
  install anything; `validate_env` runs in every preflight.
- Never resume a checkpoint on changed physics or a changed target.
- Data is not in git. `overrides/prod_*.json` is local-only.
- Command buffers are on by default (`XLA_CMDBUF=1`); off is `XLA_CMDBUF=0` (the empty
  flag). Never pass comma lists through `qsub -v`.
- Never `module load cuda` in a JAX process on the node.
- Recipes (queues, login, k-table fetch, env): notes.md, "HPC (NAS GH200) and daw".

## Layout, install, tests

- Dist `vulcan-retrieval`, import `retrieval_framework`, src layout; the engine is the
  sibling `vulcan-forward` distribution. Cases: `runs/<case>/case.py`. Data inputs in the
  repo `data/` (`$VULCAN_PROJECT_ROOT` = the directory holding this repo).
- Install: member of the root uv workspace (`uv sync` at the project root); standalone,
  `pip install --no-deps -e .` (vulcan3 and vulcan-forward are TestPyPI-only), plus
  `'.[plot]'` for plotting. Import order is guard-enforced: `vulcan_forward.vulcan_chem`
  raises if exojax was imported first.
- Run from the repo root: `python -m retrieval_framework.run_nautilus
  runs/w39b_smc_retrieval` (output in `<out_dir>_nautilus`), then
  `python -m retrieval_framework.nautilus_report <that dir> [--spectra N]` (N cold
  re-solved draws for the spectrum figure, default 32; 0 skips the forward model).
- Tests, two tiers. Normal (everyday, what CI runs): `JAX_PLATFORMS=cpu python -m pytest
  tests -n 8` (`-n` needs pytest-xdist, which the root workspace env has; CI runs
  serially); pyproject's addopts deselect `slow`. Full, before a production run or
  after a change to the chemistry/RT path or a likelihood gate: add `-m "slow or not
  slow"`. A single slow file or node id needs
  `-m slow`. Mark a test slow if it builds a real pipeline (`build_pipeline` on a real
  config); claims that share one heavy build stay in one test (xdist builds a fixture
  once per worker). The production forward build is opt-in:
  `RUN_PRODUCTION_PREFLIGHT=1 python -m pytest tests/test_set_observations.py
  tests/test_group_noise.py -m "slow or not slow"`.
- Do not edit the sibling repos from a task here; solver and engine changes go through
  their own repos.
- This repo's `grep` is ugrep honoring `.gitignore`: sweep with `--no-ignore-files`.
