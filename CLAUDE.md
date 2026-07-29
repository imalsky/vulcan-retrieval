# CLAUDE.md — vulcan-retrieval

Standing operational rules for this repo. Read before touching the retrieval or
running anything on the supercomputer. **The reasoning, dated post-mortems, and
measured numbers behind every rule below live in `notes.md`** (the dev log /
decision record) — this file is the current contract, `notes.md` is the "why".

The working documents this file cites are deliberately untracked and exist
only in a local checkout: `notes.md`, `docs/failed_approaches.md`,
`docs/job65815_badgrad_investigation.md`, and the sibling
`VULCAN-JAX/docs/vulcan_jax_notes.md`. Every pointer to them below refers to
your own disk, not to anything a fresh clone will contain. The published set a
clone does carry is `docs/forward_model.md`, `gradients_and_sampler.md`,
`wasp39b_case.md`, `limitations.md`, and `validation.md`.

## Fail fast and loud (standing rule, all sibling repos)

No behavior-changing fallback paths, ever. Missing H2-He CIA raises; planet
identity (rp_cm/rstar_cm/vulcan_cfg_name) must be set or `validate_config`
raises; a real-data run with no overlapping bins raises; RESUME with no
checkpoint raises; non-finite gradients surface as `n_bad` and the host raises.
The chemistry block `[lnZ, c_o, lnKzz]` is **load-bearing and positional** —
`pipeline.py`/`retrieval_forward.py`/`forward.vulcan_chem` unpack the parameter
vector by fixed index (`theta[0:3]`=chem, `theta[3:3+n_tp]`=T-P), so
`validate_config` **raises** on `infer_lnZ/infer_c_o/infer_lnKzz=False` (dropping
one shifts every later index and silently mislabels the posterior); `pipeline`
re-asserts the exact `[lnZ,c_o,lnKzz]+n_tp` prefix as a backstop. Keep all three
inferred (use a tight prior if you want one effectively fixed); never re-add an
independent drop path.
`set_observations` validates at the API boundary
(`pipeline.validate_observations`): raises on any non-finite depth or σ ≤ 0 /
non-finite; mask invalid bins before injection. Prefer a loud error over a
degraded result.

## Solver / numerics defaults (do not change without a measured reason)

- **`count_max = 5000`, always.** Set in `runs/w39b_smc_retrieval/case.py::gpu_config`;
  every override falls back to it. A solve that doesn't converge in 5k accepted
  steps is a **failed draw** — rejected at init, cloud oversampled to compensate.
  Do not raise it.
- **`warm_count_max = 1500`** (schema default). Warm MALA mutation solves run a
  twin runner capped here; proposals converging in (1500, 5000] become MH
  rejections. Cold / two-stage / init-phase-2 solves keep the full `count_max`.
  `warm_count_max > count_max` raises. (conv_step=500 certification window
  dominates the warm floor — see notes.md.)
- **`dt_max = 1e11 s`** (`Config.dt_max`, first-class, in `case.py::_W39B`).
  A step-size control that catches adaptive-Ros2 dt ballooning; it PRESERVES the
  longdy-defined steady state (truth bit-identical). Not a convergence criterion.
  **Conditioning note (measured 2026-07-27):** the matrix Ros2 factorizes is
  `I/(gamma*dt) - J`, so its condition number grows with dt. On a converged HD189
  state the blocks reach cond ~1.4e18 at dt=3.8e4 (a typical converged dt) and
  ~6.7e23 at dt=1e11, where the block-Thomas solve's own max relative residual is
  already ~5.5e+03 against float64's ~16 digits. This does not invalidate the cap
  — it is an upper bound the adaptive controller rarely reaches, and lowering it
  would change the steady state it is designed to preserve — but any claim that a
  solver change is "accuracy-neutral" MUST be measured at dt near this cap, not at
  small dt. See `VULCAN-JAX/docs/vulcan_jax_notes.md` (rejected inverse-carry
  optimization) for the failure this caught.
- **Convergence = VULCAN-master canonical.** `yconv_cri = 0.01` (schema default);
  `slope_cri`/`yconv_min`/`flux_cri` inherit `vulcan_cfg_W39b` — do not override.
  conv_step stays 500 (300 was probed and rejected — it certifies a less-converged
  state).
- **No T-P clipping — reject and redraw.** A draw whose T-P leaves [300, 3000] K
  on the ART grid is rejected (`pipeline.tp_valid` / `sample_prior_u`), never
  clipped. The prior rejection sampler raises if the T-P prior is mostly
  out-of-window.
- **Chemistry stays f64** (VULCAN-JAX numerical-hygiene rule; rate constants span
  ~50 dex). fp32 considered and rejected.
- **abundance_mode = "elemental"** (schema default): lnZ/c_o are exact
  conserved-ratio directions; per-proposal atmosphere rebuild; H2-He CIA required
  in every RT call (`vmr_he`).

## RT resolution — memory safety (this keeps biting)

**Keep `nu_pts ≈ 1652` (R~1000) for the production band. RT-vjp gradient memory
scales with the absolute `nu_pts`. NEVER raise `nu_pts` without `PROBE_MEMORY=1`
first.** Enforced, not just advised: `config_schema.Config.nu_pts` default is
1652; `validate_config` warns loudly above 2500. Run `PROBE_MEMORY=1` once before
the first production submit after ANY `nu_pts` / `smc_rt_vjp_chunk` / `N` change.
(History: `nu_pts=16500` once tried to allocate 343 GiB on a 96 GB GH200 — see
notes.md.)

## Evidence semantics (get this right in any paper/analysis)

- `logZ` — evidence under the OPERATIONAL prior (box ∩ T-P window ∩ converged,
  renormalized). Never difference across models with different support fractions.
- `logZ_box = logZ + ln(f_tp·f_c1·f_c2)` — zero-filled box evidence; cross-model
  Bayes factors ONLY at matched solver settings and with attrition shown
  negligible (report f_tp and f_conv alongside). SOLVER-DEPENDENT.
- There is NO f_tp-only evidence field (`logZ_box_physical` was retracted — it is
  mathematically invalid; toy counterexample in `tests/test_evidence_semantics.py`).

## Init / mutation handling (why draws get rejected)

- **Cold init: reject-and-cull + oversample.** `pipeline._init_state` rejects
  non-converged AND stall-certified draws (exit without the runner's canonical
  certification; counted as `n_stalled_init`) and draws `ceil(N·init_oversample)`
  (default 2.0) so the culled cloud holds exactly N healthy particles; raises only
  if < N survive. An INIT-LEVEL checkpoint is written right after `_init_state`
  (`last_step=-1`), so `RESUME=1` recovers a stage-0 death without re-paying the init.
- **Init phase 2 runs UNCAPPED** (`batch_eval_init_vg`) and evaluates
  `N + init_phase2_spare` (default 8) survivors, culling re-certification failures
  and backfilling from spares — survivors are proven-convergent, not disposable.
- **Warm mutation proposals** are rejected before their gradient is trusted:
  `_make_batch_eval` gates `L→-inf` at `accept_count >= warm_count_max` (warmcap)
  OR when the exit is not the runner's canonical certification
  (`pipeline._proposal_converged` on ConvDiag.conv_normal; the `stalled` class).
  A THIRD class -- `badgrad`: a certified, finite primal whose forward-mode
  TANGENT is non-finite (NAS jobs 65200/65789/65815; tangent divergence along
  the warm continuation, unflaggable by any primal-side predicate). Job 65815
  forensics measured it theta-DEPENDENT and posterior-concentrated (6.5% of
  certified proposals; 1.2% at Z=1-12x solar vs 11.7% at 67-99x; 11.3% at
  C/O 0.10-0.18; RISING along the ladder as the cloud drifts high-Z), so it
  is handled as a ZERO-DRIFT MALA move, NOT a rejection: the eval zeroes the
  non-finite gradient entries and the same zeroed drift enters both proposal
  densities (a valid MH kernel -- the drift enters the proposal, not the
  target; the certified likelihood decides acceptance); its DY rows are
  zeroed (that particle's next warm_extrapolate seed degrades to the plain
  carried column). The pre-65815 MH-reject-with-floored-L design suppressed
  the posterior bulk (metallicity bias) and its 5% per-sweep abort was
  statistically certain to kill a full ladder -- notes.md 2026-07-16 +
  docs/failed_approaches.md. Same handling at init phase 2 (keep survivors
  with zeroed drift; culling would bias the initial importance sample).
  Logged per sweep (badgrad=), forensics-dumped
  (`bad_grad_stage###_sweep#.npz`: indices, theta, ACC, longdy, chem-vs-RT
  attribution); expect it to TRACK the high-Z/low-C-O corner -- a broad
  theta-INDEPENDENT rate is the anomaly to investigate. warmcap/stalled must
  still stay ~0 in the late ladder. The loud raise remains as a
  SYSTEMATIC-BREAKAGE backstop: a single sweep exceeding
  ceil(`smc_tangent_bad_max_frac` x N) badgrad events (default 25%; measured
  worst physical sweep 7.6%) aborts -- that is AD breakage, not the corner
  class.
- **warm_extrapolate** is ON in the gpu preset (schema default off); seeds each
  warm solve at the first-order tangent prediction, gated PER PARTICLE: a
  proposal whose predicted column would need clipping (any cell driven <= 0)
  falls back to its plain carried column + carried refs. The unconditional
  clipped seed manufactured the badgrad tangent class, and a per-cell repair
  is measurably insufficient -- never reintroduce either
  (docs/job65815_badgrad_investigation.md SS10; failed_approaches #68).

## Running on the supercomputer (NAS)

- **Code updates: `git pull --ff-only` on the NAS front end** (both repos public;
  local changes are always committed + pushed). Two repos to pull —
  `vulcan-retrieval` and `VULCAN-JAX` (the clone target name `VULCAN-JAX` is
  load-bearing; the GitHub repo is `jax-vulcan`). Both are installed EDITABLE, so
  pulled changes are live with no reinstall. Jobs are read-only on the env; each
  preflight runs `python -m retrieval_framework.validate_env` (hard-fails on stale
  install / drifted metadata — a metadata change after a pull DOES need a
  bootstrap re-run).
- Front-end git: **`unset https_proxy http_proxy`** (direct https to github.com
  works; the proxy hostname does not resolve).
- **Data is NOT in git** — one-time seed of `data/opacity_cache/` (preflight errors
  without it) and `data/exojax_linelists/` into a fresh clone.
- **scp for data**, exactly this style (one command per dir, no backslashes):
  `scp -r -oProxyCommand='ssh imalsky@sfe6.nas.nasa.gov ssh-proxy %h' [local dir] imalsky@pfe.nas.nasa.gov:[hpc location]`
  **Never rsync, never tarballs, never wrap remote commands in `ssh nas '...'`**
  — give the transfer, then the commands to run while logged in on the node.
- `PROJECT_ROOT` on NAS is `/nobackup/imalsky/VULCAN_W39b_HPC`; the PBS preflight
  requires both trees under it.
- **Never profile a first/debug run with `nsys`** — it masks the exit code
  (returns 0 on a killed/crashed process). Add `NSYS=1` only once a run is
  known-good. `NSYS_DELAY` is in seconds.
- gpu preset: N=144, `smc_rt_vjp_chunk=12`, 6 sweeps/stage, 24 h PBS / 20 h
  governor. `CALIBRATE_ONLY=1` (~1 h) gives timing.json before committing a run.

## Condensation with a live T(P) (2026-07-13)

`use_condense=True` is supported for T-varying models: `build_chem_model`
extracts static conden metadata once (`vulcan_jax.conden.make_conden_spec`)
and `_prep` rebuilds every T/structure-dependent condensation array on-graph
per proposal (`conden.build_conden_profile` — saturation number densities,
Dg growth terms from the live Dzz, relax inputs, NH3 cold-trap argmin,
fix-species sat-mix rows), splicing them into the ProfileVars carry. The old
NotImplementedError + `_condense_validated_isothermal` hatch are GONE — never
reintroduce them. Loud build-time refusals remain for genuinely unsupported
configs: `use_moldiff=False` (Dg would be silently zero), empty or inert
`condense_sp`, and `use_sat_surfaceH2O` (bottom BC frozen at structural T at
ini). The cold-trap index / active-layer set are DISCRETE in T: jvp-vs-FD is
validated on the SMOOTH gas species and on the CONSERVED condensate reservoir
(gas + condensate), never on the per-species gas/condensate split -- that split
has no derivative at all, its centred FD scaling like 1/dT because the pin
captures at a discrete accepted step (`tests/test_condensation_live_tp.py`,
sweep in the test comment; failed_approaches #69);
production SMC/zco configs keep conden OFF, and Fisher through condensation
stays disabled in vulcan-jwst-tool. Converged condensing solves use the
upstream conden-window + fix_species pin methodology — without the pin the
steady state is transport-limited (reservoir drains on the Kzz timescale
while dt is capped at the condensation-front timescale) and will exhaust
count_max. The full certified recipe (measured 2026-07-13, see the test
file): whole-column pin (`fix_species_from_coldtrap_lev=False` — the
cold-trap argmin degenerates on isothermal columns), `mtol_conv=1e-15`
(glacial sub-femto NH3 drift gates forever at the 1e-20 default),
`conver_ignore` extended with the trace sulfur allotropes S/S2/S3/S4
(re-equilibrate against pinned S8 on >=1e15 s cold-top timescales), and
`trun_min = stop_conden_time` so certification can never fire before the
window + pin complete (else a half-rained S8 column gets certified). A cold
NO-PHOTO column additionally has no reachable longdy steady state at all
(well-mixed CO2 creeps toward equilibrium on >=1e17 s — the quench regime);
the synthetic test therefore integrates to a physical `runtime` cap (1e14 s)
per upstream practice. Photo-on production runs converge via the normal
longdy gate (WASP-107b Guillot+conden E2E verified). Two guardrails on the
conden path: (1) **inference is refused with conden ON** — `validate_config`
raises on `cfg_overrides["use_condense"]=True` with `run_inference=True` unless
`allow_condense_inference=True`, because gradient-MALA through the pinned S8
state is not reliably differentiable (jvp-vs-FD relative error ~0.91, i.e.
the tangent is about 91% wrong -- an order-unity failure, not a 9%
mismatch; and per the 2026-07-29 step sweep that 0.91 is itself one sample of
a step-size-dependent quantity, so the durable claim is order-unity, not the
number); conden runs as a
FORWARD model. This is enforced TWICE: the early `cfg_overrides` gate in
`validate_config`, plus a RESOLVED-config gate in
`retrieval_forward._refuse_condense_inference` (on `chem.conden_spec` after
`build_chem_model`) that also catches `use_condense=True` inherited from a base
config such as `Earth.yaml` (the `cfg_overrides`-only check would miss it). See
`../docs/condensation_differentiation.md`. (2) **conden-on does NOT reduce to conden-off when nothing
condenses** — the window+pin freezes the reservoirs at `stop_conden_time`, so a
too-hot / unsettled column is captured mid-transient, not at the conden-off
steady state; enable conden only where the species genuinely condenses (a
criterion-gated pin is a future refinement needing re-validation, not done).

## After any config/physics change — regeneration is mandatory

The elemental + atm-rebuild chemistry map means synthetic obs, `data/*.npz`
caches, zco/Fisher caches, jwst_tool model_cache, and ALL SMC checkpoints go
stale. Regenerate; do NOT resume a pre-change checkpoint (likelihoods re-anchor
mid-run). `forward._VERSION` busts the model_cache — bump it whenever the physics
changes. Before a production run: `PROBE_MEMORY=1`, then the smoke chain + suite,
then the GPU validation set (`validation/elemental_audit.py`, `resolution_ladder.py`,
`top_pressure_ladder.py --extend-chem`, `broadening_ab.py`) and post-run
`validate_warm` + `mala_reversibility.py`.

## Warm-vs-cold validation

The warm mutation kernel's likelihood is history-dependent at the convergence
tolerance (only approximately π_β-invariant). Validate once per production run:
`SMC_RETRIEVAL_PRESET=gpu python -m retrieval_framework.validate_warm runs/w39b_smc_retrieval`
PASS gate: max|dlogL| < 0.1 over the cloud. Since 2026-07-29 the cold re-solve
also recomputes the u-space GRADIENT (the MALA drift) and compares against the
checkpoint's carried grad_u — WARN above 0.1 relative (promote to a FAIL gate
once the first production baseline is recorded; zeroed badgrad rows read rel=1
by design). `VALIDATE_WARM_GRAD=0` restores the cheaper likelihood-only
re-solve. Quote the result (with the init reject fraction) in the paper.
Sampler-core edits additionally get the opt-in BlackJAX oracle:
`RUN_BLACKJAX_ORACLE=1 python -m pytest tests/test_smc_blackjax_oracle.py`
(blackjax is dev-only, not a runtime dep).

## Layout / entry points

- Dist `vulcan-retrieval`, import `retrieval_framework`, src layout. Sibling of
  `VULCAN-JAX/` and `vulcan-jwst-tool/` under the project root. SMC framework at
  `src/retrieval_framework/`; retrieval-side config + composer at
  `src/retrieval_framework/forward/` (`config`, `sensitivity`); the shared
  ENGINE is the `vulcan-forward` distribution (`vulcan_forward.vulcan_chem`,
  `.exojax_rt`, `.interp_map`, `.constants`, `.paths`), a sibling checkout. Cases: `runs/<case>/case.py`.
- Install editable: `pip install --no-deps -e .` (--no-deps because vulcan-jax +
  vulcan-forward are TestPyPI-only); plotting deps are the `[plot]` extra now,
  so `pip install -e '.[plot]'` for plot_smc / figure scripts. No sys.path bundle contract — `vulcan_jax` resolves via its own
  editable install. Import order is guard-enforced: `vulcan_forward.vulcan_chem` raises if exojax
  (or a conflicting vulcan_jax) was imported first.
- Run from repo root: `python -m retrieval_framework.run_smc runs/w39b_smc_retrieval`
  (also `calibrate_count_max`, `probe_memory`, `smoke_retrieval`, `plot_smc`,
  `validate_warm`). Suite: `python -m pytest tests -q`.
- `data/` = inputs (repo root; `$VULCAN_PROJECT_ROOT` = dir containing this repo);
  `output/` = generated npz caches (gitignored). Figures go to
  `../jax_paper/figures/`; never modify `../VULCAN-JAX`.
- **Historical/design-log content lives in `notes.md`** (this repo's dev diary);
  READMEs carry current usage only. **`docs/failed_approaches.md` is the
  append-only negative-results register** -- every failed / retracted /
  measured-worse approach (samplers, gradients, condensation/Route B, numerics,
  HPC incidents by NAS job number) with dates, measured numbers, and evidence
  pointers. When something is tried and rejected, add the entry there in the
  same pass; never delete entries.
