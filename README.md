# vulcan-retrieval

The planet-agnostic gradient-informed Bayesian retrieval framework: **live
differentiable VULCAN-JAX photochemistry** chained into **ExoJax** radiative
transfer, sampled by an adaptive-tempered SMC whose mutation kernel is **MALA
driven by forward-mode-jvp gradients** through the whole chemistry→spectrum
chain — the same pipeline/driver/PBS pattern as the SWAMPE WASP-43b phase-curve
retrieval (`SWAMPE_Project/MY_SWAMP/retrieval`), rebuilt for this forward model
with no BlackJAX dependency (the SMC
core is ~200 lines of pure JAX, validated against an analytic Gaussian
posterior including its evidence estimate).

Dist name `vulcan-retrieval`, import name `retrieval_framework`. The package
carries the retrieval framework, the shared differentiable forward-model engine
it is built on (`retrieval_framework.forward`, also the engine behind the
sibling vulcan-jwst-tool repo), the WASP-39b case, the sensitivity examples,
the validation scripts, and the Z vs C/O information analysis.

To our knowledge (literature review 2026-07) this is the first

1. **Bayesian retrieval with gradients through a full photochemical-kinetics forward
   model.** No published kinetics retrieval uses gradients: the FRECKLL line — the only
   prior full-kinetics retrievals (Al-Refaie et al. 2024; Bardet et al. 2025) — is
   gradient-free nested sampling at 5–10 parameters on 180 CPU cores × 24 h and
   ~874,000 CPU-hours respectively, and Khorshid et al. 2024 call retrieval with
   photochemistry computationally "impractical". Published gradient-based retrievals
   (ExoJAX HMC-NUTS) use free-chemistry forward models that run in ~ms, not a stiff
   kinetics solver.
2. **Atmospheric retrieval sampled with Sequential Monte Carlo** (adaptive-tempered
   SMC with preconditioned-MALA mutations; no published SMC atmospheric retrieval
   found, 2022–2026).
3. **Full-kinetics retrieval on a single GPU inside 24 hours** (10 parameters, one
   GH200), versus the CPU-cluster scale of all prior kinetics retrievals.

Statements are as of the 2026-07 review; the WASP-39b production run is the
demonstration case. The supporting citations and precedent map are preserved in
`notes.md` (section F).

## Install

Local development (repo checkout, conda env `vulcan`), from the repo root:

```
pip install --no-deps -e .
```

`--no-deps` because the chemistry dependency `vulcan-jax>=0.1.17` lives on
TestPyPI, not PyPI (the remaining dependencies are ordinary PyPI packages).
Consumer install from TestPyPI (latest published release; pin a version with
`vulcan-retrieval==X.Y.Z` if you need reproducibility):

```
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vulcan-retrieval
```

HPC runs pin the exact validated chemistry commit instead: see
`requirements-hpc.txt` and the deploy rules in `CLAUDE.md` (git pull for
code, one-time seed for data). Inputs (observations, opacity caches) live in
this repo's `data/` tree (`config.DATA_DIR`) and generated npz caches in
`output/` (`config.OUTPUTS`, gitignored). Both roots resolve via
`VULCAN_PROJECT_ROOT` (the directory containing this repo) or are inferred
from an editable checkout; `src/retrieval_framework/forward/config.py` raises
loudly when neither resolves. Provenance and regeneration policy:
`data/notes.md`.

## Package layout

| path | contents |
|------|----------|
| `src/retrieval_framework/` | the planet-agnostic framework: `config_schema.py`, `observations.py`, `tp_profile.py`, `retrieval_forward.py`, `pipeline.py`, the `run_smc.py` driver, `plot_smc.py`, and the `calibrate_count_max` / `probe_memory` / `smoke_retrieval` / `validate_warm` tools |
| `src/retrieval_framework/forward/` | the shared forward-model engine: `config`, `vulcan_chem`, `interp_map`, `exojax_rt`, `sensitivity` |
| `tests/` | fast unit tests (binning matrix, u-space prior, SMC-on-Gaussian, init reject-and-cull, warm reject, warm extrapolation, validate_warm) |
| `runs/w39b_smc_retrieval/` | the WASP-39b case directory: `case.py` (planet identity, priors, presets), the NAS PBS script, `overrides/`, run outputs |
| `examples/` | the sensitivity demo: `run_demo.py`, `run_figs.py`, `fig_exojax_sensitivity.py` |
| `validation/` | nine physics/numerics validation scripts (offline smokes plus the 2026-07-11 audit-response suite) |
| `scripts/zco_information/` | the Z vs C/O Fisher/Laplace information analysis |
| `data/` | INPUTS (`config.DATA_DIR`): the tracked `cm24_wasp39b/` observations plus the gitignored seeded caches `exojax_linelists/` and `opacity_cache/`; provenance in `data/notes.md` |
| `output/` | GENERATED npz caches from examples/validation/zco (`config.OUTPUTS`; gitignored, regenerable) |

Historical development logs live in per-directory `notes.md` files; this README
keeps only current usage and reference content.

## The shared forward engine: `retrieval_framework.forward`

The core chains the *live* VULCAN-JAX photochemical-kinetics model into the
ExoJax radiative-transfer code and propagates gradients through the whole thing:

```
physical params ─► VULCAN-JAX ─► VMR(nz, species), T(nz), P(nz)
  (lnZ, C/O, lnKzz,     (converged column, photochemistry ON)
   T-P params)                    │  log-P bridge (differentiable interp)
                                  ▼
                        ExoJax ArtTransPure ─► transit depth (Rp/Rs)²(λ)
                        (premodit line opacity + H2-H2/H2-He CIA + Rayleigh)
                                  │  jax.jvp / adjoint
                                  ▼
                        d(spectrum)/d(param)   — sensitivity, information, retrieval
```

It only imports VULCAN-JAX and ExoJax and never modifies them.

The one thing that makes all of this possible: **VULCAN-JAX's integrator is a
`lax.while_loop`, which supports forward-mode AD (`jvp`/`jacfwd`) but not reverse-mode
through the loop.** So forward-mode is the end-to-end route (and it is the right shape:
a few physical scalars → a high-dimensional spectrum). Reverse-mode is available only
*at* the converged state (the reaction-importance adjoint in VULCAN-JAX) and through
the RT alone (the staged evaluator's per-spectrum RT vjp, below).

Five planet-agnostic modules, imported as
`from retrieval_framework.forward import config, vulcan_chem, interp_map, exojax_rt, sensitivity`:

| module | what physics it owns |
|--------|----------------------|
| `config` | pure constants: molecule/isotope set + masses, wavenumber band, ART pressure grid, planet defaults, cache paths. No heavy imports (safe to load before the env-sensitive VULCAN-JAX setup). |
| `vulcan_chem` | the VULCAN-JAX side: one warm-up convergence, then `converged_ymix(theta)` / `converged_y(...)` — re-converge the closed column as a function of (lnZ, C/O, lnKzz, T-P). Sets the network env vars + jax x64 on import. |
| `interp_map` | differentiable log-pressure bridge from the VULCAN grid (nz) to the ExoJax ART grid (nlayer) — `jnp.interp`, so tangents pass across it. |
| `exojax_rt` | ExoJax `ArtTransPure` (transmission) / `ArtEmisPure` (emission) sharing one opacity set; `transmission_depth_r(...)`, `emission_flux(...)`. |
| `sensitivity` | the theta → converged VMR → transit spectrum chain composer for forward-mode jvps. |

**Import-order contract (load-bearing):** `retrieval_framework.forward.vulcan_chem`
must be imported before anything exojax. It sets the `VULCAN_JAX_*` import-frozen
env vars and jax x64 at import, and enforces the ordering with a loud
`RuntimeError` guard if exojax is already in `sys.modules`. `config` is always
safe to import first, and the `forward` package `__init__` deliberately imports
nothing, so light consumers (config readers, the jwst-tool GUI's cache face)
stay free of jax/vulcan_jax/exojax side effects.

**Cross-cutting assumptions (true for every consumer):**

- **Photochemistry is ON.** Only in the photo-on regime does the warm-started
  forward-mode jvp relax to the true steady-state sensitivity (FD-validated <0.1% at
  nz=150). It is also what produces the SO₂ that anchors the WASP-39b science.
- **Closed column.** FastChem sets the equilibrium initial abundances at 10× solar and
  stays frozen/off-graph. The runner forgets the initial speciation except through the
  conserved *elemental column totals*, so metallicity and C/O are expressed as
  initial-abundance (y₀) directions on those totals.
- **Exact-elemental abundance knobs** (`abundance_mode="elemental"`, the production /
  retrieval / jwst_tool default since 2026-07-11): after the mask-scaled guess, the
  column is renormalized to Σᵢnᵢ = P/(k_BT) per layer and linearly repaired on the
  runner's own reservoir species (He/H₂O/CO/N₂/H₂S) so the column elemental ratios hit
  He/H = base, {O,N,S}/H = Z×base, C/H = Z·e^{c_o}×base **exactly**, and the conserved
  atom anchor `pv.atom_ini` is rebuilt from that column. Cold and warm continuation
  therefore share identical conserved inventories by construction. The legacy
  `"masks"` mode (published demo caches) is kept, with its elemental leakage (~0.6%
  of H per e-fold of Z; N/S leakage through the fixed-O compensation) documented in
  `forward/vulcan_chem.py`. `chem.audit_init(theta)` and `validation/elemental_audit.py`
  verify the construction per draw.
- **Fixed-oxygen C/O guess** (`co_mode="fixed_O"`): the smooth initial guess scales
  C-bearing species and compensates O-only carriers so each layer's O total is
  invariant, valid while the compensation factor stays positive (a bound printed at
  build; the retrieval prior is capped below it). In elemental mode the projection
  makes the result exact regardless.
- **Atmospheric structure follows the proposed T-P/composition.** The runner refreshes
  the hydrostatic geometry (µ, g, H_p, dz, dzi, Hpi) in-loop from the live composition
  and the proposal's T every `update_frq` accepted steps (first firing at step 1);
  since 2026-07-11 the molecular-diffusion coefficients D_zz(T, M) (+ vm/vs where
  enabled), the convergence gate's Kzz, and the initial carry geometry are also
  rebuilt per proposal via the on-graph builders; since 2026-07-13 condensation
  (when `use_condense=True`) is rebuilt per proposal too — `_prep` regenerates the
  saturation number densities, Dg growth terms, NH3 cold-trap index, and fix-species
  sat-mix rows on-graph from the live T(P) via `vulcan_jax.conden.build_conden_profile`
  (the old loud refusal of a T-varying condensation build is gone). The one remaining
  baseline-T bake is the photolysis cross-section T-interpolation (host-side upstream
  step; second-order).
- **Opacities**: CO from cached ExoMol Li2015; H₂O/CO₂/CH₄/SO₂/HCN/C₂H₂/H₂S from HITRAN
  **main isotopologue at the 296 K reference** (intensities carry the terrestrial
  isotopic abundance factor — pairing with total molecular VMR is the standard,
  slightly conservative treatment). HITRAN under-represents hot bands at the retrieved
  ~770–1540 K limb vs HITEMP/ExoMol — a known accuracy limit for real-data inference,
  swap sources per molecule in `config.MOLECULES` when the big downloads are
  acceptable. Pressure broadening defaults to **terrestrial air**; set
  `config.BROADENING="h2he"` (or the per-run profile key) for HITRAN's planetary H₂/He
  widths where available, and measure the difference with
  `validation/broadening_ab.py`. H₂-He CIA is REQUIRED physics: every depth/flux call
  takes the He profile explicitly and raises if it is missing (a silent omission
  biased the pre-2026-07-11 sensitivity-demo caches). The premodit table is baked for
  **T ∈ [300, 3000] K** — profiles outside it are not modelable (see the retrieval's
  reject-don't-clip rule).
- **ART pressure grid** spans **1×10⁻⁸ – 7 bar**. The top sits one decade *above*
  VULCAN's 1×10⁻⁷ bar chemistry top on purpose: the interpolation CLAMPS the topmost
  chemistry values over that decade (constant-abundance + isothermal extension, a
  common transmission convention — not chemistry). Without it the strong bands (CO₂
  4.3, CO 4.7 µm) saturate into a flat wall. This is an explicit, logged modeling
  choice; `validation/top_pressure_ladder.py` measures it against chemistry actually
  solved to 1×10⁻⁸ bar.
- **T-P is ExoJax's own Guillot** (`atmprof_Guillot`, a plain `jnp.exp`, forward-mode
  clean — it bypasses the Heng+14 exponential-integral pathology in VULCAN's own
  `build_atm`). The same analytic T(P) drives both the chemistry (VULCAN grid) and the
  RT (ART grid); the precise scope of that consistency (and the f=1/4 shape-parameter
  caveat) is documented in `src/retrieval_framework/tp_profile.py`.
- **Native spectral resolution** defaults to nu_pts=1652 (R~1000 over the production
  band) — a GPU-gradient-memory bound, not a demonstrated convergence point.
  `validation/resolution_ladder.py` is the convergence test (binned depths + Jacobian
  columns vs a nu_pts ladder, optional LSF); run it before quoting few-ppm numbers.

## The retrieval framework

**Framework vs case.** Everything reusable lives in `src/retrieval_framework/`
(`config_schema.py`, `observations.py`, `tp_profile.py`, `retrieval_forward.py`,
`pipeline.py`, the `run_smc.py` driver, `plot_smc.py`, the
`calibrate_count_max.py` / `probe_memory.py` / `smoke_retrieval.py` /
`validate_warm.py` tools, and `tests/`). Everything planet-specific lives in a
CASE DIRECTORY (`runs/<case>/`): `case.py` with the `PRESETS` dict + planet
identity, the PBS submit script, `overrides/`, and run outputs. Entry points
take the case dir and run from the repo root:

```
python -m retrieval_framework.run_smc             runs/w39b_smc_retrieval [--calibrate]
python -m retrieval_framework.smoke_retrieval     runs/w39b_smc_retrieval
python -m retrieval_framework.calibrate_count_max runs/w39b_smc_retrieval --n-draws 96
python -m retrieval_framework.probe_memory        runs/w39b_smc_retrieval
python -m retrieval_framework.plot_smc            runs/w39b_smc_retrieval/data/gpu
python -m retrieval_framework.validate_warm       runs/w39b_smc_retrieval
```

`validate_warm` re-solves a finished run's checkpointed cloud COLD and compares
against the warm-carried results -- the direct measurement of the
warm-continuation history dependence. It gates on three axes (2026-07-11): max
|dlogL| < 0.1, binned-spectrum agreement < 5 ppm, and elemental-inventory
agreement. Run it once per production run; a published retrieval should quote
its result, along with the prior convergence-acceptance fraction (the
operational prior is p(theta | chemistry converges) -- init logs the reject
fraction).

Everything below documents the framework, using the original WASP-39b
application (`runs/w39b_smc_retrieval/`, real Carter & May 2024 NIRISS+G395H
data, `data/cm24_wasp39b/`) as the running example. Numbers tied to that
case (10-D theta, its priors, 152 bins, band edges) come from ITS presets in
`case.py`, not from the framework. The engineering history behind every choice
is preserved verbatim in `notes.md`.

```
theta ──► VULCAN-JAX (re-converged column, photochemistry ON, nz=50)
  │              │ VMR(nz,ni)
  │              ▼
  ├─► ExoJax Guillot T-P ──► ArtTransPure transmission (premodit opacities + CIA)
  │              │ native depth (Rp/Rs)^2 (λ)
  │              ▼
  └─► lnR0 ──► B (exact trapezoidal binning matrix) ──► + instrument offsets
                 │
                 ▼
      Gaussian likelihood on the 152 real C&M bins (93 NIRISS + 59 G395H, 1.02-5.24 um)
```

## Parameters (gpu preset: 10-D; only the first 6 are chemistry-expensive)

Priors are literature-anchored (Tsai et al. 2023 VULCAN grid + Rustamkulov et al. 2023
PRISM ERS), 2026-07-08; values live in `runs/w39b_smc_retrieval/case.py::_W39B`.

| # | name | prior | role |
|---|------|-------|------|
| 0 | `lnZ` | U(−2.303, 2.303) | metallicity about the 10×solar baseline (1–100× solar; kept wide, data localizes it) |
| 1 | `c_o` | U(−1.70, 0.24) | Δln(C/O) at **fixed O** → C/O ∈ [0.10, 0.70] (Rustamkulov C/O<0.7; upper edge below the b_z>0 validity bound 0.566) |
| 2 | `lnKzz` | U(−4.6, 4.6) | eddy-diffusion multiplier (±2 dex about the GCM baseline Kzz profile; Tsai used ±1 dex) |
| 3 | `Tirr` | U(1100, 2200) K | Guillot irradiation temperature (terminator ~770–1540 K; Teq~1100–1166, SO₂ sweet spot Teq 1000–1600) |
| 4 | `log10kappa` | U(−3.5, 0.5) | Guillot IR opacity κ_th [cm²/g] |
| 5 | `log10gamma` | U(−2, 0.301) | Guillot κ_v/κ_th → γ ∈ [0.01, 2] (weak inversion allowed) |
| 6 | `lnR0` | U(−0.08, 0.08) | reference-radius nuisance (Batalha & Line 2017) |
| 7 | `log10kappa_cloud` | U(−7, 1) | ExoJax `powerlaw_clouds` opacity at 3.5 µm [cm²/g] |
| 8 | `cloud_alpha` | U(0, 6) | cloud/haze power-law slope in ν (0 = gray deck) |
| 9 | `offset_G395H` | U(−800, 800) ppm | flat depth offset vs the NIRISS reference |

> **T-P is drawn RAW (no clip).** A profile that leaves the modelable window
> [300, 3000] K on the ART grid is REJECTED and redrawn (the prior is
> (box) ∩ (in-window); `pipeline.tp_valid`), never bent into range.

## Current configuration & numerics (authoritative, 2026-07-08)

The engineering log in `notes.md` is HISTORICAL; these are the current values (all
shown in the loud config banner every run prints, via `config_schema.describe_config`):

- **Convergence = VULCAN-master canonical** (Tsai et al. 2017 values): `yconv_cri=0.01`,
  `slope_cri=1e-4`; `yconv_min`/`flux_cri` inherit `vulcan_cfg_W39b`. (Not the sensitivity
  demo's tight `1e-3`.)
- **`count_max=5000` accepted steps, fixed.** A solve that doesn't converge is a failed
  draw — not extended, not clipped. `dt_max=1e11 s` (NOT the master default `1e17`): the
  default let the adaptive step balloon to ~1e16 s on high-Kzz columns and spin without
  settling — the bulk of the old >10k tail. Capping it converges those in ~1000 steps and
  leaves normal columns identical (a step-size control, not a convergence criterion). A
  genuine residual (marginal-`longdy` and photochemical-limit-cycle columns) still fails
  and is rejected at init. Full diagnosis + VULCAN-publication check: `CLAUDE.md`.
- **Priors** are literature-anchored (table above) and live in `case.py`.
- **Calibration runs at native R=100 by default** (`calibrate_count_max.py`): `accept_count`
  is resolution-independent, so the RT is run cheap; production uses the real `nu_pts`.
- **RT resolution `nu_pts=1652` (native R~1000) — memory-safe by DEFAULT.** RT-vjp gradient
  memory scales with `nu_pts` (the point count, NOT R), and OOM'd the 96 GB GH200 at 343 GiB
  when `nu_pts=16500`. R~1000 (~11 model pts per binned point) is ample and keeps the RT-vjp
  at ~34 GiB. The schema default is now 1652 (was 6000, a ~R10000 memory bomb) and
  `validate_config` warns above `nu_pts=2500`; run `PROBE_MEMORY=1` before ever raising it.
- **The SMC mutation rejects non-converged warm proposals** (fixed 2026-07-09). A MALA
  proposal whose warm chemistry continuation hits `count_max` is treated as an MH rejection
  (`L=-inf`), not fed into the gradient — the warm-side analogue of the cold-init
  reject-and-cull. Without it a non-converged proposal's garbage jvp/RT-vjp tripped `n_bad`
  / NaN'd at SMC stage 0 (the bug that killed job 64604 and the calibrations). See
  `CLAUDE.md` and `tests/test_warm_reject.py`.

RT carries **8 molecules** (H2O, CO2, CO, CH4, SO2 + **HCN, C2H2, H2S** — the
high-C/O and reduced-sulfur discriminators, so the likelihood can actually see the
species that decide the C/O upper tail) plus **H2-H2 and H2-He CIA**. The cloud is
ExoJax's shipped retrieval cloud (`exojax.atm.simple_clouds.powerlaw_clouds`, pRT
convention, per gram of atmosphere, uniformly mixed); cloud dims are RT-only, so
they ride the cheap gradient block like lnR0 — ~zero cost against the GPU budget.
H2/He **Rayleigh scattering** (ExoJax `xsvector_rayleigh_gas`, zero free parameters)
is on by default — required once the band reaches 1 µm, else its slope leaks into
the haze posterior.

The T-P is **ExoJax's own** `exojax.atm.atmprof.atmprof_Guillot` (Guillot 2010
Eq. 29; plain `jnp.exp`, forward-mode-clean), evaluated on *both* the VULCAN grid
(chemistry re-converges under it; rate constants rebuilt on-graph) and the ART grid
(RT scale height) — one self-consistent profile, hooked into `vulcan_chem` via the
backward-compatible `tp_eval` argument. `Tint=150 K` and `f=1/4` are fixed
(config fields). Optional: `infer_noise_inflation` adds a Line-2015-style ×σ term.

**Two-stage solve (`two_stage_z`, default on).** Measured 2026-07-05: perturbing the
cold EQ init's metals and re-converging through a *T-displaced* transient (any
retrieved T-P ≠ the baked profile) erases the inventory perturbation — converged CO
changed 1e-11 for a 5 % metals step under a Guillot T-P, vs the exact 5 % at
T=T_base. The forward therefore (1) converges the column at the retrieved T-P/Kzz
with baseline composition, then (2) applies the lnZ / C-O scaling to that converged
column and re-converges warm (`converged_y(warm_y=…)`, the Hessian-campaign
continuation pattern + `reanchor_atom_ini`). Stage 2 is cheap (warm start) and
gentle (uniform metal scaling, no species crash), so the inventory — and with it
the lnZ/C-O gradient — survives. `smoke_retrieval.py` hard-fails if those
gradients go dead again.

## Why forward-mode MALA (and how it stays affordable)

The VULCAN runner is a `lax.while_loop`: **jvp works, vjp does not**. The
likelihood gradient is therefore assembled from forward-mode jvps and exposed to
the MALA kernel through a `custom_vjp` (the SWAMPE trick). Two cost structures are
implemented (`gradient_mode`):

- **`block` (default, exact):** only the 6 chemistry+T-P directions push tangents
  through the VULCAN loop; `lnR0` is one RT-only jvp at the frozen converged
  profiles; offsets/noise gradients are analytic. ~25–35 % cheaper per MALA step.
- **`naive`:** all dims through the full chain (SWAMPE-identical), kept as a
  cross-check. `smoke_retrieval.py` asserts block ≡ naive to fp precision and
  validates both against re-converged finite differences.

**Staged batched hot path (2026-07-06 rework).** The per-particle modes above are
kept for validation, but the SMC itself runs a STAGED evaluator that splits the
chain at the chemistry/RT boundary — because the two halves have opposite
economics: the chemistry `while_loop` has tiny per-lane state (~MB; width is
nearly free and is what keeps the GPU busy) but only supports jvp, while the
ExoJax PreMODIT RT is vjp-capable but costs ~GB of intermediates per lane.
So per mutation sweep:

1. chemistry: 6 fwd-jvp lanes per particle, ALL particles in ONE wide batched
   `while_loop` (N×6 tangent-augmented columns — 864 at the gpu preset's N=144,
   nz=50). The warm accept_count diagnostic rides this same jvp'd chain (it is
   part of the runner's primal carry, integer-valued so tangent-free) — the
   2026-07-09 rework; an earlier version ran a SECOND primal-only `while_loop`
   just to read it, doubling the chemistry wall time per sweep;
2. RT: ONE reverse-mode vjp per particle (no while_loop inside the RT, so this is
   legal), `lax.map`-chunked over particles (`smc_rt_vjp_chunk`); the single
   backward pass replaces 6 forward RT tangents + the 3-dim lnR0/cloud jacfwd,
   and its aux-cotangent is contracted against the chemistry tangents
   (`dL/dθ_k = ⟨∂aux/∂θ_k, ∂L/∂aux⟩` — same chain rule, regrouped, asserted
   identical to `block` in the smoke);
3. offsets/noise: analytic.

The mutation kernel also CARRIES each particle's converged column
(`smc_chem_mode="warm"`, default): every proposal's chemistry warm-continues from
the particle's own state with incremental lnZ/C-O scaling (the validated
Hessian-campaign continuation pattern; `converged_y(warm_y=...)`) instead of
re-running the full cold two-stage solve — measured ~500–800-step re-converges
for MALA-sized moves (the `conv_step=500` longdy certification window dominates
the warm floor, not `count_min`), still ~6–8× fewer chemistry steps than cold.
Don't shrink `conv_step` to buy speed: probed 500→300 (2026-07-10), it saves
NOTHING on extrapolated seeds and 7% on plain ones while certifying a state
0.07 dex off the 500-window one — see `CLAUDE.md` for the full numbers.
MUTATION warm solves run under `warm_count_max` (default 1500, a twin runner
with the smaller cap baked in): a proposal still unconverged there is REJECTED
(-inf L) instead of dragging the whole lockstep batch to the cold `count_max` —
without this, any single bad proposal among N gated EVERY early-ladder sweep at
5000 steps (the ~3-6 h/stage pathology of job 64745; while the cloud is
prior-like that is essentially every sweep). The INIT gradient pass (phase 2)
runs the same warm map UNCAPPED (`batch_eval_init_vg`, gated at the cold
`count_max`): its inputs are phase-1 survivors re-certifying from their own
converged columns — proven-convergent particles, not disposable proposals — and
a marginal survivor can legitimately need more than the mutation cap just to
re-certify (NAS job 64854: the cap gated 5/96 healthy survivors into a spurious
"crippled cloud" raise). Some marginal columns cannot re-certify even at the
cold cap (job 64897: 3/96 — oscillating/stall-fallback certifications re-pay
the time-based window on restart and lose), so phase 2 evaluates
`N + init_phase2_spare` survivors and CULLS the re-certification failures,
backfilling from the spares — the same reject-don't-carry philosophy as
phase 1, logged as part of the operational prior. A true RT/AD death at
phase 2 (non-finite forward with a NON-exhausted accept count) still raises. Optional on top: `warm_extrapolate=true` seeds each
proposal's warm solve at the first-order prediction `Y + (dy/dθ)·Δθ` using the
tangents the gradient pass already computes (measured 1.65x fewer warm steps,
same certified state; opt-in pending a SYNTH A/B — see the lever list below and
`config_schema.py`). The cold
two-stage map runs exactly once per particle, at state initialization (and on
`smc_chem_mode="cold"`, which restores the published solve-from-baseline map for
every evaluation). The carried
likelihood also serves the tempering reweight, so a stage costs ~one mutation
call. Caveat (documented, deliberate): with warm continuation the likelihood is
defined by the continuation map from the particle's own history — path-dependent
at the yconv tolerance level; the smoke FD-checks the warm gradient against the
identical warm map.

## Sampler

`pipeline.run_smc_loop`: Del Moral resample–move SMC — ESS-bisection temperature
ladder (`target_ess_frac=0.6`), systematic resampling, then
`smc_num_mcmc_steps=6` preconditioned-MALA sweeps per stage (published
MALA-within-SMC practice is 3–10 with a good preconditioner; the old 12 was ~2×
generous and each sweep costs one full batched gradient). The proposal is
preconditioned with the **absolute per-dim std of the freshly resampled cloud**
(so it narrows in lockstep with tempering — the Gaussian test showed the
SWAMPE-style unit-geometric-mean scaling collapses acceptance after large β
jumps), plus Robbins–Monro fine-tuning of the scalar step toward 0.55 acceptance.
Every sweep logs a heartbeat line (acceptance, rejected-proposal count,
n_bad_grad) via `jax.debug.callback`, so a slow stage is visible per-sweep
instead of hours of silence. Per-stage atomic checkpointing
(`smc_checkpoint.npz`) and a **walltime governor** (gpu preset: stops cleanly at
20 h inside the 24 h PBS wall) guarantee a usable posterior from any job;
RESUME=1 continues a stopped ladder.

## Quickstart (local, vulcan env)

```bash
conda activate vulcan
cd vulcan-retrieval
python -m pytest tests -q   # binning matrix, u-space prior, SMC-on-Gaussian (~2 s)
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval   # FD + block≡naive gradient checks (~10-30 min)
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.plot_smc runs/w39b_smc_retrieval/data/smoke
```

## GH200 (NAS)

Code deploys by `git pull --ff-only` into the two clones under
`/nobackup/$USER/VULCAN_W39b_HPC` (`vulcan-retrieval` and `VULCAN-JAX`; both
clone target names are load-bearing). Data is seeded ONCE per fresh clone
(`data/opacity_cache/`, `data/exojax_linelists/`) by mv from an old
vulcan_exojax_run tree or a one-time scp; never rsync. Full transfer + setup
rules: `CLAUDE.md`.
Environment setup is a ONE-TIME bootstrap job (`qsub tools/bootstrap_nas_env.pbs`
from this repo's root): it editable-installs VULCAN-JAX and vulcan-retrieval into a
per-interpreter user site, installs the pinned exojax under a jax/jaxlib constraint,
builds FastChem for the node architecture, and validates. The run PBS reuses the
shared `pyt2_8_gh` env, is read-only on the environment, validates it with
`python -m retrieval_framework.validate_env` (re-runnable by hand any time), and
exports `VULCAN_PROJECT_ROOT` so the data paths resolve. Re-run the bootstrap only
when validate_env says so (packaging metadata changed). The staged submit order
lives in "The WASP-39b case" below.

## GPU budget (24 h PBS wall, 20 h SMC governor)

Cost ≈ `init + stages × mutation call` (the reweight uses the carried likelihood,
so it is free). `init` = one batched COLD two-stage solve per particle,
LIKELIHOOD-ONLY at full width, followed by one warm-map gradient sweep from the
just-converged columns (the 2026-07-07 fix, `notes.md` §K — the old cold-gradient
init cost >16 h). The cold pass is the only solve-from-baseline work in the run,
gated by the slowest prior-corner particle (count_max=5000 accepted steps bounds
it, and `dt_max=1e11` keeps the adaptive step from ballooning — see the
authoritative block above and `CLAUDE.md`). A mutation call = `6 MALA sweeps ×`
(ONE wide batched warm chemistry re-converge with 6 jvp lanes/particle,
`warm_count_max`-capped, + chunked RT vjp passes). The warm cap is what bounds
the early ladder: without it, any single non-convergent proposal among N gated
every sweep at the full cold count_max (job 64745: ~3-6 h/stage; see
`CLAUDE.md` "Mutation sweep cost").
The adaptive ladder typically needs ~12–25 stages. Run `CALIBRATE_ONLY=1` after
any config change — it times init + one mutation call and projects 15/25/40
stages. The governor makes the budget a guarantee rather than a hope; if
calibration says a stage is slow, the levers in order of pain:

1. `warm_extrapolate=true` (opt-in, wired 2026-07-10): seed each proposal's warm
   solve from `Y + (dy/dtheta)·dtheta` using the tangents the gradient pass
   already computes — measured 780 → ~470 typical warm steps (1.65x), same
   converged column (parity unit-tested). Validate once with a SYNTH A/B, then
   also drop `warm_count_max` 1500 → ~800 for the second half of the win.
2. `warm_count_max` 1500 → ~1000 (bounds the worst-case sweep; measured typical
   plain warm re-converge is ~500-800 steps, so watch the per-sweep rejected
   count in the heartbeat lines for collateral rejections)
3. `smc_num_mcmc_steps` 6 → 4 (linear savings, some mixing loss)
4. `nz` 50 → 40 (chemistry cost ~linear in nz)
5. band: drop to G395H-only (halves n_bin; loses NIRISS lever + offset)

(`yconv_cri` is NOT a speed lever here — it's fixed at the master `0.01`, and the
operative gate is the loose branch anyway; the real step-count lever was `dt_max`.)

(There is deliberately NO gradient-free kernel fallback — project rule: loud
errors over silent degradation.)

Note on `smc_num_particles` (updated after the 2026-07-07 probe): the chemistry
— primal AND gradient — runs full-width (`smc_chem_chunk=0`), so N widens those
kernels nearly for free (576 lanes at N=96 ≈ the paper's batch-256 benchmark
width; the GH200 power trace confirms it — wattage rises with lane count while
step time barely moves) — which is why the gpu preset ships N=144 as of
2026-07-10 (raised from 96 to spend the measured ~300-of-700 W headroom on
particles; see `CLAUDE.md` "GPU power headroom"). The ONLY N-linear cost is the
RT VJP stage (ceil(N/`smc_rt_vjp_chunk`) sequential blocks; the per-lane vjp
memory scales with nu_pts — 18.4 GiB/lane at nu_pts=5000, ~3× less at the
production 1652, so the gpu preset runs 12-wide: 96/12 = 8 blocks, the same
serial count as the old 48/6). Shrinking that per-lane cost (checkpoint
granularity / xs tables / fp32 xs inside exojax_rt) is still the top structural
speed item — it unlocks full-width RT.

Remaining documented-not-wired speedup: the reverse-mode steady-state adjoint
(`steady_state_grad`), which would make the chemistry gradient cost
dimension-independent.

## Outputs (per run dir, e.g. `data/gpu/`)

`config.json` (full config + param metadata) · `observations.npz` ·
`smc_checkpoint.npz` (atomic, per stage; carries `init_stats_*` + `warm_capped`) ·
`posterior_samples.npz` (`samples (chains, n, dim)` + `reached_beta1`) ·
`smc_extra_fields.npz` (β ladder, ESS, acceptance, step sizes, unique counts,
warm-cap counts, logZ **plus its conditioning**: `smc_log_support_fraction[_err]`
and `smc_logZ_box`) · `posterior_predictive.npz` (PPC band + median model + χ²) ·
`timing.json` (calibrate mode) · `plots/{corner, spectrum_fit, tp_posterior,
smc_diagnostics}.png` (PNG only, dpi 200).

**Evidence semantics (2026-07-11).** `smc_logZ` is the evidence under the
OPERATIONAL prior — the declared box restricted to the modelable T-P window and to
draws whose chemistry converges (the init reject-and-cull), renormalized. The
support fraction is measured at init (binomial counts persisted through
checkpoints) and `smc_logZ_box = smc_logZ + ln f_support` is the box-prior value
with the non-evaluable region assigned zero likelihood. Quote them together;
never compare `smc_logZ` across models whose support fractions differ.

## The WASP-39b case: `runs/w39b_smc_retrieval/`

Fits the **real Carter & May (2024) combined JWST transmission spectrum of
WASP-39b** (NIRISS SOSS + NIRSpec G395H, `data/cm24_wasp39b/`). The case
directory holds ONLY what is specific to this run:

- `case.py` — planet identity (gravity, radii, VULCAN cfg module, C&M product
  table) + the `smoke` / `gpu` / `prod` presets (`PRESETS` dict).
- `run_nas_w39b.pbs` — the NAS GH200 submit script (all modes: run, SYNTH,
  CALIBRATE_ONLY, CALIBRATE_COUNT_MAX, PROBE_MEMORY, NSYS profiling; the PBS
  header documents the knobs).
- `overrides/*.json` — optional Config-override files
  (`SMC_RETRIEVAL_OVERRIDES_FILE=overrides/<f>.json`, resolved against the case dir).
- `data/<preset>/` — run outputs (posterior npz, config.json, run.log, plots/).
- `logs/` — PBS live logs + GPU monitor + nsys reports.

Local smoke (offline, CPU, ~minutes; always do this after framework changes),
from the repo root:

```
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval   # gradient FD checks
```

NAS GH200 (from the case directory on the NAS tree), in the staged order for a
fresh campaign:

```
qsub -v PROBE_MEMORY=1 run_nas_w39b.pbs        # compile-only buffer report (REQUIRED after any N/chunk/nu_pts change)
qsub -v CALIBRATE_ONLY=1 run_nas_w39b.pbs      # ~1.5 h; check timing.json t_mutation_sweep_s
qsub -v SYNTH=1 run_nas_w39b.pbs               # synthetic recovery test at gpu fidelity
qsub run_nas_w39b.pbs                          # real-data production (gpu preset)
qsub -v RESUME=1 run_nas_w39b.pbs              # continue a governor-stopped ladder
qsub -v CALIBRATE_COUNT_MAX=1,CALIBRATE_COUNT_MAX_PROBE=60000,CALIBRATE_N_DRAWS=96 run_nas_w39b.pbs
```

On success, plots + the warm-vs-cold validation (`validate_warm`) run
automatically; quote the verdict and the init reject fraction in the paper.
Before trusting a real-data posterior: one clean `SYNTH=1` recovery at gpu
fidelity (injected truths inside their 90% intervals) and `VERDICT: PASS` from
the automatic warm-vs-cold validation. The case's decision history and open
items are in `runs/w39b_smc_retrieval/notes.md`; operational rules (count_max,
dt_max, nsys masking, XLA A/B candidates) are in `CLAUDE.md`.

## Sensitivity examples: `examples/`

**Question:** which wavelengths best constrain each physical parameter (an
observation-planning "which instrument/band?" view).
**Method:** forward-mode `jvp` of the WASP-39b transit depth w.r.t. (lnZ, C/O,
lnKzz, dT), coloring the spectrum by each `d(depth)/d(param)`. The scalar dT is
a uniform temperature offset, a proxy the retrieval upgrades to a full Guillot
T-P via `vulcan_chem`'s `tp_eval` hook.

- `run_demo.py`: the headline 2×2 figure (nz=150, 2.9–5.2 µm).
- `run_figs.py`: wide-band (1–15 µm, R=100) transmission AND emission
  sensitivity figures; the 4 chemistry jvps are shared between the two RT modes.
- `fig_exojax_sensitivity.py`: the single-panel manuscript figure (reads the
  cached jvp output; requires the `jax_paper` sibling checkout).

Run from the repo root: `python examples/run_demo.py`.
**Outputs:** `output/{sensitivity,wide_sensitivity}.npz` caches +
`../jax_paper/figures/exojax_sensitivity.png`. Key result: metallicity is best
measured in the 4.0–4.3 µm SO₂/CO₂ band (the same window JWST uses). The
pre-2026-07-11 caches were deleted as stale; regenerate before rebuilding
figures. Demo-specific findings: `examples/notes.md`.

## Validation scripts: `validation/`

Two groups. Offline pre-flight smokes (laptop-safe; run before trusting any
figure): `smoke_test.py` (end-to-end jvp vs re-converged FD, CO-only),
`smoke_coref.py` (the c_o_ref continuation holds C/O fixed), `smoke_zco.py`
(chemistry-tier tangents + fixed-O knob), `validate_wide_chem.py` (chemistry jvp
vs FD at nz=150). The 2026-07-11 audit-response suite (run on the GPU node
before the next production retrieval): `elemental_audit.py`,
`resolution_ladder.py`, `top_pressure_ladder.py --extend-chem`,
`broadening_ab.py`, and post-run `mala_reversibility.py`.

Run from the repo root, e.g. `python validation/smoke_test.py`.
Per-script details, PASS gates, and recorded verdicts: `validation/notes.md`.

## Z vs C/O information analysis: `scripts/zco_information/`

**Question:** the JWST spectrum constrains a Z–C/O combination; how much *unique*
information does it carry about each, and how much of it comes from disequilibrium
(quench + photochemistry) rather than equilibrium?
**Method:** Fisher/Laplace analysis on the autodiff Jacobian of the real Carter &
May 2024 combined spectrum, with a true fixed-O C/O knob, marginalizing lnKzz, dT,
a reference-radius nuisance, and per-instrument offsets. Compares equilibrium →
quench → photochem tiers.
**Assumptions (documented toy limits):** local-linear (Gaussian) Fisher; **no
clouds, no free T-P, no stellar contamination** — so *absolute* σ are best-case
lower bounds, but the *relative* statements (which wavelengths, which chemistry
tier, which parameter combination is degenerate) are robust. Equilibrium tier
drifts ~3% (moldiff off).
**Outputs:** `../jax_paper/figures/zco_{information,disequilibrium,geometry}.png`,
`output/{zco_jacobians,zco_walk}.npz`.

Build order, from the repo root:
`python scripts/zco_information/build_zco_jacobians.py` (per-tier
Jacobians; `--smoke` for the fast check), then `build_zco_walk.py`
(Gaussian-validity walk), then the three `fig_zco_*.py` scripts. The
pre-2026-07-11 caches were deleted as stale; regenerate first. Rationale and
cache policy: `scripts/zco_information/notes.md`.

## Honest limitations

- **Clouds are the parametric ExoJax power-law** (gray deck at α=0, haze slope
  otherwise), uniformly mixed — not microphysical. The ExoJax-native upgrade is the
  Ackerman & Marley stack (`exojax.atm.amclouds` + `PdbCloud("MgSiO3")` + `OpaMie`),
  with the cloud base at the retrieved T-P's `psat_enstatite_AM01` crossing and
  particle sizes from the **retrieved Kzz** — the honest "self-consistent-lite."
  Blocked only by `pip install PyMieScatt` (not in the envs) + a one-time miegrid
  build. Note VULCAN itself **cannot** do W39b clouds self-consistently: its
  condensation set is H2O/NH3/H2SO4/S2/S4/S8/C (cool-planet condensates) and Mg/Si
  aren't in the H,O,C,N,S atom set — silicate condensation would be major new
  chemistry, and the gradient path is validated conden-off.
- No stellar-contamination term (quiet G8 host; instrument offsets absorb residuals).
- HITRAN (296 K reference) opacities for H2O/CO2/CH4/SO2/HCN/C2H2/H2S, ExoMol for
  CO — adequate for the methodology; swap to HITEMP/ExoMol for publication-grade
  line fidelity (code-wise a dict entry; operationally multi-GB downloads + premodit
  memory tuning — the one upgrade with real wrangling risk).
- The chemistry's structure follows the retrieved T-P as of 2026-07-11: the runner's
  in-loop hydrostatic refresh (µ/g/Hp/dz, firing from step 1) plus the per-proposal
  on-graph rebuild of Dzz(T,M)/vm/vs, the convergence gate's Kzz, and the initial
  carry geometry. The one remaining baseline-T bake is the photolysis
  cross-section T-interpolation (host-side upstream step; second-order).
- If the governor (or a crash) stops before β=1 the samples are **tempered**
  (`reached_beta1=False` travels in BOTH `posterior_samples.npz` and
  `smc_extra_fields.npz`; every figure — corner, spectrum, T-P — is stamped and the
  PPC/recovery paths warn) — widths are lower bounds. Resubmit with
  `qsub -v RESUME=1 run_nas_w39b.pbs`: the ladder continues from the checkpointed
  cloud/β instead of restarting (validated in `test_smc_gaussian.py`).
- Observations are baked into the jitted likelihood at first trace:
  `set_observations` must be called exactly once, before inference (the driver
  enforces this ordering).
- `use_photo=True` is required — the forward-mode tangent is only validated
  photo-on (see `src/retrieval_framework/forward/config.py` FULL notes).
- **Diagonal likelihood.** The likelihood is a per-bin Gaussian: bin-to-bin
  covariance is neglected. This is common for reduced R≈100 products that provide
  only per-bin σ, and here wavelength covariance is not included because the data
  product supplies none — NOT a claim that covariance is universally irrelevant.
  Published JWST/NIRISS analyses have found significant wavelength-correlated noise
  and constructed spectral covariance matrices for retrieval; the diagonal treatment
  is a scope limit of the available product, not a statement about the underlying
  noise. An optional global multiplicative noise-inflation nuisance is available
  (`infer_noise_inflation`, off by default); there is no full/correlated covariance
  term. Correlated-systematics forecasting lives in
  the sibling `vulcan-jwst-tool` planner (its experimental scenario covariances)
  and is not ingested here.
- **Binning ≠ the planner's.** The native→bin operator (`observations.py`,
  `B @ native`) is the wavelength-space d(λ)-weighted trapezoidal average that
  reduced R≈100 products are compared against — it is *not* the stellar-count-
  weighted operator (plus native-R LSF) the `vulcan-jwst-tool` planner uses to
  forecast instrument noise. The two tools are deliberately not a matched
  injection→recovery closure pair: this framework fits real reduced spectra
  (C&M24 W39b), not planner-generated synthetic data.

## Tests

```
python -m pytest tests -q     # from the repo root; fast
```

Binning matrix ≡ trapezoid reference (incl. the real C&M bins), u-space prior
bounds/uniformity/Jacobian, Gaussian SMC recovery + evidence + governor +
resume, init reject-and-cull/backfill, warm-cap rejection, warm-extrapolation
parity, validate_warm.

## Development log

The full engineering history (the inventory-erasure finding, sampler lessons,
gradient architecture, GH200 memory probes and incident post-mortems, literature
context, the validation record, and the pre-launch review) is preserved verbatim
in `notes.md`. Read it before extending the framework.
