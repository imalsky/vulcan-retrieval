# vulcan-retrieval

`vulcan-retrieval` is a research framework for exoplanet atmospheric retrievals.
It connects [VULCAN-JAX](https://github.com/imalsky/jax-vulcan) photochemistry to
[ExoJAX](https://github.com/HajimeKawahara/exojax) radiative transfer. It uses
gradients from the full model to sample atmospheric parameters with sequential
Monte Carlo (SMC).

The repository includes:

- A reusable retrieval framework
- A WASP-39 b example that uses JWST transmission data
- A small synthetic case for local checks
- Validation scripts for the model and its gradients

This is research software. Review the model assumptions and run the validation
checks before you use results in a publication.

## How the model works

1. VULCAN-JAX solves the one-dimensional photochemical atmosphere.
2. The code maps the chemical abundances to the ExoJAX pressure grid.
3. ExoJAX calculates the transmission spectrum.
4. JAX calculates forward-mode gradients through the full model.
5. Adaptive-tempered SMC uses the Metropolis-adjusted Langevin algorithm
   (MALA) to sample the posterior distribution.

The retrieval code does not modify VULCAN-JAX or ExoJAX.

## Requirements

- Python 3.10 to 3.12
- VULCAN-JAX 0.3.0 or later
- ExoJAX 2.2.3
- JAX with a CPU or GPU backend
- A C++ compiler for FastChem, which VULCAN-JAX builds on first use

A CPU is enough for the unit tests and the small smoke checks. Use a GPU for a
production retrieval.

## Install

Clone the repository so that the code can find the tracked observation files:

```bash
git clone https://github.com/imalsky/vulcan-retrieval.git
cd vulcan-retrieval

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "vulcan-jax>=0.3.0"
python -m pip install -e ".[dev]"
```

VULCAN-JAX is on TestPyPI. The other Python packages are on PyPI. Install
VULCAN-JAX first, as shown above, so that the editable install does not look for
it on PyPI. In an environment that already has VULCAN-JAX, `pip install --no-deps
-e .` does the same job.

The code finds the `data/` and `output/` trees from an editable checkout. For a
non-editable install, set `VULCAN_PROJECT_ROOT` to the directory that contains
this repository.

### Add the opacity data

The Git repository does not contain the large opacity files. Add these files
under `data/opacity_cache/` before you build the forward model:

- `CO/12C-16O/Li2015/`
- `H2-H2_2011.cia`
- `H2-He_2011.cia`

ExoJAX can download the H2-H2 file when it is first needed. Download the H2-He
file from [HITRAN](https://hitran.org/data/CIA/main/H2-He_2011.cia).

ExoJAX also caches HITRAN line lists under `data/exojax_linelists/`. It
downloads these on first use, so you do not need to seed them by hand, but
copying an existing cache saves a long first run. See
[Data: provenance and regeneration policy](#data-provenance-and-regeneration-policy)
for the data layout and provenance.

## Quick start

Run the test suite:

```bash
python -m pytest tests -q
```

That runs everything, including three integration files that build a real
chemistry and radiative-transfer pipeline (about 10 minutes in total). For a
fast inner loop while developing, deselect them:

```bash
python -m pytest tests -q -m "not slow"     # ~10 s, 16 deselected
python -m pytest tests -q -m slow           # just the integration set
```

The marker deselects nothing by default on purpose: a silently skipped test is
a failure mode, not a convenience. Using `-m "not slow"` is an explicit choice
and pytest reports the deselected count.

Check the complete gradient path with the small WASP-39 b case:

```bash
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval
```

This check uses a CPU and usually takes 10 to 30 minutes.

Run the small synthetic retrieval:

```bash
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
```

Create the result plots:

```bash
python -m retrieval_framework.plot_smc runs/w39b_smc_retrieval/data/smoke
```

The plots are in `runs/w39b_smc_retrieval/data/smoke/plots/`.

## Entry points

Run every command from the repository root and give it a case directory, or a
run directory where the table says so.

| Command | Purpose |
| --- | --- |
| `run_smc <case>` | The retrieval driver. Add `--calibrate` to time one batch instead of running |
| `smoke_retrieval <case>` | Finite-difference and gradient-consistency checks |
| `calibrate_count_max <case>` | Choose the solver step cap from sampled draws |
| `probe_memory <case>` | Compile-only GPU buffer report. Run it after any change to `nu_pts`, the gradient chunk size, or the particle count |
| `validate_warm <case>` | Re-solve a finished run without warm starts and compare |
| `plot_smc <run dir>` | Corner, spectrum, temperature, and diagnostic figures |
| `validate_env <project root>` | Check the interpreter, installs, data files, and FastChem binary. Takes the directory that contains this repository |

## Run the WASP-39 b case

The example case defines three presets in
[`runs/w39b_smc_retrieval/case.py`](runs/w39b_smc_retrieval/case.py):

| Preset | Purpose |
| --- | --- |
| `smoke` | Small synthetic check for a local CPU |
| `gpu` | Full JWST retrieval for a GH200-class GPU |
| `prod` | Higher-resolution run without a wall-time limit |

Measure the cost of the GPU preset before you start a full run:

```bash
SMC_RETRIEVAL_PRESET=gpu python -m retrieval_framework.run_smc runs/w39b_smc_retrieval --calibrate
```

The NAS batch script is
[`runs/w39b_smc_retrieval/run_nas_w39b.pbs`](runs/w39b_smc_retrieval/run_nas_w39b.pbs).
[The WASP-39 b production case](#the-wasp-39b-production-case) covers the
submit sequence, the GPU budget, and the parameters and priors of the shipped
case.

## Create a new case

Create a new directory under `runs/`. Copy
`runs/w39b_smc_retrieval/case.py` into the new directory. Then set the planet
properties, observation files, priors, and presets for the new target. The
directory must contain:

- a `PRESETS` dictionary;
- one configuration function for each preset;
- an optional `DEFAULT_PRESET` name.

Use a JSON override for a temporary configuration change:

```bash
SMC_RETRIEVAL_OVERRIDES='{"smc_num_particles": 24}' python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
```

`SMC_RETRIEVAL_OVERRIDES_FILE` reads the same JSON from a file. The example case
keeps a few of these in `runs/w39b_smc_retrieval/overrides/`.

## Main outputs

Each run writes its files to `runs/<case>/data/<preset>/`.

| File | Contents |
| --- | --- |
| `config.json` | Complete resolved configuration |
| `run.log` | Run messages and diagnostics |
| `observations.npz` | Observation data used by the run |
| `smc_checkpoint.npz` | Restart data for the SMC run |
| `posterior_samples.npz` | Posterior or tempered samples |
| `smc_extra_fields.npz` | SMC diagnostics and evidence values |
| `posterior_predictive.npz` | Model predictions from the samples |
| `plots/` | Corner, spectrum, temperature, and SMC plots |

Check `reached_beta1` in the output before you treat samples as posterior
samples. If it is false, the run stopped before the SMC temperature reached 1.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/retrieval_framework/` | Retrieval, configuration, input, output, and plotting code |
| `src/retrieval_framework/forward/` | retrieval-side config + the sensitivity composer (the engine itself is the [vulcan-forward](https://github.com/imalsky/vulcan-forward) distribution) |
| `runs/` | Planet-specific cases and batch scripts |
| `examples/` | Spectral-sensitivity examples |
| `validation/` | Physics and gradient validation scripts |
| `tests/` | Fast automated tests |
| `scripts/zco_information/` | Metallicity and C/O information analysis |
| `data/` | Tracked observations and local opacity caches |
| `output/` | Generated example and validation data |

The forward model has one ordering rule. Import
`vulcan_forward.vulcan_chem` before anything from ExoJAX, because
it sets the VULCAN-JAX environment variables and enables float64 at import time.
It raises an error if ExoJAX is already imported.

## Reference documentation

The detailed reference lives in this file:

| Section | Contents |
| --- | --- |
| [The shared forward engine](#the-shared-forward-engine) | Forward-model modules, physics conventions, and shared assumptions |
| [Gradients and the sampler](#gradients-and-the-sampler) | Gradient architecture, warm continuation, and the SMC sampler |
| [The WASP-39 b production case](#the-wasp-39b-production-case) | The production case: submit sequence, GPU budget, and speed levers |
| [Limitations](#limitations) | Every known limitation, with its measured scope |
| [Tests and validation](#tests-and-validation) | The validation scripts and what each one checks |
| [Data: provenance and regeneration policy](#data-provenance-and-regeneration-policy) | The data layout, what is tracked, and how caches regenerate |

## The shared forward engine

The engine chains live VULCAN-JAX photochemical kinetics into ExoJax radiative
transfer and propagates gradients through the whole chain. It imports VULCAN-JAX
and ExoJax and never modifies them.

**Since 2026-07-29 the engine is its own distribution,
[`vulcan-forward`](https://github.com/imalsky/vulcan-forward)**, shared with
vulcan-jwst-tool: the two applications sit on one engine instead of the planner
depending on this framework. The physics described in this document did not
change in that move (a fresh solve reproduced the pre-split spectrum
bit-identically); only the import paths and the data contract did. What remains
in this repo is `forward/config.py` (this repo's paths, the WASP-39 b case
constants, the run profiles, the parameter-vector labels; it re-exports the
shared physics constants) and `forward/sensitivity.py`.

### Modules

Import as:

```python
from retrieval_framework.forward import config, sensitivity      # this repo
from vulcan_forward import vulcan_chem, interp_map, exojax_rt    # the engine
```

| Module | Physics it owns |
|---|---|
| `config` (this repo) | This repo's paths, WASP-39 b case constants, run profiles and parameter-vector labels, plus re-exports of the engine's shared physics constants (molecule set with masses, wavenumber band, radiative-transfer pressure grid). No heavy imports, so it is safe to load before the environment-sensitive VULCAN-JAX setup. It also hands the engine this repo's `data/` tree, so the opacity caches and line lists stay where they are |
| `vulcan_chem` (engine) | The VULCAN-JAX side. One warm-up convergence, then `converged_ymix(theta)` and `converged_y(...)` re-converge the closed column as a function of metallicity, C/O, `Kzz`, and the temperature-pressure profile. Sets the network environment variables and float64 at import |
| `interp_map` (engine) | The differentiable log-pressure bridge from the chemistry grid to the radiative-transfer grid, using `jnp.interp` so tangents pass across it |
| `exojax_rt` (engine) | ExoJax `ArtTransPure` for transmission and `ArtEmisPure` for emission, sharing one opacity set. Provides `transmission_depth_r(...)` and `emission_flux(...)` |
| `sensitivity` (this repo) | Composes the chain from parameters through the converged mixing ratios to the transit spectrum, for forward-mode gradients |

### The import-order contract

**Import `vulcan_chem` before anything from exojax.** It sets the import-frozen
`VULCAN_JAX_*` environment variables and enables float64 at import, and it
enforces the ordering with a `RuntimeError` if exojax is already in
`sys.modules`.

`config` is always safe to import first, and the shared engine's package `__init__`
deliberately imports nothing. That keeps light consumers, such as config readers
and the jwst-tool GUI's cache interface, free of jax, vulcan_jax, and exojax side
effects.

### Cross-cutting assumptions

These hold for every consumer of the engine.

#### Photochemistry is on

Only in the photochemistry-on regime does the warm-started forward-mode gradient
relax to the true steady-state sensitivity; this is validated against finite
differences to better than 0.1% at 150 layers. It is also what produces the SO2
that anchors the WASP-39b science.

#### The column is closed

FastChem sets the equilibrium initial abundances at 10x solar and then stays
frozen and off the graph. The runner forgets the initial speciation except through
the conserved **elemental column totals**, so metallicity and C/O are expressed as
initial-abundance directions on those totals.

#### Exact elemental abundance knobs

`abundance_mode="elemental"` is the default for production, retrieval, and the
jwst-tool since 2026-07-11. After the mask-scaled guess, the column is
renormalized to the hydrostatic number density per layer, then linearly repaired
on the runner's own reservoir species (He, H2O, CO, N2, H2S) so that the column
elemental ratios hit their targets **exactly**: He/H at the base value, O/H, N/H,
and S/H at metallicity times base, and C/H at metallicity times the C/O factor
times base. The conserved atom anchor `pv.atom_ini` is rebuilt from that column,
so cold and warm continuation share identical conserved inventories by
construction.

The legacy `"masks"` mode is kept for the published demo caches. Its elemental
leakage is documented in `vulcan_forward/vulcan_chem.py`: about 0.6% of H per e-fold of
metallicity, plus N and S leakage through the fixed-oxygen compensation.

Verify the construction per draw with `chem.audit_init(theta)` or
`validation/elemental_audit.py`.

#### Fixed-oxygen C/O

`co_mode="fixed_O"` scales the C-bearing species and compensates the O-only
carriers so that each layer's oxygen total is invariant. The guess is valid while
the compensation factor stays positive; the bound is printed at build time and the
retrieval prior is capped below it. In elemental mode the projection makes the
result exact regardless.

#### Atmospheric structure follows the proposal

The runner refreshes the hydrostatic geometry (mean molecular weight, gravity,
scale height, layer thicknesses) in the loop from the live composition and the
proposal's temperature, every `update_frq` accepted steps, first firing at step 1.

Since 2026-07-11 the molecular-diffusion coefficients, the convergence gate's
`Kzz`, and the initial carry geometry are also rebuilt per proposal through the
on-graph builders. Since 2026-07-13 condensation is rebuilt per proposal too:
`_prep` regenerates the saturation number densities, growth terms, NH3 cold-trap
index, and fix-species saturation rows on the graph from the live temperature
profile.

The one remaining baseline-temperature bake is the photolysis cross-section
temperature interpolation, which is a host-side upstream step and second order.

**Condensation is a forward-model capability only.** Gradient-MALA inference with
`use_condense=True` is refused, in `validate_config` and again on the fully
resolved config in `retrieval_forward`, because the pinned condensation state is
not reliably differentiable: the tangent disagrees with finite differences at
order unity, and the gas/condensate split has no derivative at all -- its centred
difference scales like `1/dT` (a fixed-size pin-capture jump, measured
2026-07-29), so any single relative-error figure is step-size dependent. Only the
conserved reservoir total is step-stable. See [Limitations](#limitations) and the
condensation contract in VULCAN-JAX `README.md` (Differentiability section).

#### Opacities

CO comes from a cached ExoMol Li2015 line list. H2O, CO2, CH4, SO2, HCN, C2H2, and
H2S come from HITRAN, main isotopologue at the 296 K reference. Those intensities
carry the terrestrial isotopic abundance factor, and pairing them with the total
molecular mixing ratio is the standard, slightly conservative treatment.

HITRAN under-represents hot bands at the retrieved 770-1540 K limb compared with
HITEMP or ExoMol. This is a known accuracy limit for real-data inference. Swap the
source per molecule in `config.MOLECULES` when the multi-gigabyte downloads are
acceptable.

Pressure broadening defaults to terrestrial air. Set `config.BROADENING="h2he"`
for HITRAN's planetary H2/He widths where available, and measure the difference
with `validation/broadening_ab.py`.

H2-He collision-induced absorption is **required** physics. Every depth and flux
call takes the helium profile explicitly and raises if it is missing; a silent
omission biased the pre-2026-07-11 sensitivity caches.

The premodit table is baked for 300-3000 K. Profiles outside that window are not
modelable, which is why the retrieval rejects rather than clips.

#### The radiative-transfer pressure grid

The grid spans 1e-8 to 7 bar. Its top sits one decade **above** the chemistry top
at 1e-7 bar on purpose: the interpolation clamps the topmost chemistry values over
that decade, a constant-abundance and isothermal extension that is a common
transmission convention rather than chemistry.

Without it, the strong CO2 4.3 um and CO 4.7 um bands saturate into a flat wall.
This is an explicit, logged modeling choice.
`validation/top_pressure_ladder.py` measures it against chemistry actually solved
down to 1e-8 bar.

#### One self-consistent temperature-pressure profile

The profile is ExoJax's own `atmprof_Guillot` (Guillot 2010, Eq. 29). It is a
plain `jnp.exp`, so it is clean under forward mode, and it bypasses the
exponential-integral pathology in VULCAN's own `build_atm`.

The same analytic profile drives both the chemistry, on the VULCAN grid, and the
radiative transfer, on the ART grid. `src/retrieval_framework/tp_profile.py`
documents the precise scope of that consistency and the shape-parameter caveat.

#### Native spectral resolution

`nu_pts` defaults to 1652, about R=1000 over the production band. That is a
GPU-gradient-memory bound, **not** a demonstrated convergence point.
`validation/resolution_ladder.py` is the convergence test: binned depths and
Jacobian columns against a ladder of `nu_pts`. Run it before quoting few-ppm
numbers.

### The two-stage solve

`two_stage_z` is on by default, and the reason is a measurement from 2026-07-05.
Perturbing the cold equilibrium initialization's metals and re-converging through
a temperature-displaced transient **erases the inventory perturbation**: converged
CO changed by 1e-11 for a 5% metals step under a Guillot profile, against the
exact 5% at the baked temperature. Any retrieved profile that differs from the
baked one triggers this.

So the forward model instead:

1. converges the column at the retrieved temperature and `Kzz` with the baseline
   composition, then
2. applies the metallicity and C/O scaling to that converged column and
   re-converges warm.

Stage 2 is cheap, because it warm-starts, and gentle, because the metal scaling is
uniform and no species crashes. The inventory survives, and with it the
metallicity and C/O gradients. `smoke_retrieval.py` hard-fails if those gradients
go dead again.

### Radiative-transfer contents

The transmission model carries eight molecules: H2O, CO2, CO, CH4, and SO2, plus
HCN, C2H2, and H2S. The last three are the high-C/O and reduced-sulfur
discriminators, included so the likelihood can see the species that decide the
C/O upper tail. H2-H2 and H2-He collision-induced absorption are always on.

The cloud is ExoJax's shipped `powerlaw_clouds`: opacity per gram of atmosphere,
uniformly mixed. Cloud dimensions are radiative-transfer only, so they ride the
cheap gradient block and cost almost nothing against the GPU budget.

H2 and He Rayleigh scattering is on by default and has no free parameters. It is
required once the band reaches 1 um, otherwise its slope leaks into the haze
posterior.

#### Height-dependent gravity

The transmission optical depth converts pressure to column mass with an
inverse-square gravity profile, `g(r) = g_btm (R_btm/r)^2`, evaluated at layer
midpoints — the same gravity law ExoJax's own height integrator uses, so heights
and opacity columns share one geometry. This is a local helper
(`exojax_rt._gravity_profile_invsq`), **not** ExoJax's `gravity_profile`: that
method (through 2.2.3) returns a profile linear in `1/r`, which leaves heights
and columns on different gravities and removes only about half of the constant-g
bias. Measured on an isothermal gray WASP-39b-like test against an independent
chord quadrature at 60 layers: constant g errs by -102 ppm, ExoJax's
`gravity_profile` by -51 ppm, the inverse-square profile by +1.5 ppm. The
wavelength-differential part (-32 ppm across a 100x opacity span for the
1/r-linear profile) is what would bias retrieved amplitude and slope parameters;
the constant part is absorbed by `lnR0`. Emission is plane-parallel and keeps the
constant bottom gravity.

## Gradients and the sampler

How the likelihood gradient is assembled, why it is forward mode, how the
warm-continuation scheme keeps it affordable, and how the SMC sampler is
configured.

### Why forward mode

VULCAN-JAX's integrator is a `lax.while_loop`. `jvp` works through it; `vjp` does
not. So the likelihood gradient is assembled from forward-mode tangents.

That is also the right shape for this problem: a few physical scalars in, a
high-dimensional spectrum out. Reverse mode is available only at the converged
state, through VULCAN-JAX's reaction-importance adjoint, and through the
radiative transfer alone.

### Two per-particle gradient modes

`gradient_mode` selects between them. Both are kept for validation.

- **`block`, the default and exact.** Only the six chemistry and
  temperature-pressure directions push tangents through the VULCAN loop. The
  reference-radius parameter is one radiative-transfer tangent at the frozen
  converged profiles, and the offset and noise gradients are analytic. This is
  25-35% cheaper per MALA step.
- **`naive`.** All dimensions through the full chain, kept as a cross-check.
  `smoke_retrieval.py` asserts that `block` equals `naive` to floating-point
  precision and validates both against re-converged finite differences.

### The staged batched hot path

The SMC itself runs a **staged** evaluator that splits the chain at the
chemistry/radiative-transfer boundary, because the two halves have opposite
economics. The chemistry loop has tiny per-lane state, on the order of megabytes,
so width is nearly free and is what keeps the GPU busy, but it only supports
`jvp`. The ExoJax PreMODIT radiative transfer is `vjp`-capable but costs about a
gigabyte of intermediates per lane.

Each mutation sweep therefore runs:

1. **Chemistry.** Six forward-mode lanes per particle, with all particles in one
   wide batched `while_loop`. At the production preset that is 864
   tangent-augmented columns. The warm convergence diagnostic rides this same
   chain, because it is part of the runner's primal carry and is integer-valued and
   therefore tangent-free. An earlier version ran a second primal-only loop just
   to read it, which doubled the chemistry wall time per sweep.
2. **Radiative transfer.** One reverse-mode pass per particle, which is legal
   because there is no `while_loop` inside the radiative transfer. It is chunked
   over particles. The single backward pass replaces six forward tangents plus the
   three-dimensional radius-and-cloud forward Jacobian, and its cotangent is
   contracted against the chemistry tangents. That is the same chain rule
   regrouped, and the smoke test asserts it is identical to `block`.
3. **Offsets and noise.** Analytic.

### Chemistry mode: cold is the default

`smc_chem_mode` selects how each likelihood evaluation gets its chemistry.

**`"cold"` is the default (since 2026-08-03).** Every evaluation runs the
published solve-from-baseline two-stage map, so the likelihood is a fixed,
deterministic function of theta. That is what MALA, SMC tempering, and a quoted
Bayesian evidence all assume, and it is the only mode whose `logZ` should be
reported without qualification.

**`"warm"` is an explicit opt-in for exploration and cost-limited work.** It is
~10-30x cheaper per sweep and remains fully supported, with two consequences:

- every artifact it produces (checkpoint, samples, extras) carries
  `approximate_history_dependent_target=1`, and `plot_smc` refuses to render
  posterior figures for it without `PLOT_SMC_ALLOW_UNCERTIFIED=1`;
- both post-run validators become MANDATORY before its numbers may be
  reported: `retrieval_framework.validate_warm` and
  `validation/mala_reversibility.py`. The PBS wrapper propagates their result
  into the job exit status, and `retrieval_framework.certificate` refuses a warm
  run that lacks either.

Cost is a real constraint, not a footnote: a cold ladder at the W39b production
size is expected to need more than one 24 h job. That is the supported route
(`RESUME=1` continues from the stage checkpoint, and the init-level checkpoint
means a restart never re-pays the init), and `run_smc --calibrate` now REFUSES
rather than warns when the projection does not fit the governor.

### Warm continuation

Under `smc_chem_mode="warm"` the mutation kernel carries each particle's
converged column. Every proposal's chemistry warm-continues from the particle's
own state with incremental metallicity and C/O scaling, rather than re-running
the full cold two-stage solve.

Measured cost for MALA-sized moves is roughly 500-800 steps to re-converge, which
is 6-8x fewer chemistry steps than cold. The certification window dominates that
warm floor, not the minimum step count.

**Do not shrink the certification window to buy speed.** Probing 500 to 300 in
2026-07-10 saved nothing on extrapolated seeds and 7% on plain ones, while
certifying a state 0.07 dex away from the 500-window result.

In warm mode the cold two-stage map runs exactly once per particle, at state
initialization.

The carried likelihood also serves the tempering reweight, so a stage costs about
one mutation call.

**The documented caveat, and why cold is now the default:** with warm
continuation the likelihood is defined by the continuation map from the
particle's own history, so it is path-dependent at the convergence-tolerance
level. A history-dependent target is not the fixed density the sampler and the
evidence integral assume, and no amount of post-hoc diagnostics converts an
approximate evidence into an exact one -- the diagnostics can only bound how
approximate it is. The smoke test finite-difference-checks the warm gradient
against the identical warm map, which validates the implementation, not the
target.

#### Rejecting non-converged proposals

A MALA proposal whose warm continuation hits the step cap is treated as a
Metropolis-Hastings rejection with `L = -inf`, and is never fed into the gradient.
This is the warm-side analogue of the cold-initialization reject-and-cull. Without
it, a non-converged proposal's meaningless tangents tripped the bad-gradient
counter or produced NaN at the first SMC stage.

Mutation solves run under `warm_count_max`, default 1500, using a twin runner with
the smaller cap baked in. A proposal still unconverged there is rejected rather
than dragging the whole lockstep batch to the cold cap. Without this, any single
bad proposal among N gated **every** early-ladder sweep at the full cold cap,
which is a 3-6 hour per stage pathology, and while the cloud is still prior-like
that is essentially every sweep.

#### The initialization gradient pass is uncapped

Phase 2 of initialization runs the same warm map **uncapped**, gated at the cold
step cap. Its inputs are phase-1 survivors re-certifying from their own converged
columns. Those are proven-convergent particles, not disposable proposals, and a
marginal survivor can legitimately need more than the mutation cap just to
re-certify; in one job the cap gated 5 of 96 healthy survivors into a spurious
failure.

Some marginal columns cannot re-certify even at the cold cap, because
oscillating and stall-fallback certifications re-pay the time-based window on
restart and lose. So phase 2 evaluates `N + init_phase2_spare` survivors and
**culls** the re-certification failures, backfilling from the spares. This is the
same reject-rather-than-carry philosophy as phase 1, and it is logged as part of
the operational prior. A true forward-model or gradient failure at phase 2, which
is a non-finite forward with a non-exhausted step count, still raises.

#### Warm extrapolation

`warm_extrapolate=true` is opt-in. It seeds each proposal's warm solve at the
first-order prediction from the tangents the gradient pass has already computed.
Measured effect is 1.65x fewer warm steps to the same certified state, with parity
unit-tested: about 780 steps down to about 470.

It remains opt-in pending a synthetic A/B test. Note that a clipped extrapolated
seed can manufacture the blown-tangent failure class; that investigation is
recorded in the negative-results register.

### The sampler

`pipeline.run_smc_loop` is a Del Moral resample-move SMC:

- **Temperature ladder** by effective-sample-size bisection, targeting 0.6 of the
  cloud.
- **Systematic resampling**, then `smc_num_mcmc_steps=6` preconditioned-MALA
  sweeps per stage. Published MALA-within-SMC practice is 3-10 with a good
  preconditioner; the earlier value of 12 was about twice as generous as needed,
  and each sweep costs one full batched gradient.
- **Preconditioning** by the absolute per-dimension standard deviation of the
  freshly resampled cloud, so the proposal narrows in lockstep with tempering. A
  Gaussian test showed that unit-geometric-mean scaling collapses acceptance after
  large temperature jumps.
- **Robbins-Monro tuning** of the scalar step toward 0.55 acceptance.
- **A heartbeat line per sweep** reporting acceptance, rejected-proposal count,
  and bad-gradient count, so a slow stage is visible per sweep instead of hours of
  silence.
- **Per-stage atomic checkpointing** plus a **walltime governor** that stops
  cleanly inside the job wall, which makes a usable posterior a guarantee rather
  than a hope. `RESUME=1` continues a stopped ladder.

There is deliberately **no gradient-free kernel fallback**. The project rule is
loud errors over silent degradation.

The SMC core is about 200 lines of pure JAX with no BlackJAX dependency, and it is
validated against an analytic Gaussian posterior including its evidence estimate.
An opt-in external-oracle test (`RUN_BLACKJAX_ORACLE=1 python -m pytest
tests/test_smc_blackjax_oracle.py`, dev-only `blackjax` install) additionally pins
the core's log-evidence against BlackJAX's adaptive tempered SMC on the identical
target; run it after any change to the sampler core. The 2026-07-28 audit measured
the two indistinguishable over 24 seeds (two-sample t, p = 0.275).

### Particle count and what actually scales

The chemistry, both primal and gradient, runs at full width, so raising the
particle count widens those kernels nearly for free. The GPU power trace confirms
it: wattage rises with lane count while step time barely moves. The production
preset ships 144 particles, raised from 96 to spend the measured power headroom on
particles.

The **only** cost that is linear in particle count is the radiative-transfer
reverse pass, which runs as sequential chunks. Per-lane memory there scales with
`nu_pts`, so at the production resolution the preset runs 12 lanes wide.

Shrinking that per-lane cost, through checkpoint granularity, cross-section
tables, or reduced precision inside `exojax_rt`, is the top remaining structural
speed item, because it would unlock a full-width radiative-transfer pass.

One documented-but-unwired speedup remains: VULCAN-JAX's reverse-mode
steady-state adjoint, which would make the chemistry gradient cost independent of
the parameter count.

### The context for these choices

As of a 2026-07 literature review, no published kinetics retrieval uses gradients.
The prior full-kinetics retrievals are gradient-free nested sampling at 5-10
parameters, costing on the order of 180 CPU cores for 24 hours in one case and
about 874,000 CPU-hours in another; another study describes retrieval with
photochemistry as computationally impractical. Published gradient-based
retrievals use free-chemistry forward models that run in milliseconds rather than
a stiff kinetics solver. No published SMC atmospheric retrieval was found over
2022-2026.

The supporting citations and precedent map are preserved in the development log.

## The WASP-39b production case

Operational reference for `runs/w39b_smc_retrieval/`: what the case directory
holds, the current numerics, the submit sequence, and the GPU budget.

The case fits the real Carter & May (2024) combined JWST transmission spectrum of
WASP-39b, NIRISS SOSS plus NIRSpec G395H, 152 bins over 1.02-5.24 um.

### What is in the case directory

| Path | Contents |
|---|---|
| `case.py` | Planet identity (gravity, radii, chemistry config module, data product table) plus the `smoke`, `gpu`, and `prod` presets |
| `run_nas_w39b.pbs` | The GPU submit script. All modes; the header documents the knobs |
| `overrides/*.json` | Optional config-override files, resolved against the case directory |
| `data/<preset>/` | Run outputs: posterior archives, resolved config, run log, plots |
| `logs/` | Live job logs, GPU monitor output, profiler reports |

### Current numerics

Every run prints these in a configuration banner via
`config_schema.describe_config`.

- **Convergence criteria are the canonical VULCAN values** (Tsai et al. 2017):
  `yconv_cri=0.01` and `slope_cri=1e-4`. These are deliberately not the
  sensitivity demo's tighter `1e-3`.
- **`count_max=5000` accepted steps, fixed.** A solve that does not converge is a
  failed draw. It is not extended and not clipped.
- **`dt_max=1e11` s**, not the VULCAN 2.0 default of `1e17`. The default let the
  adaptive step balloon to about 1e16 s on high-`Kzz` columns and spin without
  settling, which was the bulk of the old long tail. Capping it converges those in
  about 1000 steps and leaves normal columns identical. This is a step-size
  control, not a convergence criterion. A genuine residual, meaning marginal
  convergence-metric and photochemical-limit-cycle columns, still fails and is
  rejected at initialization.
- **`nu_pts=1652`, about R=1000 native, memory-safe by default.**
  Radiative-transfer gradient memory scales with the point count, not with R, and
  at `nu_pts=16500` it demanded 343 GiB and exhausted a 96 GB GPU. About 11 model
  points per binned point is ample and keeps the reverse pass near 34 GiB.
  `validate_config` warns above 2500. Run the memory probe before ever raising it.
- **Calibration runs at native R=100 by default**, because the accepted-step count
  is resolution-independent, so the radiative transfer can be run cheaply.
  Production uses the real `nu_pts`.

### Submit sequence

Local smoke first, offline and on CPU, after any framework change:

```bash
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval
```

Then, from the case directory on the GPU host, in this order for a fresh campaign:

```bash
qsub -v PROBE_MEMORY=1 run_nas_w39b.pbs     # required after any N, chunk, or nu_pts change
qsub -v CALIBRATE_ONLY=1 run_nas_w39b.pbs   # ~1.5 h; check the mutation-sweep time
qsub -v SYNTH=1 run_nas_w39b.pbs            # synthetic recovery at production fidelity
qsub run_nas_w39b.pbs                       # real-data production
qsub -v RESUME=1 run_nas_w39b.pbs           # continue a governor-stopped ladder
```

On success, the plots and the warm-versus-cold validation run automatically.

**Two gates before trusting a real-data posterior.** One clean synthetic recovery
at production fidelity, with the injected truths inside their 90% intervals, and a
passing verdict from the automatic warm-versus-cold validation. Quote both,
together with the initialization reject fraction.

### Deployment

Code deploys by `git pull --ff-only` into two clones under the scratch tree, one
for this repo and one for VULCAN-JAX. Both clone target names are load-bearing.

Data is seeded **once** per fresh clone, for the opacity cache and the line lists,
by moving an existing tree or a one-time copy. Never rsync.

Environment setup is a one-time bootstrap job, `qsub tools/bootstrap_nas_env.pbs`
from the repo root. It editable-installs both packages into a per-interpreter user
site, installs the pinned exojax under a jax and jaxlib constraint, builds FastChem
for the node architecture, and validates.

The run script reuses the shared environment, is read-only on it, validates it with
`python -m retrieval_framework.validate_env`, which is re-runnable by hand at any
time, and exports `VULCAN_PROJECT_ROOT` so the data paths resolve. Re-run the
bootstrap only when `validate_env` says to, meaning the packaging metadata changed.

### GPU budget

Cost is approximately `init + stages x mutation call`. The reweight uses the
carried likelihood, so it is free. The adaptive ladder typically needs 12-25
stages.

**Initialization** is one batched cold two-stage solve per particle,
likelihood-only at full width, followed by one warm-map gradient sweep from the
just-converged columns. That ordering matters: the older cold-gradient
initialization cost more than 16 hours. The cold pass is the only
solve-from-baseline work in the run, and it is gated by the slowest prior-corner
particle.

**A mutation call** is six MALA sweeps, each one wide batched warm chemistry
re-converge with six tangent lanes per particle, capped at `warm_count_max`, plus
the chunked radiative-transfer reverse passes. The warm cap is what bounds the
early ladder.

Run `CALIBRATE_ONLY=1` after any config change. It times initialization plus one
mutation call and projects 15, 25, and 40 stages.

The walltime governor makes the budget a guarantee rather than a hope: the
production preset stops cleanly at 20 hours inside a 24-hour wall.

### Speed levers, in order of pain

If calibration says a stage is slow:

1. **`warm_extrapolate=true`.** Seeds each proposal's warm solve from the
   first-order prediction using tangents the gradient pass already computes.
   Measured 780 to about 470 typical warm steps, 1.65x, to the same converged
   column. Validate once with a synthetic A/B, then also drop `warm_count_max`
   from 1500 to about 800 for the second half of the win.
2. **`warm_count_max` 1500 to about 1000.** Bounds the worst-case sweep. Typical
   plain warm re-converge is 500-800 steps, so watch the per-sweep rejected count
   in the heartbeat lines for collateral rejections.
3. **`smc_num_mcmc_steps` 6 to 4.** Linear savings, some mixing loss.
4. **`nz` 50 to 40.** Chemistry cost is roughly linear in layer count.
5. **Band: drop to G395H only.** Halves the bin count, but loses the NIRISS lever
   and the offset parameter.

`yconv_cri` is not a speed lever here. It is fixed at the canonical `0.01`, the
operative gate is the loose branch anyway, and the real step-count lever was
`dt_max`.

The validation scripts to run before a production retrieval are listed under
[Tests and validation](#tests-and-validation).

### Companion analyses

**Sensitivity figures** (`examples/`) answer which wavelengths best constrain each
parameter, an observation-planning view. Forward-mode tangents of the transit
depth with respect to metallicity, C/O, `Kzz`, and a temperature offset color the
spectrum by each derivative. The key result is that metallicity is best measured
in the 4.0-4.3 um SO2 and CO2 band, which is the window JWST uses. `run_demo.py`
is the headline figure; `run_figs.py` produces wide-band transmission and emission
versions sharing the chemistry tangents.

**Metallicity versus C/O information** (`scripts/zco_information/`) asks how much
unique information the spectrum carries about each, and how much of it comes from
disequilibrium rather than equilibrium. It is a Fisher and Laplace analysis on the
autodiff Jacobian of the real spectrum, with a true fixed-oxygen C/O knob,
marginalizing `Kzz`, a temperature offset, a reference-radius nuisance, and
per-instrument offsets, and it compares equilibrium, quench, and photochemistry
tiers.

Its documented toy limits: the analysis is local-linear and Gaussian, with no
clouds, no free temperature-pressure profile, and no stellar contamination.
**Absolute uncertainties are therefore best-case lower bounds**, but the relative
statements — which wavelengths, which chemistry tier, which parameter combination
is degenerate — are robust. The equilibrium tier drifts about 3% with molecular
diffusion off.

Build order from the repo root: `build_zco_jacobians.py`, which takes `--smoke`
for a fast check, then `build_zco_walk.py`, then the three figure scripts. Caches
predating 2026-07-11 were deleted as stale; regenerate before rebuilding figures.

## Limitations

Every known limitation of the framework, with its measured scope. Read this before
publishing a number.

### Clouds are parametric

The cloud is ExoJax's power-law deck: a gray deck at zero slope, a haze slope
otherwise, uniformly mixed. It is not microphysical.

The ExoJax-native upgrade is the Ackerman & Marley stack, with the cloud base at
the retrieved profile's enstatite saturation crossing and particle sizes from the
**retrieved** `Kzz`. That would be an honest self-consistent-lite treatment. It is
blocked only by a missing Mie-scattering package in the environments plus a
one-time lookup-grid build.

Note that VULCAN itself **cannot** do WASP-39b clouds self-consistently. Its
condensation set is H2O, NH3, H2SO4, S2, S4, S8, and C, which are cool-planet
condensates, and Mg and Si are not in its atom set. Silicate condensation would be
major new chemistry, and the gradient path is validated with condensation off.

### No stellar contamination term

The host is a quiet G8 star, and the instrument offsets absorb residuals. This is
a scope limit, not a claim that the transit light source effect is negligible in
general.

### Opacity fidelity

HITRAN at the 296 K reference is used for H2O, CO2, CH4, SO2, HCN, C2H2, and H2S;
ExoMol for CO. This is adequate for the methodology but under-represents hot bands
at the retrieved limb temperature.

Swapping to HITEMP or ExoMol is one dictionary entry in the code. Operationally it
means multi-gigabyte downloads and premodit memory tuning, and it is the one
upgrade with real wrangling risk.

### The chemistry structure follows the retrieved profile, with one exception

As of 2026-07-11 the runner refreshes the hydrostatic geometry in the loop from
step 1, and rebuilds the molecular-diffusion coefficients, the convergence gate's
`Kzz`, and the initial carry geometry per proposal on the graph.

The one remaining baseline-temperature bake is the photolysis cross-section
temperature interpolation, which is a host-side upstream step and second order.

### Condensation blocks gradient-based inference

Condensation is a forward-model capability. Gradient-MALA inference with
`use_condense=True` is refused, in `validate_config` and again on the fully
resolved config, because the pinned condensation state is not reliably
differentiable: the tangent disagrees with re-converged finite differences at
order unity. The disagreement is worse than any single relative-error figure
suggests. Re-measured 2026-07-29 by sweeping the finite-difference step, the
gas/condensate split has no derivative at all: its centred difference scales
like `1/dT` (the signature of a fixed-size jump, since the `fix_species` pin
captures the column at a discrete accepted step), so a quoted relative error is
really a statement about the step size chosen. Only the conserved reservoir
total (gas + condensate) is step-stable, and the tangent reproduces that to
18-22%. See the condensation contract in VULCAN-JAX `README.md` (Differentiability section).

### Tempered output when a run stops early

If the walltime governor or a crash stops the ladder before the temperature
reaches 1, the samples are **tempered** and the reported widths are lower bounds.

`reached_beta1=False` travels in both `posterior_samples.npz` and
`smc_extra_fields.npz`. Every figure — corner, spectrum, temperature-pressure — is
stamped, and the posterior-predictive and recovery paths warn.

Resubmit with `RESUME=1` and the ladder continues from the checkpointed cloud and
temperature instead of restarting. This is validated in the Gaussian test.

### Observations are baked in at first trace

`set_observations` must be called exactly once, before inference, because the
observations are baked into the jitted likelihood at first trace. The driver
enforces this ordering.

### Photochemistry must be on

`use_photo=True` is required. The forward-mode tangent is validated only in the
photochemistry-on regime. See the full notes in `vulcan_forward/constants.py`.

### The likelihood is diagonal

The likelihood is a per-bin Gaussian, so bin-to-bin covariance is neglected.

This is common for reduced R≈100 products that supply only per-bin uncertainties,
and here wavelength covariance is not included **because the data product supplies
none**. That is a scope limit of the available product, not a claim that covariance
is universally irrelevant: published JWST and NIRISS analyses have found
significant wavelength-correlated noise and have constructed spectral covariance
matrices for retrieval.

An optional global multiplicative noise-inflation nuisance is available
(`infer_noise_inflation`, off by default). There is no full or correlated
covariance term. Correlated-systematics forecasting lives in the sibling
vulcan-jwst-tool planner and is not ingested here.

### Binning does not match the planner's

The native-to-bin operator in `observations.py` is the wavelength-space,
width-weighted trapezoidal average that reduced R≈100 products are compared
against. It is **not** the stellar-count-weighted operator, with a native-resolution
line-spread function, that the vulcan-jwst-tool planner uses to forecast
instrument noise.

The two tools are deliberately **not** a matched injection-recovery closure pair.
This framework fits real reduced spectra; it does not fit planner-generated
synthetic data.

### Evidence semantics

`smc_logZ` is the evidence under the **operational** prior: the declared box
restricted to the modelable temperature window and to draws whose chemistry
converges, then renormalized.

The support fraction is measured at initialization, with binomial counts persisted
through the checkpoints. `smc_logZ_box = smc_logZ + ln f_support` is the box-prior
value, with the non-evaluable region assigned zero likelihood.

Quote them together. **Never compare `smc_logZ` across models whose support
fractions differ.**

### Warm continuation is path-dependent

With warm continuation the likelihood is defined by the continuation map from each
particle's own history, so it is path-dependent at the convergence-tolerance
level. The smoke test finite-difference-checks the warm gradient against the
identical warm map, and `validate_warm` measures the effect directly by re-solving
a finished run's cloud cold — comparing likelihoods, binned spectra, elemental
inventories, and (since 2026-07-29) the u-space gradients that drive the MALA
proposals.

### Some prior-corner draws have a finite likelihood but no usable tangent

A particle can certify a finite likelihood and still return a non-finite
forward-mode tangent. This is expected in the high-metallicity, low-C/O corner of
the prior, and it is not a bug.

Those particles are **kept with their gradient entries zeroed**, which makes their
first move a zero-drift one, rather than being discarded. The initialization logs
the affected indices, so the count is visible per run. Treat a large count as a
signal that the prior reaches into a corner the chemistry cannot linearize, not as
a failed run.

A related interaction is why `warm_extrapolate` remains opt-in: a clipped
extrapolated seed can itself manufacture this class, and a per-cell fallback is
measurably insufficient. The forensic investigation behind both findings is in the
development log rather than here.

### Spectral resolution is a memory bound, not a convergence point

`nu_pts=1652` was chosen to fit GPU gradient memory. It is not a demonstrated
convergence point. Run `validation/resolution_ladder.py` before quoting few-ppm
numbers.

## Tests and validation

What the unit tests cover, what each validation script gates, and which ones to
run before a production retrieval.

### Unit tests

```bash
python -m pytest tests -q     # from the repo root; about 2 s
```

They cover: the binning matrix against a trapezoid reference, including the real
data bins; unit-space prior bounds, uniformity, and Jacobian; Gaussian SMC
recovery with its evidence, governor, and resume path; initialization
reject-and-cull with backfill; warm-cap rejection; warm-extrapolation parity; and
the warm-versus-cold validator.

### Gradient checks

```bash
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval
```

This is the gate after any framework change, and it takes 10-30 minutes. It
finite-difference-checks the end-to-end gradient, asserts that the `block` and
`naive` gradient modes agree to floating-point precision, and asserts that the
staged evaluator agrees with `block`.

It also hard-fails if the metallicity and C/O gradients go dead, which is the
regression that the two-stage solve exists to prevent.

### Offline pre-flight smokes

Laptop-safe. Run these before trusting any figure.

| Script | What it checks |
|---|---|
| `smoke_test.py` | End-to-end tangent against re-converged finite differences, CO only |
| `smoke_coref.py` | The reference continuation holds C/O fixed |
| `smoke_zco.py` | Chemistry-tier tangents and the fixed-oxygen knob |
| `validate_wide_chem.py` | Chemistry tangent against finite differences at 150 layers |

### The audit-response suite

Added 2026-07-11. Run these on the GPU node before the next production retrieval.

| Script | What it measures |
|---|---|
| `elemental_audit.py` | The per-draw elemental construction hits its targets exactly |
| `resolution_ladder.py` | Spectral-resolution convergence: binned depths and Jacobian columns against a `nu_pts` ladder |
| `top_pressure_ladder.py --extend-chem` | The clamped top decade, against chemistry actually solved to 1e-8 bar |
| `broadening_ab.py` | Terrestrial-air versus H2/He pressure broadening |
| `mala_reversibility.py` | Post-run kernel reversibility |

Run them from the repo root, for example `python validation/smoke_test.py`.

### Warm-versus-cold validation

```bash
python -m retrieval_framework.validate_warm runs/w39b_smc_retrieval
```

This re-solves a finished run's checkpointed cloud **cold** and compares against
the warm-carried results. It is the direct measurement of warm-continuation
history dependence.

It gates on all four axes:

- maximum log-likelihood difference below 0.1,
- binned-spectrum agreement within 5 ppm,
- elemental-inventory agreement,
- warm-versus-cold **gradient** agreement (the MALA drift): the cold re-solve
  recomputes each particle's u-space gradient and reports the relative
  discrepancy and drift-direction cosine against the checkpoint's carried
  gradient. The current 0.1 relative threshold and zeroed-drift-fraction
  threshold are certificate failures, not warnings. A likelihood gate alone
  does not validate a gradient-driven kernel; this axis closes that hole.

Run it once per production run. A published retrieval should quote its verdict,
along with the prior convergence-acceptance fraction, which the initialization log
reports as the reject fraction. On the production case it runs automatically after
a successful job.

### External sampler oracle

```bash
pip install blackjax   # dev-only dependency
RUN_BLACKJAX_ORACLE=1 python -m pytest tests/test_smc_blackjax_oracle.py
```

Opt-in regression test (~30 s): the repo's tempered-SMC core and BlackJAX's
adaptive tempered SMC integrate the identical analytic Gaussian-box target, and
both log-evidences must agree with the exact value and with each other within
seed scatter. Run it after any change to the sampler core
(`pipeline.run_smc_loop`, `_make_mutation`, `_next_dbeta`,
`_systematic_resample_idx`, `make_uspace`). A 5% error injected into the
evidence increment fails it (verified at creation).

### Production-fidelity convergence results (`validation/results/`)

Archived, provenance-bearing results from the three checks that measure the
production choices which were made for reasons other than accuracy:

| artifact | script | what it decides |
|---|---|---|
| `resolution_ladder.*` | `validation/resolution_ladder.py --jacobian` | whether `nu_pts = 1652` (chosen for GPU gradient MEMORY) is spectrally converged |
| `top_pressure_ladder.*` | `validation/top_pressure_ladder.py --extend-chem` | whether the one-decade constant-VMR/isothermal ART clamp above the chemistry grid is a faithful stand-in for real chemistry |
| `broadening_ab.*` | `validation/broadening_ab.py` | whether terrestrial-air HITRAN widths are visible in an H2/He atmosphere at this precision |

Each run writes `<name>.json` (machine-readable, with full provenance) and
`<name>.md` (a short human summary). Commit both.

#### Run them

On the intended GPU node with the production data installation, from the repo
root:

```
python validation/resolution_ladder.py --jacobian
python validation/top_pressure_ladder.py --extend-chem
python validation/broadening_ab.py
```

The flags are not optional decoration. Without `--jacobian` the resolution check
certifies the depth only, not the gradient the sampler actually uses; without
`--extend-chem` the model-top check measures only that the CLAMP is internally
converged, which says nothing about whether the clamp matches chemistry. Both
scripts record that in the artifact rather than letting a partial run read as a
pass.

#### Decision rules

These are the rules the artifacts feed. Applying them is the point of running
the checks.

- **Resolution.** Keep `nu_pts = 1652` only if the depth gate (< 5 ppm against
  the next rung) AND the Jacobian gate (< 1% in significant bins) both pass.
  Otherwise adopt the lowest tested rung that passes. Do not invent a spectral
  algorithm; and re-run `PROBE_MEMORY=1` before raising `nu_pts`, because the
  RT-vjp memory scales with it.
- **Model top.** Keep the one-decade clamp only if the extended-chemistry
  comparison passes below 5 ppm. If it fails, set the production chemistry top
  to the already-tested extended value (`cfg P_t`). Do not add an extrapolation
  model.
- **Broadening.** If air vs H2/He differs by tens of ppm, select the existing
  `broadening="h2he"` mode for production molecules that have coverage, and fail
  loudly for a requested molecule that does not. Never mix a claim of H2/He
  broadening with all-air data, and do not fetch new databases as part of this.

#### Status

No artifacts are committed yet. Until they are, no few-ppm accuracy claim and no
converged-evidence claim is supported: a check that has not been run at
production settings is not a check that passed. The resolved production config
must cite these three files.

## Data: provenance and regeneration policy

Inputs for this repo live in `data/` (`config.DATA_DIR`); generated npz caches
go to `output/` (`config.OUTPUTS`, gitignored). Code resolves both roots via
`VULCAN_PROJECT_ROOT` (the directory containing the `vulcan-retrieval/`
checkout) or, in an editable checkout, by inferring the repo root from the
installed package location; `src/retrieval_framework/forward/config.py` raises
loudly when neither resolves. Data does not travel with pip installs on
purpose.

### Tracked in git

- `cm24_wasp39b/`: the real Carter & May (2024) WASP-39b transmission products
  (Zenodo 10161743, Fixed_LimbDarkening CSVs: NIRISS SOSS orders 1/2, NIRSpec
  G395H NRS1/NRS2, NIRCam, all R=100), plus `PRISM_native.csv` (the
  Rustamkulov et al. 2023 PRISM spectrum). Published measurements, small, the
  retrieval's observation source. Never regenerate; replace only with a newer
  published reduction.

### Gitignored seeded caches

- `exojax_linelists/` (~190 MB): HITRAN line-list caches. ExoJax re-downloads
  them on first use (through the NAS proxy on HPC). The `h2he` broadening knob
  adds separate `<db>_h2he` cache dirs on its first use.
- `opacity_cache/` (~170 MB): offline CO ExoMol Li2015 plus H2-H2 and H2-He CIA
  tables. Regenerable by download; the H2-He CIA file's canonical URL is
  `https://hitran.org/data/CIA/main/H2-He_2011.cia` (147 MB; note the `/main/`
  path segment, the bare `/data/CIA/` URL 404s).

### Generated caches: `output/`

The sensitivity, zco-jacobian, and zco-walk npz caches are GENERATED and live
in `output/` (`config.OUTPUTS`), gitignored. The previously tracked set was
deleted as stale on 2026-07-11 (every chemistry/spectrum cache predating the
scientific-correctness pass is invalid). Regenerate with
`examples/run_demo.py` / `run_figs.py` and
`scripts/zco_information/build_zco_jacobians.py` / `build_zco_walk.py`; never
re-track them.

### HPC seeding

Code deploys to the NAS by `git pull`; data does not ride along. Seed the big
caches ONCE into a fresh clone, either by moving them from an old
vulcan_exojax_run tree on /nobackup (`mv <old tree>/data/opacity_cache
vulcan-retrieval/data/`, same for `exojax_linelists`) or by a one-time scp
from local (see `CLAUDE.md` for the exact scp proxy command; never rsync,
never tarballs). The PBS preflight errors without `opacity_cache/`; a missing
`exojax_linelists/` just re-downloads through the proxy.

## Support

Open a [GitHub issue](https://github.com/imalsky/vulcan-retrieval/issues) for a
bug or question. Include the command, resolved configuration, software
versions, and full error message.

## License

This project uses the [GNU General Public License v3.0](LICENSE).
