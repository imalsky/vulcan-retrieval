# Gradients and the sampler

How the likelihood gradient is assembled, why it is forward mode, how the
warm-continuation scheme keeps it affordable, and how the SMC sampler is
configured. Moved out of the README in 2026-07.

## Why forward mode

VULCAN-JAX's integrator is a `lax.while_loop`. `jvp` works through it; `vjp` does
not. So the likelihood gradient is assembled from forward-mode tangents.

That is also the right shape for this problem: a few physical scalars in, a
high-dimensional spectrum out. Reverse mode is available only at the converged
state, through VULCAN-JAX's reaction-importance adjoint, and through the
radiative transfer alone.

## Two per-particle gradient modes

`gradient_mode` selects between them. Both are kept for validation.

- **`block`, the default and exact.** Only the six chemistry and
  temperature-pressure directions push tangents through the VULCAN loop. The
  reference-radius parameter is one radiative-transfer tangent at the frozen
  converged profiles, and the offset and noise gradients are analytic. This is
  25-35% cheaper per MALA step.
- **`naive`.** All dimensions through the full chain, kept as a cross-check.
  `smoke_retrieval.py` asserts that `block` equals `naive` to floating-point
  precision and validates both against re-converged finite differences.

## The staged batched hot path

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

## Chemistry mode: cold is the default

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

## Warm continuation

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

### Rejecting non-converged proposals

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

### The initialization gradient pass is uncapped

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

### Warm extrapolation

`warm_extrapolate=true` is opt-in. It seeds each proposal's warm solve at the
first-order prediction from the tangents the gradient pass has already computed.
Measured effect is 1.65x fewer warm steps to the same certified state, with parity
unit-tested: about 780 steps down to about 470.

It remains opt-in pending a synthetic A/B test. Note that a clipped extrapolated
seed can manufacture the blown-tangent failure class; that investigation is
recorded in the negative-results register.

## The sampler

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

## Particle count and what actually scales

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

## The context for these choices

As of a 2026-07 literature review, no published kinetics retrieval uses gradients.
The prior full-kinetics retrievals are gradient-free nested sampling at 5-10
parameters, costing on the order of 180 CPU cores for 24 hours in one case and
about 874,000 CPU-hours in another; another study describes retrieval with
photochemistry as computationally impractical. Published gradient-based
retrievals use free-chemistry forward models that run in milliseconds rather than
a stiff kinetics solver. No published SMC atmospheric retrieval was found over
2022-2026.

The supporting citations and precedent map are preserved in the development log.
