# Limitations

Every known limitation of the framework, with its measured scope. Read this before
publishing a number. Moved out of the README in 2026-07.

## Clouds are parametric

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

## No stellar contamination term

The host is a quiet G8 star, and the instrument offsets absorb residuals. This is
a scope limit, not a claim that the transit light source effect is negligible in
general.

## Opacity fidelity

HITRAN at the 296 K reference is used for H2O, CO2, CH4, SO2, HCN, C2H2, and H2S;
ExoMol for CO. This is adequate for the methodology but under-represents hot bands
at the retrieved limb temperature.

Swapping to HITEMP or ExoMol is one dictionary entry in the code. Operationally it
means multi-gigabyte downloads and premodit memory tuning, and it is the one
upgrade with real wrangling risk.

## The chemistry structure follows the retrieved profile, with one exception

As of 2026-07-11 the runner refreshes the hydrostatic geometry in the loop from
step 1, and rebuilds the molecular-diffusion coefficients, the convergence gate's
`Kzz`, and the initial carry geometry per proposal on the graph.

The one remaining baseline-temperature bake is the photolysis cross-section
temperature interpolation, which is a host-side upstream step and second order.

## Condensation blocks gradient-based inference

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
18-22%. See the project-level `condensation_differentiation.md`.

## Tempered output when a run stops early

If the walltime governor or a crash stops the ladder before the temperature
reaches 1, the samples are **tempered** and the reported widths are lower bounds.

`reached_beta1=False` travels in both `posterior_samples.npz` and
`smc_extra_fields.npz`. Every figure — corner, spectrum, temperature-pressure — is
stamped, and the posterior-predictive and recovery paths warn.

Resubmit with `RESUME=1` and the ladder continues from the checkpointed cloud and
temperature instead of restarting. This is validated in the Gaussian test.

## Observations are baked in at first trace

`set_observations` must be called exactly once, before inference, because the
observations are baked into the jitted likelihood at first trace. The driver
enforces this ordering.

## Photochemistry must be on

`use_photo=True` is required. The forward-mode tangent is validated only in the
photochemistry-on regime. See the full notes in `forward/config.py`.

## The likelihood is diagonal

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

## Binning does not match the planner's

The native-to-bin operator in `observations.py` is the wavelength-space,
width-weighted trapezoidal average that reduced R≈100 products are compared
against. It is **not** the stellar-count-weighted operator, with a native-resolution
line-spread function, that the vulcan-jwst-tool planner uses to forecast
instrument noise.

The two tools are deliberately **not** a matched injection-recovery closure pair.
This framework fits real reduced spectra; it does not fit planner-generated
synthetic data.

## Evidence semantics

`smc_logZ` is the evidence under the **operational** prior: the declared box
restricted to the modelable temperature window and to draws whose chemistry
converges, then renormalized.

The support fraction is measured at initialization, with binomial counts persisted
through the checkpoints. `smc_logZ_box = smc_logZ + ln f_support` is the box-prior
value, with the non-evaluable region assigned zero likelihood.

Quote them together. **Never compare `smc_logZ` across models whose support
fractions differ.**

## Warm continuation is path-dependent

With warm continuation the likelihood is defined by the continuation map from each
particle's own history, so it is path-dependent at the convergence-tolerance
level. The smoke test finite-difference-checks the warm gradient against the
identical warm map, and `validate_warm` measures the effect directly by re-solving
a finished run's cloud cold — comparing likelihoods, binned spectra, elemental
inventories, and (since 2026-07-29) the u-space gradients that drive the MALA
proposals.

## Some prior-corner draws have a finite likelihood but no usable tangent

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

## Spectral resolution is a memory bound, not a convergence point

`nu_pts=1652` was chosen to fit GPU gradient memory. It is not a demonstrated
convergence point. Run `validation/resolution_ladder.py` before quoting few-ppm
numbers.
