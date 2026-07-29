# vulcan-retrieval

vulcan-retrieval is a Bayesian atmospheric retrieval framework that uses
gradients through a full photochemical kinetics forward model.

The forward model chains live VULCAN-JAX photochemistry into ExoJax radiative
transfer. The sampler is adaptive-tempered Sequential Monte Carlo, and its
mutation kernel is MALA driven by forward-mode gradients through the whole
chemistry-to-spectrum chain. A 10-parameter retrieval on real JWST data runs on
one GPU inside 24 hours.

The distribution is named `vulcan-retrieval` and imports as
`retrieval_framework`. The package carries the retrieval framework, the shared
differentiable forward-model engine underneath it
(`retrieval_framework.forward`, also used by the sibling vulcan-jwst-tool), the
WASP-39b case, and the validation and information-analysis scripts.

```
physical params ─► VULCAN-JAX ─► VMR(nz, species), T(nz), P(nz)
  (lnZ, C/O, lnKzz,     (converged column, photochemistry ON)
   T-P params)                    │  log-pressure bridge (differentiable interp)
                                  ▼
                        ExoJax ArtTransPure ─► transit depth (Rp/Rs)²(λ)
                        (premodit lines + H2-H2/H2-He CIA + Rayleigh)
                                  │  jax.jvp
                                  ▼
                        d(spectrum)/d(param) ─► SMC + MALA
```

## Install

In a checkout, for development:

```bash
pip install --no-deps -e .
```

`--no-deps` is needed because the chemistry dependency `vulcan-jax` lives on
TestPyPI rather than PyPI. To install as a consumer:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple vulcan-retrieval
```

Inputs (observations and opacity caches) live in this repo's `data/` tree, and
generated caches go to `output/`. Both roots resolve from
`VULCAN_PROJECT_ROOT`, which is the directory containing this repo, or are
inferred from an editable checkout. `forward/config.py` raises a clear error when
neither resolves.

HPC runs pin the exact validated chemistry commit instead; see
`requirements-hpc.txt`.

## Quickstart

```bash
python -m pytest tests -q                                              # ~2 s
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval  # 10-30 min
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.plot_smc runs/w39b_smc_retrieval/data/smoke
```

Run every entry point from the repo root and give it a case directory:

| Command | Purpose |
|---|---|
| `run_smc <case>` | The retrieval driver. Add `--calibrate` to time it instead |
| `smoke_retrieval <case>` | Finite-difference and gradient-consistency checks |
| `calibrate_count_max <case>` | Choose the solver step cap from sampled draws |
| `probe_memory <case>` | Compile-only GPU buffer report |
| `plot_smc <run dir>` | Corner, spectrum fit, T-P, and diagnostics figures |
| `validate_warm <case>` | Re-solve a finished run's cloud cold and compare |

## Framework and case

Everything reusable lives in `src/retrieval_framework/`. Everything
planet-specific lives in a **case directory** under `runs/`, which holds
`case.py` (planet identity, priors, and presets), the submit script,
`overrides/`, and the run outputs.

| Path | Contents |
|---|---|
| `src/retrieval_framework/` | The planet-agnostic framework: config schema, observations, T-P profile, likelihood, SMC pipeline, driver, plotting, and tools |
| `src/retrieval_framework/forward/` | The shared forward engine: `config`, `vulcan_chem`, `interp_map`, `exojax_rt`, `sensitivity` |
| `runs/w39b_smc_retrieval/` | The WASP-39b case |
| `examples/` | Sensitivity figures: which wavelengths constrain which parameter |
| `validation/` | Physics and numerics validation scripts |
| `scripts/zco_information/` | Fisher and Laplace analysis of metallicity vs C/O information |
| `data/` | Observations, plus the seeded opacity and line-list caches |

The forward engine imports VULCAN-JAX and ExoJax and never modifies either. It has
one load-bearing ordering rule: **import
`retrieval_framework.forward.vulcan_chem` before anything from exojax.**

`vulcan_chem` sets the import-frozen `VULCAN_JAX_*` environment variables and
enables float64 at import. It raises if exojax is already imported.
`forward.config` is always safe to import first.

## The WASP-39b case

The shipped case fits the real Carter & May (2024) combined JWST transmission
spectrum of WASP-39b: NIRISS SOSS plus NIRSpec G395H, 152 bins from 1.02 to
5.24 um. The `gpu` preset samples 10 parameters, of which only the first six are
chemistry-expensive.

| # | Name | Prior | Role |
|---|---|---|---|
| 0 | `lnZ` | U(-2.303, 2.303) | Metallicity about the 10x solar baseline |
| 1 | `c_o` | U(-1.70, 0.24) | Change in ln(C/O) at fixed oxygen, giving C/O in [0.10, 0.70] |
| 2 | `lnKzz` | U(-4.6, 4.6) | Eddy-diffusion multiplier, +/-2 dex about the GCM baseline |
| 3 | `Tirr` | U(1100, 2200) K | Guillot irradiation temperature |
| 4 | `log10kappa` | U(-3.5, 0.5) | Guillot infrared opacity |
| 5 | `log10gamma` | U(-2, 0.301) | Guillot opacity ratio, allowing a weak inversion |
| 6 | `lnR0` | U(-0.08, 0.08) | Reference-radius nuisance |
| 7 | `log10kappa_cloud` | U(-7, 1) | Power-law cloud opacity at 3.5 um |
| 8 | `cloud_alpha` | U(0, 6) | Cloud slope, where 0 is a gray deck |
| 9 | `offset_G395H` | U(-800, 800) ppm | Flat depth offset against the NIRISS reference |

Priors are anchored to Tsai et al. (2023) and Rustamkulov et al. (2023) and live
in `case.py`. Temperature-pressure profiles are drawn raw and never clipped: a
profile that leaves the modelable 300-3000 K window is rejected and redrawn.

Before trusting a real-data posterior, require two things: one clean synthetic
recovery at production fidelity, with injected truths inside their 90% intervals,
and a passing verdict from the automatic warm-versus-cold validation. Quote both,
together with the prior convergence-acceptance fraction.

See [`docs/wasp39b_case.md`](docs/wasp39b_case.md) for the submit sequence, the
GPU budget, and the speed levers.

## What to know before you trust a result

Six assumptions hold for every consumer of the forward engine. Each one is a
modeling choice with a measured consequence, not a detail.

- **Photochemistry must be on.** The warm-started forward-mode gradient relaxes
  to the true steady-state sensitivity only in the photochemistry-on regime.
- **The column is closed.** Metallicity and C/O act on the conserved elemental
  column totals, because a converged column forgets its initial speciation.
- **Condensation is forward-model only.** Gradient-based inference with
  `use_condense=True` is refused, because the pinned condensation state is not
  reliably differentiable.
- **The radiative-transfer pressure grid extends one decade above the chemistry
  top**, and the interpolation clamps the topmost chemistry values over that
  decade. Without it the strong CO2 and CO bands saturate into a flat wall.
- **Opacities are HITRAN at the 296 K reference** for most molecules, which
  under-represents hot bands at the retrieved limb temperature. This is the main
  accuracy limit for real-data inference.
- **The likelihood is diagonal.** The data product supplies no bin-to-bin
  covariance, so none is used.

The full statements, with the measurements behind them, are in
[`docs/forward_model.md`](docs/forward_model.md). The complete limitations list is
in [`docs/limitations.md`](docs/limitations.md). Read both before publishing a
number.

## Why forward-mode gradients

VULCAN-JAX's integrator is a `lax.while_loop`. It supports `jvp` but not `vjp`, so
reverse mode cannot run through the loop. Forward mode is therefore the
end-to-end route, and it is the right shape for this problem: a few physical
scalars in, a high-dimensional spectrum out.

The sampler exploits the fact that the two halves of the chain have opposite
economics. The chemistry loop has tiny per-lane state, so width is nearly free,
but it only supports `jvp`. The radiative transfer is `vjp`-capable but costs
about a gigabyte of intermediates per lane.

Each mutation sweep therefore runs the chemistry as six forward-mode lanes per
particle, with all particles in one wide batched loop. It then takes one
reverse-mode pass per particle through the radiative transfer, chunked over
particles. Offset and noise gradients are analytic.

Details, including the warm-continuation scheme and its path-dependence caveat:
[`docs/gradients_and_sampler.md`](docs/gradients_and_sampler.md).

## Outputs

Each run directory holds the resolved `config.json`, the observations, a
per-stage atomic `smc_checkpoint.npz`, `posterior_samples.npz`,
`smc_extra_fields.npz` (temperature ladder, effective sample size, acceptance,
evidence and its conditioning), `posterior_predictive.npz`, and `plots/`.

Two output conventions matter when reporting. `smc_logZ` is the evidence under
the **operational** prior, which is the declared box restricted to the modelable
temperature window and to draws whose chemistry converges, then renormalized.
`smc_logZ_box` is the box-prior value with the non-evaluable region assigned zero
likelihood. Quote them together, and never compare `smc_logZ` across models whose
support fractions differ.

Separately: if a run stops before the temperature ladder reaches 1, the samples
are tempered and the widths are lower bounds. `reached_beta1=False` travels with
the samples, and every figure is stamped.

## Documentation

| File | Contents |
|---|---|
| [`docs/forward_model.md`](docs/forward_model.md) | The forward engine: modules, physics conventions, and cross-cutting assumptions |
| [`docs/gradients_and_sampler.md`](docs/gradients_and_sampler.md) | Gradient architecture, the staged evaluator, warm continuation, and the sampler |
| [`docs/wasp39b_case.md`](docs/wasp39b_case.md) | The production case: submit sequence, GPU budget, numerics, speed levers |
| [`docs/limitations.md`](docs/limitations.md) | Every known limitation, with its measured scope |
| [`docs/validation.md`](docs/validation.md) | The validation scripts and what each one gates |

## Citation and license

GPLv3, inherited from VULCAN. Cite the VULCAN 3.0 paper for the chemistry, and
ExoJax (Kawahara et al. 2022) for the radiative transfer.
