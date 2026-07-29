# The WASP-39b production case

Operational reference for `runs/w39b_smc_retrieval/`: what the case directory
holds, the current numerics, the submit sequence, and the GPU budget. Moved out of
the README in 2026-07.

The case fits the real Carter & May (2024) combined JWST transmission spectrum of
WASP-39b, NIRISS SOSS plus NIRSpec G395H, 152 bins over 1.02-5.24 um.

## What is in the case directory

| Path | Contents |
|---|---|
| `case.py` | Planet identity (gravity, radii, chemistry config module, data product table) plus the `smoke`, `gpu`, and `prod` presets |
| `run_nas_w39b.pbs` | The GPU submit script. All modes; the header documents the knobs |
| `overrides/*.json` | Optional config-override files, resolved against the case directory |
| `data/<preset>/` | Run outputs: posterior archives, resolved config, run log, plots |
| `logs/` | Live job logs, GPU monitor output, profiler reports |

## Current numerics

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

## Submit sequence

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

## Deployment

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

## GPU budget

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

## Speed levers, in order of pain

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

The validation scripts to run before a production retrieval are listed in
[`validation.md`](validation.md).

## Companion analyses

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
