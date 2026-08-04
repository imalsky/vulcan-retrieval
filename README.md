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
[`data/notes.md`](data/notes.md) for the data layout and provenance.

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
[`docs/wasp39b_case.md`](docs/wasp39b_case.md) covers the submit sequence, the
GPU budget, and the parameters and priors of the shipped case.

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
| `docs/` | Model, sampler, case, limitation, and validation documents |

The forward model has one ordering rule. Import
`vulcan_forward.vulcan_chem` before anything from ExoJAX, because
it sets the VULCAN-JAX environment variables and enables float64 at import time.
It raises an error if ExoJAX is already imported.

## Important limits

- The validated gradient path requires photochemistry.
- Gradient-based inference does not support condensation. The code refuses it
  and runs condensation as a forward model only.
- The likelihood uses independent Gaussian errors for each spectral bin. It
  does not use wavelength covariance.
- The cloud model is a simple power-law opacity model. It is not a
  microphysical cloud model.
- Most molecular opacities use HITRAN line lists at the 296 K reference
  temperature. Check the effect of the opacity source for precision work.
- A run that stops before `beta = 1` contains tempered samples, not posterior
  samples.
- The mutation kernel warm-starts each particle from its carried chemical
  state, so the sampled posterior is exact only up to the measured agreement
  between warm and cold solves. Run `validate_warm` after a production run and
  quote the result.
- Two evidence values are reported. `logZ` uses the operational prior, which
  removes draws outside the modelable temperature window and draws whose
  chemistry does not converge. `logZ_box` fills that region with zero
  likelihood. Do not compare `logZ` across models with different support
  fractions.

## Documentation

| File | Contents |
| --- | --- |
| [`docs/forward_model.md`](docs/forward_model.md) | Forward-model modules, physics conventions, and shared assumptions |
| [`docs/gradients_and_sampler.md`](docs/gradients_and_sampler.md) | Gradient architecture, warm continuation, and the SMC sampler |
| [`docs/wasp39b_case.md`](docs/wasp39b_case.md) | The production case: submit sequence, GPU budget, and speed levers |
| [`docs/limitations.md`](docs/limitations.md) | Every known limitation, with its measured scope |
| [`docs/validation.md`](docs/validation.md) | The validation scripts and what each one checks |

## Support

Open a [GitHub issue](https://github.com/imalsky/vulcan-retrieval/issues) for a
bug or question. Include the command, resolved configuration, software
versions, and full error message.

## License

This project uses the [GNU General Public License v3.0](LICENSE).
