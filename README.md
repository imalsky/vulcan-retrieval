# vulcan-retrieval

`vulcan-retrieval` fits exoplanet spectra with a live photochemical forward
model. It uses VULCAN-JAX for chemical kinetics and ExoJAX for radiative
transfer through the shared
[`vulcan-forward`](https://github.com/imalsky/vulcan-forward) package.

The sampler uses adaptive-tempered sequential Monte Carlo (SMC). Optional
MALA moves use forward derivatives from the atmosphere model. The repository
contains a WASP-39 b case, a small synthetic case, validation scripts, and a
run-certificate check.

This is research software. Validate a new target and model setup before using
the posterior in a publication.

## Install

Use Python 3.10 to 3.12 and a C++ compiler for FastChem.

```bash
git clone https://github.com/imalsky/vulcan-retrieval.git
cd vulcan-retrieval
python -m pip install \
  -i https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "vulcan-jax>=0.3.0" "vulcan-forward>=0.11.0"
python -m pip install -e ".[dev,plot]"
```

Opacity data are not stored in Git. Set `VULCAN_FORWARD_DATA`, then fetch the
needed ExoMolOP tables. CIA data go in `opacity_cache/`.

```bash
export VULCAN_FORWARD_DATA="$PWD/data"
python -m vulcan_forward.fetch_exomolop \
  --molecules H2O,CO2,CO,CH4,SO2,HCN,C2H2,H2S
python -m retrieval_framework.validate_env ..
```

`validate_env` takes the project root: the directory that contains the
`VULCAN-JAX`, `vulcan-forward`, and `vulcan-retrieval` checkouts side by side
(`..` when run from this repo).

## Test and run

Start with the fast tests and the gradient smoke check:

```bash
python -m pytest tests -q -m "not slow"
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval
```

Run the small SMC preset and plot it:

```bash
SMC_RETRIEVAL_PRESET=smoke \
  python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.plot_smc \
  runs/w39b_smc_retrieval/data/smoke
```

The `smoke`, `gpu`, and `prod` presets are defined in
[`runs/w39b_smc_retrieval/case.py`](runs/w39b_smc_retrieval/case.py). Copy the
case file to `runs/<new-case>/` for another target.

Each run writes its resolved configuration, checkpoints, posterior samples,
diagnostics, predictions, and plots to the case data directory. Check
`reached_beta1` before treating samples as posterior samples. Run
`python -m retrieval_framework.certificate` before reporting a production
result.

## Limits

- The likelihood is diagonal Gaussian.
- Spectral binning uses trapezoidal integration. No instrument line-spread
  function is applied, so a product whose resolution approaches the model band
  R is refused rather than modelled without it.
- The reported posterior is CONDITIONED, not just the evidence: draws whose
  T-P profile leaves the modelable window, or whose chemistry does not
  converge, are rejected and the target is renormalized over what remains.
  The figures carry the two surviving fractions.
- Condensation is refused during gradient inference.
- A certificate checks required artifacts and numerical gates. It does not
  prove that the physical model is complete.
- Results depend on the reaction network, opacity coverage, pressure and
  temperature profiles, clouds, priors, and data-reduction assumptions.

## Papers and citation

Published work should cite this repository and the model components used:

- VULCAN: [Tsai et al. (2017)](https://doi.org/10.3847/1538-4365/228/2/20)
  and [Tsai et al. (2021)](https://doi.org/10.3847/1538-4357/ac29bc)
- ExoJAX: [Kawahara et al. (2022)](https://arxiv.org/abs/2105.14782) and
  [Kawahara et al. (2025)](https://arxiv.org/abs/2410.06900)
- ExoMolOP tables: [Chubb et al. (2021)](https://doi.org/10.1051/0004-6361/202038350)
- FastChem initialization: [Stock et al. (2018)](https://doi.org/10.1093/mnras/sty1531)

Record the repository commit, package versions, data releases, reaction
network, priors, and full resolved configuration.

## Support and license

Open a [GitHub issue](https://github.com/imalsky/vulcan-retrieval/issues) and
include the case configuration, package versions, run log, and error message.

`vulcan-retrieval` is released under GPLv3.
