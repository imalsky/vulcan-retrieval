# vulcan-retrieval

vulcan-retrieval runs exoplanet atmospheric retrievals with gradients from a
full photochemical forward model: VULCAN-JAX kinetics and ExoJAX radiative
transfer through the shared
[vulcan-forward](https://github.com/imalsky/vulcan-forward) engine, sampled
with adaptive-tempered SMC and MALA. It ships a reusable framework, a
WASP-39 b JWST case, a small synthetic case, and validation scripts. This is
research software; run the validation checks before publishing results.

## Install

Requires Python 3.10-3.12, ExoJAX 2.2.3, JAX (CPU or GPU), and a C++
compiler for FastChem. Clone the repository (the code finds `data/` and
`output/` from an editable checkout):

```bash
git clone https://github.com/imalsky/vulcan-retrieval.git
cd vulcan-retrieval
python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "vulcan-jax>=0.3.0" "vulcan-forward>=0.4.0"
python -m pip install -e ".[dev]"
```

Opacity data is not tracked. Retrievals use correlated-k tables from
[ExoMolOP](https://www.exomol.com/data/data-types/opacity/), fetched once with
`python -m vulcan_forward.fetch_exomolop --molecules H2O,CO2,CO,CH4,SO2,HCN,C2H2,H2S`
into `data/exomolop/`; the two CIA tables go under `data/opacity_cache/` (the
H2-He file comes from HITRAN). `validate_env` checks the whole setup.

## Quick start

```bash
python -m pytest tests -q -m "not slow"
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.plot_smc runs/w39b_smc_retrieval/data/smoke
```

Entry points (all take a case directory): `run_smc` (add `--calibrate` to
time one batch), `smoke_retrieval` (gradient checks), `calibrate_count_max`,
`probe_memory`, `validate_warm`, `plot_smc`, `validate_env`, and `certificate`
(the PASS/FAIL gate a run must clear before its numbers may be reported). The WASP-39 b
presets (`smoke`, `gpu`, `prod`) live in `runs/w39b_smc_retrieval/case.py`;
copy that file into a new `runs/<case>/` directory for a new target.
`SMC_RETRIEVAL_OVERRIDES` applies a temporary JSON configuration change.

## Outputs

Each run writes `runs/<case>/data/<preset>/`: the resolved `config.json`,
checkpoints, posterior samples, diagnostics, predictions, and `plots/`.
Check `reached_beta1` before treating samples as posterior samples.

## Limitations

The likelihood is diagonal Gaussian; binning is trapezoidal; condensation is
refused for gradient inference; the certificate gate refuses runs missing
their production-fidelity validation artifacts. The full limitation list
with measured scopes is kept in the maintainer's log.

## Support and license

Open a [GitHub issue](https://github.com/imalsky/vulcan-retrieval/issues).
GPLv3, inherited from VULCAN.
