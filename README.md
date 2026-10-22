# vulcan-retrieval

`vulcan-retrieval` fits exoplanet spectra with a live photochemical forward
model. It uses VULCAN-JAX for chemical kinetics and ExoJAX for radiative
transfer through the shared
[`vulcan-forward`](https://github.com/imalsky/vulcan-forward) package.

The sampler is nested sampling with nautilus (`run_nautilus`, which the NAS
job script runs). The repository contains a WASP-39 b case (the production
run specification and a small synthetic one), validation scripts, and a run
report (`nautilus_report`) that writes a certificate and figures.

This is research software. Validate a new target and model setup before using
the posterior in a publication.

## Install

Use Python 3.10 to 3.12. The retrieval runs from checkouts of the three
repositories side by side, installed in editable mode (the `VULCAN-JAX`
directory name is required):

```bash
mkdir vulcan && cd vulcan
git clone https://github.com/imalsky/VULCAN3.git VULCAN-JAX
git clone https://github.com/imalsky/vulcan-forward.git
git clone https://github.com/imalsky/vulcan-retrieval.git
python -m pip install -e ./VULCAN-JAX -e ./vulcan-forward \
  -e "./vulcan-retrieval[dev,plot]"
cd vulcan-retrieval
```

Opacity data are not stored in Git. Set `VULCAN_FORWARD_DATA`, then fetch the
needed ExoMolOP tables. CIA data go in `opacity_cache/`.

```bash
export VULCAN_FORWARD_DATA="$PWD/data"
python -m vulcan_forward.fetch_exomolop \
  --molecules H2O,CO2,CO,CH4,SO2,HCN,C2H2,H2S,NH3,OCS,SH,SO
python -m retrieval_framework.validate_env ..
```

`validate_env` takes the project root: the directory that contains the
`VULCAN-JAX`, `vulcan-forward`, and `vulcan-retrieval` checkouts side by side
(`..` when run from this repo).

## Test and run

Start with the fast tests:

```bash
python -m pytest tests -q -m "not slow"
```

Run the small synthetic case and report it:

```bash
python -m retrieval_framework.run_nautilus runs/w39b_smc_retrieval/smoke.yaml
python -m retrieval_framework.nautilus_report \
  output/w39b_smc_retrieval/smoke_nautilus
```

A run is one YAML specification in `runs/<case>/`: `smoke.yaml` is the small
synthetic case and [`v5.yaml`](runs/w39b_smc_retrieval/v5.yaml) the WASP-39 b
production run (`--dry-run` prints the resolved settings). Copy a specification
to `runs/<new-case>/` for another target.

Each run writes its resolved configuration, target manifest, checkpoint, and
rejection tally to `output/<case>/<name>_nautilus/`, and when it finishes the
weighted posterior (`nautilus_posterior.npz`), a summary, and
`nautilus_certificate.json`.
`python -m retrieval_framework.nautilus_report <out_dir> [--spectra N]` writes
the certificate again and draws the corner, T-P, and spectrum figures (N cold
re-solved posterior draws, default 32; 0 skips the spectrum). The certificate
has no pass/fail verdict: read its caveats before reporting a production result.

The sampler is stock `nautilus-sampler` 1.0.6 from PyPI with its default
networks, and the exploration phase is discarded (`make_sampler` in
`run_nautilus.py`).
The padding rows of a short chunk are masked: they take no chemistry step and
no RT. The RT runs only on the rows whose likelihood can be kept. Each program
compiles at its first batch.

Each draw's chemistry is one solve at the specification's `rtol` (in
`cfg_overrides`) within `count_max` accepted steps. A draw is kept only if
VULCAN's own convergence test certifies that solve; an uncertified draw is
rejected, never extended. The WASP-39 b run uses rtol 0.05 within 5000 steps,
with photolysis every step: on the low-Kzz draws, the default photolysis cadence
set off the step-size collapses that stalled the solves. The studies, with
measurements, are summarized in `notes.md` ("Low-Kzz hole", "Design records").

## Limits

- The likelihood is diagonal Gaussian.
- Spectral binning uses trapezoidal integration. No instrument line-spread
  function is applied, so a product whose resolution approaches the model band
  R is refused rather than modelled without it.
- The reported posterior is CONDITIONED, not just the evidence: draws whose
  T-P profile leaves the modelable window, or whose chemistry does not
  converge, are rejected and the target is renormalized over what remains.
  The certificate records the rejected fractions.
- Condensation is refused during inference.
- The certificate records the run's settings, rejection fractions, and
  caveats. It has no pass/fail verdict and does not prove that the physical
  model is complete.
- Results depend on the reaction network, opacity coverage, pressure and
  temperature profiles, clouds, priors, and data-reduction assumptions.

## Papers and citation

Published work should cite this repository and the model components used:

- VULCAN: [Tsai et al. (2017)](https://doi.org/10.3847/1538-4365/228/2/20)
  and [Tsai et al. (2021)](https://doi.org/10.3847/1538-4357/ac29bc)
- ExoJAX: [Kawahara et al. (2022)](https://arxiv.org/abs/2105.14782) and
  [Kawahara et al. (2025)](https://arxiv.org/abs/2410.06900)
- ExoMolOP tables: [Chubb et al. (2021)](https://doi.org/10.1051/0004-6361/202038350)
- ExoGibbs equilibrium initialization: [ExoGibbs](https://github.com/HajimeKawahara/exogibbs)

Record the repository commit, package versions, data releases, reaction
network, priors, and full resolved configuration.

## Support and license

Open a [GitHub issue](https://github.com/imalsky/vulcan-retrieval/issues) and
include the case configuration, package versions, run log, and error message.

`vulcan-retrieval` is released under GPLv3.
