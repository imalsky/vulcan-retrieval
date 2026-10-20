# vulcan-retrieval

`vulcan-retrieval` fits exoplanet spectra with a live photochemical forward
model. It uses VULCAN-JAX for chemical kinetics and ExoJAX for radiative
transfer through the shared
[`vulcan-forward`](https://github.com/imalsky/vulcan-forward) package.

The sampler is nested sampling with nautilus (`run_nautilus`, which the NAS
job script runs). The repository contains a WASP-39 b case with a small
synthetic preset, validation scripts, and a run report (`nautilus_report`)
that writes a certificate and figures.

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

Run the small preset and report it:

```bash
SMC_RETRIEVAL_PRESET=smoke \
  python -m retrieval_framework.run_nautilus runs/w39b_smc_retrieval
python -m retrieval_framework.nautilus_report \
  runs/w39b_smc_retrieval/data/smoke_nautilus
```

The `smoke` and `gpu` presets are defined in
[`runs/w39b_smc_retrieval/case.py`](runs/w39b_smc_retrieval/case.py). Copy the
case file to `runs/<new-case>/` for another target.

Each run writes its resolved configuration, target manifest, checkpoint, and
rejection tally to `<out_dir>_nautilus`, and when it finishes the weighted
posterior (`nautilus_posterior.npz`), a summary, and `nautilus_certificate.json`.
`python -m retrieval_framework.nautilus_report <out_dir> [--spectra N]` writes
the certificate again and draws the corner, T-P, and spectrum figures (N cold
re-solved posterior draws, default 32; 0 skips the spectrum). The certificate
has no pass/fail verdict: read its caveats before reporting a production result.

The sampler is stock `nautilus-sampler` 1.0.6 from PyPI with its default
networks, and the exploration phase is discarded (`make_sampler` in
`run_nautilus.py`).
The padding rows of a short chunk are masked: they take no chemistry step and
no RT. Each pass runs the RT only on the rows whose likelihood it can keep.
`NAUTILUS_PRECOMPILE=1` (the default) compiles the warm programs at the batch
width on a host thread while the cold first batch runs; `NAUTILUS_PRECOMPILE=0`
compiles each program at its first use.

`chem_presolve` and `chem_retry` (off by default; set together) turn the first pass into
a presolve: a chemistry-only solve at the base `rtol` and `count_max` whose
column is only a starting state (no RT, never a likelihood). Every in-window
draw is then solved and certified by the `chem_retry` program
(`chem_retry_rtol`, `chem_retry_count_max`) warm from that column, on one fixed
width of `n_batch` rows, so nothing compiles mid-run. Only that solve's
VULCAN certificate keeps a draw, and every kept column carries
`chem_retry_rtol`. Off, it stays out of the target manifest and nothing changes.
The WASP-39b setting is
`runs/w39b_smc_retrieval/overrides/w39b_baseline_v4_nz100.json`: presolve at
rtol 0.2, then rtol 0.05, both with photolysis every step. On the low-Kzz
draws, the presolve gets past two step-size traps that stall direct rtol-0.05
solves: photolysis-triggered step collapses, and a near-atol trace cell at the
model top. The certified solve then starts near its steady state. The study,
with measurements, is summarized in `notes.md` (lowkzz_fix).

## Limits

- The likelihood is diagonal Gaussian.
- Spectral binning uses trapezoidal integration. No instrument line-spread
  function is applied, so a product whose resolution approaches the model band
  R is refused rather than modelled without it.
- The reported posterior is CONDITIONED, not just the evidence: draws whose
  T-P profile leaves the modelable window, or whose chemistry does not
  converge, are rejected and the target is renormalized over what remains.
  The figures carry the two surviving fractions.
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
