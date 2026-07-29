# Tests and validation

What the unit tests cover, what each validation script gates, and which ones to
run before a production retrieval. Moved out of the README in 2026-07.

## Unit tests

```bash
python -m pytest tests -q     # from the repo root; about 2 s
```

They cover: the binning matrix against a trapezoid reference, including the real
data bins; unit-space prior bounds, uniformity, and Jacobian; Gaussian SMC
recovery with its evidence, governor, and resume path; initialization
reject-and-cull with backfill; warm-cap rejection; warm-extrapolation parity; and
the warm-versus-cold validator.

## Gradient checks

```bash
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval
```

This is the gate after any framework change, and it takes 10-30 minutes. It
finite-difference-checks the end-to-end gradient, asserts that the `block` and
`naive` gradient modes agree to floating-point precision, and asserts that the
staged evaluator agrees with `block`.

It also hard-fails if the metallicity and C/O gradients go dead, which is the
regression that the two-stage solve exists to prevent.

## Offline pre-flight smokes

Laptop-safe. Run these before trusting any figure.

| Script | What it checks |
|---|---|
| `smoke_test.py` | End-to-end tangent against re-converged finite differences, CO only |
| `smoke_coref.py` | The reference continuation holds C/O fixed |
| `smoke_zco.py` | Chemistry-tier tangents and the fixed-oxygen knob |
| `validate_wide_chem.py` | Chemistry tangent against finite differences at 150 layers |

## The audit-response suite

Added 2026-07-11. Run these on the GPU node before the next production retrieval.

| Script | What it measures |
|---|---|
| `elemental_audit.py` | The per-draw elemental construction hits its targets exactly |
| `resolution_ladder.py` | Spectral-resolution convergence: binned depths and Jacobian columns against a `nu_pts` ladder |
| `top_pressure_ladder.py --extend-chem` | The clamped top decade, against chemistry actually solved to 1e-8 bar |
| `broadening_ab.py` | Terrestrial-air versus H2/He pressure broadening |
| `mala_reversibility.py` | Post-run kernel reversibility |

Run them from the repo root, for example `python validation/smoke_test.py`.

## Warm-versus-cold validation

```bash
python -m retrieval_framework.validate_warm runs/w39b_smc_retrieval
```

This re-solves a finished run's checkpointed cloud **cold** and compares against
the warm-carried results. It is the direct measurement of warm-continuation
history dependence.

It gates on three axes:

- maximum log-likelihood difference below 0.1,
- binned-spectrum agreement within 5 ppm,
- elemental-inventory agreement.

Run it once per production run. A published retrieval should quote its verdict,
along with the prior convergence-acceptance fraction, which the initialization log
reports as the reject fraction. On the production case it runs automatically after
a successful job.
