# Vertical-grid (art_nlayer) convergence of the binned depth

**VERDICT: PASS** -- opacity_mode=exomolop; art_nlayer ladder [67, 101, 151]; production is 67. The production resolution is converged at the declared gates. Jacobian axis included.

Generated 2026-08-27T21:12:01Z by `resolution_ladder.py --jacobian`.

## Measurements

| quantity | value | gate |
|---|---|---|
| max \|Delta binned depth\|, art_nlayer 67 -> 101 | 1.02 ppm | < 5.0 ppm |
| max \|Delta binned depth\|, art_nlayer 101 -> 151 | 0.68 ppm | (informational) |
| max rel Jacobian (dlnZ) change, art_nlayer 67 -> 101 | 0.933% | < 1% |

## Provenance

| key | value |
|---|---|
| jax-vulcan | cf993e24d0b6 |
| vulcan-forward | fb244c761b35 |
| vulcan-retrieval | 833862870ba9 |
| vulcan-jwst-tool | 8f6b8e543413 |
| jax | 0.6.2 |
| jaxlib | 0.6.2 |
| numpy | 1.26.4 |
| exojax | 2.2.3 |
| python | 3.11.4 |
| devices | cpu:cpu |
| host | Alizas-MacBook-Pro-2.local (macOS-14.2.1-arm64-arm-64bit) |
| opacity_cache | 11 files, 178817677 bytes, newest 2026-07-07T23:52:00Z |
| exomolop | 26 files, 9724969441 bytes, newest 2026-08-25T19:48:40Z |
| resolved config sha256 | 009ef6d973710e48... |

Machine-readable: `resolution_ladder.json`
