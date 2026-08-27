# Vertical-grid (art_nlayer) convergence of the binned depth

**VERDICT: FAIL** -- opacity_mode=exomolop; art_nlayer ladder [60, 90, 135]; production is 60. NOT converged at the declared gates -- adopt the lowest tested passing rung as the production art_nlayer. Jacobian axis included.

Generated 2026-08-27T20:56:26Z by `resolution_ladder.py --jacobian`.

## Measurements

| quantity | value | gate |
|---|---|---|
| max \|Delta binned depth\|, art_nlayer 60 -> 90 | 1.41 ppm | < 5.0 ppm |
| max \|Delta binned depth\|, art_nlayer 90 -> 135 | 1.07 ppm | (informational) |
| max rel Jacobian (dlnZ) change, art_nlayer 60 -> 90 | 1.345% | < 1% |

## Provenance

| key | value |
|---|---|
| jax-vulcan | cf993e24d0b6 |
| vulcan-forward | fb244c761b35 |
| vulcan-retrieval | e4d590757343 |
| vulcan-jwst-tool | ecf0cf752b9e |
| jax | 0.6.2 |
| jaxlib | 0.6.2 |
| numpy | 1.26.4 |
| exojax | 2.2.3 |
| python | 3.11.4 |
| devices | cpu:cpu |
| host | Alizas-MacBook-Pro-2.local (macOS-14.2.1-arm64-arm-64bit) |
| opacity_cache | 11 files, 178817677 bytes, newest 2026-07-07T23:52:00Z |
| exomolop | 26 files, 9724969441 bytes, newest 2026-08-25T19:48:40Z |
| resolved config sha256 | 8476d69c62293250... |

Machine-readable: `resolution_ladder.json`
