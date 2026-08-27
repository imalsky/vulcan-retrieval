# Model-top treatment: clamped ART extension vs real chemistry

**VERDICT: FAIL** -- FAIL -- extend the production chemistry grid (cfg P_t) instead of clamping

Generated 2026-08-27T18:11:34Z by `top_pressure_ladder.py --extend-chem`.

## Measurements

| quantity | value | gate |
|---|---|---|
| clamp ladder, ART top 1e-07 -> 1e-08 bar | 75.28 ppm | (informational) |
| clamp ladder, ART top 1e-08 -> 1e-09 bar | 22.00 ppm | (informational) |
| clamped extension vs REAL chemistry over 1e-7..1e-8 bar | 73.47 ppm | < 5.0 ppm |

## Provenance

| key | value |
|---|---|
| jax-vulcan | cf993e24d0b6 |
| vulcan-forward | aedbcbc70a1b |
| vulcan-retrieval | 5f5eb37a5301 |
| vulcan-jwst-tool | 341aed2a94f2 |
| jax | 0.6.2 |
| jaxlib | 0.6.2 |
| numpy | 1.26.4 |
| exojax | 2.2.3 |
| python | 3.11.4 |
| devices | cpu:cpu |
| host | Alizas-MacBook-Pro-2.local (macOS-14.2.1-arm64-arm-64bit) |
| opacity_cache | 11 files, 178817677 bytes, newest 2026-07-07T23:52:00Z |
| exomolop | 26 files, 9724969441 bytes, newest 2026-08-25T19:48:40Z |
| resolved config sha256 | 615908f2153d30c2... |

Machine-readable: `top_pressure_ladder.json`
