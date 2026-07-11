# examples: sensitivity-demo notes

Findings recorded while validating the forward-mode jvp chain these scripts
exercise. The fragments below are verbatim; path references in the old README
fragment are historical.

## Tangent-validity requirements

From `../src/retrieval_framework/forward/config.py` (Run profiles section),
verbatim:

```
# Two non-obvious requirements, both about keeping the forward-mode tangent valid:
#   * Photochemistry must be ON. Only in the photo-on regime does the warm-started jvp
#     relax to the true steady-state sensitivity (validated: jvp vs re-converged FD <0.1%
#     at nz=150). With photo OFF the W39b column lands in a regime where the tangent is
#     under-relaxed/unstable.
#   * Let convergence happen naturally (default count_min/count_max). Do NOT pin a fixed
#     step count -- forcing dt to dt_max drives the Ros2 step's forward tangent singular.
```

## The sensitivity-demo science thread (old bundle README, verbatim)

### `sensitivity_demo/` — where does the spectrum's information live?
**Question:** which wavelengths best constrain each physical parameter (an
observation-planning "which instrument/band?" view).
**Method:** forward-mode `jvp` of the WASP-39b transit depth w.r.t. (lnZ, C/O, lnKzz,
T_int), coloring the spectrum by each `d(depth)/d(param)`.
**Assumptions:** the four y₀/profile knobs above; scalar-`T_int` temperature shift (a
proxy; the retrieval upgrades this to a full Guillot T-P via `vulcan_chem`'s `tp_eval`
hook). nz=150, 2.9–5.2 µm for the headline, 1–15 µm for the wide-band figure.
**Outputs:** `../jax_paper/figures/exojax_sensitivity.png` (the manuscript figure) +
`data/{sensitivity,wide_sensitivity}.npz`. Key result: metallicity is best measured in
the 4.0–4.3 µm SO₂/CO₂ band (the same window JWST uses).

## Cache status (2026-07-11)

The demo npz caches (`data/sensitivity.npz`, `data/wide_sensitivity.npz`)
predate the 2026-07-11 scientific-correctness pass (the silently skippable
H2-He CIA term biased them) and were deleted as stale; rerun `run_demo.py` /
`run_figs.py` to regenerate before rebuilding any figure.
