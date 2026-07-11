# zco_information: science-thread rationale

The thread asks how much independent information the JWST spectrum carries
about Z vs C/O, and how much of it comes from disequilibrium chemistry. The
rationale below was moved verbatim from the pre-0.6 bundle README (paths
historical):

### `zco_information/` — how much *independent* info about Z vs C/O, and from where?
**Question:** the JWST spectrum constrains a Z–C/O combination; how much *unique*
information does it carry about each, and how much of it comes from disequilibrium
(quench + photochemistry) rather than equilibrium?
**Method:** Fisher/Laplace analysis on the autodiff Jacobian of the real Carter & May
2024 combined spectrum, with a true fixed-O C/O knob, marginalizing lnKzz, T_int, a
reference-radius nuisance, and per-instrument offsets. Compares equilibrium → quench →
photochem tiers.
**Assumptions (documented toy limits):** local-linear (Gaussian) Fisher; **no clouds,
no free T-P, no stellar contamination** — so *absolute* σ are best-case lower bounds,
but the *relative* statements (which wavelengths, which chemistry tier, which parameter
combination is degenerate) are robust. Equilibrium tier drifts ~3% (moldiff off).
**Outputs:** `../jax_paper/figures/zco_{information,disequilibrium,geometry}.png`,
`data/{zco_jacobians,zco_walk}.npz`. Guide: `../jax_paper/docs/ZCO_Guide.md`.

## Cache status (2026-07-11)

The npz caches this thread writes (`data/zco_jacobians.npz`,
`data/zco_walk.npz`) predate the 2026-07-11 scientific-correctness pass and
were deleted as stale. Regenerate with `build_zco_jacobians.py` (per-tier
Jacobians) and `build_zco_walk.py` (Gaussian-validity walk) before rebuilding
the figures. The old `fisher_forecast/` thread was removed the same day as
superseded by this directory plus vulcan-jwst-tool's live Fisher forecast.
