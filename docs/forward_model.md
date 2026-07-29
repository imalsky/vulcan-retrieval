# The shared forward engine

`retrieval_framework.forward` chains live VULCAN-JAX photochemical kinetics into
ExoJax radiative transfer and propagates gradients through the whole chain. It is
also the engine behind the sibling vulcan-jwst-tool. It imports VULCAN-JAX and
ExoJax and never modifies them.

Moved out of the README in 2026-07.

## Modules

Import as
`from retrieval_framework.forward import config, vulcan_chem, interp_map, exojax_rt, sensitivity`.

| Module | Physics it owns |
|---|---|
| `config` | Pure constants: molecule and isotope set with masses, wavenumber band, radiative-transfer pressure grid, planet defaults, cache paths. No heavy imports, so it is safe to load before the environment-sensitive VULCAN-JAX setup |
| `vulcan_chem` | The VULCAN-JAX side. One warm-up convergence, then `converged_ymix(theta)` and `converged_y(...)` re-converge the closed column as a function of metallicity, C/O, `Kzz`, and the temperature-pressure profile. Sets the network environment variables and float64 at import |
| `interp_map` | The differentiable log-pressure bridge from the chemistry grid to the radiative-transfer grid, using `jnp.interp` so tangents pass across it |
| `exojax_rt` | ExoJax `ArtTransPure` for transmission and `ArtEmisPure` for emission, sharing one opacity set. Provides `transmission_depth_r(...)` and `emission_flux(...)` |
| `sensitivity` | Composes the chain from parameters through the converged mixing ratios to the transit spectrum, for forward-mode gradients |

## The import-order contract

**Import `vulcan_chem` before anything from exojax.** It sets the import-frozen
`VULCAN_JAX_*` environment variables and enables float64 at import, and it
enforces the ordering with a `RuntimeError` if exojax is already in
`sys.modules`.

`config` is always safe to import first, and the `forward` package `__init__`
deliberately imports nothing. That keeps light consumers, such as config readers
and the jwst-tool GUI's cache interface, free of jax, vulcan_jax, and exojax side
effects.

## Cross-cutting assumptions

These hold for every consumer of the engine.

### Photochemistry is on

Only in the photochemistry-on regime does the warm-started forward-mode gradient
relax to the true steady-state sensitivity; this is validated against finite
differences to better than 0.1% at 150 layers. It is also what produces the SO2
that anchors the WASP-39b science.

### The column is closed

FastChem sets the equilibrium initial abundances at 10x solar and then stays
frozen and off the graph. The runner forgets the initial speciation except through
the conserved **elemental column totals**, so metallicity and C/O are expressed as
initial-abundance directions on those totals.

### Exact elemental abundance knobs

`abundance_mode="elemental"` is the default for production, retrieval, and the
jwst-tool since 2026-07-11. After the mask-scaled guess, the column is
renormalized to the hydrostatic number density per layer, then linearly repaired
on the runner's own reservoir species (He, H2O, CO, N2, H2S) so that the column
elemental ratios hit their targets **exactly**: He/H at the base value, O/H, N/H,
and S/H at metallicity times base, and C/H at metallicity times the C/O factor
times base. The conserved atom anchor `pv.atom_ini` is rebuilt from that column,
so cold and warm continuation share identical conserved inventories by
construction.

The legacy `"masks"` mode is kept for the published demo caches. Its elemental
leakage is documented in `forward/vulcan_chem.py`: about 0.6% of H per e-fold of
metallicity, plus N and S leakage through the fixed-oxygen compensation.

Verify the construction per draw with `chem.audit_init(theta)` or
`validation/elemental_audit.py`.

### Fixed-oxygen C/O

`co_mode="fixed_O"` scales the C-bearing species and compensates the O-only
carriers so that each layer's oxygen total is invariant. The guess is valid while
the compensation factor stays positive; the bound is printed at build time and the
retrieval prior is capped below it. In elemental mode the projection makes the
result exact regardless.

### Atmospheric structure follows the proposal

The runner refreshes the hydrostatic geometry (mean molecular weight, gravity,
scale height, layer thicknesses) in the loop from the live composition and the
proposal's temperature, every `update_frq` accepted steps, first firing at step 1.

Since 2026-07-11 the molecular-diffusion coefficients, the convergence gate's
`Kzz`, and the initial carry geometry are also rebuilt per proposal through the
on-graph builders. Since 2026-07-13 condensation is rebuilt per proposal too:
`_prep` regenerates the saturation number densities, growth terms, NH3 cold-trap
index, and fix-species saturation rows on the graph from the live temperature
profile.

The one remaining baseline-temperature bake is the photolysis cross-section
temperature interpolation, which is a host-side upstream step and second order.

**Condensation is a forward-model capability only.** Gradient-MALA inference with
`use_condense=True` is refused, in `validate_config` and again on the fully
resolved config in `retrieval_forward`, because the pinned condensation state is
not reliably differentiable: the tangent disagrees with finite differences at 0.91
relative. See the project-level `condensation_differentiation.md` contract.

### Opacities

CO comes from a cached ExoMol Li2015 line list. H2O, CO2, CH4, SO2, HCN, C2H2, and
H2S come from HITRAN, main isotopologue at the 296 K reference. Those intensities
carry the terrestrial isotopic abundance factor, and pairing them with the total
molecular mixing ratio is the standard, slightly conservative treatment.

HITRAN under-represents hot bands at the retrieved 770-1540 K limb compared with
HITEMP or ExoMol. This is a known accuracy limit for real-data inference. Swap the
source per molecule in `config.MOLECULES` when the multi-gigabyte downloads are
acceptable.

Pressure broadening defaults to terrestrial air. Set `config.BROADENING="h2he"`
for HITRAN's planetary H2/He widths where available, and measure the difference
with `validation/broadening_ab.py`.

H2-He collision-induced absorption is **required** physics. Every depth and flux
call takes the helium profile explicitly and raises if it is missing; a silent
omission biased the pre-2026-07-11 sensitivity caches.

The premodit table is baked for 300-3000 K. Profiles outside that window are not
modelable, which is why the retrieval rejects rather than clips.

### The radiative-transfer pressure grid

The grid spans 1e-8 to 7 bar. Its top sits one decade **above** the chemistry top
at 1e-7 bar on purpose: the interpolation clamps the topmost chemistry values over
that decade, a constant-abundance and isothermal extension that is a common
transmission convention rather than chemistry.

Without it, the strong CO2 4.3 um and CO 4.7 um bands saturate into a flat wall.
This is an explicit, logged modeling choice.
`validation/top_pressure_ladder.py` measures it against chemistry actually solved
down to 1e-8 bar.

### One self-consistent temperature-pressure profile

The profile is ExoJax's own `atmprof_Guillot` (Guillot 2010, Eq. 29). It is a
plain `jnp.exp`, so it is clean under forward mode, and it bypasses the
exponential-integral pathology in VULCAN's own `build_atm`.

The same analytic profile drives both the chemistry, on the VULCAN grid, and the
radiative transfer, on the ART grid. `src/retrieval_framework/tp_profile.py`
documents the precise scope of that consistency and the shape-parameter caveat.

### Native spectral resolution

`nu_pts` defaults to 1652, about R=1000 over the production band. That is a
GPU-gradient-memory bound, **not** a demonstrated convergence point.
`validation/resolution_ladder.py` is the convergence test: binned depths and
Jacobian columns against a ladder of `nu_pts`. Run it before quoting few-ppm
numbers.

## The two-stage solve

`two_stage_z` is on by default, and the reason is a measurement from 2026-07-05.
Perturbing the cold equilibrium initialization's metals and re-converging through
a temperature-displaced transient **erases the inventory perturbation**: converged
CO changed by 1e-11 for a 5% metals step under a Guillot profile, against the
exact 5% at the baked temperature. Any retrieved profile that differs from the
baked one triggers this.

So the forward model instead:

1. converges the column at the retrieved temperature and `Kzz` with the baseline
   composition, then
2. applies the metallicity and C/O scaling to that converged column and
   re-converges warm.

Stage 2 is cheap, because it warm-starts, and gentle, because the metal scaling is
uniform and no species crashes. The inventory survives, and with it the
metallicity and C/O gradients. `smoke_retrieval.py` hard-fails if those gradients
go dead again.

## Radiative-transfer contents

The transmission model carries eight molecules: H2O, CO2, CO, CH4, and SO2, plus
HCN, C2H2, and H2S. The last three are the high-C/O and reduced-sulfur
discriminators, included so the likelihood can see the species that decide the
C/O upper tail. H2-H2 and H2-He collision-induced absorption are always on.

The cloud is ExoJax's shipped `powerlaw_clouds`: opacity per gram of atmosphere,
uniformly mixed. Cloud dimensions are radiative-transfer only, so they ride the
cheap gradient block and cost almost nothing against the GPU budget.

H2 and He Rayleigh scattering is on by default and has no free parameters. It is
required once the band reaches 1 um, otherwise its slope leaks into the haze
posterior.
