# Production-fidelity convergence results

Archived, provenance-bearing results from the three checks that measure the
production choices which were made for reasons other than accuracy:

| artifact | script | what it decides |
|---|---|---|
| `resolution_ladder.*` | `validation/resolution_ladder.py --jacobian` | whether `nu_pts = 1652` (chosen for GPU gradient MEMORY) is spectrally converged |
| `top_pressure_ladder.*` | `validation/top_pressure_ladder.py --extend-chem` | whether the one-decade constant-VMR/isothermal ART clamp above the chemistry grid is a faithful stand-in for real chemistry |
| `broadening_ab.*` | `validation/broadening_ab.py` | whether terrestrial-air HITRAN widths are visible in an H2/He atmosphere at this precision |

Each run writes `<name>.json` (machine-readable, with full provenance) and
`<name>.md` (a short human summary). Commit both.

## Run them

On the intended GPU node with the production data installation, from the repo
root:

```
python validation/resolution_ladder.py --jacobian
python validation/top_pressure_ladder.py --extend-chem
python validation/broadening_ab.py
```

The flags are not optional decoration. Without `--jacobian` the resolution check
certifies the depth only, not the gradient the sampler actually uses; without
`--extend-chem` the model-top check measures only that the CLAMP is internally
converged, which says nothing about whether the clamp matches chemistry. Both
scripts record that in the artifact rather than letting a partial run read as a
pass.

## Decision rules

These are the rules the artifacts feed. Applying them is the point of running
the checks.

- **Resolution.** Keep `nu_pts = 1652` only if the depth gate (< 5 ppm against
  the next rung) AND the Jacobian gate (< 1% in significant bins) both pass.
  Otherwise adopt the lowest tested rung that passes. Do not invent a spectral
  algorithm; and re-run `PROBE_MEMORY=1` before raising `nu_pts`, because the
  RT-vjp memory scales with it.
- **Model top.** Keep the one-decade clamp only if the extended-chemistry
  comparison passes below 5 ppm. If it fails, set the production chemistry top
  to the already-tested extended value (`cfg P_t`). Do not add an extrapolation
  model.
- **Broadening.** If air vs H2/He differs by tens of ppm, select the existing
  `broadening="h2he"` mode for production molecules that have coverage, and fail
  loudly for a requested molecule that does not. Never mix a claim of H2/He
  broadening with all-air data, and do not fetch new databases as part of this.

## Status

No artifacts are committed yet. Until they are, no few-ppm accuracy claim and no
converged-evidence claim is supported: a check that has not been run at
production settings is not a check that passed. The resolved production config
must cite these three files.
