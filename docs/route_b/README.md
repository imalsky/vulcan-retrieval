# Route B smooth-rainout condensation — B0A artifacts

Tracked home of the B0A (physics decision) phase artifacts for the Route B
plan (`../../../docs/route_b_smooth_condensation_plan.txt`, umbrella
workspace). Status: B0A analysis and draft decision record complete; the
B0A gate remains open pending the D3b sign-off. B0B is blocked until then.

## Contents

- `b0a_decision_record.txt` — the AUTHORITATIVE decision record (revised
  per the collaborator's round-3 review). A copy sits next to the plan in
  the workspace `docs/`; this one is the tracked original.
- `h2s_dominance_sweep.py` / `h2s_dominance_results.json` — gas-phase
  H2S-dominance sweep over the supported (T_bottom, Z, C/O) domain at the
  engine bottom pressure, plus the S-condensable saturation-ratio
  diagnostic (max 3.5e-9 anywhere: the gas-phase equilibrium is the
  complete sulfur equilibrium at the boundary node) and a full provenance
  block. 1260 nodes, 0 nonconverged.
- `h2s_boundary_table.py` / `h2s_boundary_table.json` — the proposed
  production boundary condition: ln x_H2S(T_bottom, lnZ, c_o) lookup at
  P_b = 7.6 bar (17x9x7, trilinear in ln x) with off-node validation of
  values AND partial derivatives against FastChem finite differences.
  The B1 JAX port must reproduce the trilinear rule and gate on the
  table checksum.
- `harness/` — the B0-6 derivative-probe drivers (chemistry endpoint +
  one binned-spectrum row; env-gated, see `harness/README.md`).

## Re-running

Both scripts drive the VULCAN-vendored gas-phase FastChem binary and must
run against a PRIVATE copy of the tree (they overwrite its input files):

```
cp -R ../../../VULCAN-JAX/src/vulcan_jax/fastchem_vulcan /tmp/fastchem_check
cp /tmp/fastchem_check/input/parameters_wo_ion.dat /tmp/fastchem_check/input/parameters.dat
H2S_CHECK_FASTCHEM_DIR=/tmp/fastchem_check python h2s_dominance_sweep.py
H2S_CHECK_FASTCHEM_DIR=/tmp/fastchem_check python h2s_boundary_table.py
```

Note: the vendored FastChem abundance reader consumes the first line of the
element-abundance file as a header; both scripts therefore write a leading
comment line (dropping it silently loses hydrogen).

Runtime is about a minute each on a laptop; the sweep is 30 FastChem
invocations, the table builder about 143 (grid + validation FD stencils).
