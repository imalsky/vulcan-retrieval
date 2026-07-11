# validation: what each script measures

Reference notes for the nine scripts in this directory, with the verdicts
recorded in the pre-0.6 docs. Two groups: offline pre-flight smokes
(laptop-safe) and the 2026-07-11 audit-response suite. The audit-response
scripts must run on the GPU node before the next production retrieval: every
chemistry/spectrum cache that predates the 2026-07-11 scientific-correctness
pass is stale (see the repo-root `CLAUDE.md`).

## Offline pre-flight smokes

- `smoke_test.py`: end-to-end FD check of the whole chain at the reduced SMOKE
  config (nz=40, CO-only, photo on). Proves the composed forward runs, all four
  parameter tangents (lnZ, C/O, lnKzz, dT) are finite through chemistry, bridge,
  and RT, and the forward-mode derivative matches a re-converged central finite
  difference for lnZ and dT. Recorded verdict (pre-0.6 README): jvp vs FD to
  ~2% on responding levels, machine precision on T; writes `data/smoke.npz`.
- `smoke_coref.py`: checks the c_o_ref continuation preserves C/O along a
  metallicity march (a bug that re-applies c_o every step would make C/O drift
  multiplicatively). nz=40, a few minutes.
- `smoke_zco.py`: chemistry-only checks for the Z and C/O Fisher figures: the
  fixed-O C/O knob conserves O while scaling C on the converged column, the
  forward-mode tangents are finite in all three chemistry tiers (including
  photo off), and AD matches central FD on an SO2-column readout.
- `validate_wide_chem.py`: chemistry jvp vs re-converged central FD at the real
  figure resolution (config.WIDE, nz=150, photo on) for lnZ, C/O, lnKzz.
  Recorded interpretation: lnZ and C/O match to <1%; lnKzz shows a few percent
  of FD noise on a near-zero column-mean derivative (the Kzz tangent itself is
  validated to <0.1% on the responding levels in VULCAN-JAX's
  fig_kzz_jvp_validate.py). ~13 min.

## Audit-response suite (2026-07-11)

- `elemental_audit.py`: per-draw audit of the abundance map: exact column
  elemental ratios (He/H, O/H, C/H, N/H, S/H), achieved dln(C/O) == c_o,
  per-layer density closure sum_i n_i == P/(kB T), and pv.atom_ini ==
  atoms(y_init); `--converge` adds post-convergence inventory drift. Exit 0 =
  all gates pass in elemental mode; `--mode masks` measures the legacy knob's
  documented leakage without failing.
- `resolution_ladder.py`: native-nu_pts convergence of the BINNED transit depth
  (the production nu_pts=1652, R~1000, was set by GPU gradient memory, not by a
  convergence test). PASS gates: binned-depth change <5 ppm between the top two
  ladder rungs; Jacobian direction change <1% where the depth response is
  significant.
- `top_pressure_ladder.py`: the clamped 1e-7 to 1e-8 bar model-top extension vs
  chemistry actually solved to 1e-8 bar (`--extend-chem`, the decisive
  comparison). PASS gate: |Delta binned depth| <5 ppm clamped vs extended.
- `broadening_ab.py`: terrestrial-air vs H2/He pressure broadening on the binned
  spectrum over the production band, same converged chemistry both ways.
  Reported, not gated; as a guide, differences below ~5 ppm are invisible under
  the CM24 error bars, tens of ppm mean production should switch to
  broadening="h2he". Recorded decision: the default stays "air" until this A/B
  is run and judged (`CLAUDE.md`, `forward/config.py`). The h2he build downloads
  separate `<db>_h2he` line-list caches on first use.
- `mala_reversibility.py`: warm-cap reversibility probe on a finished run's
  checkpointed cloud (nearest-neighbor particle pairs warm-solved in both
  directions). PASS: no asymmetric converged/capped classification and
  |L(fwd) - L(carried)| consistent with validate_warm's gate; asymmetric pairs
  at production settings mean raise warm_count_max or run the final ladder
  stages with smc_chem_mode="cold".

The framework-level warm-vs-cold gate is not here but in the package proper:
`python -m retrieval_framework.validate_warm <case dir>` gates on three axes
(max|dlogL| < 0.1, binned-spectrum ppm < 5, elemental-inventory agreement).

## Historical fragment (old bundle README, verbatim)

### `validation/` — is the gradient right?
**Question:** does the end-to-end forward-mode derivative equal a re-converged finite
difference, and does the C/O continuation actually hold C/O fixed?
**Method:** offline, CO-only (fully cached) FD checks of `d/dlnZ`, `d/dT_int`, and the
`c_o_ref` continuation; a full-resolution (nz=150) chemistry-jvp check.
**Assumptions:** none beyond the shared library; deliberately offline so it runs on a
laptop as the pre-flight before trusting any figure.
**Outputs:** pass/fail to stdout (jvp vs FD to ~2% on responding levels, machine-precision
on T), `data/smoke.npz`.

