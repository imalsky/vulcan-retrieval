# Route B harness — B0-6 derivative probes + B0C gate runners

Env-gated drivers; nothing runs on import; every script owns its process
(the SNCHO network and atom list are import-locked in VULCAN-JAX).

## B0C gate runners (primal-first order, round-5 review)

Shared fixture: `w107b_fixture.py` — the MANDATORY scientific fixture
(WASP-107b, Guillot T-P, photochemistry ON, equilibrium-table H2S deep
reservoir, smooth rainout) built through the PRODUCTION forward path
(`retrieval_framework.forward.vulcan_chem`, `conden_mode="smooth_rainout"`).
theta = [lnZ, c_o, lnKzz, Tirr, Tint, log_kappa, log_gamma]. G1 honesty:
no runtime cap, no conver_ignore additions; mtol_conv sits at the repo's
certified 1e-15 cold-column floor after the first run MEASURED the stall
to be sub-floor trace radicals (N2H 1.6e-17) — convergence is earned or
the gate fails with measured numbers. Shared reporting:
`gate_common.py` (provenance, termination derivation, ledger/residual/
agreement reports, artifact I/O to `results/`).

All heavy runners gate on `ROUTE_B_W107B=1` (Isaac schedules):

| Script | Gate | Cost |
|---|---|---|
| `g1_convergence.py` | G1 convergence + B0-5 direct residual (saves state) | 1 build + 1 solve |
| `g3_flux_closure.py` | G3 D6 sulfur/H flux closure (reads G1 artifact) | cheap, no solve |
| `g2_seeds.py` | G2 five-seed agreement (equilibrium / master_pin twin / S-rich / S-poor / continuation) | 3 builds + ~7 solves |
| `g4_subsaturated.py` | G4 hot subsaturated limit == conden-off (same pin both sides) | 2 builds + 2 solves |
| `g5_ladders.py` | G5a w-ladder kill test; G5b scale/supply/Kzz ladders + lookup-cell probe | ~5 builds + ~11 solves |

Verdict discipline: PASS/FAIL only when the documented thresholds are
supplied (`ROUTE_B_G1_MAX_R`, `ROUTE_B_G3_TOL`, `ROUTE_B_G2_RTOL/ATOL`,
`ROUTE_B_G4_RTOL/ATOL`, `ROUTE_B_G5A_RTOL/ATOL`); otherwise the artifact
records the measured numbers and an INCOMPLETE verdict for the record
review to set thresholds against.

Spectrum + derivative + assembly layer:

| Script | Duty | Cost |
|---|---|---|
| `w107b_spectrum.py npz <artifact.npz>` | measured spectrum-agreement columns for G2/G5a from saved gate states | cheap (1 build, no solves) |
| `w107b_spectrum.py fd` | directive K: reconverged-FD binned-spectrum derivative rows (h, h/2; per-endpoint termination recorded) | 9 solves |
| `g6_sensitivity.py` | G6: solver-map adjoint (rainout in the body map, theta-map extras for n_sat/C/pin derivatives, closed-element deflation) vs independently reconverged FD on 4 losses incl. one spectrum band; refuses on a non-converged or zero-rain nominal | ~18 solves + 4 adjoint ensembles |
| `eta_c_linearity.py` | plan section 9 Fisher validity guard (library, unit-checked; exercised at Fisher-enablement) | caller-priced |
| `b0c_artifact.py` | assembles THE single B0C reproducibility artifact from the latest gate artifacts; refuses mixed provenance; directive-N go/no-go | cheap |

Documented thresholds (PROPOSED 2026-07-14, pending Isaac + collaborator
sign-off; all conservative relative to measured noise, none weakens a
gate):

- G4: `ROUTE_B_G4_RTOL=1e-2`, `ROUTE_B_G4_ATOL=1.0` (cm^-3). Justification:
  measured max endpoint deviation smooth-vs-conden-off on converged hot
  solves is 5.28e-4 in |dln n| (artifact w107b_g4_20260714_080347) -- the
  1e-2 threshold is the yconv certification scale, ~19x above measured
  noise, far below science relevance; atol 1.0 cm^-3 only guards exact-zero
  trace cells.
- G2: `ROUTE_B_G2_RTOL=1e-2`, `ROUTE_B_G2_ATOL=1.0` (same metric class as
  G4: endpoint state agreement at the certification scale).
- G5a: `ROUTE_B_G5A_RTOL=1e-2`, `ROUTE_B_G5A_ATOL=1.0` (endpoint stability
  across w at the same scale; instability beyond it IS numerical failure).
- G3: `ROUTE_B_G3_TOL=1e-2` (fraction of boundary/rain throughput left
  unaccounted at the endpoint; the ledger telescoping identity is exact to
  ~1e-16, so 1e-2 headroom is purely for genuine slow drift at a certified
  endpoint).
- G1 `ROUTE_B_G1_MAX_R`: NOT yet proposed -- needs the first certified
  endpoint's measured residual scale (a threshold proposed blind would be
  arbitrary).

G1 measured history (2026-07-14): the tool-default hot Guillot fiducial
(Tirr 1046.5) converged nothing sulfurous — zero rainout, S8 < 4e-17
(SO2 1.1e-4: photochemistry fine, column too hot for S8) — so the fixture
was redesigned to the measured coolest-valid candidate (Tirr 500, Tint
150, log_kappa -2.5, log_gamma 0). On that column the cold-EQ G1 solve
exhausted 15000 steps mid-transient: the boundary reservoir fills the
column on ~3e9 s while dt is pinned at ~1e2 s by stiff S3/S4 front
chemistry (worst residual S4 z=43). Continuation laddering and the
sanctioned knobs are the open G1 avenues; every attempt is archived in
`results/`.

## B0-6 derivative probes (B0B)

## b0_6_chem_probe.py — chemistry-endpoint derivatives

```
ROUTE_B_PROBE=1 python b0_6_chem_probe.py
```

Isothermal 400 K SNCHO S8 mini-column (local-derivative fixture only),
theta = (T_iso, ln x_pin[H2S]). Reports, on the losses
[ln N_S8, ln N_H2S, ln Phi_S,rain]:

- primal convergence facts (steps, runtime-cap termination), the sulfur
  budget closure from the ledger, and the B0-5 scaled residual;
- route 1: unrolled forward-mode jvp through the full runner;
- route 2: the D9 implicit fixed-point prototype — dense (I - G_eta)
  solve in log coordinates on the smooth-rainout body map, with the null
  space MEASURED by SVD (expected rank 3 = {O, C, N}; loud failure on
  mismatch, including S/H measuring null);
- route 3: independently re-run centered FD at h and h/2;
- a comparison table with relative deviations vs FD(h/2).

First full run archived: `probe_results_2026_07_13.txt` (FD ground truth
clean and physical — d ln N_S8/dT = +0.0731/K vs saturation-curve slope
0.0738, d ln N_H2S/d ln x_pin ~ 1.0; unrolled jvp fails by 6-9 orders on
this cold fixture; dense implicit prototype reaches right order on the
best-conditioned entry only — the production solver-map LGMRES route is
required for G6).

## b0_6_spectrum_probe.py — one binned-spectrum derivative row

```
ROUTE_B_SPECTRUM_PROBE=1 python b0_6_spectrum_probe.py
```

Full-chain mechanics check: theta -> chemistry with smooth rainout ->
exojax transmission depth (retrieval `exojax_rt.build_rt_model`, 3.6-4.2
um window with H2O/CO/CH4/H2S/SO2) -> one plain band-average row; jvp vs
centered FD on that row. Heavy (opacity build + linelist data from
`data/opacity_cache/` and `data/exojax_linelists/`); Isaac schedules.

The W107b chemistry plumbing this probe used to lack now EXISTS
(`vulcan_chem` smooth-rainout passthrough + on-graph pin, 2026-07-13; see
`w107b_fixture.py`), but this script's W107b spectrum path is still not
wired — the full binned-spectrum derivative on the active-rainout fixture
is the dedicated spectrum-derivative work item and does NOT reuse this
scratch driver. The `ROUTE_B_FIXTURE` hatch still raises loudly.
