# Route B B0-6 harness — early derivative probes

Scratch-level drivers (plan Section 5: "NOT production wiring") for the
B0B go/no-go derivative checks. Both own their process (the SNCHO network
and atom list are import-locked in VULCAN-JAX) and are env-gated; nothing
runs on import.

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

The W107b Guillot + photochemistry fixture (the G1/G6 science column) is
NOT wired here: it needs the B1-7 production plumbing (conden_mode
passthrough + on-graph pin in `vulcan_chem._prep`). The script raises
loudly if asked for it (`ROUTE_B_FIXTURE`).
