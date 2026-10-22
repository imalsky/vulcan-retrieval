# todo

The evidence behind each item is in `notes.md` (sections named below); solver and engine
items live in VULCAN3's and vulcan-forward's own `todo.md`. The v5 model options and the
report changes from the external science review are on main ("Design records"); what
remains is the runs, the checks and one decision.

1. [owed] Warm-vs-cold check: a small script that re-solves a weighted subsample of 500-1000
   posterior draws cold through the production chain at unchanged tolerances (low-Kzz rows
   reported apart), CPU-tested; run on v5 before any v5 number is quoted. v4 (job 94946)
   stays unquotable until the same check runs on it, which waits until a v4 number is needed.
2. [owed] The v5 run (`runs/w39b_smc_retrieval/v5.yaml`, `python tools/submit.py <spec>`); numbers to "Runs".
3. [owed] Forward checks at v4's posterior draws (`validation/posterior_forward_check.py`, `validation/opacity_leave_one_out.py --from-posterior`), each a number under "Physics assumptions and limitations".
4. [science] Conditioning map: the rejected fraction over the posterior from v5's `rejected/` log, and whether v4's checkpoint keeps its rejected points.
5. [decision] Presolve chain in v5: the no-presolve A/B of v4 (notes "Runs", v4_nopresolve_test) ran 4 h faster with 1.0 point more in-window rejection and logZ_box 0.064 lower, every median within 0.1 sigma; keep the chain (the v5 files) or run v5 on the plain rtol-0.05 program.

## 2. The v5 run

Evidence: notes "Design records" (the decisions and the options), "Current state" (the
launch recipe). Before launch: the full tier here (`-m "slow or not slow"`; it includes
`tests/test_v5_options.py`, every option on the real smoke pipeline), vulcan-forward's
full tier, `tools/gpu_kernel_test.pbs` on the GPU, and the new launch path end-to-end on
the CPU: `smoke.yaml` through `run_nautilus` with a short governor, then `--resume`
(CLAUDE.md: a new job path runs on the CPU first; the submit wrapper's `-o <dir>/` and
`afterany` chaining are confirmed on the first live submit). The run: `v5` (final, 17
parameters). When it finishes: `nautilus_report` (the certificate's `derived` block
holds the quoted quantities), the warm-vs-cold check (item 1), and the "Physics
assumptions and limitations" list beside the posterior; Kzz as an upper limit if the
weight at the floor grows; 17 parameters may need more than v4's 24 h, so watch the
governor. Done when the run's numbers are in "Runs".

## 3. Forward checks

Evidence: notes "Physics assumptions and limitations", VULCAN3 notes (a one-sided
rate-range clamp moved the nominal W39b SO2 by +1.5%; vulcan3 has no switch for it, so
it is not a variant). `posterior_forward_check.py run <out_dir> --variant base`, then
`network2025`, `dayside`, `twolimb`, `mass`, `spot` (one process each; the network
variant freezes at import), then `compare`; weighted draws, the script's 16 by default.
`opacity_leave_one_out.py --from-posterior <out_dir>/nautilus_posterior.npz` screens
the 13 omitted absorbers at the gate min(5 ppm, 0.1 sigma); one over it joins
`molecules` and v5 is re-run with it. Done when each result is a number under "Physics
assumptions and limitations".

## 4. Conditioning map

Evidence: notes "Design records" (the rejected-draw log). `rejected/*.npz` carry the
unit-cube point, the class, the accepted steps and the start mode per chunk; map the
rejected fraction over (lnKzz, T1bar, lnZ) against the posterior, and the paper states
where the conditioning bites. v4 wrote no log: check whether its nautilus checkpoint
keeps the log L = -inf points, else the tally is v4's record. Done when the figure
exists for v5 and the statement is in "Physics assumptions and limitations".

## 5. Presolve chain in v5

Evidence: notes "Runs" (baseline_v4 and v4_nopresolve_test), "Low-Kzz hole" (the chain
was adopted on a 264-draw hard set: 263 certified against 203 for a plain rtol-0.05
solve at cap 15000, and the per-batch cost of a plain rtol-0.05 program was 2.8x; the
full-run A/B shows the opposite wall time, so the hard set over-weighted the low-Kzz
cases). The plain program certifies at rtol 0.05 with cap 5000 and no second compile.
Both v5 arms are one override edit: the chain (as written) or `chem_retry` and
`chem_presolve` false with `cfg_overrides.rtol` 0.05 and `count_max` 5000. logZ_box is
comparable only within one arm. The maintainer decides before the v5 launch; the
decision goes to notes "Design records".
