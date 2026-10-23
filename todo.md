# todo

The evidence behind each item is in `notes.md` (sections named below); solver and engine
items live in VULCAN3's and vulcan-forward's own `todo.md`. The v5 model options and the
report changes from the external science review are on main ("Design records"); what
remains is the runs and the checks.

1. [owed] Warm-vs-cold check: a small script that re-solves a weighted subsample of 500-1000
   posterior draws cold through the production program at unchanged tolerances (low-Kzz rows
   reported apart), CPU-tested; run on v5 before any v5 number is quoted. v4 (job 94946)
   stays unquotable until the same check runs on it, at `0a2dbf3` (the last commit with
   v4's presolve chain), which waits until a v4 number is needed.
2. [owed] The v5 run (`runs/w39b_smc_retrieval/wasp39_test1.yaml`, `python tools/submit.py <spec>`); numbers to "Runs".
3. [owed] Forward checks at v5's posterior draws, after v5 finishes (`validation/posterior_forward_check.py`, `validation/opacity_leave_one_out.py --from-posterior`), each a number under "Physics assumptions and limitations".
4. [science] Conditioning map: the rejected fraction over the posterior from v5's `rejected/` log, and whether v4's checkpoint keeps its rejected points.

## 2. The v5 run

Evidence: notes "Design records" (the decisions and the options), "Current state" (the
launch recipe). Before launch: the full tier here (`-m "slow or not slow"`; it includes
`tests/test_v5_options.py`, every option on the real smoke pipeline), vulcan-forward's
full tier, and the new launch path end-to-end on the CPU: `smoke.yaml` through
`run_nautilus` with a short governor, then `--resume`
(CLAUDE.md: a new job path runs on the CPU first; the submit wrapper's `-o <dir>/` and
`afterany` chaining are confirmed on the first live submit). The NAS environment and the
GPU kernel test pass on the v5 code (notes "HPC (NAS GH200) and daw"). The run: `v5` (final, 17
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
They wait on v5 (the maintainer decided): v4 is superseded, and its presolve-chain run
would need `0a2dbf3`, since the script refuses a chain run on main.
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
