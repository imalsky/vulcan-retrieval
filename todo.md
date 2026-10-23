# todo

The evidence behind each item is in `notes.md` (sections named below); solver and engine
items live in VULCAN3's and vulcan-forward's own `todo.md`. Two parts (the maintainer
decided, notes "Design records"): the model, the checks and changes that make the
pipeline right in general, comes first; everything about WASP-39b's results (the v5
run, checks at its posterior draws, sensitivity runs, how a number is quoted) waits for
the paper plan, where those tests are designed together.

Model

1. [owed] The end-to-end check of the production stack: `validation/e2e_check.py`'s ladder (`tools/check_nas.pbs -v CHECK=e2e`; the CPU rungs on daw) and the CLI against VULCAN 2; a disagreement is localised before a production run.
2. [decision] Integrate the 2 offsets and 3 noise inflations out of the likelihood so nautilus samples 12 parameters (notes "Sampler", the stand-in study); decided before the v5 launch.

Paper: WASP-39b results (after the paper plan)

3. [owed] The v5 run (`runs/w39b_smc_retrieval/wasp39_test1.yaml`, `python tools/submit.py <spec>`), once the model part is closed; numbers to "Runs".
4. [owed] Warm-vs-cold check on v5 before any v5 number is quoted: `validation/e2e_check.py warmcold` on 1000 weighted posterior rows re-solved cold.
5. [owed] Forward checks at v5's posterior draws (`validation/posterior_forward_check.py`, `validation/opacity_leave_one_out.py --from-posterior`), each a number under "Physics assumptions and limitations".
6. [owed] The rate-range clamp at v5's hottest and coolest posterior draws (the external reviewer's ask; the +1.5% SO2 on the nominal column does not cover the T-window edges).
7. [science] The conditioning map on v5 (the certificate's `conditioning` block) and what it means for the Kzz floor.
8. [decision] The four sensitivity runs beside v5 (`wasp39_test1_kzz1000`, `_deck`, `_kzz_powerlaw`, `_adiabat`; specs ready, notes "Design records"): the paper decides which it reports.

## 1. End-to-end check

Evidence: notes "Design records" (the ladder, the gates, the VULCAN 2 envelope). The
GPU rungs in one NAS job (`qsub -W "group_list=${PBS_GROUP:?}" -v CHECK=e2e
tools/check_nas.pbs`), then on daw the CPU rungs on the first 128 rows of the GPU's
draw set (`--u-from` its `e2e_production.npz`), then `compare` down the ladder and
`VULCAN-JAX/tools/compare_vul.py` on the CLI's W39b and HD189 columns. The job only
solves; `compare` exits 1 over a gate or with no draw certified on both sides. A rung
over a gate is localised (the rung names the change; replay the draw with its queue
neighbours) before a production run. Done when every comparison and the VULCAN 2
envelope are numbers in notes "Design records".

## 2. Nuisance integral

Evidence: notes "Sampler" (the 17-D stand-in: 12-D nautilus recovered the evidence and
a weak mode that 17-D nautilus missed in every seed, with 24% fewer calls). Each
instrument group's likelihood depends on its own offset (uniform +-800 ppm; none for
NIRISS, the reference) and inflation (log10-uniform 0.5-3), so the offset integral is
closed-form (erf) and the inflation one a 1-D quadrature per group; their posteriors are
drawn afterwards, conditional on each row. Before adopting it: the quadrature against
a brute-force integral of the 17-D likelihood at a few draws, the normalisation, and the
report and replay tools (`nautilus_report`, `e2e_check.py --posterior`, the forward
checks) on the 12-D layout. A new spec and a new run name. Done when decided (notes
"Design records") and, if adopted, those checks pass before v5.

## 3. The v5 run

Evidence: notes "Design records" (the decisions and the options), "Current state" (the
launch recipe). Before launch: the full tier here (`-m "slow or not slow"`; it includes
`tests/test_v5_options.py`), vulcan-forward's full tier, and the launch path on the CPU
(`smoke.yaml` through `run_nautilus` with a short governor, then `--resume`). 17
parameters may need more than v4's 24 h, so watch the governor. When it finishes:
`nautilus_report` (the certificate's `derived` block holds the quoted quantities) and
the "Physics assumptions and limitations" list beside the posterior. Done when the run's
numbers are in "Runs".

## 4. Warm-vs-cold on v5

Evidence: notes "Design records" (the end-to-end check, `warmcold`), "Runs"
(postrun_93269, the method). On a GPU: `python validation/e2e_check.py solve <spec>
--rung production --posterior <out_dir> --draws 1000`, then `warmcold` on its npz;
gate: every parameter's reweighted median within 0.1 sigma. Done when it passes and the
numbers are in "Runs".

## 5. Forward checks

Evidence: notes "Physics assumptions and limitations". `posterior_forward_check.py run
<out_dir> --variant base`, then `network2025`, `dayside`, `twolimb`, `mass`, `spot` (one
process each; the network variant freezes at import), then `compare`; weighted draws,
the script's 16 by default. `opacity_leave_one_out.py --from-posterior
<out_dir>/nautilus_posterior.npz` screens the 13 omitted absorbers at the gate min(5 ppm,
0.1 sigma); one over it joins `molecules` and v5 is re-run with it. Done when each result
is a number under "Physics assumptions and limitations".

## 6. Rate clamp at the T extremes

Evidence: VULCAN3 notes "Parity numbers" (P16 and the nominal patch study). The
reviewer's design: 8-16 weighted posterior draws at the T(1 bar) and Tirr extremes,
re-solved with and without the clamp, quoting the SO2 change and the per-band spectrum
change in sigma. vulcan3 has no switch: the study patch evaluates `rates_jax._arrhenius`
(k0 and k_inf) at min(T, hi) for the 345 rates of the W39b network with a stated range,
matched by equation text. Done when the numbers are under "Physics assumptions and
limitations".

## 7. Conditioning map on v5

Evidence: notes "Design records" (the rejected-draw log). The certificate's
`conditioning` block bins the in-window proposals in lnKzz, T1bar and lnZ (one at a
time) with the rejected fraction and the posterior weight per bin. The reviewer's
reading: rejections that crowd x0.1-x0.2 make the floor a solver cut, so either Kzz is
reported as an upper limit or those draws are certified with a longer budget (a cap
change, the maintainer's call). v4 wrote no log. Done when the statement is in "Physics
assumptions and limitations".

## 8. Sensitivity runs

Evidence: notes "Design records" (the four variants, their order and what each is
compared on). The external review asked for each; the reply to the reviewer promised
them. Each answers whether a modelling choice moves a headline number, so the paper
decides which it reports (a sensitivity table: O/H, total metals, C/O, S/O, Kzz, SO2,
evidence difference; an evidence difference needs cold runs and >= 3 seeds). One submit
each, `python tools/submit.py runs/w39b_smc_retrieval/<name>.yaml`, at v5's code. Done
when each run that the paper uses has its row.
