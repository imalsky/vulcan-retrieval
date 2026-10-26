# todo

The evidence behind each item is in `notes.md` (sections named below); solver and engine
items live in VULCAN3's and vulcan-forward's own `todo.md`. Two parts (the maintainer
decided, notes "Design records"): the model, the checks and changes that make the
pipeline right in general, comes first; everything about WASP-39b's results (the v5
run, checks at its posterior draws, sensitivity runs, how a number is quoted) waits for
the paper plan, where those tests are designed together.

Model

1. [owed] The end-to-end check of the production stack: `validation/e2e_check.py`'s ladder (`tools/check_nas.pbs -v CHECK=e2e`; the CPU rungs on daw) and the CLI against VULCAN 2; a disagreement is localised before a production run.

Paper: WASP-39b results (after the paper plan)

2. [owed] The rate-range clamp at v5's hottest and coolest posterior draws (the external reviewer's ask; the +1.5% SO2 on the nominal column does not cover the T-window edges).
3. [science] The conditioning map on v5 (the certificate's `conditioning` block) and what it means for the Kzz floor.
4. [decision] The four sensitivity runs beside v5 (`wasp39_test1_kzz1000`, `_deck`, `_kzz_powerlaw`, `_adiabat`; specs ready, notes "Design records"): the paper decides which it reports.

WASP-107b: Kzz from the two 2024 papers' data

5. [owed] WASP-107b, before a Kzz number is quoted: the solver-cut test (re-solve the rejected in-window draws below log10 Kzz ~8.1 with a longer cap against the posterior's log L) and the rtol 0.05 re-solve of the posterior rows with the warm-vs-cold check, both on the three finished runs at their source.
6. [decision] WASP-107b prior-edge rerun: a joint-fit variant with the Tirr edge lowered (the 320 K window meets ~400 K for half the posterior), the C/O floor below 0.1 and the log10 gamma ceiling above 0.3; about 24 GPU-h.

## 1. End-to-end check

Evidence: notes "Design records" (the ladder, the gates, the VULCAN 2 envelope). The
GPU rungs in one NAS job (`qsub -W "group_list=${PBS_GROUP:?}" -v CHECK=e2e
tools/check_nas.pbs`), then on daw the CPU rungs on the first 128 rows of the GPU's
draw set (`--u-from` its `e2e_production.npz`), then `compare` down the ladder and
`VULCAN-JAX/tools/compare_vul.py` on the CLI's W39b and HD189 columns. The job only
solves; `compare` exits 1 over a gate or with no draw certified on both sides. A rung
over a gate is localised (the rung names the change; replay the draw with its queue
neighbours) before a production run. Done: the batch rungs and the CLI against the CPU
port (notes "Design records"; v5 launched on them). Left: the one-at-a-time GPU rung
(the first job hit its wall; rerun on gpu_normal), the daw CPU rungs, and a replay of
row 1352 under ffi and fast. Done when those are numbers in notes "Design records".

## 2. Rate clamp at the T extremes

Evidence: VULCAN3 notes "Parity numbers" (P16 and the nominal patch study). The
reviewer's design: 8-16 weighted posterior draws at the T(1 bar) and Tirr extremes,
re-solved with and without the clamp, quoting the SO2 change and the per-band spectrum
change in sigma. vulcan3 has no switch: the study patch evaluates `rates_jax._arrhenius`
(k0 and k_inf) at min(T, hi) for the 345 rates of the W39b network with a stated range,
matched by equation text. Done when the numbers are under "Physics assumptions and
limitations".

## 3. Conditioning map on v5

Evidence: notes "Design records" (the rejected-draw log). The certificate's
`conditioning` block bins the in-window proposals in lnKzz, T1bar and lnZ (one at a
time) with the rejected fraction and the posterior weight per bin. The reviewer's
reading: rejections that crowd x0.1-x0.2 make the floor a solver cut, so either Kzz is
reported as an upper limit or those draws are certified with a longer budget (a cap
change, the maintainer's call). v4 wrote no log. Done when the statement is in "Physics
assumptions and limitations".

## 4. Sensitivity runs

Evidence: notes "Design records" (the four variants, their order and what each is
compared on). The external review asked for each; the reply to the reviewer promised
them. Each answers whether a modelling choice moves a headline number, so the paper
decides which it reports (a sensitivity table: O/H, total metals, C/O, S/O, Kzz, SO2,
evidence difference; an evidence difference needs cold runs and >= 3 seeds). One submit
each, `python tools/submit.py runs/w39b_smc_retrieval/<name>.yaml`, at v5's code. Done
when each run that the paper uses has its row.

## 5. WASP-107b: the solver-cut test and the re-solve

Evidence: notes "Runs" (the w107b entry: the Kzz plateau whose lower edge sits on the
solver cut, the rejected fraction per bin) and "Design records" (the case; the
baseline_v2 correction; the fixedmass rejected-draw study). Runs on the three finished
runs, which are the final ones (the maintainer decided; MIRI is left out, notes "Design
records", WASP-107b case), at their launch commits (the reweighting is exp(L_new -
L_run): vulcan-retrieval ea0b96b, VULCAN3 fed3860, vulcan-forward c5770a4). The
cap for both is `count_max` 30000 (a cold 10^8.6 column certifies in 13085 steps at
rtol 0.05, "Design records"). Each run's results go to `output/w107b/<run>_validation/`
(`e2e_check.py --results`), never to WASP-39b's directory. From the repo root, `<run>`
the run name:
(1) The cut: the draws it removed are in each run's `rejected/` log, not in the
posterior, so re-solve the in-window rejected draws below log10 Kzz 8.1 cold at the
run's rtol (0.2) and compare their log L with the posterior rows' at the same Kzz (0.35
dex bins); the result says whether the plateau extends below 10^7.4 or Kzz is quoted
as an upper limit. `<run>_cut.yaml` is the run's spec with only `count_max: 30000`
(not written yet; `--dry-run` against the run's must differ only there):

    python validation/rejected_resolve.py select output/w107b/<run>_nautilus --results output/w107b/<run>_validation/solver_cut --per-bin 300
    python validation/e2e_check.py solve runs/w107b/<run>_cut.yaml --rung production --u-from output/w107b/<run>_validation/solver_cut/rejected_draws.npz --results output/w107b/<run>_validation/solver_cut
    python validation/rejected_resolve.py compare output/w107b/<run>_nautilus output/w107b/<run>_validation/solver_cut/e2e_production.npz

On the current three [M]: 3053, 1221 and 5246 such draws (joint, Welbanks's bands,
G395H), 95-96% exhausted at the cap and the rest stalled, 89-98% rejected warm;
`--per-bin 300` keeps 1214, 1109 and 1211.
(2) The rtol bias of the accepted draws: the weighted posterior rows re-solved at rtol
0.05 (the v5 gate 0.1 sigma); `warmcold` gives zero weight to a row that does not
certify, so the cap must let the plateau rows certify. `<run>_resolve.yaml` is the
run's spec with rtol 0.05 and `count_max` 30000, nothing else (written for the current
three):

    python validation/e2e_check.py solve runs/w107b/<run>_resolve.yaml --rung production --posterior output/w107b/<run>_nautilus --draws 1000 --results output/w107b/<run>_validation
    python validation/e2e_check.py warmcold output/w107b/<run>_validation/e2e_production_posterior.npz --results output/w107b/<run>_validation

Cost [E]: the runs' warm batches move ~3000 lane-steps/s on 528 lanes (their
`batches.jsonl`; cold unmeasured) and v5's 1000-row re-solve at cap 5000 took 31 min,
so about 3 h per run for (2) if the slowest lanes reach the cap, and at most ~7 h per
run for (1) (two cold stages of up to 30000 steps per draw). One GPU job per run in
the v5 postrun job's environment, its log under `scratch/` (`qsub -o`). Done when both
results are in "Runs".

## 6. WASP-107b: the prior-edge rerun

Evidence: notes "Runs" (the w107b entry: Tirr at the 600 K edge, C/O at the 0.10
floor, log10 gamma at the +0.30 ceiling, T(1 bar) against the ~1680 K window ceiling,
the window's own Tirr floor) and "Design records" (the case). One joint-fit variant
spec (`base: w107b_joint.yaml`) with the Tirr edge lowered, the C/O floor below 0.1 and
the log10 gamma ceiling above 0.3; cold-certify columns at the new prior corners before
submitting, as was done for the case; `python tools/submit.py runs/w107b/<name>.yaml`
(about 24 GPU-h on gpu_long; rtol 0.2 like the runs, so item 5's re-solve applies to it
too). Lowering Tirr below ~450 K trades the prior edge for the window edge for about
half the posterior, so the shift it can show is bounded; the 10 bar bottom is a
separate variant if wanted. Reported as the shifts in O/H, C/O and Kzz against the
joint fit; a shift inside the 68% widths is not evidence without seeds. Closes when the
maintainer decides to run it or not; the decision goes to "Design records".
