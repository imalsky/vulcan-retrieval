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

5. [owed] WASP-107b MIRI-era runs: the joint fit and Welbanks's bands with MIRI, the joint fit with the Tirr and gamma edges widened, G395H alone at R300; each product as its authors fitted it, the published errors (inflated per group in the joint fits).
6. [owed] WASP-107b, before a Kzz number is quoted: the solver-cut test (re-solve the rejected in-window draws below log10 Kzz ~8.1 with a longer cap against the posterior's log L) and the rtol 0.05 re-solve of the posterior rows with the warm-vs-cold check, on the MIRI-era runs at their source.

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

## 5. WASP-107b: the MIRI-era runs

Evidence: notes "Runs" (the w107b entry: the prior edges pressed, the Kzz plateau) and
"Design records" (the case). The maintainer decided to refit with MIRI and every product
as its authors fitted it: Welbanks+2024's NIRCam at its native 0.015 um bins (F322W2
offset by -81 ppm as in his retrievals, F444W the 72 bins to 4.97 um), MIRI LRS the 47
bins to ~12 um and G141 the 20 bins to 1.629 um, WFC3 and NIRCam in one frame; Sing+2024's
G395H at R300 per detector as one block; the published errors, with one inflation per
group in the two joint fits (NRS1 and the native F322W2 disagree in shape at 2.85-3.6 um,
chi2 108/54 about a constant); the C/O floor 0.05 (`prepare.py` under
`scratch/w107b_data/`; the binning operator now admits bins down to two model bands).
Four specs in `runs/w107b/`, launched together from one commit (`python
tools/submit.py runs/w107b/<name>.yaml [--queue gpu_normal]`):
`w107b_joint_miri` (gpu_long), the joint fit; `w107b_joint_miri_edges` (gpu_long), Tirr
from 400 K and log10 gamma to 1.0, read as the shifts in O/H, C/O and Kzz against
`w107b_joint_miri` (a shift inside the 68% widths is not evidence without seeds);
`w107b_welbanks_miri` (gpu_normal), Welbanks's data as he fitted them (one MIRI offset on
his prior), against his log10 Kzz 8.6; `w107b_g395h_r300` (gpu_normal), Sing's data alone,
against his 11.6. The two gpu_normal runs have a 22 h governor and a resume watcher under
`scratch/` (resumes a job that ends without a certificate while the checkout stays at the
launch commit, at most 4 times). Expected wall [E]: 1.3-1.6x the finished runs' (23.9 h
joint, 18.1 h Welbanks's bands; the RT is x1.52 on the 2515-band grid to 12.5 um). The
model has no 6-11 um aerosol absorber (Welbanks's skewed-normal term, Dyrek+2024's
silicates), which overlaps the SO2 7.35/8.69 um and CH4 7.7 um bands, and no nitrogen
freedom (Welbanks scaled NH3 x29): the absorber is built if the posterior-predictive MIRI
residuals give chi2/47 > 1.5 or |sum r / sigma| / sqrt(N) > 3 over 7-11 um, or if a
MIRI-free twin of `w107b_welbanks_miri` (a later run) moves S/O, O/H, the total metals,
C/O or log10 Kzz by more than its 68% half-width. Done when each run's numbers, chi2/N
per group and the MIRI residual test are in notes "Runs".

## 6. WASP-107b: the solver-cut test and the re-solve

Evidence: notes "Runs" (the w107b entry: the Kzz plateau whose lower edge sits on the
solver cut, the rejected fraction per bin) and "Design records" (the case; the
baseline_v2 correction; the fixedmass rejected-draw study). Runs on the MIRI-era runs
(item 5), each at its own launch commits (the reweighting is exp(L_new - L_run)). The
cap for both is `count_max` 30000 (a cold 10^8.6 column certifies in 13085 steps at
rtol 0.05, "Design records"). Each run's results go to `output/w107b/<run>_validation/`
(`e2e_check.py --results`), never to WASP-39b's directory. The analysis specs are the
run's spec with the changes below and nothing else; a base has no base of its own, so
each restates its run's keys on `base: w107b_joint.yaml` (`--dry-run` against the run's
must differ only there). From the repo root, `<run>` the run name:
(1) The cut: the draws it removed are in each run's `rejected/` log, not in the
posterior, so re-solve the in-window rejected draws below log10 Kzz 8.1 cold at the
run's rtol (0.2) and compare their log L with the posterior rows' at the same Kzz (0.35
dex bins); the result says whether the plateau extends below 10^7.4 or Kzz is quoted
as an upper limit. `<run>_cut.yaml`: `count_max: 30000`.

    python validation/rejected_resolve.py select output/w107b/<run>_nautilus --results output/w107b/<run>_validation/solver_cut --per-bin 300
    python validation/e2e_check.py solve runs/w107b/<run>_cut.yaml --rung production --u-from output/w107b/<run>_validation/solver_cut/rejected_draws.npz --results output/w107b/<run>_validation/solver_cut
    python validation/rejected_resolve.py compare output/w107b/<run>_nautilus output/w107b/<run>_validation/solver_cut/e2e_production.npz

On the finished three [M]: 3053, 1221 and 5246 such draws (joint, Welbanks's bands,
G395H), 95-96% exhausted at the cap and the rest stalled, 89-98% rejected warm;
`--per-bin 300` keeps 1214, 1109 and 1211.
(2) The rtol bias of the accepted draws: the weighted posterior rows re-solved at rtol
0.05 (the v5 gate 0.1 sigma); `warmcold` gives zero weight to a row that does not
certify, so the cap must let the plateau rows certify. `<run>_resolve.yaml`: rtol 0.05
(`cfg_overrides` restated whole) and `count_max` 30000.

    python validation/e2e_check.py solve runs/w107b/<run>_resolve.yaml --rung production --posterior output/w107b/<run>_nautilus --draws 1000 --results output/w107b/<run>_validation
    python validation/e2e_check.py warmcold output/w107b/<run>_validation/e2e_production_posterior.npz --results output/w107b/<run>_validation

Cost [E]: the finished runs' warm batches move ~3000 lane-steps/s on 528 lanes (their
`batches.jsonl`; cold unmeasured) and v5's 1000-row re-solve at cap 5000 took 31 min,
so about 3 h per run for (2) if the slowest lanes reach the cap, and at most ~7 h per
run for (1) (two cold stages of up to 30000 steps per draw). One GPU job per run in
the v5 postrun job's environment, its log under `scratch/` (`qsub -o`). Done when both
results are in "Runs".
