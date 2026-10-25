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

5. [owed] The three WASP-107b runs (`runs/w107b/`: the joint fit, Sing+2024's G395H alone, Welbanks+2024's bands alone), then whether the data constrain Kzz and which paper's value they support.

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

## 5. WASP-107b runs

Evidence: notes "Design records" (the WASP-107b case: data, system, the 10 bar bottom,
priors, what is not modelled). Submit from cghfe02: `python tools/submit.py
runs/w107b/w107b_joint.yaml` (gpu_long, 46 h governor), `w107b_g395h.yaml` and
`w107b_welbanks.yaml` with `--queue gpu_normal` (22 h governor; `--resume` until each
finishes). Each run binds the NAS checkout's commit, so keep the checkout fixed until
all three finish. Then `nautilus_report` on each and the Kzz marginal and the joint
Kzz-T1bar posterior side by side, with the certificate's `conditioning` block (the
rejected fraction per lnKzz bin; columns below ~10^8.3 do not certify within 5000
steps at rtol 0.2, a solver cut, not evidence against low Kzz) and the weight above
log10 Kzz 12, where the 10 bar bottom truncates the quench. The runs use rtol 0.2:
re-solve the weighted posterior rows at rtol 0.05 (the baseline_v2 correction, a
higher cap for those solves only) before any number is quoted. A statement about which paper is right needs the separate-data runs to agree
with the joint one. Done when the three runs' numbers are in "Runs".
