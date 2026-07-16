# Job 65815: the tangent-blown (badgrad) class -- investigation, fix, and operating guide

Date: 2026-07-16. Status: complete. Sampler fix `941c847` (zero-drift
handling); root-cause fix `881042e` (per-particle warm_extrapolate gate,
SS10).

This is the consolidated record of the NAS job 65815 abort: what happened,
what the forensics measured, why the previous handling was wrong, what the
fix is, how to operate the retrieval now, and how to reproduce every number
in this document. Short forms of this record: `notes.md` (2026-07-16 entry,
the dev-diary version), `docs/failed_approaches.md` (#67 + E-section 65815),
and the badgrad paragraph in `CLAUDE.md` (the current contract).

---

## 1. Executive summary

- Job 65815 (real-data W39b, N=144, resumed init) was deliberately aborted
  at stage 2 sweep 2 by the per-sweep badgrad gate: 11/144 proposals had a
  finite, canonically certified chemistry primal but a non-finite
  forward-mode tangent, exceeding the tolerance 8 = ceil(0.05 x 144).
- Forensics over all 13 dumped sweeps (64 events) show the class is
  **theta-dependent and posterior-concentrated**, not the ~1%
  theta-independent stochastic tail the 5% gate was designed around
  (NAS 65200/65789). It is ~10x denser at high metallicity and low C/O --
  exactly where the W39b data pulls the sampler -- and its rate RISES along
  the tempering ladder as the cloud drifts there.
- Two consequences followed: (a) MH-rejecting these proposals multiplied
  the target by a theta-correlated indicator, i.e. a systematic bias
  against the high-metallicity posterior bulk; (b) the 5% per-sweep abort
  was statistically near-certain to fire somewhere in a full ladder. The
  abort was therefore CORRECT as a refusal to produce biased science, and
  the fix is not "raise the tolerance".
- **Fix (commit `941c847`)**: badgrad proposals are handled as ZERO-DRIFT
  MALA moves -- kept, not rejected. The eval zeroes the non-finite gradient
  entries and that same zeroed drift enters both proposal densities, which
  is a valid MH kernel (the drift enters the proposal, not the target).
  The raise survives only as a systematic-breakage backstop
  (`smc_tangent_bad_max_frac`, default 0.25/sweep).
- Local replays reproduce the certified primals but NOT the tangent
  blowup at zero increment: the divergence needs the actual mutation-move
  warm trajectory. There is no pointwise chemistry fix (the K_eq overflow
  clip was already in place and is unrelated).

## 2. The incident

Job 65815 resumed the 65789 init-level checkpoint (SMC_RESUME=1), ran
stage 0 (accept 0.15, ESS 15.7/144) and stage 1 (accept 0.21, ESS 87.5),
~2 h per stage, and died 39 min into stage 2:

```
RuntimeError: 11 finite-likelihood/non-finite-gradient event(s) during SMC
stage 2 (beta=3.357e-04), sweep 2/6 exceed the tangent-reject tolerance
(8 = ceil(0.05 x 144)) ...
```

Per-sweep badgrad counts: stage 0: 1, 6, 4, 0, 4, 3; stage 1: 5, 5, 5, 7,
2, 4; stage 2: 7, 11. Every event was chemistry-tangent side (never the RT
vjp), and every offender was canonically certified (conv_ok=True).

Independently of the abort, the job was pacing ~2 h/stage against a
40-stage cap and a 24 h walltime, and 68% of warm proposals burned all the
way to the warm_count_max=1500 cap before rejection (see SS8).

## 3. Forensics: what the 13 dumps measured

Source data: `runs/w39b_smc_retrieval/forensics_65815/bad_grad_stage*.npz`
(13 files, each the full 144-particle sweep state: `bad_grad`,
`chem_tan_bad`, `acc`, `longdy`, `conv_ok`, `theta_proposal`,
`loglik_proposal`). Analysis scripts: `validation/badgrad_65815/`.

Headline numbers (1872 proposals, 990 certified, 64 badgrad):

| measurement | value |
|---|---|
| badgrad among CERTIFIED proposals, pooled | 6.5% |
| by stage (0 / 1 / 2) | 5.1% / 5.9% / 11.0% |
| by lnZ quartile (certified) | 1.2% (Z 1-12x) -> 7.3% -> 5.7% -> 11.7% (Z 67-99x) |
| by C/O quartile (certified) | 11.3% (0.10-0.18) / 5.7% / 4.5% / 4.4% (0.49-0.70) |
| joint high-Z+low-C/O half-cell vs opposite | 11.6% vs 2.6% |
| chemistry-side attribution | 64/64 |
| offenders canonically certified | 64/64 |
| repeat-particle structure | none (57 distinct slots / 64 events) |
| offenders with longdy in 0.05-0.1 (loose branch) | 38% (vs 13% of certified non-offenders) |
| offenders with longdy < 0.01 (well-converged) | 44% |
| offender accept counts | p5 478, median 810, p95 1330 (not warmcap-adjacent) |

Kzz and Tirr have NO effect once you condition on certification. (A naive
percentile test flags high Kzz, but that is composition: the
low-C/O/high-Kzz corner CERTIFIES far more easily -- 78% vs 47% -- so it is
overrepresented among certified proposals. Among certified proposals, the
high-Kzz half is if anything safer.)

The cloud is moving INTO the dense region: the certified-cloud median
drifted lnZ +0.77 -> +1.32 -> +1.71 (Z 22x -> 37x -> 55x solar) over stages
0-2, tracking the real W39b posterior pull toward high metallicity. So the
badgrad rate rises along the ladder; the design hope "watch this stay ~0 in
the late ladder" can never hold on this dataset.

Offender medians in physical units: Z ~ 57x solar, C/O ~ 0.21, Kzz ~ 19x.

## 4. Why the old handling was wrong

**Bias.** MH-rejecting a proposal because its tangent blew up multiplies
the target density by the indicator of "tangent computable". For a sparse
theta-independent class that is a negligible perturbation; for a class
that holds 6-12% of certified proposals concentrated exactly in the
posterior bulk, it is a systematic suppression of the retrieved
high-metallicity region -- a bias in the headline W39b parameter.

**Statistical certainty of the abort.** At the pooled 3.4% per-proposal
rate, a Binomial(144, 0.034) sweep exceeds 8 events with probability
~6%. The probability of at least one such sweep within the 14 sweeps the
job ran was ~58%; over a full 6x40-sweep ladder it is ~1. Stage 2's
11-event sweep was not an anomaly -- it was the first draw over a line the
run was guaranteed to cross. At the stage-2 conditional rate (11%), the
gate would have tripped with ~55% probability on EVERY subsequent sweep.

**No primal-side flag exists.** 44% of offenders are well-converged
(longdy < 0.01), so tightening certification (e.g. refusing the loose
longdy < 0.1 branch) removes at most a third of the class while rejecting
~13% of healthy certified proposals. This confirms the 65789 finding with
much more data.

## 5. What the mechanism is (and is not)

The chemistry gradient rides `jax.jvp` through the warm-continuation
`lax.while_loop` (`pipeline._chem_one` -> `chem.converged_y(warm_cap=True)`).
The loop stops on the PRIMAL certification; the tangent has no stopping
criterion of its own and is amplified or contracted by the linearization
of every accepted step. When the linearized step map has transiently
expanding modes along the trajectory, the tangent grows geometrically;
~1000 steps at an average factor as small as ~2 overflows float64 (the
healthy tangent scale is ~1e15-1e19; overflow is ~1e308). The same
recurrence, iterated as a raw-Neumann adjoint, diverges to 1e57
(failed_approaches #28). This is a trajectory property, not a pointwise
bug:

- Pointwise tangents at these thetas are finite (prep-stage jvp of the
  initial-state build: all finite).
- The K_eq overflow NaN-tangent class (cold networks) was already fixed by
  the `_EXP_ARG_MAX` clip in `vulcan_jax/rates_jax.py` and is unrelated.
- Replay v1 (13 cases: 6 offenders spanning both longdy sub-classes, 4
  controls, plus spares): a zero-increment warm re-certification from each
  theta's OWN cold-certified column is tangent-finite in every lane, every
  case. This matches field behavior -- init phase 2 has the same structure
  (zero-increment re-cert) and rarely produces badgrad.
- The NAS events occur on real mutation moves: warm start from the
  particle's carried column at theta_from, solve at theta_prop, with
  early-ladder increments of ~0.5 cloud-std per dimension (step~0.1
  preconditioned), plus the warm_extrapolate first-order seed. Replay v2
  reproduces exactly that structure (outcome in SS9).

Why high-Z/low-C/O: not pinned down mechanistically. The plausible reading
is that metal-dominated, oxygen-rich columns carry stiffer chemistry whose
step-map linearization is closer to marginal stability near the fixed
point, so the warm transient picks up expanding modes more often. The
replay harness (SS9) is the tool for going deeper if it ever matters.

## 6. The fix (commit 941c847)

**Principle.** In MALA, the gradient enters only the PROPOSAL (the drift),
never the target. An MH kernel with drift b(x) is valid for any measurable
b, provided the same b is used in the forward and reverse proposal
densities. So define b(x) = "the AD gradient with non-finite entries
zeroed" and let the certified, finite likelihood decide acceptance. A
badgrad proposal then costs mixing speed (locally a random-walk step), not
correctness. Caveat: b is warm-history-dependent, exactly like the warm
likelihood itself -- the same approximation class already accepted for the
warm kernel and gated per production run by `validate_warm`
(max|dlogL| < 0.1) and `mala_reversibility`.

**Mechanics** (all in `src/retrieval_framework/pipeline.py`):

- The sweep no longer floors a badgrad proposal's L to -1e30. The eval
  already zeroes non-finite gradient entries (`_rt_val_grad`), so the
  reverse density (GT_new) and -- on acceptance -- the carried G use the
  zeroed drift consistently.
- `eval_batch` zeroes the DY rows of badgrad proposals: DY is exactly the
  non-finite tangent stack, and an accepted badgrad particle must not
  poison its next warm_extrapolate seed (its seed degrades to the plain
  carried column).
- `_init_state` phase 2 applies the same policy: badgrad survivors are
  KEPT with zeroed drift (their first MALA move is prior-drift only) and
  zeroed DY rows. Culling or raising would bias the initial importance
  sample against the corner. The raise survives above the same backstop.
- The stub evaluator (`_get_batch_evals`) mirrors the zeroing contract so
  unit tests exercise the real path.
- Config: `smc_tangent_reject_max_frac` (0.05) renamed to
  **`smc_tangent_bad_max_frac`** (default **0.25**). Semantics: a single
  sweep exceeding ceil(0.25 x N) badgrad events is far beyond the measured
  physical class (worst sweep 7.6%) and means systematic AD breakage (e.g.
  a regression NaN-ing every tangent) -- loud raise. 0.0 restores
  zero-tolerance.
- Nothing in the forward model, likelihood, or evidence bookkeeping
  changed: **existing SMC checkpoints remain valid; RESUME=1 is correct.**
  The `tangent_rejected` checkpoint/results key keeps its name for resume
  compatibility (it now counts zero-drift-handled events).

**Validation run for this commit**: `tests/test_smc_gaussian.py` (7,
including the two rewritten badgrad tests), `tests/test_init_state.py` +
`test_uspace_prior.py` + `test_evidence_semantics.py` (15),
`tests/test_warm_extrap.py` + `test_warm_reject.py` +
`test_validate_warm.py` (12, real chemistry) -- all green, twice.

## 7. Operating guide (what to expect in logs now)

- `badgrad=k` per sweep with a WARNING "handled as ZERO-DRIFT MALA moves"
  is NORMAL, especially as the cloud reaches the high-Z posterior region.
  Expect it to TRACK that corner and grow modestly along the ladder.
  Forensics still dump on every occurrence
  (`bad_grad_stage###_sweep#.npz`).
- The anomaly worth investigating is a broad theta-INDEPENDENT badgrad
  rate, or any RT-vjp-side attribution (all 65815 events were
  chemistry-side).
- The run aborts only if one sweep exceeds ceil(0.25 x N) events (36 at
  N=144) -- treat that as a code regression, not physics.
- warmcap and stalled remain REJECTION classes and must still stay ~0 in
  the late ladder (unchanged detailed-balance contract).
- After any production run with this kernel: run `validate_warm` (PASS:
  max|dlogL| < 0.1) and `validation/mala_reversibility.py`, and quote both
  (this is the standing rule, doubly so after a kernel change).
- Ladder pacing on 65815 hardware: ~2 h/stage, beta roughly tripling per
  stage at target ESS 0.6 -> expect ~10-12 stages, i.e. one RESUME=1
  resubmission past the 20 h governor.

## 8. Known open issues (measured on 65815, NOT fixed here)

1. **Warmcap burn**: 68% of (non-badgrad) warm proposals run all 1500
   warm steps before being rejected at the cap -- the dominant per-sweep
   cost. Candidate levers (untested): smaller early-ladder step size,
   lower warm_count_max, better seeding. Needs its own measured pass.
2. **warm_extrapolate A/B**: the gpu preset has warm_extrapolate=True; the
   CLAUDE.md contract has always said to validate it with a SYNTH=1 A/B
   before relying on it, and that A/B has not been run. Replay v2 (SS9)
   informs whether it also aggravates the badgrad class.
3. **Root-cause chemistry investigation**: why the high-Z/low-C/O warm
   trajectories carry expanding tangent modes. The replay harness below is
   the starting point; not blocking production.

## 9. Reproducing every number

- Raw evidence: `runs/w39b_smc_retrieval/forensics_65815/` (13 npz dumps +
  the full job log `VJAX_W39B_SMC.o65815`), versioned in this repo.
- `validation/badgrad_65815/analyze_badgrad.py` -- per-sweep table, repeat
  offenders, longdy/acc distributions, per-dimension percentile z-scores,
  binomial gate statistics (SS3, SS4).
- `validation/badgrad_65815/analyze_badgrad2.py` -- certified-conditional
  rates, corner localization 2x2s, cloud drift by stage, certification
  rate inside/outside the corner.
- `validation/badgrad_65815/analyze_badgrad3.py` -- lnZ/C-O quartile risk
  gradients, offender sub-classes (marginal vs well-converged), stage-2
  splits, backstop projections.
- `validation/badgrad_65815/replay_badgrad.py` -- replay v1:
  zero-increment warm re-certification jvp at offender/control thetas
  (production `chem.converged_y` path, CPU, ~2-4 min/case).
- `validation/badgrad_65815/replay_badgrad_v2.py` -- replay v2: mutation-
  move structure (warm start from a same-stage sibling proposal), plain
  seed vs warm_extrapolate first-order seed, per-lane tangent forensics.
- `validation/badgrad_65815/replay_badgrad_v3.py` -- replay v3: seed-repair
  A/B/C (production clip vs per-cell fallback vs plain) on the v2-reproduced
  cases; reads replay v2's jsonl output.

All scripts run from the repo root in the standard env and default to the
versioned forensics directory.

## 10. Replay v2/v3 outcome: warm_extrapolate's clipped seed manufactures the class

**v2 (move structure, 11 completed cases: 7 offenders + 4 controls).** Each
case warm-starts at a synthesized realistic move (theta_from = the same
particle's proposal in another sweep of the same stage) and jvp's the
production warm-capped solve at theta_prop, once with the plain carried
column and once with the warm_extrapolate first-order seed
max(Y + DY.dC, 0):

- plain seed: **0/11 non-finite** (24 total warm-solve jvps finite across
  v1+v2);
- extrapolated seed: **3/11 non-finite** (off s0w2i7, ctl s1w2i0,
  ctl s1w4i0 -- all at lnZ +1.6..+2.2, exactly the field corner), each with
  the exact production signature: certified finite primal, all-lane NaN
  tangents, chemistry side, whole column.

The reproductions hit field-controls as much as field-offenders: blowing up
is a stochastic property of the (from-state, move, seed) triple, not of the
proposal theta alone -- which is why the field sees a rate, not a region
boundary. The reproduced cases' predictions had 872-1509 cells driven
non-positive by the linear extrapolation (clipped to exactly 0 in
production). Every plain-seed tangent stayed at the healthy 1e15-1e19
scale. Two same-theta solves from different seeds can differ by 15 orders
of magnitude in finite tangent norm (1e3 vs 1e18) -- the through-loop
tangent is strongly trajectory-dependent in this corner even when finite.

**v3 (seed-repair test on the 3 reproduced cases).** Three seeds at
identical (theta_prop, y_from, refs):

| case | max(pred,0) (production) | per-cell fallback where(pred>0, pred, y) | plain |
|---|---|---|---|
| off s0w2i7  | NaN | finite | finite |
| ctl s1w2i0  | NaN | NaN    | finite |
| ctl s1w4i0  | NaN | NaN    | finite |

The per-cell fallback FAILS 2/3: when the prediction is that far out, even
its positive cells are toxic. Plain is 3/3 finite -- and FASTER than the
poisoned extrapolations (accept counts 376-1141 vs cap-burn 1501).

**Fix adopted (this commit): per-particle extrapolation gate** in the
mutation sweep. A proposal extrapolates only if its predicted column needs
no clipping -- every cell either strictly positive or exactly unchanged
from the carried value (so the runner's own clipped zeros and DY=0 rows
from accepted badgrad particles do not disqualify). Any cell driven to
<= 0 makes that particle fall back to its plain carried column WITH its
carried refs (seed and refs switch together; proposal refs are only correct
for extrapolated seeds). This provably converts all reproduced failures to
the never-observed-to-fail plain path, keeps the measured ~1.65x
extrapolation saving on the small late-ladder moves where it matters, and
any residual badgrad tail is still handled unbiasedly by the zero-drift
kernel (SS6) under the 25% backstop.

Expected observable effect on production runs: the per-sweep badgrad count
(7-15/sweep on job 66062 before this change) should drop substantially
once this commit is pulled; whatever remains is the true trajectory-tail
class, still chemistry-side and high-Z-corner-concentrated.
