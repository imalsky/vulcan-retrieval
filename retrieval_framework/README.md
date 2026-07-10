# retrieval_framework/ — reusable differentiable-VULCAN retrieval framework (VULCAN-JAX → ExoJax, SMC + MALA)

The planet-agnostic gradient-informed Bayesian retrieval framework: **live
differentiable VULCAN-JAX photochemistry** chained into **ExoJax** radiative
transfer, sampled by an adaptive-tempered SMC whose mutation kernel is **MALA
driven by forward-mode-jvp gradients** through the whole chemistry→spectrum
chain — the same pipeline/driver/PBS pattern as the SWAMPE WASP-43b phase-curve
retrieval (`SWAMPE_Project/MY_SWAMP/retrieval`), rebuilt for this forward model
with no BlackJAX dependency (the SMC core is ~200 lines of pure JAX, validated
against an analytic Gaussian posterior including its evidence estimate).

**Framework vs case (2026-07-08 reorg).** Everything reusable lives HERE
(`config_schema.py`, `observations.py`, `tp_profile.py`, `retrieval_forward.py`,
`pipeline.py`, the `run_smc.py` driver, `plot_smc.py`, the
`calibrate_count_max.py` / `probe_memory.py` / `smoke_retrieval.py` /
`validate_warm.py` tools, and
`tests/`). Everything planet-specific lives in a CASE DIRECTORY
(`../runs/<case>/`): `case.py` with the `PRESETS` dict + planet identity, the PBS
submit script, `overrides/`, and run outputs. Entry points take the case dir:

```
cd vulcan_exojax_run/
python -m retrieval_framework.run_smc             runs/w39b_smc_retrieval [--calibrate]
python -m retrieval_framework.smoke_retrieval     runs/w39b_smc_retrieval
python -m retrieval_framework.calibrate_count_max runs/w39b_smc_retrieval --n-draws 96
python -m retrieval_framework.probe_memory        runs/w39b_smc_retrieval
python -m retrieval_framework.plot_smc            runs/w39b_smc_retrieval/data/gpu
python -m retrieval_framework.validate_warm       runs/w39b_smc_retrieval
```

`validate_warm` re-solves a finished run's checkpointed cloud COLD and compares
against the warm-carried log-likelihoods -- the direct measurement of the
warm-continuation history dependence (PASS gate: max|dlogL| < 0.1). Run it once
per production run; a published retrieval should quote its result, along with the
prior convergence-acceptance fraction (the operational prior is
p(theta | chemistry converges) -- init logs the reject fraction).

Everything below documents the framework and its engineering history, using the
original WASP-39b application (`../runs/w39b_smc_retrieval/`, real Carter & May
2024 NIRISS+G395H data, `../data/cm24_wasp39b/`) as the running example. Numbers
tied to that case (10-D theta, its priors, 152 bins, band edges) come from ITS
presets in `case.py`, not from the framework.

```
theta ──► VULCAN-JAX (re-converged column, photochemistry ON, nz=50)
  │              │ VMR(nz,ni)
  │              ▼
  ├─► ExoJax Guillot T-P ──► ArtTransPure transmission (premodit opacities + CIA)
  │              │ native depth (Rp/Rs)^2 (λ)
  │              ▼
  └─► lnR0 ──► B (exact trapezoidal binning matrix) ──► + instrument offsets
                 │
                 ▼
      Gaussian likelihood on the 152 real C&M bins (93 NIRISS + 59 G395H, 1.02-5.24 um)
```

## Parameters (gpu preset: 10-D; only the first 6 are chemistry-expensive)

Priors are literature-anchored (Tsai et al. 2023 VULCAN grid + Rustamkulov et al. 2023
PRISM ERS), 2026-07-08; values live in `runs/w39b_smc_retrieval/case.py::_W39B`.

| # | name | prior | role |
|---|------|-------|------|
| 0 | `lnZ` | U(−2.303, 2.303) | metallicity about the 10×solar baseline (1–100× solar; kept wide, data localizes it) |
| 1 | `c_o` | U(−1.70, 0.24) | Δln(C/O) at **fixed O** → C/O ∈ [0.10, 0.70] (Rustamkulov C/O<0.7; upper edge below the b_z>0 validity bound 0.566) |
| 2 | `lnKzz` | U(−4.6, 4.6) | eddy-diffusion multiplier (±2 dex about the GCM baseline Kzz profile; Tsai used ±1 dex) |
| 3 | `Tirr` | U(1100, 2200) K | Guillot irradiation temperature (terminator ~770–1540 K; Teq~1100–1166, SO₂ sweet spot Teq 1000–1600) |
| 4 | `log10kappa` | U(−3.5, 0.5) | Guillot IR opacity κ_th [cm²/g] |
| 5 | `log10gamma` | U(−2, 0.301) | Guillot κ_v/κ_th → γ ∈ [0.01, 2] (weak inversion allowed) |
| 6 | `lnR0` | U(−0.08, 0.08) | reference-radius nuisance (Batalha & Line 2017) |
| 7 | `log10kappa_cloud` | U(−7, 1) | ExoJax `powerlaw_clouds` opacity at 3.5 µm [cm²/g] |
| 8 | `cloud_alpha` | U(0, 6) | cloud/haze power-law slope in ν (0 = gray deck) |
| 9 | `offset_G395H` | U(−800, 800) ppm | flat depth offset vs the NIRISS reference |

> **T-P is drawn RAW (no clip).** A profile that leaves the modelable window
> [300, 3000] K on the ART grid is REJECTED and redrawn (the prior is
> (box) ∩ (in-window); `pipeline.tp_valid`), never bent into range.

## Current configuration & numerics (authoritative, 2026-07-08)

The engineering log further down is HISTORICAL; these are the current values (all
shown in the loud config banner every run prints, via `config_schema.describe_config`):

- **Convergence = VULCAN-master canonical** (Tsai et al. 2017 values): `yconv_cri=0.01`,
  `slope_cri=1e-4`; `yconv_min`/`flux_cri` inherit `vulcan_cfg_W39b`. (Not the sensitivity
  demo's tight `1e-3`.)
- **`count_max=5000` accepted steps, fixed.** A solve that doesn't converge is a failed
  draw — not extended, not clipped. `dt_max=1e11 s` (NOT the master default `1e17`): the
  default let the adaptive step balloon to ~1e16 s on high-Kzz columns and spin without
  settling — the bulk of the old >10k tail. Capping it converges those in ~1000 steps and
  leaves normal columns identical (a step-size control, not a convergence criterion). A
  genuine residual (marginal-`longdy` and photochemical-limit-cycle columns) still fails
  and is rejected at init. Full diagnosis + VULCAN-publication check: `../../CLAUDE.md`.
- **Priors** are literature-anchored (table above) and live in `case.py`.
- **Calibration runs at native R=100 by default** (`calibrate_count_max.py`): `accept_count`
  is resolution-independent, so the RT is run cheap; production uses the real `nu_pts`.
- **RT resolution `nu_pts=1652` (native R~1000) — memory-safe by DEFAULT.** RT-vjp gradient
  memory scales with `nu_pts` (the point count, NOT R), and OOM'd the 96 GB GH200 at 343 GiB
  when `nu_pts=16500`. R~1000 (~11 model pts per binned point) is ample and keeps the RT-vjp
  at ~34 GiB. The schema default is now 1652 (was 6000, a ~R10000 memory bomb) and
  `validate_config` warns above `nu_pts=2500`; run `PROBE_MEMORY=1` before ever raising it.
- **The SMC mutation rejects non-converged warm proposals** (fixed 2026-07-09). A MALA
  proposal whose warm chemistry continuation hits `count_max` is treated as an MH rejection
  (`L=-inf`), not fed into the gradient — the warm-side analogue of the cold-init
  reject-and-cull. Without it a non-converged proposal's garbage jvp/RT-vjp tripped `n_bad`
  / NaN'd at SMC stage 0 (the bug that killed job 64604 and the calibrations). See
  `../../CLAUDE.md` and `tests/test_warm_reject.py`.

RT carries **8 molecules** (H2O, CO2, CO, CH4, SO2 + **HCN, C2H2, H2S** — the
high-C/O and reduced-sulfur discriminators, so the likelihood can actually see the
species that decide the C/O upper tail) plus **H2-H2 and H2-He CIA**. The cloud is
ExoJax's shipped retrieval cloud (`exojax.atm.simple_clouds.powerlaw_clouds`, pRT
convention, per gram of atmosphere, uniformly mixed); cloud dims are RT-only, so
they ride the cheap gradient block like lnR0 — ~zero cost against the GPU budget.
H2/He **Rayleigh scattering** (ExoJax `xsvector_rayleigh_gas`, zero free parameters)
is on by default — required once the band reaches 1 µm, else its slope leaks into
the haze posterior.

The T-P is **ExoJax's own** `exojax.atm.atmprof.atmprof_Guillot` (Guillot 2010
Eq. 29; plain `jnp.exp`, forward-mode-clean), evaluated on *both* the VULCAN grid
(chemistry re-converges under it; rate constants rebuilt on-graph) and the ART grid
(RT scale height) — one self-consistent profile, hooked into `vulcan_chem` via the
new backward-compatible `tp_eval` argument. `Tint=150 K` and `f=1/4` are fixed
(config fields). Optional: `infer_noise_inflation` adds a Line-2015-style ×σ term.

**Two-stage solve (`two_stage_z`, default on).** Measured 2026-07-05: perturbing the
cold EQ init's metals and re-converging through a *T-displaced* transient (any
retrieved T-P ≠ the baked profile) erases the inventory perturbation — converged CO
changed 1e-11 for a 5 % metals step under a Guillot T-P, vs the exact 5 % at
T=T_base. The forward therefore (1) converges the column at the retrieved T-P/Kzz
with baseline composition, then (2) applies the lnZ / C-O scaling to that converged
column and re-converges warm (`converged_y(warm_y=…)`, the Hessian-campaign
continuation pattern + `reanchor_atom_ini`). Stage 2 is cheap (warm start) and
gentle (uniform metal scaling, no species crash), so the inventory — and with it
the lnZ/C-O gradient — survives. `smoke_retrieval.py` hard-fails if those
gradients go dead again.

## Why forward-mode MALA (and how it stays affordable)

The VULCAN runner is a `lax.while_loop`: **jvp works, vjp does not**. The
likelihood gradient is therefore assembled from forward-mode jvps and exposed to
the MALA kernel through a `custom_vjp` (the SWAMPE trick). Two cost structures are
implemented (`gradient_mode`):

- **`block` (default, exact):** only the 6 chemistry+T-P directions push tangents
  through the VULCAN loop; `lnR0` is one RT-only jvp at the frozen converged
  profiles; offsets/noise gradients are analytic. ~25–35 % cheaper per MALA step.
- **`naive`:** all dims through the full chain (SWAMPE-identical), kept as a
  cross-check. `smoke_retrieval.py` asserts block ≡ naive to fp precision and
  validates both against re-converged finite differences.

**Staged batched hot path (2026-07-06 rework).** The per-particle modes above are
kept for validation, but the SMC itself runs a STAGED evaluator that splits the
chain at the chemistry/RT boundary — because the two halves have opposite
economics: the chemistry `while_loop` has tiny per-lane state (~MB; width is
nearly free and is what keeps the GPU busy) but only supports jvp, while the
ExoJax PreMODIT RT is vjp-capable but costs ~GB of intermediates per lane.
So per mutation sweep:

1. chemistry: 6 fwd-jvp lanes per particle, ALL particles in ONE wide batched
   `while_loop` (N×6 tangent-augmented columns — 864 at the gpu preset's N=144,
   nz=50). The warm accept_count diagnostic rides this same jvp'd chain (it is
   part of the runner's primal carry, integer-valued so tangent-free) — the
   2026-07-09 rework; an earlier version ran a SECOND primal-only `while_loop`
   just to read it, doubling the chemistry wall time per sweep;
2. RT: ONE reverse-mode vjp per particle (no while_loop inside the RT, so this is
   legal), `lax.map`-chunked over particles (`smc_rt_vjp_chunk`); the single
   backward pass replaces 6 forward RT tangents + the 3-dim lnR0/cloud jacfwd,
   and its aux-cotangent is contracted against the chemistry tangents
   (`dL/dθ_k = ⟨∂aux/∂θ_k, ∂L/∂aux⟩` — same chain rule, regrouped, asserted
   identical to `block` in the smoke);
3. offsets/noise: analytic.

The mutation kernel also CARRIES each particle's converged column
(`smc_chem_mode="warm"`, default): every proposal's chemistry warm-continues from
the particle's own state with incremental lnZ/C-O scaling (the validated
Hessian-campaign continuation pattern; `converged_y(warm_y=...)`) instead of
re-running the full cold two-stage solve — measured ~500–800-step re-converges
for MALA-sized moves (the `conv_step=500` longdy certification window dominates
the warm floor, not `count_min`), still ~6–8× fewer chemistry steps than cold.
Don't shrink `conv_step` to buy speed: probed 500→300 (2026-07-10), it saves
NOTHING on extrapolated seeds and 7% on plain ones while certifying a state
0.07 dex off the 500-window one — see `../CLAUDE.md` for the full numbers.
MUTATION warm solves run under `warm_count_max` (default 1500, a twin runner
with the smaller cap baked in): a proposal still unconverged there is REJECTED
(-inf L) instead of dragging the whole lockstep batch to the cold `count_max` —
without this, any single bad proposal among N gated EVERY early-ladder sweep at
5000 steps (the ~3-6 h/stage pathology of job 64745; while the cloud is
prior-like that is essentially every sweep). The INIT gradient pass (phase 2)
runs the same warm map UNCAPPED (`batch_eval_init_vg`, gated at the cold
`count_max`): its inputs are phase-1 survivors re-certifying from their own
converged columns — proven-convergent particles, not disposable proposals — and
a marginal survivor can legitimately need more than the mutation cap just to
re-certify (NAS job 64854: the cap gated 5/96 healthy survivors into a spurious
"crippled cloud" raise). Optional on top: `warm_extrapolate=true` seeds each
proposal's warm solve at the first-order prediction `Y + (dy/dθ)·Δθ` using the
tangents the gradient pass already computes (measured 1.65x fewer warm steps,
same certified state; opt-in pending a SYNTH A/B — see the lever list below and
`config_schema.py`). The cold
two-stage map runs exactly once per particle, at state initialization (and on
`smc_chem_mode="cold"`, which restores the published solve-from-baseline map for
every evaluation). The carried
likelihood also serves the tempering reweight, so a stage costs ~one mutation
call. Caveat (documented, deliberate): with warm continuation the likelihood is
defined by the continuation map from the particle's own history — path-dependent
at the yconv tolerance level; the smoke FD-checks the warm gradient against the
identical warm map.

## Sampler

`pipeline.run_smc_loop`: Del Moral resample–move SMC — ESS-bisection temperature
ladder (`target_ess_frac=0.6`), systematic resampling, then
`smc_num_mcmc_steps=6` preconditioned-MALA sweeps per stage (published
MALA-within-SMC practice is 3–10 with a good preconditioner; the old 12 was ~2×
generous and each sweep costs one full batched gradient). The proposal is
preconditioned with the **absolute per-dim std of the freshly resampled cloud**
(so it narrows in lockstep with tempering — the Gaussian test showed the
SWAMPE-style unit-geometric-mean scaling collapses acceptance after large β
jumps), plus Robbins–Monro fine-tuning of the scalar step toward 0.55 acceptance.
Every sweep logs a heartbeat line (acceptance, rejected-proposal count,
n_bad_grad) via `jax.debug.callback`, so a slow stage is visible per-sweep
instead of hours of silence. Per-stage atomic checkpointing
(`smc_checkpoint.npz`) and a **walltime governor** (gpu preset: stops cleanly at
20 h inside the 24 h PBS wall) guarantee a usable posterior from any job;
RESUME=1 continues a stopped ladder.

## Quickstart (local, vulcan env)

```bash
conda activate vulcan
cd vulcan_exojax_run
python -m pytest retrieval_framework/tests/ -q   # binning matrix, u-space prior, SMC-on-Gaussian (~2 s)
python -m retrieval_framework.smoke_retrieval runs/w39b_smc_retrieval   # FD + block≡naive gradient checks (~10-30 min)
SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
python -m retrieval_framework.plot_smc runs/w39b_smc_retrieval/data/smoke
```

## GH200 (NAS)

One-time: `scp -r` the tree (at minimum `VULCAN-JAX/` and `vulcan_exojax_run/`
incl. `data/`, which carries its own opacity cache -- no other project needed) to
`/nobackup/$USER/VULCAN_Project` (NEVER rsync -- see `../CLAUDE.md`).
The PBS reuses the shared `pyt2_8_gh` env (+ `pip install --user vulcan-jax`,
exojax already present) exactly like the working SWAMPE and VULCAN-JAX-benchmark
jobs, and exports `VULCAN_PROJECT_ROOT` so the bundle's paths resolve.

```bash
cd /nobackup/$USER/VULCAN_Project/vulcan_exojax_run/runs/w39b_smc_retrieval
qsub -v CALIBRATE_ONLY=1 run_nas_w39b.pbs   # 1) timing projection (timing.json)
qsub -v SYNTH=1 run_nas_w39b.pbs            # 2) gpu-fidelity injection/recovery test
qsub run_nas_w39b.pbs                       # 3) the real-data production run
tail -f logs/W39B_SMC_<jobid>.log
```

Run 1 reports measured per-stage cost and projected totals for 15/25/40 stages
against the 20 h governor **before** you commit a full job. Run 2 must put the
injected truths inside their 90 % intervals before run 3's posterior is trusted.

## GPU budget (24 h PBS wall, 20 h SMC governor)

Cost ≈ `init + stages × mutation call` (the reweight uses the carried likelihood,
so it is free). `init` = one batched COLD two-stage solve per particle,
LIKELIHOOD-ONLY at full width, followed by one warm-map gradient sweep from the
just-converged columns (the 2026-07-07 fix, §K — the old cold-gradient init cost
>16 h). The cold pass is the only solve-from-baseline work in the run, gated by
the slowest prior-corner particle (count_max=5000 accepted steps bounds it, and
`dt_max=1e11` keeps the adaptive step from ballooning — see the authoritative block
above and `../../CLAUDE.md`). A mutation call = `6 MALA sweeps ×` (ONE wide
batched warm chemistry re-converge with 6 jvp lanes/particle, `warm_count_max`-
capped, + chunked RT vjp passes). The warm cap is what bounds the early ladder:
without it, any single non-convergent proposal among N gated every sweep at the
full cold count_max (job 64745: ~3-6 h/stage; see `../../CLAUDE.md` "Mutation
sweep cost").
The adaptive ladder typically needs ~12–25 stages. Run `CALIBRATE_ONLY=1` after
any config change — it times init + one mutation call and projects 15/25/40
stages. The governor makes the budget a guarantee rather than a hope; if
calibration says a stage is slow, the levers in order of pain:

1. `warm_extrapolate=true` (opt-in, wired 2026-07-10): seed each proposal's warm
   solve from `Y + (dy/dtheta)·dtheta` using the tangents the gradient pass
   already computes — measured 780 → ~470 typical warm steps (1.65x), same
   converged column (parity unit-tested). Validate once with a SYNTH A/B, then
   also drop `warm_count_max` 1500 → ~800 for the second half of the win.
2. `warm_count_max` 1500 → ~1000 (bounds the worst-case sweep; measured typical
   plain warm re-converge is ~500-800 steps, so watch the per-sweep rejected
   count in the heartbeat lines for collateral rejections)
3. `smc_num_mcmc_steps` 6 → 4 (linear savings, some mixing loss)
4. `nz` 50 → 40 (chemistry cost ~linear in nz)
5. band: drop to G395H-only (halves n_bin; loses NIRISS lever + offset)

(`yconv_cri` is NOT a speed lever here — it's fixed at the master `0.01`, and the
operative gate is the loose branch anyway; the real step-count lever was `dt_max`.)

(There is deliberately NO gradient-free kernel fallback — project rule: loud
errors over silent degradation.)

Note on `smc_num_particles` (updated after the 2026-07-07 probe): the chemistry
— primal AND gradient — runs full-width (`smc_chem_chunk=0`), so N widens those
kernels nearly for free (576 lanes at N=96 ≈ the paper's batch-256 benchmark
width; the GH200 power trace confirms it — wattage rises with lane count while
step time barely moves) — which is why the gpu preset ships N=144 as of
2026-07-10 (raised from 96 to spend the measured ~300-of-700 W headroom on
particles; see `../CLAUDE.md` "GPU power headroom"). The ONLY N-linear cost is the
RT VJP stage (ceil(N/`smc_rt_vjp_chunk`) sequential blocks; the per-lane vjp
memory scales with nu_pts — 18.4 GiB/lane at nu_pts=5000, ~3× less at the
production 1652, so the gpu preset runs 12-wide: 96/12 = 8 blocks, the same
serial count as the old 48/6). Shrinking that per-lane cost (checkpoint
granularity / xs tables / fp32 xs inside exojax_rt) is still the top structural
speed item — it unlocks full-width RT.

Remaining documented-not-wired speedup: the reverse-mode steady-state adjoint
(`steady_state_grad`), which would make the chemistry gradient cost
dimension-independent.

## Outputs (per run dir, e.g. `data/gpu/`)

`config.json` (full config + param metadata) · `observations.npz` ·
`smc_checkpoint.npz` (atomic, per stage) · `posterior_samples.npz`
(`samples (chains, n, dim)`) · `smc_extra_fields.npz` (β ladder, ESS, acceptance,
step sizes, unique counts, logZ) · `posterior_predictive.npz` (PPC band + median
model + χ²) · `timing.json` (calibrate mode) · `plots/{corner, spectrum_fit,
tp_posterior, smc_diagnostics}.png` (PNG only, dpi 200).

## Honest limitations

- **Clouds are the parametric ExoJax power-law** (gray deck at α=0, haze slope
  otherwise), uniformly mixed — not microphysical. The ExoJax-native upgrade is the
  Ackerman & Marley stack (`exojax.atm.amclouds` + `PdbCloud("MgSiO3")` + `OpaMie`),
  with the cloud base at the retrieved T-P's `psat_enstatite_AM01` crossing and
  particle sizes from the **retrieved Kzz** — the honest "self-consistent-lite."
  Blocked only by `pip install PyMieScatt` (not in the envs) + a one-time miegrid
  build. Note VULCAN itself **cannot** do W39b clouds self-consistently: its
  condensation set is H2O/NH3/H2SO4/S2/S4/S8/C (cool-planet condensates) and Mg/Si
  aren't in the H,O,C,N,S atom set — silicate condensation would be major new
  chemistry, and the gradient path is validated conden-off.
- No stellar-contamination term (quiet G8 host; instrument offsets absorb residuals).
- HITRAN (296 K reference) opacities for H2O/CO2/CH4/SO2/HCN/C2H2/H2S, ExoMol for
  CO — adequate for the methodology; swap to HITEMP/ExoMol for publication-grade
  line fidelity (code-wise a dict entry; operationally multi-GB downloads + premodit
  memory tuning — the one upgrade with real wrangling risk).
- The chemistry's internal transport geometry (dz, Dzz structure) stays at the
  baseline W39b profile; the retrieved T-P drives kinetics, densities, and the RT
  scale height (the spectrum's T response is fully carried). Same approximation as
  the validated `fig_so2_temperature` pattern.
- If the governor (or a crash) stops before β=1 the samples are **tempered**
  (`reached_beta1=False` in the outputs) — widths are lower bounds. Resubmit with
  `qsub -v RESUME=1 run_nas_w39b.pbs`: the ladder continues from the checkpointed
  cloud/β instead of restarting (validated in `test_smc_gaussian.py`).
- Observations are baked into the jitted likelihood at first trace:
  `set_observations` must be called exactly once, before inference (the driver
  enforces this ordering).
- `use_photo=True` is required — the forward-mode tangent is only validated
  photo-on (see `../config.py` FULL notes).

---

# Development log & findings (2026-07-04/05) — read before extending

> **HISTORICAL.** Config values quoted below (e.g. `count_max=3e4`/`5000`,
> `yconv_cri=1e-3`, the old Tirr/C-O priors, the T-clip) are the values *at the time of
> each entry* and trace the evolution. For the CURRENT config see the "Current
> configuration & numerics (authoritative)" section above and `../../CLAUDE.md`.

Everything below is the full record of what was found, measured, decided, and
verified while building this. Nothing here is speculative; every number was
measured in this tree.

## A. The inventory-erasure finding (the big one)

**Symptom.** With the Guillot T-P hooked into the chemistry, `dL/dlnZ ≈ 1e-20` —
and central finite differences AGREED it was zero (ΔL ~ 4e-11 for a 0.5 % metals
step). Not an AD bug: the converged *primal* did not respond to the initial-metals
scaling at all. Meanwhile `lnKzz`/`Tirr`/`kappa` gradients were healthy.

**Why the knob was expected to work.** The lnZ / C-O knobs scale the *initial*
abundances (`y0p = y0·exp(lnZ·metal_mask)`), relying on: zero-flux boundaries ⇒
the elemental column inventory is conserved ⇒ the converged state remembers the
init exactly through the conserved totals. At `T = T_base` this is true and
measured: +5 % lnZ step → **+5.068 %** converged CO (index ≈ 1.0), total atom
drift 5e-7 over a full run.

**Bisection (nz=30, CO-only, ±0.05 lnZ steps, separate processes):**

| probe | config | converged-CO response |
|---|---|---|
| A | legacy interface (tp_eval=None, proxy C/O, no reanchor) | **5.068e-2 (alive)** |
| B | A + Guillot `tp_eval` ONLY | **1.419e-11 (dead)** |
| C | A + fixed_O C/O + reanchor ONLY | **5.068e-2 (alive, bit-identical to A)** |

The T-P hook alone kills it. C also proves the `reanchor_atom_ini` einsum and the
fixed-O `b_z` machinery are exact no-ops at baseline.

**Mechanism (deep-diag proven, `run_diag` on the dead config).** Every accepted
step the runner rebuilds each layer's total density hydrostatically:
`sol_balanced = M[:,None]·ymix`, `M = pco/(kB·T)`. This is a faithful port of
VULCAN-master's own step (`op.py:909`, comment `# MAINTAINING HYDROSTATIC
BALANCE`, `var.y = np.vstack(atm.n_0)*var.ymix`). Consequences, measured:

- init atom totals carried the injection correctly (C: 3.047e17 → 3.204e17, +5.1 %),
- final totals were **identical to all printed digits** between perturbed and
  unperturbed runs (C: 4.53897e17 both; even the solver counters matched:
  accept=4484, delta=306, loss=6, t=1.023e9),
- final totals sat **+49 % above the init totals for BOTH runs** (H: 1.033e20 →
  1.539e20): under a displaced T the renorm+transport combination is NOT
  conservative — the column's elemental content relaxes to an attractor set by
  M(T) and the baseline's basin, forgetting the init entirely.

With the conserved-inventory channel destroyed, the fixed point is unique and
init-independent ⇒ the derivative is *exactly* zero (tangent contracts like any
non-conserved direction), and FD agrees. `T`/`Kzz` survive because they enter the
loop body every step (atm arrays, rate table), not through the init. The
atom-loss accept/reject can't catch it: `loss_diff` is *incremental*
(`|atom_loss_new − atom_loss_prev| < loss_eps`), so a slow secular drift passes.

**Why nobody ever saw this.** (i) Every published VULCAN(-JAX) lnZ result
(fig_metallicity_sens, the sensitivity demo, the README FD numbers) evaluates at
`T ≡ T_base`, where the init is M-consistent, the renorm is a no-op, and the
inventory really is conserved. (ii) In normal VULCAN usage metallicity is set via
the elemental-abundance file → FastChem EQ init built ON the run's own T-P —
init and n₀ always agree by construction. (iii) Most other photochem codes
(Atmos, photochem) anchor composition at *boundary conditions*, not the init, so
init-forgetting is a feature for them. The failure needs the retrieval-specific
combination: T(θ) moving AND composition encoded in the init, in one call.

**Fix: the two-stage solve (`two_stage_z=True`, default).**
1. converge at (T(θ), Kzz(θ)) with baseline composition — the violent
   T-relaxation happens on the baseline, where the inventory rebuild is harmless;
2. scale metals / C-O on that **converged** column and re-converge warm
   (`converged_y(warm_y=…, lnZ_ref=0, c_o_ref=0)` + `reanchor_atom_ini`). The
   start state is M(T)-consistent, so the renorm only rescales totals while the
   enrichment lives in the **ratios**, which it preserves; the gentle re-converge
   re-partitions speciation without a violent transient.

Measured after the fix (same dead configuration): +5 % lnZ → **+5.291 %** CO;
a full e-fold metals kick (×2.72) retains **98.4 %** of its carbon through
stage 2 (C ratio 2.676 vs 2.718 ideal); FD table below. Cost: stage 2 is warm,
~1.2–1.4× one solve total. This is the SO2-Hessian-campaign continuation pattern
(which hit the same class of init-forgetting as "snap-to-baseline" and validated
warm-started jvp at nz=100).

**Rule for all future work:** any retrieval_framework/scan that moves T *and* uses the
y0-composition knobs must apply the composition perturbation to a T-converged
column, never to the cold EQ init. `smoke_retrieval.py` hard-fails (liveness
guard, threshold 1e-3) if `|dL/dlnZ|` or `|dL/dc_o|` ever go dead again.

## B. Sampler lessons (self-contained SMC core)

- **No BlackJAX** in the `vulcan` env or NAS `pyt2_8_gh` → `pipeline.py` carries a
  ~200-line pure-JAX Del Moral resample–move SMC (ESS-bisection β ladder,
  systematic resampling, preconditioned MALA mutation). Validated on an analytic
  Gaussian: posterior mean/std recovered, β ladder strictly increasing to 1 in
  7 stages, 256/256 unique particles, **logZ = −8.219 vs analytic −8.374** (the
  evidence bookkeeping is right, not just the samples).
- **Preconditioner must be ABSOLUTE, not shape-only.** The SWAMPE kernel
  normalizes the diagonal scale to unit geometric mean, leaving the overall
  proposal *width* to Robbins–Monro on the scalar step — which lags the ladder
  and collapses acceptance after large β jumps. Reproduced here: final-stage
  acceptance **0.019** (the same failure signature as SWAMPE's WASP-43b pilot,
  accept=0.001 / 25 unique). Fix: `scale_diag = per-dim std of the freshly
  RESAMPLED cloud` (clip [1e-3, 20]) so the proposal narrows in lockstep with
  tempering; RM only fine-tunes toward target accept 0.55. After: acceptance
  0.62–0.91 across every stage.
- Governor (`walltime_seconds`) + atomic per-stage checkpoint + `resume_from`
  (RESUME=1) are all unit-tested (`test_smc_gaussian.py`): a killed ladder
  resumes from its tempered cloud and completes with the correct posterior.

## C. Gradient architecture

- The runner's `lax.while_loop` has jvp but **no vjp** → likelihood gradients are
  forward-mode, exposed to MALA's `value_and_grad` via `custom_vjp` whose fwd
  computes (value, full gradient) from n forward passes (the SWAMPE trick).
- **`gradient_mode="block"` (default, exact):** only the `n_chem_tp` chem+T-P
  directions push tangents through the VULCAN loop; `lnR0` is a single RT-only
  jvp at the frozen ART-grid aux profiles (`native_depth_aux` / `rt_depth`);
  instrument offsets and noise-inflation are analytic. Exact because the blocks
  enter μ through disjoint sub-graphs. Verified block ≡ naive to **3e-8**
  (2-stage) / 3.3e-11 (1-stage). Savings ≈ (n_dim − n_chem_tp)/n_dim: 2/8 = 25 %
  on the gpu preset (in the 6-dim smoke only lnR0 is cheap, so block≈naive there).
- Non-finite-gradient policy (loud-error rule, 2026-07-06): a non-finite DEPTH is
  an MH rejection (−1e30 sentinel — principled, documented). A finite depth with a
  non-finite GRADIENT is an AD pathology: the staged evaluator counts these
  in-jit (`n_bad_grad`) and the driver **raises `RuntimeError`** — it is never
  silently zeroed into a random-walk step. (The legacy per-particle `_fwd`
  custom_vjp still zeroes — it survives only as a smoke-test validation path, not
  in the SMC hot path.) Cold initialization likewise raises on any non-finite
  particle instead of letting resampling silently cull it.
- **GH200 post-mortem (2026-07-06 jobs 63886/63972/63995/63997) — why the
  all-in-one gradient was redesigned.** Job 63886 (N=48, all-particles vmap):
  `jit_mutate` requested **1.52 TiB**; XLA rematerialization bottomed out at
  1.06 TiB and the executable exceeded the 2 GB protobuf cap. Chunking to 4
  particles (job 63997) still peaked at **120.9 GiB** vs the ~87 GiB pool →
  ~25–30 GB per particle-gradient, i.e. ~3–5 GB per FORWARD tangent lane through
  the ExoJax PreMODIT cross-section math (the chemistry lanes are ~MB). Meanwhile
  chunking serialized the expensive `while_loop` into 24 sequential 14-lane
  calls: nvidia-smi showed "100 % util" at **~200 W / 700 W** — launch-latency-
  bound tiny kernels, not compute (the flat 88.5–89.8 GB "memory used" was just
  the `XLA_PYTHON_CLIENT_PREALLOCATE=true, MEM_FRACTION=0.90` pool; the
  `MEM_FRACTION=0.98` experiment broke executable-constant allocation — keep
  0.90). The staged evaluator fixes both ends: chemistry full-width (wide batched
  kernels), RT gradient via ONE reverse-mode vjp per particle instead of 9–10
  forward lanes.
- **`smc_rt_chunk` (16) / `smc_rt_vjp_chunk` (schema 6, gpu preset 12 at nu_pts=1652):** particles per `lax.map` chunk
  through the RT stage (primal / gradient sweeps). The RT vjp tape is bounded by
  per-molecule `jax.checkpoint` in `exojax_rt._accumulate_dtau` (without it the
  backward pass stores every molecule's PreMODIT intermediates: ~30–50 GB per
  spectrum). PROBE-MEASURED 2026-07-07 (compile-only, nu_pts=5000): even WITH the
  checkpoint the RT VJP costs **18.4 GiB for the first lane + ~9.4 GiB per
  additional lane** (65.4 GiB at 6-wide, vs the ~81 GiB pool) — it is THE memory
  wall of the whole evaluator, and it scales with n_nu. Do not raise
  `smc_rt_vjp_chunk` without a fresh `PROBE_MEMORY=1` pass. RT PRIMAL is only
  ~0.22 GiB/lane (full width fine). `0` = single all-particles vmap. Verified
  chunk-invariant to fp64 precision (padding included).
- **`smc_chem_chunk` (0 = full width, the default since 2026-07-07):** particles
  per `lax.map` chunk through the CHEMISTRY GRADIENT stage. CORRECTION
  (probe-measured 2026-07-07): staged chem tangent lanes cost **~20 MB per
  lane-pair** (0.78 GiB at 36 lanes; nu-independent), NOT the ~1.3 GB this doc
  previously claimed — that figure was the 2026-07-06 all-in-one architecture's
  PreMODIT tangents (the 390 GiB OOM) misattributed to photo temporaries. The
  chemistry gradient therefore runs UNCHUNKED: 288 lanes at N=48, 576 at N=96
  (~12 GiB), one wide while_loop — no sequential chem blocks at all. Primal
  chemistry is ~55 MB/lane (5.3 GiB at 96-wide).
- The binning is a precomputed exact linear matrix **B** (trapezoidal bin-average
  as a matrix; tested to 1e-12 against a trapz reference on the real C&M bins),
  so the binned depth's derivative is exact and free.
- Final FD validation (smoke, 2-stage, h=1e-3 in u, re-converged central diffs):

  | dim | AD | FD | rel |
  |---|---|---|---|
  | lnZ | −2.06293e+1 | −2.06331e+1 | 1.9e-4 |
  | c_o | −4.18315e+1 | −4.18315e+1 | 6.5e-8 |
  | lnKzz | −1.42502e+0 | −1.42113e+0 | 2.7e-3 |
  | Tirr | −3.22068e+2 | −3.22091e+2 | 7.0e-5 |
  | log10kappa | −4.67619e+1 | −4.67620e+1 | 1.6e-6 |
  | lnR0 | −3.55280e+2 | −3.55280e+2 | 1.9e-7 |

- Legacy path regression: the parent `smoke_test.py` (tp_eval=None) still passes
  (lnZ jvp 5.10e-4 vs FD −5.28e-4, rel 3.4e-2 — its historical tolerance).

## D. Model-setup gotchas

- **NIRISS SOSS order 1 ends at 2.83 µm.** A 2.9–5.2 µm model band silently drops
  ALL NIRISS bins → no inter-instrument offset, no 2.7 µm water lever. The gpu
  band is 1.01–5.26 µm (nu 1900–9900, 1652 pts / R~1000; was 16500 pre-OOM) → 93 NIRISS + 59 G395H bins
  (widened from 2.0 µm at the 2026-07-05 pre-launch review: +64 NIRISS bins with
  median σ 70 ppm covering the 1.1–1.9 µm water bands + the haze-slope lever, at
  zero chemistry cost — the band only touches the cheap RT).
- **Guillot priors:** Teq(W39b) ≈ 1166 K ⇒ Tirr = √2·Teq ≈ 1650 K; with f=1/4 the
  skin T ≈ 0.70·Tirr ≈ 1150 K (the JWST limb estimate). Truth Guillot at
  (1650, κ=10⁻², γ=10⁻⁰·⁴, Tint=150): isothermal ~1057 K aloft → ~1320 K at 7 bar
  — a sane W39b terminator, well inside the T-clip [320, 2980] K (opacity-table
  bounds; clip gradient is zero at the rails by design).
- T-P is evaluated by the SAME ExoJax `atmprof_Guillot` on both the VULCAN grid
  (chemistry: rates rebuilt on-graph via `rates_jax`, M = P/kT, pv carry) and the
  ART grid (RT scale height) — one self-consistent profile; ExoJax's Guillot uses
  a plain `jnp.exp` (no exponential-integral E₂), so it is forward-mode-clean
  (the Heng+14 `expn` jvp pathology lives in VULCAN's own `build_atm`, bypassed
  entirely by feeding `Tco`).
- `vulcan_chem.build_chem_model` gained `tp_eval`/`n_tp_params` —
  backward-compatible: `tp_eval=None` reproduces the published scalar-`T_int`
  path bit-for-bit (regression-checked via the parent smoke).
- Observations are baked into the jitted likelihood at first trace (closure):
  `set_observations` exactly once, before any jitted call.

## E. Environment / deployment facts

- NAS env: the same **`pyt2_8_gh`** shared conda env as SWAMPE and the VULCAN-JAX
  GPU benchmark (`module use -a /swbuild/analytix/tools/modulefiles; module load
  miniconda3/gh2`), caches at `/nobackup/$USER/.vulcan` (PYTHONUSERBASE, pip,
  **JAX_COMPILATION_CACHE_DIR** — persists compiles across calibrate/synth/real
  jobs), NAS proxy for first-run HITRAN downloads. GH200 XLA knobs from the
  validated benchmark: `PREALLOCATE=true`, `MEM_FRACTION=0.90`,
  `--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0`.
- **FastChem is a per-architecture C++ binary** (the EQ init runs it once at
  build). The repo checkout ships macOS-arm64; on GH200 the PBS probes candidate
  trees **by executing them** (OSError = wrong arch) and prefers the
  pip-installed vulcan-jax's tree (built by vulcan-emulator `run_install.pbs`),
  exported via `VULCAN_JAX_FASTCHEM_DIR` (read at first import).
- `config.py` / `zco_lib.py` roots are overridable via **`VULCAN_PROJECT_ROOT`**
  (default = the local absolute path). rsync list for NAS: `VULCAN-JAX/` and
  `vulcan_exojax_run/` (incl. `data/`, which carries its own
  `data/opacity_cache/` -- cached CO ExoMol + H2-H2/H2-He CIA, no other project
  needed as of 2026-07-07).
- numpy 1.x/2.x split: `np.trapz` was removed in numpy 2 and `np.trapezoid`
  doesn't exist in 1.26 (the `vulcan` env). The tree uses a module-level
  `_trapezoid = getattr(np, "trapezoid", None) or np.trapz` alias
  (`zco_lib`, `fig_fisher_forecast`, `test_binning`) so both majors work --
  use that pattern in new code, never either name directly.
- Memory on GH200 at N=48/nz=50: dominated by the convergence ring buffer
  (`y_time_ring`, ~18 MB/lane) × ~288 tangent-augmented lanes × primal+tangent
  ≈ 10–12 GB — comfortable; no device-batch tiling needed (the 512×nz150
  benchmark OOM regime is far away).

## F. Literature context (verified online, 2026-07-05)

**The two-stage/continuation pattern is field-standard:** Atmos's own convergence
criterion re-runs from the previous output and its workflows step parameters with
re-convergence; the Agúndez pseudo-2D method (basis of the Baeyens grids) *is*
warm-start-under-changing-T — a converged substellar column re-converged
continuously as T changes with longitude; CHEMKIN flame solvers' `CNTN`
continuation from previous solutions is canonical in combustion. Every published
VULCAN-style metallicity study regenerates a consistent EQ init at the target Z
on the run's own T-P — the community never relied on inventory-scaling through a
violent T-transient either.

**Precedent map for the retrieval itself:**
- Kinetics inside a retrieval: **once** — FRECKLL/TauREx (Al-Refaie, Venot,
  Changeat & Edwards, ApJ 2024): "the first time a full disequilibrium kinetic
  retrieval … is attempted." MultiNest (gradient-free, 750 live pts, ~40k
  samples), **simulated** JWST HD 189733 b (NIRISS+G395M), **isothermal** T,
  5 free params; forward = 23 s (reduced net) / 3.2 min (full); retrieval 8 h /
  24 h on **180 CPU cores**. Their bias result motivates kinetics retrievals:
  fitting their solar-Z kinetics truth with equilibrium chemistry returned
  Z ≈ 32 (reduced network: Z ≈ 6).
- Gradient-based retrievals: established via ExoJAX (HMC-NUTS on real spectra,
  incl. Gl 229 B) — but gradients through **RT only**, parametric/equilibrium
  chemistry; never through a kinetics solver.
- Tempered SMC: textbook Bayesian computation (Del Moral; PyMC ships adaptive
  tempered SMC) but not established in exoplanet atmospheric retrieval.
- Wogan's `photochem`: forward grids feeding separate retrievals (K2-18b);
  kinetics not in the sampler; no AD; composition anchored at BCs so the
  inventory trap structurally can't occur there.

⇒ **Real JWST data + full photochemical kinetics + AD gradients through the
solver + SMC-MALA appears unprecedented.** Frame any writeup carefully with
FRECKLL as the nearest precedent (and note the jax_paper deliberately avoids
claiming HMC-style retrievals are enabled — this bundle is the actual attempt,
made feasible by vmapped particles + forward-mode MALA + a 24 h GH200 budget;
FRECKLL's 8–24 h × 180 cores for gradient-free sampling of a simpler model is a
useful external calibration).

Key links: [FRECKLL (arXiv:2209.11203)](https://arxiv.org/abs/2209.11203) ·
[Agúndez+14 pseudo-2D (arXiv:1403.0121)](https://arxiv.org/abs/1403.0121) ·
[Baeyens+22 grid II](https://academic.oup.com/mnras/article-abstract/512/4/4877/6554558) ·
[ExoJAX (ApJS 2022)](https://iopscience.iop.org/article/10.3847/1538-4365/ac3b4d) ·
[ExoJAX2 (arXiv:2410.06900)](https://arxiv.org/abs/2410.06900) ·
[Gl 229B HMC retrieval (arXiv:2410.11561)](https://arxiv.org/abs/2410.11561) ·
[PyMC tempered SMC](https://www.pymc.io/projects/examples/en/latest/samplers/SMC2_gaussians.html) ·
[Wogan+24 K2-18b (ApJL)](https://iopscience.iop.org/article/10.3847/2041-8213/ad2616) ·
[Photochem code paper (PSJ)](https://iopscience.iop.org/article/10.3847/PSJ/ae0e1c) ·
[CHEMKIN PREMIX manual](http://www.cvd.louisville.edu/Course/Chemical%20Vapour%20Deposition/Manuals/chemkin/chemkin7premix.pdf)

## G. Validation record (what was actually run, 2026-07-05)

**Physics-completion re-certification (second session, 8-D smoke with clouds +
H2-He CIA + merged gradient):** ALL CHECKS PASSED. block ≡ naive to **1.6e-9**
across all 8 dims; FD table — lnZ 2.0e-5, c_o 1.7e-7, lnKzz 2.9e-3, Tirr 3.0e-5,
log10kappa 1.9e-5, lnR0 9.3e-8, **log10kappa_cloud 8.4e-8, cloud_alpha 8.8e-8**
(machine-grade, as expected for RT-only dims; the cloud was optically ACTIVE at
the test point, κ≈3e-3 cm²/g — a real-signal test, not 0≈0). Measured bonus: the
merged block gradient ran **5.8× faster than naive** (439 s vs 2532 s) — one
batched program + 3 of 8 dims off the chemistry path. Opacity build over the full
gpu band: all 8 molecules + both CIAs in 45 s from local caches.

1. `pytest tests/` — **9/9**: binning matrix ≡ trapezoid reference (incl. real
   C&M bins, row sums = 1, edge-bin drops); u-space bounds/uniformity/Jacobian;
   Gaussian SMC recovery + evidence + governor + resume.
2. `smoke_retrieval.py` — **ALL CHECKS PASSED** (~19 min CPU): block ≡ naive,
   6-dim FD table above, lnZ/c_o liveness.
3. Parent `smoke_test.py` — PASS (legacy tp_eval=None path bit-compatible).
4. Bisect + deep-diag probes (section A numbers).
5. Stage-2 conservation probe (5.29 % response; 98.4 % C retention at ×2.72).
6. `plot_smc.py` — all 4 figures render from the driver schema (validated on a
   schema-true synthetic bundle).
7. End-to-end driver (smoke preset, 12 particles): built, injected synthetic
   obs, entered the SMC loop cleanly; full completion is a GPU-scale job on a
   laptop CPU (each mutation sweep ≈ 60 batched 2-stage tangent solves) — run
   `--calibrate` on the GH200 before the first real submission.

## H. Physics completion pass (2026-07-05, second session)

Added after the litmus question "what's missing for a real run":

- **Clouds via ExoJax's shipped retrieval cloud** (`exojax.atm.simple_clouds.
  powerlaw_clouds`, pRT convention): κ(ν) = κ0·(ν/2857 cm⁻¹)^α per gram of
  atmosphere, uniformly mixed; dtau = κ·dP_cgs/g using `art.dParr` (bar→cgs ×1e6;
  do NOT reuse exojax's `opacity_factor`, which folds in 1/m_u for per-molecule
  cross sections). 2 new dims (`log10kappa_cloud`, `cloud_alpha`), kind "cloud",
  RT-only → cheap gradient block. Decision per Isaac's rule "clouds only if ExoJax
  ships the methods": it ships two — this one, and the AM01+Mie stack
  (`amclouds` + `PdbCloud` + `OpaMie`), the latter blocked by PyMieScatt (absent)
  + miegrid generation → documented as the upgrade, keyed to the retrieved T-P
  (`psat_enstatite_AM01` base) and retrieved Kzz (particle sizes). VULCAN cannot
  do silicate clouds self-consistently (condensates H2O/NH3/H2SO4/S2/S4/S8/C only;
  no Mg/Si in the atom set).
- **H2-He CIA** wired (second `CdbCIA`/`OpaCIA` + `opacity_profile_cia` term with
  vmr_h2×vmr_he; He VMR threaded through the aux tuple). File downloaded from the
  canonical `https://hitran.org/data/CIA/main/H2-He_2011.cia` (147 MB — note the
  `/main/` path segment; the bare `/data/CIA/` URL 404s). Graceful skip + warning
  if the file is absent, so legacy callers never break.
- **HCN, C2H2, H2S opacities** (high-C/O + reduced-S discriminators) — HITRAN
  main-isotopologue entries in `config.MOLECULES`, added to the gpu preset
  (8 molecules total). Caches downloaded (55k/15k/41k lines); the four older .h5
  caches already covered the 1900–5000 cm⁻¹ band from the June WIDE build. Full
  8-molecule RT builds in ~45 s; **the GH200 needs no internet** for any opacity
  (everything under the rsync paths).
- **Merged chem-gradient device call**: `_value_and_grad_block` now does ONE
  vmapped jvp over all n_chem_tp directions (primal + aux read from lane 0)
  instead of an unbatched dir-0 call + a separate (n−1)-lane call — one batched
  while-loop program instead of two (the two-call form paid nearly double wall
  when latency-bound). lnR0 + cloud dims are one RT-only `jacfwd` at the frozen
  aux. gpu preset is now 10-D with still only 6 chemistry-expensive directions.
- All RT signature changes are backward-compatible kwargs (`vmr_he=None`,
  `cloud=None`) — the parent demo's positional callers are untouched.

## J. Pre-launch review for the 48 h production job (2026-07-05, third session)

Full audit before the real-data submission; every item below was CHECKED, not assumed.

**Data (Carter & May 2024 fixed-LD products, NIRISS+G395H):**
- 152 bins after the 1.02–5.24 µm cut (93 NIRISS + 59 G395H); depths 20,696–22,676
  ppm; per-bin σ 37–436 ppm (median 70); all finite, all σ>0; no intra-instrument
  bin overlaps; the 0.114 µm G395H gap is the NRS1/NRS2 detector gap (expected).
- Error-bar asymmetry |σ₋−σ₊|/mean: median 1.4 % (G395H) / 0.9 % (NIRISS), max
  8 % → the Gaussian likelihood is safe. Bin-to-bin covariance is neglected
  (universal practice for these R=100 products; note in the paper).
- All depths sit above the model's bottom-of-atmosphere floor (19,887 ppm at
  7 bar) — the model can reach the data with physical photosphere heights.
- PRISM's <2 µm saturation issue does NOT apply (we use NIRISS, not PRISM).

**Chemistry configuration = Tsai 2023's published setup (verified in the cfg):**
`sl_angle = 83°` (their terminator-mean, cited in-file), their stellar UV
(`sflux-W39b_Tsai2023.txt`), their Kzz profile (`Kzz_prof="file"`; our lnKzz is a
multiplier on it), zero-flux BCs, SNCHO photo network. VULCAN-JAX's canonical W39b
`count_max = 3e4` (~6× headroom over typical ~5k-step convergence) is what the
paper's own single-column benchmarks use; the SMC suite overrides it down to
`count_max = 5000` (`gpu_config()`, right at "typical", deliberately tight) so one
pathological prior corner in the phase-1 lockstep batch can't turn into a
many-hour hang (see sec K). A prior draw that doesn't converge in 5000 accepted
steps is REJECTED at init (not carried, not raised on) and the init OVERSAMPLES
(`init_oversample`, default 2.0) so the culled cloud still holds N healthy
particles — `batch_eval_cold_l_diag` supplies the per-draw `worst_accept` that
`pipeline._init_state` thresholds against `count_max` to decide the rejection.
The W39b calibration (job 64575) measured ~27% cold-init non-convergence at
count_max=5000, comfortably within the 50% the 2.0 oversample tolerates.
`init_max_nonconverged_frac` is now a WARNING threshold on the observed reject
fraction (the run continues; it raises only if fewer than N survive).

**Vertical resolution (measured, this session):** nz=50 vs nz=150 at the truth θ
through the FULL real pipeline (152 real bins): median |Δdepth| = 2.6 ppm,
max = 72.4 ppm vs median σ = 70 ppm (worst bin 0.74σ). Sub-noise systematic;
aggregate Δχ² ~ few. ACCEPTED for production; the nz-convergence study stays as
referee-proofing follow-up (raise nz if calibration shows headroom).

**Pre-launch changes made:**
1. **Band widened 2.02→1.02 µm** (nu 1900–9900, 16 500 pts): +64 NIRISS bins
   (88→152, median σ 91→70 ppm) — the 1.1–1.9 µm water bands + haze-slope lever,
   at zero chemistry cost. All opacity caches verified to cover the wider band
   (8-molecule + 2-CIA build in 9.5 s, offline).
2. **H2/He Rayleigh scattering added** (ExoJax `xsvector_rayleigh_gas` +
   polarizabilities; King factor 1.0, a ≤2 ppm approximation): zero-free-parameter
   physics, mandatory once the band reaches 1 µm (else its slope biases the cloud
   posterior). Opt-in per profile (`use_rayleigh`), legacy demo untouched.
3. **48 h job mechanics**: PBS walltime 24→48 h; SMC governor 20→44 h (leaves ~4 h
   for build/compile/PPC/plots); H2-He CIA promoted to a REQUIRED preflight check
   (exojax_rt's graceful skip is not acceptable in production); HCN/C2H2/H2S added
   to the cache preflight.
4. Re-verified: 9/9 pytest; observation layer at the widened band (152 bins,
   exact binning operator, 10-D layout); final FD smoke re-run with Rayleigh in
   the graph (see §G for the numbers).

**Deliberate, documented modeling choices (defensible as-published-practice):**
noise inflation OFF (ERS practice: offsets yes, inflation rarely — enable via
`{"infer_noise_inflation": true}` overrides if the χ² demands it); uniform (not
patchy) cloud (add a coverage-fraction dim later if a referee asks); bin
covariance neglected; HITRAN line lists (documented caveat; ExoMol/HITEMP is the
post-first-run upgrade); Tint=150 K, f=1/4 fixed; 1D terminator model vs
limb-combined data.

**Launch runbook (in order):**
```
rsync -a --exclude '.git' VULCAN-JAX vulcan_exojax_run jax_paper /nobackup/$USER/VULCAN_Project/
cd /nobackup/$USER/VULCAN_Project/vulcan_exojax_run/runs/w39b_smc_retrieval
qsub -v CALIBRATE_ONLY=1 run_nas_w39b.pbs     # ~1-2 h; read timing.json vs the 20 h governor
qsub -v SYNTH=1 run_nas_w39b.pbs              # injection recovery MUST bracket truths
qsub run_nas_w39b.pbs                         # the real-data production run
qsub -v RESUME=1 run_nas_w39b.pbs             # only if the governor stopped before beta=1
```
Gate between steps: calibrate projection must fit 20 h for >=15 stages (else
warm_count_max 1500→1000 / mcmc_steps 6→4 / nz 50→40 via overrides); SYNTH must recover injected truths in
the 90 % CIs; only then trust the real-data posterior.

## K. Init-stall incident + fix (2026-07-07)

**Incident (job 64073, gpu_r3000 real-data run).** 16.7 h at "100 % GPU util" /
163 W / 700 W with NO log line after "Running adaptive-tempered SMC..." and no
`smc_checkpoint.npz`. The GPU monitor showed a ~4 min 0 %-util window (the XLA
compile, 18:08–18:11) then continuous 100 %. Diagnosis: the run never left
`_init_state`. The old init computed the gradient through the COLD map
(`batch_eval_cold_vg`): 6 tangent lanes × TWO full while_loop solves per
particle (`chem_solve_cold` = stage-1 T-relax + stage-2 re-converge),
`lax.map`-chunked at `smc_chem_chunk=6` → 8 SEQUENTIAL lockstep blocks of 36
lanes, each gated by its slowest prior-corner lane. `count_max=3e4` bounds
ACCEPTED steps only; with `batch_max_retries=64`, rejected iterations can
inflate the body-iteration count well past it. Eight sums-of-maxima over prior
corners × two stages × tens of ms per launch-bound iteration = tens of hours.
The 163 W at 100 % util is the tiny-kernel signature (§C post-mortem): the
while_loop body is ~O(100) µs-scale kernels + a per-iteration predicate sync,
so "utilization" is pegged while SMs sit idle.

**Fix (implemented).** `_init_state` is now two phases: (1) cold LIKELIHOOD-ONLY
pass at full width (1 primal lane per particle, no chunking — one lockstep max
over N draws instead of eight, no 6× tangent redundancy through cold solves);
(2) gradient via the MOVE evaluator (warm continuation) at the same cloud —
each particle re-converges from its own phase-1 column in ~count_min steps and
the jvp lanes ride that short warm map, which is the SAME map every subsequent
MALA proposal uses (consistent by construction; MALA is exact for any
consistently-used drift). Expected init: tens of minutes, phase-logged.
Companions: `calibrate()` now derives its cloud from the run's own seed exactly
as `run_smc_loop` does (the PRNGKey(0) pilot cloud is why the timing gate never
saw the bad corners); `build_retrieval_forward` fails fast if `prior_c_o[1]`
reaches the fixed-O b_z positivity bound (beyond it, prior corners get negative
O-carrier abundances that the runner clips into silently-wrong finite-likelihood
states); the PBS GPU monitor now records power.draw + clocks.sm.

**Guard tripped on first contact (probe job 64144, 2026-07-07).** The b_z
positivity bound on the real 10x-solar column is **+0.566** — INSIDE the old
prior_c_o=(−1.6, 0.6). Every prior draw / proposal with c_o ∈ (0.566, 0.6] in
the killed 16 h run was silently clip-mangled (negative O-only carriers →
runner clip → wrong inventory, finite likelihood). Priors now capped at
**c_o ≤ 0.45** (default + override files): worst-layer b_z ≈ 0.25, margin for
hot stage-1 columns where the O-in-C-carriers share rises above the baseline
0.568, and C/O coverage up to ~0.86 about the 0.549 baseline (the old 0.6 edge
was C/O ≈ 1.00, which the fixed-O knob structurally cannot reach — reformulate
the knob or use proxy mode if a carbon-rich prior ever becomes a science
requirement). Bonus calibration from the same log: the nz=50 warm-up converge
is 2667 steps in 84 s on the GH200 ≈ **31 ms per solver step single-lane** —
the launch-bound per-step cost that sets init/stage wall-time expectations.

**Memory probe results (job 64144, 2026-07-07, nu_pts=5000, compile-only) — the
numbers that rewired the chunking:**

| case | temp GiB |
|---|---|
| chem GRAD ×1/×2/×6 particles (6/12/36 lanes) | 0.15 / 0.26 / 0.78 |
| chem PRIMAL ×96 | 5.32 |
| RT VJP ×1 / ×6 | **18.40 / 65.43** |
| RT PRIMAL ×16 | 3.58 |
| FULL cold_vg at (chem 8, rt_vjp 12) | **195.25 — would have OOM'd** |
| FULL cold_l ×96 | 5.37 |

Takeaways: (1) staged chemistry tangents are ~60× cheaper than believed — the
gradient chemistry now runs UNCHUNKED (`smc_chem_chunk=0`); (2) the RT VJP is
the sole memory wall (18.4 GiB/lane; `smc_rt_vjp_chunk` stays 6 = 65.4 GiB);
(3) the full program can stack stage peaks beyond the naive component sum
(195 vs ~123 naive at the failed setting) — ALWAYS re-probe `FULL cold_vg`
after chunk/nu changes, and fall back to `smc_rt_vjp_chunk=4` if it exceeds
~72 GiB; (4) with 576 full-width gradient lanes at N=96, the chemistry stage
finally runs at the gpu_throughput-benchmark width. (Historical note: N=96 was
the recommendation here; the gpu preset moved to N=144 on 2026-07-10 on the
power-headroom evidence.)

**count_max tightened + reject-on-nonconverged closed (2026-07-07, same day).**
Diagnosed live against a real production run (job 64163, gpu_r3000_n96): even the
FIXED two-phase init above can legitimately sit in phase 1 for hours at N=96,
since wall time is bounded by the single slowest of N independent cold prior
draws under the canonical `count_max=3e4`. Two changes, SMC-suite-scoped only
(VULCAN-JAX's own W39b default and the paper's benchmarks are untouched):
1. `gpu_config()` now sets `count_max=5000` — right at the documented "typical
   ~5k-step convergence" mark rather than 6× above it, so the worst-case phase-1
   wall time is bounded to roughly the cost of one healthy convergence instead of
   an open-ended tail.
2. `vulcan_chem.converged_y(..., return_diag=True)` exposes `accept_count`;
   `retrieval_forward.chem_solve_cold_diag` threads the worse of the two
   two-stage-solve stages; `pipeline.batch_eval_cold_l_diag` carries it through
   the batched phase-1 evaluator.

**Reject-and-cull + oversample (2026-07-08, replaces the raise-on-nonconverged
gate).** The R=100 calibration (job 64575, `dt_max=1e11` live) measured **27% of
cold draws non-convergent at count_max=5000** — a real minority for a full-kinetics
forward, not a bug (they cluster at hot + extreme-Kzz prior corners). Raising the
whole run over that is wrong, and so is the old fallback of *carrying* an
unconverged state as finite L (a silent bias). `_init_state` now does what every
retrieval code does with a failed forward: **reject it with `-inf` likelihood and
oversample so the culled cloud still holds N healthy particles** (petitRADTRANS /
nested-sampling `-inf`-for-invalid + Herbst-Schorfheide oversample-for-ESS). It
draws `ceil(N * init_oversample)` (`init_oversample` default 2.0 → tolerates up to
50% non-convergence), thresholds each draw's `worst_accept` against `count_max` to
reject the non-converged, pays the expensive phase-2 gradient on the N survivors
only, and RAISES only if fewer than N survive (systemic). `init_max_nonconverged_frac`
is demoted to a WARNING threshold on the observed reject fraction. Unit-tested in
`tests/test_init_state.py` (reject / cull / raise-if-too-few / oversample-count);
full suite 14/14. Stub pipelines (`has_chem_state=False`) never reject — the diag
path only engages for real chem-backed pipelines.

**Still open (deliberately deferred):** re-running the SYNTH injection-recovery
gate against the new count_max=5000 (should still recover truths given the
tolerance, but unverified end-to-end), chunk widening from a PROBE_MEMORY pass at
the r3000 grid, XLA command-buffer / autotune A/B, sort-by-cost chunk
permutation, and the adjoint lane reduction (Future work #3, with the caveat that
`steady_state_grad`'s LGMRES is host-side scipy — not hot-path-usable without a
JAX-native batched solve, and lnZ/c_o are conserved-inventory directions, the
documented ill-posed adjoint case).

## I. Future work (ranked)

1. **AM01 + Mie clouds** (ExoJax-native self-consistent-lite; see above) — needs
   `pip install PyMieScatt` + one-time miegrid; +2–3 dims (fsed, σg, base scale).
   - **VULCAN-grown clouds in the retrieval** (assessed 2026-07-05, deliberately NOT
     done): VULCAN(-JAX) has full condensation (H2O/NH3/H2SO4/S2/S4/S8/C + settling,
     ported incl. batched NH3 cold-trap) — but it yields condensate MASS only, never
     optics (single fixed r_p, no size distribution/refractive indices), so ExoJax
     OpaMie/PdbCloud is still required for opacity (same PyMieScatt blocker). For
     W39b it's moot: none of VULCAN's condensables condense at the ~1050–1300 K
     terminator (the real cloud is silicate; no Mg/Si in the network). And every FD
     certification here is conden-OFF — the conden kernels are switch-heavy
     (saturation crossings, cold traps, fix-species pins) with unvalidated
     forward-mode tangents (the 2026-07-05 adjoint audit flags conden as
     wrong-when-active on the reverse side), plus untested interaction with the
     two-stage warm start. WHERE IT SHINES: a cooler target (H2O on a temperate
     sub-Neptune, NH3 on a cold Jupiter, S8 haze at 500–700 K) — "VULCAN grows the
     cloud, ExoJax shines light through it, one gradient through both" would be a
     first; prerequisites are an FD campaign through the conden branches (likely
     smoothing the switches) + the Mie stack.
2. **Per-particle warm-starting across MCMC steps** (`converged_y(warm_y=…)` per
   particle, threaded through resampling) — potentially large speedup; needs
   care that warm-started tangents stay relaxed (count_min guards).
3. **Reverse-mode steady-state adjoint** for the likelihood gradient
   (`steady_state_grad` machinery: solver_map="renorm" + photo_recompute_k) —
   would make gradient cost dimension-independent; needs generalization from
   d/d(ln k) to the retrieval θ and validation.
4. HITEMP/ExoMol opacities; free Tint; PRISM/NIRCam groups; noise-inflation on.
5. `nz` convergence study (50 vs 100) on the recovered posterior.
