# vulcan-retrieval: historical development log

This file is the package's development diary: design choices, incidents, what
worked and what did not, with every measured number preserved. Everything below
the rule was moved VERBATIM from the pre-0.6 READMEs (the old
`retrieval_framework/README.md` development log, then the old bundle-level
`README.md` audit summary), so path references (`retrieval_framework/...`,
`runs/...`, relative `CLAUDE.md` paths, bare `config.py` imports) are
historical. Current usage and the authoritative configuration live in
`README.md`; operational rules live in the repo-root `CLAUDE.md`.

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

---

The following section is the old bundle-level README's audit-response summary,
moved verbatim:

## Scientific-correctness pass (2026-07-11, external-audit response)

An external scientific audit of this bundle was answered in full; the load-bearing
changes (details in each module's docstring):

1. **Abundance knobs are exact elemental directions** (`abundance_mode="elemental"`,
   default for retrieval + jwst_tool): exact conserved ratios, Σn=P/k_BT, exact
   `atom_ini`, path-independent inventories. Legacy `"masks"` kept for the published
   demo caches.
2. **Atmospheric structure rebuilt per proposal**: D_zz(T,M), vm/vs, convergence-gate
   Kzz, and the initial carry geometry now follow the retrieved T-P (the in-loop
   hydrostatic refresh already did µ/g/H_p/dz from step 1 — the audit's "frozen
   structure" claim was narrower than stated, but real for these pieces).
3. **H₂-He CIA is required** everywhere (was silently skippable and WAS skipped by the
   sensitivity demo / zco / Fisher-forecast caches — all those caches are stale and
   must be regenerated); emission shares the transmission opacity terms (CIA + cloud;
   Rayleigh deliberately transmission-only) and is labeled emergent flux.
4. **Broadening knob** (`air`/`h2he`) + A/B script; hot-line-list limits documented as
   accuracy caveats, per-molecule source swap points marked.
5. **Evidence semantics fixed**: `logZ` is reported as evidence under the OPERATIONAL
   prior (T-P window × converged support, renormalized); the measured support fraction
   (persisted from init through checkpoints) and `logZ_box = logZ + ln f` ride with
   every output. Warm-cap rejections are counted separately per sweep/stage
   (`warmcap=`), and `validation/mala_reversibility.py` probes cap symmetry.
   Tempered (β<1) draws are labeled on every figure/export path.
6. **validate_warm now gates on three axes**: Δ logL (<0.1), binned-spectrum ppm
   (<5), and elemental-inventory agreement — not logL alone.
7. **jwst_tool v5**: floor-aware transits-to-target (photon term averages down, the
   R=100-anchored floor does not — "never" is a possible answer), offset-marginalized
   detection Δχ², d(λ)-weighted model binning, saturated modes excluded from all
   forecasts consistently.
8. **New validation suite** (`validation/`): `elemental_audit.py`,
   `resolution_ladder.py`, `top_pressure_ladder.py`, `broadening_ab.py`,
   `mala_reversibility.py` — the numerical-convergence and statistical checks the
   audit required before interpreting a real-data posterior. **Run them on the GPU
   node before the next production retrieval**; every pre-existing chemistry/spectrum
   cache (demo npz, zco/Fisher caches, jwst_tool model cache, SMC checkpoints)
   predates the physics fixes and is stale.

## Maximal cross-repo audit response (2026-07-12)

The tri-repo "maximally intensive" audit added two retrieval fixes on top of the
2026-07-11 pass (jwst-tool items handled in the sibling repo):

1. **Observation-injection validation** (item 4). `pipeline.set_observations`
   only checked vector length, so a non-finite depth or a σ ≤ 0 / non-finite
   slipped into the Gaussian likelihood (which divides by σ and logs it) and
   silently produced NaN/Inf likelihoods. The guard is now the module-level
   `validate_observations()` (unit-testable without the forward model;
   `tests/test_set_observations.py`) and RAISES on any invalid entry. Mask
   invalid bins before injection.

2. **Box-prior evidence: physical vs numerical support separated** (item 5,
   P0 for model comparison). The 2026-07-11 pass reported one
   `logZ_box = logZ + ln(f_tp·f_c1·f_c2)`, which folded solver-dependent
   convergence success (cold-converge `f_c1`, warm-recert `f_c2` — they move
   with count_max/warm_count_max/tolerances/history) into a quantity called a
   box-prior evidence. Now: `logZ_box_physical = logZ + ln(f_tp)` restores ONLY
   the physical, solver-INDEPENDENT T-P-window domain (no premodit opacities
   outside [300,3000] K) — the number to difference across models; the
   convergence attrition `ln(f_c1)+ln(f_c2)` is reported separately as a
   numerical diagnostic; the old convergence-inclusive `logZ_box` is retained
   but flagged not-for-Bayes-factors. New npz fields `smc_logZ_box_physical`,
   `smc_log_support_physical[_err]`, `smc_log_conv_attrition[_err]`; plot_smc
   headlines the physical value.

---

Note: `fisher_forecast/` was removed 2026-07-11 as superseded by
`scripts/zco_information/` (the Z-C/O science) and vulcan-jwst-tool's live
Fisher forecast (the instrument forecasting).


---
---

# MIGRATED FROM CLAUDE.md (2026-07-13)

The content below was moved verbatim out of `CLAUDE.md` during a cleanup that
distilled `CLAUDE.md` down to standing operational rules. These are dated
debugging narratives, job post-mortems, and the reasoning behind the rules —
the "why" that now lives here. Some episodes (the 2026-07-11 scientific-correctness
pass, the 2026-07-12 cross-repo audit) also appear in the sections above; where
they overlap, both accounts are preserved. `CLAUDE.md` now carries only the
current rules and points back here for detail.

---

Critical rules and decisions for this repo. Read before touching the retrieval or
running anything on the supercomputer.

## 2026-07-11 scientific-correctness pass — EVERY cache/checkpoint is stale

Audit-response changes (README "Scientific-correctness pass" + module docstrings):
exact-elemental abundance map (`abundance_mode="elemental"`, schema default — lnZ/c_o
are now exact conserved-ratio directions, atom_ini exact, Σn=P/kT at init), per-proposal
atmosphere rebuild (Dzz(T,M)/vm/vs + pv.Kzz + initial carry geometry; conden refuses
T-varying builds), H2-He CIA REQUIRED in every RT call (vmr_he is a required arg),
broadening knob ("air"/"h2he"), evidence reported as OPERATIONAL-prior conditioned with
measured support fraction (`logZ_box = logZ + ln f` in results/npz; init cull counts
persisted through checkpoints), per-sweep/stage `warmcap=` counters, tempered-draw
labels on every output path, validate_warm gates on logL + spectrum-ppm + inventories,
jwst_tool v5 (floor-aware transits, R=100-anchored floors, offset-marginalized detect,
saturation-consistent Fisher; `_VERSION=5` -- the tool now lives in the SIBLING
vulcan-jwst-tool repo). This repo is the standalone vulcan-retrieval package since
2026-07-11 (`pip install --no-deps -e .`); see "Layout / entry points" below.

Operational consequences:
- **The chemistry map changed** (elemental + atm rebuild): synthetic obs, demo npz
  caches (`data/*.npz`), zco/Fisher caches, jwst_tool model_cache, and ALL SMC
  checkpoints are STALE. Regenerate; do NOT resume a pre-pass checkpoint into the new
  map (likelihoods re-anchor mid-run). `overwrite=True` handles synthetic obs.
- **Before the next production run**: `PROBE_MEMORY=1` (the evaluator gained small
  per-proposal structure rebuilds), then the smoke chain + suite, then on the GPU node
  the new validation set: `validation/elemental_audit.py`,
  `resolution_ladder.py`, `top_pressure_ladder.py --extend-chem`,
  `broadening_ab.py`, and post-run `validate_warm` + `mala_reversibility.py`.
- **`h2he` broadening** downloads separate `<db>_h2he` line-list caches on first use
  (network / NAS proxy); default stays "air" until broadening_ab.py is run and judged.

## 2026-07-12 maximal cross-repo audit response (obs validation + evidence split)

Two retrieval fixes from the tri-repo "maximally intensive" audit (the jwst-tool
items are fixed in the sibling repo):

- **`set_observations` validates at the API boundary** (audit item 4). The
  Gaussian likelihood divides by σ and logs it, so a non-finite depth or a
  non-positive/non-finite σ used to silently poison every likelihood with
  NaN/Inf (mass rejection / pathological SMC). The check now lives in the
  module-level `pipeline.validate_observations(depth, sigma, n_bin, npdtype)`
  (so it is unit-testable without the forward model — `tests/test_set_observations.py`):
  RAISES on any non-finite depth, or any σ ≤ 0 / non-finite. Mask invalid bins
  BEFORE injection. Fail-loud rule, standard.

- **Box-prior evidence — `logZ_box_physical` RETRACTED same day it shipped
  (2026-07-12 recheck P0-B).** The first fix split the support into
  `logZ_box_physical = logZ + ln(f_tp)` ("use for Bayes factors") and a
  separate convergence diagnostic. That construction is INVALID:
  `P(A)·E[L|A∩C]` restores the T-P prior mass while silently keeping the
  convergence conditioning renormalized — it is neither the box integral
  over A (the likelihood on the non-converged set was never evaluated) nor
  the A-conditioned evidence; a support fraction cannot reconstruct an
  unevaluated likelihood (toy counterexample pinned in
  `tests/test_evidence_semantics.py`). Current semantics
  (`pipeline.evidence_report`, module-level + unit-testable):
  - `logZ` — evidence under the OPERATIONAL prior (box ∩ T-P window ∩
    converged, renormalized). Never difference across models with different
    support fractions.
  - `logZ_box = logZ + ln(f_tp·f_c1·f_c2)` — the ZERO-FILLED box evidence,
    the exact integral of π·L·1[valid] over the declared box (the sampler
    defines non-converged = rejected = zero likelihood). SOLVER-DEPENDENT
    via the convergence indicator. Cross-model Bayes factors ONLY at matched
    solver settings AND with attrition shown likelihood-negligible (report
    f_tp and f_conv alongside any comparison; headlined by plot_smc).
  - There is NO f_tp-only evidence field anywhere (`smc_logZ_box_physical`
    removed from run_smc.py npz; support fractions still exported
    separately: `smc_log_support_physical[_err]`,
    `smc_log_conv_attrition[_err]`).

## Supercomputer sync — git pull for CODE (preferred, 2026-07-10), scp for DATA

**Code updates: `git pull` on the NAS front end** (all repos are public on GitHub and
every local change is committed + pushed, so GitHub is always current). Since the
2026-07-11 sibling-repo split there are TWO repos to pull for the retrieval (the
jwst tool is local-only and never deployed to NAS):

```
cd /nobackup/imalsky/VULCAN_W39b_HPC/vulcan-retrieval
git pull --ff-only
cd ../VULCAN-JAX
git pull --ff-only
```

No manual install step after a pull: both packages are installed EDITABLE (once, by
`tools/bootstrap_nas_env.pbs`), so pulled code changes are live immediately. Jobs are
READ-ONLY on the environment (2026-07-11 redesign: the old per-job `pip install
--user -e` raced across concurrent jobs on the shared userbase and crashed
mid-uninstall — the job-64961-era "OSError: .../userbase/bin/vulcan-jax" failure);
every job preflight runs `python -m retrieval_framework.validate_env` instead, which
HARD-fails with a pointer at the bootstrap when a stale install shadows the clone or
packaging metadata drifted (version/deps/entry-point changes after a pull DO need a
bootstrap re-run — validate_env detects exactly that). The user site is
per-interpreter (`/nobackup/$USER/.vulcan/userbase-py3.10` etc.); the old shared
`userbase` dir is dead and can be `rm -rf`'d.

One-time setup (front end). MEASURED 2026-07-10 on cghfe02: **direct https to
github.com works and the proxy hostname does NOT resolve** -- make sure
`https_proxy`/`http_proxy` are UNSET for git (the "Could not resolve proxy" failure
mode), no proxy exports needed:

```
cd /nobackup/imalsky/VULCAN_W39b_HPC
unset https_proxy http_proxy
git clone https://github.com/imalsky/vulcan-retrieval.git
git clone https://github.com/imalsky/jax-vulcan.git VULCAN-JAX
```

**2026-07-11 sibling-split migration from an existing vulcan_exojax_run tree**: clone
vulcan-retrieval as above, then MOVE the seeded data across (no re-transfer needed):

```
mv vulcan_exojax_run/data/opacity_cache vulcan-retrieval/data/
mv vulcan_exojax_run/data/exojax_linelists vulcan-retrieval/data/
```

The old vulcan_exojax_run clone can then be parked/removed.

- The **`VULCAN-JAX` clone target name is load-bearing** (the PBS preflight hard-codes
  it; the GitHub repo is named `jax-vulcan`). Same for `vulcan-retrieval` (repo and
  clone name match).
- The NAS clones are **read-only deploys**: never edit there; `--ff-only` guarantees a
  pull can never merge; run outputs / PBS `.o` files / caches are all gitignored so the
  tree stays clean.
- **Data is NOT in git** and needs a ONE-TIME seed into a fresh clone —
  `data/opacity_cache/` (preflight ERRORS without it) and `data/exojax_linelists/`
  (else re-downloaded via the proxy). Move from the old tree as above, or scp once
  from local.

**scp fallback / data transfers** (also if git https is ever blocked), **exactly** this
style (one command per dir, no backslashes):

```
scp -r -oProxyCommand='ssh imalsky@sfe6.nas.nasa.gov ssh-proxy %h' [local dir] imalsky@pfe.nas.nasa.gov:[hpc location]
```

- **Never use rsync.** Never make tarballs.
- **Never** wrap remote commands in `ssh nas '...'`. Give the transfer, then the
  commands to run **while logged in on the node** (`cd …`, `qsub …`) as plain commands.
- `PROJECT_ROOT` is currently `/nobackup/imalsky/VULCAN_W39b_HPC`; the PBS preflight
  requires both trees under it.

## nsys masks the exit code — never profile a first/debug run (learned 2026-07-09)

`nsys profile ... python ...` returns **0 even when the profiled process is killed or
crashes**. Job 64604 ran under `NSYS=1`, so when SMC stage 0 died the wrapper saw `rc=0`,
ran the plot step, and printed `job finished rc=0` — a green result on a dead run (no
posterior, no checkpoint), and the real error was swallowed. That masking hid the stage-0
mutation bug for ~15 calibration retries. **Rule: run calibrations and first/debug runs
WITHOUT `NSYS`** so a real failure surfaces (a `RuntimeError` traceback, a CUDA error, or a
`Killed`/OOM line). Add `NSYS=1` only once a run is known-good and you specifically want a
kernel timeline. The always-on `nvidia-smi` monitor (`logs/gpu_monitor_*.log`) already
gives util/power/clocks without it. Also: `NSYS_DELAY` is in **seconds** (64604 used
6000 = 100 min, not 6 s).

## Calibration mutation runs at stage-0 conditions (job 64961, fixed 2026-07-11)

`run_smc.calibrate()` used to benchmark the mutation at hard-coded `(beta=0.5,
step=mala_step_size=0.2, scale=1)`. The MALA drift is `step*scale^2*beta*G`, and a
prior-like cloud carries |L| (hence |G|) up to ~1e6 — those proposals land so far off
the converged map that a few per sweep (8/144 in job 64961) returned a finite spectrum
with a non-finite end-to-end gradient, and `_check_mutation_health` aborted with the
AD-pathology RuntimeError (48 = 8 x 6 sweeps; accept=0.00 throughout). NOT an AD bug:
init was healthy (205/288 converged, phase-2 gradient pass clean), and the production
ladder never makes such moves — its stage 0 uses an ESS-bisected first beta (~1e-5 for
a real-data cloud), a stage-0 resample, the `_abs_scale_diag` cloud-width
preconditioner, and a clamped step. **Fix: `calibrate()` now reproduces
`run_smc_loop`'s stage 0 exactly** (same beta bisection, systematic resample,
preconditioner, step clamp); the chosen beta/step/scale are logged and land in
timing.json (`calibration_beta_stage0`, `calibration_step`, `calibration_scale_*`).
Regression: `tests/test_smc_gaussian.py::test_calibrate_benchmarks_stage0_conditions`.
Corollary: `warm_extrapolate` was an amplifier, not the cause — an absurd delta-theta
makes the first-order seed garbage; at stage-0-sized moves the seed-vs-plain parity is
gated by `tests/test_warm_extrap.py`.

## Retrieval — critical decisions (2026-07-08)

- **count_max = 5000, always** (lowered from 10000, Isaac 2026-07-08). Set in `vulcan-retrieval/runs/w39b_smc_retrieval/case.py::gpu_config`;
  every override file falls back to it. Do NOT raise it. A solve that doesn't converge in
  5k accepted steps is a **failed draw** — acceptable, not a bug to chase: it is REJECTED
  at init and the cloud is oversampled to compensate (see "Cold-init reject-and-cull").
- **Convergence = VULCAN-master canonical criteria.** `yconv_cri = 0.01` (schema default),
  and `slope_cri` / `yconv_min` / `flux_cri` are NOT overridden (inherit
  `vulcan_cfg_W39b`). The old 1e-3 (from the sensitivity demo's tight-jvp needs) is gone —
  it barely changed gradient quality but ground out extra thousands of steps.
- **No T-P clipping — reject and redraw.** `tp_profile` returns the raw Guillot profile.
  A draw whose T-P leaves the modelable window **[300, 3000] K** (premodit table range,
  inset 20 K) on the ART grid is REJECTED, never clipped: rejection-sampled away at the
  prior (`pipeline.sample_prior_u` redraws) and given `-inf` likelihood as a MALA proposal
  (`pipeline.tp_valid` gates every likelihood path). The prior rejection sampler raises
  loudly if the T-P prior is mostly out-of-window (a mis-specified prior fails fast).
- **Realistic priors** live in `case.py::_W39B`, literature-anchored (Tsai et al. 2023
  VULCAN grid + Rustamkulov et al. 2023 PRISM ERS): C/O ∈ [0.10, 0.70], Kzz ±2 dex,
  Tirr ∈ [1100, 2200] K, γ ≤ ~2 (weak inversion allowed, Isaac 2026-07-08). Metallicity kept wide (1–100×
  solar) so the data localizes it.
- **Fail-fast everywhere** (standing rule): no silent fallbacks. Missing H2-He CIA raises;
  planet identity (rp_cm/rstar_cm/vulcan_cfg_module) must be set or `validate_config`
  raises; a real-data run with no overlapping bins raises; RESUME with no checkpoint
  raises; non-finite gradients surface as `n_bad` and the host raises.

## Why the >10k-step tail happened — dt_max ballooning (diagnosed 2026-07-08, local)

Local chemistry-only diagnostics (`vulcan` env, no RT; scratchpad `diag_*.py`,
`sweep_dtmax.py`): the tail is MOSTLY a numerical artifact, not slow physics. The
VULCAN-master default `dt_max = runtime*1e-5 = 1e17 s` lets the adaptive Ros2 step
balloon to ~1e16 s on high-Kzz columns (per-step local error stays tiny — the published
Tsai+2017 adaptive-stepping behavior), so the solver SPINS in a large-dt oscillation
(`longdy` stuck ~2-4, marched to t~1e17 s) instead of settling. Capping `dt_max`
converges those ballooning draws in ~1000 steps (d10/d19/d59 from calib job 64523; d19:
>11000 → 986). The cap VALUE in [1e9, 1e12] doesn't change WHICH draws converge, only
convergence tightness. **Fix applied: `Config.dt_max` (first-class, banner-shown) = 1e11
s in `case.py::_W39B`** (catches ballooning, reaches t~5e14 in 5000 steps >> physical
settling ~1e13, leaves the truth at 4275 steps identically). This is a STEP-SIZE control,
NOT a convergence criterion (yconv_cri/slope_cri stay master).

**A residual fraction still fails and `dt_max` CANNOT fix it** — two distinct modes:
(1) `longdy` stuck just above the 0.1 loose gate (~0.13, moderate t, aflux fine — a
marginal oscillation, e.g. d5 +Kzz2.0); (2) hot + low-Kzz photolysis limit cycles
(`aflux_change` stuck ~0.2-0.36, e.g. d40). These are genuine non-converging columns at
extreme prior corners (hot + extreme-Kzz), inherent to a full-kinetics forward.

## Cold-init reject-and-cull + oversample (measured + WIRED 2026-07-08)

The R=100 calibration (job 64575, dt_max=1e11 live, N=192 draws, probe 12000) MEASURED
the residual: **27.1% of cold draws don't converge at count_max=5000** (12.5% even at
12000; heavy tail — p50=2525 fine, but p75=5420 busts 5000). That is a real minority, not
a bug. **Fix (best practice, WIRED): `pipeline._init_state` now REJECTS non-converged
draws (-inf) and OVERSAMPLES so the culled cloud still holds exactly N healthy
particles** — the same handling every retrieval code uses for a failed forward
(petitRADTRANS / nested sampling `-inf` for invalid outputs) plus the Herbst-Schorfheide
SMC oversample-for-ESS rule. It draws `ceil(N * init_oversample)` (default
`init_oversample=2.0`, tolerates up to 50% non-convergence), pays the expensive gradient
pass on the N survivors only, and RAISES only if fewer than N survive (systemic). The old
behavior (raise at >10%, or carry unconverged states as finite L) is gone — carrying a
non-converged spectrum is a silent bias, rejecting it is correct. `init_max_nonconverged_frac`
is now just a WARNING threshold on the observed reject fraction. Unit-tested in
`vulcan-retrieval/tests/test_init_state.py` (5 tests: reject, cull, raise-if-too-few,
oversample count). Full suite 14/14 green.

**FIXED 2026-07-09 (was the deferred residual, and it was NOT low-risk):** a warm MALA
*mutation* proposal that hits a non-convergent corner is now count_max-REJECTED before its
gradient is trusted — the warm-side analogue of the cold-init reject-and-cull. Previously
the warm continuation returned a finite-but-off L **and had its jvp/RT-vjp computed
anyway**; that garbage gradient tripped `n_bad` (a spurious `RuntimeError`) or NaN'd at SMC
**stage 0** — which is what killed the tempering ladder in job 64604 and made every timing
calibration fail (~15 retries). It was invisible because the failure was masked by `nsys`
(see below) and because the memory probe never covered the mutation kernel. Root-caused by
ruling OUT memory (the mutation kernel's compiled footprint is byte-identical to the
cold-init gradient that already fit) and the sampler logic (14/14 unit tests), then
reproduced on the smoke pipeline. **Fix:** `retrieval_forward.chem_solve_warm_diag` (warm
twin of `chem_solve_cold_diag`) reports the warm accept_count, and
`pipeline._make_batch_eval` (warm+want_grad) gates `L→-inf`, drops it from `n_bad`, and
pins its carried state when `accept_count >= count_max`. Regression-tested in
`tests/test_warm_reject.py` on the real smoke pipeline.

**Init phase 2 must run UNCAPPED (learned from NAS job 64854, 2026-07-10).** The claim
that phase 2 "is unaffected because survivors re-converge at ~zero increment" was WRONG
once the cap tightened to `warm_count_max=1500`: a marginal survivor (slow phase-1
converger / stall-fallback certification) can need >1500 accepted steps just to
RE-CERTIFY convergence from its own converged column (the criterion is time-based --
unchanged vs the run at half its integrated time -- so a fresh warm restart of a wobbly
column re-pays the certification window). Job 64854: 5/96 healthy survivors gated at the
cap -> mislabeled "non-finite forward" -> spurious "RT/AD problem" RuntimeError; the tell
was phase-2 wall time sitting exactly at the cap (~780 s ≈ 1500 steps). Fix: phase 2 uses
`batch_eval_init_vg` (`_make_batch_eval(..., mutation_cap=False)` ->
`chem_solve_warm_diag_full`, gated at the cold `count_max`) -- survivors are
proven-convergent particles, not disposable proposals. Mutation proposals keep the
`warm_count_max` cap unchanged. Cost: phase 2 can run to ~5000 steps (~40 min) instead of
~13 min when a marginal survivor is present -- once per run, and correctness is not
negotiable. Regression: `test_warm_reject.py::test_init_eval_is_uncapped`.

**...and uncapped is still not enough: cull-and-backfill (job 64897, same day).** With
phase 2 uncapped, 3/96 survivors STILL died -- these columns certify cold from baseline
within 5000 steps but cannot RE-certify warm from their own converged column even in
5000 (marginal oscillators / stall-fallback certifications re-pay the time-based window
on restart and lose). A repeatable class (5/96, then 3/96 on a different seed), so it is
handled like phase 1 handles non-convergence: **phase 2 now evaluates
`N + init_phase2_spare` survivors (default spare 8; width ~free in lockstep), culls the
re-certification failures, and backfills from the spares** -- part of the operational
prior, logged loudly, reported alongside the phase-1 reject fraction. The init eval
threads the accept counts out so a TRUE RT/AD death (non-finite forward with a
NON-exhausted count) still raises. PROBE_MEMORY now probes the widened init eval (the
widest gradient batch in the run: N+8 = 152 at the gpu preset, projected ~80.5 GiB).
Raise `init_phase2_spare` only with a probe. Tests: `test_init_state.py` (cull/backfill,
RT-death raise, spares-exhausted raise).

**warm_extrapolate is ON in the gpu preset (Isaac, 2026-07-10).** The schema default
stays False; the gpu preset enables it, so the staged CALIBRATE -> SYNTH -> production
sequence exercises it end-to-end (and validate_warm gates the result) before real data.
`warm_count_max` stays 1500 -- drop toward ~800 only after the per-sweep heartbeat's
rejected-counts confirm typical warm solves sit well under it.

## Mutation sweep cost — the <24 h rework (2026-07-09, after job 64745)

Job 64745 (N=48, 12 sweeps/stage, 44 h governor) sat >3 h in SMC stage 1 at 100%
util / ~300 W: every early-ladder sweep step was gated at the full `count_max=5000`,
because a warm MALA proposal had NO cap of its own and ~30% of prior mass is
non-convergent — with 48 proposals per sweep, P(≥1 bad) ≈ 1 while the cloud is
prior-like, and the full-width lockstep while_loop runs at the slowest lane. On top of
that the warm gradient ran the chemistry TWICE (a separate primal-only diag solve just
to read accept_count). Projected ~3-6 h/stage × 15-40 stages ≫ any wall. Fixes (all
wired, suite 18/18):

- **`warm_count_max` = 1500 (schema default, banner-shown).** Warm mutation solves run a
  TWIN runner whose `_Statics` snapshot the smaller cap (`vulcan_chem.build_chem_model`
  builds it by temporarily setting `cfg.count_max`; `converged_y(..., warm_cap=True)`);
  the gate in `pipeline._make_batch_eval` rejects at `ACC >= warm_count_max`.
  MEASURED (smoke chain, same day): a MALA-small warm move needs ~780 accepted steps —
  the **conv_step=500 longdy certification window dominates the warm floor**, not
  count_min — so the first-guess cap of 800 would have rejected typical GOOD proposals;
  1500 gives ~2x margin and still cuts the gated worst case 3.3x vs 5000.
  Proposals that would converge in (1500, 5000] become extra MH rejections — a valid
  kernel either way. Cold/two-stage solves keep `count_max` (init phase 1 and the
  two_stage_z stage-2 increment genuinely need it), and so does the INIT phase-2
  gradient pass (see "Init phase 2 must run UNCAPPED" above — job 64854).
  `warm_count_max > count_max` raises (schema + build).
- **`warm_extrapolate` (opt-in, WIRED 2026-07-10, default off).** Seeds each proposal's
  warm solve at the first-order prediction `Y + (dy/dθ)·Δθ`, where dy/dθ = the converged
  column's parameter tangents read off the SAME jvp lanes that produce the gradient
  (zero extra compute; ~14 MB carried at N=96; `y_tangents` added to the checkpoint —
  resuming an extrapolated run from a tangent-less checkpoint raises). The seed's refs
  are set to the PROPOSAL's (lnZ, c_o) so the solver's refs-rescale is a no-op — the
  no-double-scaling recipe; getting this wrong silently double-applies the composition
  shift, which is why `tests/test_warm_extrap.py` pins seed-vs-plain likelihood parity
  on the real smoke chain. Measured: ~780 → ~470 warm steps (1.65x) on a MALA-small
  move. Flag off compiles today's exact kernel (trace-time gating). VALIDATION before
  production use: one `SYNTH=1` A/B (same seed, flag on vs off — same posterior, faster
  sweeps), then optionally `warm_count_max` → ~800 for the second half of the win.
- **accept_count rides the jvp chain** (`chem_solve_warm_diag` IS the warm gradient
  solve now): it is part of the runner's primal carry, integer-valued (tangent-free;
  stop_gradient + cast inside `_chain`). The duplicate diag while_loop is gone — ~2× on
  the chemistry per sweep step.
- **6 sweeps/stage** (was 12; schema + gpu preset). Published MALA-within-SMC practice
  is 3-10 with a good preconditioner; each sweep costs one full batched gradient.
- **N=96 (2026-07-09) → 144 (2026-07-10), `smc_rt_vjp_chunk=12`** at nu_pts=1652
  (8 serialized RT chunks at 96, 12 at 144; chemistry is full-width so N is nearly
  free — the wattage evidence and memory projection live in "GPU power headroom"
  below). Run `PROBE_MEMORY=1` once before the
  first production submit after ANY nu_pts / chunk / N change.
- **Per-sweep heartbeat**: `_make_mutation` logs `sweep k/n: accept= rejected= n_bad_grad=`
  via `jax.debug.callback` — a slow stage is visible as it happens, never hours of silence.
- **Walltime: 24 h PBS / 20 h governor** (gpu preset). Projected ~15-25 min/stage after
  the fixes; `CALIBRATE_ONLY=1` (~1 h) gives timing.json before committing a run.
- **conv_step 500 → 300 probed and REJECTED (2026-07-10, measured).** conv_step is the
  convergence ring depth; the criterion itself is time-based (y unchanged vs the run at
  t·st_factor=0.5, lookback clamped to the ring). Smoke-chain probe (same small MALA
  move): extrapolated warm 472 → 472 steps (ZERO saving — the ring never binds once the
  seed starts at the answer), plain warm 779 → 722 (7%), cold 4484 → 2885 — and that
  cold "saving" is the tell: the 300-window certifies a state that differs from the
  500-certified one by up to **0.072 dex**, 7x the yconv tolerance. It is not the same
  answer faster; it is a less-converged answer, which inflates exactly the warm-vs-cold
  path dependence validate_warm gates. Unlike dt_max (validated state-preserving), this
  changes the certified state — keep the master default 500. The step-count lever is
  warm_extrapolate (+ warm_count_max→800 after its A/B), not the certification window.
- **fp32 considered and REJECTED** (Isaac: only if much faster — it isn't): chemistry
  must stay f64 (VULCAN-JAX numerical-hygiene rule; rate constants span ~50 dex), and
  the RT is not the dominant cost, so fp32-RT is <2× on a minority term. Precedent
  exists (ExoJAX Gl229B ran fp32) if the RT ever dominates.

## Warm-vs-cold validation + exactness hygiene (2026-07-09, external-review response)

The warm mutation kernel's likelihood is history-dependent at the convergence
tolerance, so it is only approximately pi_beta-invariant — the one substantive point
from the external review. The measurement tool is `validate_warm`:

```
SMC_RETRIEVAL_PRESET=gpu python -m retrieval_framework.validate_warm vulcan-retrieval/runs/w39b_smc_retrieval
```

It cold re-solves the checkpointed cloud (init phase-1-equivalent, ~minutes) and
compares against the warm-carried logL. PASS gate: max|dlogL| < 0.1 over the cloud
(tolerance predicts ~1e-2; the cloud's logL spread is ~n_dim/2 ≈ 5). Run once per
production run; quote the result in the paper together with the init reject fraction
(the operational prior is p(theta | chemistry converges)). FAIL exits nonzero and
says what to do (tighten yconv_cri or rerun `smc_chem_mode="cold"`).

Same pass fixed: pilot-tuner PRNG-key reuse (dormant — only the non-default
`mcmc_auto_tune and not mcmc_stage_adapt` path; now `fold_in`-decorrelated), the
silent nonfinite-L floor + logZ-increment skip in the SMC loop are now raises (both
unreachable in a healthy run — invariant checks, not normalizations), and plot_smc
stamps `[TEMPERED beta=...]` on corner + spectrum figures when a governor-stopped
run hasn't reached beta=1. Rejected from the same review, with measured reasons (see
memory/README): steady-state-adjoint gradient swap, stiffness bucketing, delayed
acceptance (all void under the lockstep per-step cost model), fp32 RT, exojax
unpinning, remat retuning.

## GPU power headroom → N=144 + XLA A/B candidates (2026-07-10)

Reading the GH200 monitor correctly: nvidia-smi "100% util" only means SOME kernel was
resident each sample — the WATTAGE is the honest saturation signal (700 W cap). Job
64854's trace: ~290-300 W during the 192-lane init-1 primal, ~360-390 W during the
672-lane gradient pass. That jump is the load-bearing observation: **batch width fills
the GPU**; the sequential solver chain itself cannot be shortened by idle silicon (a
step can't start before the previous finishes), so headroom converts to WIDTH
(statistics) or to per-step LAUNCH-OVERHEAD reduction (speed) — nothing else.

- **Width: gpu preset raised N 96 → 144 (2026-07-10, "more aggressive" per Isaac).**
  Chemistry rides ~free; RT-vjp goes 8 → 12 serialized chunks of 12 (tail ×1.5).
  **MEASURED (probe job 64944): peak memory is WIDTH-INDEPENDENT** — FULL cold_vg at
  N=144 and FULL init_vg at 152 are both **73.25 GiB, byte-identical to N=96**. The
  peak lives inside the fixed-width RT-vjp chunk stage; the chemistry tangent buffers
  (~0.13 GiB/particle) are freed before it and only become the peak owner near
  N≈500. So N buys memory-free width; its only real cost is the serialized RT chunk
  count (N/12). **N=192 is memory-viable** (16 chunks, RT tail ×2 vs 96) if more
  particles are ever wanted. PROBE_MEMORY=1 stays REQUIRED after any N / chunk /
  nu_pts change — nu_pts and rt_vjp_chunk DO move the peak.
- **Speed: two untested XLA A/B experiments** (launch-overhead reduction; judged purely
  by `t_mutation_sweep_s` vs a baseline calibration — they change scheduling, not math;
  the PBS XLA_FLAGS line is `${XLA_FLAGS:-...}` so qsub -v overrides it cleanly):
  `qsub -v CALIBRATE_ONLY=1,XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=4' run_nas_w39b.pbs`
  `qsub -v CALIBRATE_ONLY=1,XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL,WHILE' run_nas_w39b.pbs`
  If a combo crashes or shows nothing, discard it; if command buffers capture the
  while_loop body as a CUDA graph, every stage gets faster at identical physics.
  Adopt a winner by editing the PBS default, with a note here.

## RT resolution — R~1000 (nu_pts~1652) is the MEMORY-SAFE DEFAULT (this keeps biting)

**RULE: keep R~1000, i.e. `nu_pts`~1652 for the production band. The RT-vjp gradient
memory scales with `nu_pts` (the absolute point count — NOT R; a narrow-band smoke can be
high-R at tiny nu_pts). NEVER raise `nu_pts` without `PROBE_MEMORY=1` first.** This has
OOM'd the run more than once, so as of 2026-07-09 it is enforced, not just documented:
- `config_schema.Config.nu_pts` DEFAULT is now **1652** (was 6000, itself a ~R10000 memory
  bomb). Any preset that forgets to set it gets the safe value.
- `validate_config` WARNS loudly when `nu_pts > 2500`, pointing at `PROBE_MEMORY`. The
  `runs/*/overrides/r3000_*.json` files (`nu_pts=5000`) trip this on purpose — they are
  experimental high-res configs and need a lowered `smc_rt_vjp_chunk` to fit.

History: the first SYNTH run past the reject-cull fix (job 64601) died in init phase 2 (the
RT-vjp gradient), allocating **343 GiB on the 96 GB GH200**, because `gpu_config` used
`nu_pts=16500` (native ~R10000). The OLD init masked it (it raised at phase 1 before
reaching the phase-2 gradient); the reject-cull fix exposed it. `nu_pts=1652` drops the
RT-vjp to ~34 GiB → fits the ~81 GiB pool (0.90 × 96 GB) with wide margin at the default
chunk. The data is ~150 binned points, so R~1000 (~11 model pts/bin) is ample; 16500 was
overkill. `overwrite=True` regenerates synthetic obs at the new resolution automatically.
`case.py::gpu_config` still sets `nu_pts=1652` explicitly (belt and suspenders).

**VULCAN-publication check (2026-07-08):** yconv_cri=0.01 + slope_cri=1e-4 are the
published Tsai+2017 steady-state values; the ballooning is the published adaptive-step
behavior; priors trace to Tsai 2023; baseline Kzz = GCM file (5e7 deep, Tsai-consistent);
metallicity/photochem match Tsai 2023. The `dt_max` cap is a documented deviation from
upstream's default that PRESERVES the longdy-defined steady state (truth bit-identical).

## Which parameter space fails (determined 2026-07-08)

- **T-P window failures = the HOT DEEP atmosphere**, not cold corners (cheap numpy Guillot
  sweep). High Tirr + low γ (strong greenhouse) push the deep (7 bar) layer above 3000 K.
  Under the new priors ~5.6% of draws are rejected (fine for the redraw sampler);
  physically those are implausible for W39b anyway. `too_cold` is ~0%.
- **Convergence tail (>10k steps)** is a *separate* axis (stiff transients relaxing from
  the ~1100 K baked baseline to a hot in-window T). Determine it empirically with the
  count_max calibration (`CALIBRATE_COUNT_MAX=1`, `CALIBRATE_COUNT_MAX_PROBE=5000`), which
  logs the per-draw θ of every censored draw. Expect it to be much smaller now that the
  hottest (out-of-window) draws are rejected before the chemistry and yconv_cri is 0.01.

## Layout / entry points (standalone sibling repo since 2026-07-11)

- THIS repo (dist vulcan-retrieval, import `retrieval_framework`, src layout) is a
  sibling of `VULCAN-JAX/` and `vulcan-jwst-tool/` under the project root: the SMC
  framework at `src/retrieval_framework/`, the shared forward-model engine at
  `src/retrieval_framework/forward/` (config, vulcan_chem, exojax_rt, interp_map,
  sensitivity -- import as `from retrieval_framework.forward import config`).
  Cases: `runs/<case>/case.py`. Also `tests/`, `examples/` (ex sensitivity_demo),
  `validation/`, `scripts/zco_information/`.
- The jwst tool lives in the SIBLING `vulcan-jwst-tool/` repo (dist vulcan-jwst-tool,
  import `jwst_tool`, console script `jwst-tool`); it depends on this dist for the
  forward engine and runs locally only (never deployed to NAS).
- Install editable (`pip install --no-deps -e .`; --no-deps because vulcan-jax is
  TestPyPI-only). The old sys.path "bundle import contract" is GONE: vulcan_chem no
  longer inserts VULCAN-JAX/src, so vulcan_jax resolves via its own (editable)
  install -- the PBS preflight installs and hard-checks this on NAS.
- Import order is guard-enforced: `retrieval_framework.forward.vulcan_chem` raises if
  exojax was imported first.
- Run from the repo root:
  `python -m retrieval_framework.run_smc runs/w39b_smc_retrieval`
  (also `calibrate_count_max`, `probe_memory`, `smoke_retrieval`, `plot_smc`,
  `validate_warm`). Suite: `python -m pytest tests -q`.
- `data/` = INPUTS at the repo root (cm24 obs + NAS-seeded opacity caches);
  `output/` = GENERATED npz caches (config.OUTPUTS, gitignored). Roots are portable
  via `$VULCAN_PROJECT_ROOT` = the directory CONTAINING this repo (forward/config.py
  raises loudly if the data tree is missing).
- Historical/design-log content lives in per-directory `notes.md` files (user
  convention, 2026-07-11); READMEs carry current usage only, one per repo.
- Figures still go to `../jax_paper/figures/`; never modify `../VULCAN-JAX`.

---

# 2026-07-13 — Live-T(P) condensation: on-graph rebuild replaces the isothermal escape hatch

Collaborator-requested refactor: condensation must be correct for arbitrary,
dynamically evaluated non-isothermal T-P profiles (Guillot included), on-graph,
not via a fixed-profile rebuild workaround.

**What was wrong.** `_prep` rebuilt rates, n_0, hydrostatic structure, Dzz, vm,
vs on the JAX graph per proposal, but every condensation quantity (the
ProfileVars `c_*` arrays: per-reaction sat_n and Dg, H2O/NH3 relax inputs, NH3
cold-trap argmin, fix-species sat-mix rows) was frozen at the baseline
structural T by the host packer. `build_chem_model` refused `use_condense`
except through `profile['_condense_validated_isothermal']=True` — and that
hatch was itself unsafe for WASP-39b, whose structural baseline was the GCM
file, not the requested isothermal profile (saturation tables at GCM T,
chemistry at T_iso).

**Fix.** VULCAN-JAX now splits the condensation state:
`conden.make_conden_spec` (static: species identity, k_arr rows, particle
m/(rho r^2) coefficients, relax/fix flags) + pure-JAX
`conden.build_conden_profile(spec, Tco, pco, n_0, Dzz)` (dynamic: every
T/structure-dependent array; jit/vmap/jvp-compatible; the one discrete output
is the NH3 cold-trap argmin). `OuterLoop._build_conden_static` delegates to
the same functions — verified bit-exact against the pre-refactor packer on
isothermal AND non-isothermal columns. The runner already read the conden
arrays from the ProfileVars carry each step, so `_prep` now rebuilds them at
the proposed T and splices them in. The NotImplementedError and the hatch are
gone; build-time refusals remain for moldiff-off (Dg would be silently zero),
empty/inert condense_sp, and use_sat_surfaceH2O.

**Convergence of condensing solves (all numbers measured on the nz=32
isothermal-400K S8 test column, S8 VMR 1e-4, 5→0.01 bar).** Naive conden-on
solves DO NOT converge:
- 1 um S8 particles: dt pinned at ~0.4 s (condensation-front timescale
  ~ 1/(Dg·(m/rho r^2)·(y−sat)) ≈ 0.1 s at 5 bar); t reached 1.1e3 s in 4000
  steps, longdy 10.4.
- 50 um (rainout-sized) particles: dt cap ~270 s; t = 1e6 s in 4000 steps,
  gas/sat down from 6.7 to 1.67 but longdy 1.55 — the steady state is
  TRANSPORT-LIMITED: the subsaturated upper S8 reservoir drains through the
  condensation front on the Kzz timescale (~L²/Kzz ≈ 1e9-1e10 s) while dt
  stays capped, so no step budget reaches it.
- Upstream's own answer (Earth/Jupiter cfgs) is the conden window +
  fix_species pin. With `fix_species_from_coldtrap_lev=True` the gas pin is
  EMPTY on an isothermal column (the cold-trap argmin degenerates to layer 0)
  and post-pin chemistry re-supersaturated S8 to 2560x against frozen k rows
  — use the whole-column variant (`=False`, master's op.py "TEST2022" branch).
- With the pin alone, longdy plateaued at 0.90: N2→NH3 kinetics at 400 K
  leave NH3 drifting at ~6e-19 VMR forever → `mtol_conv=1e-15`. Next gaters:
  S3 (2.6e-12) then S2 (7e-8) re-equilibrating against the pinned S8 at the
  cold top — still 18%/window at t=1.6e15 s, i.e. physically unreachable →
  `conver_ignore` extended with S/S2/S3/S4 (none is an RT molecule;
  SO2/H2S/SO stay gated). With the allotropes ignored the gate could fire at
  t≈2.3e3 s, BEFORE the window — certifying a half-rained column →
  `trun_min = stop_conden_time` bounds certification below.

The full recipe lives in `tests/test_condensation_live_tp.py` (retrieval) and
`jwst_tool.forward.CONDEN_CFG` (tool, stop_conden_time=1e6, trun_min=1e6).
Truncation caveat, documented everywhere: the secular reservoir drainage
(centuries) is cut off at stop_conden_time, exactly as in upstream conden
runs; and on planets too hot to condense the pin still freezes S8/S8_l_s at
their stop-time transient.

**AD.** jvp through the builder is exact vs FD (VULCAN-JAX
tests/test_conden_profile_builder.py); jvp through a condensing+pinned steady
state vs warm-started reconverged centered FD is asserted in the retrieval
test (sign + 15% relative on S8 gas/condensate columns; valid only away from
active-layer/cold-trap switches, which move discretely with T). Fisher through
condensation stays disabled in the tool for exactly that discreteness reason.

**Amendment (same day): the anchor-free cold column has NO reachable longdy
steady state.** After all of the above, the gate was held by well-mixed CO2
(1.7e-8 VMR, an RT species — cannot be ignored) creeping toward
thermochemical equilibrium at ~18% per time-doubling even at t = 1.6e15 s:
the 400 K no-photo synthetic column lacks the hot deep anchor / photolysis
sources that quench real columns, so its equilibration time (>=1e17 s)
exceeds any planet age. Resolution: upstream's own `runtime` cap —
`runtime = 1e14 s` (~3 Myr, far beyond every transport/condensation
timescale here); the runner terminates there (end_case 2) with the
condensation observables settled, ~2000 accepted steps, well under
count_max. The test asserts termination at the runtime cap, NOT by step-cap
exhaustion. The TOOL does not need this: its production runs have
photochemistry ON and warmer profiles — the WASP-107b Guillot + condensation
end-to-end run converged normally (longdy gate) and is cached under
forward v7. A cold no-photo tool corner would exhaust count_max and raise
loudly, which is the correct behavior.

## NAS job 65200 post-mortem + certification-gate rework (2026-07-15)

**Incident.** Job 65200 (N=144 real-data gpu run, 2026-07-13) died at SMC
stage 0 after 4.2 h: 16 of ~864 warm MALA proposals across the 6 sweeps
returned a FINITE likelihood but a NON-FINITE gradient, tripping the
zero-tolerance `_check_mutation_health` raise. The ~2.1 h two-phase init was
lost (the only checkpoint was per-stage, written after the health check), and
no per-particle forensics existed (the bads mask was reduced to a scalar).

**Root cause (mechanism).** The warm mutation usable gate trusted the accept
count alone (`ACC < warm_count_max`). The runner can certify an exit via the
stall fallback or the loose longdy branch on a marginally-stable column: the
clip-bounded PRIMAL certifies while the jvp TANGENT -- which relaxes through
the same while_loop with no stopping criterion of its own -- has not settled
and can amplify to Inf/NaN. Ruled out for that build: the K_eq exponent clip
(landed 2026-07-09, present), hybrid vm_mol (defaults flipped 2026-07-14,
after the run), dt ballooning (dt_max=1e11 first-class since 07-09),
condensation (off), init sizing (init passed; 27/192 oscillating/
stall-fallback columns culled with 48 spares -- the same fragile class the
kept survivors border on).

**Fixes (this pass).**
1. `ConvDiag` through the hot path: `vulcan_chem.converged_y(...,
   return_conv_diag=True)` returns (accept_count, longdy, longdydt,
   count_since_new_min, conv_normal), all free reads off the primal carry;
   `conv_normal` is the runner's canonical two-branch certification
   recomputed at the exit state (mirrors outer_loop._convergence_ok).
   The usable gate is now `valid & (ACC < wcmax) & conv_normal`
   (`pipeline._proposal_converged` -- ONE swappable predicate). Stalled
   proposals become an MH rejection class (`n_stalled`), logged per sweep
   next to warmcap, exported as `warm_stalled`, and covered by
   `mala_reversibility.py`'s asymmetry check. The init gates got the same
   treatment: phase 1 rejects stall-certified cold draws
   (`n_stalled_init` in init_stats -> f_conv accounting stays exact);
   phase 2's recert_fail includes non-certified exits.
2. Bad-grad forensics: the mutation runs as a host loop over a single-sweep
   jit (bit-identical RNG: same pre-split keys), so a bad-gradient event
   fails FAST at the offending sweep and dumps per-particle forensics
   (indices, theta, ACC, longdy, chem-tangent-vs-RT-vjp attribution via
   DAUX finiteness) to `bad_grad_stage###_sweep#.npz` before the loud raise.
3. Init-level checkpoint (`last_step=-1`, `init_checkpoint=1`), written right
   after `_init_state`; `RESUME=1` now recovers the init on a stage-0 death.
   Single `_write_checkpoint` writer keeps the schemas in lockstep.
4. **Runner-carry seeding regression fix (load-bearing).** The 2026-07-14
   VULCAN-JAX vm_branch port moved the termination budget to DYNAMIC carry
   fields (`count_max_dyn` etc., seeded at `_pack_state_from_runstate` from
   the packing runner's statics) and made the diffusion blend carry-driven
   (`hybrid_use_vm`). `state0` is packed ONCE under the COLD statics, so at
   current HEAD the warm twin's 1500-step cap was silently unbound (proposals
   marched to the cold 5000) and, under the new hybrid defaults, every solve
   would restart in upwind phase 0. `vulcan_chem._runner_carry_seed` now
   re-seeds the budget per solve and pins warm CONTINUATIONS to the
   central-difference operator when hybrid is resolved (the converged phase-1
   operator: same fixed point, no phase-0 re-preconditioning; pure-upwind
   configs keep upwind). Cold solves follow the resolved defaults (hybrid
   preconditioning, per Isaac's decision to track the current Shami
   defaults). `tests/test_warm_reject.py::test_warm_cap_binds_not_cold_cap`
   fails without the seed and passes with it. The resolved scheme is printed
   at build (`[chem] diffusion scheme: ...`).
5. Evidence semantics: the certification gate is part of the convergence
   indicator C -- logZ_box comparisons need MATCHED gate predicates; noted in
   run_smc_loop's docstring.

**Diagnostics.** `validation/diag_warm_stall_tangent.py`: the production
warm-capped continuation + jvp at gpu-preset prior corners, classifying every
solve against P0 (longdy<yconv_min), P1 (conv_normal -- the wired gate), P2
(tight branch); `--fd` cross-checks finite tangents; `--equivalence` compares
hybrid-vs-central cold fixed points (the re-baseline evidence for following
the new defaults). `calibrate_count_max` now also reports exit-longdy
percentiles + the stall-certified count.

**Caveats.** The 16/864 class is stochastic; if the local corner probe does
not reproduce a non-finite tangent, the in-run forensics dump makes the next
HPC occurrence self-diagnosing. If bad gradients persist WITH the conv_normal
gate (true tangent divergence at certified marginally-stable fixed points),
the follow-up is a tangent-norm-based REJECTION (never a zeroing). Hybrid
cold-solve cost on the real prior is unmeasured -- read CALIBRATE_ONLY=1
timing + attrition before the next 24 h submit; all pre-change checkpoints/
synthetic caches are stale per the standing regeneration rule.

**Measured (2026-07-15, local corner probe, `diag_warm_stall_tangent.py
--scheme default`, gpu preset).** Under the new hybrid defaults the full
warm kernel works end-to-end (hybrid cold solves, central warm continuation,
warm cap binding at 1501). Predicate table over 8 warm probes at 4 prior
corners: every HEALTHY warm proposal certifies on the LOOSE branch
(longdy 0.04-0.09 >> yconv_cri=0.01), so P2 (tight-branch-only) would reject
the ENTIRE warm kernel -- ruled out. P1 (conv_normal at exit, the wired gate)
kept all clean solves and rejected only the genuinely capped one: zero false
rejections. No non-finite tangent reproduced locally (expected: the
production class was ~2% stochastic); the in-run forensics dump owns that
case. NEW observation: hybrid's phase-flip budget extension let a cold
stage-1 solve run to 6002 accepted steps (past the static count_max=5000)
and exit UNcertified while stage 2 then certified at 4062 -- the worst-stage
`ACC >= count_max` phase-1 gate rejects such draws (conservative, absorbed
by oversample), so expect the phase-1 reject fraction to RISE somewhat vs
job 65200's central-diffusion 26%; read it off CALIBRATE_ONLY before the
24 h submit and resize init_oversample if needed.

**Measured (same day, `--scheme central` + `--equivalence`, RT-masked
VMR > 1e-12).** Central scheme (job-65200 physics): the hot+lowKzz corner
warm probe caps at 1501 with longdy = 1.31 -- far from steady state, primal
finite: the pathology family, correctly labeled non-usable by every
predicate. Zero P1 false rejections under either scheme; P2 rejects EVERY
healthy warm proposal in both (loose-branch certification is the norm) --
ruled out permanently. Hybrid-vs-central cold fixed points on CERTIFIED
points: worst 0.182 dex on RT-relevant cells (baseline and hiCO corners at
0.001 dex; RT tracers <= 0.095 dex) -- within the documented ~0.16 dex
central-scheme convergence floor. The only larger value (0.205 dex) is the
corner where NEITHER scheme certifies (acc 5001/6002, both
conv_normal=False; a production phase-1 reject under either scheme).
Hybrid cold-solve overhead: ~ +110 accepted steps of phase-0 preconditioning
(~6% at the baseline). Verdict: following the hybrid defaults with the
central-pinned warm twin is SAFE for the retrieval forward map, subject to
the CALIBRATE_ONLY attrition read on the real prior before the 24 h submit.

## 2026-07-15: run_diag(return_atm=True) export for adjoint callers

`forward.vulcan_chem` `run_diag` gained an optional `return_atm=True` that
additionally returns the theta-dependent `AtmStatic` (`atm_T` from `_prep`:
live Tco/Ti/M/Kzz/Dzz/vm/vs + refreshed geometry) the runner was actually
driven with. Reverse-mode adjoint calls (`vulcan_jax.steady_state_grad`)
must linearize around exactly that object; before this export a downstream
caller could only reach the setup-time baseline AtmStatic, which the
adjoint scope audit correctly flags as `stale_geometry` whenever theta
carries a T-P or Kzz offset. The two-tuple `(final, init)` contract is
unchanged for existing callers. First consumer: vulcan-jwst-tool's
`adjoint_diag.py` diagnostics panel (scope-audited dL/dlnk + dL/dT of the
target molecule's photosphere abundance). Wording fix in CLAUDE.md's
condensation guardrail while here: the 0.91 jvp-vs-FD figure is a RELATIVE
ERROR (~91% wrong tangent), stated so it cannot be misread as a 0.91
agreement ratio.

## NAS job 65789: the residual class identified -- tangent-blown CERTIFIED proposals (2026-07-15)

The gate-rework rerun died at stage 0 sweep 1 -- and the new instrumentation
did its job: 21 min to a named offender instead of 4.2 h to a bare count, the
init preserved by the init-level checkpoint, forensics dumped. The offender:
particle 12, accept_count 470 (well under the warm cap), longdy 0.087 (a
GENUINE loose-branch canonical certification; stalled=0 that sweep),
non-finite on the CHEMISTRY-TANGENT side. Conclusion: the 65200/65789 class
is NOT stall exits -- it is forward-mode tangent divergence at
marginally-stable CERTIFIED fixed points. No primal-side predicate can flag
it (the tight branch rejects every healthy proposal -- measured), and at
~1%/sweep at prior-like beta a zero-tolerance raise means NO production
ladder can pass stage 0.

FIX (the plan's pre-authorized contingency): the sweep now MH-REJECTS a
bad_grad proposal by flooring its L to -1e30 -- never a zeroed gradient: the
old eval-level zero-for-hygiene meant a zeroed-gradient proposal could be
ACCEPTED with a corrupted MH ratio, which is exactly the silent random-walk
degradation the raise guarded against; the floor closes it properly. The
class is a third state-dependent rejection alongside warmcap/stalled:
logged per sweep (badgrad=), per-stage history checkpointed + exported
(tangent_rejected / smc_tangent_rejected), forensics dumped on every
occurrence. The loud raise is retained for the systematic regime: a sweep
exceeding ceil(smc_tangent_reject_max_frac x N) events (Config field,
default 0.05 -> 8 at N=144; 0.0 restores zero-tolerance) aborts.
Detailed-balance status: same contract as warmcap (visible, ~0 late-ladder,
mala_reversibility). Evidence for the tolerance: 65200 saw 1-4 events/sweep
(0.7-2.8%) at beta=3e-5; a systematic AD bug shows as tens of percent.
Resume semantics: the fix changes only proposal REJECTION handling, not the
forward map or likelihood anchoring -- the 65789 init-level checkpoint
remains valid; RESUME=1 after pulling this commit re-enters the ladder at
beta=0 without re-paying the 2.1 h init.
