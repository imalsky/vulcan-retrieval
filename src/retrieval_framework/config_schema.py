"""Static configuration schema for one VULCAN-JAX -> ExoJax transmission SMC retrieval.

This mirrors the SWAMPE ``pipeline.Config`` pattern: a single frozen dataclass, with
per-planet ``*_config()`` PRESETS living in each run's ``case.py`` (see
``runs/w39b_smc_retrieval/case.py``), not here. The parameter vector is

    theta = [ lnZ, dln(C/O), lnKzz,   <T-P params>,   lnR0,  offset_g ... ]
            |------ VULCAN chemistry (3) ------|      radius   inter-instrument
                                 + ExoJax Guillot T-P            offsets (G-1)

The chemistry parameters (lnZ, dln(C/O), lnKzz) and the T-P parameters all require
re-converging VULCAN and are the *expensive* directions of every forward-mode
gradient; ``lnR0`` and the instrument offsets are applied analytically after the
spectrum and are cheap.

Planet identity enters through explicit fields a case preset sets: the observed
spectrum (``obs_dir`` + ``obs_products`` + ``combo``), the VULCAN baseline config
module (``vulcan_cfg_name``), gravity/radii (``tp_gravity_cgs``, ``rp_cm``,
``rstar_cm``), and the priors. Field defaults document the shapes/scales of the
original WASP-39b application; every case preset overrides what defines its planet.

All fields are overridable per preset via kwargs, and at run time via the
``SMC_RETRIEVAL_OVERRIDES`` / ``SMC_RETRIEVAL_OVERRIDES_FILE`` JSON hooks read by
``retrieval.run_smc`` (identical mechanism to the SWAMPE driver).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Config:
    """Everything static for one retrieval run. A case preset builds one of these;
    the driver fills ``out_dir`` (default: <run_dir>/data/<preset>) when unset."""

    # ---- I/O & reproducibility ------------------------------------------------
    out_dir: Optional[Path] = None     # set by the driver from the run dir + preset
    run_label: str = ""                # short human label for plot titles (e.g. "WASP-39b")
    seed: int = 20260704
    log_level: str = "INFO"
    overwrite: bool = True

    # Precision is not a knob: VULCAN-JAX forces jax_enable_x64=True on import
    # (vulcan_chem), so the whole chemistry+RT chain runs in float64 unconditionally;
    # XLA preallocation is set by the PBS env. Neither is a Config field.

    # ---- forward-model fidelity (the "profile" dict consumed by vulcan_chem /
    #      exojax_rt; see config.FULL in the parent package) --------------------
    nz: int = 50                       # VULCAN vertical layers (50 -> ~1/3 the nz=150 cost)
    use_photo: bool = True             # REQUIRED for a correct forward-mode tangent (and for SO2)
    # Convergence uses the VULCAN-master canonical W39b criteria: yconv_cri=0.01 (NOT the
    # 1e-3 the sensitivity demo used for tight jvps). The operative convergence gate is
    # the loose branch (longdy<yconv_min=0.1) + photo-flux settling, so 1e-3 vs 0.01
    # barely changes gradient quality but the looser value avoids grinding extra
    # thousands of steps toward a criterion the run rarely reaches. slope_cri / yconv_min
    # / flux_cri are NOT overridden -> they inherit the vulcan_cfg_W39b master defaults.
    yconv_cri: float = 0.01
    molecules: Tuple[str, ...] = ("H2O", "CO2", "CO", "CH4", "SO2")
    nu_min: float = 1923.0             # ~5.2 um
    nu_max: float = 3450.0             # ~2.9 um  (NIRSpec G395H/PRISM red)
    # DEFAULT nu_pts is MEMORY-SAFE by design. R~1000 (nu_pts~1652 over the production
    # NIRISS+G395H band) is the standard resolution: the data is binned to ~150 points, so
    # R~1000 (~11 model pts/bin) is ample. The RT-vjp gradient memory scales with nu_pts
    # (NOT with R -- a narrow-band smoke can be high-R at tiny nu_pts), and blew to 343 GiB
    # (OOM on the 96 GB GH200, job 64601) at the old ~R10000 native resolution. The old
    # default here (6000) was exactly that memory bomb. NEVER raise nu_pts without
    # PROBE_MEMORY=1 first. See the RT-resolution note in ../../CLAUDE.md -- this has bitten
    # the run repeatedly.
    nu_pts: int = 1652
    art_nlayer: int = 60
    use_rayleigh: bool = True          # H2/He Rayleigh scattering (ExoJax; zero free params)
    co_mode: str = "fixed_O"           # C/O GUESS construction (elemental mode repairs it exactly)
    # Abundance-knob semantics. "elemental" (production default) makes lnZ / c_o EXACT
    # column elemental directions: after the mask-scaled guess the column is renormalized
    # to sum_i n_i = P/(kB T) per layer and linearly repaired on the runner's reservoir
    # species so the column ratios hit He/H = base, {O,N,S}/H = Z x base,
    # C/H = Z e^{c_o} x base exactly, and pv.atom_ini is rebuilt from that column --
    # conserved inventories are then path-independent (cold == warm by construction).
    # "masks" reproduces the legacy species-mask knob (published demo caches), whose
    # elemental leakage (~0.6%/e-fold of Z into H, N/S leakage via the fixed-O b_z)
    # and sum(n) != M init are documented in vulcan_chem. See chem.audit_init.
    abundance_mode: str = "elemental"
    reanchor_atom_ini: bool = True     # masks-mode only (elemental always re-anchors exactly)
    # Pressure-broadening perturber for HITRAN opacities: "air" (terrestrial widths,
    # the documented default approximation) or "h2he" (HITRAN planetary H2/He widths
    # where available, blended per config.H2HE_BROADENING_MIX; per-molecule coverage
    # printed at build; run validation/broadening_ab.py for the measured A/B).
    broadening: str = "air"
    # Two-stage solve (REQUIRED for a live lnZ/C-O response when the T-P is retrieved):
    # stage 1 converges the column at (T(theta), Kzz(theta)) with BASELINE composition;
    # stage 2 applies the lnZ/C-O scaling to that converged column and re-converges warm.
    # Rationale (measured, 2026-07-05): perturbing the cold EQ init and re-converging
    # through the violent T-displacement transient ERASES the initial-inventory
    # perturbation (converged CO change 1e-11 for a 5% metals step under a Guillot T-P,
    # vs the exact 5% response at T=T_base) -- the same class of init-forgetting the
    # SO2 Hessian campaign dodged with warm continuation. Stage 2 from the T-consistent
    # converged state is gentle, keeps the inventory, and is also the validated
    # warm-started-jvp pattern.
    two_stage_z: bool = True
    count_min: Optional[int] = None
    count_max: Optional[int] = None
    # Warm-continuation step cap for the MUTATION path (accepted steps). A proposal
    # still unconverged at warm_count_max is rejected there (-inf L, same convention as
    # the count_max reject, just a tighter threshold) instead of dragging the whole
    # full-width lockstep while_loop to the cold cap. This is THE early-ladder
    # wall-clock lever (diagnosed 2026-07-09, job 64745): while the cloud is prior-like,
    # essentially every sweep step contains at least one non-convergent-corner proposal,
    # so every sweep step used to pay count_max (5000) chemistry steps -- ~hours per
    # stage. VALUE (measured, smoke chain 2026-07-09): a MALA-small warm move needs
    # ~780 accepted steps -- the conv_step=500 longdy certification window dominates the
    # warm floor, NOT count_min -- so an 800 cap would reject typical GOOD proposals;
    # 1500 keeps ~2x margin while still cutting the gated worst case 3.3x vs 5000.
    # (warm_extrapolate=True converges the same move in ~470 steps -- once A/B-validated
    # on the GPU, this cap can drop to ~800 alongside it.) Statistical effect:
    # proposals converging in (warm_count_max, count_max] become extra MH rejections --
    # a valid kernel either way; watch the per-sweep heartbeat's rejected counts.
    # Cold/two-stage solves are NOT affected (they keep count_max). Must be <= count_max
    # (build_chem_model raises); a cap below the effective warm floor rejects every
    # proposal, which the init gradient pass catches loudly within minutes.
    warm_count_max: int = 1500
    # Tangent-extrapolated warm starts (OPT-IN). Seed each MALA proposal's warm solve
    # from Y + (dy/dtheta)·dtheta -- dy = the converged column's parameter tangents,
    # which the gradient pass already computes per particle (and otherwise discards).
    # The proposal then starts a first-order prediction away from its own answer.
    # MEASURED (smoke chain 2026-07-09): the ~780-step MALA-small warm re-converge
    # drops to ~470 steps (1.65x), same column to 9e-3 dex; once validated on a GH200
    # SYNTH A/B, warm_count_max can drop toward ~800 for the second half of the win.
    # First-order in the move: exactly right for MALA-sized steps, useless for large
    # jumps (those hit warm_count_max either way). Requires smc_chem_mode="warm".
    # The extrapolated column carries the predicted lnZ/C-O shift, so the solver's own
    # refs-rescale is bypassed (refs are set to the proposal's values -- the validated
    # no-double-scaling recipe). Costs ~14 MB of carried tangents at N=96 and adds a
    # y_tangents field to the checkpoint; resuming an extrapolated run from a
    # checkpoint without tangents raises (restart or resume with this off).
    warm_extrapolate: bool = False
    # Max integrator step size (s). None -> VULCAN-master default (runtime*1e-5 = 1e17 s).
    # DIAGNOSED 2026-07-08: that master default is physically absurd for a photochemical
    # column (nothing evolves on >~1e12 s timescales) and is the ROOT of most of the
    # >10k-step tail -- at high Kzz the Ros2 local error stays tiny so dt balloons to
    # ~1e16 s and the solver SPINS in a large-dt oscillation (longdy stuck ~2-4, t marched
    # to ~1e17 s) instead of settling. Capping dt_max to ~1e12 converged a censored
    # high-Kzz draw in 986 steps (was >11000 uncapped) and does NOT touch normal draws
    # (their converged dt is ~1e8, well below the cap). This is a STEP-SIZE control, not
    # one of the master convergence CRITERIA (yconv_cri/slope_cri/... stay at master).
    dt_max: Optional[float] = None
    # Cold-init handling of prior draws whose chemistry doesn't converge within
    # count_max (a real, expected minority at extreme prior corners -- hot + extreme-Kzz
    # -- for a full-kinetics forward; see README.md sec K). Best practice (petitRADTRANS,
    # nested-sampling codes, Herbst-Schorfheide SMC): REJECT the failed draw with -inf
    # likelihood and OVERSAMPLE the init so the culled cloud still has N healthy
    # particles. pipeline._init_state draws ceil(N*init_oversample), rejects the
    # non-converged/non-finite draws, and keeps the first N survivors; it raises ONLY if
    # fewer than N survive (a systemic prior/config problem, not a few hard corners).
    #   init_oversample            -- draw factor for the cold init (>=1). 2.0 tolerates
    #                                 up to 50% non-convergence before the floor bites;
    #                                 the W39b calibration (job 64575) measured ~27% at
    #                                 count_max=5000, so 2.0 fills N=48 with margin.
    #   init_max_nonconverged_frac -- WARNING threshold on the observed reject fraction:
    #                                 above it, a loud warning fires (the prior reaches
    #                                 many non-convergent corners) but the run continues.
    # Both only apply when has_chem_state (real pipelines); stubs draw exactly N.
    init_max_nonconverged_frac: float = 0.1
    init_oversample: float = 2.0
    # Phase 2 (the init gradient pass) evaluates target_n + init_phase2_spare survivors
    # and culls the ones that certify cold but cannot RE-certify warm within the cold
    # count_max (marginal oscillators / stall-fallback columns; NAS jobs 64854 + 64897
    # saw 5/96 and 3/96 respectively -- a repeatable class, not a fluke). Width is
    # nearly free in the lockstep chemistry, and MEASURED width-free in memory too
    # (probe 64944: the 152-wide init eval peaks at the same 73.25 GiB as N=96 --
    # the peak is the fixed-width RT-vjp chunk stage; PROBE_MEMORY covers it). A true RT/AD failure at phase 2 (non-finite
    # forward WITHOUT a count_max-exhausted accept count) still raises.
    init_phase2_spare: int = 8
    fastchem_met_scale: float = 10.0   # BASELINE metallicity (x solar); lnZ is relative to this
    cfg_overrides: Dict[str, Any] = field(default_factory=dict)

    # ---- planet identity (case presets set these; empty -> shared-lib defaults) ---
    # VULCAN baseline config name for the chemistry pre-loop, loaded via
    # vulcan_jax.load_config (e.g. "W39b"); "" -> config.py's W39B_CFG_NAME.
    vulcan_cfg_name: str = ""
    # Planet/stellar radii for the RT depth normalization (cm); None -> config.py's
    # RP_CM / RSTAR_CM. tp_gravity_cgs below doubles as the RT's g_btm.
    rp_cm: Optional[float] = None
    rstar_cm: Optional[float] = None

    # ---- observed spectrum source ---------------------------------------------
    # obs_dir holds per-instrument product CSVs in the (Rp/Rs)-format observations.py
    # documents; obs_products maps group label -> csv filenames within obs_dir; combo
    # selects the groups to fit (order sets the offset reference = combo[0]).
    # obs_dir=None (or empty products) -> purely synthetic bin grid (offline smokes).
    obs_dir: Optional[Path] = None
    obs_products: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    combo: Tuple[str, ...] = ("NIRISS", "G395H")   # instrument groups -> two offsets
    obs_wl_lo: float = 1.00            # um (H2-H2 CIA short edge)
    obs_wl_hi: float = 5.28            # um (model band red edge)

    # ---- T-P profile (ExoJax built-ins: exojax.atm.atmprof) -------------------
    # "guillot" : atmprof_Guillot(P, g, kappa, gamma, Tint, Tirr, f) -- the built-in
    #             irradiated analytic profile (uses jnp.exp, forward-mode-clean).
    # "powerlaw": atmprof_powerlow(P, T0, alpha).
    tp_model: str = "guillot"
    tp_gravity_cgs: float = 422.0      # WASP-39b surface gravity (config.GS_CGS)
    tp_f: float = 0.25                 # 1/4 = whole-planet average (transmission terminator)
    tp_Tint_K: float = 150.0           # fixed interior temperature (transmission barely constrains it)
    tp_infer_gamma: bool = True        # retrieve log10(gamma); False -> fixed no-inversion
    tp_gamma_fixed: float = 0.4        # used only when tp_infer_gamma=False

    # ---- clouds (ExoJax powerlaw_clouds: kappa(nu)=kappac0*(nu/CLOUD_NUC0)^alphac,
    #      cm^2 per gram of atmosphere, uniformly mixed; alphac=0 -> gray deck) -----
    use_clouds: bool = True
    prior_log10kappa_cloud: Tuple[float, float] = (-7.0, 1.0)   # cm^2/g at 3.5 um
    truth_log10kappa_cloud: float = -6.5                        # ~cloud-free injection
    prior_cloud_alpha: Tuple[float, float] = (0.0, 6.0)         # 0=gray, 4~Rayleigh haze
    truth_cloud_alpha: float = 1.0

    # ---- inference toggles ----------------------------------------------------
    infer_lnZ: bool = True
    infer_c_o: bool = True
    infer_lnKzz: bool = True
    infer_lnR0: bool = True
    infer_offsets: bool = True         # one flat depth offset per instrument group beyond the reference

    # ---- priors (all bounded; uniform unless noted). Truth = synthetic-injection
    #      value, ignored for real-data runs -----------------------------------
    # lnZ is relative to the fastchem_met_scale baseline: lnZ=0 -> 10x solar here.
    prior_lnZ: Tuple[float, float] = (-2.303, 2.303)     # ~1x .. ~100x solar
    truth_lnZ: float = 0.0
    # dln(C/O) about the fixed-O baseline. UPPER BOUND CONSTRAINT: the fixed-O knob's
    # O-only compensation b_z must stay positive, which on the 10x-solar W39b column
    # holds only for c_o < 0.566 (printed at build; retrieval_forward raises if the
    # prior reaches it). 0.45 leaves margin for hotter columns (worst-layer b_z ~ 0.25)
    # and spans C/O up to ~0.86 about the 0.549 baseline.
    prior_c_o: Tuple[float, float] = (-1.6, 0.45)
    truth_c_o: float = 0.0
    prior_lnKzz: Tuple[float, float] = (-5.0, 5.0)        # eddy-diffusion multiplier
    truth_lnKzz: float = 0.0
    # Guillot irradiation temperature: WASP-39b Teq ~ 1166 K -> Tirr = sqrt(2)*Teq ~ 1650 K
    # (with f=0.25 the skin temperature is ~0.70*Tirr ~ 1150 K, the JWST-inferred limb T).
    prior_Tirr: Tuple[float, float] = (800.0, 2600.0)     # K
    truth_Tirr: float = 1650.0
    prior_log10kappa: Tuple[float, float] = (-3.5, 0.5)   # IR opacity, cm^2/g (log10)
    truth_log10kappa: float = -2.0
    prior_log10gamma: Tuple[float, float] = (-2.0, 0.7)   # kappa_v/kappa_th (log10)
    truth_log10gamma: float = -0.4
    prior_lnR0: Tuple[float, float] = (-0.08, 0.08)       # reference-radius log scaling
    truth_lnR0: float = 0.0
    prior_offset_ppm: Tuple[float, float] = (-800.0, 800.0)   # per-group depth offset, ppm
    truth_offset_ppm: float = 0.0

    # A free multiplicative noise-inflation term (Line 2015 style) guards against
    # underestimated JWST error bars; k multiplies every sigma. Off by default.
    infer_noise_inflation: bool = False
    prior_noise_inflation: Tuple[float, float] = (0.5, 3.0)   # log10_uniform
    truth_noise_inflation: float = 1.0

    # ---- data source ----------------------------------------------------------
    # False : fit the real observed spectrum (obs_dir/obs_products above).
    # True  : inject the truth_* parameters, add Gaussian noise at the real per-bin
    #         sigma (or the synthetic grid's), and fit that (recovery self-test).
    generate_synthetic_data: bool = False

    # ---- inference: BlackJAX adaptive-tempered SMC + forward-mode-jvp MALA -----
    run_inference: bool = True
    # Expert override: allow gradient-MALA inference with condensation ON. OFF by
    # default because the only SMC mutation kernel is gradient-based MALA and the
    # forward-mode gradient through a condensing+pinned steady state is NOT
    # reliably differentiable -- the pinned S8 state's jvp disagrees with FD at
    # O(1) (0.91 relative measured; tests/test_condensation_live_tp.py), the same
    # reason Fisher-through-condensation is refused in vulcan-jwst-tool. Condensation
    # FORWARD solves (run_inference=False, synthetic generation) are always allowed.
    allow_condense_inference: bool = False
    smc_num_particles: int = 48
    smc_target_ess_frac: float = 0.6
    # MALA sweeps per tempering stage. Each sweep costs one full batched gradient
    # (chem jvp lanes + RT vjp) -- the dominant per-stage cost -- so this is a LINEAR
    # wall-clock knob. Published practice for preconditioned-MALA-within-SMC is 3-10
    # steps per stage (k=3 is Chopin & Ridgway's floor, called "very sub-optimal" only
    # for HARD stages by Dau & Chopin; Buchholz+ 2018 adaptively stop near ~5 on
    # well-preconditioned targets). With the absolute-std preconditioner + per-stage
    # step adaptation here, 6 is the right planning number; the old 12 was ~2x generous.
    smc_num_mcmc_steps: int = 6
    # Preconditioned MALA with the staged forward-jvp(chem)+vjp(RT) gradient -- the
    # only supported kernel. No gradient-free fallback exists ON PURPOSE: a flagged
    # gradient pathology raises loudly instead of degrading the sampler.
    smc_mcmc_kernel: str = "mala"
    mala_step_size: float = 0.2
    smc_max_steps: int = 40             # max tempering stages before giving up on beta=1
    smc_use_custom_gradients: bool = True   # forward-mode value-and-grad wrapper (memory-stable, no vjp tape)
    smc_custom_grad_max_dim: int = 16
    # "block": only chem+T-P dims take tangents through the VULCAN while_loop; lnR0 is
    #          one RT-only jvp; offsets/noise are analytic (exact, ~25-35% cheaper).
    # "naive": every u-dimension through the full chain (the SWAMPE pattern; cross-check).
    # (These per-particle paths remain for validation; the SMC hot path is the staged
    # batched evaluator -- see smc_chem_mode / smc_rt_chunk below.)
    gradient_mode: str = "block"
    # "warm": every MCMC proposal's chemistry re-converges by CONTINUATION from the
    #         particle's carried converged column (incremental lnZ/C-O scaling -- the
    #         validated Hessian-campaign pattern). ~count_min-step solves instead of
    #         full cold two-stage solves; the cold map is used once, at initialization.
    # "cold": the published solve-from-baseline (two-stage) map for EVERY evaluation
    #         (the pre-2026-07-06 behavior; ~10-30x more chemistry steps per sweep).
    smc_chem_mode: str = "warm"
    # Particles per lax.map chunk through the ExoJax RT. 0 = no chunking (single
    # all-particles vmap). Compile-only probe 2026-07-07 (nu_pts=5000): RT PRIMAL
    # ~0.22 GiB/lane (full width is fine); RT VJP is THE memory wall -- 18.4 GiB
    # for the first lane + ~9.4 GiB per additional (65.4 GiB at 6-wide vs the
    # ~81 GiB pool), even with the per-molecule jax.checkpoint in exojax_rt.
    # Scales with n_nu: re-probe (PROBE_MEMORY=1) before raising or changing nu_pts.
    smc_rt_chunk: int = 16              # primal-likelihood RT chunk
    # Gradient-sweep RT chunk. The default 6 is the value PROBED SAFE at nu_pts=5000;
    # the vjp memory scales ~linearly with nu_pts, so at the production nu_pts=1652 a
    # chunk of 12 fits with margin (the gpu preset sets it -- ~half the serialized RT
    # chunks per sweep). PROBE_MEMORY=1 before raising it further or changing nu_pts.
    smc_rt_vjp_chunk: int = 6
    # Particles per lax.map chunk through the CHEMISTRY GRADIENT stage. 0 = no
    # chunking (full width) -- default since the 2026-07-07 probe: STAGED chem
    # tangent lanes cost ~20 MB per lane-pair (0.78 GiB at 36 lanes, and
    # nu-independent), NOT the ~1.3 GB previously claimed here -- that figure was
    # the old all-in-one architecture's PreMODIT tangents (the 390 GiB OOM),
    # misattributed to photo temporaries. Primal chemistry is ~55 MB/lane
    # (5.3 GiB at 96-wide).
    smc_chem_chunk: int = 0

    # step-size auto-tuning (one-shot pilot) + per-stage adaptation
    mcmc_auto_tune: bool = True
    mcmc_tune_beta: float = 0.3
    mcmc_tune_particles: int = 12
    mcmc_tune_steps: int = 6
    mcmc_tune_iters: int = 6
    mcmc_target_accept_mala: float = 0.55
    mcmc_step_size_min: float = 1.0e-3
    mcmc_step_size_max: float = 3.0
    mcmc_tune_gain: float = 0.7
    # Per-stage adaptation: the MALA proposal is preconditioned with the ABSOLUTE
    # per-dim std of the freshly resampled cloud (clipped to [1e-3, mcmc_scale_clip]),
    # so the proposal narrows in lockstep with tempering; the scalar step size is then
    # only Robbins-Monro fine-tuned toward mcmc_target_accept_mala.
    mcmc_stage_adapt: bool = True
    mcmc_stage_adapt_gain: float = 1.0
    mcmc_scale_clip: float = 20.0

    # posterior draws
    num_samples: int = 48
    num_chains: int = 2

    # ---- posterior predictive -------------------------------------------------
    do_ppc: bool = True
    ppc_draws: int = 64
    ppc_chunk_size: int = 16

    # ---- walltime governor ----------------------------------------------------
    # Soft wall-clock budget (seconds). run_smc_loop stops cleanly after a tempering
    # stage once this is exceeded, so a 24 h PBS job always writes usable partial
    # output (0 -> no limit).
    walltime_seconds: float = 0.0

    # -------------------------------------------------------------------------
    def profile(self) -> Dict[str, Any]:
        """The dict consumed by vulcan_chem.build_chem_model / exojax_rt.build_rt_model."""
        p: Dict[str, Any] = dict(
            use_photo=bool(self.use_photo),
            nz=int(self.nz),
            yconv_cri=float(self.yconv_cri),
            molecules=list(self.molecules),
            nu_min=float(self.nu_min),
            nu_max=float(self.nu_max),
            nu_pts=int(self.nu_pts),
            art_nlayer=int(self.art_nlayer),
            use_rayleigh=bool(self.use_rayleigh),
            co_mode=str(self.co_mode),
            abundance_mode=str(self.abundance_mode),
            broadening=str(self.broadening),
            reanchor_atom_ini=bool(self.reanchor_atom_ini),
            fastchem_met_scale=float(self.fastchem_met_scale),
            cfg_overrides=dict(self.cfg_overrides),
            gs_cgs=float(self.tp_gravity_cgs),   # RT g_btm = the T-P gravity
        )
        if self.count_min:
            p["count_min"] = int(self.count_min)
        if self.count_max:
            p["count_max"] = int(self.count_max)
        p["warm_count_max"] = int(self.warm_count_max)
        if self.dt_max:
            p["dt_max"] = float(self.dt_max)
        if self.vulcan_cfg_name:
            p["vulcan_cfg_name"] = str(self.vulcan_cfg_name)
        if self.rp_cm is not None:
            p["rp_cm"] = float(self.rp_cm)
        if self.rstar_cm is not None:
            p["rstar_cm"] = float(self.rstar_cm)
        return p


# ---------------------------------------------------------------------------
# Presets live with each case (runs/<case>/case.py), not here: a preset IS the
# planet-specific part of a retrieval. The framework only defines the schema.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Parameter specification (the ordered, active parameter list + priors)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    prior_type: str   # {"uniform", "log10_uniform"}
    lo: float
    hi: float
    truth: float
    kind: str         # {"chem", "tp", "lnR0", "offset", "noise"} -- how the forward consumes it


def specs_from_config(cfg: Config, groups: Optional[List[str]] = None) -> List[ParamSpec]:
    """Build the ordered active parameter list. ``groups`` is the ordered instrument-group
    list from the observations (offsets are added for groups[1:] relative to groups[0])."""
    specs: List[ParamSpec] = []

    def add(name, label, lo, hi, truth, kind, prior_type="uniform"):
        if not (lo < hi):
            raise ValueError(f"prior bounds for {name}: lo={lo} !< hi={hi}")
        if prior_type == "log10_uniform" and (lo <= 0 or hi <= 0):
            raise ValueError(f"log10_uniform prior needs positive bounds for {name}")
        specs.append(ParamSpec(name, label, prior_type, float(lo), float(hi), float(truth), kind))

    # --- chemistry (order matters: converged_ymix expects [lnZ, c_o, lnKzz, <tp...>]) ---
    if cfg.infer_lnZ:
        add("lnZ", r"$\ln Z$", *cfg.prior_lnZ, cfg.truth_lnZ, "chem")
    if cfg.infer_c_o:
        add("c_o", r"$\Delta\ln(\mathrm{C/O})$", *cfg.prior_c_o, cfg.truth_c_o, "chem")
    if cfg.infer_lnKzz:
        add("lnKzz", r"$\ln K_{zz}$", *cfg.prior_lnKzz, cfg.truth_lnKzz, "chem")

    # --- T-P (ExoJax Guillot or power-law) ---
    if cfg.tp_model == "guillot":
        add("Tirr", r"$T_{\rm irr}$ [K]", *cfg.prior_Tirr, cfg.truth_Tirr, "tp")
        add("log10kappa", r"$\log_{10}\kappa_{\rm IR}$", *cfg.prior_log10kappa, cfg.truth_log10kappa, "tp")
        if cfg.tp_infer_gamma:
            add("log10gamma", r"$\log_{10}\gamma$", *cfg.prior_log10gamma, cfg.truth_log10gamma, "tp")
    elif cfg.tp_model == "powerlaw":
        add("T0", r"$T_0$ [K]", *cfg.prior_Tirr, cfg.truth_Tirr, "tp")
        add("alpha", r"$\alpha$", *cfg.prior_log10gamma, cfg.truth_log10gamma, "tp")
    else:
        raise ValueError(f"unknown tp_model {cfg.tp_model!r}")

    # --- radius nuisance ---
    if cfg.infer_lnR0:
        add("lnR0", r"$\ln R_0$", *cfg.prior_lnR0, cfg.truth_lnR0, "lnR0")

    # --- clouds (RT-only, like lnR0: cheap gradient dims) ---
    if cfg.use_clouds:
        add("log10kappa_cloud", r"$\log_{10}\kappa_{\rm cl}$", *cfg.prior_log10kappa_cloud,
            cfg.truth_log10kappa_cloud, "cloud")
        add("cloud_alpha", r"$\alpha_{\rm cl}$", *cfg.prior_cloud_alpha,
            cfg.truth_cloud_alpha, "cloud")

    # --- inter-instrument offsets (ppm), one per group beyond the reference ---
    if cfg.infer_offsets and groups is not None and len(groups) > 1:
        for g in groups[1:]:
            add(f"offset_{g}", rf"$\delta_{{{g}}}$ [ppm]", *cfg.prior_offset_ppm,
                cfg.truth_offset_ppm, "offset")

    # --- optional noise inflation ---
    if cfg.infer_noise_inflation:
        add("noise_inflation", r"$b$", *cfg.prior_noise_inflation, cfg.truth_noise_inflation,
            "noise", prior_type="log10_uniform")

    if not specs:
        raise ValueError("no parameters enabled for inference")
    return specs


def validate_config(cfg: Config) -> None:
    if cfg.smc_num_particles <= 0:
        raise ValueError("smc_num_particles must be > 0")
    if str(cfg.smc_mcmc_kernel).strip().lower() != "mala":
        raise ValueError("this retrieval only supports smc_mcmc_kernel='mala' "
                         "(staged fwd-jvp chemistry + vjp RT gradient); there is "
                         "deliberately no gradient-free fallback kernel")
    if not (0.0 < cfg.smc_target_ess_frac <= 1.0):
        raise ValueError("smc_target_ess_frac must be in (0, 1]")
    # Condensation forward solves are supported (on-graph rebuild from the live
    # T(P)), but gradient-MALA INFERENCE through a condensing+pinned steady state
    # is NOT validated: the fix_species pin captures the column at the first
    # accepted step past stop_conden_time, so a T perturbation shifts the accepted
    # step sequence and the forward-mode tangent for the pinned species disagrees
    # with finite differences at O(1) (0.91 relative measured;
    # tests/test_condensation_live_tp.py). The only mutation kernel is gradient
    # MALA, so an inference run would sample against unreliable gradients. Refuse
    # by default (loud-errors rule); allow_condense_inference=True is the explicit
    # expert opt-in for anyone who has independently validated their column.
    if (bool(cfg.cfg_overrides.get("use_condense", False))
            and cfg.run_inference and not cfg.allow_condense_inference):
        raise ValueError(
            "use_condense=True with run_inference=True is refused: condensation "
            "forward solves are supported, but gradient-MALA inference through the "
            "condensing+pinned steady state is not validated (the pinned-species "
            "forward-mode tangent disagrees with FD at O(1) -- 0.91 relative; the "
            "same reason Fisher-through-condensation is disabled in vulcan-jwst-tool). "
            "Run condensation as a FORWARD model (run_inference=False), or set "
            "allow_condense_inference=True only if you have independently validated "
            "the gradient on your column.")
    if int(cfg.init_phase2_spare) < 0:
        raise ValueError("init_phase2_spare must be >= 0")
    if not (1.0 <= cfg.init_oversample <= 10.0):
        raise ValueError("init_oversample must be in [1, 10] (draw factor for the cold "
                         "init so the reject-and-cull leaves N healthy particles)")
    if not (0.0 <= cfg.init_max_nonconverged_frac <= 1.0):
        raise ValueError("init_max_nonconverged_frac must be in [0, 1]")
    if str(cfg.smc_chem_mode).strip().lower() not in ("warm", "cold"):
        raise ValueError("smc_chem_mode must be 'warm' or 'cold'")
    if int(cfg.warm_count_max) < 1:
        raise ValueError("warm_count_max must be >= 1")
    if cfg.count_max is not None and int(cfg.warm_count_max) > int(cfg.count_max):
        raise ValueError(
            f"warm_count_max={cfg.warm_count_max} exceeds count_max={cfg.count_max}: "
            "the warm mutation cap exists to reject doomed proposals EARLIER than the "
            "cold cap, never later (build_chem_model enforces the same against the "
            "vulcan_cfg default when count_max is inherited)")
    if cfg.warm_extrapolate and str(cfg.smc_chem_mode).strip().lower() != "warm":
        raise ValueError("warm_extrapolate=True requires smc_chem_mode='warm' -- the "
                         "extrapolation seeds the warm continuation; there is nothing "
                         "to seed on the cold map")
    # The chemistry block [lnZ, c_o, lnKzz] is LOAD-BEARING and POSITIONAL:
    # pipeline.py / retrieval_forward.py / forward.vulcan_chem unpack the
    # parameter vector by fixed index (theta[0]=lnZ, theta[1]=c_o, theta[2]=lnKzz,
    # theta[3:3+n_tp]=T-P) and assume a length-(3+n_tp) chem+T-P prefix. Dropping
    # any one via specs_from_config shortens the vector and shifts every later
    # index, so the forward path silently reinterprets the parameters (and the
    # gradient path shape-errors). These three toggles were never meant to be
    # flipped independently; refuse loudly here rather than sample a mislabeled
    # posterior. There is deliberately no supported way to drop a chem dimension.
    if not (cfg.infer_lnZ and cfg.infer_c_o and cfg.infer_lnKzz):
        off = [n for n, on in (("infer_lnZ", cfg.infer_lnZ),
                               ("infer_c_o", cfg.infer_c_o),
                               ("infer_lnKzz", cfg.infer_lnKzz)) if not on]
        raise ValueError(
            f"{', '.join(off)}=False is not supported: the chemistry block "
            "[lnZ, c_o, lnKzz] is unpacked by fixed position downstream, so "
            "disabling one shifts the T-P and nuisance indices and silently "
            "reinterprets the parameter vector. Keep all three inferred (use a "
            "tight prior range if you want one effectively fixed).")
    if cfg.tp_model not in ("guillot", "powerlaw"):
        raise ValueError(f"unknown tp_model {cfg.tp_model!r}")
    if str(cfg.abundance_mode) not in ("elemental", "masks"):
        raise ValueError(f"unknown abundance_mode {cfg.abundance_mode!r} "
                         "(expected 'elemental' or 'masks')")
    if str(cfg.broadening) not in ("air", "h2he"):
        raise ValueError(f"unknown broadening {cfg.broadening!r} (expected 'air' or 'h2he')")
    # Planet identity must be declared explicitly by the case: without these the RT
    # would silently normalize with the shared-lib WASP-39b radius/gravity and the
    # chemistry would run WASP-39b's baseline column -- a silently-wrong retrieval of
    # the wrong planet. Fail loud instead (the case's PRESETS must set them).
    if not str(cfg.vulcan_cfg_name).strip():
        raise ValueError("vulcan_cfg_name is unset -- the case must name its VULCAN "
                         "baseline config (e.g. 'W39b', loaded from vulcan_jax/configs/); "
                         "refusing to silently fall back to the shared-lib WASP-39b default")
    if cfg.rp_cm is None or cfg.rstar_cm is None:
        raise ValueError(f"planet radii unset (rp_cm={cfg.rp_cm}, rstar_cm={cfg.rstar_cm}) -- "
                         "the case must set both (cm); refusing to silently normalize the "
                         "transit depth with the shared-lib WASP-39b radii")
    if not cfg.use_photo:
        # not fatal, but the forward-mode tangent is only validated photo-on.
        import warnings
        warnings.warn("use_photo=False: the forward-mode tangent is only validated with "
                      "photochemistry ON (see config.FULL notes). Proceed with caution.")
    # RT-vjp gradient memory scales with nu_pts (job 64601: nu_pts=16500 -> 343 GiB ->
    # OOM on the 96 GB GH200). R~1000 == nu_pts~1652 for the production band is the
    # memory-safe DEFAULT; warn loudly above it so a resolution bump can't silently
    # reintroduce the OOM (this has recurred). Not fatal -- a deliberate high-res run with
    # a heavier smc_rt_vjp_chunk can be valid -- but PROBE_MEMORY=1 FIRST.
    if int(cfg.nu_pts) > 2500:
        import warnings
        warnings.warn(
            f"nu_pts={cfg.nu_pts} exceeds the memory-safe ~1652 (R~1000) default: the "
            "RT-vjp gradient memory scales with nu_pts and OOM'd the 96 GB GH200 at 343 "
            "GiB when nu_pts=16500 (job 64601). Run PROBE_MEMORY=1 to confirm the RT-vjp "
            "fits (lower smc_rt_vjp_chunk if needed) before a production run.")


# ---------------------------------------------------------------------------
# Loud config banner (printed at the top of every run so nothing is a surprise)
# ---------------------------------------------------------------------------
def describe_config(cfg: Config, preset: str = "", specs: Optional[List[ParamSpec]] = None) -> str:
    """A prominent, human-readable dump of the RESOLVED configuration (after preset +
    overrides) -- forward-model fidelity, convergence criteria, T-P handling, data
    source, SMC settings, and the full parameter/prior table. Every entry point logs
    this so the exact numbers a run uses (nu_pts/resolution, count_max, priors, ...)
    are visible up front rather than buried in the code. Pure string formatting."""
    if specs is None:
        try:
            specs = specs_from_config(cfg, groups=list(cfg.combo))
        except Exception:
            specs = []
    W = 84
    bar = "=" * W

    def rule(title=""):
        return f"  --- {title} " + "-" * max(0, W - 8 - len(title)) if title else "  " + "-" * (W - 2)

    R = (int(cfg.nu_pts) - 1) / math.log(float(cfg.nu_max) / float(cfg.nu_min))
    wl_lo, wl_hi = 1e4 / float(cfg.nu_max), 1e4 / float(cfg.nu_min)
    cmax = "(vulcan_cfg default)" if cfg.count_max is None else str(int(cfg.count_max))
    cmin = "(vulcan_cfg default)" if cfg.count_min is None else str(int(cfg.count_min))
    dtmax = "(master default 1e17)" if cfg.dt_max is None else f"{cfg.dt_max:g}"
    data = ("SYNTHETIC (inject-and-recover at truth_*)" if cfg.generate_synthetic_data
            else "REAL observed spectrum")

    lines = [
        "", bar,
        f"  RETRIEVAL CONFIG    preset={preset or '?'}    {cfg.run_label or 'run'}"
        f"    out_dir={cfg.out_dir}",
        bar,
        rule("forward model"),
        f"    nz={cfg.nz}   nu_pts={cfg.nu_pts} (native R~{R:.0f}, {wl_lo:.2f}-{wl_hi:.2f} um)"
        f"   art_nlayer={cfg.art_nlayer}",
        f"    molecules: {' '.join(cfg.molecules)}",
        f"    photo={'ON' if cfg.use_photo else 'OFF'}   rayleigh={'on' if cfg.use_rayleigh else 'off'}"
        f"   co_mode={cfg.co_mode}   two_stage_z={'on' if cfg.two_stage_z else 'off'}"
        f"   reanchor_atom_ini={'on' if cfg.reanchor_atom_ini else 'off'}",
        f"    fastchem baseline metallicity: {cfg.fastchem_met_scale:g}x solar   "
        f"(lnZ is relative to this)",
        rule("convergence  (VULCAN-master criteria; slope_cri/yconv_min/flux_cri inherit vulcan_cfg)"),
        f"    yconv_cri={cfg.yconv_cri:g}   count_max={cmax}   count_min={cmin}   "
        f"warm_count_max={int(cfg.warm_count_max)} (mutation-proposal cap)",
        f"    dt_max={dtmax} s   (physical step cap; master default 1e17 balloons dt -> "
        "high-Kzz non-convergence)",
        f"    cold-init: draw {cfg.init_oversample:g}xN, REJECT non-converged draws (-inf), "
        f"keep first N healthy   (warn if reject frac > {cfg.init_max_nonconverged_frac:.0%}; "
        "raise only if < N survive)",
        rule("T-P profile"),
        f"    model={cfg.tp_model}   Tint={cfg.tp_Tint_K:g}K   f={cfg.tp_f:g}   g={cfg.tp_gravity_cgs:g}cgs"
        f"   infer_gamma={'on' if cfg.tp_infer_gamma else 'off'}",
        f"    drawn RAW (no clip); profiles leaving the modelable T window are REJECTED + REDRAWN",
        rule("data"),
        f"    {data}   band {cfg.obs_wl_lo:g}-{cfg.obs_wl_hi:g} um   groups={list(cfg.combo)}",
        rule("SMC  (adaptive-tempered + forward-jvp MALA)"),
        f"    N={cfg.smc_num_particles}   mcmc_steps={cfg.smc_num_mcmc_steps}   "
        f"max_stages={cfg.smc_max_steps}   target_ess_frac={cfg.smc_target_ess_frac:g}   "
        f"step={cfg.mala_step_size:g}",
        f"    gradient_mode={cfg.gradient_mode}   chem_mode={cfg.smc_chem_mode}"
        f"   warm_extrapolate={'on' if cfg.warm_extrapolate else 'off'}   "
        f"rt_chunk={cfg.smc_rt_chunk}   rt_vjp_chunk={cfg.smc_rt_vjp_chunk}   chem_chunk={cfg.smc_chem_chunk}",
        f"    walltime governor: {cfg.walltime_seconds / 3600.0:.1f} h"
        + ("  (no limit)" if cfg.walltime_seconds <= 0 else ""),
        rule(f"parameters ({len(specs)})   [prior : truth]"),
    ]
    for s in specs:
        pt = "log10U" if s.prior_type == "log10_uniform" else "U"
        note = ""
        if s.name == "lnZ":
            note = (f"  [{math.exp(s.lo) * cfg.fastchem_met_scale:.2g}-"
                    f"{math.exp(s.hi) * cfg.fastchem_met_scale:.2g}x solar]")
        elif s.name == "c_o":
            note = f"  [C/O {math.exp(s.lo) * 0.549:.2g}-{math.exp(s.hi) * 0.549:.2g}]"
        elif s.name == "log10gamma":
            note = f"  [gamma {10 ** s.lo:.2g}-{10 ** s.hi:.2g}]"
        elif s.name == "lnKzz":
            note = f"  [Kzz x{math.exp(s.lo):.2g}-x{math.exp(s.hi):.2g}]"
        tr = "n/a" if not math.isfinite(s.truth) else f"{s.truth:g}"
        lines.append(f"    {s.name:<17s} {pt}({s.lo:g}, {s.hi:g}){note:<26s}  truth={tr}")
    lines += [bar, ""]
    return "\n".join(lines)
