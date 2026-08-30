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

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# stdlib-only re-export of the engine constants (see forward/config.py); this
# keeps the schema import-light while the RT grid bottom stays single-sourced
from vulcan_forward.constants import ART_PBTM_BAR, ART_PTOP_BAR


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
    nz: int = 62                       # VULCAN vertical layers (62 -> ~1/3 the nz=188 cost; 6.3 layers/decade over 1e-9..7.6 bar)
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
    # Opacity treatment: correlated-k from the published ExoMolOP tables
    # (ExoMol/HITEMP high-temperature line lists with H2/He broadening already
    # applied, integrated over each R=1000 band offline; the band grid and the
    # 16-point quadrature come from the files, so there is no spectral-resolution
    # knob). This is the only value the engine accepts; validate_config refuses
    # anything else.
    opacity_mode: str = "exomolop"
    art_nlayer: int = 67
    art_ptop_bar: float = ART_PTOP_BAR   # model top: chemistry AND RT end here (engine rule)
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
    # Two-stage solve (REQUIRED for a live lnZ/C-O response when the T-P is retrieved):
    # stage 1 converges the column at (T(theta), Kzz(theta)) with BASELINE composition;
    # stage 2 applies lnZ/C-O to that T-consistent state and re-converges warm.
    # Starting the composition perturbation before the large T displacement can erase
    # its inventory response; the second stage preserves it.
    two_stage_z: bool = True
    count_min: Optional[int] = None
    count_max: Optional[int] = None
    # Warm-continuation step cap for the MUTATION path (accepted steps). A proposal
    # still unconverged at warm_count_max is rejected there (-inf L, same convention as
    # the count_max reject, just a tighter threshold) instead of dragging the whole
    # full-width lockstep while_loop to the cold cap. The conv_step=500 certification
    # window sets the effective warm floor; 1500 keeps margin without paying count_max.
    # Proposals needing more become ordinary MH rejections. Cold/two-stage solves keep
    # count_max, and validate_config requires this cap not to exceed it.
    warm_count_max: int = 1500
    # Tangent-extrapolated warm starts (OPT-IN). Seed each MALA proposal's warm solve
    # from Y + (dy/dtheta)·dtheta -- dy = the converged column's parameter tangents,
    # which the gradient pass already computes per particle (and otherwise discards).
    # It helps MALA-sized moves, not large jumps, and requires smc_chem_mode="warm".
    # The predicted state already carries the lnZ/C-O shift, so refs are updated to
    # avoid double scaling. Its tangents ride in checkpoints; a legacy checkpoint
    # without them is refused.
    warm_extrapolate: bool = False
    # Max integrator step size (s). None inherits the VULCAN default. Cases should cap
    # physically meaningless large-dt Ros2 oscillations without changing the canonical
    # convergence criteria.
    dt_max: Optional[float] = None
    # Cold-init handling of prior draws whose chemistry doesn't converge within
    # count_max (a real, expected minority at extreme prior corners -- hot + extreme-Kzz
    # -- for a full-kinetics forward). Best practice (petitRADTRANS,
    # nested-sampling codes, Herbst-Schorfheide SMC): REJECT the failed draw with -inf
    # likelihood and OVERSAMPLE the init so the culled cloud still has N healthy
    # particles. pipeline._init_state draws ceil(N*init_oversample), rejects the
    # non-converged/non-finite draws, and keeps the first N survivors; it raises ONLY if
    # fewer than N survive (a systemic prior/config problem, not a few hard corners).
    #   init_oversample            -- draw factor for the cold init (>=1). 2.0 tolerates
    #                                 up to 50% non-convergence before the floor bites.
    #   init_max_nonconverged_frac -- GATE on the observed reject fraction: above it
    #                                 _init_state RAISES. Conditioning on convergence
    #                                 removes part of the DECLARED prior, so a run that
    #                                 rejects heavily is sampling a different support.
    #                                 This is the per-run floor; certificate.validate
    #                                 applies the tighter release gates
    #                                 (CONV_ATTRITION_JUSTIFY / CONV_ATTRITION_FAIL).
    # Both only apply when has_chem_state (real pipelines); stubs draw exactly N.
    init_max_nonconverged_frac: float = 0.1
    init_oversample: float = 2.0
    # The independent demonstration that the region the solver rejected carries
    # negligible posterior mass (an artifact name, job number or DOI).
    # certificate.validate REFUSES a run whose attrition exceeds
    # CONV_ATTRITION_JUSTIFY with this left empty.
    attrition_justification: str = ""
    # Phase 2 evaluates target_n + init_phase2_spare survivors, culls columns that
    # cannot re-certify within count_max, and backfills from the spares. A true RT/AD
    # failure still raises.
    init_phase2_spare: int = 8
    fastchem_met_scale: float = 10.0   # BASELINE metallicity (x solar); lnZ is relative to this
    cfg_overrides: Dict[str, Any] = field(default_factory=dict)

    # ---- planet identity (every case MUST set these; unset is a hard error) ----
    # These defaults exist only so the dataclass is constructible; they are not
    # fallbacks. validate_config REFUSES an unset value rather than substituting
    # a shared-lib WASP-39b one, because silently modelling a different planet is
    # the failure this repo most wants to make impossible.
    # VULCAN baseline config name for the chemistry pre-loop, loaded via
    # vulcan_jax.load_config (e.g. "W39b").
    vulcan_cfg_name: str = ""
    # Planet/stellar radii for the RT depth normalization (cm).
    # tp_gravity_cgs below doubles as the RT's g_btm.
    rp_cm: Optional[float] = None
    rstar_cm: Optional[float] = None
    # Pressure (bar) at which rp_cm and tp_gravity_cgs are taken to apply.
    #
    # Pinned to the RT grid bottom, where ExoJAX defines radius_btm. Because lnR0
    # is inferred, changing the reference pressure only re-anchors that posterior;
    # doing so requires an explicit re-baseline and a compatible prior_lnR0.
    p_ref_bar: float = ART_PBTM_BAR

    # ---- observed spectrum source ---------------------------------------------
    # obs_dir holds per-instrument product CSVs in the (Rp/Rs)-format observations.py
    # documents; obs_products maps group label -> csv filenames within obs_dir; combo
    # selects WHICH groups to fit. The offset REFERENCE group is NOT combo[0]: it is
    # the group of the shortest-wavelength kept bin (observations.load_real_observations
    # orders groups by first appearance after the wavelength sort). With G groups the
    # likelihood gets G-1 offset parameters, one per non-reference group.
    # obs_dir=None (or empty products) -> purely synthetic bin grid (offline smokes).
    obs_dir: Optional[Path] = None
    obs_products: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    combo: Tuple[str, ...] = ("NIRISS", "G395H")   # 2 groups -> ONE offset (on the non-reference group)
    obs_wl_lo: float = 1.00            # um (H2-H2 CIA short edge)
    obs_wl_hi: float = 5.28            # um (model band red edge)
    # Drop product bins finer than this R before they reach the binning operator.
    # Published R100 products carry truncation remnants where a masked region cuts a
    # bin short: they are a few times narrower than the nominal grid, carry a much
    # larger sigma, and are fine enough that the missing LSF matters
    # (observations._refuse_unresolved_products refuses them). 0 disables the cut.
    obs_max_bin_R: float = 0.0

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
    # Power-law T-P index in T = T0 * P^alpha (tp_model="powerlaw"). Its own
    # prior: alpha is a dimensionless T(P) slope, not the Guillot opacity ratio
    # it used to borrow. 0 is isothermal; ~0.1 is the radiative-region slope of
    # an irradiated giant; negatives allow a mild inversion.
    prior_alpha: Tuple[float, float] = (-0.10, 0.40)
    truth_alpha: float = 0.10
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
    # step adaptation here, 6 is the right planning number.
    smc_num_mcmc_steps: int = 6
    # Preconditioned MALA with the staged forward-jvp(chem)+vjp(RT) gradient -- the
    # only supported kernel. No gradient-free fallback exists ON PURPOSE: a flagged
    # gradient pathology raises loudly instead of degrading the sampler.
    smc_mcmc_kernel: str = "mala"
    mala_step_size: float = 0.2
    smc_max_steps: int = 40             # max tempering stages before giving up on beta=1
    # Per-sweep systematic-breakage BACKSTOP for the tangent-blown class
    # (finite certified primal, non-finite forward-mode tangent). Such
    # proposals are handled as ZERO-DRIFT MALA moves -- eval-zeroed gradient
    # entries used consistently in both proposal densities, certified
    # likelihood decides acceptance -- logged per sweep as badgrad= with
    # per-particle forensics dumped. A sweep exceeding ceil(this * N) indicates
    # systematic AD breakage rather than the known theta-corner class and raises.
    smc_tangent_bad_max_frac: float = 0.25
    # "block": only chem+T-P dims take tangents through the VULCAN while_loop; lnR0 is
    #          one RT-only jvp; offsets/noise are analytic (exact, ~25-35% cheaper).
    # "naive": every u-dimension through the full chain (the SWAMPE pattern; cross-check).
    # (These per-particle paths remain for validation; the SMC hot path is the staged
    # batched evaluator -- see smc_chem_mode / smc_rt_chunk below.)
    gradient_mode: str = "block"
    # "cold": the published solve-from-baseline (two-stage) map for EVERY
    #         evaluation. The likelihood is then a FIXED, DETERMINISTIC function
    #         of theta -- the target MALA, SMC tempering, and a quoted Bayesian
    #         evidence all assume. THE DEFAULT.
    # "warm": every proposal re-converges by continuation from the particle's
    #         carried column. It is cheaper, but a likelihood evaluation depends on
    # the particle's CARRIED chemistry column, hence on sampler history, at the
    # convergence tolerance. A history-dependent target is not the fixed density
    # the sampler/evidence assume. Warm runs are stamped approximate and require
    # both post-run validators; cold is the publication default.
    smc_chem_mode: str = "cold"
    # Particles per lax.map chunk through the ExoJAX RT. 0 = one all-particle
    # batch. RT VJP is the memory wall; run PROBE_MEMORY=1 before raising widths
    # or changing the spectral band/art_nlayer.
    smc_rt_chunk: int = 16              # primal-likelihood RT chunk
    # Gradient-sweep RT chunk. Correlated-k carries a 16-point g axis through the
    # random-overlap folds; PROBE_MEMORY=1 before raising this or changing the grid.
    smc_rt_vjp_chunk: int = 6
    # Particles per chemistry-gradient chunk. 0 keeps the full-width staged batch;
    # chemistry memory is independent of the spectral grid.
    smc_chem_chunk: int = 0

    # MALA step size: the per-stage Robbins-Monro adaptation below is the only
    # tuner. mala_step_size seeds it.
    mcmc_target_accept_mala: float = 0.55
    mcmc_step_size_min: float = 1.0e-3
    mcmc_step_size_max: float = 3.0
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

    def profile(self) -> Dict[str, Any]:
        """The dict consumed by vulcan_chem.build_chem_model / exojax_rt.build_rt_model."""
        p: Dict[str, Any] = dict(
            use_photo=bool(self.use_photo),
            nz=int(self.nz),
            yconv_cri=float(self.yconv_cri),
            molecules=list(self.molecules),
            nu_min=float(self.nu_min),
            nu_max=float(self.nu_max),
            opacity_mode=str(self.opacity_mode),
            art_nlayer=int(self.art_nlayer),
            art_ptop_bar=float(self.art_ptop_bar),
            use_rayleigh=bool(self.use_rayleigh),
            co_mode=str(self.co_mode),
            abundance_mode=str(self.abundance_mode),
            reanchor_atom_ini=bool(self.reanchor_atom_ini),
            fastchem_met_scale=float(self.fastchem_met_scale),
            cfg_overrides=dict(self.cfg_overrides),
            gs_cgs=float(self.tp_gravity_cgs),   # RT g_btm = the T-P gravity
            p_ref_bar=float(self.p_ref_bar),      # where rp_cm/gs_cgs apply
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


# Presets live with each case (runs/<case>/case.py), not here: a preset IS the
# planet-specific part of a retrieval. The framework only defines the schema.


# Parameter specification (the ordered, active parameter list + priors)
@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    prior_type: str   # {"uniform", "log10_uniform"}
    lo: float
    hi: float
    truth: float
    kind: str         # {"chem", "tp", "lnR0", "cloud", "offset", "noise"} -- how the forward consumes it


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
        # T0 is the 1-bar temperature, so it legitimately shares the Tirr box.
        add("T0", r"$T_0$ [K]", *cfg.prior_Tirr, cfg.truth_Tirr, "tp")
        add("alpha", r"$\alpha$", *cfg.prior_alpha, cfg.truth_alpha, "tp")
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
    # Counts that silently produce a broken or empty run if they reach zero: a
    # 0-sweep ladder never mutates, 0 stages never tempers, 0 PPC draws writes an
    # empty envelope. Chunk sizes are batch splits where 0 means "one batch".
    for name in ("smc_num_mcmc_steps", "smc_max_steps", "ppc_draws", "ppc_chunk_size"):
        if int(getattr(cfg, name)) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(cfg, name)!r}")
    for name in ("smc_rt_chunk", "smc_rt_vjp_chunk", "smc_chem_chunk"):
        if int(getattr(cfg, name)) < 0:
            raise ValueError(f"{name} must be >= 0 (0 = no chunking), "
                             f"got {getattr(cfg, name)!r}")
    if not math.isfinite(cfg.walltime_seconds):
        raise ValueError("walltime_seconds must be finite (<= 0 means no limit)")
    if not (0.0 < cfg.smc_tangent_bad_max_frac <= 1.0):
        raise ValueError("smc_tangent_bad_max_frac must be in (0, 1] -- it is the "
                         "systematic-breakage backstop, not an off switch")
    if not (0.0 < cfg.mcmc_step_size_min < cfg.mcmc_step_size_max):
        raise ValueError(
            f"need 0 < mcmc_step_size_min < mcmc_step_size_max, got "
            f"{cfg.mcmc_step_size_min!r} and {cfg.mcmc_step_size_max!r}")
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
    # pipeline.py / retrieval_forward.py / vulcan_forward.vulcan_chem unpack the
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
    if str(cfg.opacity_mode) != "exomolop":
        raise ValueError(
            f"opacity_mode={cfg.opacity_mode!r} is not available: the sampled "
            "line-by-line path ('lbl') was removed with vulcan-forward 0.11.0 "
            "(measured 857 ppm rms / 3177 ppm max binned-shape error and 1.30x "
            "too much feature contrast on the production band, non-convergent; "
            "notes.md). Correlated-k over the ExoMolOP tables ('exomolop', the "
            "default) is the only opacity path -- drop the key.")


# Loud config banner (printed at the top of every run so nothing is a surprise)
def describe_config(cfg: Config, preset: str = "", specs: Optional[List[ParamSpec]] = None) -> str:
    """A prominent, human-readable dump of the RESOLVED configuration (after preset +
    overrides) -- forward-model fidelity, convergence criteria, T-P handling, data
    source, SMC settings, and the full parameter/prior table. Every entry point logs
    this so the exact numbers a run uses (band, count_max, priors, ...)
    are visible up front rather than buried in the code. Pure string formatting."""
    if specs is None:
        # Offset parameters are named per non-REFERENCE group, and the reference is
        # the wavelength-first group (see the combo field comment), NOT combo[0] --
        # naming the banner's offsets from cfg.combo prints the WRONG parameter for
        # any combo not already in wavelength order. Derive
        # the order the pipeline will actually use by reading the product CSVs (cheap,
        # numpy-only); band-edge bin drops can still differ slightly from the built
        # pipeline, which logs its resolved groups after build.
        groups = list(cfg.combo)
        if cfg.obs_dir and cfg.obs_products:
            try:
                from retrieval_framework import observations as OBS
                groups = list(OBS.load_real_observations(cfg)["groups"])
            except Exception:
                pass   # banner stays provisional; the pipeline logs resolved groups
        try:
            specs = specs_from_config(cfg, groups=groups)
        except Exception:
            specs = []
    W = 84
    bar = "=" * W

    def rule(title=""):
        return f"  --- {title} " + "-" * max(0, W - 8 - len(title)) if title else "  " + "-" * (W - 2)

    opa = "correlated-k (ExoMolOP tables, R=1000 bands, H2/He broadening)"
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
        f"    nz={cfg.nz}   band {wl_lo:.2f}-{wl_hi:.2f} um   art_nlayer={cfg.art_nlayer}",
        f"    opacity: {opa}",
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
        f"keep first N healthy   (RAISES if reject frac > {cfg.init_max_nonconverged_frac:.0%}; "
        "raise only if < N survive)",
        rule("T-P profile"),
        f"    model={cfg.tp_model}   Tint={cfg.tp_Tint_K:g}K   f={cfg.tp_f:g}   g={cfg.tp_gravity_cgs:g}cgs"
        f"   infer_gamma={'on' if cfg.tp_infer_gamma else 'off'}",
        "    drawn RAW (no clip); profiles leaving the modelable T window are REJECTED + REDRAWN",
        rule("data"),
        f"    {data}   band {cfg.obs_wl_lo:g}-{cfg.obs_wl_hi:g} um   groups={list(cfg.combo)}",
        rule("SMC  (adaptive-tempered + forward-jvp MALA)"),
        f"    N={cfg.smc_num_particles}   mcmc_steps={cfg.smc_num_mcmc_steps}   "
        f"max_stages={cfg.smc_max_steps} (per JOB; RESUME continues)   "
        f"target_ess_frac={cfg.smc_target_ess_frac:g}   step={cfg.mala_step_size:g}",
        f"    preconditioner: full cloud covariance (Cholesky)   "
        f"step tuning: {'per-stage Robbins-Monro' if cfg.mcmc_stage_adapt else 'fixed'}",
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
