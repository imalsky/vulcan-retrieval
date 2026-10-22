"""Static configuration schema for one VULCAN-JAX -> ExoJax transmission retrieval.

A single frozen dataclass; the planet and the run live in the run specification
(``runs/<case>/<name>.yaml``, ``run_setup.load_spec``), not here. The parameter vector is

    theta = [ lnZ, dln(C/O), lnKzz, [s_o], [kzz_slope],  <T-P params>,  lnR0, [lnM],
              VULCAN chemistry (3 + the optional entries)  Guillot T-P    radius, mass
              <cloud params>,  offset_g ...,   noise_inflation ]
              power law (2)    offsets (G-1)   optional: one b, or
              or deck (1)                      one b per group (G)

The chemistry parameters (lnZ, dln(C/O), lnKzz and the optional s_o = ln of the
S/O multiple and kzz_slope) and the T-P parameters all require re-converging
VULCAN and are the *expensive* part of every likelihood call; ``lnR0``, the
optional ``lnM`` (Gaussian, RT-only) and the cloud parameters enter only the
RT (no chemistry solve), and the instrument offsets are applied analytically
after the spectrum; all are cheap. The T-P block's deep parameter is Tint or,
with ``tp_deep_param="T1bar"``, the temperature at ``tp_deep_p_bar``.

Planet identity enters through explicit fields the spec sets: the observed
spectrum (``obs_dir`` + ``obs_products`` + ``combo``), the VULCAN baseline config
YAML (``vulcan_cfg_name``), gravity/radii (``tp_gravity_cgs``, ``rp_cm``,
``rstar_cm``), and the priors. Field defaults document the shapes/scales of the
original WASP-39b application; every spec sets what defines its planet.

Every field is set from the spec's ``config`` block; a field not named there keeps
its default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# stdlib-only re-export of the engine constants (see forward/config.py); this
# keeps the schema import-light while the RT grid bottom stays single-sourced
from vulcan_forward.constants import ART_PBTM_BAR, ART_PTOP_BAR

# Default lane count (the smoke spec's); the production spec sets cold_lanes itself
# (one kernel wave of a GH200 is 132 SMs).
OFF_GPU_LANES = 144


@dataclass(frozen=True)
class Config:
    """Everything static for one retrieval run; run_setup.load_spec builds one from
    the spec's ``config`` block and sets ``out_dir`` to output/<case>/<name>_nautilus."""

    # ---- I/O & reproducibility ------------------------------------------------
    out_dir: Optional[Path] = None     # set by load_spec, never by a spec key
    run_label: str = ""                # short human label for plot titles (e.g. "WASP-39b")
    seed: int = 20260704

    # Precision is not a knob: VULCAN-JAX forces jax_enable_x64=True on import
    # (vulcan_chem), so the whole chemistry+RT chain runs in float64 unconditionally;
    # XLA preallocation is set by the PBS env. Neither is a Config field.

    # ---- forward-model fidelity (the "profile" dict consumed by vulcan_chem /
    #      exojax_rt; the photo-on requirement: forward/config.py) ----------------
    nz: int = 62                       # VULCAN vertical layers (62 -> ~1/3 the nz=188 cost; 6.3 layers/decade over 1e-9..7.6 bar)
    use_photo: bool = True             # photochemistry (SO2 is photochemical)
    # VULCAN-master canonical W39b convergence (yconv_cri=0.01; 1e-3 costs thousands of
    # steps for no gradient gain). slope_cri / yconv_min / flux_cri are not
    # overridden: they inherit the vulcan_cfg_name config (vulcan3/configs/W39b.yaml).
    yconv_cri: float = 0.01
    molecules: Tuple[str, ...] = ("H2O", "CO2", "CO", "CH4", "SO2")
    nu_min: float = 1923.0             # ~5.2 um
    nu_max: float = 3450.0             # ~2.9 um  (NIRSpec G395H/PRISM red)
    # Opacity treatment: correlated-k from the published ExoMolOP tables
    # (ExoMol/HITEMP high-temperature line lists with H2/He broadening already
    # applied, integrated over each R=1000 band offline; the band grid and the
    # 16-point quadrature come from the files, so there is no spectral-resolution
    # knob). This is the only value the engine accepts (build_rt_model refuses
    # anything else).
    opacity_mode: str = "exomolop"
    art_nlayer: int = 67
    art_ptop_bar: float = ART_PTOP_BAR   # model top: chemistry AND RT end here (engine rule)
    use_rayleigh: bool = True          # H2/He Rayleigh scattering (ExoJax; zero free params)
    co_mode: str = "fixed_O"           # C/O GUESS construction (the engine repairs it exactly)
    count_min: Optional[int] = None
    count_max: Optional[int] = None
    # Max integrator step size (s). None inherits the VULCAN default. Cases should cap
    # physically meaningless large-dt Ros2 oscillations without changing the canonical
    # convergence criteria.
    dt_max: Optional[float] = None
    cfg_overrides: Dict[str, Any] = field(default_factory=dict)

    # ---- planet identity (every case MUST set these; unset is a hard error) ----
    # Placeholders so the dataclass is constructible; validate_config refuses an
    # unset value.
    # VULCAN baseline config name for the chemistry pre-loop, loaded via
    # vulcan3.load_config (e.g. "W39b").
    vulcan_cfg_name: str = ""
    # Planet/stellar radii for the RT depth normalization (cm).
    # tp_gravity_cgs below doubles as the RT's gravity at rp_cm; the planet mass
    # GM = g rp_cm^2 stays fixed when lnR0 scales the radius.
    rp_cm: Optional[float] = None
    rstar_cm: Optional[float] = None
    # Pressure (bar) at which rp_cm and tp_gravity_cgs are taken to apply.
    #
    # Default: the RT grid bottom, where ExoJAX defines radius_btm. lnR0 is the
    # log radius change at this level, so moving it changes what lnR0 means and
    # needs a compatible prior_lnR0 (the W39b case anchors at 1 mbar).
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
    obs_wl_hi: float = 5.28            # um (data-window red edge; bins past the band are dropped)
    # Drop product bins finer than this R before they reach the binning operator.
    # Published R100 products carry truncation remnants where a masked region cuts a
    # bin short: they are a few times narrower than the nominal grid, carry a much
    # larger sigma, and are fine enough that the missing LSF matters
    # (observations._refuse_unresolved_products refuses them). 0 disables the cut.
    obs_max_bin_R: float = 0.0

    # ---- T-P profile: exojax.atm.atmprof.atmprof_Guillot(P, g, kappa, gamma, Tint,
    #      Tirr, f), the built-in irradiated analytic profile (uses jnp.exp,
    #      forward-mode-clean); f = GUILLOT_F below --------------------------------
    tp_gravity_cgs: float = 422.0      # WASP-39b surface gravity (config.GS_CGS)
    tp_Tint_K: float = 150.0           # interior temperature when tp_infer_Tint=False
    tp_infer_gamma: bool = True        # retrieve log10(gamma); False -> fixed no-inversion
    tp_gamma_fixed: float = 0.4        # used only when tp_infer_gamma=False
    # Retrieve the deep T-P parameter. It sets the deep profile, and so the
    # CO/CH4/NH3 quench, through 3/4 Tint^4 tau at large tau; the T-P sub-vector
    # gains a 4th entry: Tint itself (tp_deep_param="Tint") or the temperature
    # at tp_deep_p_bar ("T1bar"), from which Tint follows in closed form
    # (T^4 is linear in Tint^4; a draw whose Tint^4 would be negative is
    # rejected as an out-of-window profile). The T1bar prior is uniform in a
    # quantity the window acts on directly, so it does not inherit the
    # window-shaped Tint prior (tp_profile).
    tp_infer_Tint: bool = False
    tp_deep_param: str = "Tint"        # {"Tint", "T1bar"}, with tp_infer_Tint
    tp_deep_p_bar: float = 1.0         # the level T1bar is quoted at (bar)
    # An adiabat below a free radiative-convective boundary pressure, matched
    # to the Guillot profile there (one more T-P entry, log10P_rcb): a
    # sensitivity model. With constant kappa nothing in the model makes the
    # column convective, so the join is imposed, and the adiabat (R/cp ~0.29)
    # is steeper than Guillot's deep 0.25.
    tp_adiabat: bool = False
    tp_nabla_ad: float = 0.29          # dlnT/dlnP of the H2-He adiabat near 1000-2000 K

    # ---- clouds: cloud_model "powerlaw" (ExoJax powerlaw_clouds:
    #      kappa(nu)=kappac0*(nu/CLOUD_NUC0)^alphac, cm^2 per gram of atmosphere,
    #      uniformly mixed; alphac=0 -> gray) or "deck" (an opaque grey deck
    #      below a free pressure, engine constants CLOUD_DECK_*) -------------------
    cloud_model: str = "powerlaw"
    prior_log10kappa_cloud: Tuple[float, float] = (-7.0, 1.0)   # cm^2/g at 3.5 um
    truth_log10kappa_cloud: float = -6.5                        # ~cloud-free injection
    prior_cloud_alpha: Tuple[float, float] = (0.0, 6.0)         # 0=gray, 4~Rayleigh haze
    truth_cloud_alpha: float = 1.0
    prior_log10P_deck: Tuple[float, float] = (-6.0, 0.8)        # bar; 0.8 ~ the grid bottom
    truth_log10P_deck: float = 0.5

    # ---- inference toggles (the chemistry block is always inferred) -----------
    infer_offsets: bool = True         # one flat depth offset per instrument group beyond the reference

    # ---- priors (all bounded; uniform unless noted). Truth = synthetic-injection
    #      value, ignored for real-data runs -----------------------------------
    # lnZ is relative to the case's BASELINE composition (the vulcan cfg's <X>_H,
    # 10x solar here): lnZ=0 -> that baseline.
    prior_lnZ: Tuple[float, float] = (-math.log(10.0), math.log(10.0))   # 1x-100x solar
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
    # Sulfur freed from the metallicity scaling: s_o = ln of the S/O multiple of
    # the baseline (S/H = e^(lnZ + s_o) times the baseline's), a fourth chemistry
    # entry after lnKzz. Off, sulfur scales with lnZ like O and N.
    infer_s_o: bool = False
    prior_s_o: Tuple[float, float] = (-math.log(10.0), math.log(10.0))
    truth_s_o: float = 0.0
    # Kzz shape: "scale" multiplies the vulcan cfg's profile by e^lnKzz;
    # "powerlaw" is Kzz0(kzz_p_ref_bar) e^lnKzz (P / P_ref)^-kzz_slope, with the
    # slope a chemistry entry after s_o (engine kzz_mode).
    kzz_mode: str = "scale"
    kzz_p_ref_bar: float = 1.0e-3
    prior_kzz_slope: Tuple[float, float] = (-1.0, 1.0)
    truth_kzz_slope: float = 0.0
    # The vulcan cfg's baseline metallicity in solar units (its <X>_H over the
    # Lodders 2020 photospheric values): the report quotes e^lnZ times this.
    baseline_metallicity_solar: float = 10.0
    # Guillot irradiation temperature: WASP-39b Teq ~ 1166 K -> Tirr = sqrt(2)*Teq ~ 1650 K
    # (with f=0.25 the skin temperature is ~0.6*Tirr: 0.59 at gamma << 1, 0.64 at 0.4).
    prior_Tirr: Tuple[float, float] = (800.0, 2600.0)     # K
    truth_Tirr: float = 1650.0
    prior_log10kappa: Tuple[float, float] = (-3.5, 0.5)   # IR opacity, cm^2/g (log10)
    truth_log10kappa: float = -2.0
    prior_log10gamma: Tuple[float, float] = (-2.0, 0.7)   # kappa_v/kappa_th (log10)
    truth_log10gamma: float = -0.4
    prior_Tint: Tuple[float, float] = (100.0, 500.0)     # K, used when tp_infer_Tint
    truth_Tint: float = 150.0
    # T at tp_deep_p_bar (K), used with tp_deep_param="T1bar": the default is
    # the engine's valid RT window; the irradiation term and the window shape
    # the accepted prior (report it).
    prior_T1bar: Tuple[float, float] = (320.0, 2980.0)
    truth_T1bar: float = 1500.0
    prior_log10P_rcb: Tuple[float, float] = (-1.0, 0.88)  # bar; used with tp_adiabat
    truth_log10P_rcb: float = 0.5
    prior_lnR0: Tuple[float, float] = (-0.08, 0.08)       # reference-radius log scaling
    truth_lnR0: float = 0.0
    # Planet mass as a nuisance: lnM = ln(M / M0) ~ N(0, prior_lnM_sigma), applied
    # in the RT at the fixed reference radius (GM scales e^lnM; engine
    # transmission_depth_r lnM). The chemistry's hydrostatic column and the
    # Guillot tau keep M0. Off, the mass is fixed at M0 = g rp^2 / G.
    infer_lnM: bool = False
    prior_lnM_sigma: float = 0.11                          # the literature mass error
    truth_lnM: float = 0.0
    prior_offset_ppm: Tuple[float, float] = (-800.0, 800.0)   # per-group depth offset, ppm
    truth_offset_ppm: float = 0.0

    # A free multiplicative noise-inflation term (Line 2015 style) guards against
    # underestimated JWST error bars; k multiplies every sigma. Off by default.
    infer_noise_inflation: bool = False
    prior_noise_inflation: Tuple[float, float] = (0.5, 3.0)   # log10_uniform
    truth_noise_inflation: float = 1.0
    # One inflation per observation group instead of one for every bin: b_g
    # multiplies the sigmas of group g's bins, every group included (the
    # offsets' reference group too), each with the prior above. Named
    # noise_inflation_<group>, in the pipeline's group order, after the
    # offsets. Needs infer_noise_inflation=True; off keeps the scalar b.
    noise_inflation_per_group: bool = False

    # ---- data source ----------------------------------------------------------
    # False : fit the real observed spectrum (obs_dir/obs_products above).
    # True  : inject the truth_* parameters, add Gaussian noise at the real per-bin
    #         sigma (or the synthetic grid's), and fit that (recovery self-test).
    generate_synthetic_data: bool = False

    # ---- inference ------------------------------------------------------------
    # Expert opt-in: a pipeline with condensation ON. Off by default: no
    # retrieval with condensation has been validated (certification and start
    # dependence of condensing columns are untested).
    # A condensation forward model goes through the engine directly
    # (vulcan_forward.vulcan_chem.build_chem_model), which has no such gate.
    allow_condense_inference: bool = False
    # Rows per lax.map chunk through the ExoJAX RT. 0 = one all-row batch.
    smc_rt_chunk: int = 16
    # Lanes the chemistry batches run on (> 0): min(k, draws) lanes, and a lane
    # that certifies is refilled with the next draw inside the same while loop
    # (vulcan_forward.converged_y_queue), so wall time follows total work / k.
    # The production spec sets it (one kernel wave of a GH200 is 132 SMs;
    # run_nautilus multiplies it by FFI_BLOCKS_PER_SM under the ffi solver).
    # The schema default is the off-GPU width.
    cold_lanes: int = OFF_GPU_LANES
    # Lanes refilled per refill pass. Bigger amortizes the refill over more
    # lanes; it is capped at cold_lanes.
    cold_refill_chunk: int = 8

    # ---- walltime governor ----------------------------------------------------
    # Soft wall-clock budget (seconds). run_nautilus stops once this is exceeded,
    # with its checkpoint written, and exits 0; RESUME=1 continues (0 -> no limit).
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
            kzz_mode=str(self.kzz_mode),
            kzz_p_ref_bar=float(self.kzz_p_ref_bar),
            cfg_overrides=dict(self.cfg_overrides),
            gs_cgs=float(self.tp_gravity_cgs),   # RT g at rp_cm = the T-P gravity
            p_ref_bar=float(self.p_ref_bar),      # where rp_cm/gs_cgs apply
        )
        # `is not None`, not truthiness: an explicit 0 must reach the engine,
        # never fall back to the vulcan_cfg value while the banner prints 0
        if self.count_min is not None:
            p["count_min"] = int(self.count_min)
        if self.count_max is not None:
            p["count_max"] = int(self.count_max)
        if self.dt_max is not None:
            p["dt_max"] = float(self.dt_max)
        if self.vulcan_cfg_name:
            p["vulcan_cfg_name"] = str(self.vulcan_cfg_name)
        if self.rp_cm is not None:
            p["rp_cm"] = float(self.rp_cm)
        if self.rstar_cm is not None:
            p["rstar_cm"] = float(self.rstar_cm)
        return p


# The planet and the run live in the spec (runs/<case>/<name>.yaml), not here: the
# framework only defines the schema.

# Fixed choices, not Config fields: no spec sets them.
GUILLOT_F = 0.25        # Guillot f: 1/4 = whole-planet average irradiation
# Floor for a denominator or log argument that may be exactly zero.
UNDERFLOW_DENOM = 1e-300
# The CUDA compute capability the `ffi` block-Thomas kernel runs on.
FFI_COMPUTE_CAPABILITY = "9.0"
# Baseline C/O of the W39b column, W39b.yaml C_H / O_H = 0.00295 / 0.00537;
# only the banner's C/O range uses it.
_BASELINE_C_TO_O = 0.549


# Parameter specification (the ordered, active parameter list + priors)
@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    prior_type: str   # {"uniform", "log10_uniform", "gaussian"}; gaussian: lo = mean, hi = sigma
    lo: float
    hi: float
    truth: float
    kind: str         # {"chem", "tp", "lnR0", "lnM", "cloud", "offset", "noise"} -- how the forward consumes it


def chem_extras(cfg: Config) -> Tuple[str, ...]:
    """The optional chemistry entries this Config carries, in the engine's
    layout order (vulcan_chem.CHEM_EXTRAS): s_o, then kzz_slope."""
    out = []
    if cfg.infer_s_o:
        out.append("s_o")
    if cfg.kzz_mode == "powerlaw":
        out.append("kzz_slope")
    return tuple(out)


def specs_from_config(cfg: Config, groups: Optional[List[str]] = None) -> List[ParamSpec]:
    """Build the ordered active parameter list. ``groups`` is the ordered instrument-group
    list from the observations (offsets are added for groups[1:] relative to groups[0])."""
    specs: List[ParamSpec] = []

    def add(name, label, lo, hi, truth, kind, prior_type="uniform"):
        if prior_type == "gaussian":
            if not (hi > 0.0):
                raise ValueError(f"gaussian prior for {name} needs sigma > 0, got {hi}")
        elif not (lo < hi):
            raise ValueError(f"prior bounds for {name}: lo={lo} !< hi={hi}")
        if prior_type == "log10_uniform" and (lo <= 0 or hi <= 0):
            raise ValueError(f"log10_uniform prior needs positive bounds for {name}")
        specs.append(ParamSpec(name, label, prior_type, float(lo), float(hi), float(truth), kind))

    # --- chemistry, always inferred: converged_y and the pipeline unpack
    # [lnZ, c_o, lnKzz, <extras...>, <tp...>] by position ---
    add("lnZ", r"$\ln Z$", *cfg.prior_lnZ, cfg.truth_lnZ, "chem")
    add("c_o", r"$\Delta\ln(\mathrm{C/O})$", *cfg.prior_c_o, cfg.truth_c_o, "chem")
    add("lnKzz", r"$\ln K_{zz}$", *cfg.prior_lnKzz, cfg.truth_lnKzz, "chem")
    for e in chem_extras(cfg):
        if e == "s_o":
            add("s_o", r"$\Delta\ln(\mathrm{S/O})$", *cfg.prior_s_o, cfg.truth_s_o, "chem")
        else:
            add("kzz_slope", r"$\alpha_{K}$", *cfg.prior_kzz_slope, cfg.truth_kzz_slope,
                "chem")

    # --- T-P (ExoJax Guillot) ---
    add("Tirr", r"$T_{\rm irr}$ (K)", *cfg.prior_Tirr, cfg.truth_Tirr, "tp")
    add("log10kappa", r"$\log_{10}\kappa_{\rm IR}$", *cfg.prior_log10kappa, cfg.truth_log10kappa, "tp")
    if cfg.tp_infer_gamma:
        add("log10gamma", r"$\log_{10}\gamma$", *cfg.prior_log10gamma, cfg.truth_log10gamma, "tp")
    if cfg.tp_infer_Tint:
        if cfg.tp_deep_param == "T1bar":
            add("T1bar", rf"$T_{{{cfg.tp_deep_p_bar:g}\,\rm bar}}$ (K)", *cfg.prior_T1bar,
                cfg.truth_T1bar, "tp")
        else:
            add("Tint", r"$T_{\rm int}$ (K)", *cfg.prior_Tint, cfg.truth_Tint, "tp")
    if cfg.tp_adiabat:
        add("log10P_rcb", r"$\log_{10}P_{\rm rcb}$", *cfg.prior_log10P_rcb,
            cfg.truth_log10P_rcb, "tp")

    # --- radius nuisance ---
    add("lnR0", r"$\ln R_0$", *cfg.prior_lnR0, cfg.truth_lnR0, "lnR0")
    # --- planet mass nuisance (RT-only, Gaussian) ---
    if cfg.infer_lnM:
        add("lnM", r"$\ln M$", 0.0, cfg.prior_lnM_sigma, cfg.truth_lnM, "lnM",
            prior_type="gaussian")

    # --- clouds (RT-only, like lnR0: no chemistry solve) ---
    if cfg.cloud_model == "deck":
        add("log10P_deck", r"$\log_{10}P_{\rm deck}$", *cfg.prior_log10P_deck,
            cfg.truth_log10P_deck, "cloud")
    else:
        add("log10kappa_cloud", r"$\log_{10}\kappa_{\rm cl}$", *cfg.prior_log10kappa_cloud,
            cfg.truth_log10kappa_cloud, "cloud")
        add("cloud_alpha", r"$\alpha_{\rm cl}$", *cfg.prior_cloud_alpha,
            cfg.truth_cloud_alpha, "cloud")

    # --- inter-instrument offsets (ppm), one per group beyond the reference ---
    if cfg.infer_offsets and groups is not None and len(groups) > 1:
        for g in groups[1:]:
            add(f"offset_{g}", rf"$\delta_{{{g}}}$ (ppm)", *cfg.prior_offset_ppm,
                cfg.truth_offset_ppm, "offset")

    # --- optional noise inflation: one b, or one per instrument group ---
    if cfg.infer_noise_inflation and cfg.noise_inflation_per_group:
        if not groups:
            raise ValueError("noise_inflation_per_group needs the observation groups: "
                             "call specs_from_config(cfg, groups=...)")
        for g in groups:
            tex = g.replace("_", r"\_")            # mathtext: a literal underscore
            add(f"noise_inflation_{g}", rf"$b_{{\rm {tex}}}$",
                *cfg.prior_noise_inflation, cfg.truth_noise_inflation, "noise",
                prior_type="log10_uniform")
    elif cfg.infer_noise_inflation:
        add("noise_inflation", r"$b$", *cfg.prior_noise_inflation, cfg.truth_noise_inflation,
            "noise", prior_type="log10_uniform")
    return specs


def hardware_profile() -> dict:
    """The hardware this process runs on, read from JAX and the OS: backend,
    device kind, host cores and RAM, and on a CUDA device its SM count,
    compute capability and memory. run_nautilus logs it and records it in
    config.json."""
    import jax

    dev = jax.devices()[0]
    prof = {"backend": jax.default_backend(), "device_kind": dev.device_kind,
            "n_devices": len(jax.devices()),
            # sched_getaffinity is Linux-only (absent on macOS)
            "host_cores": (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                           else os.cpu_count()),
            "host_ram_gib": round(os.sysconf("SC_PAGE_SIZE")
                                  * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
            "vulcan3_solver": os.environ.get("VULCAN3_SOLVER", "fast"),
            "vulcan3_chem_rhs": os.environ.get("VULCAN3_CHEM_RHS", ""),
            "xla_flags": os.environ.get("XLA_FLAGS", "")}
    if prof["backend"] in ("gpu", "cuda"):
        cc = str(getattr(dev, "compute_capability", ""))
        stats = dev.memory_stats() or {}
        prof.update(sm_count=getattr(dev, "core_count", None), compute_capability=cc,
                    device_mem_gib=round(stats.get("bytes_limit", 0) / 2**30, 1))
    return prof


def choose_solver() -> str:
    """Set VULCAN3_SOLVER for this process from the device, unless the user
    set it: `ffi` (the block-Thomas CUDA kernel) on compute capability 9.0,
    the only card it is validated on, and only when its CUDA library is
    built; `fast` everywhere else. Import-frozen by VULCAN-JAX, so this runs before
    the first vulcan3 import (the analysis tools call it before they rebuild a
    run's forward; a run's own solver is its spec's runtime block). Returns the choice and
    why, for the log."""
    import importlib.util
    import jax

    if "VULCAN3_SOLVER" in os.environ:
        return f"{os.environ['VULCAN3_SOLVER']} (set by the user)"
    step = sys.modules.get("vulcan3.jax_step")
    if step is not None:        # already frozen by an earlier import: report it
        os.environ["VULCAN3_SOLVER"] = step._SOLVER
        return f"{step._SOLVER} (vulcan3 was imported before the choice)"
    dev = jax.devices()[0]
    cc = str(getattr(dev, "compute_capability", ""))
    spec = importlib.util.find_spec("vulcan3")
    lib = Path(spec.submodule_search_locations[0]) / "csrc" / "libblock_thomas_cuda.so"
    if (jax.default_backend() in ("gpu", "cuda") and cc == FFI_COMPUTE_CAPABILITY
            and lib.is_file()):
        os.environ["VULCAN3_SOLVER"] = "ffi"
        return f"ffi (compute capability {cc}, {lib.name} built)"
    os.environ["VULCAN3_SOLVER"] = "fast"
    why = ("not a CUDA device" if jax.default_backend() not in ("gpu", "cuda") else
           f"compute capability {cc} is not {FFI_COMPUTE_CAPABILITY}"
           if cc != FFI_COMPUTE_CAPABILITY else
           f"{lib} is not built (python -c 'from vulcan3 import solver_fast; "
           "solver_fast.build(cuda=True)' on the GPU host)")
    return f"fast ({why})"


def validate_config(cfg: Config) -> None:
    if int(cfg.cold_refill_chunk) < 1:
        raise ValueError(f"cold_refill_chunk must be >= 1, got {cfg.cold_refill_chunk!r}")
    # The RT chunk size is a batch split where 0 means "one batch".
    if int(cfg.smc_rt_chunk) < 0:
        raise ValueError(f"smc_rt_chunk must be >= 0 (0 = no chunking), got {cfg.smc_rt_chunk!r}")
    if int(cfg.cold_lanes) < 1:
        raise ValueError("cold_lanes must be > 0 (the lane count of the chemistry "
                         f"queue every batch runs on), got {cfg.cold_lanes!r}")
    if not math.isfinite(cfg.walltime_seconds):
        raise ValueError("walltime_seconds must be finite (<= 0 means no limit)")
    # Every pipeline evaluates a likelihood, so condensation ON is refused here
    # unless allow_condense_inference=True (the opt-in).
    if (bool(cfg.cfg_overrides.get("use_condense", False))
            and not cfg.allow_condense_inference):
        raise ValueError(
            "use_condense=True is refused in a retrieval: no retrieval with "
            "condensation has been validated (certification and start dependence "
            "of condensing columns are untested). Run a condensation forward model "
            "through vulcan_forward.vulcan_chem.build_chem_model, or set "
            "allow_condense_inference=True after validating it on your case.")
    if cfg.noise_inflation_per_group and not cfg.infer_noise_inflation:
        raise ValueError("noise_inflation_per_group=True needs infer_noise_inflation=True "
                         "(it splits the inferred inflation by group; alone it would "
                         "silently infer none)")
    if cfg.tp_deep_param not in ("Tint", "T1bar"):
        raise ValueError(f"tp_deep_param={cfg.tp_deep_param!r}: expected 'Tint' or 'T1bar'")
    if cfg.tp_deep_param == "T1bar" and not cfg.tp_infer_Tint:
        raise ValueError("tp_deep_param='T1bar' needs tp_infer_Tint=True (it names the "
                         "retrieved deep parameter; with a fixed Tint it would be ignored)")
    if not (float(cfg.tp_deep_p_bar) > 0.0):
        raise ValueError(f"tp_deep_p_bar must be > 0 bar, got {cfg.tp_deep_p_bar!r}")
    if cfg.kzz_mode not in ("scale", "powerlaw"):
        raise ValueError(f"kzz_mode={cfg.kzz_mode!r}: expected 'scale' or 'powerlaw'")
    if cfg.cloud_model not in ("powerlaw", "deck"):
        raise ValueError(f"cloud_model={cfg.cloud_model!r}: expected 'powerlaw' or 'deck'")
    if cfg.infer_lnM and not (float(cfg.prior_lnM_sigma) > 0.0):
        raise ValueError(f"prior_lnM_sigma must be > 0 with infer_lnM, got "
                         f"{cfg.prior_lnM_sigma!r}")
    # Planet identity must be set by the spec.
    if not str(cfg.vulcan_cfg_name).strip():
        raise ValueError("vulcan_cfg_name is unset: the case must name its VULCAN "
                         "baseline config (e.g. 'W39b', loaded from vulcan3/configs/)")
    if cfg.rp_cm is None or cfg.rstar_cm is None:
        raise ValueError(f"planet radii unset (rp_cm={cfg.rp_cm}, rstar_cm={cfg.rstar_cm}): "
                         "the case must set both (cm)")


# Loud config banner (printed at the top of every run so nothing is a surprise)
def describe_config(cfg: Config, name: str = "") -> str:
    """A prominent, human-readable dump of the RESOLVED configuration (the spec over
    the defaults) -- forward-model fidelity, convergence criteria, T-P handling, data
    source, run settings, and the full parameter/prior table. Every entry point logs
    this so the exact numbers a run uses (band, count_max, priors, ...)
    are visible up front."""
    # Offsets are named per non-reference group, and the reference is the
    # wavelength-first group, not combo[0]; read the product CSVs (numpy-only) to
    # get the pipeline's order. Band-edge drops can still differ; the built
    # pipeline logs its resolved groups.
    groups = list(cfg.combo)
    if cfg.obs_dir and cfg.obs_products:
        from retrieval_framework import observations as OBS
        groups = list(OBS.load_real_observations(cfg)["groups"])
    specs = specs_from_config(cfg, groups=groups)
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
    lanes = f"{int(cfg.cold_lanes)} lanes, refill {int(cfg.cold_refill_chunk)}"

    lines = [
        "", bar,
        f"  RETRIEVAL CONFIG    spec={name or '?'}    {cfg.run_label or 'run'}"
        f"    out_dir={cfg.out_dir}",
        bar,
        rule("forward model"),
        f"    nz={cfg.nz}   band {wl_lo:.2f}-{wl_hi:.2f} um   art_nlayer={cfg.art_nlayer}",
        f"    opacity: {opa}",
        f"    molecules: {' '.join(cfg.molecules)}",
        f"    photo={'ON' if cfg.use_photo else 'OFF'}   rayleigh={'on' if cfg.use_rayleigh else 'off'}"
        f"   co_mode={cfg.co_mode}"
        f"   lane queue: {lanes}",
        rule("convergence  (VULCAN-master criteria; slope_cri/yconv_min/flux_cri inherit vulcan_cfg)"),
        f"    yconv_cri={cfg.yconv_cri:g}   count_max={cmax}   count_min={cmin}",
        f"    dt_max={dtmax} s   (physical step cap; master default 1e17 balloons dt -> "
        "high-Kzz non-convergence)",
        rule("T-P profile"),
        f"    model=guillot   deep={('free ' + cfg.tp_deep_param) if cfg.tp_infer_Tint else f'Tint {cfg.tp_Tint_K:g}K'}"
        f"   f={GUILLOT_F:g}   g={cfg.tp_gravity_cgs:g}cgs"
        f"   infer_gamma={'on' if cfg.tp_infer_gamma else 'off'}"
        + (f"   adiabat below a free P_rcb (nabla {cfg.tp_nabla_ad:g})" if cfg.tp_adiabat else ""),
        "    drawn RAW (no clip); profiles leaving the modelable T window on the RT or the "
        "chemistry grid are REJECTED + REDRAWN",
        rule("composition, Kzz, mass, clouds"),
        f"    sulfur={'free (s_o)' if cfg.infer_s_o else 'scales with lnZ'}"
        f"   kzz_mode={cfg.kzz_mode}"
        + (f" (ref {cfg.kzz_p_ref_bar:g} bar)" if cfg.kzz_mode == 'powerlaw' else "")
        + f"   mass={('N(0, ' + format(cfg.prior_lnM_sigma, 'g') + ') in lnM') if cfg.infer_lnM else 'fixed'}"
        f"   cloud_model={cfg.cloud_model}",
        rule("data"),
        f"    {data}   band {cfg.obs_wl_lo:g}-{cfg.obs_wl_hi:g} um   groups={list(cfg.combo)}",
        rule("run"),
        f"    rt_chunk={cfg.smc_rt_chunk}   walltime governor: "
        f"{cfg.walltime_seconds / 3600.0:.1f} h"
        + ("  (no limit)" if cfg.walltime_seconds <= 0 else ""),
        rule(f"parameters ({len(specs)})   [prior : truth]"),
    ]
    for s in specs:
        pt = {"log10_uniform": "log10U", "gaussian": "N"}.get(s.prior_type, "U")
        note = ""
        if s.name == "c_o":
            note = (f"  [C/O {math.exp(s.lo) * _BASELINE_C_TO_O:.2g}-"
                    f"{math.exp(s.hi) * _BASELINE_C_TO_O:.2g}]")
        elif s.name == "log10gamma":
            note = f"  [gamma {10 ** s.lo:.2g}-{10 ** s.hi:.2g}]"
        elif s.name == "lnKzz":
            note = f"  [Kzz x{math.exp(s.lo):.2g}-x{math.exp(s.hi):.2g}]"
        elif s.name == "s_o":
            note = f"  [S/O x{math.exp(s.lo):.2g}-x{math.exp(s.hi):.2g}]"
        elif s.name == "lnM":
            note = "  [mean, sigma]"
        tr = "n/a" if not math.isfinite(s.truth) else f"{s.truth:g}"
        lines.append(f"    {s.name:<17s} {pt}({s.lo:g}, {s.hi:g}){note:<26s}  truth={tr}")
    lines += [bar, ""]
    return "\n".join(lines)
