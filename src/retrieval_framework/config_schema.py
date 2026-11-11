"""The static configuration of one retrieval: the frozen ``Config``, the parameter list and the run banner.

The planet and the run live in the spec, ``runs/<case>/<name>.yaml`` (``run_setup.load_spec``). The parameter vector
is

    theta = [lnZ, dln(C/O), lnKzz, [s_o], [kzz_slope], <T-P>, lnR0, [lnM], <cloud>, offset_g ..., <noise inflation>]

The chemistry and T-P entries re-converge VULCAN, the expensive part of a likelihood call; the rest enter only the RT or
apply after it. Imports no jax. docs/method.md describes the parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003  # run_setup resolves the Config annotations at run time
from typing import Any, NamedTuple, TypeVar

from vulcan_forward.constants import ART_PTOP_BAR, check_band, check_core_molecules

from retrieval_framework import constants

__all__ = [
    "Config",
    "ParamSpec",
    "Parameter",
    "active_parameters",
    "chem_extras",
    "describe_config",
    "export_network",
    "need",
    "specs_from_config",
    "validate_config",
]


_T = TypeVar("_T")


def need(value: _T | None, key: str) -> _T:
    """Return a Config value a case states, refusing it unset.

    ``validate_config`` refuses every unset one first, so None here is a caller that skipped it.

    Parameters
    ----------
    value : object or None
        The Config field's value.
    key : str
        Its name, for the message.

    Returns
    -------
    object
        ``value``.

    Raises
    ------
    ValueError
        If ``value`` is None.
    """
    if value is None:
        raise ValueError(
            f"{key} is unset: run_setup.load_spec and validate_config refuse such a Config; validate it first"
        )
    return value


@dataclass(frozen=True)
class Config:
    """Everything static for one retrieval run; ``run_setup.load_spec`` builds one from the spec's ``config`` block.

    A key a case states (``case_keys.required_keys``) has no default: None, refused unset by ``validate_config``.
    """

    # ---- I/O and reproducibility ----
    out_dir: Path | None = None  # set by load_spec, never by a spec key
    run_label: str = ""  # short human label for plot titles (e.g. "WASP-39b")
    seed: int | None = None  # nautilus and the synthetic injection

    # ---- forward-model fidelity: the engine's profile dict ----
    nz: int = 62  # VULCAN vertical layers (62 -> ~1/3 the nz=188 cost; 6.3 layers/decade over 1e-9..7.6 bar)
    use_photo: bool = True  # photochemistry (SO2 is photochemical)
    # VULCAN master's W39b convergence (1e-3 costs thousands of steps for no gradient gain);
    # slope_cri, yconv_min and flux_cri come from the vulcan_cfg_name config.
    yconv_cri: float = 0.01
    # The RT's absorbers: the engine's core ones its band needs and the minor species its planet needs.
    molecules: tuple[str, ...] | None = None
    # Core absorbers the case leaves out of the RT on purpose; the engine refuses a band that needs one otherwise
    # (vulcan_forward.constants.CORE_MOLECULE_BANDS_UM).
    omit_core_molecules: tuple[str, ...] = ()
    nu_min: float | None = None  # cm^-1
    nu_max: float | None = None  # cm^-1
    art_nlayer: int = 67
    art_ptop_bar: float = ART_PTOP_BAR  # model top: chemistry AND RT end here (engine rule)
    # RT grid bottom, bar; no default: it must sit above the case's chemistry bottom (the engine refuses one below).
    art_pbtm_bar: float | None = None
    # exojax ArtTransPure's chord quadrature: "simpson" or "trapezoid".
    rt_integration: str = "simpson"
    use_rayleigh: bool = True  # H2/He Rayleigh scattering (ExoJax; zero free params)
    co_mode: str = "fixed_O"  # how the C/O guess is built (the engine repairs it exactly)
    count_min: int | None = None
    count_max: int | None = None
    # Largest integrator step, s; None is the VULCAN default. A cap stops large-dt Ros2 oscillations
    # without touching the convergence criteria.
    dt_max: float | None = None
    cfg_overrides: dict[str, Any] = field(default_factory=dict)

    # ---- planet identity: validate_config refuses an unset value ----
    # The VULCAN baseline config, a name or path vulcan3.load_config takes (e.g. "W39b").
    vulcan_cfg_name: str | None = None
    # The kinetics network, a vulcan3 path (thermo/...), and its elements; no default: the spec names both
    # (validate_config). Import-frozen: export_network hands them to the engine before vulcan3 is imported, and the
    # engine refuses a later call with another pair.
    network: str | None = None
    atom_list: tuple[str, ...] | None = None
    # Planet/stellar radii for the RT depth normalization (cm).
    # tp_gravity_cgs below doubles as the RT's gravity at rp_cm; the planet mass
    # GM = g rp_cm^2 stays fixed when lnR0 scales the radius.
    rp_cm: float | None = None
    rstar_cm: float | None = None
    # Pressure (bar) where rp_cm and tp_gravity_cgs apply. lnR0 is the log radius change here, so moving it needs a
    # matching prior_lnR0 (the W39b case anchors at 1 mbar).
    p_ref_bar: float | None = None

    # ---- observed spectrum source ----
    # obs_products maps a group label to its product CSVs in obs_dir; combo selects the groups to
    # fit. The offset reference is the group of the shortest-wavelength kept bin, not combo[0];
    # G groups give G-1 offsets. No obs_dir or products: a synthetic bin grid.
    obs_dir: Path | None = None
    obs_products: dict[str, tuple[str, ...]] = field(default_factory=dict)
    combo: tuple[str, ...] | None = None
    obs_wl_lo: float | None = None  # um: the fit window's edges
    obs_wl_hi: float | None = None
    # Drop product bins finer than this R before they reach the binning operator.
    # Published R100 products carry truncation remnants where a masked region cuts a
    # bin short: they are a few times narrower than the nominal grid, carry a much
    # larger sigma, and come close to the model R, where the binning operator has no
    # LSF (observations._refuse_unresolved_products). 0 disables the cut.
    obs_max_bin_R: float = 0.0

    # ---- T-P profile: exojax's atmprof_Guillot(P, g, kappa, gamma, Tint, Tirr, f), f = constants.GUILLOT_F ----
    tp_gravity_cgs: float | None = None  # cm/s^2, at rp_cm
    tp_Tint_K: float | None = None  # interior temperature, K, when tp_infer_Tint=False
    tp_infer_gamma: bool = True  # retrieve log10(gamma); False -> fixed
    tp_gamma_fixed: float | None = None  # used only when tp_infer_gamma=False
    # Retrieve the deep T-P parameter, which sets the CO/CH4/NH3 quench: Tint itself
    # (tp_deep_param="Tint") or the temperature at tp_deep_p_bar ("T1bar"; docs/method.md).
    tp_infer_Tint: bool = False
    tp_deep_param: str = "Tint"  # {"Tint", "T1bar"}, with tp_infer_Tint
    tp_deep_p_bar: float | None = None  # bar: the level T1bar is quoted at, and the report's deep temperature
    # A sensitivity model: an adiabat below a free boundary pressure (one more T-P entry,
    # log10P_rcb), matched to the Guillot profile there; with constant kappa the join is imposed.
    tp_adiabat: bool = False
    tp_nabla_ad: float | None = None  # dlnT/dlnP of the adiabat

    # ---- clouds: "powerlaw" is ExoJax's kappa(nu) = kappac0 (nu / CLOUD_NUC0)^alphac, cm^2 per gram
    #      of atmosphere, uniformly mixed; "deck" is an opaque grey deck below a free pressure; "none" is a clear
    #      atmosphere with no cloud term ----
    cloud_model: str | None = None
    prior_log10kappa_cloud: tuple[float, float] | None = None  # cm^2/g at 3.5 um
    truth_log10kappa_cloud: float | None = None
    prior_cloud_alpha: tuple[float, float] | None = None  # 0=gray, 4~Rayleigh haze
    truth_cloud_alpha: float | None = None
    prior_log10P_deck: tuple[float, float] | None = None  # bar
    truth_log10P_deck: float | None = None

    # ---- inference toggles: the chemistry block is always inferred ----
    infer_offsets: bool = True  # one flat depth offset per instrument group beyond the reference

    # ---- priors, bounded and uniform unless noted; truth_* is the synthetic-injection value ----
    # lnZ is relative to the vulcan cfg's baseline composition (its <X>_H).
    prior_lnZ: tuple[float, float] | None = None
    truth_lnZ: float | None = None
    # dln(C/O) about the fixed-O baseline. Upper bound: the fixed-O knob's O-only compensation b_z must stay positive
    # (the bound is printed at build; retrieval_forward raises if the prior reaches it).
    prior_c_o: tuple[float, float] | None = None
    truth_c_o: float | None = None
    prior_lnKzz: tuple[float, float] | None = None  # ln of the eddy-diffusion multiplier
    truth_lnKzz: float | None = None
    # Sulfur freed from the metallicity scaling: s_o = ln of the S/O multiple of
    # the baseline (S/H = e^(lnZ + s_o) times the baseline's), a fourth chemistry
    # entry after lnKzz. Off, sulfur scales with lnZ like O and N.
    infer_s_o: bool = False
    prior_s_o: tuple[float, float] | None = None
    truth_s_o: float | None = None
    # Kzz shape: "scale" multiplies the vulcan cfg's profile by e^lnKzz;
    # "powerlaw" is Kzz0(kzz_p_ref_bar) e^lnKzz (P / P_ref)^-kzz_slope, with the
    # slope a chemistry entry after s_o (engine kzz_mode).
    kzz_mode: str = "scale"
    kzz_p_ref_bar: float | None = None  # bar
    prior_kzz_slope: tuple[float, float] | None = None
    truth_kzz_slope: float | None = None
    # The vulcan cfg's baseline metallicity in solar units (its <X>_H over the
    # Lodders 2020 photospheric values): the report quotes e^lnZ times this.
    baseline_metallicity_solar: float | None = None
    # Guillot irradiation temperature, sqrt(2) Teq for full redistribution
    # (with f=0.25 the skin temperature is ~0.6*Tirr: 0.59 at gamma << 1, 0.64 at 0.4).
    prior_Tirr: tuple[float, float] | None = None  # K
    truth_Tirr: float | None = None
    prior_log10kappa: tuple[float, float] | None = None  # IR opacity, cm^2/g (log10)
    truth_log10kappa: float | None = None
    prior_log10gamma: tuple[float, float] | None = None  # kappa_v/kappa_th (log10)
    truth_log10gamma: float | None = None
    prior_Tint: tuple[float, float] | None = None  # K
    truth_Tint: float | None = None
    # T at tp_deep_p_bar (K), used with tp_deep_param="T1bar": the irradiation term and the [220, 2980] K T-P window
    # shape the accepted prior (report it).
    prior_T1bar: tuple[float, float] | None = None
    truth_T1bar: float | None = None
    prior_log10P_rcb: tuple[float, float] | None = None  # bar
    truth_log10P_rcb: float | None = None
    prior_lnR0: tuple[float, float] | None = None  # reference-radius log scaling
    truth_lnR0: float | None = None
    # Planet mass as a nuisance: lnM = ln(M / M0) ~ N(0, prior_lnM_sigma), applied
    # in the RT at the fixed reference radius (GM scales e^lnM; engine
    # transmission_depth_r lnM). The chemistry's hydrostatic column and the
    # Guillot tau keep M0. Off, the mass is fixed at M0 = g rp^2 / G.
    infer_lnM: bool = False
    prior_lnM_sigma: float | None = None  # the mass error, in ln M
    truth_lnM: float | None = None
    prior_offset_ppm: tuple[float, float] | None = None  # per-group depth offset, ppm
    truth_offset_ppm: float | None = None

    # A free multiplicative noise-inflation term (Line 2015 style) guards against
    # underestimated JWST error bars; b multiplies every sigma.
    infer_noise_inflation: bool = False
    prior_noise_inflation: tuple[float, float] | None = None  # log10-uniform
    truth_noise_inflation: float | None = None
    # One inflation b_g per observation group, the reference group included, each with the
    # prior above. Needs infer_noise_inflation=True.
    noise_inflation_per_group: bool = False
    # Integrate the offsets and the noise inflation(s) out of the likelihood at the same priors
    # (nuisance.make_nuisance_marginal); their posteriors are drawn after the run. Needs
    # infer_offsets or infer_noise_inflation.
    integrate_nuisance: bool = False

    # ---- data source ----
    # False : fit the real observed spectrum (obs_dir/obs_products above).
    # True  : inject the truth_* parameters, add Gaussian noise at the real per-bin
    #         sigma (or the synthetic grid's), and fit that (recovery self-test).
    generate_synthetic_data: bool = False

    # ---- inference ----
    # Opt in to a retrieval with condensation; none has been validated. A condensation forward
    # model goes through vulcan_forward.vulcan_chem.build_chem_model instead.
    allow_condense_inference: bool = False
    # Rows per lax.map chunk through the ExoJAX RT. 0 = one all-row batch.
    smc_rt_chunk: int = 16
    # Lanes of the chemistry queue, the number that runs: a lane that certifies is refilled with the next draw in the
    # same while loop, so wall time follows total work / lanes.
    cold_lanes: int | None = None
    # Lanes refilled per pass, capped at cold_lanes; more amortizes the refill.
    cold_refill_chunk: int = 8

    # ---- report only ----
    # A Kzz ceiling, a multiple of the case's base Kzz: the report also quotes the posterior truncated there. None: no
    # truncated block.
    report_kzz_ceiling_x: float | None = None

    # ---- walltime governor ----
    # Soft wall-clock budget, s (0: none); run_nautilus stops at it with its checkpoint written and
    # exits 0, and --resume continues.
    walltime_seconds: float = 0.0

    def profile(self) -> dict[str, Any]:
        """Return the settings dict the engine's model builders take.

        Returns
        -------
        dict
            Keys in ``vulcan_forward.constants.PROFILE_KEYS``; optional fields only when set.
        """
        p: dict[str, Any] = {
            "use_photo": bool(self.use_photo),
            "nz": int(self.nz),
            "yconv_cri": float(self.yconv_cri),
            "molecules": list(need(self.molecules, "molecules")),
            "omit_core_molecules": list(self.omit_core_molecules),
            "nu_min": float(need(self.nu_min, "nu_min")),
            "nu_max": float(need(self.nu_max, "nu_max")),
            "art_nlayer": int(self.art_nlayer),
            "art_ptop_bar": float(self.art_ptop_bar),
            "rt_integration": str(self.rt_integration),
            "use_rayleigh": bool(self.use_rayleigh),
            "co_mode": str(self.co_mode),
            "kzz_mode": str(self.kzz_mode),
            "cfg_overrides": dict(self.cfg_overrides),
            "gs_cgs": float(need(self.tp_gravity_cgs, "tp_gravity_cgs")),  # RT g at rp_cm = the T-P gravity
            "network": str(self.network),
            "atom_list": list(need(self.atom_list, "atom_list")),
        }
        if self.kzz_mode == "powerlaw":  # the engine refuses the key under any other mode
            p["kzz_p_ref_bar"] = float(need(self.kzz_p_ref_bar, "kzz_p_ref_bar"))
        # is not None: an explicit 0 must reach the engine.
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
        if self.p_ref_bar is not None:
            p["p_ref_bar"] = float(self.p_ref_bar)  # where rp_cm/gs_cgs apply
        if self.art_pbtm_bar is not None:
            p["art_pbtm_bar"] = float(self.art_pbtm_bar)
        return p


@dataclass(frozen=True)
class ParamSpec:
    """One sampled parameter and its prior.

    Attributes
    ----------
    name : str
        The parameter's name.
    label : str
        Its mathtext label.
    prior_type : str
        "uniform", "log10_uniform" or "gaussian".
    lo : float
        The lower bound; a gaussian's mean.
    hi : float
        The upper bound; a gaussian's sigma.
    truth : float
        The synthetic-injection value.
    kind : str
        How the forward consumes it: "chem", "tp", "lnR0", "lnM", "cloud", "offset" or "noise".
    """

    name: str
    label: str
    prior_type: str
    lo: float
    hi: float
    truth: float
    kind: str


def chem_extras(cfg: Config) -> tuple[str, ...]:
    """Return the optional chemistry entries this Config carries, in the engine's layout order.

    Parameters
    ----------
    cfg : Config
        The run's configuration.

    Returns
    -------
    tuple of str
        A subset of ``vulcan_chem.CHEM_EXTRAS``.
    """
    out = []
    if cfg.infer_s_o:
        out.append("s_o")
    if cfg.kzz_mode == "powerlaw":
        out.append("kzz_slope")
    return tuple(out)


class Parameter(NamedTuple):
    """One parameter a Config samples, and the Config keys of its prior and its synthetic-injection value."""

    name: str
    label: str
    kind: str
    prior_key: str
    truth_key: str
    prior_type: str = "uniform"


def active_parameters(cfg: Config, groups: list[str] | None = None) -> list[Parameter]:  # noqa: C901  # one branch per optional parameter
    """Return the parameters this Config samples, in theta order: the one place a mode decides what is inferred.

    Parameters
    ----------
    cfg : Config
        The run's configuration.
    groups : list of str, optional
        The ordered instrument groups; offsets go on ``groups[1:]``, relative to ``groups[0]``.

    Returns
    -------
    list of Parameter
        The parameters, in theta order.

    Raises
    ------
    ValueError
        If per-group noise is asked for without the groups.
    """

    def p(name: str, label: str, kind: str, prior_type: str = "uniform") -> Parameter:
        return Parameter(name, label, kind, f"prior_{name}", f"truth_{name}", prior_type)

    # Chemistry: always inferred, unpacked by position.
    out = [
        p("lnZ", r"$\ln Z$", "chem"),
        p("c_o", r"$\Delta\ln(\mathrm{C/O})$", "chem"),
        p("lnKzz", r"$\ln K_{zz}$", "chem"),
    ]
    for e in chem_extras(cfg):
        if e == "s_o":
            out.append(p("s_o", r"$\Delta\ln(\mathrm{S/O})$", "chem"))
        else:
            out.append(p("kzz_slope", r"$\alpha_{K}$", "chem"))

    out += [p("Tirr", r"$T_{\rm irr}$ (K)", "tp"), p("log10kappa", r"$\log_{10}\kappa_{\rm IR}$", "tp")]
    if cfg.tp_infer_gamma:
        out.append(p("log10gamma", r"$\log_{10}\gamma$", "tp"))
    if cfg.tp_infer_Tint:
        if cfg.tp_deep_param == "T1bar":
            level = "?" if cfg.tp_deep_p_bar is None else format(cfg.tp_deep_p_bar, "g")
            out.append(p("T1bar", rf"$T_{{{level}\,\rm bar}}$ (K)", "tp"))
        else:
            out.append(p("Tint", r"$T_{\rm int}$ (K)", "tp"))
    if cfg.tp_adiabat:
        out.append(p("log10P_rcb", r"$\log_{10}P_{\rm rcb}$", "tp"))

    out.append(p("lnR0", r"$\ln R_0$", "lnR0"))
    if cfg.infer_lnM:
        out.append(Parameter("lnM", r"$\ln M$", "lnM", "prior_lnM_sigma", "truth_lnM", "gaussian"))

    if cfg.cloud_model == "deck":
        out.append(p("log10P_deck", r"$\log_{10}P_{\rm deck}$", "cloud"))
    elif cfg.cloud_model == "powerlaw":
        out += [
            p("log10kappa_cloud", r"$\log_{10}\kappa_{\rm cl}$", "cloud"),
            p("cloud_alpha", r"$\alpha_{\rm cl}$", "cloud"),
        ]

    # the integrated nuisances are not parameters (nuisance.make_nuisance_marginal)
    if cfg.integrate_nuisance:
        return out

    if cfg.infer_offsets and groups is not None and len(groups) > 1:
        for g in groups[1:]:
            out.append(
                Parameter(f"offset_{g}", rf"$\delta_{{{g}}}$ (ppm)", "offset", "prior_offset_ppm", "truth_offset_ppm")
            )

    noise = ("prior_noise_inflation", "truth_noise_inflation", "log10_uniform")
    if cfg.infer_noise_inflation and cfg.noise_inflation_per_group:
        if not groups:
            raise ValueError(
                "noise_inflation_per_group needs the observation groups: call specs_from_config(cfg, groups=...)"
            )
        for g in groups:
            tex = g.replace("_", r"\_")  # mathtext: a literal underscore
            out.append(Parameter(f"noise_inflation_{g}", rf"$b_{{\rm {tex}}}$", "noise", *noise))
    elif cfg.infer_noise_inflation:
        out.append(Parameter("noise_inflation", r"$b$", "noise", *noise))
    return out


def specs_from_config(cfg: Config, groups: list[str] | None = None) -> list[ParamSpec]:
    """Build the ordered active parameter list with its priors.

    Parameters
    ----------
    cfg : Config
        The run's configuration.
    groups : list of str, optional
        The ordered instrument groups; offsets go on ``groups[1:]``, relative to ``groups[0]``.

    Returns
    -------
    list of ParamSpec
        The parameters, in theta order; a truth the Config leaves unset is NaN.

    Raises
    ------
    ValueError
        On a prior that is unset or has bad bounds.
    """
    specs = []
    for par in active_parameters(cfg, groups):
        prior = getattr(cfg, par.prior_key)
        if prior is None:
            raise ValueError(f"{par.prior_key} is unset: the {par.name} parameter needs its prior")
        lo, hi = (0.0, prior) if par.prior_type == "gaussian" else prior
        if par.prior_type == "gaussian":
            if not (hi > 0.0):
                raise ValueError(f"gaussian prior for {par.name} needs sigma > 0, got {hi}")
        elif not (lo < hi):
            raise ValueError(f"prior bounds for {par.name}: lo={lo} !< hi={hi}")
        if par.prior_type == "log10_uniform" and (lo <= 0 or hi <= 0):
            raise ValueError(f"log10_uniform prior needs positive bounds for {par.name}")
        truth = getattr(cfg, par.truth_key)
        specs.append(
            ParamSpec(
                par.name,
                par.label,
                par.prior_type,
                float(lo),
                float(hi),
                math.nan if truth is None else float(truth),
                par.kind,
            )
        )
    return specs


def export_network(cfg: Config) -> None:
    """Hand ``cfg.network`` and ``cfg.atom_list`` to the engine, which freezes both at vulcan3's first import.

    Imports the engine's ``vulcan_chem``, and with it jax: call it after ``run_setup.export_runtime``, on a Config
    ``validate_config`` accepted. The engine's ``select_network`` raises ``RuntimeError`` when the environment, or a
    vulcan3 already imported, holds another pair.

    Parameters
    ----------
    cfg : Config
        The run's configuration.
    """
    from vulcan_forward.vulcan_chem import select_network

    select_network(cfg.network, cfg.atom_list)


def validate_config(cfg: Config) -> None:  # noqa: C901  # one refusal per knob
    """Refuse a configuration that is inconsistent or leaves the planet unset.

    Parameters
    ----------
    cfg : Config
        The run's configuration.

    Raises
    ------
    ValueError
        Naming the setting and what it needs.
    """
    from retrieval_framework import case_keys, provenance

    unset = sorted(k for k in case_keys.required_keys(cfg) if getattr(cfg, k) in (None, "", ()))
    if unset:
        raise ValueError(
            f"{unset} unset: each belongs to this case, so its spec (or its base) states it; a schema default would be "
            "another case's"
        )
    if int(cfg.cold_refill_chunk) < 1:
        raise ValueError(f"cold_refill_chunk must be >= 1, got {cfg.cold_refill_chunk!r}")
    if int(cfg.smc_rt_chunk) < 0:
        raise ValueError(f"smc_rt_chunk must be >= 0 (0 = no chunking), got {cfg.smc_rt_chunk!r}")
    if int(need(cfg.cold_lanes, "cold_lanes")) < 1:
        raise ValueError(
            "cold_lanes must be > 0 (the lane count of the chemistry "
            f"queue every batch runs on), got {cfg.cold_lanes!r}"
        )
    if not math.isfinite(cfg.walltime_seconds):
        raise ValueError("walltime_seconds must be finite (<= 0 means no limit)")
    if bool(cfg.cfg_overrides.get("use_condense", False)) and not cfg.allow_condense_inference:
        raise ValueError(
            "use_condense=True is refused in a retrieval: no retrieval with "
            "condensation has been validated (certification and start dependence "
            "of condensing columns are untested). Run a condensation forward model "
            "through vulcan_forward.vulcan_chem.build_chem_model, or set "
            "allow_condense_inference=True after validating it on your case."
        )
    if cfg.integrate_nuisance:
        if not (cfg.infer_offsets or cfg.infer_noise_inflation):
            raise ValueError(
                "integrate_nuisance=True needs infer_offsets or infer_noise_inflation: there is nothing to integrate"
            )
        off, noise = cfg.prior_offset_ppm, cfg.prior_noise_inflation
        if cfg.infer_offsets and not need(off, "prior_offset_ppm")[0] < need(off, "prior_offset_ppm")[1]:
            raise ValueError(f"prior_offset_ppm needs lo < hi, got {off}")
        if (
            cfg.infer_noise_inflation
            and not 0 < need(noise, "prior_noise_inflation")[0] < need(noise, "prior_noise_inflation")[1]
        ):
            raise ValueError(f"prior_noise_inflation needs 0 < lo < hi, got {noise}")
    if cfg.tp_deep_param not in ("Tint", "T1bar"):
        raise ValueError(f"tp_deep_param={cfg.tp_deep_param!r}: expected 'Tint' or 'T1bar'")
    if cfg.kzz_mode not in ("scale", "powerlaw"):
        raise ValueError(f"kzz_mode={cfg.kzz_mode!r}: expected 'scale' or 'powerlaw'")
    if cfg.cloud_model not in ("powerlaw", "deck", "none"):
        raise ValueError(f"cloud_model={cfg.cloud_model!r}: expected 'powerlaw', 'deck' or 'none'")
    if not cfg.use_rayleigh:
        raise ValueError(
            "use_rayleigh=False is refused: H2 and He Rayleigh scattering is in every retrieval (oracle.md, the "
            "standard opacity terms). Drop the key."
        )
    if bool(cfg.obs_dir) != bool(cfg.obs_products):
        raise ValueError(
            f"obs_dir={cfg.obs_dir!r}, obs_products={cfg.obs_products!r}: a data source names both, a synthetic grid "
            "neither"
        )
    if cfg.generate_synthetic_data and not (
        math.isfinite(need(cfg.truth_noise_inflation, "truth_noise_inflation"))
        and need(cfg.truth_noise_inflation, "truth_noise_inflation") > 0.0
    ):
        raise ValueError(
            f"truth_noise_inflation must be finite and > 0 (it scales the noise), got {cfg.truth_noise_inflation}"
        )
    if cfg.report_kzz_ceiling_x is not None and not float(cfg.report_kzz_ceiling_x) > 0.0:
        raise ValueError(
            f"report_kzz_ceiling_x must be > 0 (a multiple of the base Kzz), got {cfg.report_kzz_ceiling_x}"
        )
    if cfg.infer_lnM and not (float(need(cfg.prior_lnM_sigma, "prior_lnM_sigma")) > 0.0):
        raise ValueError(f"prior_lnM_sigma must be > 0 with infer_lnM, got {cfg.prior_lnM_sigma!r}")
    if cfg.rt_integration not in ("simpson", "trapezoid"):
        raise ValueError(f"rt_integration={cfg.rt_integration!r}: expected 'simpson' or 'trapezoid'")
    positive = {
        "rp_cm": cfg.rp_cm,
        "rstar_cm": cfg.rstar_cm,
        "tp_gravity_cgs": cfg.tp_gravity_cgs,
        "p_ref_bar": cfg.p_ref_bar,
        "art_pbtm_bar": cfg.art_pbtm_bar,
        "tp_deep_p_bar": cfg.tp_deep_p_bar,
    }
    bad = {k: v for k, v in positive.items() if not (math.isfinite(float(need(v, k))) and float(need(v, k)) > 0.0)}
    if bad:
        raise ValueError(f"{bad} must be finite and positive: a radius, a gravity or a pressure cannot be otherwise")
    # The engine's own refusals, raised here so a spec fails at load and not after the opacity build; the core
    # absorbers the band wants and the case leaves unnamed are logged by run_setup.load_spec.
    check_band(float(need(cfg.nu_min, "nu_min")), float(need(cfg.nu_max, "nu_max")))
    check_core_molecules(
        list(need(cfg.molecules, "molecules")),
        list(cfg.omit_core_molecules),
        float(need(cfg.nu_min, "nu_min")),
        float(need(cfg.nu_max, "nu_max")),
    )
    # The report quotes C/O and metallicity against the case file's start composition: reading it refuses a case whose
    # start is not C_H and O_H (provenance.baseline_c_to_o shares that reader).
    z = provenance.baseline_metallicity_solar(need(cfg.vulcan_cfg_name, "vulcan_cfg_name"), cfg.cfg_overrides)
    if (
        abs(float(need(cfg.baseline_metallicity_solar, "baseline_metallicity_solar")) / z - 1.0)
        > constants.BASELINE_METALLICITY_RTOL
    ):
        raise ValueError(
            f"baseline_metallicity_solar={cfg.baseline_metallicity_solar} disagrees with the case file's O/H, "
            f"{z:.4g} x solar (Lodders 2020 O), beyond {constants.BASELINE_METALLICITY_RTOL:.0%}: state the file's "
            "value"
        )


def describe_config(cfg: Config, name: str = "") -> str:
    """Return the resolved configuration as text: the banner every entry point logs.

    Parameters
    ----------
    cfg : Config
        The run's configuration.
    name : str, optional
        The spec's name.

    Returns
    -------
    str
        Forward-model fidelity, convergence criteria, T-P handling, data source, run settings and the parameter
        table.
    """
    # The offset reference is the wavelength-first group, not combo[0], so the group order comes from
    # the product CSVs; band-edge drops can still differ, and the built pipeline logs its groups.
    groups = ["SYNTH"]  # the synthetic grid's one group
    if cfg.obs_dir and cfg.obs_products:
        from retrieval_framework import observations as OBS

        groups = list(OBS.load_real_observations(cfg)["groups"])
    specs = specs_from_config(cfg, groups=groups)
    from retrieval_framework.provenance import baseline_c_to_o

    c_to_o = baseline_c_to_o(need(cfg.vulcan_cfg_name, "vulcan_cfg_name"), cfg.cfg_overrides)
    W = 84
    bar = "=" * W

    def rule(title: str = "") -> str:
        """Return one section rule of the banner."""
        return f"  --- {title} " + "-" * max(0, W - 8 - len(title)) if title else "  " + "-" * (W - 2)

    opa = "correlated-k (ExoMolOP tables, R=1000 bands, H2/He broadening)"
    wl_lo, wl_hi = 1e4 / float(need(cfg.nu_max, "nu_max")), 1e4 / float(need(cfg.nu_min, "nu_min"))
    cmax = "(vulcan_cfg default)" if cfg.count_max is None else str(int(cfg.count_max))
    cmin = "(vulcan_cfg default)" if cfg.count_min is None else str(int(cfg.count_min))
    dtmax = "(master default 1e17)" if cfg.dt_max is None else f"{cfg.dt_max:g}"
    data = "SYNTHETIC (inject-and-recover at truth_*)" if cfg.generate_synthetic_data else "REAL observed spectrum"
    lanes = f"{need(cfg.cold_lanes, 'cold_lanes')} lanes, refill {int(cfg.cold_refill_chunk)}"

    lines = [
        "",
        bar,
        f"  RETRIEVAL CONFIG    spec={name or '?'}    {cfg.run_label or 'run'}    out_dir={cfg.out_dir}",
        bar,
        rule("forward model"),
        f"    nz={cfg.nz}   band {wl_lo:.2f}-{wl_hi:.2f} um   art_nlayer={cfg.art_nlayer}",
        f"    network: {cfg.network or '(UNSET)'}   baseline: {cfg.vulcan_cfg_name}",
        f"    opacity: {opa}",
        f"    molecules: {' '.join(need(cfg.molecules, 'molecules'))}",
        (
            f"    photo={'ON' if cfg.use_photo else 'OFF'}   rayleigh={'on' if cfg.use_rayleigh else 'off'}"
            f"   co_mode={cfg.co_mode}"
            f"   lane queue: {lanes}"
        ),
        rule("convergence  (VULCAN-master criteria; slope_cri/yconv_min/flux_cri inherit vulcan_cfg)"),
        f"    yconv_cri={cfg.yconv_cri:g}   count_max={cmax}   count_min={cmin}",
        f"    dt_max={dtmax} s   (physical step cap; master default 1e17 balloons dt -> high-Kzz non-convergence)",
        rule("T-P profile"),
        f"    model=guillot   deep={('free ' + cfg.tp_deep_param) if cfg.tp_infer_Tint else f'Tint {cfg.tp_Tint_K:g}K'}"
        f"   f={constants.GUILLOT_F:g}   g={cfg.tp_gravity_cgs:g}cgs"
        f"   infer_gamma={'on' if cfg.tp_infer_gamma else 'off'}"
        + (f"   adiabat below a free P_rcb (nabla {cfg.tp_nabla_ad:g})" if cfg.tp_adiabat else ""),
        (
            "    drawn RAW (no clip); profiles leaving the modelable T window on the RT or the "
            "chemistry grid are REJECTED + REDRAWN"
        ),
        rule("composition, Kzz, mass, clouds"),
        f"    sulfur={'free (s_o)' if cfg.infer_s_o else 'scales with lnZ'}"
        f"   kzz_mode={cfg.kzz_mode}"
        + (f" (ref {cfg.kzz_p_ref_bar:g} bar)" if cfg.kzz_mode == "powerlaw" else "")
        + f"   mass={('N(0, ' + format(cfg.prior_lnM_sigma, 'g') + ') in lnM') if cfg.infer_lnM else 'fixed'}"
        f"   cloud_model={cfg.cloud_model}",
        rule("data"),
        f"    {data}   "
        + (f"band {cfg.obs_wl_lo:g}-{cfg.obs_wl_hi:g} um   groups={groups}" if cfg.obs_dir else "synthetic bin grid"),
        rule("run"),
        f"    rt_chunk={cfg.smc_rt_chunk}   walltime governor: "
        f"{cfg.walltime_seconds / 3600.0:.1f} h" + ("  (no limit)" if cfg.walltime_seconds <= 0 else ""),
        rule(f"parameters ({len(specs)})   [prior : truth]"),
    ]
    for s in specs:
        pt = {"log10_uniform": "log10U", "gaussian": "N"}.get(s.prior_type, "U")
        note = ""
        if s.name == "c_o":
            note = f"  [C/O {math.exp(s.lo) * c_to_o:.2g}-{math.exp(s.hi) * c_to_o:.2g}]"
        elif s.name == "log10gamma":
            note = f"  [gamma {10**s.lo:.2g}-{10**s.hi:.2g}]"
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
