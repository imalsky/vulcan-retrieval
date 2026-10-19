"""WASP-39b SMC retrieval case: everything planet-specific for this run, and ONLY
that. The reusable machinery lives in the ``vulcan-retrieval`` package
(``retrieval_framework``).

theta (gpu preset, 13-D): [lnZ, dln(C/O), lnKzz, Tirr, log10kappa, log10gamma, Tint,
lnR0, log10kappa_cloud, cloud_alpha, offset_G395H_NRS1, offset_G395H_NRS2,
noise_inflation].

Run (from the repo root, or via the PBS script in this directory):

    SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
    SMC_RETRIEVAL_PRESET=gpu   python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
"""
from __future__ import annotations

import math
from typing import Any

from retrieval_framework.config_schema import Config, device_lane_count  # light import; jax only inside gpu_config
from retrieval_framework.forward import config as fwd_config  # pure constants + repo paths

# Planet + data identity (WASP-39b, Carter & May 2024 combined JWST spectrum);
# the geometry is fwd_config's, never re-typed here (one planet, one place).
_W39B = dict(
    run_label="WASP-39b",
    vulcan_cfg_name="W39b",
    tp_gravity_cgs=fwd_config.GS_CGS,           # cm/s^2 (also the RT g at rp_cm)
    rp_cm=fwd_config.RP_CM,
    rstar_cm=fwd_config.RSTAR_CM,
    # rp_cm/gs_cgs apply at 1 mbar, near the transmission photosphere (the
    # engine's own default). lnR0 then scales that radius at fixed planet mass,
    # so it reads as ln(R(1 mbar) / rp_cm) and stays nearly independent of T-P;
    # at the 7 bar grid bottom it also carried the whole column's thickness.
    p_ref_bar=1.0e-3,
    # Cap physically meaningless large-dt Ros2 oscillations on high-Kzz columns.
    # This is not a convergence criterion: yconv_cri/slope_cri remain the canonical
    # Tsai et al. (2017) values, and genuinely non-convergent draws are rejected at init.
    dt_max=1.0e11,
    # Carter & May (2024) fixed-limb-darkening products (repo data/); NIRISS
    # O1+O2 share NIRISS. G395H is the two detectors as one group (one offset);
    # the gpu preset fits NRS1 and NRS2 as separate groups instead: Sarkar+2024
    # (arXiv:2405.06737 §5.2) find a ~40-50 ppm NRS1-NRS2 offset in their
    # reductions of this G395H visit, and it sits between the H2O (NRS1) and
    # CO2/SO2/CO (NRS2) bands.
    obs_dir=fwd_config.DATA_DIR / "cm24_wasp39b",
    obs_products={
        "PRISM":  ("PRISM_native.csv",),
        "NIRISS": ("NIRISS_O1_R100.csv", "NIRISS_O2_R100.csv"),
        "G395H":  ("G395H_NRS1_R100.csv", "G395H_NRS2_R100.csv"),
        "G395H_NRS1": ("G395H_NRS1_R100.csv",),
        "G395H_NRS2": ("G395H_NRS2_R100.csv",),
        "NIRCam": ("NIRCam_R100.csv",),
    },
    # Drop two masked-gap truncation remnants (R=252 and 464, not R=100 bins).
    # R=200 removes both; the highest retained bin is R=103, safely below the
    # model/LSF refusal limit of R=333.
    obs_max_bin_R=200.0,
    # The chemistry's own geometry: VULCAN puts Rp at the layer nearest 1 bar and
    # derives g from Mp. The cfg's Rp is the 1.279 RJ transit radius; at fixed
    # mass the RT puts 1 bar at 1.20-1.25 RJ (a hydrostatic estimate on the
    # posterior), so 1.21 RJ keeps the chemistry's scale heights on the RT's.
    cfg_overrides={"Rp": 1.21 * fwd_config.R_JUP_CM},

    # ---- realistic WASP-39b priors (literature-anchored) ---------------------
    # Sources: Tsai et al. 2023 (Nature, VULCAN photochemistry grid for W39b) and
    # Rustamkulov et al. 2023 (Nature, NIRSpec PRISM ERS retrieval). All bounded,
    # kept wide enough not to pre-decide the posterior but physical enough that the
    # forward model converges.
    #
    #   metallicity : Tsai nominal 10x solar (tested 5-20x); ERS ~10x solar. Kept WIDE
    #                 1-100x solar (lnZ rel. to the 10x baseline) so the data localizes it.
    prior_lnZ=(-math.log(10.0), math.log(10.0)),   # 1x-100x solar
    #   C/O : Rustamkulov+2023 upper limit 0.7 (at 10x); Tsai tested 0.25-0.75; solar 0.55.
    #         dln(C/O) about the 0.549 baseline -> C/O in [0.10, 0.70]. Upper edge 0.24
    #         stays below the fixed-O b_z positivity bound (~0.566) too.
    prior_c_o=(-1.70, 0.24),
    #   Kzz : x0.1..x1000 about the VULCAN W39b baseline profile (3.4e9 cm^2/s
    #         at 1 mbar). The lower edge is Tsai's tested x0.1, a solver limit:
    #         below it most columns do not certify within count_max. Tsai's
    #         profile is w_rms x 0.1 H from their GCM, so x10 is the full
    #         H w_rms; x1000 is a loose exploratory ceiling, not a physical one
    #         (a 56x-solar mode peaked at x200-x300).
    prior_lnKzz=(math.log(0.1), math.log(1000.0)),
    #   T-P (Guillot, f = 1/4, so the skin temperature is ~0.59 Tirr at
    #         gamma << 1): Teq ~1100-1166 K, published limbs 889 / 1068 K
    #         (Espinoza+2024). With lnR0 at fixed g the posterior (T ~847 K at
    #         1 mbar) lay flat to an 800 K floor; fixed-mass gravity should
    #         lift T by ~20%, and 700 K keeps that ridge's tail without chasing a
    #         flat direction. Any out-of-window profile is REJECTED, not
    #         clipped (pipeline.tp_valid).
    prior_Tirr=(700.0, 2200.0),         # K
    #   kappa_th: the schema's -3.5..0.5. With Tint fixed at 150 K the posterior
    #         piled at 0.5, the only route to a hot deep column; the free Tint
    #         below gives that directly, and a higher ceiling only adds
    #         out-of-window volume (19.6% of the box at 0.5, 26.4% at 1.0).
    #   gamma >= 0.01: the deep temperature saturates at ~(3/16 / (gamma
    #         sqrt 3))^(1/4) Tirr, 1.8 Tirr at 0.01, already above Tsai's GCM
    #         evening profile at 1 bar (1605 K); lower gamma only heats the
    #         quench region further, which Tint now covers. gamma up to 2 lets
    #         the data prefer a weak inversion.
    prior_log10gamma=(-2.0, math.log10(2.0)),
    #   Tint (gpu preset: free): Tsai+2023 adopt 358 K from the Thorngren+2019
    #         Tint-flux relation; an interior fit gives ~300 K (arXiv:2410.04470).
    #         Fixed at 358 K in this radiative Guillot profile (no convective
    #         adjustment) it puts the preferred profiles' 7 bar level at ~4400 K,
    #         so it is retrieved instead, uniform over 50-600 K: 50 K is about
    #         where a 7-9 Gyr, 0.28 MJ planet cools to with no extra heating
    #         (Fortney+2020 tracks, extrapolated), 600 K is above the
    #         heated-interior estimates (~300-400 K). The 320-2980 K window
    #         (tp_profile.py) cuts most Tint above ~280 K at the preferred kappa.
    prior_Tint=(50.0, 600.0),           # K
    #   lnR0 = ln(R(1 mbar) / rp_cm) at fixed mass: the continuum depths put
    #         the transit radius ~3% above the 1.279 RJ literature radius (with
    #         this rstar), so expect ~+0.02-0.03.
    prior_lnR0=(-0.05, 0.15),
    # Cloud and offset priors keep the schema defaults (generic nuisances).
)


# Presets
def smoke_config(**overrides: Any) -> Config:
    """Tiny fully-offline preset: CO-only opacity (cached), coarse column, a small SMC
    swarm. Proves the chain end-to-end and FD-checks the gradient in minutes on CPU."""
    base = dict(
        _W39B,
        nz=30,
        molecules=("CO",),
        nu_min=4280.0, nu_max=4360.0,   # the CO 2-0 band (CO k-table only, fully offline)
        art_nlayer=20,
        combo=("G395H",),          # single group -> no offsets in the smoke
        infer_offsets=False,
        obs_wl_lo=2.28, obs_wl_hi=2.36,             # overlap the cached CO band
        tp_infer_gamma=False,      # 8-D smoke: lnZ, c_o, lnKzz, Tirr, log10kappa, lnR0, 2 cloud
        generate_synthetic_data=True,               # smoke always self-tests on an injection
        smc_num_particles=12, smc_num_mcmc_steps=4, smc_max_steps=8,
        smc_target_ess_frac=0.5,
        num_samples=12, num_chains=1, ppc_draws=12, ppc_chunk_size=6,
    )
    base.update(overrides)
    return Config(**base)


def gpu_config(**overrides: Any) -> Config:
    """GH200 production preset for the Carter & May NIRISS+G395H spectrum.

    Band note: the native model band is 1.01-5.26 um (nu 1900-9900). NIRISS SOSS
    order 1 supplies the short-wavelength water bands and offset/cloud leverage;
    the 1.02 um edge stays inside the H2-H2 CIA table. count_max=5000 is a hard
    convergence gate, not a reason to extend failed draws. Run CALIBRATE_ONLY
    before a full submission.
    """
    lanes = device_lane_count()
    base = dict(
        _W39B,
        nz=62,
        count_max=5000,
        # + HCN/C2H2 (high-C/O discriminators) + H2S (reduced-S reservoir): without
        # them the likelihood is blind to the species that rule the C/O upper tail
        # in or out. All have ExoMolOP k-tables, same path as the first five.
        # + NH3/OCS/SH/SO: each exceeds the min(5 ppm, 0.1 sigma) leave-one-out
        # gate at the nominal state. Omitting a produced absorber biases the
        # abundances that must absorb its opacity. A species is omitted only with a
        # recorded leave-one-out bound below the gate
        # (validation/opacity_leave_one_out.py).
        molecules=("H2O", "CO2", "CO", "CH4", "SO2", "HCN", "C2H2", "H2S",
                   "NH3", "OCS", "SH", "SO"),
        # ExoMolOP correlated-k is the only opacity path. Its 16-point g axis is
        # carried through the RT vjp, so the vjp chunk below sets its memory.
        nu_min=1900.0, nu_max=9900.0, art_nlayer=67,
        combo=("NIRISS", "G395H_NRS1", "G395H_NRS2"),
        obs_wl_lo=1.02, obs_wl_hi=5.24,   # strictly inside the native span (1.01-5.26)
        tp_infer_Tint=True,
        generate_synthetic_data=False,
        # N = the lane count = one kernel wave of the card (132 on the GH200,
        # 144 off the GPU): every particle's chemistry runs in one wave and the
        # queue never waits on a late starter. Both divide into exact RT-vjp
        # chunks of 4.
        # smc_max_steps is a per-JOB cap, not a per-run one (a RESUME job gets a
        # fresh budget and the stage index continues). Exhausting it yields a
        # TEMPERED cloud the certificate refuses and an unused stage costs
        # nothing, so it sits well above the expected stage count and the
        # governor is the real limit.
        smc_num_particles=lanes, cold_lanes=lanes, smc_max_steps=80,
        # Draw 2.5 N cold candidates (330 at N = 132) and reserve 48 phase-2
        # spares: the N + 48 pool can cull 48 (27% at N = 132) and still return
        # N. Peak memory remains set by the fixed RT-vjp chunk, not the number
        # of serialized chunks.
        init_oversample=2.5,
        init_phase2_spare=48,
        # Declared convergence attrition: pipeline._init_state raises above it.
        # 0.35 covers the measured rates with margin; the certificate
        # warns above CONV_ATTRITION_WARN until the removed region is shown to carry
        # negligible posterior mass.
        init_max_nonconverged_frac=0.35,
        # Cold chemistry: a draw's column never depends on sampler history, up to
        # the lane queue's refill tick (convergence scale). "warm" is
        # history-dependent and its logZ approximate.
        smc_chem_mode="cold",
        # Four sequential cold sweeps preserve particle count but may require
        # multiple 24 h jobs. RESUME continues the absolute stage index from the
        # checkpoint; CALIBRATE_ONLY must fit the configured governor before submit.
        smc_num_mcmc_steps=4,
        smc_target_ess_frac=0.6,
        # Free multiplicative error inflation (Line+2015). ON for real data: the
        # forward is a stiff self-consistent kinetics model, so its
        # misspecification has nowhere to go but into lnZ / C-O / the cloud deck
        # unless the noise scale is free to say the residuals are larger than the
        # quoted sigma. Analytic gradient (no chemistry solve), so it costs one
        # extra dimension and nothing else.
        infer_noise_inflation=True,
        # RT-vjp chunk: memory is linear in this width; a wider chunk
        # buys no time. Numerically identical at any width.
        smc_rt_vjp_chunk=4,
        num_samples=144, num_chains=2, ppc_draws=64, ppc_chunk_size=16,
        walltime_seconds=20.0 * 3600.0,   # SMC governor; leaves ~4 h of a 24 h PBS wall
    )                                     # for build/compile + init + PPC + plots
    base.update(overrides)
    return Config(**base)


def prod_config(**overrides: Any) -> Config:
    """Higher-fidelity variant (nz=100, more stages, no governor) for when >24 h is
    available. Inherits the gpu preset's particle/chunk settings and cold target,
    but explicitly raises the mutation budget from 4 to 8 sweeps per stage."""
    base = dict(
        nz=100, smc_num_mcmc_steps=8, smc_max_steps=96,
        ppc_draws=96,
        walltime_seconds=0.0,
    )
    base.update(overrides)
    return gpu_config(**base)


PRESETS = {"smoke": smoke_config, "gpu": gpu_config, "prod": prod_config}
DEFAULT_PRESET = "smoke"
