"""WASP-39b SMC retrieval case: everything planet-specific for this run, and ONLY
that. The reusable machinery lives in the ``vulcan-retrieval`` package
(``retrieval_framework``).

theta (gpu preset, 11-D): [lnZ, dln(C/O), lnKzz, Tirr, log10kappa, log10gamma,
lnR0, log10kappa_cloud, cloud_alpha, offset_G395H, noise_inflation].

Run (from the repo root, or via the PBS script in this directory):

    SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
    SMC_RETRIEVAL_PRESET=gpu   python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
"""
from __future__ import annotations

from typing import Any

from retrieval_framework.config_schema import Config      # light import, no jax
from retrieval_framework.forward import config as fwd_config  # pure constants + repo paths

# Planet + data identity (WASP-39b, Carter & May 2024 combined JWST spectrum)
R_SUN_CM = 6.957e10
_W39B = dict(
    run_label="WASP-39b",
    vulcan_cfg_name="W39b",
    tp_gravity_cgs=422.0,                       # cm/s^2 (also the RT g_btm)
    rp_cm=1.279 * 7.1492e9,                     # planet radius at P_btm
    rstar_cm=0.932 * R_SUN_CM,
    fastchem_met_scale=10.0,                    # baseline 10x solar; lnZ relative to it
    # Cap physically meaningless large-dt Ros2 oscillations on high-Kzz columns.
    # This is not a convergence criterion: yconv_cri/slope_cri remain the canonical
    # Tsai et al. (2017) values, and genuinely non-convergent draws are rejected at init.
    dt_max=1.0e11,
    # Carter & May (2024) fixed-limb-darkening products (repo data/):
    # NRS1/NRS2 share the G395H group (one offset); NIRISS O1+O2 share NIRISS.
    obs_dir=fwd_config.DATA_DIR / "cm24_wasp39b",
    obs_products={
        "PRISM":  ("PRISM_native.csv",),
        "NIRISS": ("NIRISS_O1_R100.csv", "NIRISS_O2_R100.csv"),
        "G395H":  ("G395H_NRS1_R100.csv", "G395H_NRS2_R100.csv"),
        "NIRCam": ("NIRCam_R100.csv",),
    },
    # Drop two masked-gap truncation remnants (R=252 and 464, not R=100 bins).
    # R=200 removes both; the highest retained bin is R=103, safely below the
    # model/LSF refusal limit of R=333.
    obs_max_bin_R=200.0,

    # ---- realistic WASP-39b priors (literature-anchored) ---------------------
    # Sources: Tsai et al. 2023 (Nature, VULCAN photochemistry grid for W39b) and
    # Rustamkulov et al. 2023 (Nature, NIRSpec PRISM ERS retrieval). All bounded,
    # kept wide enough not to pre-decide the posterior but physical enough that the
    # forward model converges.
    #
    #   metallicity : Tsai nominal 10x solar (tested 5-20x); ERS ~10x solar. Kept WIDE
    #                 1-100x solar (lnZ rel. to the 10x baseline) so the data localizes it.
    prior_lnZ=(-2.303, 2.303),          # 1x .. 100x solar
    #   C/O : Rustamkulov+2023 upper limit 0.7 (at 10x); Tsai tested 0.25-0.75; solar 0.55.
    #         dln(C/O) about the 0.549 baseline -> C/O in [0.10, 0.70]. Upper edge 0.24
    #         stays below the fixed-O b_z positivity bound (~0.566) too.
    prior_c_o=(-1.70, 0.24),
    #   Kzz : Tsai nominal Kzz(P) scaled x0.1..x10; widened to x0.01..x100 (+/-2 dex)
    #         about the VULCAN W39b baseline profile.
    prior_lnKzz=(-4.6, 4.6),
    #   T-P (Guillot) : Teq ~1100-1166 K; SO2 photochemistry sweet spot Teq 1000-1600 K
    #         (Tsai 2023). With f=1/4 the terminator ~0.7*Tirr, so Tirr in [1100, 2200] K
    #         gives a limb T ~770-1540 K -- physical for W39b, no unmodelably cold/hot
    #         corners. gamma up to ~2 lets the data prefer a WEAK thermal inversion;
    #         a mild inversion actually cools the deep atmosphere, so
    #         it slightly LOWERS the reject rate. Any residual out-of-window profile is
    #         REJECTED, not clipped (pipeline.tp_valid).
    prior_Tirr=(1100.0, 2200.0),        # K
    prior_log10gamma=(-2.0, 0.301),     # gamma = kappa_v/kappa_th in [0.01, 2.0]
    # prior_log10kappa (IR opacity), prior_lnR0, cloud, and offset priors keep the
    # schema defaults (generic nuisances, not W39b-specific).
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
        tp_infer_gamma=False,      # 5-D smoke: lnZ, c_o, lnKzz, Tirr, log10kappa (+lnR0)
        generate_synthetic_data=True,               # smoke always self-tests on an injection
        smc_num_particles=12, smc_num_mcmc_steps=4, smc_max_steps=8,
        smc_target_ess_frac=0.5,
        num_samples=12, num_chains=1, ppc_draws=12, ppc_chunk_size=6,
        do_ppc=True,
    )
    base.update(overrides)
    return Config(**base)


def gpu_config(**overrides: Any) -> Config:
    """GH200 production preset for the Carter & May NIRISS+G395H spectrum.

    Band note: the native model band is 1.01-5.26 um (nu 1900-9900). NIRISS SOSS
    order 1 supplies the short-wavelength water bands and offset/cloud leverage;
    the 1.02 um edge stays inside the H2-H2 CIA table. count_max=5000 is a hard
    convergence gate, not a reason to extend failed draws. Run PROBE_MEMORY after
    any band/chunk/N change and CALIBRATE_ONLY before a full submission.
    """
    base = dict(
        _W39B,
        nz=62,
        count_max=5000,
        # + HCN/C2H2 (high-C/O discriminators) + H2S (reduced-S reservoir): without
        # them the likelihood is blind to the species that rule the C/O upper tail
        # in or out. All have ExoMolOP k-tables, same path as the first five.
        # + NH3/OCS/SH/SO: each exceeds the min(5 ppm, 0.1 sigma) leave-one-out
        # gate at the nominal state. Omitting a produced absorber biases the
        # abundances that must absorb its opacity.
        # NOT YET SCREENED: 13 further species have both a solved abundance and an
        # installed ExoMolOP table (CS N2O NO NS NH CN OH CH3 H2CO C2H4 CH C2
        # H2O2). validation/opacity_leave_one_out.py measures them; a species may
        # be dropped only with a recorded bound below the gate.
        molecules=("H2O", "CO2", "CO", "CH4", "SO2", "HCN", "C2H2", "H2S",
                   "NH3", "OCS", "SH", "SO"),
        # ExoMolOP correlated-k is the only opacity path. Its 16-point g axis is
        # carried through the RT vjp, so PROBE_MEMORY must certify the chunk below.
        nu_min=1900.0, nu_max=9900.0, art_nlayer=67,
        combo=("NIRISS", "G395H"),
        obs_wl_lo=1.02, obs_wl_hi=5.24,   # strictly inside the native span (1.01-5.26)
        generate_synthetic_data=False,
        # N=144 gives 12 exact RT-vjp chunks and reduces small-cloud SMC noise.
        # smc_max_steps is a per-JOB cap, not a per-run one (a RESUME job gets a
        # fresh budget and the stage index continues). 40 sits right at the edge
        # of what a 10-D ladder at target_ess_frac=0.6 needs, and exhausting it
        # yields a TEMPERED cloud the certificate refuses; an unused stage costs
        # nothing, so the governor is left as the real limit.
        smc_num_particles=144, smc_max_steps=80,
        # Draw 360 cold candidates and reserve 48 phase-2 spares. The 192-column
        # phase-2 pool can cull 25% and still return N=144; peak memory remains set
        # by the fixed RT-vjp chunk, not the number of serialized chunks.
        init_oversample=2.5,
        init_phase2_spare=48,
        # DECLARED convergence attrition. The cold reject fraction is a gate, not a
        # warning (pipeline._init_state raises above it), because conditioning on
        # convergence removes part of the declared prior. 0.35 covers the measured
        # 29% with margin. It is NOT a claim that the removed region is negligible:
        # certificate.validate still fails any run above CONV_ATTRITION_FAIL until
        # that region is shown to carry negligible posterior mass.
        init_max_nonconverged_frac=0.35,
        # COLD chemistry. Every likelihood evaluation uses the published
        # solve-from-baseline map, so the target is a fixed deterministic
        # function of theta -- what MALA, SMC tempering, and the evidence
        # integral all assume. Under "warm" the likelihood depends on each
        # particle's carried column, hence on sampler history, at the
        # convergence tolerance; the resulting logZ is approximate in a way
        # diagnostics cannot repair.
        smc_chem_mode="cold",
        # warm_extrapolate is a WARM-only optimization (validate_config raises
        # if it is on in cold mode): it seeds each warm solve at the first-order
        # tangent prediction of the carried column, and there is no carried
        # column to extrapolate from in cold mode.
        warm_extrapolate=False,
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
        # PROBE_MEMORY=1 must certify this RT-vjp width for all 12 absorbers before
        # production; the compile-only probe cannot OOM.
        smc_rt_vjp_chunk=12,
        mcmc_stage_adapt=True,
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
