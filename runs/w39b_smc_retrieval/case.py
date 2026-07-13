"""WASP-39b SMC retrieval case: everything planet-specific for this run, and ONLY
that. The reusable machinery lives in the ``vulcan-retrieval`` package
(``retrieval_framework``).

theta (gpu preset, 10-D): [lnZ, dln(C/O), lnKzz, Tirr, log10kappa, log10gamma,
lnR0, log10kappa_cloud, cloud_alpha, offset_G395H].

Run (from the repo root, or via the PBS script in this directory):

    SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
    SMC_RETRIEVAL_PRESET=gpu   python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval_framework.config_schema import Config      # light import, no jax
from retrieval_framework.forward import config as fwd_config  # pure constants + repo paths

# ---------------------------------------------------------------------------
# Planet + data identity (WASP-39b, Carter & May 2024 combined JWST spectrum)
# ---------------------------------------------------------------------------
R_SUN_CM = 6.957e10
_W39B = dict(
    run_label="WASP-39b",
    vulcan_cfg_module="vulcan_jax.cfg_examples.vulcan_cfg_W39b",
    tp_gravity_cgs=422.0,                       # cm/s^2 (also the RT g_btm)
    rp_cm=1.279 * 7.1492e9,                     # planet radius at P_btm
    rstar_cm=0.932 * R_SUN_CM,
    fastchem_met_scale=10.0,                    # baseline 10x solar; lnZ relative to it
    # Physical step-size cap (s). The VULCAN-master default (runtime*1e-5 = 1e17 s) lets
    # the adaptive Ros2 step balloon to ~1e16 s on high-Kzz columns (per-step local error
    # stays tiny), so the solver SPINS in a large-dt oscillation and never converges --
    # this was the BULK of the >10k-step tail. Local diagnostic 2026-07-08 (chemistry-only,
    # scratchpad diag_*.py): capping dt_max converges the ballooning draws in ~1000 steps
    # (d10/d19/d59 from calib job 64523; d19 was >11000 uncapped) and leaves normal draws
    # untouched (truth 4275 steps identically capped/uncapped; converged dt ~1e8 << cap).
    # 1e11 s catches the ballooning with margin (reaches t~1e15 in 10k steps, >> physical
    # settling ~1e13). NOT a convergence criterion -- yconv_cri/slope_cri stay master
    # (Tsai+2017: 0.01, 1e-4). A residual minority still fails for OTHER reasons dt_max
    # can't fix (longdy stuck ~0.13 just above the 0.1 gate; hot/low-Kzz photolysis limit
    # cycles with aflux oscillating) -- those get rejected at init.
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

    # ---- realistic WASP-39b priors (literature-anchored 2026-07-08) -----------
    # Sources: Tsai et al. 2023 (Nature, VULCAN photochemistry grid for W39b) and
    # Rustamkulov et al. 2023 (Nature, NIRSpec PRISM ERS retrieval). All bounded,
    # kept wide enough not to pre-decide the posterior but physical enough that the
    # forward model converges. See CLAUDE.md "priors" for the mapping notes.
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
    #         corners. gamma up to ~2 lets the data prefer a WEAK thermal inversion
    #         (Isaac, 2026-07-08); a mild inversion actually cools the deep atmosphere, so
    #         it slightly LOWERS the reject rate. Any residual out-of-window profile is
    #         REJECTED, not clipped (pipeline.tp_valid).
    prior_Tirr=(1100.0, 2200.0),        # K
    prior_log10gamma=(-2.0, 0.301),     # gamma = kappa_v/kappa_th in [0.01, 2.0]
    # prior_log10kappa (IR opacity), prior_lnR0, cloud, and offset priors keep the
    # schema defaults (generic nuisances, not W39b-specific).
)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def smoke_config(**overrides: Any) -> Config:
    """Tiny fully-offline preset: CO-only opacity (cached), coarse column, a small SMC
    swarm. Proves the chain end-to-end and FD-checks the gradient in minutes on CPU."""
    base = dict(
        _W39B,
        nz=30,
        molecules=("CO",),
        nu_min=4280.0, nu_max=4360.0, nu_pts=400,   # cached CO 2-0 band (fully offline)
        art_nlayer=20,
        combo=("G395H",),          # single group -> no offsets in the smoke
        infer_offsets=False,
        obs_wl_lo=2.28, obs_wl_hi=2.36,             # overlap the cached CO band
        tp_infer_gamma=False,      # 5-D smoke: lnZ, c_o, lnKzz, Tirr, log10kappa (+lnR0)
        generate_synthetic_data=True,               # smoke always self-tests on an injection
        smc_num_particles=12, smc_num_mcmc_steps=4, smc_max_steps=8,
        smc_target_ess_frac=0.5, mcmc_tune_particles=6, mcmc_tune_steps=4, mcmc_tune_iters=4,
        num_samples=12, num_chains=1, ppc_draws=12, ppc_chunk_size=6,
        do_ppc=True,
    )
    base.update(overrides)
    return Config(**base)


def gpu_config(**overrides: Any) -> Config:
    """The <24 h GH200 production preset: nz=50, photo on, full molecule set over the
    real Carter & May combined NIRISS+G395H band, N=144 forward-mode-jvp MALA particles
    with per-stage adaptation. Fits the real spectrum (generate_synthetic_data=False).

    Band note: the native model band is 1.01-5.26 um (nu 1900-9900). NIRISS SOSS
    order 1 spans 0.85-2.83 um -- extending the band to ~1 um keeps 93 NIRISS bins
    (vs 29 at a 2.0 um cut): the 1.1-1.9 um water bands, the inter-instrument offset
    lever, and the cloud/haze slope. Chemistry cost is band-independent; only the
    (cheap) RT grows. The short edge stays >=1 um (H2-H2 CIA table edge at 10000
    cm^-1; PRISM's <2 um saturation is a PRISM issue -- NIRISS is unaffected).

    count_max=5000 (Isaac, 2026-07-08). The old >10k-step tail was diagnosed as a
    dt_max-ballooning numerical artifact (see _W39B dt_max + ../../CLAUDE.md): with
    dt_max=1e11 the ballooning draws converge in ~1000 steps, well under 5000, and the
    truth needs 4275. So 5000 gives ~700 steps over the truth -- draws harder than the
    truth (needing 5000-10000 steps) and the genuine residual (marginal-longdy /
    photolysis limit cycles) will FAIL at 5000, which is accepted (they are rejected at
    init). Measure the real failure fraction with the calibration (now cheap at R=100),
    probing AT the production cap: `qsub -v CALIBRATE_COUNT_MAX=1,
    CALIBRATE_COUNT_MAX_PROBE=5000,CALIBRATE_N_DRAWS=96 run_nas_w39b.pbs`.

    Sweep-cost budget (2026-07-09 rework; see ../../CLAUDE.md "Mutation sweep cost"):
    warm proposals are warm_count_max-capped (schema default 1500; measured typical
    warm re-converge ~500-800 steps, conv_step-window-dominated -- the cold count_max
    only ever binds at init), the warm gradient runs ONE chemistry while_loop (the
    accept_count diag rides the jvp chain), 6 sweeps/stage (was 12), and the RT vjp
    runs 12-wide at nu_pts=1652 (12 serialized chunks at N=144). Projected ~20-40
    min/stage vs the ~3-6 h/stage job 64745 showed.
    Run `qsub -v PROBE_MEMORY=1` once after any nu_pts / chunk / N change, and
    `qsub -v CALIBRATE_ONLY=1` (~1 h) to get timing.json before committing a full run.
    """
    base = dict(
        _W39B,
        nz=50,
        count_max=5000,
        # + HCN/C2H2 (high-C/O discriminators) + H2S (reduced-S reservoir): without
        # them the likelihood is blind to the species that rule the C/O upper tail
        # in or out. All HITRAN main-isotopologue, same path as the first five.
        molecules=("H2O", "CO2", "CO", "CH4", "SO2", "HCN", "C2H2", "H2S"),
        # nu_pts=1652 -> native R~1000. The data is 152 binned points, so R~1000 native
        # (~11 model pts per bin) is ample; the old nu_pts=16500 (R~10000) was overkill
        # AND blew the RT-VJP gradient to 343 GiB (OOM on the 96 GB GH200 -- job 64601).
        # RT-vjp memory scales with nu_pts. (Isaac, 2026-07-08: "don't do that high
        # resolution.")
        nu_min=1900.0, nu_max=9900.0, nu_pts=1652, art_nlayer=60,
        combo=("NIRISS", "G395H"),
        obs_wl_lo=1.02, obs_wl_hi=5.24,   # strictly inside the native span (1.01-5.26)
        generate_synthetic_data=False,
        # N=144 (raised from 96, 2026-07-10): the GPU power trace showed real headroom
        # (~300 W of 700 W during the 192-lane primal, ~360-390 W at 672 gradient
        # lanes) -- width fills the GPU nearly for free in the launch-bound
        # while_loop, so spend the idle silicon on particles. 144 = 12 exact RT-vjp
        # chunks of 12 (RT tail x1.5 vs N=96's 8 chunks). MEASURED (probe job 64944):
        # peak memory is WIDTH-INDEPENDENT -- 73.25 GiB at N=96, 144, and the
        # 152-wide init eval alike (the peak is the fixed-width RT-vjp chunk stage;
        # chem tangents are freed before it). N=192 is memory-viable (16 chunks) if
        # ever wanted. More particles also directly answer the small-N SMC criticism
        # (ladder-adaptation noise, evidence variance).
        smc_num_particles=144, smc_num_mcmc_steps=6, smc_max_steps=40,
        # Init reject-and-cull sizing (raised 2026-07-12 after job on real data died
        # in init phase 2: 84/288 cold-rejected [29%, expected] then 21/152 warm
        # re-certification failures [14%] vs the schema default init_phase2_spare=8 ->
        # 131/144 healthy, spares exhausted). The 8-spare default was sized for the
        # 3-5% warm-recert rate measured pre-pass (jobs 64854/64897, synthetic);
        # the 2026-07-12 elemental-map + per-proposal atm rebuild and REAL prior
        # corners push it to ~14%. count_max stays 5000 (Isaac), so absorb the higher
        # attrition here: oversample 2.5 -> ceil(144*2.5)=360 cold draws (~256 alive
        # at 29% reject), spare 48 -> phase-2 pool min(n_alive, 144+48)=192, tolerating
        # a warm-recert cull up to (192-144)/192 = 25% and still leaving 144. Both are
        # ~free: phase-1 wall = the slowest lane (lockstep max, batch width irrelevant),
        # phase-2 memory is width-independent to N~500 (probe 64944 measured 73.25 GiB
        # flat at N=96/144/152). No memory probe needed for this bump: the 192-wide
        # init eval only adds serialized RT-vjp chunks (not chunk width) and touches
        # neither nu_pts nor smc_rt_vjp_chunk -- the only two knobs that move the peak.
        init_oversample=2.5, init_phase2_spare=48,
        # Tangent-extrapolated warm starts ON (Isaac, 2026-07-10): proposals seed at
        # Y + (dy/dtheta).dtheta -- measured 1.65x fewer warm steps, same certified
        # state (parity unit-tested; validate_warm gates the production result and
        # the staged CALIBRATE->SYNTH->production sequence exercises it end-to-end
        # before real data). warm_count_max stays 1500; drop toward ~800 only after
        # heartbeat rejected-counts confirm typical warm solves sit well under it.
        warm_extrapolate=True,
        smc_target_ess_frac=0.6,
        # 12-wide RT vjp at nu_pts=1652 (~half the 5000-probed per-lane cost applies;
        # est. ~40-55 GiB vs the ~81 GiB pool). PROBE_MEMORY=1 once before the first
        # production submit -- the probe is compile-only and cannot OOM.
        smc_rt_vjp_chunk=12,
        mcmc_stage_adapt=True, mcmc_auto_tune=True,
        num_samples=144, num_chains=2, ppc_draws=64, ppc_chunk_size=16,
        walltime_seconds=20.0 * 3600.0,   # SMC governor; leaves ~4 h of a 24 h PBS wall
    )                                     # for build/compile + init + PPC + plots
    base.update(overrides)
    return Config(**base)


def prod_config(**overrides: Any) -> Config:
    """Higher-fidelity variant (nz=100, more stages, no governor) for when >24 h is
    available. Inherits the gpu preset's sweep-cost settings (N=144, 6 sweeps/stage,
    warm_count_max)."""
    base = dict(
        nz=100, smc_num_mcmc_steps=8, smc_max_steps=48,
        ppc_draws=96,
        walltime_seconds=0.0,
    )
    base.update(overrides)
    return gpu_config(**base)


PRESETS = {"smoke": smoke_config, "gpu": gpu_config, "prod": prod_config}
DEFAULT_PRESET = "smoke"
