"""W107b Guillot photo-on source-bearing fixture (B0C, plan 2f / directive C2).

The MANDATORY scientific fixture for the B0C feasibility gates: a WASP-107b
column with photochemistry ON, a Guillot T(P), the equilibrium-table lower
H2S reservoir (nonzero deep sulfur supply), and active smooth rainout --
built through retrieval_framework.forward.vulcan_chem, i.e. the SAME
production forward path the tool will use. The 400 K isothermal column
(tests/test_smooth_rainout_prep.py) is the local mechanics fixture only and
is BANNED as a convergence fixture (measured nonconvergent from CO2 quench
creep unrelated to rainout; plan round-2).

System parameters follow the jwst-tool planet registry (Piaulet+2021):
gs = 270 cm/s^2, Rp = 0.94 R_Jup, R* = 0.67 R_Sun, a = 0.0553 au, eps Eri
UV proxy.

Guillot fiducial (REDESIGNED 2026-07-14 on measured evidence): the tool's
canonical hot defaults (Tirr 1046.5, log_gamma -1.0) produce a ~1000-1100 K
near-isothermal column whose G1 run MEASURED zero rainout -- sulfur goes to
SO2 (1.1e-4 peak at z=72, real photochemistry) but S8 never exceeds 3.8e-17
(saturation needs <~450 K; results/w107b_g1_20260714_064031.json). The
fixture demand (directive C2) is ACTIVE rainout, so the fiducial is the
measured coolest-valid candidate from a (Tirr, Tint, kappa, gamma) scan:
Tirr = 500 K, Tint = 150 K, log10 kappa = -2.5, log10 gamma = 0.0, f = 0.25
-- whole column inside the validated T window ([349.0, 469.6] K measured),
T_bottom = 469.6 K (69.6 K above the lookup-domain edge), minimum S8
saturation mixing ratio 6.4e-6 at 0.46 bar (4.7x supersaturation headroom
at a conservative 3e-5 photochemical S8 yield; 16x at the measured 1e-4
sulfur-funneling scale). Physically motivated by W107b's JWST ~500 K
photosphere; chosen for feasibility, not fitted to data. Structural
baseline is isothermal at 500 K -- the coolest FastChem-sane seed (the EQ
init MEASURED a 99.98% carbon collapse at 470 K, exact at 500 K); the
structural profile only sets the hydrostatic grid + EQ init and the
on-graph tp_eval supplies the live T(P).

theta = [lnZ, c_o, lnKzz, Tirr, Tint, log_kappa, log_gamma]   (n_tp = 4)

G1 honesty: NO runtime cap, NO step-cap relaxation, NO conver_ignore
additions beyond the module defaults. In smooth mode S8 is live (not
pinned), so the sulfur allotropes re-equilibrate against it normally -- the
master_pin-era S/S2/S3/S4 conver_ignore extension does not apply and is
deliberately absent. mtol_conv = 1e-15: the repo's certified convergence
floor for cold/condensing columns (CLAUDE.md; measured 2026-07-13), adopted
after the first G1 run MEASURED the stall to be sub-floor trace radicals
(N2H max mix 1.6e-17, CH3CO max 1.2e-15 -- results/
w107b_g1_20260714_064031.json), not sulfur physics. Convergence must be
earned or the gate fails with measured numbers.

Heavy: one build is a full photo-on pre-loop + warm-up converge (minutes on
CPU). Every consumer script is env-gated (ROUTE_B_W107B=1) -- scheduled runs
are Isaac's.
"""
from __future__ import annotations

import os

import numpy as np

# --- system (jwst-tool planets registry, WASP-107b) ------------------------
GS_CGS = 270.0
RP_RJUP = 0.94
RSTAR_RSUN = 0.67
ORBIT_AU = 0.0553
SFLUX = "atm/stellar_flux/sflux-epseri.txt"
R_JUP_CM = 7.1492e9
R_SUN_CM = 6.957e10

# --- Guillot fiducial (measured cool source-bearing design; see docstring) --
TIRR_FID = 500.0
TINT_FID = 150.0
LOG_KAPPA_FID = -2.5
LOG_GAMMA_FID = 0.0
# Structural isothermal baseline: keeps the warm-up EQ init and static pin
# inside the lookup domain (Tirr/sqrt(2) = 354 K would sit BELOW the 400 K
# table edge) AND above the vendored FastChem's measured carbon-collapse
# threshold: the EQ init at Tiso = 470 K loses 99.98% of elemental carbon
# to underflow-scale CH4 columns (mix ~1e-98; the build's lnZ-basis gate
# caught it), while 500 K reproduces C/H = 2.9499e-3 exactly. 500 K is the
# coolest measured-sane structural seed; the live Guillot T(P) is what the
# solves actually use.
T_STRUCT_ISO = 500.0

FIDUCIAL_THETA = np.array(
    [0.0, 0.0, 0.0, TIRR_FID, TINT_FID, LOG_KAPPA_FID, LOG_GAMMA_FID],
    dtype=np.float64)
N_TP = 4
THETA_NAMES = ["lnZ", "c_o", "lnKzz", "Tirr", "Tint", "log_kappa", "log_gamma"]

# Declared lookup domain for the whole gate campaign (validated at build):
# T_bottom spans the 469.6 K fiducial deep temperature (5 K guard above the
# 400 K table edge for FD steps / Tint-Tirr ladders) up through the G4 hot
# variant (~1465 K at Tirr 1560); lnZ covers the +-0.5 supply ladder with
# margin; c_o small. Any gate point outside this box is a REFUSED
# configuration, not a clamp.
THETA_BOX = {
    "T_bottom": (405.0, 1800.0),
    "lnZ": (-1.0, 1.0),
    "c_o": (-0.4, 0.4),
}

ENV_GATE = "ROUTE_B_W107B"


def require_env_gate():
    if os.environ.get(ENV_GATE) != "1":
        raise SystemExit(
            f"{ENV_GATE}=1 not set: this is a heavy scheduled run (photo-on "
            "W107b build + solves), not a local test. Isaac schedules these.")


def make_tp_eval():
    """The tool's exact Guillot hook (exojax atmprof_Guillot, f = 0.25)."""
    import jax.numpy as jnp
    from exojax.atm.atmprof import atmprof_Guillot

    def tp_eval(tp, p_bar):
        p = jnp.asarray(p_bar)
        Tirr, Tint = tp[0], tp[1]
        kappa, gamma = 10.0 ** tp[2], 10.0 ** tp[3]
        return atmprof_Guillot(p, GS_CGS, kappa, gamma, Tint, Tirr, 0.25)

    return tp_eval


def smooth_cfg_overrides(pin0: float, kzz_const: float = 1.0e10,
                         **extra) -> dict:
    """Smooth-rainout channel + W107b system overrides (production path).

    Deliberately ABSENT vs the master_pin CONDEN_CFG: fix_species window/pin
    knobs, stop/start_conden_time, trun_min, the sulfur conver_ignore
    extension, any runtime cap -- G1 measures NORMAL convergence or fails
    loudly. mtol_conv 1e-15 IS carried: the certified trace floor (module
    docstring; first-run stall measured sub-floor N2H/CH3CO radicals).
    """
    over = {
        # system identity (tool cfg_overrides hook)
        "gs": GS_CGS,
        "Rp": RP_RJUP * R_JUP_CM,
        "r_star": RSTAR_RSUN,
        "orbit_radius": ORBIT_AU,
        "sflux_file": SFLUX,
        "use_moldiff": True,
        # photolysis geometry (tool defaults)
        "sl_angle": float(np.deg2rad(83.0)),
        "f_diurnal": 1.0,
        # structural baseline: isothermal near the fiducial T_bottom
        "atm_type": "isothermal",
        "Tiso": T_STRUCT_ISO,
        "Kzz_prof": "const",
        # 1e10: physically motivated for W107b (its CH4 depletion implies
        # very strong mixing) AND measured necessary -- at 1e9 the cold G1
        # solve is dt-pinned (~1e2 s) by stiff S3/S4 front chemistry while
        # the boundary fills the column on ~3e9 s (artifact
        # w107b_g1_20260714_070038); at 1e10 dt balloons to ~7e7 s, rainout
        # activates, and warm re-certification converges via the normal
        # longdy gate in ~121 steps with Phi_rain,S steady at ~1.97e14
        # (continuation probe, 2026-07-14).
        "const_Kzz": kzz_const,
        # smooth-rainout channel
        "use_condense": True,
        "condense_sp": ["S8"],
        "non_gas_sp": ["S8_l_s"],
        "r_p": {"S8_l_s": 5.0e-3},
        "rho_p": {"S8_l_s": 2.07},
        "conden_mode": "smooth_rainout",
        "conden_smooth_width": 0.1,
        "rainout_rate_scale": 1.0,
        "use_fix_sp_bot": {"H2S": float(pin0)},
        # neutralize master_pin machinery explicitly (also refused at build)
        "use_relax": [],
        "use_settling": False,
        "fix_species": [],
        # certified trace-floor for cold columns (see module docstring)
        "mtol_conv": 1.0e-15,
    }
    over.update(extra)
    return over


def build(nz: int = 100, count_max: int = 5000, use_photo: bool = True,
          cfg_extra: dict | None = None, profile_extra: dict | None = None):
    """Build the fixture chem model. Returns (chem, meta dict).

    Import order matters: vulcan_chem BEFORE exojax (make_tp_eval).
    """
    from retrieval_framework.forward import config, vulcan_chem
    from retrieval_framework.forward import h2s_boundary as hb

    table = hb.load_h2s_boundary_table(config.H2S_BOUNDARY_TABLE)
    t_struct = T_STRUCT_ISO
    pin0 = float(hb.h2s_pin_mix(table, t_struct, 0.0, 0.0))

    profile = dict(config.SMOKE)
    profile.update(
        nz=int(nz),
        use_photo=bool(use_photo),
        yconv_cri=1.0e-2,
        count_max=int(count_max),
        abundance_mode="elemental",
        cfg_overrides=smooth_cfg_overrides(pin0, **(cfg_extra or {})),
        h2s_boundary={"theta_box": {k: tuple(v) for k, v in THETA_BOX.items()}},
    )
    profile.update(profile_extra or {})

    tp_eval = make_tp_eval()
    chem = vulcan_chem.build_chem_model(profile, tp_eval=tp_eval,
                                        n_tp_params=N_TP)
    meta = {
        "fixture": "w107b_guillot_photo_smooth_rainout",
        "theta_names": THETA_NAMES,
        "fiducial_theta": FIDUCIAL_THETA.tolist(),
        "theta_box": {k: list(v) for k, v in THETA_BOX.items()},
        "nz": int(nz),
        "count_max": int(count_max),
        "use_photo": bool(use_photo),
        "structural_Tiso_K": float(t_struct),
        "static_pin_H2S": pin0,
        "table_sha256": table.sha256,
        "cfg_overrides": {k: (v if not isinstance(v, np.floating) else float(v))
                          for k, v in profile["cfg_overrides"].items()},
    }
    return chem, meta
