"""VULCAN-JAX side of the demo: the differentiable physics-parameters -> converged-VMR map.

``build_chem_model(profile)`` runs the one-time WASP-39b pre-loop + a single warm-up
convergence (which compiles & caches the JIT'd inner runner), then returns a model whose
``converged_ymix(theta)`` re-converges the closed column as a function of

    theta = [lnZ, c_o, lnKzz, T_int]   (all scalars)

Abundance knobs -- two modes (``profile["abundance_mode"]``):
  * ``"masks"`` (legacy default): multiplicative species-mask y0 directions, the
    validated jax_paper patterns (fig_metallicity_sens.py / fixed-O C/O). These are
    NOT exact elemental directions: scaling every C/N/O/S-bearing species also moves
    the hydrogen those molecules carry (~0.6% of elemental H per e-fold of Z at the
    10x-solar baseline), the fixed-O b_z compensation leaks into N/S through NO/SO/SO2,
    and the scaled column no longer sums to M = P/(kB T) until the runner's per-step
    hydrostatic renorm restores it. Kept for reproducing the published demo caches.
  * ``"elemental"`` (retrieval / production default via config_schema): the mask scaling
    is only a smooth initial GUESS; the column is then renormalized to sum_i n_i = M
    per layer and repaired (three fixed Newton-style iterations of a small linear solve
    on the runner's own reservoir species He/H2O/CO/N2/H2S) so the column-integrated
    elemental ratios hit the targets EXACTLY:
        He/H = baseline,  O/H = Z x baseline,  N/H = Z x baseline,
        S/H  = Z x baseline,  C/H = Z e^{c_o} x baseline   (=> dln(C/O) = c_o at fixed O/H)
    with Z = e^lnZ relative to the FastChem baseline (fastchem_met_scale). The conserved
    atom totals ``pv.atom_ini`` are rebuilt from the repaired column, so the runner's
    atom-conservation anchor, the third-body density, the pressure, and the initial
    composition all describe the same gas -- and cold/warm paths share identical
    conserved inventories by construction (targets depend on theta only, never on the
    warm-start history). ``reanchor_atom_ini`` is moot in this mode. Residuals after the
    fixed iterations are ~1e-8 relative; measure them with ``audit_init``.

Temperature / atmosphere: rate constants are rebuilt on-graph (rates_jax) and the
ATMOSPHERIC STRUCTURE now follows the proposed T-P/composition too. The runner itself
refreshes the hydrostatic geometry (mu, g, Hp, dz, dzi, Hpi) in-loop from the live
composition and ``pv.r_Tco`` every ``update_frq`` accepted steps (first firing on the
first accepted step), so the converged column was already hydrostatically consistent;
what was frozen at the baseline was (a) the molecular-diffusion coefficients Dzz(T, M)
(+ the derived vm / settling vs), (b) the convergence gate's ``pv.Kzz``, and (c) the
initial carry geometry for step 1. All three are now rebuilt per proposal via the
committed on-graph builders (vulcan_jax.atm_jax / atm_refresh).

Condensation follows the live T(P) too (2026-07-13): with ``use_condense=True``,
``vulcan_jax.conden.make_conden_spec`` extracts the static metadata once at build
and ``_prep`` rebuilds every T/structure-dependent condensation array on-graph per
proposal (``conden.build_conden_profile``: saturation number densities, Dg
growth/diffusion terms from the live Dzz, H2O/NH3 relax inputs, the NH3 cold-trap
argmin, fix-species saturation mixing ratios), splicing them into the ProfileVars
carry the runner reads each step. Baseline-frozen condensation tables never reach
a live-T solve. Genuinely unsupported condensation configs still refuse loudly at
build time: ``use_moldiff=False`` (the growth term Dg IS the molecular-diffusion
coefficient -- it would be silently zero), an empty/inert ``condense_sp``, and
``use_sat_surfaceH2O`` (a bottom BC frozen at the structural T at ini time). The
cold-trap index and the active-condensation layer set are discrete: a jvp through
a condensing state is valid only away from those switches (validated in
tests/test_condensation_live_tp.py; Fisher forecasts through condensation stay
disabled in vulcan-jwst-tool).

KNOWN LIMITATION -- condensation-enabled does NOT reduce to condensation-off when
nothing condenses. The upstream conden-window + whole-column ``fix_species`` pin
freezes the condensable reservoirs at their ``stop_conden_time`` state. On a column
too hot for the species to supersaturate (or one whose chemistry has not settled by
the window's end), that pin still captures a transient state rather than the
condensation-off steady state, so ``use_condense=True`` and ``use_condense=False``
can differ even where no mass actually rained out. Enable condensation only for
columns where the species genuinely condenses. A criterion-gated pin (activate only
on measurable condensate/supersaturation) is a possible future refinement but would
change the certified convergence recipe (the pin is what makes a condensing column
converge) and needs measured re-validation before adoption -- not done here.

Still frozen by design: the photolysis cross-section T-interpolation (host-side
re-bake upstream; second-order).

The runner's lax.while_loop supports jvp/jacfwd but NOT vjp, so forward-mode is the
end-to-end route -- which is also optimal here (few scalar inputs -> high-dim spectrum).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from retrieval_framework.forward import config

# This module must own the first jax import: it fixes the VULCAN_JAX_* import-frozen
# env vars and jax x64. If exojax got in first those knobs are already baked wrong.
if "exojax" in sys.modules:
    raise RuntimeError(
        "retrieval_framework.forward.vulcan_chem must be imported BEFORE exojax: "
        "it fixes jax x64 and the VULCAN_JAX_* import-frozen env vars. "
        "Import order: forward.config, forward.vulcan_chem, then forward.exojax_rt.")

# --- env setup MUST happen before importing vulcan_jax / jax ------------------
os.environ["VULCAN_JAX_NETWORK"] = config.VULCAN_NETWORK
os.environ["VULCAN_JAX_ATOM_LIST"] = config.VULCAN_ATOM_LIST
os.environ.setdefault("OMP_NUM_THREADS", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Column-repair pairs for the exact-elemental mode: element -> adjuster species.
# These are the runner's own atom-conservation reservoirs (jax_step._ATOM_RESERVOIRS),
# i.e. the abundant carrier of each element in an H2-dominated gas, so the linear
# repair stays tiny and well-conditioned across the retrieval prior box. H is the
# reference element (its absolute density is set by sum_i n_i = M); He preserves the
# baseline He/H.
_ELEMENTAL_REPAIR = (("He", "He"), ("O", "H2O"), ("C", "CO"), ("N", "N2"), ("S", "H2S"))
# Number of renorm+repair iterations. Each iteration nails the column ratios exactly,
# then the per-layer renorm to M perturbs them by O(alpha x layer heterogeneity); the
# residual contracts geometrically (~1e-2 -> ~1e-8 by three passes; see audit_init).
_ELEMENTAL_REPAIR_ITERS = 3


class ConvDiag(NamedTuple):
    """Per-solve convergence diagnostics read off the runner's final carry.

    Every field rides the runner's primal carry, so reading them is free inside a
    forward-mode jvp chain. ``accept_count`` and ``count_since_new_min`` are
    integer-valued (no tangent); ``longdy``/``longdydt`` are floats and DO carry a
    tangent -- callers on an AD path must ``stop_gradient`` them.

    ``accept_count`` alone is NOT a convergence test: the stall fallback (and the
    hybrid vm_mol phase flip) can terminate the runner with accept_count well under
    the cap on a state whose tangent has not settled. ``conv_normal`` is the
    runner's canonical two-branch certification (tight yconv_cri/slope_cri OR
    loose yconv_min/slope_min, AND the photo-flux gate) recomputed at the exit
    state: False on an exit that only certified via the stall fallback or that
    exhausted a budget.
    """

    accept_count: jnp.ndarray         # () int32   accepted steps taken
    longdy: jnp.ndarray               # () float64 runner's convergence metric at exit
    longdydt: jnp.ndarray             # () float64 longdy per lookback time
    count_since_new_min: jnp.ndarray  # () int32   steps since longdy improved >= 5%
    conv_normal: jnp.ndarray          # () bool    canonical certification at exit


def build_chem_model(profile: dict, tp_eval=None, n_tp_params: int = 0) -> SimpleNamespace:
    """Build the converged WASP-39b model and the differentiable converged_ymix(theta).

    Parameters
    ----------
    profile : dict
        One of ``config.SMOKE`` / ``config.FULL`` -- supplies ``use_photo`` and
        ``yconv_cri``. ``profile["abundance_mode"]`` selects "masks" (legacy) or
        "elemental" (exact conserved-inventory construction; see module docstring).
    tp_eval : callable or None, optional
        Temperature-profile hook. When ``None`` (default, unchanged behavior) the
        temperature is the validated uniform shift ``T = T_base + theta[3]`` (theta[3]
        is a bulk offset; the demo's historical "T_int" label). When supplied,
        ``tp_eval(theta[3:3+n_tp_params], p_bar)`` returns the full (nz,) T-P profile
        (bar-indexed) that replaces the scalar shift -- used by the retrieval framework
        to retrieve an ExoJax Guillot/power-law T-P. Either way the rate table AND the
        T/composition-dependent atmospheric structure are rebuilt on-graph.
    n_tp_params : int, optional
        Number of T-P parameters consumed from ``theta[3:]`` when ``tp_eval`` is given.

    Returns
    -------
    SimpleNamespace with fields:
        converged_ymix(theta) -> (nz, ni) linear VMR, float64, differentiable
        audit_init(theta) -> host-side dict of elemental/density residuals at init
        T_base   : (nz,) baseline temperature (np.float64)
        p_bar    : (nz,) pressure grid in bar (np.float64)
        sidx     : dict species-name -> column index
        species_masses : (ni,) jnp molar mass per species (g/mol)
        nz, ni   : ints
    """
    t0 = time.time()
    import vulcan_jax

    # Baseline VULCAN config, loaded by name from vulcan_jax/configs/*.yaml
    # (overridable per profile; the case presets set this). Env VULCAN_JAX_* was
    # set above, so this first vulcan_jax import freezes the SNCHO network.
    cfg = vulcan_jax.load_config(profile.get("vulcan_cfg_name") or config.W39B_CFG_NAME)
    cfg.use_live_plot = cfg.use_live_flux = cfg.use_print_prog = False
    cfg.use_photo = bool(profile["use_photo"])
    cfg.yconv_cri = float(profile["yconv_cri"])
    if profile.get("nz"):
        cfg.nz = int(profile["nz"])
    if profile.get("count_min"):
        cfg.count_min = int(profile["count_min"])
    if profile.get("count_max"):
        cfg.count_max = int(profile["count_max"])
    # Warm-continuation step cap (accepted steps) for the MUTATION path only. A warm
    # re-converge from a particle's own converged column normally needs ~count_min-300
    # steps; a proposal still not converged at warm_count_max is headed for rejection
    # anyway, so cutting the loop there (instead of at the full cold count_max) stops a
    # single bad lane from dragging the whole lockstep batch through thousands of wasted
    # steps. Realized via a SECOND runner whose _Statics snapshot the smaller cap (the
    # cap is baked into the jitted while_loop at trace time); the cold/two-stage solves
    # keep count_max. warm_count_max == count_max (or absent) means one shared runner.
    warm_count_max = int(profile.get("warm_count_max") or cfg.count_max)
    if warm_count_max > int(cfg.count_max):
        raise ValueError(
            f"warm_count_max={warm_count_max} exceeds count_max={int(cfg.count_max)}: "
            "the warm mutation cap must be at most the cold cap (it exists to REJECT "
            "doomed proposals earlier, not to extend them)")
    if profile.get("dt_max"):               # physical step-size cap (prevents the dt-balloon
        cfg.dt_max = float(profile["dt_max"])  # non-convergence at high Kzz; see config_schema)
    if profile.get("yconv_min"):           # close the loose convergence OR-branch (default 0.1)
        cfg.yconv_min = float(profile["yconv_min"])
    if profile.get("slope_cri"):
        cfg.slope_cri = float(profile["slope_cri"])
    if profile.get("fastchem_met_scale"):  # BASELINE metallicity (x solar); W39b default 10.0.
        cfg.fastchem_met_scale = float(profile["fastchem_met_scale"])  # build at the bottom -> march up
    # Generic cfg overrides (e.g. use_moldiff=False for the no-transport equilibrium tier).
    # Applied BEFORE the pre-loop build, so they reach make_atm_static / OuterLoop exactly
    # like use_photo does. The fisher_zco tier configs are the only users.
    for _k, _v in (profile.get("cfg_overrides") or {}).items():
        setattr(cfg, _k, _v)

    import vulcan_jax
    from vulcan_jax.state import RunState, legacy_view
    from vulcan_jax import network as net_mod, composition, rates_jax
    from vulcan_jax import atm_jax, atm_refresh as atm_refresh_mod
    from vulcan_jax import conden as conden_mod
    from vulcan_jax.atm_setup import _VISCOSITY_TABLE, settling_velocity_jax
    from vulcan_jax.jax_step import make_atm_static
    from vulcan_jax.gibbs import load_nasa9
    from vulcan_jax._paths import resolve_data_path
    from vulcan_jax.phy_const import kb
    import vulcan_jax.legacy_io as op
    import vulcan_jax.op_jax as op_jax
    import vulcan_jax.outer_loop as outer_loop

    # Condensation with a live T(P) needs configuration that can actually
    # condense; refuse the silently-inert combinations upfront (standing rule:
    # loud errors, no silent fallbacks). The dynamic rebuild itself happens in
    # _prep below via conden.build_conden_profile.
    use_condense = bool(getattr(cfg, "use_condense", False))
    if use_condense:
        if not bool(getattr(cfg, "use_moldiff", True)):
            raise ValueError(
                "use_condense=True requires use_moldiff=True: the condensation "
                "growth term Dg IS the species' molecular-diffusion coefficient "
                "(op.conden's continuum-regime rate), so with molecular diffusion "
                "off every condensation rate would silently be zero.")
        if bool(getattr(cfg, "use_sat_surfaceH2O", False)):
            raise NotImplementedError(
                "use_condense with use_sat_surfaceH2O=True is unsupported in the "
                "T-varying model: it rewrites the fixed-bottom H2O boundary "
                "condition from the STRUCTURAL temperature at ini time, which a "
                "live T(P) does not rebuild. Disable use_sat_surfaceH2O.")
        if not list(getattr(cfg, "condense_sp", []) or []):
            raise ValueError(
                "use_condense=True with an empty condense_sp: nothing would "
                "condense. List the condensable gas species (network "
                "condensation reactions and/or use_relax species).")

    rs = RunState.with_pre_loop_setup(cfg)
    var, atm, para = legacy_view(rs)
    network = net_mod.parse_network(str(resolve_data_path(cfg.network)))
    nz, ni = atm.Tco.shape[0], network.ni
    sidx = dict(network.species_idx)

    # Static condensation metadata (species identity, particle coefficients,
    # relax/fix flags) -- extracted once; _prep rebuilds the dynamic arrays
    # from it at every proposed T. None when condensation is off.
    conden_spec = None
    if use_condense:
        conden_spec = conden_mod.make_conden_spec(cfg, var, atm, sidx)
        relax_set = set(getattr(cfg, "use_relax", []) or [])
        inert = [sp for sp in cfg.condense_sp
                 if sp not in conden_spec.gas_names and sp not in relax_set]
        if inert:
            raise ValueError(
                f"condense_sp entries {inert} have no condensation reaction in "
                f"network {cfg.network!r} and are not in use_relax -- they would "
                "silently not condense. Remove them or add the reaction/relax.")
        print(f"[chem] condensation ON: kinetics rows {list(conden_spec.gas_names)}, "
              f"relax H2O={conden_spec.h2o_active} NH3={conden_spec.nh3_active}, "
              f"fix_species={list(conden_spec.fix_names)}; conden arrays rebuilt "
              "on-graph at each proposed T", flush=True)

    pco = jnp.asarray(np.asarray(atm.pco, dtype=np.float64))
    p_bar = np.asarray(atm.pco, dtype=np.float64) / 1.0e6
    p_bar_j = jnp.asarray(p_bar)   # bar-indexed grid for the optional tp_eval hook

    thermo_dir = resolve_data_path(cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = Path(vulcan_jax.__file__).resolve().parent / "thermo"
    nasa9, _ = load_nasa9(network.species, thermo_dir)
    remove_list = getattr(cfg, "remove_list", None)

    # --- one warm-up run: compiles/caches integ._runner and confirms the primal converges
    solver = op_jax.Ros2JAX()
    if rs.photo_static is not None:
        solver._photo_static = rs.photo_static
    integ = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
    solver.naming_solver(para)
    print(f"[chem] setup {time.time() - t0:.1f}s; nz={nz} ni={ni} photo={cfg.use_photo}; "
          f"warming up runner ...", flush=True)
    tw = time.time()
    _ = integ(rs)
    print(f"[chem] warm-up converge {time.time() - tw:.1f}s", flush=True)

    # Warm-capped twin runner for the mutation path. OuterLoop._Statics snapshots
    # int(cfg.count_max) at _ensure_runner time, so the temporary mutation is safe: the
    # smaller cap is frozen into integ_warm's while_loop and cfg is restored right after.
    # Host-side closure construction only -- no extra XLA compile (the retrieval traces
    # integ_warm._runner inside its own jitted evaluators, exactly like integ._runner).
    if warm_count_max != int(cfg.count_max):
        _cold_cap = int(cfg.count_max)
        cfg.count_max = warm_count_max
        integ_warm = outer_loop.OuterLoop(solver, op.Output(cfg=cfg), cfg=cfg)
        integ_warm._ensure_runner(var, atm)
        cfg.count_max = _cold_cap
    else:
        integ_warm = integ

    # --- runner-carry budget/scheme seeding --------------------------------
    # The runner's termination budget and diffusion-scheme blend live on the
    # CARRY, not the statics: _pack_state_from_runstate seeds count_min_dyn /
    # count_max_dyn / runtime_dyn from the packing runner's statics and
    # hybrid_use_vm = 1.0 iff use_vm_mol (upwind / phase 0), and the runner's
    # cond_fn/_real_terminate read ONLY those carry fields (the hybrid phase
    # flip mutates them mid-run). state0 below is packed ONCE under the COLD
    # statics, so every per-proposal solve must re-seed these for the runner
    # that consumes it -- otherwise the warm twin runs to the cold count_max
    # (its warm_count_max cap silently unbound) and, with the hybrid default
    # on, every warm continuation restarts in upwind phase 0 and exhausts the
    # warm cap before the phase flip's budget extension (~count+2000) can
    # certify. Warm continuations start from a column already converged on
    # the CENTRAL-difference operator (a completed hybrid run always ends in
    # phase 1), so they continue on that operator: same fixed point, no
    # phase-0 re-preconditioning. Pure-upwind configs (use_vm_mol on, hybrid
    # off) keep upwind for warm continuation -- their steady state IS the
    # upwind fixed point.
    cold_count_max = int(cfg.count_max)
    count_min_v = int(cfg.count_min)
    runtime_v = float(cfg.runtime)
    use_vm_mol_v = bool(cfg.use_vm_mol)
    hybrid_v = use_vm_mol_v and bool(getattr(cfg, "use_hybrid_vm_mol", False))
    yconv_cri_v = float(cfg.yconv_cri)
    yconv_min_v = float(cfg.yconv_min)
    slope_cri_v = float(cfg.slope_cri)
    flux_cri_v = float(cfg.flux_cri)
    conv_stall_window_v = int(cfg.conv_stall_window)
    _warm_note = ("; warm continuation pinned to central difference (the "
                  "converged phase-1 operator)" if hybrid_v else "")
    print(f"[chem] diffusion scheme: use_vm_mol={use_vm_mol_v} "
          f"hybrid={hybrid_v}{_warm_note}", flush=True)

    def _runner_carry_seed(init, *, warm_continuation, warm_cap):
        """Re-seed the carry's live termination budget + diffusion blend for the
        runner about to consume ``init`` (see the block comment above)."""
        blend = 1.0 if use_vm_mol_v else 0.0
        if warm_continuation and hybrid_v:
            blend = 0.0   # continue on the converged (phase-1, central) operator
        return init._replace(
            hybrid_use_vm=jnp.float64(blend),
            count_min_dyn=jnp.int32(count_min_v),
            count_max_dyn=jnp.int32(warm_count_max if warm_cap else cold_count_max),
            runtime_dyn=jnp.float64(runtime_v),
        )

    def _conv_normal_at_exit(final):
        """The runner's canonical two-branch certification, recomputed at the exit
        state. Mirrors vulcan_jax.outer_loop._convergence_ok's ``conv_normal``
        (keep in sync with that predicate). True only for a tight- or
        loose-branch certified exit; False when the exit certified via the stall
        fallback or exhausted a count/runtime budget."""
        slope_min = jnp.minimum(
            jnp.min(final.pv.Kzz / (0.1 * final.Hp[:-1]) ** 2), jnp.float64(1e-8))
        slope_min = jnp.maximum(slope_min, jnp.float64(1e-10))
        conv = (((final.longdy < yconv_cri_v) & (final.longdydt < slope_cri_v))
                | ((final.longdy < yconv_min_v) & (final.longdydt < slope_min)))
        return conv & (final.aflux_change < flux_cri_v)

    atm_static = make_atm_static(atm, ni, nz, cfg=integ._cfg)
    state0 = integ._pack_state_from_runstate(rs)
    y0 = state0.y
    Kzz0 = atm_static.Kzz
    pv0 = state0.pv
    T_base = jnp.asarray(np.asarray(atm.Tco, dtype=np.float64))

    # --- on-graph atmosphere rebuild inputs -------------------------------
    # refresh_static packs the runner's own hydrostatic-refresh kernel inputs (pico,
    # gs, Rp, pref anchor, species masses); update_mu_dz_jax(ymix, st) is exactly what
    # the runner fires in-loop every update_frq accepted steps, so seeding the initial
    # carry with it makes step 1 consistent with what the loop maintains thereafter.
    # phys0/spec_atm feed atm_jax._mol_diff, the committed on-graph Dzz(T, M) builder
    # (field-for-field equal to the host make_atm_static for this atm_type; validated
    # in VULCAN-JAX tests/test_atm_jax.py).
    refresh_static = integ._build_refresh_static(var, atm)
    phys0, spec_atm = atm_jax.make_physical_inputs(cfg, var, atm, list(network.species))
    use_vm = bool(spec_atm.use_vm_mol and spec_atm.use_moldiff)
    use_set = bool(spec_atm.use_settling and spec_atm.use_moldiff)

    # --- composition masks for the y0 knobs -------------------------------
    compo = np.asarray(composition.compo_array)
    metal_cols = [config.ATOM_COLS[a] for a in ("O", "C", "N", "S")]
    # Scales every C/N/O/S-bearing species. NOTE (elemental accounting): the hydrogen
    # bound in those molecules (H2O, CH4, NH3, H2S, OH, ...) scales along with them, so
    # this is NOT an exact "metals only, H/He fixed" elemental direction -- standalone
    # H2/He are untouched but elemental H shifts by the bound-H fraction (~0.6% per
    # e-fold of Z at the 10x-solar baseline, growing toward the 100x edge). In
    # abundance_mode="elemental" this is only the initial guess and the exact repair
    # below removes the leakage; in legacy "masks" mode it IS the knob definition.
    metal_mask = jnp.asarray((compo[:, metal_cols].sum(axis=1) > 0).astype(np.float64))
    carbon_mask = jnp.asarray((compo[:, config.ATOM_COLS["C"]] > 0).astype(np.float64))  # C/O proxy
    # fixed-O C/O mode ("co_mode": "fixed_O"): every C atom lives in a C-bearing species,
    # and the O-carriers holding no C (H2O, OH, O2, SO, SO2, NO, ...) are disjoint from
    # them -- the two masks partition all O between "dragged along by C-carriers" and
    # "free to compensate".
    nO_per_species = jnp.asarray(np.asarray(compo[:, config.ATOM_COLS["O"]], dtype=np.float64))
    o_only_mask = jnp.asarray(((compo[:, config.ATOM_COLS["O"]] > 0)
                               & (compo[:, config.ATOM_COLS["C"]] == 0)).astype(np.float64))
    co_fixed_o = str(profile.get("co_mode", "proxy")) == "fixed_O"
    atomic_masses = jnp.asarray(np.asarray(config.ATOMIC_MASSES, dtype=np.float64))
    species_masses = jnp.asarray(np.asarray(compo, dtype=np.float64)) @ atomic_masses  # (ni,)
    # runner's own (ni, n_atoms) composition table, columns in its internal _atom_order --
    # used to rebuild the conserved atom totals (atom_ini) in the runner's exact basis.
    compo_run = jnp.asarray(np.asarray(integ._compo_arr, dtype=np.float64))
    # Legacy-mode opt-in: re-anchor the conserved atom totals to the perturbed column
    # (needed for finite metallicity/C-O steps in "masks" mode; see _prep). Moot in
    # "elemental" mode, where atom_ini is ALWAYS rebuilt from the repaired column.
    reanchor_atom_ini = bool(profile.get("reanchor_atom_ini", False))
    # Opt-in: zero the eddy-diffusion profile entirely (the no-transport equilibrium tier;
    # combine with cfg_overrides={"use_moldiff": False} so Dzz is off too). lnKzz is then inert.
    zero_kzz = bool(profile.get("zero_Kzz", False))
    abundance_mode = str(profile.get("abundance_mode", "masks"))
    if abundance_mode not in ("masks", "elemental"):
        raise ValueError(f"abundance_mode={abundance_mode!r}: expected 'masks' or 'elemental'")

    # --- exact-elemental targets + repair tables (abundance_mode="elemental") ----
    # Baseline column-integrated elemental totals from the pristine y0 (which sums to
    # M_base per layer by construction: FastChem mixing ratios x n_0). Targets are
    # RATIOS to elemental H; absolute densities follow from sum_i n_i = M.
    elem_pairs = [(e, sp) for e, sp in _ELEMENTAL_REPAIR
                  if sp in sidx and compo[:, config.ATOM_COLS[e]].sum() > 0]
    _y0_np = np.asarray(y0, dtype=np.float64)
    _elem_cols = [config.ATOM_COLS["H"]] + [config.ATOM_COLS[e] for e, _ in elem_pairs]
    # (ni, 1+nrep) atoms-per-molecule for [H, He, O, C, N, S]-as-present
    E_mat = jnp.asarray(np.asarray(compo[:, _elem_cols], dtype=np.float64))
    rep_cols = np.asarray([sidx[sp] for _, sp in elem_pairs], dtype=np.int64)
    A0 = _y0_np @ np.asarray(compo[:, _elem_cols], dtype=np.float64)  # per-layer (nz, 1+nrep)
    A0 = A0.sum(axis=0)                                               # column totals
    if abundance_mode == "elemental":
        missing = [sp for _, sp in _ELEMENTAL_REPAIR if sp not in sidx]
        if not elem_pairs:
            raise RuntimeError("elemental mode: no repair species found in the network")
        R0_ratios = A0[1:] / A0[0]
        # per-element theta-scaling kind: He fixed; O/N/S x Z; C x Z e^{c_o}
        _zk = np.asarray([0.0 if e == "He" else 1.0 for e, _ in elem_pairs])
        _ck = np.asarray([1.0 if e == "C" else 0.0 for e, _ in elem_pairs])
        zscale_kind = jnp.asarray(_zk)
        cscale_kind = jnp.asarray(_ck)
        R0_j = jnp.asarray(R0_ratios)
        print("[chem] elemental mode: exact column ratios to H via repair species "
              f"{[sp for _, sp in elem_pairs]}"
              + (f" (absent: {missing})" if missing else "")
              + "; baseline C/O = "
              f"{A0[1 + [e for e, _ in elem_pairs].index('C')] / A0[1 + [e for e, _ in elem_pairs].index('O')]:.4f}",
              flush=True)

    co_bz_bound = float("inf")   # proxy mode has no b_z compensation -> no bound
    if co_fixed_o:
        # Build-time diagnostics for the fixed-O C/O knob: baseline C/O, how much of the
        # column's O sits in C-carriers (sets the b_z compensation), and the worst-layer
        # O-only share (b_z blows up where O-only carriers vanish).
        _y0n = _y0_np
        _nC = np.asarray(compo[:, config.ATOM_COLS["C"]], dtype=np.float64)
        _nO = np.asarray(compo[:, config.ATOM_COLS["O"]], dtype=np.float64)
        _mC = np.asarray(carbon_mask); _mOo = np.asarray(o_only_mask)
        _C_tot = float((_y0n * _nC[None, :]).sum())
        _O_tot = float((_y0n * _nO[None, :]).sum())
        _OC_z = (_y0n * (_nO * _mC)[None, :]).sum(axis=1)
        _OO_z = (_y0n * (_nO * _mOo)[None, :]).sum(axis=1)
        with np.errstate(divide="ignore"):
            co_bz_bound = float(np.log(1.0 + np.min(_OO_z / _OC_z)))
        print(f"[chem] fixed-O C/O knob: baseline C/O = {_C_tot/_O_tot:.4f} "
              f"(ln = {np.log(_C_tot/_O_tot):+.4f}); O-in-C-carriers share "
              f"median {np.median(_OC_z/(_OC_z+_OO_z)):.3f}, max {np.max(_OC_z/(_OC_z+_OO_z)):.3f} "
              f"(b_z stays positive for c_o < {co_bz_bound:.2f})", flush=True)

    rep_cols_j = jnp.asarray(rep_cols)

    def _elemental_project(y_in, M, lnZ, c_o):
        """Renormalize to sum_i n_i = M and repair the column elemental ratios exactly.

        y_in : (nz, ni) guessed absolute densities. Returns (y_out, min_adj) where
        y_out rows sum to M and the column ratios-to-H equal the theta targets to the
        fixed-iteration residual (~1e-8 rel; audit_init measures it), and min_adj is
        the smallest per-species repair factor (must stay > 0 for a physical column;
        it is ~1 +/- the mask-leakage scale everywhere in the shipped prior boxes).
        """
        targets = R0_j * jnp.exp(lnZ * zscale_kind + c_o * cscale_kind)  # (nrep,)
        y = y_in * (M / jnp.sum(y_in, axis=1))[:, None]
        min_adj = jnp.asarray(1.0, dtype=jnp.float64)
        for _ in range(_ELEMENTAL_REPAIR_ITERS):
            A = jnp.einsum("zi,ie->e", y, E_mat)                # [H, e1..] column totals
            col_tot = jnp.sum(y[:, rep_cols_j], axis=0)         # (nrep,) adjuster columns
            B = E_mat[rep_cols_j, :].T * col_tot[None, :]       # (1+nrep, nrep)
            Msys = B[1:, :] - targets[:, None] * B[0:1, :]
            rhs = targets * A[0] - A[1:]
            alpha = jnp.linalg.solve(Msys, rhs)                 # (nrep,) additive factors
            min_adj = jnp.minimum(min_adj, jnp.min(1.0 + alpha))
            scale_vec = jnp.ones(ni, dtype=jnp.float64).at[rep_cols_j].set(1.0 + alpha)
            y = y * scale_vec[None, :]
            y = y * (M / jnp.sum(y, axis=1))[:, None]
        return y, min_adj

    def _guess_y0(lnZ, c_o, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Mask-scaled initial-composition GUESS (shared by _prep and audit_init).

        From the baseline (warm_y=None) the full lnZ/c_o is applied; in continuation
        only the increments (lnZ - lnZ_ref, c_o - c_o_ref) are, so a large absolute
        perturbation is reached by small steps from a nearby converged state."""
        c_o_inc = c_o - c_o_ref     # incremental C/O relative to the warm state
        base = y0 if warm_y is None else warm_y
        if co_fixed_o:
            # c_o == delta ln(C/O) at fixed O, EXACTLY, layer by layer: scaling every
            # C-bearing species by e^c multiplies each layer's C total by e^c (all C lives
            # there); the O those species drag along (CO, CO2, ...) is compensated by
            # scaling the O-only carriers by b_z = 1 + (1 - e^c)*O_Ccarriers/O_Oonly, which
            # keeps each layer's O total invariant. Leakage is only into H (via H2O's H,
            # ~1e-3 relative per unit c) and N/S (via trace NO/SO/SO2 in the equilibrium
            # init). Smooth in c_o -> AD-safe; b_z > 0 within the range printed at build.
            OC_z = (base * (nO_per_species * carbon_mask)[None, :]).sum(axis=1)
            OO_z = (base * (nO_per_species * o_only_mask)[None, :]).sum(axis=1)
            b_z = 1.0 + (1.0 - jnp.exp(c_o_inc)) * OC_z / OO_z                # (nz,)
            cofac = jnp.where(carbon_mask[None, :] > 0, jnp.exp(c_o_inc), 1.0)  # (1, ni)
            cofac = jnp.where(o_only_mask[None, :] > 0, b_z[:, None], cofac)  # (nz, ni)
            y0p = base * jnp.exp((lnZ - lnZ_ref) * metal_mask)[None, :] * cofac
        else:
            scale = jnp.exp((lnZ - lnZ_ref) * metal_mask + c_o_inc * carbon_mask)  # (ni,)
            y0p = base * scale[None, :]
        return y0p

    def _prep(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Build the perturbed initial runner state + atm for theta=[lnZ, c_o, lnKzz, T...].

        Continuation: pass warm_y = a previously-CONVERGED y (and its lnZ_ref / c_o_ref)
        to warm-start from there instead of the fixed baseline y0 (see _guess_y0). In
        "elemental" mode the guess is then projected onto the EXACT theta targets (which
        depend on theta only), so the conserved inventory is path-independent; in legacy
        "masks" mode the incremental scaling IS the map (avoids the runner's
        snap-back-to-baseline, and avoids double-applying C/O to a warm_y that already
        carries it -- a fixed-C/O metallicity march passes c_o_ref = c_o)."""
        lnZ, c_o, lnKzz = theta[0], theta[1], theta[2]

        # Temperature: default is the validated uniform T shift theta[3]; with a tp_eval
        # hook the full differentiable T-P profile theta[3:3+n_tp_params] is used instead.
        # Either way rate constants are rebuilt ON-GRAPH (rates_jax), with n_0 = pco/(kb T),
        # Ti, and the pv carry (fig_so2_temperature pattern).
        if tp_eval is None:
            T = T_base + theta[3]
        else:
            T = tp_eval(theta[3:3 + n_tp_params], p_bar_j)
        M = pco / (kb * T)
        # Honor cfg.use_lowT_limit_rates (2026-07-19): build_rate_array
        # defaults use_lowT_caps=False, so a config with the flag ON was
        # silently solved with uncapped low-T rates (no shipped config sets
        # it, but a silent config-ignore violates the loud-errors rule; the
        # adjoint caller in vulcan-jwst-tool already passed it explicitly).
        k_arr = rates_jax.build_rate_array(
            network, T, M, nasa9, remove_list,
            use_lowT_caps=bool(cfg.use_lowT_limit_rates))
        Ti = 0.5 * (T[:-1] + T[1:])
        Kzz_eff = Kzz0 * 0.0 if zero_kzz else Kzz0 * jnp.exp(lnKzz)

        y0p = _guess_y0(lnZ, c_o, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)

        if abundance_mode == "elemental":
            # Exact construction: sum_i n_i = M per layer AND exact column elemental
            # ratios; atom_ini rebuilt from the repaired column so the conservation
            # anchor matches the actual initial gas (no reanchor knob needed).
            y0p, _min_adj = _elemental_project(y0p, M, lnZ, c_o)
            ymix0 = y0p / M[:, None]
            atom_ini_new = jnp.einsum("zi,ia->a", y0p, compo_run)  # runner atom order
            pv_T = pv0._replace(n_0=M, r_Tco=T, Kzz=Kzz_eff, atom_ini=atom_ini_new)
        else:
            ymix0 = y0p / jnp.sum(y0p, axis=1, keepdims=True)
            # Legacy masks mode, opt-in: re-anchor the conserved atom totals to the
            # PERTURBED column. The runner's atom-conservation (_compute_atom_loss)
            # measures drift from pv.atom_ini; if atom_ini stays at the baked baseline,
            # finite metallicity/C-O steps that exceed the loss threshold get the added
            # metals driven back to baseline (the "snap to baseline" seen for
            # lnZ >= +0.10). NOTE: y0p is NOT renormalized to M here (published-demo
            # behavior); the runner's first hydrostatic renorm rescales it, so atom_ini
            # computed from the raw y0p mismatches the post-renorm gas by the scaled
            # metal fraction. The "elemental" mode removes this inconsistency.
            if reanchor_atom_ini:
                atom_ini_new = jnp.einsum("zi,ia->a", y0p, compo_run)
                pv_T = pv0._replace(n_0=M, r_Tco=T, Kzz=Kzz_eff, atom_ini=atom_ini_new)
            else:
                pv_T = pv0._replace(n_0=M, r_Tco=T, Kzz=Kzz_eff)

        # --- atmospheric structure at the proposed T + composition --------
        # Hydrostatic geometry via the runner's OWN refresh kernel (so the initial
        # carry equals what the in-loop refresh maintains); Dzz/vm/vs via the
        # committed on-graph builder at the proposed (T, M). The runner splices the
        # carry geometry into every step and recomputes vm in-loop from atm.Dzz, so
        # rebuilding Dzz here fixes the whole molecular-diffusion channel.
        refresh_lane = refresh_static._replace(Tco=T)
        mu_i, g_i, Hp_i, dz_i, zco_i, dzi_i, Hpi_i = atm_refresh_mod.update_mu_dz_jax(
            ymix0, refresh_lane)
        Dzz_new, _Dzz_cen, vm_new = atm_jax._mol_diff(
            phys0._replace(Tco=T), spec_atm, M, g_i, Hp_i, dz_i)
        if not use_vm:
            vm_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        if use_set:
            _na, _a, _b = _VISCOSITY_TABLE[spec_atm.atm_base]
            vs_new = settling_velocity_jax(_na, _a, _b, T, g_i, spec_atm.settle_coeff)
        else:
            vs_new = jnp.zeros((nz - 1, ni), dtype=jnp.float64)
        pv_T = pv_T._replace(r_Dzz_top=Dzz_new[-1])

        # --- condensation at the proposed T ---------------------------------
        # Rebuild every T/structure-dependent condensation array from the SAME
        # live temperature and structure the chemistry uses (saturation number
        # densities, Dg growth terms from the live Dzz, relax inputs, NH3
        # cold-trap argmin, fix-species sat-mix rows) and splice them into the
        # ProfileVars carry the runner reads each step. No baseline-frozen
        # condensation table survives into a live-T solve.
        if conden_spec is not None:
            cprof = conden_mod.build_conden_profile(conden_spec, T, pco, M, Dzz_new)
            pv_T = pv_T._replace(
                c_Dg_per_re=cprof.Dg_per_re,
                c_sat_n_per_re=cprof.sat_n_per_re,
                c_h2o_Dg=cprof.h2o_Dg,
                c_h2o_sat=cprof.h2o_sat,
                c_nh3_Dg=cprof.nh3_Dg,
                c_nh3_sat=cprof.nh3_sat,
                c_nh3_conden_top=cprof.nh3_conden_top,
                fix_species_sat_mix=cprof.fix_species_sat_mix,
            )
        atm_T = atm_static._replace(Tco=T, Ti=Ti, M=M, Kzz=Kzz_eff, Dzz=Dzz_new,
                                    vm=vm_new, vs=vs_new, g=g_i, dzi=dzi_i, Hpi=Hpi_i)

        init = state0._replace(y=y0p, ymix=ymix0, k_arr=k_arr, pv=pv_T,
                               mu=mu_i, g=g_i, Hp=Hp_i, dz=dz_i, zco=zco_i,
                               dzi=dzi_i, Hpi=Hpi_i, vs=vs_new)
        return init, atm_T

    def converged_ymix(theta):
        """Re-converge the WASP-39b column under theta=[lnZ, c_o, lnKzz, T...].

        Returns linear VMR (nz, ni). Differentiable end-to-end via forward-mode.
        """
        init, atm_T = _prep(theta)
        init = _runner_carry_seed(init, warm_continuation=False, warm_cap=False)
        final = integ._runner(init, atm_T)
        return final.y / jnp.sum(final.y, axis=1, keepdims=True)

    def run_diag(theta, return_atm=False):
        """Diagnostic twin of converged_ymix: returns (final_runner_state, init_state).

        Lets a caller inspect convergence (longdy/accept_count/t), whether the runner
        actually moved off the init, and whether the metallicity perturbation changed the
        conserved element totals. Not on any AD path.

        ``return_atm=True`` additionally returns the theta-dependent AtmStatic the
        runner was actually driven with (``atm_T`` from ``_prep``: live Tco/Ti/M/
        Kzz/Dzz/vm/vs + refreshed geometry) -- the ``atm`` a reverse-mode adjoint
        call (vulcan_jax.steady_state_grad) must linearize around. Without it a
        caller can only reach the setup-time baseline AtmStatic, which is the
        WRONG operating point whenever theta carries a T-P or Kzz offset (the
        scope audit flags that as stale_geometry). Added 2026-07-15 for the
        jwst-tool adjoint diagnostics; the (final, init) two-tuple contract is
        unchanged for existing callers."""
        init, atm_T = _prep(theta)
        init = _runner_carry_seed(init, warm_continuation=False, warm_cap=False)
        final = integ._runner(init, atm_T)
        return (final, init, atm_T) if return_atm else (final, init)

    def prep_pv(theta):
        """The initial-carry ProfileVars for ``theta`` -- the per-proposal arrays
        (n_0, Kzz, atom_ini, and with condensation on the live-rebuilt c_* conden
        arrays + fix_species_sat_mix) WITHOUT running the solver. Pure function
        of theta; jit/vmap/jvp-traceable. Diagnostics/tests only."""
        init, _atm_T = _prep(jnp.asarray(theta, dtype=jnp.float64))
        return init.pv

    def converged_y(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0, return_diag=False,
                    warm_cap=False, return_longdy=False, return_conv_diag=False):
        """Converged ABSOLUTE number densities y (nz, ni), with optional continuation
        warm-start (warm_y at lnZ_ref / c_o_ref). Differentiable via forward-mode w.r.t. theta.
        The SO2 column number density is then jnp.sum(y[:, so2] * dz); jvp gives both y (for
        chaining the next continuation step) and its lnZ-derivative (the index) in one pass.

        The carry's live termination budget + diffusion blend are re-seeded per solve
        (``_runner_carry_seed``): the warm-capped path gets count_max_dyn=warm_count_max,
        and under the hybrid vm_mol default a warm continuation runs on the
        central-difference operator (the converged phase-1 operator) instead of
        re-entering upwind phase 0.

        ``return_diag=True`` additionally returns ``final.accept_count`` (int32 scalar) so a
        caller can detect a count_max-exhausted (not-actually-converged) solve without
        re-deriving the runner's own termination ladder. AD-safe inside a forward-mode jvp
        chain: accept_count rides the runner's primal carry (no extra work) and, being
        integer-valued, carries no tangent -- callers on an AD path should wrap it in
        ``stop_gradient`` and cast, and must not differentiate w.r.t. it.

        ``warm_cap=True`` runs the warm-capped twin runner (count_max=warm_count_max) --
        the SMC mutation path, where a proposal that hasn't converged in warm_count_max
        steps is rejected rather than marched to the full cold cap.

        ``return_longdy=True`` returns ``(y, accept_count, final.longdy)``. accept_count alone
        is NOT a convergence test: the hybrid vm_mol phase-flip (and the stall fallback)
        terminate the runner EARLY (accept_count ~ count_min+2000 << count_max) even when the
        column has not settled, so ``accept_count < count_max`` can be True for a non-steady
        state. ``longdy`` is the runner's own convergence metric -- gate it against ``yconv_min``
        (converged states have ``longdy < yconv_min``) to catch that early-terminate case.

        ``return_conv_diag=True`` returns ``(y, ConvDiag)`` -- the full per-solve
        convergence diagnostics (accept_count, longdy, longdydt, count_since_new_min,
        conv_normal), all free reads off the primal carry. This is the SMC hot-path
        surface: ``conv_normal`` is the runner's canonical certification recomputed at
        the exit state, so a stall-fallback or budget exit reads False even when
        ``longdy < yconv_min``. Supersedes return_longdy/return_diag for new callers."""
        init, atm_T = _prep(jnp.asarray(theta, dtype=jnp.float64), warm_y=warm_y,
                            lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
        init = _runner_carry_seed(init, warm_continuation=warm_y is not None,
                                  warm_cap=warm_cap)
        final = (integ_warm if warm_cap else integ)._runner(init, atm_T)
        if return_conv_diag:
            return final.y, ConvDiag(
                accept_count=final.accept_count,
                longdy=final.longdy,
                longdydt=final.longdydt,
                count_since_new_min=final.count_since_new_min,
                conv_normal=_conv_normal_at_exit(final),
            )
        if return_longdy:
            return final.y, final.accept_count, final.longdy
        if return_diag:
            return final.y, final.accept_count
        return final.y

    def audit_init(theta, warm_y=None, lnZ_ref=0.0, c_o_ref=0.0):
        """Host-side audit of the initial column built for ``theta`` (not on any AD path).

        Returns a dict with the quantities the science review asked to see verified at
        every retrieval point: relative density-closure error max_z |sum_i n_i - M|/M,
        the achieved-vs-target column elemental ratios (elemental mode) or the raw
        achieved ratios (masks mode), the achieved dln(C/O) vs theta, the smallest
        elemental-repair factor (elemental mode; must be > 0), and the atom_ini
        consistency |atoms(y_init) - atom_ini|/atom_ini in the runner's atom basis.
        """
        th = jnp.asarray(theta, dtype=jnp.float64)
        init, _atm_T = _prep(th, warm_y=warm_y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
        y = np.asarray(init.y, dtype=np.float64)
        Mn = np.asarray(init.pv.n_0, dtype=np.float64)
        A = (y @ np.asarray(compo[:, _elem_cols], dtype=np.float64)).sum(axis=0)
        ratios = A[1:] / A[0]
        names = [e for e, _ in elem_pairs]
        out = {
            "density_closure_max_rel": float(np.max(np.abs(y.sum(axis=1) - Mn) / Mn)),
            "ratios_to_H": dict(zip(names, ratios.tolist())),
            "baseline_ratios_to_H": dict(zip(names, (A0[1:] / A0[0]).tolist())),
        }
        if "C" in names and "O" in names:
            r_now = ratios[names.index("C")] / ratios[names.index("O")]
            r_base = (A0[1:] / A0[0])[names.index("C")] / (A0[1:] / A0[0])[names.index("O")]
            out["dln_CO_achieved"] = float(np.log(r_now / r_base))
        if abundance_mode == "elemental":
            tg = np.asarray(R0_j) * np.exp(float(th[0]) * np.asarray(zscale_kind)
                                           + float(th[1]) * np.asarray(cscale_kind))
            out["target_ratios_to_H"] = dict(zip(names, tg.tolist()))
            out["ratio_max_rel_err"] = float(np.max(np.abs(ratios / tg - 1.0)))
            # Re-run the projection from the raw GUESS to expose the actual repair
            # magnitude (projecting the already-repaired y would always report ~1).
            y_guess = _guess_y0(th[0], th[1], warm_y=warm_y,
                                lnZ_ref=lnZ_ref, c_o_ref=c_o_ref)
            _yg, min_adj = _elemental_project(y_guess, jnp.asarray(Mn), th[0], th[1])
            out["min_repair_factor"] = float(min_adj)
        ai = np.asarray(init.pv.atom_ini, dtype=np.float64)
        a_run = y @ np.asarray(integ._compo_arr, dtype=np.float64)
        out["atom_ini_max_rel_err"] = float(np.max(np.abs(a_run.sum(axis=0) - ai) / ai))
        return out

    return SimpleNamespace(
        converged_ymix=converged_ymix,
        run_diag=run_diag,
        converged_y=converged_y,
        conv_normal_at_exit=_conv_normal_at_exit,  # canonical certification of a
        #                                            raw run_diag final carry --
        #                                            exported 2026-07-19 so
        #                                            adjoint callers can gate on
        #                                            conv_normal, not longdy alone
        audit_init=audit_init,
        conden_spec=conden_spec,   # static conden metadata (None when conden off)
        prep_pv=prep_pv,           # theta -> initial ProfileVars (no solve; tests)
        _integ=integ,              # the OuterLoop (baked statics access; tests only)
        abundance_mode=abundance_mode,
        co_bz_bound=co_bz_bound,   # fixed-O knob validity: b_z > 0 iff c_o < this (baseline column)
        y0=np.asarray(y0, dtype=np.float64),   # baked baseline column (warm-start fallback)
        compo_array=compo,
        T_base=np.asarray(T_base),
        p_bar=p_bar,
        dz=np.asarray(atm.dz, dtype=np.float64),   # layer thickness (cm); for n0*dz column weighting
        sidx=sidx,
        species_masses=species_masses,
        nz=nz, ni=ni,
        count_max=int(cfg.count_max),   # the resolved (profile-overridden or module-default) cap
        warm_count_max=warm_count_max,  # mutation-path cap (== count_max when no twin runner)
        yconv_min=float(cfg.yconv_min), # loose convergence gate: a converged solve has longdy<this
        yconv_cri=yconv_cri_v,          # tight convergence branch threshold
        slope_cri=slope_cri_v,          # tight-branch longdydt threshold
        conv_stall_window=conv_stall_window_v,  # stall-fallback lookback (accepted steps)
        use_vm_mol=use_vm_mol_v,        # resolved COLD diffusion scheme (upwind on/off)
        use_hybrid_vm_mol=hybrid_v,     # resolved hybrid phase-flip (warm continuation runs central)
    )
