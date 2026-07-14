"""Live-T(P) condensation through the chemistry model.

Validates the 2026-07-13 on-graph condensation rebuild end-to-end:

* the ProfileVars conden arrays in a live solve correspond to the PROPOSED
  temperature, never the baseline structural one;
* at theta whose T(P) equals the structural baseline, the rebuilt arrays are
  bit-compatible with the host-baked CondenStatic (previous-isothermal parity:
  identical runner input state => identical solve);
* an isothermal AND a Guillot-profile condensation run complete end-to-end
  (terminating at a physical runtime cap -- this anchor-free synthetic column
  has no reachable full thermochemical equilibrium; see _CFG_OVERRIDES), move
  mass into the condensate, and relax supersaturation toward the LIVE
  saturation curve;
* jit / vmap / forward-mode jvp execute without tracer errors, and the jvp
  through a condensing steady state matches warm-started centered finite
  differences away from active-layer switches.

Setup mirrors VULCAN-JAX's condensation runtime test: a small synthetic
column on the production SNCHO network (its one condensation reaction is
S8 -> S8_l_s), const_mix init (offline, no FastChem), photochemistry off.
Convergence uses the upstream conden-window + whole-column fix_species pin
(same methodology jwst_tool.forward.CONDEN_CFG ships): without the pin the
steady state is transport-limited -- the upper S8 reservoir drains through
the condensation front on the Kzz timescale (~1e9 s) while dt stays capped
at the front's condensation timescale, so every solve exhausts count_max.
Builds two chem models (~1-2 min each, compile-dominated); every test skips
cleanly when the chem stack is unavailable (same policy as test_warm_extrap).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Structural/thermal setup: T where the S8 saturation boundary sits inside the
# grid for an S8 VMR of 1e-4 (sat_mix ~ 1.5e-5 at 5 bar, ~7.4e-3 at 0.01 bar).
T_STRUCT = 400.0
S8_VMR = 1.0e-4
NZ = 32
STOP_CONDEN = 1.0e5   # conden window end; S8/S8_l_s pinned whole-column after

_CFG_OVERRIDES = {
    "atm_type": "isothermal",
    "Tiso": T_STRUCT,
    "P_b": 5.0e6,          # 5 bar
    "P_t": 1.0e4,          # 0.01 bar
    "Kzz_prof": "const",
    "const_Kzz": 1.0e7,
    "use_photo": False,
    "use_moldiff": True,
    "ini_mix": "const_mix",
    # He/H2O/CO/N2/H2S are the elemental-repair species and must be present.
    "const_mix": {"H2": 0.85877, "He": 0.14, "H2O": 5e-4, "CO": 5e-4,
                  "N2": 1e-4, "H2S": 1e-4, "S8": S8_VMR},
    "use_condense": True,
    "condense_sp": ["S8"],
    "non_gas_sp": ["S8_l_s"],
    # Rainout-sized particles: smaller radii stiffen the growth term (~1/r^2)
    # until dt is pinned at the condensation-front timescale.
    "r_p": {"S8_l_s": 5.0e-3},   # 50 um
    "rho_p": {"S8_l_s": 2.07},   # orthorhombic sulfur, g/cm^3
    "use_relax": [],
    "use_settling": False,
    "fix_species": ["S8", "S8_l_s"],
    "fix_species_from_coldtrap_lev": False,   # whole-column pin (isothermal-safe)
    "start_conden_time": 0.0,
    "stop_conden_time": STOP_CONDEN,
    # Convergence mixing-ratio floor raised from the 1e-20 default: at 400 K
    # the glacial N2 -> NH3 kinetics leave NH3 drifting at ~6e-19 VMR forever
    # (a cold-atmosphere property, independent of condensation), which would
    # gate longdy at ~0.9 indefinitely. Species below 1e-15 VMR are far
    # beneath any observable/RT relevance.
    "mtol_conv": 1.0e-15,
    # Default heavy-hydrocarbon offenders + the trace sulfur allotropes:
    # against a pinned S8, S (ppq), S3 (ppt), S4 (ppb), and S2 (~1e-7)
    # re-equilibrate on cold-top thermal timescales measured at >=1e15 s --
    # physically unreachable, and all far below RT relevance (none is an RT
    # molecule; the observable sulfur species SO2/H2S/SO stay in the gate).
    "conver_ignore": ["C6H6", "C2H2", "C6H5", "C2H", "C2H4", "C2H5", "C2H6",
                      "C3H2", "C3H3", "C4H5", "CH2NH", "CH3NH2", "H2CCO",
                      "S", "S2", "S3", "S4"],
    # With the allotropes ignored the gate could fire before the conden
    # window ends (measured: t ~ 2.3e3 s), leaving a half-rained S8 column.
    # Bound certification from below so the window + whole-column pin always
    # complete first: the certified S8 state is then deterministic
    # (end-of-window rainout, drizzle truncated at STOP_CONDEN).
    "trun_min": STOP_CONDEN,
    # Physical integration cap. This anchor-free synthetic column (400 K,
    # no photochemistry, no hot deep boundary) has NO reachable longdy
    # steady state: after the pin, well-mixed CO2 at ~1.7e-8 VMR keeps
    # creeping toward thermochemical equilibrium at ~18% per time-doubling
    # even at t = 1.6e15 s (equilibration ~1e17+ s -- older than any
    # planet; real columns are anchored by a hot interior or photolysis
    # sources, cf. the converged WASP-107b tool run). Upstream VULCAN's own
    # mechanism for such regimes is the runtime cap: integrate to a
    # physically-sufficient time and take that state. 1e14 s (~3 Myr) is
    # far beyond every transport/condensation timescale in the column.
    "runtime": 1.0e14,
}


def _profile(**extra):
    from retrieval_framework.forward import config

    prof = dict(config.SMOKE)
    prof.update(
        nz=NZ,
        use_photo=False,
        yconv_cri=1.0e-2,
        count_max=5000,
        abundance_mode="elemental",
        cfg_overrides=dict(_CFG_OVERRIDES),
    )
    prof.update(extra)
    return prof


@pytest.fixture(scope="module")
def stack():
    """Import the chem stack (vulcan_chem BEFORE jax/exojax) or skip."""
    try:
        from retrieval_framework.forward import vulcan_chem  # noqa: F401
        import jax
        import jax.numpy as jnp
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"chem stack unavailable: {e}")
    return vulcan_chem, jax, jnp


@pytest.fixture(scope="module")
def chem_iso(stack):
    """Isothermal tp_eval model: theta = [lnZ, c_o, lnKzz, T]."""
    vulcan_chem, jax, jnp = stack

    def tp_eval(tp, p_bar):
        return jnp.zeros_like(jnp.asarray(p_bar)) + tp[0]

    try:
        return vulcan_chem.build_chem_model(_profile(), tp_eval=tp_eval,
                                            n_tp_params=1)
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"chem model build failed: {e}")


def _theta(T):
    return np.array([0.0, 0.0, 0.0, float(T)], dtype=np.float64)


def _s8_cols(chem):
    return chem.sidx["S8"], chem.sidx["S8_l_s"]


def _sat_n_s8(jnp, T):
    from vulcan_jax.atm_setup import sat_p_jax
    from vulcan_jax.phy_const import kb

    return sat_p_jax("S8", jnp.asarray(T)) / kb / jnp.asarray(T)


def test_conden_spec_extracted(chem_iso):
    spec = chem_iso.conden_spec
    assert spec is not None
    assert spec.gas_names == ("S8",)
    assert spec.coeff_per_re[0] > 0.0
    assert not spec.h2o_active and not spec.nh3_active


def test_live_arrays_follow_proposed_temperature(stack, chem_iso):
    """No baseline conden array survives a live solve: the carry's saturation
    row equals sat(T_live) exactly, and differs from the baseline-T one."""
    _, jax, jnp = stack
    pv_400 = chem_iso.prep_pv(_theta(400.0))
    pv_430 = chem_iso.prep_pv(_theta(430.0))
    sat_400 = np.asarray(pv_400.c_sat_n_per_re[0])
    sat_430 = np.asarray(pv_430.c_sat_n_per_re[0])
    want_400 = np.asarray(_sat_n_s8(jnp, np.full(NZ, 400.0)))
    want_430 = np.asarray(_sat_n_s8(jnp, np.full(NZ, 430.0)))
    np.testing.assert_allclose(sat_400, want_400, rtol=1e-14)
    np.testing.assert_allclose(sat_430, want_430, rtol=1e-14)
    assert np.all(sat_430 > sat_400)  # warmer => higher saturation density
    # boundary moves: supersaturated layer count changes at fixed abundance
    y_s8 = S8_VMR * np.asarray(pv_400.n_0)
    assert (y_s8 > sat_400).sum() != (S8_VMR * np.asarray(pv_430.n_0) > sat_430).sum()


def test_baseline_T_parity_with_baked_static(chem_iso):
    """theta at the structural temperature reproduces the host-baked conden
    arrays bit-compatibly => the runner input equals the pre-change isothermal
    path and previous numerical parity is retained by construction."""
    baked = chem_iso._integ._conden_static
    pv = chem_iso.prep_pv(_theta(T_STRUCT))
    np.testing.assert_allclose(np.asarray(pv.c_sat_n_per_re),
                               np.asarray(baked.sat_n_per_re), rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(np.asarray(pv.c_Dg_per_re),
                               np.asarray(baked.Dg_per_re), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(pv.n_0),
                               np.asarray(baked.n_0), rtol=1e-12, atol=0.0)


def test_isothermal_condensation_converges_and_rains_out(stack, chem_iso):
    """End-to-end isothermal condensing solve: completes within the step
    budget (terminating at the physical runtime cap, NOT by exhausting
    count_max -- see the `runtime` note in _CFG_OVERRIDES), with the pin
    activated and the S8 rainout observables settled."""
    _, jax, jnp = stack
    s8, s8_ls = _s8_cols(chem_iso)
    final, _init = chem_iso.run_diag(jnp.asarray(_theta(T_STRUCT)))
    assert int(final.accept_count) < chem_iso.count_max, \
        "condensing solve must complete within the step budget"
    assert float(final.t) >= float(_CFG_OVERRIDES["runtime"]), \
        "solve must reach the physical runtime cap"
    assert bool(final.fix_species_started), \
        "the conden-window + fix pin must have activated"
    y = np.asarray(final.y)
    assert np.all(np.isfinite(y))
    assert y[:, s8_ls].sum() > 0.0, "mass must move into the condensate"
    # Supersaturation relaxed toward the live saturation curve. The pin
    # freezes the state at the end of the conden window, where the front
    # layer still carries residual supersaturation (~1.6 here); the initial
    # column starts ~6.7x supersaturated at the bottom.
    pv = chem_iso.prep_pv(_theta(T_STRUCT))
    sat = np.asarray(pv.c_sat_n_per_re[0])
    init_ratio = (S8_VMR * np.asarray(pv.n_0) / sat).max()
    final_ratio = (y[:, s8] / sat).max()
    assert init_ratio > 2.0, "test setup must start supersaturated"
    assert final_ratio < 2.0, f"gas must relax to ~saturation (got {final_ratio:.2f})"


# The Guillot column (346-586 K, cold condensing top) caps dt at ~4e5 s
# (measured; vs ~4e10 for the isothermal column -- a stiffness property of
# the cold-top chemistry, identical in upstream's scheme), so the isothermal
# model's 1e14 s runtime is unreachable. 1e9 s is still four decades past
# the conden window + pin (1e5 s), which is what this test certifies.
GUILLOT_RUNTIME = 1.0e9
GUILLOT_COUNT_MAX = 15000


@pytest.fixture(scope="module")
def chem_guillot(stack):
    """Guillot tp_eval model (exojax parameterization, jwst-tool pattern):
    theta = [lnZ, c_o, lnKzz, Tirr, Tint, log_kappa, log_gamma]."""
    vulcan_chem, jax, jnp = stack
    try:
        from exojax.atm.atmprof import atmprof_Guillot
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"exojax unavailable: {e}")
    gs_cgs = 1000.0

    def tp_eval(tp, p_bar):
        p = jnp.asarray(p_bar)
        kappa, gamma = 10.0 ** tp[2], 10.0 ** tp[3]
        return atmprof_Guillot(p, gs_cgs, kappa, gamma, tp[1], tp[0], 0.25)

    try:
        return vulcan_chem.build_chem_model(
            _profile(cfg_overrides=dict(_CFG_OVERRIDES,
                                        # VULCAN derives gs = G*Mp/Rp^2; set Mp/Rp
                                        # to reproduce gs_cgs at the W39b radius.
                                        Rp=1.279 * 7.1492e9,
                                        Mp=gs_cgs * (1.279 * 7.1492e9) ** 2 / 6.67430e-8,
                                        runtime=GUILLOT_RUNTIME),
                     count_max=GUILLOT_COUNT_MAX),
            tp_eval=tp_eval, n_tp_params=4)
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"chem model build failed: {e}")


# Guillot parameters giving a ~360-450 K column over 5 -> 0.01 bar: cold
# enough aloft to supersaturate S8 at 1e-4 VMR, warm enough deep to stay
# inside the thermo tables.
_GUILLOT = dict(Tirr=560.0, Tint=80.0, log_kappa=-2.3, log_gamma=-1.0)


def _theta_guillot():
    return np.array([0.0, 0.0, 0.0, _GUILLOT["Tirr"], _GUILLOT["Tint"],
                     _GUILLOT["log_kappa"], _GUILLOT["log_gamma"]],
                    dtype=np.float64)


def test_guillot_condensation_end_to_end(stack, chem_guillot):
    """A non-isothermal (Guillot) T-P condensation solve: converges, rains
    out, and the carry saturation row follows the Guillot temperatures."""
    _, jax, jnp = stack
    th = _theta_guillot()
    pv = chem_guillot.prep_pv(th)
    T_live = np.asarray(pv.r_Tco)
    assert T_live.std() > 5.0, "Guillot profile must be genuinely non-isothermal"
    want_sat = np.asarray(_sat_n_s8(jnp, T_live))
    np.testing.assert_allclose(np.asarray(pv.c_sat_n_per_re[0]), want_sat,
                               rtol=1e-12)
    s8, s8_ls = _s8_cols(chem_guillot)
    final, _init = chem_guillot.run_diag(jnp.asarray(th))
    assert int(final.accept_count) < chem_guillot.count_max, \
        "Guillot condensing solve must complete within the step budget"
    assert float(final.t) >= GUILLOT_RUNTIME, \
        "solve must reach the physical runtime cap"
    assert bool(final.fix_species_started)
    y = np.asarray(final.y)
    assert np.all(np.isfinite(y))
    assert y[:, s8_ls].sum() > 0.0
    final_ratio = (y[:, s8] / want_sat).max()
    assert final_ratio < 2.0, f"gas must relax to ~saturation (got {final_ratio:.2f})"


def test_prep_jit_vmap_jvp_traceable(stack, chem_iso):
    """jit / vmap / jvp through the live conden rebuild: no concretization or
    tracer errors, vmap lanes match solo calls, tangents are finite."""
    _, jax, jnp = stack
    th0 = jnp.asarray(_theta(400.0))
    th1 = jnp.asarray(_theta(430.0))

    sat_of = lambda th: chem_iso.prep_pv(th).c_sat_n_per_re  # noqa: E731
    jit_sat = jax.jit(sat_of)(th0)
    np.testing.assert_allclose(np.asarray(jit_sat), np.asarray(sat_of(th0)),
                               rtol=1e-14)
    batched = jax.vmap(sat_of)(jnp.stack([th0, th1]))
    np.testing.assert_allclose(np.asarray(batched[0]), np.asarray(sat_of(th0)),
                               rtol=1e-14)
    np.testing.assert_allclose(np.asarray(batched[1]), np.asarray(sat_of(th1)),
                               rtol=1e-14)
    e_T = jnp.zeros(4, dtype=jnp.float64).at[3].set(1.0)
    _, dsat = jax.jvp(sat_of, (th0,), (e_T,))
    dsat = np.asarray(dsat)
    assert np.all(np.isfinite(dsat)) and np.all(dsat > 0.0)


def test_jvp_matches_finite_difference_through_condensing_state(stack, chem_iso):
    """Forward-mode d/dT through the condensing+pinned steady state vs
    warm-started centered finite differences (both re-converged from the same
    converged column -- the validated Fisher continuation pattern).

    Two regimes, asserted separately:

    * SMOOTH observables -- column totals of unpinned gas species (H2O, CO,
      H2S) whose T response flows through the on-graph rates/structure/
      saturation -- must match FD to 15%.
    * The PINNED species (S8, S8_l_s): the fix pin captures the column at the
      first accepted step past stop_conden_time, and a T perturbation shifts
      the accepted-step sequence, so the FD endpoints capture at slightly
      different drainage states. Measured here: jvp and FD agree in SIGN but
      differ by O(1) in magnitude (0.91 relative on first measurement).
      This measured instability -- discrete active-layer/cold-trap switches
      plus pin-capture jitter -- is exactly why Fisher forecasts with
      condensation stay loudly unsupported in vulcan-jwst-tool
      (forward.canonical_params raises). Only finiteness and sign are
      asserted for the pinned pair.
    """
    _, jax, jnp = stack
    s8, s8_ls = _s8_cols(chem_iso)
    smooth_cols = [chem_iso.sidx[m] for m in ("H2O", "CO", "H2S")]
    th0 = jnp.asarray(_theta(T_STRUCT))
    y_base = chem_iso.converged_y(th0)

    def f(th):
        y = chem_iso.converged_y(th, warm_y=y_base)
        smooth = jnp.stack([jnp.sum(y[:, c]) for c in smooth_cols])
        pinned = jnp.stack([jnp.sum(y[:, s8]), jnp.sum(y[:, s8_ls])])
        return smooth, pinned

    e_T = jnp.zeros(4, dtype=jnp.float64).at[3].set(1.0)
    _, (jv_smooth, jv_pin) = jax.jvp(f, (th0,), (e_T,))
    jv_smooth, jv_pin = np.asarray(jv_smooth), np.asarray(jv_pin)
    assert np.all(np.isfinite(jv_smooth)) and np.all(np.isfinite(jv_pin)), \
        "tangent through condensing state must be finite"

    dT = 0.5
    hi_s, hi_p = (np.asarray(a) for a in f(jnp.asarray(_theta(T_STRUCT + dT))))
    lo_s, lo_p = (np.asarray(a) for a in f(jnp.asarray(_theta(T_STRUCT - dT))))
    fd_smooth = (hi_s - lo_s) / (2.0 * dT)
    fd_pin = (hi_p - lo_p) / (2.0 * dT)

    for i, name in enumerate(("H2O", "CO", "H2S")):
        rel = abs(jv_smooth[i] - fd_smooth[i]) / max(abs(fd_smooth[i]), 1e-300)
        assert rel < 0.15, (f"{name} column: jvp {jv_smooth[i]:.6e} vs FD "
                            f"{fd_smooth[i]:.6e} (rel {rel:.3f}) -- exceeds 15%")

    for i, name in enumerate(("S8 gas", "S8_l_s")):
        assert np.sign(jv_pin[i]) == np.sign(fd_pin[i]), \
            f"{name}: sign(jvp) != sign(FD)"
