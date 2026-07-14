"""Smooth-rainout plumbing through the retrieval forward path (Route B G0).

Validates that conden_mode="smooth_rainout" runs the TOOL's production
build/_prep path with the equilibrium-lookup deep boundary ON-GRAPH:

* the bottom-node H2S pin in the initial ProfileVars carry equals the
  checksum-gated lookup evaluated at the LIVE proposal (T_bottom, lnZ, c_o),
  and the live-rebuilt saturation row (the sink's n_sat input) tracks the
  proposed temperature;
* d(pin)/d(theta) rides the graph: jacfwd through _prep is finite, the lnZ
  slope is the measured ~1 physics, the T and c_o partials match the lookup's
  own derivatives, and the lnKzz slot is exactly zero;
* pin_value() is the host-side evaluation-boundary domain guard;
* a short solve on the retrieval path ENFORCES the pin at the bottom node
  (the fix_sp_bot commit rule) with an actively supersaturated S8 column
  (bottom s = n/n_sat - 1 ~ 1.3 by construction);
* every unsupported configuration is refused loudly BEFORE the pre-loop.

Setup mirrors tests/test_condensation_live_tp.py: a 32-layer isothermal
400 K synthetic column on the production SNCHO network, const_mix init
(no FastChem), photochemistry off -- but P_b = 7.6 bar (the lookup table's
build pressure, enforced) and const_mix constructed so the column elemental
ratios equal the table provenance baseline_X_H (the lnZ-basis gate,
tolerance 2%). This is the LOCAL mechanics fixture only; the mandatory
scientific fixture for the B0C gates is the source-bearing W107b
Guillot+photo column (heavy, scheduled runs). NO derivative through the
SOLVE is asserted here: the unrolled jvp through the smooth-rainout
steady state is measured invalid (B0-6), and the solver-map G6 route is
pending -- _prep differentiability (this file) is a necessary but not
sufficient condition.

Builds one chem model (~1-2 min, compile-dominated); skips cleanly when
the chem stack is unavailable (same policy as test_condensation_live_tp).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

T_STRUCT = 400.0
NZ = 32

# const_mix constructed so the column elemental ratios-to-H equal the lookup
# table's provenance baseline_X_H (O 5.37e-3, C 2.95e-3, N 7.08e-4, S
# 1.41e-4) to <0.1% -- the build's lnZ-basis gate. S is split S8-heavy
# (8 x 2.5e-5 in S8, rest H2S) so the bottom node is genuinely
# supersaturated: s = x_S8 * n_0 / n_sat - 1 ~ 1.3 at 400 K, ~6.85 bar
# (measured; sat mix there is 1.09e-5). He/H2O/CO/N2/H2S are the
# elemental-repair species and must be present.
CONST_MIX = {
    "H2": 0.8501538,
    "He": 0.14,
    "H2O": 4.13493e-3,
    "CO": 5.04052e-3,
    "N2": 6.04862e-4,
    "H2S": 4.0920e-5,
    "S8": 2.5e-5,
}

THETA_BOX = {"T_bottom": (400.0, 500.0), "lnZ": (-0.5, 0.5), "c_o": (-0.3, 0.3)}

# Every cfg-module attribute any override dict in this file touches --
# snapshot/restore so the shared vulcan_cfg_W39b module never leaks
# smooth-rainout state into other test files (or inherits theirs).
_TOUCHED_CFG_KEYS = (
    "atm_type", "Tiso", "P_b", "P_t", "Kzz_prof", "const_Kzz", "use_photo",
    "use_moldiff", "ini_mix", "const_mix", "use_condense", "condense_sp",
    "non_gas_sp", "r_p", "rho_p", "conden_mode", "conden_smooth_width",
    "rainout_rate_scale", "use_fix_sp_bot", "use_relax", "use_settling",
    "fix_species", "mtol_conv", "conver_ignore", "runtime", "yconv_cri",
    "nz", "count_max", "count_min",
)


def _cfg_overrides(pin, **extra):
    over = {
        "atm_type": "isothermal",
        "Tiso": T_STRUCT,
        "P_b": 7.6e6,          # bar -> dyn/cm2; MUST equal the table's P_bar
        "P_t": 1.0e4,
        "Kzz_prof": "const",
        "const_Kzz": 1.0e7,
        "use_photo": False,
        "use_moldiff": True,
        "ini_mix": "const_mix",
        "const_mix": dict(CONST_MIX),
        "use_condense": True,
        "condense_sp": ["S8"],
        "non_gas_sp": ["S8_l_s"],
        "r_p": {"S8_l_s": 5.0e-3},
        "rho_p": {"S8_l_s": 2.07},
        "conden_mode": "smooth_rainout",
        "conden_smooth_width": 0.1,
        "rainout_rate_scale": 1.0,
        # Static warm-up pin: the lookup value at the structural baseline
        # (T=400 K, lnZ=0, c_o=0); _prep overrides it on-graph per proposal.
        "use_fix_sp_bot": {"H2S": pin},
        # neutralize any master_pin window/pin state on the shared cfg module
        "use_relax": [],
        "use_settling": False,
        "fix_species": [],
        # cold-column convergence hygiene (same rationale as the live-T(P)
        # conden test): glacial trace kinetics below RT relevance
        "mtol_conv": 1.0e-15,
        "conver_ignore": ["C6H6", "C2H2", "C6H5", "C2H", "C2H4", "C2H5",
                          "C2H6", "C3H2", "C3H3", "C4H5", "CH2NH", "CH3NH2",
                          "H2CCO", "S", "S2", "S3", "S4"],
        # physical integration cap: a cold no-photo synthetic column has no
        # reachable longdy steady state (quench creep); upstream practice.
        "runtime": 1.0e13,
    }
    over.update(extra)
    return over


def _profile(pin, **extra):
    from retrieval_framework.forward import config

    prof = dict(config.SMOKE)
    prof.update(
        nz=NZ,
        use_photo=False,
        yconv_cri=1.0e-2,
        count_max=4000,
        abundance_mode="elemental",
        cfg_overrides=_cfg_overrides(pin),
        h2s_boundary={"theta_box": dict(THETA_BOX)},
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


@pytest.fixture(scope="module", autouse=True)
def _restore_cfg_module(stack):
    """Snapshot/restore the shared vulcan_cfg module attributes this file
    mutates (build_chem_model setattrs cfg_overrides onto the module)."""
    from retrieval_framework.forward import config as fconfig

    mod = importlib.import_module(fconfig.W39B_CFG_MODULE)
    snap = {k: (hasattr(mod, k), getattr(mod, k, None))
            for k in _TOUCHED_CFG_KEYS}
    yield
    for k, (had, v) in snap.items():
        if had:
            setattr(mod, k, v)
        elif hasattr(mod, k):
            delattr(mod, k)


@pytest.fixture(scope="module")
def table(stack):
    from retrieval_framework.forward import config, h2s_boundary as hb

    return hb.load_h2s_boundary_table(config.H2S_BOUNDARY_TABLE)


@pytest.fixture(scope="module")
def chem(stack, table):
    """Smooth-rainout chem model, isothermal tp_eval: theta=[lnZ,c_o,lnKzz,T]."""
    vulcan_chem, jax, jnp = stack
    from retrieval_framework.forward import h2s_boundary as hb

    pin0 = float(hb.h2s_pin_mix(table, T_STRUCT, 0.0, 0.0))

    def tp_eval(tp, p_bar):
        return jnp.zeros_like(jnp.asarray(p_bar)) + tp[0]

    try:
        return vulcan_chem.build_chem_model(_profile(pin0), tp_eval=tp_eval,
                                            n_tp_params=1)
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"chem model build failed: {e}")


def _theta(lnZ=0.0, c_o=0.0, T=T_STRUCT):
    return np.array([lnZ, c_o, 0.0, float(T)], dtype=np.float64)


def test_pin_reaches_carry_at_live_theta(chem, stack, table):
    """pv.bot_pin_mix equals the lookup at the proposal's (T, lnZ, c_o), and
    the sink's saturation row is rebuilt at the LIVE temperature."""
    _, jax, jnp = stack
    from retrieval_framework.forward import h2s_boundary as hb
    from vulcan_jax.atm_setup import sat_p_jax
    from vulcan_jax.phy_const import kb

    assert chem.conden_mode == "smooth_rainout"
    assert chem.h2s_table is not None
    for lnZ, c_o, T in [(0.0, 0.0, 400.0), (0.3, -0.2, 437.5),
                        (-0.4, 0.25, 462.0)]:
        pv = chem.prep_pv(_theta(lnZ, c_o, T))
        assert pv.bot_pin_mix.shape == (1,)
        want = float(hb.h2s_pin_mix(table, T, lnZ, c_o))
        np.testing.assert_allclose(float(pv.bot_pin_mix[0]), want, rtol=1e-14)
        n_sat = float(sat_p_jax("S8", jnp.asarray(T)) / kb / T)
        np.testing.assert_allclose(np.asarray(pv.c_sat_n_per_re[0]),
                                   n_sat, rtol=1e-12)


def test_bottom_node_supersaturated(chem, stack):
    """The fixture is a genuinely active-sink column: bottom-node S8 exceeds
    saturation (else the integration test would only exercise a dead sink)."""
    pv = chem.prep_pv(_theta())
    n_s8_bot = CONST_MIX["S8"] * float(pv.n_0[0])
    s = n_s8_bot / float(pv.c_sat_n_per_re[0][0]) - 1.0
    assert s > 0.5, f"bottom supersaturation s={s}: fixture not sink-active"


def test_pin_theta_gradient_through_prep(chem, stack, table):
    """d(ln pin)/d(theta) through _prep: finite, lnZ slope ~ 1 (measured
    physics), T/c_o partials equal the lookup's own, lnKzz exactly zero."""
    _, jax, jnp = stack
    from retrieval_framework.forward import h2s_boundary as hb

    theta0 = jnp.asarray([0.3, -0.2, 0.0, 437.5])

    def ln_pin(th):
        return jnp.log(chem.prep_pv(th).bot_pin_mix[0])

    g = np.asarray(jax.jacfwd(ln_pin)(theta0))
    assert np.all(np.isfinite(g))
    assert 0.95 < g[0] < 1.05          # d ln x_H2S / d lnZ, measured ~1
    assert g[2] == 0.0                 # lnKzz does not touch the boundary

    def ln_direct(p):                  # p = (T, lnZ, c_o), lookup only
        return jnp.log(hb.h2s_pin_mix(table, p[0], p[1], p[2]))

    gd = np.asarray(jax.jacfwd(ln_direct)(jnp.asarray([437.5, 0.3, -0.2])))
    np.testing.assert_allclose(g[3], gd[0], rtol=1e-10)   # T partial
    np.testing.assert_allclose(g[1], gd[2], rtol=1e-10)   # c_o partial
    np.testing.assert_allclose(g[0], gd[1], rtol=1e-10)   # lnZ partial


def test_pin_value_host_guard(chem):
    """pin_value validates each actually-visited point (loud) and reports
    the pin with its lookup inputs."""
    d = chem.pin_value(_theta(0.3, -0.2, 437.5))
    assert d["T_bottom"] == 437.5 and d["lnZ"] == 0.3 and d["c_o"] == -0.2
    pv = chem.prep_pv(_theta(0.3, -0.2, 437.5))
    np.testing.assert_allclose(d["x_pin"], float(pv.bot_pin_mix[0]),
                               rtol=1e-14)
    with pytest.raises(ValueError, match="outside the validated"):
        chem.pin_value(_theta(T=350.0))


def test_short_solve_enforces_pin(chem, stack):
    """A solve on the retrieval path commits the pin at the bottom node
    exactly (fix_sp_bot rule: sol[0, idx] = bot_pin_mix * n_0[0]).

    theta = (0, 0, 0, 400): unit mask scaling, so the init S8 column IS
    CONST_MIX * n_0 (to the ~1e-8 elemental-repair residual) and the
    "column moved" assertion genuinely detects the active sink (bottom
    s ~ 1.3), not the theta perturbation."""
    theta = _theta(0.0, 0.0, 400.0)
    y, nacc = chem.converged_y(theta, return_diag=True)
    y = np.asarray(y, dtype=np.float64)
    assert np.all(np.isfinite(y))
    assert int(nacc) >= 1
    pv = chem.prep_pv(theta)
    want = float(pv.bot_pin_mix[0]) * float(pv.n_0[0])
    np.testing.assert_allclose(y[0, chem.sidx["H2S"]], want, rtol=1e-10)
    # the sink + boundary moved the S8 column off its init
    y0 = CONST_MIX["S8"] * np.asarray(pv.n_0, dtype=np.float64)
    assert not np.allclose(y[:, chem.sidx["S8"]], y0, rtol=1e-3)


def test_build_refusals_are_loud(stack, table):
    """Every unsupported smooth-mode configuration dies loudly at build.

    The first five refusals fire BEFORE the pre-loop (config-only); the
    static-pin sanity refusal is a measured check and needs the pre-loop
    (structural T), but still fires before the expensive warm-up solve."""
    vulcan_chem, jax, jnp = stack
    from retrieval_framework.forward import h2s_boundary as hb

    pin0 = float(hb.h2s_pin_mix(table, T_STRUCT, 0.0, 0.0))

    def tp_eval(tp, p_bar):
        return jnp.zeros_like(jnp.asarray(p_bar)) + tp[0]

    def build(prof):
        return vulcan_chem.build_chem_model(prof, tp_eval=tp_eval,
                                            n_tp_params=1)

    prof = _profile(pin0)
    del prof["h2s_boundary"]
    with pytest.raises(ValueError, match="h2s_boundary"):
        build(prof)

    prof = _profile(pin0)
    prof["cfg_overrides"]["use_fix_sp_bot"] = {"SO2": 1e-5}
    with pytest.raises(ValueError, match="exactly H2S"):
        build(prof)

    prof = _profile(pin0)
    prof["cfg_overrides"]["fix_species"] = ["S8", "S8_l_s"]
    with pytest.raises(ValueError, match="rejects fix_species"):
        build(prof)

    prof = _profile(pin0)
    prof["cfg_overrides"]["P_b"] = 5.0e6
    with pytest.raises(ValueError, match="build pressure"):
        build(prof)

    prof = _profile(pin0)
    prof["h2s_boundary"] = {"theta_box": dict(THETA_BOX,
                                              T_bottom=(300.0, 500.0))}
    with pytest.raises(ValueError, match="exceeds the validated"):
        build(prof)

    prof = _profile(pin0)
    prof["cfg_overrides"]["use_fix_sp_bot"] = {"H2S": pin0 * 10.0}
    with pytest.raises(ValueError, match="inconsistent with the lookup"):
        build(prof)
