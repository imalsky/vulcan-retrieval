"""Differentiable T-P profile built from ExoJax's own atmosphere built-ins.

Uses ExoJax's ``exojax.atm.atmprof`` profiles:

    atmprof_Guillot(P, g, kappa, gamma, Tint, Tirr, f)
        the built-in Guillot (2010) irradiated analytic profile. ExoJax implements it
        with a plain ``jnp.exp`` (NOT the E2 exponential integral), so it is
        forward-mode-clean -- which matters because the same T(P) is pushed as a
        forward-mode tangent through the VULCAN-JAX ``lax.while_loop``. (VULCAN's
        own ``build_atm`` is bypassed by supplying ``Tco`` directly.)

``build_tp_model(cfg)`` returns an object whose ``eval(tp_params, p_bar_grid)`` maps the
*retrieved* T-P sub-vector + the fixed constants to a temperature array on ANY pressure
grid (bar). The retrieval evaluates the SAME analytic curve on both the VULCAN grid (for
chemistry) and the ExoJax ART grid (for the RT), so the two sides sample one T(P)
function. Scope of that consistency, stated precisely:
  * the chemistry side rebuilds the rate table AND the T/composition-dependent
    atmospheric structure (hydrostatic geometry via the runner's in-loop refresh +
    seeded initial carry, Dzz/vm/vs on-graph) from this T(P) -- see vulcan_chem;
    the photolysis cross sections are T-independent (vulcan_chem refuses a
    non-empty T_cross_sp);
  * the ExoJax side builds its own hydrostatic transit geometry from the SAME T(P)
    and the chemistry MMW, interpolated per interp_map (the chemistry grid reaches
    the ART top, so nothing is extrapolated).
Physical interpretation caveat: f = config_schema.GUILLOT_F = 1/4, the GLOBAL-average
irradiation convention -- an analytic-shape choice, not a terminator measurement, so
retrieved (Tirr, kappa, gamma) are flexible shape parameters of the limb profile
rather than literal disk-average properties. The deep parameter (fixed Tint, or
retrieved last with ``tp_infer_Tint``: Tint itself, or with
``tp_deep_param="T1bar"`` the temperature at ``tp_deep_p_bar``) adds 3/4 Tint^4
tau and so sets the deep column; with constant kappa the deep gradient is 0.25,
below the adiabat, so the profile never turns convective on its own and Tint is a
deep-temperature knob, not the planet's measured intrinsic flux. Retrieving T at a
deep level puts the prior on the quantity the validity window acts on: Tint
follows in closed form (T^4 is linear in Tint^4) and a draw whose Tint^4 would be
negative evaluates to NaN, which ``pipeline.tp_valid`` rejects. ``tp_adiabat``
replaces the profile below a free boundary pressure by the adiabat matched there
(a sensitivity model: the join is imposed, nothing makes the column convective).

Import order: ``vulcan_chem`` (which sets env + jax x64) must be imported before this,
because ExoJax is imported lazily inside ``build_tp_model``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from retrieval_framework.config_schema import GUILLOT_F
from vulcan_forward.constants import BAR_CGS, T_OPA_MAX_K, T_OPA_MIN_K
import jax.numpy as jnp

# [T_OPA_MIN_K, T_OPA_MAX_K] is the engine's valid RT window (the k-tables span
# 100-3400 K and clamp at their edges). The profile is not clipped (a clip invents
# an isothermal wall); the pipeline rejects a profile with a layer outside the
# window inset by T_EDGE_INSET_K, i.e. [320, 2980] K (pipeline.tp_valid).
T_EDGE_INSET_K = 20.0
_T_MIN = float(T_OPA_MIN_K) + T_EDGE_INSET_K
_T_MAX = float(T_OPA_MAX_K) - T_EDGE_INSET_K


def guillot_terms(p_bar, g, kappa, gamma, f):
    """Guillot (2010) Eq. 29 split as T^4 = a Tint^4 + b Tirr^4: ``(a, b)`` at
    ``p_bar``, the algebra of exojax's ``atmprof_Guillot`` (tau = P kappa / g)."""
    tau = jnp.asarray(p_bar) * BAR_CGS * kappa / g
    invsq3 = 1.0 / jnp.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (1.0 / gamma
                                + (gamma - 1.0 / gamma) * jnp.exp(-gamma * tau / invsq3))
    return 0.75 * (2.0 / 3.0 + tau), 0.75 * f * fac


def tint_from_deep(T_deep, Tirr, kappa, gamma, g, f, p_deep_bar):
    """Tint of the Guillot profile whose temperature at ``p_deep_bar`` is
    ``T_deep``: NaN where the irradiation term alone exceeds T_deep^4 (no such
    profile exists; the pipeline rejects the draw)."""
    a, b = guillot_terms(p_deep_bar, g, kappa, gamma, f)
    tint4 = (jnp.asarray(T_deep) ** 4 - b * jnp.asarray(Tirr) ** 4) / a
    return jnp.where(tint4 > 0.0, jnp.abs(tint4), jnp.nan) ** 0.25


def build_tp_model(cfg: Any) -> SimpleNamespace:
    """Build the differentiable T-P evaluator for this Config.

    Returns SimpleNamespace with:
        eval(tp_params, p_bar) -> T (len(p_bar),)   pure-JAX, differentiable
        n_params : int
        tint_of(tp_params) -> Tint (the retrieved one, or reconstructed from T_deep)
        deep_param : "fixed", "Tint" or "T1bar"; p_deep_bar; adiabat; nabla_ad

    The T-P sub-vector is [Tirr, log10kappa, (log10gamma), (Tint | T1bar),
    (log10P_rcb)], the order of config_schema.specs_from_config.
    """
    from exojax.atm.atmprof import atmprof_Guillot  # lazy: after vulcan_chem

    g = float(cfg.tp_gravity_cgs)
    f = GUILLOT_F
    Tint = float(cfg.tp_Tint_K)
    infer_gamma = bool(cfg.tp_infer_gamma)
    gamma_fixed = float(cfg.tp_gamma_fixed)
    infer_Tint = bool(cfg.tp_infer_Tint)
    deep_param = str(cfg.tp_deep_param) if infer_Tint else "fixed"
    p_deep = float(cfg.tp_deep_p_bar)
    adiabat = bool(cfg.tp_adiabat)
    nabla = float(cfg.tp_nabla_ad)
    i_deep = 2 + int(infer_gamma)          # the T-P order of config_schema.specs_from_config
    i_rcb = i_deep + int(infer_Tint)
    n_params = i_rcb + int(adiabat)

    def _phys(tp):
        Tirr = tp[0]
        kappa = 10.0 ** tp[1]
        gamma = (10.0 ** tp[2]) if infer_gamma else jnp.asarray(gamma_fixed, dtype=tp.dtype)
        if not infer_Tint:
            Tint_t = jnp.asarray(Tint, dtype=tp.dtype)
        elif deep_param == "T1bar":
            T_inv, p_inv = tp[i_deep], p_deep
            if adiabat:
                # the deep level may sit on the adiabat: then the Guillot part
                # must pass through T_deep (P_rcb / p_deep)^nabla at P_rcb, so the
                # adiabat below it passes through T_deep at p_deep
                p_inv = jnp.minimum(10.0 ** tp[i_rcb], p_deep)
                T_inv = T_inv * (p_inv / p_deep) ** nabla
            Tint_t = tint_from_deep(T_inv, Tirr, kappa, gamma, g, f, p_inv)
        else:
            Tint_t = tp[i_deep]
        return Tirr, kappa, gamma, Tint_t

    def eval_fn(tp_params, p_bar):
        tp = jnp.asarray(tp_params)
        p = jnp.asarray(p_bar, dtype=tp.dtype)
        Tirr, kappa, gamma, Tint_t = _phys(tp)
        # RAW profile -- no clip. Out-of-window draws are rejected upstream, not bent
        # into range (see pipeline.tp_valid).
        T = atmprof_Guillot(p, g, kappa, gamma, Tint_t, Tirr, f)
        if adiabat:
            p_rcb = 10.0 ** tp[i_rcb]
            T_rcb = atmprof_Guillot(p_rcb, g, kappa, gamma, Tint_t, Tirr, f)
            T = jnp.where(p > p_rcb, T_rcb * (p / p_rcb) ** nabla, T)
        return T

    def tint_of(tp_params):
        return _phys(jnp.asarray(tp_params))[3]

    return SimpleNamespace(eval=eval_fn, n_params=int(n_params), T_min=_T_MIN, T_max=_T_MAX,
                           tint_of=tint_of, deep_param=deep_param, p_deep_bar=p_deep,
                           adiabat=adiabat, nabla_ad=nabla)
