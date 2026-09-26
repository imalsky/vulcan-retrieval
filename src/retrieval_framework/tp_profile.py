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
    the photolysis cross-section T-interpolation remains baked at the baseline T
    (host-side upstream step; a documented second-order approximation);
  * the ExoJax side builds its own hydrostatic transit geometry from the SAME T(P)
    and the chemistry MMW, interpolated per interp_map (the chemistry grid reaches
    the ART top, so nothing is extrapolated).
Physical interpretation caveat: f = config_schema.GUILLOT_F = 1/4, the GLOBAL-average
irradiation convention -- an analytic-shape choice, not a terminator measurement, so
retrieved (Tirr, kappa, gamma) are flexible shape parameters of the limb profile
rather than literal disk-average properties.

Import order: ``vulcan_chem`` (which sets env + jax x64) must be imported before this,
because ExoJax is imported lazily inside ``build_tp_model``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from retrieval_framework.config_schema import GUILLOT_F
from vulcan_forward.constants import T_OPA_MAX_K, T_OPA_MIN_K
import jax.numpy as jnp

# [T_OPA_MIN_K, T_OPA_MAX_K] is the engine's valid RT window (the k-tables span
# 100-3400 K and clamp at their edges). The profile is not clipped (a clip invents
# an isothermal wall); the pipeline rejects a profile with a layer outside the
# window inset by T_EDGE_INSET_K, i.e. [320, 2980] K (pipeline.tp_valid).
T_EDGE_INSET_K = 20.0
_T_MIN = float(T_OPA_MIN_K) + T_EDGE_INSET_K
_T_MAX = float(T_OPA_MAX_K) - T_EDGE_INSET_K


def build_tp_model(cfg: Any) -> SimpleNamespace:
    """Build the differentiable T-P evaluator for this Config.

    Returns SimpleNamespace with:
        eval(tp_params, p_bar) -> T (len(p_bar),)   pure-JAX, differentiable
        n_params : int
    """
    from exojax.atm.atmprof import atmprof_Guillot  # lazy: after vulcan_chem

    g = float(cfg.tp_gravity_cgs)
    f = GUILLOT_F
    Tint = float(cfg.tp_Tint_K)
    infer_gamma = bool(cfg.tp_infer_gamma)
    gamma_fixed = float(cfg.tp_gamma_fixed)
    n_params = 3 if infer_gamma else 2

    def _phys(tp):
        Tirr = tp[0]
        kappa = 10.0 ** tp[1]
        gamma = (10.0 ** tp[2]) if infer_gamma else jnp.asarray(gamma_fixed, dtype=tp.dtype)
        return Tirr, kappa, gamma

    def eval_fn(tp_params, p_bar):
        tp = jnp.asarray(tp_params)
        p = jnp.asarray(p_bar, dtype=tp.dtype)
        Tirr, kappa, gamma = _phys(tp)
        # RAW profile -- no clip. Out-of-window draws are rejected upstream, not bent
        # into range (see pipeline.tp_valid).
        return atmprof_Guillot(p, g, kappa, gamma, jnp.asarray(Tint, dtype=tp.dtype), Tirr, f)

    return SimpleNamespace(eval=eval_fn, n_params=int(n_params), T_min=_T_MIN, T_max=_T_MAX)
