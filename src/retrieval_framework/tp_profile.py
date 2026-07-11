"""Differentiable T-P profile built from ExoJax's own atmosphere built-ins.

Per the retrieval design we use ExoJax's ``exojax.atm.atmprof`` profiles rather than
rolling our own:

    tp_model="guillot"  -> atmprof_Guillot(P, g, kappa, gamma, Tint, Tirr, f)
        the built-in Guillot (2010) irradiated analytic profile. ExoJax implements it
        with a plain ``jnp.exp`` (NOT the E2 exponential integral), so it is
        forward-mode-clean -- which matters because the same T(P) is pushed as a
        forward-mode tangent through the VULCAN-JAX ``lax.while_loop``. (The Heng+14
        exponential-integral pathology flagged in the atmosphere-differentiability
        work lives in VULCAN's own ``build_atm``; we bypass it entirely by supplying
        ``Tco`` directly.)

    tp_model="powerlaw" -> atmprof_powerlow(P, T0, alpha)

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
    and the chemistry MMW, interpolated per interp_map (constant-VMR clamp above the
    chemistry top, reported loudly at build).
Physical interpretation caveat: cfg.tp_f defaults to 1/4, the GLOBAL-average
irradiation convention -- an analytic-shape choice, not a terminator measurement, so
retrieved (Tirr, kappa, gamma) are flexible shape parameters of the limb profile
rather than literal disk-average properties.

Import order: ``vulcan_chem`` (which sets env + jax x64) must be imported before this,
because ExoJax is imported lazily inside ``build_tp_model``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from retrieval_framework.forward import config as _pkg_config   # pure constants (T_OPA_MIN_K/MAX_K, GS_CGS)
import jax.numpy as jnp

# The premodit opacity table is baked for [T_OPA_MIN_K, T_OPA_MAX_K]; outside it the RT
# would extrapolate. We DO NOT clip the profile into this range (a clip silently invents a
# fake isothermal wall and a zero-gradient plateau). Instead these bounds define the
# MODELABLE window, and the pipeline rejects any drawn profile with a layer outside it
# (rejection-sampled at init, -inf likelihood for a MALA proposal) -- see pipeline.tp_valid.
# The 20 K inset keeps us off the exact table edge where premodit accuracy degrades.
_T_MIN = float(_pkg_config.T_OPA_MIN_K) + 20.0
_T_MAX = float(_pkg_config.T_OPA_MAX_K) - 20.0


def build_tp_model(cfg: Any) -> SimpleNamespace:
    """Build the differentiable T-P evaluator for this Config.

    Returns SimpleNamespace with:
        eval(tp_params, p_bar) -> T (len(p_bar),)   pure-JAX, differentiable
        n_params : int
        model    : str
        unpack(tp_params) -> dict of the physical T-P quantities (for logging/plots)
    """
    from exojax.atm.atmprof import atmprof_Guillot, atmprof_powerlow  # lazy: after vulcan_chem

    model = str(cfg.tp_model).strip().lower()
    g = float(cfg.tp_gravity_cgs)

    if model == "guillot":
        f = float(cfg.tp_f)
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

        def unpack(tp_params):
            tp = jnp.asarray(tp_params)
            Tirr, kappa, gamma = _phys(tp)
            return dict(Tirr=float(Tirr), kappa=float(kappa), gamma=float(gamma),
                        Tint=Tint, f=f, gravity=g, model="guillot")

    elif model == "powerlaw":
        n_params = 2

        def eval_fn(tp_params, p_bar):
            tp = jnp.asarray(tp_params)
            p = jnp.asarray(p_bar, dtype=tp.dtype)
            return atmprof_powerlow(p, tp[0], tp[1])   # RAW -- no clip (see pipeline.tp_valid)

        def unpack(tp_params):
            tp = jnp.asarray(tp_params)
            return dict(T0=float(tp[0]), alpha=float(tp[1]), gravity=g, model="powerlaw")

    else:
        raise ValueError(f"unknown tp_model {cfg.tp_model!r}")

    return SimpleNamespace(eval=eval_fn, n_params=int(n_params), model=model, unpack=unpack,
                           T_min=_T_MIN, T_max=_T_MAX)
