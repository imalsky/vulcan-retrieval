"""Differentiable T-P profile from exojax's ``atmprof_Guillot``, with an optional deep level and adiabat.

``build_tp_model`` maps the retrieved T-P entries and the fixed constants to a temperature on any pressure grid, bar.
The retrieval evaluates the same curve on the VULCAN grid and the ART grid. exojax implements Guillot (2010) with a
plain ``jnp.exp``, so the profile is forward-mode clean. exojax is imported inside ``build_tp_model``: ``vulcan_chem``
must come first. docs/method.md states what the parameters mean and what they do not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import jax.numpy as jnp
from vulcan_forward.constants import BAR_CGS, T_OPA_MAX_K, T_OPA_MIN_K

from retrieval_framework.config_schema import GUILLOT_F

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike

    from retrieval_framework.config_schema import Config

__all__ = ["T_EDGE_INSET_K", "build_tp_model", "tint_from_deep"]

# The profile is not clipped, since a clip invents an isothermal wall; the pipeline rejects a profile with a layer
# outside the engine's valid RT window inset by this, [320, 2980] K (pipeline.tp_valid).
T_EDGE_INSET_K = 20.0
_T_MIN = float(T_OPA_MIN_K) + T_EDGE_INSET_K
_T_MAX = float(T_OPA_MAX_K) - T_EDGE_INSET_K


def _guillot_terms(
    p_bar: ArrayLike, g: float, kappa: Array | float, gamma: Array | float, f: float
) -> tuple[Array, Array]:
    """Return Guillot (2010) Eq. 29 split as T^4 = a Tint^4 + b Tirr^4, the algebra of exojax's ``atmprof_Guillot``.

    Parameters
    ----------
    p_bar : array-like
        Pressure, bar.
    g : float
        Gravity, cm/s^2.
    kappa : array-like
        IR opacity, cm^2/g; tau = P kappa / g.
    gamma : array-like
        The visible-to-IR opacity ratio.
    f : float
        The irradiation redistribution factor.

    Returns
    -------
    tuple of jax.Array
        ``(a, b)`` at ``p_bar``.
    """
    tau = jnp.asarray(p_bar) * BAR_CGS * kappa / g
    invsq3 = 1.0 / jnp.sqrt(3.0)
    fac = 2.0 / 3.0 + invsq3 * (1.0 / gamma + (gamma - 1.0 / gamma) * jnp.exp(-gamma * tau / invsq3))
    return 0.75 * (2.0 / 3.0 + tau), 0.75 * f * fac


def tint_from_deep(
    T_deep: ArrayLike,
    Tirr: ArrayLike,
    kappa: Array | float,
    gamma: Array | float,
    g: float,
    f: float,
    p_deep_bar: ArrayLike,
) -> Array:
    """Return the Tint of the Guillot profile whose temperature at ``p_deep_bar`` is ``T_deep``.

    NaN where the irradiation term alone exceeds T_deep^4: no such profile exists, and the pipeline rejects the draw.

    Parameters
    ----------
    T_deep : array-like
        Temperature at the deep level, K.
    Tirr : array-like
        Irradiation temperature, K.
    kappa : array-like
        IR opacity, cm^2/g.
    gamma : array-like
        The visible-to-IR opacity ratio.
    g : float
        Gravity, cm/s^2.
    f : float
        The irradiation redistribution factor.
    p_deep_bar : array-like
        The deep level, bar.

    Returns
    -------
    jax.Array
        Tint, K.
    """
    a, b = _guillot_terms(p_deep_bar, g, kappa, gamma, f)
    tint4 = (jnp.asarray(T_deep) ** 4 - b * jnp.asarray(Tirr) ** 4) / a
    return jnp.where(tint4 > 0.0, jnp.abs(tint4), jnp.nan) ** 0.25


def build_tp_model(cfg: Config) -> SimpleNamespace:
    """Build the T-P evaluator for this Config.

    The entries are ``[Tirr, log10kappa, (log10gamma), (Tint | T1bar), (log10P_rcb)]``, the order of
    ``config_schema.specs_from_config``.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    types.SimpleNamespace
        ``eval(tp_params, p_bar)``, ``n_params``, ``T_min``, ``T_max``, ``tint_of(tp_params)``, ``deep_param``,
        ``p_deep_bar``, ``adiabat``, ``nabla_ad``.
    """
    from exojax.atm.atmprof import atmprof_Guillot

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
    i_deep = 2 + int(infer_gamma)
    i_rcb = i_deep + int(infer_Tint)
    n_params = i_rcb + int(adiabat)

    def _phys(tp: Array) -> tuple[Array, Array, Array, Array]:
        """Return (Tirr, kappa, gamma, Tint) of one T-P vector."""
        Tirr = tp[0]
        kappa = 10.0 ** tp[1]
        gamma = (10.0 ** tp[2]) if infer_gamma else jnp.asarray(gamma_fixed, dtype=tp.dtype)
        if not infer_Tint:
            Tint_t = jnp.asarray(Tint, dtype=tp.dtype)
        elif deep_param == "T1bar":
            T_inv = tp[i_deep]
            p_inv: Array | float = p_deep
            if adiabat:
                # The deep level may sit on the adiabat: the Guillot part must then pass through
                # T_deep (P_rcb / p_deep)^nabla at P_rcb.
                p_inv = jnp.minimum(10.0 ** tp[i_rcb], p_deep)
                T_inv = T_inv * (p_inv / p_deep) ** nabla
            Tint_t = tint_from_deep(T_inv, Tirr, kappa, gamma, g, f, p_inv)
        else:
            Tint_t = tp[i_deep]
        return Tirr, kappa, gamma, Tint_t

    def eval_fn(tp_params: ArrayLike, p_bar: ArrayLike) -> Array:
        """Return T(p_bar), K: the raw profile, since out-of-window draws are rejected, never clipped."""
        tp = jnp.asarray(tp_params)
        p = jnp.asarray(p_bar, dtype=tp.dtype)
        Tirr, kappa, gamma, Tint_t = _phys(tp)
        T = atmprof_Guillot(p, g, kappa, gamma, Tint_t, Tirr, f)
        if adiabat:
            p_rcb = 10.0 ** tp[i_rcb]
            T_rcb = atmprof_Guillot(p_rcb, g, kappa, gamma, Tint_t, Tirr, f)
            T = jnp.where(p > p_rcb, T_rcb * (p / p_rcb) ** nabla, T)
        return T

    def tint_of(tp_params: ArrayLike) -> Array:
        """Return the Tint of one T-P vector, retrieved or reconstructed from the deep level."""
        return _phys(jnp.asarray(tp_params))[3]

    return SimpleNamespace(
        eval=eval_fn,
        n_params=int(n_params),
        T_min=_T_MIN,
        T_max=_T_MAX,
        tint_of=tint_of,
        deep_param=deep_param,
        p_deep_bar=p_deep,
        adiabat=adiabat,
        nabla_ad=nabla,
    )
