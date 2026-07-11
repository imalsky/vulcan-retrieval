"""Differentiable bridge: VULCAN-JAX (nz) pressure grid -> ExoJax ART (nlayer) grid.

The chemistry runs on VULCAN's vertical grid; ExoJax's ``ArtTransPure`` has its own
log-spaced pressure grid. We map per-layer profiles (T, VMR, mean molecular weight)
between them with a log-pressure linear interpolation that is pure JAX (``jnp.interp``)
and therefore differentiable -- so forward-mode tangents pass cleanly across the bridge.

``jnp.interp`` requires ascending sample points, so we sort the VULCAN grid by log10(P)
once (static index array) and CLAMP outside the range. Clamping is NOT a no-op in the
shipped configuration: the ART top (config.ART_PTOP_BAR = 1e-8 bar) deliberately sits
one decade ABOVE VULCAN's chemistry top (P_t = 1e-7 bar), so every ART layer above the
chemistry top receives the topmost VULCAN value -- a constant-VMR / isothermal upper
extension (see the ART_PTOP_BAR comment in config.py for the rationale and
validation/top_pressure_ladder.py for the convergence test). Any clamped span is
reported loudly at build time so the choice is visible in every run log; a clamped
BOTTOM (chemistry not covering the ART bottom) is refused -- that is a mis-set grid,
not a modeling convention.

Interpolation caveats (documented, not silent): linear-in-log-P is not column- or
mass-conservative and can smear photochemical transitions sharper than the ART layer
spacing; the vertical-grid convergence test in validation/ is the check that neither
matters at the quoted precision.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def make_to_art(p_bar_vulcan: np.ndarray, p_bar_art: np.ndarray):
    """Build a differentiable ``to_art(profile_nz) -> profile_nlayer`` interpolator.

    Parameters
    ----------
    p_bar_vulcan : (nz,) np.ndarray
        VULCAN cell-center pressures in bar (any monotonic order).
    p_bar_art : (nlayer,) np.ndarray
        ExoJax ART layer pressures in bar (top-to-bottom).

    Returns
    -------
    callable
        ``to_art(profile)`` where ``profile`` is a JAX array of shape (nz,);
        returns a JAX array of shape (nlayer,) interpolated in log-pressure.

    Raises
    ------
    ValueError
        If the ART grid extends BELOW the VULCAN bottom (deep clamping would
        silently fabricate deep-atmosphere chemistry; fix the grids instead).
    """
    order = np.argsort(p_bar_vulcan)                       # static
    logP_v_sorted = jnp.asarray(np.log10(p_bar_vulcan[order]))
    logP_art = jnp.asarray(np.log10(np.asarray(p_bar_art)))
    order_j = jnp.asarray(order)

    # Loud, host-side accounting of the clamped span (runs once at build).
    p_top_v, p_btm_v = float(np.min(p_bar_vulcan)), float(np.max(p_bar_vulcan))
    n_top_clamp = int(np.sum(np.asarray(p_bar_art) < p_top_v))
    n_btm_clamp = int(np.sum(np.asarray(p_bar_art) > p_btm_v))
    if n_btm_clamp:
        raise ValueError(
            f"ART grid bottom ({np.max(p_bar_art):.3e} bar) lies below the VULCAN "
            f"chemistry bottom ({p_btm_v:.3e} bar): {n_btm_clamp} layers would clamp "
            "to the deepest chemistry value. Shrink ART_PBTM_BAR or extend the "
            "chemistry grid.")
    if n_top_clamp:
        print(f"[interp] NOTE: {n_top_clamp}/{len(p_bar_art)} ART layers sit above the "
              f"VULCAN chemistry top ({p_top_v:.1e} bar) and use the constant-VMR/"
              "isothermal clamp extension (deliberate; see interp_map docstring + "
              "config.ART_PTOP_BAR).", flush=True)

    def to_art(profile_nz):
        fp = profile_nz[order_j]
        return jnp.interp(logP_art, logP_v_sorted, fp)

    return to_art
