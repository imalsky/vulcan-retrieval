"""The retrieval forward: chemistry and T-P parameters plus lnR0 to the native transit spectrum.

``chem_theta = [lnZ, dln(C/O), lnKzz, <extras>, <T-P>]``; one T(P) drives the chemistry on the VULCAN grid and the RT on
the ART grid. Pure JAX and forward-mode differentiable: the runner's ``lax.while_loop`` has no vjp. ``vulcan_chem`` is
imported before ``exojax_rt`` and ``tp_profile``, in that order on purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

import numpy as np

__all__ = ["build_retrieval_forward"]

logger = logging.getLogger("retrieval")

from retrieval_framework.forward import config  # noqa: F401  (hands the engine its data root)
from retrieval_framework import config_schema as C
from vulcan_forward import constants
from vulcan_forward import vulcan_chem  # must precede exojax
import jax
import jax.numpy as jnp

from retrieval_framework import tp_profile
from retrieval_framework.pipeline import _TERM_DT_MIN_FROZEN
from vulcan_forward import exojax_rt
from vulcan_forward import interp_map

# Whether this vulcan-forward reports why a solve stopped: the commit requirements-hpc.txt pins does not.
_REPORTS_EXIT = "termination_reason" in vulcan_chem.ConvDiag._fields

if TYPE_CHECKING:
    from jax import Array


def _refuse_condense_inference(chem: SimpleNamespace, cfg: C.Config) -> None:
    """Refuse a retrieval through a condensing steady state, on the resolved engine config.

    ``config_schema.validate_config`` catches ``cfg_overrides``; a base VULCAN config can default ``use_condense`` too.
    No retrieval with condensation has been validated.
    """
    if chem.conden_spec is not None and not cfg.allow_condense_inference:
        raise ValueError(
            "condensation is active in the RESOLVED VULCAN config "
            f"(vulcan_cfg_name={cfg.vulcan_cfg_name!r}): no "
            "retrieval with condensation has been validated (certification and "
            "start dependence of condensing columns are untested). Run a "
            "condensation forward model through "
            "vulcan_forward.vulcan_chem.build_chem_model, or set "
            "allow_condense_inference=True."
        )


def _rtol_accept(chem: SimpleNamespace) -> float:
    """Return the step-acceptance tolerance baked into a chemistry program."""
    # The private name: the vulcan3 commit in requirements-hpc.txt has no public `statics`.
    return float(chem._integ._statics.rtol_accept)


def _refuse_c_o_bound(chem: SimpleNamespace, cfg: C.Config) -> None:
    """Refuse a C/O prior that reaches the fixed-O positivity bound; ``co_mode="proxy"`` has no such bound.

    b_z must stay positive, else prior-corner columns get negative O carriers that the runner's clip turns into a
    wrong-inventory, finite-likelihood state. First order: on the baseline column; hot columns can bind slightly
    tighter.
    """
    if str(cfg.co_mode) == "fixed_O":
        bound = float(chem.co_bz_bound)
        hi = float(cfg.prior_c_o[1])
        if hi >= bound:
            raise ValueError(
                f"prior_c_o upper bound {hi:+.3f} reaches the fixed-O b_z positivity "
                f"bound {bound:+.3f} (baseline column): beyond it the O-only "
                "compensation goes negative and the solver clips the inventory into "
                f"silently-wrong states. Lower prior_c_o[1] below {bound:.3f} (with "
                "margin), or use co_mode='proxy'."
            )


def _refuse_unreported_freezes() -> None:
    """Refuse a vulcan3 with the dt_min guard under a forward whose ``ConvDiag`` cannot report a frozen column."""
    import vulcan3

    if hasattr(vulcan3, "DtMinFrozenError") and not _REPORTS_EXIT:
        raise RuntimeError(
            "this vulcan3 stops a column frozen at dt_min and this vulcan-forward's ConvDiag has no exit reason, "
            "so frozen draws would be recorded as rejected: update vulcan-forward"
        )


def _with_stage1_freeze(cd1: Any, cd2: Any) -> Any:
    """Return stage 2's ``ConvDiag``, reporting a stage 1 freeze: stage 1 is not gated, so it would go unseen."""
    if not _REPORTS_EXIT:
        return cd2
    frozen = cd1.termination_reason == _TERM_DT_MIN_FROZEN
    return cd2._replace(
        termination_reason=jnp.where(frozen, cd1.termination_reason, cd2.termination_reason),
        last_delta=jnp.where(frozen, cd1.last_delta, cd2.last_delta),
    )


def _check_frozen_network(cfg: C.Config) -> None:
    """Refuse an engine frozen on another network than the config's: it would solve other chemistry."""
    import vulcan3
    from vulcan3 import chem_funs

    # The private name: the vulcan3 commit in requirements-hpc.txt has no public `NETWORK`.
    frozen = Path(chem_funs._NETWORK.network_path).resolve()
    want = (Path(vulcan3.__file__).parent / cfg.network).resolve()
    if frozen != want:
        raise RuntimeError(
            f"vulcan3 is frozen on {frozen} but the config's network is "
            f"{cfg.network!r}: call config_schema.export_network(cfg) "
            "before the first vulcan3 import"
        )


# size-limit-ok: the closures of one forward share its chemistry and RT models
def build_retrieval_forward(cfg: C.Config) -> SimpleNamespace:  # noqa: C901  # counts its closures
    """Build the theta-space forward model for a Config.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    types.SimpleNamespace
        ``native_depth_aux``, ``rt_depth``, ``rt_kwargs``, ``aux_from_y``, the chemistry solves (cold single, cold
        batched, warm batched), ``chem_rtol``, ``y_baseline``, ``wl_um``, ``n_tp``, ``n_chem``, ``chem_extras``,
        ``tp_model``, ``chem``, ``rt``, ``p_art_bar``, ``p_chem_bar``.
    """
    _check_frozen_network(cfg)
    _refuse_unreported_freezes()
    profile = cfg.profile()

    tpm = tp_profile.build_tp_model(cfg)
    n_tp = tpm.n_params
    # Where the composition knobs sit in the chemistry block: stage 1 zeroes them, the warm references carry them.
    extras = C.chem_extras(cfg)
    n_chem = 3 + len(extras)
    comp_idx = (0, 1) + ((3 + extras.index("s_o"),) if "s_o" in extras else ())

    chem = vulcan_chem.build_chem_model(profile, tp_eval=tpm.eval, n_tp_params=n_tp, chem_extras=extras)

    _refuse_condense_inference(chem, cfg)

    _refuse_c_o_bound(chem, cfg)

    rt = exojax_rt.build_rt_model(profile)
    to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)

    mol_cols = {key: chem.sidx[constants.MOLECULES[key]["vulcan"]] for key in rt.molecules}
    h2_col = chem.sidx[constants.BULK_H2_VULCAN]
    he_col = chem.sidx["He"]
    species_masses = chem.species_masses
    # Gas-phase normalization: the condensed reservoir columns (*_l_s, *_l, *_s) are particles. Counting them would
    # dilute every gas VMR and inflate the mean molecular weight (S8_l_s carries 256 g/mol).
    _gas = np.ones(int(np.asarray(species_masses).size))
    for _sp, _i in chem.sidx.items():
        if _sp.endswith(("_l_s", "_l", "_s")):
            _gas[int(_i)] = 0.0
    gas_mask = jnp.asarray(_gas)
    p_art_bar_j = jnp.asarray(rt.p_art_bar)

    def _baseline_composition(C_: Array, axis: int) -> Array:
        """Return ``C_`` with the composition knobs set to 0 along ``axis`` (0: one chem_theta; 1: a stack)."""
        for i in comp_idx:
            C_ = C_.at[i].set(0.0) if axis == 0 else C_.at[:, i].set(0.0)
        return C_

    def chem_stage1(chem_theta: Array) -> Array:
        """Return stage 1 of the cold two-stage map: the column at the retrieved (lnKzz, T-P) with baseline composition.

        Two proposals sharing those entries share this subproblem bit for bit. Stage 1 is not gated.
        """
        return chem.converged_y(_baseline_composition(chem_theta, 0))

    def chem_stage2_diag(chem_theta: Array, y_relaxed: Array) -> tuple[Array, Any]:
        """Return stage 2: warm re-convergence at the proposal's composition, with the certifying ConvDiag."""
        return chem.converged_y(
            chem_theta, warm_y=y_relaxed, lnZ_ref=0.0, c_o_ref=0.0, s_o_ref=0.0, return_conv_diag=True
        )

    def chem_solve_cold_diag(chem_theta: Array) -> tuple[Array, Any]:
        """Return the converged column and its ``ConvDiag``, in two stages.

        (1) Converge at the retrieved T-P and Kzz with baseline composition; (2) apply the composition to that column
        and re-converge warm. A one-stage solve loses the lnZ and C/O response under a retrieved T-P. Every ConvDiag
        field describes the final stage.
        """
        return chem_stage2_diag(chem_theta, chem_stage1(chem_theta))

    lanes = int(cfg.cold_lanes)
    refill_chunk = int(cfg.cold_refill_chunk)

    def _chem_batch_route(C_: Array, active: Array | None, **kw: Any) -> tuple[Array, Any]:
        """Run one batched chemistry stage on the lane queue: the single routing point for every batched solve.

        ``min(cold_lanes, draws)`` lanes; at or above the draw count nothing is refilled, so a narrow call takes the
        production route. A False row of ``active`` is a placeholder whose lane starts done.
        """
        return chem.converged_y_queue(C_, min(lanes, int(C_.shape[0])), chunk=refill_chunk, active=active, **kw)

    def chem_solve_cold_diag_batch(C_: Array, active: Array | None = None) -> tuple[Array, Any]:
        """Return ``chem_solve_cold_diag`` for a stack, on the lane queue.

        The while loop sits above the lane vmap, so a lane is not bit-identical to its solo solve (agreement at the
        convergence scale). Stage 2's ConvDiag certifies each draw. ``active`` masks padded rows in both stages.
        """
        Y1, cd1 = _chem_batch_route(_baseline_composition(C_, 1), active)
        Y2, cd2 = _chem_batch_route(C_, active, warm_y=Y1, lnZ_ref=0.0, c_o_ref=0.0, s_o_ref=0.0)
        return Y2, _with_stage1_freeze(cd1, cd2)

    def chem_solve_warm_diag_full_batch(
        C_: Array, Y: Array, lnZ_ref: Array, c_o_ref: Array, s_o_ref: Array, active: Array | None = None
    ) -> tuple[Array, Any]:
        """Re-converge every row's carried column at its own theta from its own reference composition."""
        return _chem_batch_route(C_, active, warm_y=Y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref, s_o_ref=s_o_ref)

    def _rt_kwargs(cloud: Array | None, lnM: Array | float) -> dict[str, Any]:
        """Return the RT's cloud and mass keywords for this Config's cloud model."""
        kw: dict[str, Any] = {"lnM": jnp.asarray(lnM)}
        if str(cfg.cloud_model) == "deck":
            kw["deck"] = None if cloud is None else jnp.asarray(cloud)[0]
        else:
            kw["cloud"] = cloud
        return kw

    def aux_from_y(y: Array, chem_theta: Array) -> tuple[Any, ...]:
        """Return the ART-grid profiles (vmr, vmr_h2, vmr_he, T_art, mmw_art) from an absolute column."""
        y_gas = y * gas_mask[None, :]
        ymix = y_gas / jnp.sum(y_gas, axis=1, keepdims=True)
        T_art = tpm.eval(chem_theta[n_chem : n_chem + n_tp], p_art_bar_j)
        mmw_v = ymix @ species_masses
        mmw_art = to_art(mmw_v)
        vmr = {key: to_art(ymix[:, col]) for key, col in mol_cols.items()}
        vmr_h2 = to_art(ymix[:, h2_col])
        vmr_he = to_art(ymix[:, he_col])
        return (vmr, vmr_h2, vmr_he, T_art, mmw_art)

    def native_depth_aux(
        chem_theta: Array, lnR0: Array | float, cloud: Array | None = None, lnM: Array | float = 0.0
    ) -> tuple[Array, tuple[Any, ...], Array]:
        """Return (native depth, aux, ok) through the full chain.

        ``aux`` lets a caller re-run the RT alone, without the while loop. ``ok`` is the cold certificate bit, 0 or 1
        and stop-gradient: ``conv_normal`` is False on every cap exit, so no accept_count clause is needed.
        ``cloud`` is None, or ``[log10 kappac0, alphac]``, or ``[log10 P_deck]`` with ``cloud_model="deck"``; ``lnM`` is
        the RT-only mass nuisance.
        """
        chem_theta = jnp.asarray(chem_theta)
        y, cd = chem_solve_cold_diag(chem_theta)
        ok = (jnp.asarray(cd.conv_normal) > 0.5).astype(y.dtype)
        aux = aux_from_y(y, chem_theta)
        vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
        depth = rt.transmission_depth_r(
            vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0), vmr_he=vmr_he, **_rt_kwargs(cloud, lnM)
        )
        return depth, aux, jax.lax.stop_gradient(ok)

    def rt_depth(
        aux: tuple[Any, ...], lnR0: Array | float, cloud: Array | None = None, lnM: Array | float = 0.0
    ) -> Array:
        """Return the RT-only depth at frozen chemistry and T-P profiles."""
        vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
        return rt.transmission_depth_r(
            vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0), vmr_he=vmr_he, **_rt_kwargs(cloud, lnM)
        )

    return SimpleNamespace(
        native_depth_aux=native_depth_aux,
        rt_depth=rt_depth,
        rt_kwargs=_rt_kwargs,
        chem_solve_cold_diag=chem_solve_cold_diag,
        chem_solve_cold_diag_batch=chem_solve_cold_diag_batch,
        chem_solve_warm_diag_full_batch=chem_solve_warm_diag_full_batch,
        chem_rtol=_rtol_accept(chem),
        aux_from_y=aux_from_y,
        y_baseline=np.asarray(chem.y0, dtype=np.float64),
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        n_tp=int(n_tp),
        n_chem=int(n_chem),
        chem_extras=extras,
        tp_model=tpm,
        chem=chem,
        rt=rt,
        p_art_bar=np.asarray(rt.p_art_bar, dtype=np.float64),
        p_chem_bar=np.asarray(chem.p_bar, dtype=np.float64),
    )
