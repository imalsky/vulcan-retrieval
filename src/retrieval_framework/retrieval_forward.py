"""The retrieval forward: chemistry and T-P parameters plus lnR0 to the native transit spectrum.

``chem_theta = [lnZ, dln(C/O), lnKzz, <extras>, <T-P>]``; one T(P) drives the chemistry on the VULCAN grid and the RT on
the ART grid. Pure JAX and forward-mode differentiable: the runner's ``lax.while_loop`` has no vjp. ``vulcan_chem`` is
imported before ``exojax_rt`` and ``tp_profile``, in that order on purpose.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

__all__ = ["RetrievalForward", "build_retrieval_forward"]

logger = logging.getLogger("retrieval")

from retrieval_framework.forward import config  # noqa: F401  (hands the engine its data root)
from retrieval_framework import config_schema as C
from vulcan_forward import constants
from vulcan_forward import vulcan_chem  # must precede exojax
import jax
import jax.numpy as jnp

from retrieval_framework import tp_profile
from vulcan3.state import TERM_DT_MIN_FROZEN
from vulcan_forward import exojax_rt
from vulcan_forward import interp_map

if TYPE_CHECKING:
    from types import SimpleNamespace

    from jax import Array
    from jax.typing import ArrayLike


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
    return float(chem._integ.statics.rtol_accept)


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


def _with_stage1_freeze(cd1: Any, cd2: Any) -> Any:
    """Return stage 2's ``ConvDiag``, reporting a stage 1 freeze: stage 1 is not gated, so it would go unseen."""
    frozen = cd1.termination_reason == TERM_DT_MIN_FROZEN
    return cd2._replace(
        termination_reason=jnp.where(frozen, cd1.termination_reason, cd2.termination_reason),
        last_delta=jnp.where(frozen, cd1.last_delta, cd2.last_delta),
    )


def _check_frozen_network(cfg: C.Config) -> None:
    """Refuse an engine frozen on another network than the config's: it would solve other chemistry."""
    import vulcan3
    from vulcan3 import chem_funs

    frozen = Path(chem_funs.NETWORK.network_path).resolve()
    want = (Path(vulcan3.__file__).parent / cfg.network).resolve()
    if frozen != want:
        raise RuntimeError(
            f"vulcan3 is frozen on {frozen} but the config's network is "
            f"{cfg.network!r}: call config_schema.export_network(cfg) "
            "before the first vulcan3 import"
        )


class RetrievalForward:
    """One Config's forward: the two-stage chemistry, the lane queue and the RT on the ART-grid profiles it gives.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.
    tpm : types.SimpleNamespace
        The T-P model.
    extras : tuple of str
        The optional chemistry entries.
    chem : types.SimpleNamespace
        The ``build_chem_model`` result.
    rt : types.SimpleNamespace
        The ``build_rt_model`` result.

    Attributes
    ----------
    chem, rt, tp_model
        The engine's models, and the T-P model.
    chem_rtol, y_baseline, wl_um, p_art_bar, p_chem_bar
        The chemistry program's tolerance, the build's column, the native wavelengths (um) and both grids (bar).
    n_tp, n_chem, chem_extras
        The sizes of the T-P and chemistry blocks, and the optional chemistry entries.
    """

    def __init__(
        self, cfg: C.Config, tpm: Any, extras: tuple[str, ...], chem: SimpleNamespace, rt: SimpleNamespace
    ) -> None:
        self._cfg = cfg
        self.tp_model = tpm
        self.chem = chem
        self.rt = rt
        self.chem_extras = extras
        self.n_tp = int(tpm.n_params)
        self.n_chem = 3 + len(extras)
        # Where the composition knobs sit in the chemistry block: stage 1 zeroes them, the warm references carry them.
        self._comp_idx = (0, 1) + ((3 + extras.index("s_o"),) if "s_o" in extras else ())
        self._to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)
        self._mol_cols = {key: chem.sidx[constants.MOLECULES[key]["vulcan"]] for key in rt.molecules}
        self._h2_col = chem.sidx[constants.BULK_H2_VULCAN]
        self._he_col = chem.sidx["He"]
        self._species_masses = chem.species_masses
        # Gas-phase normalization: the condensed reservoir columns (*_l_s, *_l, *_s) are particles. Counting them would
        # dilute every gas VMR and inflate the mean molecular weight (S8_l_s carries 256 g/mol).
        gas = np.ones(int(np.asarray(chem.species_masses).size))
        for species, i in chem.sidx.items():
            if species.endswith(("_l_s", "_l", "_s")):
                gas[int(i)] = 0.0
        self._gas_mask = jnp.asarray(gas)
        self._p_art_bar_j = jnp.asarray(rt.p_art_bar)
        self._lanes = int(cfg.cold_lanes)
        self._refill_chunk = int(cfg.cold_refill_chunk)
        self.chem_rtol = _rtol_accept(chem)
        self.y_baseline = np.asarray(chem.y0, dtype=np.float64)
        self.wl_um = np.asarray(rt.wl_um, dtype=np.float64)
        self.p_art_bar = np.asarray(rt.p_art_bar, dtype=np.float64)
        self.p_chem_bar = np.asarray(chem.p_bar, dtype=np.float64)

    def _baseline_composition(self, C_: Array, axis: int) -> Array:
        """Return ``C_`` with the composition knobs set to 0 along ``axis`` (0: one chem_theta; 1: a stack)."""
        for i in self._comp_idx:
            C_ = C_.at[i].set(0.0) if axis == 0 else C_.at[:, i].set(0.0)
        return C_

    def chem_solve_cold_diag(self, chem_theta: Array) -> tuple[Array, Any]:
        """Return the converged column and its ``ConvDiag``, in two stages.

        (1) Converge at the retrieved T-P and Kzz with baseline composition: two proposals sharing those entries share
        this subproblem bit for bit, and it is not gated. (2) Apply the composition to that column and re-converge warm.
        A one-stage solve loses the lnZ and C/O response under a retrieved T-P. Every ConvDiag field describes the final
        stage.

        Parameters
        ----------
        chem_theta : jax.Array [n_chem + n_tp]
            The chemistry and T-P entries.

        Returns
        -------
        y : jax.Array [nz, ni]
            The converged column, number densities in cm^-3.
        diag : ConvDiag
            Stage 2's convergence record.
        """
        y_relaxed = self.chem.converged_y(self._baseline_composition(chem_theta, 0))
        return self.chem.converged_y(
            chem_theta, warm_y=y_relaxed, lnZ_ref=0.0, c_o_ref=0.0, s_o_ref=0.0, return_conv_diag=True
        )

    def _chem_batch_route(self, C_: Array, active: ArrayLike | None, **kw: Any) -> tuple[Array, Any]:
        """Run one batched chemistry stage on the lane queue: the single routing point for every batched solve.

        ``min(cold_lanes, draws)`` lanes; at or above the draw count nothing is refilled, so a narrow call takes the
        production route. A False row of ``active`` is a placeholder whose lane starts done.
        """
        lanes = min(self._lanes, int(C_.shape[0]))
        return self.chem.converged_y_queue(C_, lanes, chunk=self._refill_chunk, active=active, **kw)

    def chem_solve_cold_diag_batch(self, C_: Array, active: ArrayLike | None = None) -> tuple[Array, Any]:
        """Return ``chem_solve_cold_diag`` for a stack, on the lane queue.

        The while loop sits above the lane vmap, so a lane is not bit-identical to its solo solve (agreement at the
        convergence scale). Stage 2's ConvDiag certifies each draw.

        Parameters
        ----------
        C_ : jax.Array [N, n_chem + n_tp]
            The chemistry and T-P entries of each draw.
        active : array-like of bool [N], optional
            False marks a padded row, in both stages.

        Returns
        -------
        Y : jax.Array [N, nz, ni]
            The converged columns, cm^-3.
        diag : ConvDiag
            Stage 2's record per row, with a stage 1 freeze carried into its termination reason.
        """
        Y1, cd1 = self._chem_batch_route(self._baseline_composition(C_, 1), active)
        Y2, cd2 = self._chem_batch_route(C_, active, warm_y=Y1, lnZ_ref=0.0, c_o_ref=0.0, s_o_ref=0.0)
        return Y2, _with_stage1_freeze(cd1, cd2)

    def chem_solve_warm_diag_full_batch(
        self, C_: Array, Y: Array, lnZ_ref: Array, c_o_ref: Array, s_o_ref: Array, active: ArrayLike | None = None
    ) -> tuple[Array, Any]:
        """Re-converge every row's carried column at its own theta from its own reference composition.

        Parameters
        ----------
        C_ : jax.Array [N, n_chem + n_tp]
            The chemistry and T-P entries of each draw.
        Y : jax.Array [N, nz, ni]
            The carried columns, cm^-3.
        lnZ_ref : jax.Array [N]
            The lnZ each carried column was solved at.
        c_o_ref : jax.Array [N]
            Its dln(C/O).
        s_o_ref : jax.Array [N]
            Its s_o.
        active : array-like of bool [N], optional
            False marks a padded row.

        Returns
        -------
        Y : jax.Array [N, nz, ni]
            The converged columns, cm^-3.
        diag : ConvDiag
            The convergence record per row.
        """
        return self._chem_batch_route(C_, active, warm_y=Y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref, s_o_ref=s_o_ref)

    def rt_kwargs(self, cloud: Array | None, lnM: Array | float) -> dict[str, Any]:
        """Return the RT's cloud and mass keywords for this Config's cloud model."""
        kw: dict[str, Any] = {"lnM": jnp.asarray(lnM)}
        if str(self._cfg.cloud_model) == "deck":
            kw["deck"] = None if cloud is None else jnp.asarray(cloud)[0]
        else:
            kw["cloud"] = cloud
        return kw

    def aux_from_y(self, y: Array, chem_theta: Array) -> tuple[Any, ...]:
        """Return the ART-grid profiles the RT takes, from a column of number densities.

        Parameters
        ----------
        y : jax.Array [nz, ni]
            The column, cm^-3; the condensed reservoirs are left out of the gas.
        chem_theta : jax.Array [n_chem + n_tp]
            The chemistry and T-P entries.

        Returns
        -------
        tuple
            ``(vmr, vmr_h2, vmr_he, T_art, mmw_art)``: a dict of [nlayer] mixing ratios per RT molecule, the H2 and He
            mixing ratios [nlayer], the temperature [nlayer] in K and the mean molecular weight [nlayer] in amu.
        """
        to_art = self._to_art
        y_gas = y * self._gas_mask[None, :]
        ymix = y_gas / jnp.sum(y_gas, axis=1, keepdims=True)
        T_art = self.tp_model.eval(chem_theta[self.n_chem : self.n_chem + self.n_tp], self._p_art_bar_j)
        mmw_v = ymix @ self._species_masses
        mmw_art = to_art(mmw_v)
        vmr = {key: to_art(ymix[:, col]) for key, col in self._mol_cols.items()}
        vmr_h2 = to_art(ymix[:, self._h2_col])
        vmr_he = to_art(ymix[:, self._he_col])
        return (vmr, vmr_h2, vmr_he, T_art, mmw_art)

    def native_depth_aux(
        self, chem_theta: Array, lnR0: Array | float, cloud: Array | None = None, lnM: Array | float = 0.0
    ) -> tuple[Array, tuple[Any, ...], Array]:
        """Return the native depth through the full chain, with the profiles and the cold certificate.

        Parameters
        ----------
        chem_theta : jax.Array [n_chem + n_tp]
            The chemistry and T-P entries.
        lnR0 : float
            ln of the radius multiple at ``p_ref_bar``.
        cloud : jax.Array [2] or [1], optional
            ``[log10 kappac0, alphac]``, or ``[log10 P_deck]`` with ``cloud_model="deck"``.
        lnM : float, optional
            ln of the planet-mass multiple, RT-only.

        Returns
        -------
        depth : jax.Array [nband]
            The transit depth on the native bands.
        aux : tuple
            ``aux_from_y``'s profiles, so a caller can re-run the RT alone, without the while loop.
        ok : jax.Array
            The cold certificate, 0 or 1 and stop-gradient: ``conv_normal`` is False on every cap exit, so no
            accept_count clause is needed.
        """
        chem_theta = jnp.asarray(chem_theta)
        y, cd = self.chem_solve_cold_diag(chem_theta)
        ok = (jnp.asarray(cd.conv_normal) > 0.5).astype(y.dtype)
        aux = self.aux_from_y(y, chem_theta)
        return self.rt_depth(aux, lnR0, cloud, lnM), aux, jax.lax.stop_gradient(ok)

    def rt_depth(
        self, aux: tuple[Any, ...], lnR0: Array | float, cloud: Array | None = None, lnM: Array | float = 0.0
    ) -> Array:
        """Return the RT-only depth at frozen chemistry and T-P profiles.

        Parameters
        ----------
        aux : tuple
            ``aux_from_y``'s profiles.
        lnR0 : float
            ln of the radius multiple at ``p_ref_bar``.
        cloud : jax.Array [2] or [1], optional
            The cloud block, as in ``native_depth_aux``.
        lnM : float, optional
            ln of the planet-mass multiple.

        Returns
        -------
        jax.Array [nband]
            The transit depth on the native bands.
        """
        vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
        return self.rt.transmission_depth_r(
            vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0), vmr_he=vmr_he, **self.rt_kwargs(cloud, lnM)
        )


def build_retrieval_forward(cfg: C.Config) -> RetrievalForward:
    """Build the theta-space forward model for a Config, refusing what the retrieval cannot trust.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    RetrievalForward
        The forward.
    """
    _check_frozen_network(cfg)
    profile = cfg.profile()

    tpm = tp_profile.build_tp_model(cfg)
    extras = C.chem_extras(cfg)
    chem = vulcan_chem.build_chem_model(profile, tp_eval=tpm.eval, n_tp_params=tpm.n_params, chem_extras=extras)

    _refuse_condense_inference(chem, cfg)

    _refuse_c_o_bound(chem, cfg)

    rt = exojax_rt.build_rt_model(profile)
    return RetrievalForward(cfg, tpm, extras, chem, rt)
