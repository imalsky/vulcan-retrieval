"""The differentiable retrieval forward: (chemistry+T-P params, lnR0) -> native transit
spectrum, composing the *live* VULCAN-JAX chemistry with the ExoJax RT.

    depth(chem_theta, lnR0) = transmission_depth_r(
        bridge( VULCAN.converged_y(chem_theta) ),              # VMR(nz, ni) -> ART grid
        T_art = Guillot(chem_theta[3:]),                        # same T-P on the ART grid
        lnR0 )                                                  # reference-radius nuisance

``chem_theta = [lnZ, dln(C/O), lnKzz, <T-P params>]`` is what the (tp_eval-hooked)
``vulcan_chem.converged_y`` consumes; the T-P sub-vector ``chem_theta[3:3+n_tp]`` is
evaluated by the SAME ExoJax profile on both the VULCAN pressure grid (inside the
chemistry) and the ART grid (here, for the RT), so one self-consistent T(P) drives both.

Everything is pure JAX and supports forward-mode ``jvp`` end-to-end (the VULCAN runner's
``lax.while_loop`` supports jvp but not vjp -- forward-mode is the only route for a
gradient through the chemistry).

Import order matters: ``vulcan_chem`` (env + jax x64) is imported before ``exojax_rt`` /
``tp_profile`` (which import ExoJax).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np

logger = logging.getLogger("retrieval")

# import order matters: vulcan_chem (env + jax x64) before anything exojax
from retrieval_framework.forward import config  # noqa: F401  (hands the engine its data root)
from vulcan_forward import constants
from vulcan_forward import vulcan_chem   # sets env + jax x64; MUST precede exojax imports
import jax
import jax.numpy as jnp

from retrieval_framework import tp_profile   # ExoJax Guillot T-P
from vulcan_forward import exojax_rt     # ExoJax ArtTransPure model
from vulcan_forward import interp_map    # differentiable log-P bridge


def _refuse_condense_inference(chem, cfg) -> None:
    """Refuse a retrieval through a condensing steady state.

    The early ``cfg_overrides`` gate in ``config_schema.validate_config`` catches
    the common case, but a base VULCAN config can default ``use_condense=True``
    without the flag ever appearing in ``cfg_overrides``.
    ``chem.conden_spec`` is the resolved truth (``build_chem_model`` builds it iff
    condensation is active), so gating on it closes that bypass. No nautilus
    retrieval with condensation has been validated: whether condensing columns
    certify, and how they depend on their start under the presolve chain, is
    untested.
    """
    if (getattr(chem, "conden_spec", None) is not None
            and not bool(getattr(cfg, "allow_condense_inference", False))):
        raise ValueError(
            "condensation is active in the RESOLVED VULCAN config "
            f"(vulcan_cfg_name={getattr(cfg, 'vulcan_cfg_name', '?')!r}): no "
            "retrieval with condensation has been validated (certification and "
            "start dependence of condensing columns are untested). Run a "
            "condensation forward model through "
            "vulcan_forward.vulcan_chem.build_chem_model, or set "
            "allow_condense_inference=True."
        )


# _Statics fields a retry program may differ in from the main one: the step
# acceptance tolerance and the cap (and the evolution buffer sized from the cap,
# a length-1 placeholder unless save_evolution).
RETRY_STATICS_FREE = frozenset({"rtol_accept", "count_max", "save_evo_n_max"})


def statics_diff(sa, sb) -> list:
    """Names of the fields in which two VULCAN-JAX ``_Statics`` differ."""
    out = []
    for f in sa._fields:
        va, vb = getattr(sa, f), getattr(sb, f)
        try:
            xa, xb = np.asarray(va), np.asarray(vb)
            same = xa.shape == xb.shape and bool(np.array_equal(xa, xb))
        # broad: a non-array leaf (a callable, a tuple of names) compares by repr
        except Exception:
            same = repr(va) == repr(vb)
        if not same:
            out.append(f)
    return out


def _rtol_accept(chem) -> float | None:
    """The step-acceptance tolerance baked into a chemistry program (VULCAN-JAX
    ``_Statics.rtol_accept``), or None when the engine does not expose it."""
    st = getattr(getattr(chem, "_integ", None), "_statics", None)
    return None if st is None else float(st.rtol_accept)


def build_retrieval_forward(cfg: Any) -> SimpleNamespace:
    """Build the theta-space forward model for a Config.

    Returns SimpleNamespace with:
        native_depth_aux(chem_theta, lnR0) -> ((n_nu,) native transit depth, aux, ok)
        rt_depth, aux_from_y, the cold and warm chemistry solves (single and batched)
        wl_um     : (n_nu,) native wavelengths (um)
        n_tp      : number of T-P parameters
        tp_model  : the tp_profile object (eval)
        chem, rt  : the engine's chemistry and RT models
    """
    profile = cfg.profile()

    # T-P first (grid-agnostic evaluator), then hook it into the chemistry so the VULCAN
    # column re-converges under the retrieved T-P.
    tpm = tp_profile.build_tp_model(cfg)
    n_tp = tpm.n_params

    chem = vulcan_chem.build_chem_model(profile, tp_eval=tpm.eval, n_tp_params=n_tp)

    # Condensation is an engine (forward-model) capability only; refuse it here on
    # the RESOLVED config (closes the base-config bypass of the early
    # cfg_overrides gate).
    _refuse_condense_inference(chem, cfg)

    # chem_retry (the presolve chain's certifying program): the SAME chemistry at
    # another step tolerance and cap. rtol is
    # a compile-time static of the step acceptance, so it is a second program
    # (its own runner, compiled on its first solve). Refuse anything else
    # differing: only numerical settings may change.
    chem_retry = None
    if bool(getattr(cfg, "chem_retry", False)):
        from retrieval_framework.config_schema import retry_profile
        chem_retry = vulcan_chem.build_chem_model(retry_profile(cfg),
                                                  tp_eval=tpm.eval, n_tp_params=n_tp)
        st_main = getattr(getattr(chem, "_integ", None), "_statics", None)
        st_retry = getattr(getattr(chem_retry, "_integ", None), "_statics", None)
        if st_main is not None and st_retry is not None:
            extra = sorted(set(statics_diff(st_main, st_retry)) - RETRY_STATICS_FREE)
            if extra:
                raise RuntimeError(
                    "chem_retry: the retry chemistry differs from the main program "
                    f"beyond the step tolerance and the cap ({extra}); refusing")
        logger.info(f"[chem retry] retry program: rtol {_rtol_accept(chem)} -> "
                    f"{_rtol_accept(chem_retry)}, count_max {int(chem.count_max)} -> "
                    f"{int(chem_retry.count_max)} (solves and certifies every draw "
                    "after the presolve)")

    # Fail fast if the C/O prior can leave the fixed-O knob's validity range: b_z
    # (the O-only compensation factor) must stay positive, else prior-corner columns
    # get negative O-carrier abundances that the runner's clip silently turns into a
    # wrong-inventory, finite-likelihood state. First-order guard: the bound is
    # computed on the baseline column (hot stage-2 columns can bind slightly tighter).
    if str(profile["co_mode"]) == "fixed_O":
        bound = float(chem.co_bz_bound)
        hi = float(cfg.prior_c_o[1])
        if hi >= bound:
            raise ValueError(
                f"prior_c_o upper bound {hi:+.3f} reaches the fixed-O b_z positivity "
                f"bound {bound:+.3f} (baseline column): beyond it the O-only "
                "compensation goes negative and the solver clips the inventory into "
                f"silently-wrong states. Lower prior_c_o[1] below {bound:.3f} (with "
                "margin), or use co_mode='proxy'.")

    rt = exojax_rt.build_rt_model(profile)
    to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)

    mol_cols = {key: chem.sidx[constants.MOLECULES[key]["vulcan"]] for key in rt.molecules}
    h2_col = chem.sidx[constants.BULK_H2_VULCAN]
    he_col = chem.sidx["He"]          # H2-He CIA partner (He is inert in the network)
    species_masses = chem.species_masses
    # GAS-phase normalization: the network's condensed-phase reservoir columns
    # (*_l_s) are particles, not gas. Counting them dilutes every gas VMR and
    # inflates the RT mean molecular weight as if the condensate were vapor
    # (S8_l_s carries ~256 g/mol). The condensate's aerosol opacity is excluded:
    # that is the cloud deck's job.
    _gas = np.ones(int(np.asarray(species_masses).size))
    for _sp, _i in chem.sidx.items():
        if _sp.endswith("_l_s"):
            _gas[int(_i)] = 0.0
    gas_mask = jnp.asarray(_gas)
    p_art_bar_j = jnp.asarray(rt.p_art_bar)

    def chem_stage1(chem_theta):
        """Stage 1 of the cold two-stage map: the converged column at the
        retrieved (lnKzz, T-P) with baseline composition. A function of
        chem_theta[2:] only (theta[0:2] are overwritten with 0), so two proposals
        sharing theta[2:] share this subproblem bit for bit. No ConvDiag: stage 1
        is not gated."""
        return chem.converged_y(chem_theta.at[0].set(0.0).at[1].set(0.0))

    def chem_stage2_diag(chem_theta, y_relaxed):
        """Stage 2: warm re-convergence of y_relaxed at the proposal's own
        (lnZ, c_o); returns (y, ConvDiag), the ConvDiag that certifies the draw."""
        return chem.converged_y(chem_theta, warm_y=y_relaxed, lnZ_ref=0.0,
                                c_o_ref=0.0, return_conv_diag=True)

    def chem_solve_cold_diag(chem_theta):
        """Converged absolute column y (nz, ni) and its ``ConvDiag``, in two
        stages: (1) converge at the retrieved T-P/Kzz with baseline composition;
        (2) apply the lnZ / C-O scaling to that column and re-converge warm. A
        one-stage solve loses the lnZ/C-O response under a retrieved T-P.

        Every ConvDiag field, ``accept_count`` included, describes the final
        stage, the state ``y`` is; stage 1's step count says nothing about the
        draw's own column. accept_count alone is not a convergence
        test; gate on ``conv_normal`` too.

        Used by native_depth_aux (the synthetic truth spectrum and the
        plots); the batched evaluators use chem_solve_cold_diag_batch and
        reject a non-converged cold proposal as a warm one is rejected."""
        return chem_stage2_diag(chem_theta, chem_stage1(chem_theta))

    lanes = int(cfg.cold_lanes)
    refill_chunk = int(cfg.cold_refill_chunk)

    def _chem_batch_route(ch, C, active, **kw):
        """One batched chemistry stage -> ``(y, ConvDiag)``: the single routing
        point for every batched solve, cold stage or warm continuation, on
        ``ch`` (the main chemistry, or the chem_retry program). Every call runs
        the lane queue (``ch.converged_y_queue``) at ``min(cold_lanes, draws)``
        lanes; with the lane count at or above the draw count ``run_queue``
        refills nothing and runs the plain batch's ticks, so a narrow call (a
        padded last chunk, a few model spectra) takes the production route.
        ``kw`` carries ``warm_y`` / ``lnZ_ref`` / ``c_o_ref``.

        ``active`` ((N,) bool, run_nautilus's padded and out-of-window rows): a
        False row is a placeholder whose lane starts done and takes no step;
        the caller discards it. None is the program without the mask."""
        return ch.converged_y_queue(C, min(lanes, int(C.shape[0])),
                                    chunk=refill_chunk, active=active, **kw)

    def chem_solve_cold_diag_batch(C, active=None):
        """``chem_solve_cold_diag`` for a STACK of chem_thetas ``C`` (N, n_chem_tp):
        ``(y, ConvDiag)`` with a leading particle axis on every field.

        Same map, on the lane queue (``_chem_batch_route``). The while loop sits
        above the lane vmap, so photolysis and the geometry refresh fire once
        per cadence for the whole batch instead of on every lane every
        iteration. A lane is NOT bit-identical to its solo solve: the cadence
        rides the loop's iteration tick, and a refilled draw starts at the tick
        its lane was freed at (agreement at the convergence scale). Stage 1 is
        not gated (it carries no certificate); stage 2's ConvDiag certifies each
        draw. ``active`` masks padded rows in both stages."""
        Y1, _cd = _chem_batch_route(chem, C.at[:, 0].set(0.0).at[:, 1].set(0.0), active)
        return _chem_batch_route(chem, C, active, warm_y=Y1, lnZ_ref=0.0, c_o_ref=0.0)

    def chem_solve_warm_diag_full_batch(C, Y, lnZ_ref, c_o_ref, active=None):
        """Every row's carried column ``Y`` (N, nz, ni) re-converged at its own
        ``C`` (N, n_chem_tp) from its own reference composition ``lnZ_ref`` /
        ``c_o_ref`` ((N,) arrays), under the cold count_max: run_nautilus's
        anchored warm starts. ``active`` masks padded rows."""
        return _chem_batch_route(chem, C, active, warm_y=Y, lnZ_ref=lnZ_ref,
                                 c_o_ref=c_o_ref)

    chem_solve_warm_diag_full_batch_retry = None
    if chem_retry is not None:
        def chem_solve_warm_diag_full_batch_retry(C, Y, lnZ_ref, c_o_ref, active=None):
            """``chem_solve_warm_diag_full_batch`` on the chem_retry program: the
            presolve chain's certifying pass, under the retry count_max."""
            return _chem_batch_route(chem_retry, C, active, warm_y=Y, lnZ_ref=lnZ_ref,
                                     c_o_ref=c_o_ref)

    def aux_from_y(y, chem_theta):
        """ART-grid primal profiles aux = (vmr dict, vmr_h2, vmr_he, T_art, mmw_art)
        from an absolute column y (nz, ni). Differentiable and cheap (normalize +
        T-P eval + log-P interpolation); the RT consumes exactly this tuple."""
        y_gas = y * gas_mask[None, :]
        ymix = y_gas / jnp.sum(y_gas, axis=1, keepdims=True)
        T_art = tpm.eval(chem_theta[3:3 + n_tp], p_art_bar_j)      # (nlayer,)
        mmw_v = ymix @ species_masses                              # (nz,)
        mmw_art = to_art(mmw_v)
        vmr = {key: to_art(ymix[:, col]) for key, col in mol_cols.items()}
        vmr_h2 = to_art(ymix[:, h2_col])
        vmr_he = to_art(ymix[:, he_col])
        return (vmr, vmr_h2, vmr_he, T_art, mmw_art)

    def native_depth_aux(chem_theta, lnR0, cloud=None):
        """Full chain -> (native depth, aux, ok) where aux = (vmr, vmr_h2, vmr_he,
        T_art, mmw_art) are the ART-grid primal profiles. The aux lets a caller take
        an RT-ONLY jvp (e.g. d/dlnR0 or the cloud parameters, which do not touch the
        chemistry) without re-running or re-differentiating the VULCAN while_loop.

        ``ok`` (float 0/1, stop_gradient'ed) is the cold certificate: the runner's
        canonical convergence bit, the predicate the staged batch evaluators
        apply. No accept_count clause: the runner stops on accept_count >
        count_max with convergence first, so a certified solve can exit at cap
        or cap + 1, and conv_normal is False on every cap exit. The synthetic
        injection refuses an uncertified truth on it
        (pipeline.generate_observations). Reading the diag is free: every
        field rides the primal carry.

        ``cloud`` is None (off) or a (2,) array [log10 kappac0, alphac] for the
        ExoJax powerlaw_clouds term (see exojax_rt /
        vulcan_forward.constants.CLOUD_NUC0)."""
        chem_theta = jnp.asarray(chem_theta)
        y, cd = chem_solve_cold_diag(chem_theta)                   # (nz, ni) absolute
        ok = (jnp.asarray(cd.conv_normal) > 0.5).astype(y.dtype)
        aux = aux_from_y(y, chem_theta)                            # ART-grid profiles
        vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
        depth = rt.transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0),
                                        vmr_he=vmr_he, cloud=cloud)
        return depth, aux, jax.lax.stop_gradient(ok)

    def rt_depth(aux, lnR0, cloud=None):
        """RT-only depth at frozen chemistry/T-P profiles (for the cheap lnR0/cloud jvps)."""
        vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
        return rt.transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0),
                                       vmr_he=vmr_he, cloud=cloud)

    return SimpleNamespace(
        native_depth_aux=native_depth_aux,
        rt_depth=rt_depth,
        chem_solve_cold_diag=chem_solve_cold_diag,
        chem_solve_cold_diag_batch=chem_solve_cold_diag_batch,
        chem_solve_warm_diag_full_batch=chem_solve_warm_diag_full_batch,
        # chem_retry (None when off): the presolve chain's certifying program and
        # its warm solve
        chem_retry=chem_retry,
        chem_rtol=_rtol_accept(chem),
        chem_retry_rtol=None if chem_retry is None else _rtol_accept(chem_retry),
        chem_solve_warm_diag_full_batch_retry=chem_solve_warm_diag_full_batch_retry,
        aux_from_y=aux_from_y,
        y_baseline=np.asarray(chem.y0, dtype=np.float64),
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        n_tp=int(n_tp),
        tp_model=tpm,
        chem=chem, rt=rt,
        p_art_bar=np.asarray(rt.p_art_bar, dtype=np.float64),
    )
