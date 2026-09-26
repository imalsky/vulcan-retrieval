"""The differentiable retrieval forward: (chemistry+T-P params, lnR0) -> native transit
spectrum, composing the *live* VULCAN-JAX chemistry with the ExoJax RT.

    depth(chem_theta, lnR0) = transmission_depth_r(
        bridge( VULCAN.converged_y(chem_theta) ),              # VMR(nz, ni) -> ART grid
        T_art = Guillot(chem_theta[3:]),                        # same T-P on the ART grid
        lnR0 )                                                  # reference-radius nuisance

``chem_theta = [lnZ, dln(C/O), lnKzz, <T-P params>]`` is exactly what the (tp_eval-hooked)
``vulcan_chem.converged_y`` consumes; the T-P sub-vector ``chem_theta[3:3+n_tp]`` is
evaluated by the SAME ExoJax profile on both the VULCAN pressure grid (inside the
chemistry) and the ART grid (here, for the RT), so one self-consistent T(P) drives both.

Everything is pure JAX and supports forward-mode ``jvp`` end-to-end (the VULCAN runner's
``lax.while_loop`` supports jvp but not vjp -- forward-mode is the only route, which is
also why the retrieval's MALA gradient is built from forward-mode jvps).

Import order matters: ``vulcan_chem`` (env + jax x64) is imported before ``exojax_rt`` /
``tp_profile`` (which import ExoJax).
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np

logger = logging.getLogger("retrieval")

# import order is load-bearing: vulcan_chem (env + jax x64) before anything exojax
from retrieval_framework.forward import config  # noqa: F401  (hands the engine its data root)
from vulcan_forward import constants
from vulcan_forward import vulcan_chem   # sets env + jax x64; MUST precede exojax imports
import jax
import jax.numpy as jnp

from retrieval_framework import tp_profile   # ExoJax Guillot / power-law T-P
from vulcan_forward import exojax_rt     # ExoJax ArtTransPure model
from vulcan_forward import interp_map    # differentiable log-P bridge


def _refuse_condense_inference(chem, cfg) -> None:
    """Refuse gradient-MALA inference through a condensing steady state.

    The early ``cfg_overrides`` gate in ``config_schema.validate_config`` catches
    the common case, but a base VULCAN config can default ``use_condense=True``
    (e.g. ``Earth.yaml``) without the flag ever appearing in ``cfg_overrides``.
    ``chem.conden_spec`` is the RESOLVED truth (``build_chem_model`` builds it iff
    condensation is actually active), so gating on it closes that bypass. The
    pinned condensation state is not reliably differentiable (0.91 rel jvp-vs-FD
    on pinned species) and gradient-MALA is the default mutation kernel, so an
    inference run would sample against unreliable gradients (VULCAN-JAX
    notes §2.5-2.6).
    """
    if (getattr(chem, "conden_spec", None) is not None
            and bool(getattr(cfg, "run_inference", False))
            and not bool(getattr(cfg, "allow_condense_inference", False))):
        raise ValueError(
            "condensation is active in the RESOLVED VULCAN config "
            f"(vulcan_cfg_name={getattr(cfg, 'vulcan_cfg_name', '?')!r}) but "
            "run_inference=True: gradient-MALA inference through the "
            "condensing+pinned steady state is not validated (the pinned-species "
            "forward-mode tangent disagrees with finite differences at order "
            "unity, 0.91 relative). Run condensation as a FORWARD model "
            "(run_inference=False), or set allow_condense_inference=True only "
            "with an independently validated gradient. This gate reads the "
            "resolved conden_spec, so it also catches use_condense=True inherited "
            "from the base config (cfg_overrides need not restate it)."
        )


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

    # Condensation is a FORWARD-model capability only; refuse inference on the
    # RESOLVED config (closes the base-config bypass of the early cfg_overrides
    # gate). VULCAN-JAX notes §2.5-2.6.
    _refuse_condense_inference(chem, cfg)

    # Surface a failed warm-up check. NOT a refusal: nothing consumes the
    # warm-up column (y_baseline is the pre-loop column, and cold solves start
    # from the equilibrium seed), and every draw certifies itself; the offline
    # smoke preset builds on a cap-exit warm-up (longdy~0.11 at nz=30) and
    # passes its gradient checks. A failure only flags a configuration that
    # may not converge (vulcan-forward notes §2).
    if bool(cfg.run_inference) and not bool(
            getattr(chem, "baseline_conv_normal", True)):
        logger.warning(
            "the chemistry warm-up solve failed its check (see the [chem] "
            "WARNING above for end_case/termination_reason/longdy). Its column is not "
            "used and every draw certifies itself, but this configuration may "
            "not converge: check the T-P window / Kzz / dt_max settings.")

    # Fail fast if the C/O prior can leave the fixed-O knob's validity range: b_z
    # (the O-only compensation factor) must stay positive, else prior-corner columns
    # get negative O-carrier abundances that the runner's clip silently turns into a
    # wrong-inventory, finite-likelihood state. First-order guard: the bound is
    # computed on the baseline column (hot stage-2 columns can bind slightly tighter).
    if cfg.infer_c_o and str(profile["co_mode"]) == "fixed_O":
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
    # (S8_l_s carries ~256 g/mol). The condensate's aerosol OPACITY stays
    # deliberately excluded -- that is the cloud deck's job.
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
        is not gated (notes §2.12)."""
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
        one-stage solve loses the lnZ/C-O response under a retrieved T-P (notes
        §2.1, §2.12).

        Every ConvDiag field, ``accept_count`` included, describes the final
        stage, the state ``y`` is; stage 1's step count says nothing about the
        draw's own column (notes §1.1). accept_count alone is not a convergence
        test; gate on ``conv_normal`` too.

        Used by native_depth_aux (scalar likelihood and block gradient); the
        batched evaluators use chem_solve_cold_diag_batch and reject a
        non-converged cold proposal as a warm one is rejected."""
        return chem_stage2_diag(chem_theta, chem_stage1(chem_theta))

    def _chem_batch_route(C, **kw):
        """One batched chemistry stage -> ``(y, ConvDiag)``: the single routing
        point for every batched solve, cold stage or warm continuation (the
        knob keeps its ``cold_lanes`` name; ``kw`` carries ``warm_y`` /
        ``lnZ_ref`` / ``c_o_ref`` / ``warm_cap``). ONE route per config:
        ``cfg.cold_lanes > 0`` ALWAYS takes the lane queue, at
        ``min(cold_lanes, draws)`` lanes -- with the lane count at or above the
        draw count ``run_queue`` refills nothing and runs the plain batch's
        ticks. A narrow call (the certificate's 4-particle replay, a
        validate_warm chunk) therefore takes the same route as the production
        run instead of silently switching to the lockstep batch.
        ``cold_lanes = 0`` is the lockstep batch, call for call."""
        lanes = int(cfg.cold_lanes)
        if lanes > 0:
            return chem.converged_y_queue(C, min(lanes, int(C.shape[0])),
                                          chunk=int(cfg.cold_refill_chunk), **kw)
        return chem.converged_y_batch(C, return_conv_diag=True, **kw)

    def chem_solve_cold_diag_batch(C):
        """``chem_solve_cold_diag`` for a STACK of chem_thetas ``C`` (N, n_chem_tp):
        ``(y, ConvDiag)`` with a leading particle axis on every field.

        Same map, run through the solver's BATCHED runner
        (``vulcan_chem.converged_y_batch``): the while loop sits above the lane
        vmap, so photolysis and the geometry refresh fire once per cadence for
        the whole batch instead of on every lane every iteration. Each lane
        freezes at its own exit, so a particle's column does not depend on the
        others, but it is NOT bit-identical to its solo solve: the cadence rides
        the loop's iteration tick (agreement at the convergence scale, 5.4e-5
        over ymix > 1e-10 on vulcan-jax's HD189 batch, its notes 2.9). The cold
        GRADIENT path takes the same route, one ``jax.jvp`` per direction
        through the stage twins below, and so do the batched WARM
        continuations (``chem_solve_warm_diag_batch``).

        With ``cfg.cold_lanes`` above 0 EVERY cold batch -- both stages, any
        width -- runs on ``min(cold_lanes, draws)`` lanes with refill
        (``vulcan_chem.converged_y_queue``): a lane that certifies takes the
        next draw inside the same while loop, so wall time follows total work /
        lanes instead of the slowest draw. One route per config, so a narrow
        replay agrees with the run. cold_lanes = 0 keeps the single lockstep
        batch, call for call."""
        return chem_stage2_diag_batch(C, chem_stage1_batch(C))

    def chem_stage1_batch(C):
        """``chem_stage1`` for a STACK of chem_thetas ``C`` (N, n_chem_tp) ->
        Y1 (N, nz, ni), through ``_chem_batch_route``. The batched twin the cold
        GRADIENT path jvp's through: one solve for the whole cloud instead of
        a per-particle vmap of solves, so the photolysis and geometry
        cadences follow the loop tick shared by the cloud. Not gated (stage 1
        carries no certificate)."""
        Y1, _cd = _chem_batch_route(C.at[:, 0].set(0.0).at[:, 1].set(0.0))
        return Y1

    def chem_stage2_diag_batch(C, Y1):
        """``chem_stage2_diag`` for a STACK: warm re-convergence of each
        particle's own stage-1 column ``Y1`` (N, nz, ni) at its (lnZ, c_o) ->
        ``(y (N, nz, ni), ConvDiag with a leading particle axis)``, the
        ConvDiag that certifies each draw."""
        return _chem_batch_route(C, warm_y=Y1, lnZ_ref=0.0, c_o_ref=0.0)

    def chem_solve_warm_diag(chem_theta, y_warm, lnZ_ref, c_o_ref):
        """Converged absolute column y (nz, ni) and its ``ConvDiag`` by warm
        continuation from a converged column ``y_warm`` whose inventory
        corresponds to (lnZ_ref, c_o_ref); the lnZ / C-O scalings apply
        incrementally (theta[0]-lnZ_ref, theta[1]-c_o_ref). Runs under the
        warm_count_max cap (warm_cap=True). The ConvDiag lets a caller reject a
        proposal that is not at a certified steady state (warm_count_max-exhausted
        or a stall/budget exit) before trusting its jvp; it rides the primal
        carry. The per-particle map for mala_reversibility and the tests; the
        SMC runs chem_solve_warm_diag_batch."""
        return chem.converged_y(chem_theta, warm_y=y_warm,
                                lnZ_ref=lnZ_ref, c_o_ref=c_o_ref,
                                return_conv_diag=True, warm_cap=True)

    def chem_solve_warm_diag_full(chem_theta, y_warm, lnZ_ref, c_o_ref):
        """chem_solve_warm_diag WITHOUT the mutation cap: runs under the cold
        count_max. For the INIT gradient pass only (pipeline._init_state phase 2):
        its inputs are phase-1 SURVIVORS re-certifying from their own converged
        columns -- proven-convergent states, not disposable proposals -- and a
        marginal survivor (a slow phase-1 converger) can
        legitimately need more than warm_count_max accepted steps to re-certify.
        Capping them mislabels healthy particles as blown forwards (5 of 96
        survivors gated at 1500 -> a spurious 'RT/AD problem' RuntimeError)."""
        return chem.converged_y(chem_theta, warm_y=y_warm,
                                lnZ_ref=lnZ_ref, c_o_ref=c_o_ref,
                                return_conv_diag=True, warm_cap=False)

    def chem_solve_warm_diag_batch(C, Y, lnZ_ref, c_o_ref):
        """``chem_solve_warm_diag`` for a STACK: every particle's own carried
        column ``Y`` (N, nz, ni) re-converged at its own proposal ``C``
        (N, n_chem_tp) with its own reference composition ``lnZ_ref`` /
        ``c_o_ref`` ((N,) arrays, one per carried column) ->
        ``(y (N, nz, ni), ConvDiag with a leading particle axis)``.

        Runs under the mutation cap (``warm_cap=True``), which rides the runner
        carry per lane, and through ``_chem_batch_route``: the batched runner,
        or the lane queue when ``cfg.cold_lanes > 0``. THE warm solve on the
        SMC mutation path, primal and gradient (pipeline._make_batch_eval jvps
        through this for the whole cloud at once)."""
        return _chem_batch_route(C, warm_y=Y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref,
                                 warm_cap=True)

    def chem_solve_warm_diag_full_batch(C, Y, lnZ_ref, c_o_ref):
        """``chem_solve_warm_diag_batch`` WITHOUT the mutation cap (the cold
        count_max): the INIT phase-2 pass, whose inputs are phase-1 survivors
        re-certifying from their own converged columns."""
        return _chem_batch_route(C, warm_y=Y, lnZ_ref=lnZ_ref, c_o_ref=c_o_ref,
                                 warm_cap=False)

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
        chemistry) without re-running or re-differentiating the VULCAN while_loop --
        the block-structured likelihood gradient in pipeline.py relies on this split.

        ``ok`` (float 0/1, stop_gradient'ed) is the cold certificate: the runner's
        canonical convergence bit AND accept_count under count_max -- the same
        predicate the staged batch evaluators apply (pipeline._proposal_converged
        plus the cap). The scalar likelihood and the block gradient reject on it,
        so no likelihood entry point accepts a finite spectrum from an uncertified
        solve. Reading the diag is free: every field rides the primal carry.

        ``cloud`` is None (off) or a (2,) array [log10 kappac0, alphac] for the
        ExoJax powerlaw_clouds term (see exojax_rt /
        vulcan_forward.constants.CLOUD_NUC0)."""
        chem_theta = jnp.asarray(chem_theta)
        y, cd = chem_solve_cold_diag(chem_theta)                   # (nz, ni) absolute
        ok = ((jnp.asarray(cd.conv_normal) > 0.5)
              & (cd.accept_count < int(chem.count_max))).astype(y.dtype)
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
        chem_stage1_batch=chem_stage1_batch,
        chem_stage2_diag_batch=chem_stage2_diag_batch,
        chem_solve_warm_diag=chem_solve_warm_diag,
        chem_solve_warm_diag_full=chem_solve_warm_diag_full,
        chem_solve_warm_diag_batch=chem_solve_warm_diag_batch,
        chem_solve_warm_diag_full_batch=chem_solve_warm_diag_full_batch,
        aux_from_y=aux_from_y,
        y_baseline=np.asarray(chem.y0, dtype=np.float64),
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        n_tp=int(n_tp),
        tp_model=tpm,
        chem=chem, rt=rt,
        p_art_bar=np.asarray(rt.p_art_bar, dtype=np.float64),
    )
