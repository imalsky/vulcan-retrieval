"""The differentiable retrieval forward: (chemistry+T-P params, lnR0) -> native transit
spectrum, composing the *live* VULCAN-JAX chemistry with the ExoJax RT.

    native_depth(chem_theta, lnR0) = transmission_depth_r(
        bridge( VULCAN.converged_ymix(chem_theta) ),           # VMR(nz, ni) -> ART grid
        T_art = Guillot(chem_theta[3:]),                        # same T-P on the ART grid
        lnR0 )                                                  # reference-radius nuisance

``chem_theta = [lnZ, dln(C/O), lnKzz, <T-P params>]`` is exactly what the (tp_eval-hooked)
``vulcan_chem.converged_ymix`` consumes; the T-P sub-vector ``chem_theta[3:3+n_tp]`` is
evaluated by the SAME ExoJax profile on both the VULCAN pressure grid (inside the
chemistry) and the ART grid (here, for the RT), so one self-consistent T(P) drives both.

Everything is pure JAX and supports forward-mode ``jvp`` end-to-end (the VULCAN runner's
``lax.while_loop`` supports jvp but not vjp -- forward-mode is the only route, which is
also why the retrieval's MALA gradient is built from forward-mode jvps).

Import order matters: ``vulcan_chem`` (env + jax x64) is imported before ``exojax_rt`` /
``tp_profile`` (which import ExoJax).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

# import order is load-bearing: vulcan_chem (env + jax x64) before anything exojax
from retrieval_framework.forward import config        # constants (MOLECULES, ...)
from retrieval_framework.forward import vulcan_chem   # sets env + jax x64; MUST precede exojax imports
import jax.numpy as jnp

from retrieval_framework import tp_profile   # ExoJax Guillot / power-law T-P
from retrieval_framework.forward import exojax_rt     # ExoJax ArtTransPure model
from retrieval_framework.forward import interp_map    # differentiable log-P bridge


def _refuse_condense_inference(chem, cfg) -> None:
    """Refuse gradient-MALA inference through a condensing steady state.

    The early ``cfg_overrides`` gate in ``config_schema.validate_config`` catches
    the common case, but a base VULCAN config can default ``use_condense=True``
    (e.g. ``Earth.yaml``) without the flag ever appearing in ``cfg_overrides``.
    ``chem.conden_spec`` is the RESOLVED truth (``build_chem_model`` builds it iff
    condensation is actually active), so gating on it closes that bypass. The
    pinned condensation state is not reliably differentiable (0.91 rel jvp-vs-FD
    on pinned species) and gradient-MALA is the only mutation kernel, so an
    inference run would sample against unreliable gradients. See
    ``../docs/condensation_differentiation.md``.
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
            "from the base config (cfg_overrides need not restate it). See "
            "docs/condensation_differentiation.md."
        )


def build_retrieval_forward(cfg: Any) -> SimpleNamespace:
    """Build the theta-space forward model for a Config.

    Returns SimpleNamespace with:
        native_depth(chem_theta, lnR0) -> (n_nu,) transit depth on the native nu grid
        wl_um     : (n_nu,) native wavelengths (um)
        n_tp      : number of T-P parameters
        tp_model  : the tp_profile object (eval/unpack)
        chem, rt, to_art, mol_cols, h2_col, species_masses, p_bar_vulcan
    """
    profile = cfg.profile()

    # T-P first (grid-agnostic evaluator), then hook it into the chemistry so the VULCAN
    # column re-converges under the retrieved T-P (not the old scalar T_int shift).
    tpm = tp_profile.build_tp_model(cfg)
    n_tp = tpm.n_params

    chem = vulcan_chem.build_chem_model(profile, tp_eval=tpm.eval, n_tp_params=n_tp)

    # Condensation is a FORWARD-model capability only; refuse inference on the
    # RESOLVED config (closes the base-config bypass of the early cfg_overrides
    # gate). See docs/condensation_differentiation.md.
    _refuse_condense_inference(chem, cfg)

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

    mol_cols = {key: chem.sidx[config.MOLECULES[key]["vulcan"]] for key in rt.molecules}
    h2_col = chem.sidx[config.BULK_H2_VULCAN]
    he_col = chem.sidx["He"]          # H2-He CIA partner (He is inert in the network)
    species_masses = chem.species_masses
    p_art_bar_j = jnp.asarray(rt.p_art_bar)

    two_stage = bool(cfg.two_stage_z)

    def chem_solve_cold(chem_theta):
        """Converged ABSOLUTE column y (nz, ni) from the baked baseline init. Default
        two-stage: (1) converge at the retrieved T-P/Kzz with BASELINE composition (the
        violent T-relaxation, which measurably erases init-inventory perturbations);
        (2) apply the lnZ / C-O scaling to that converged column and re-converge warm
        (gentle -> inventory survives; also the validated warm-started-jvp pattern).
        See config_schema.two_stage_z."""
        if not two_stage:
            return chem.converged_y(chem_theta)
        th_relax = chem_theta.at[0].set(0.0).at[1].set(0.0)        # baseline lnZ, c_o
        y_relaxed = chem.converged_y(th_relax)                     # stage 1 (nz, ni) abs
        return chem.converged_y(chem_theta, warm_y=y_relaxed,
                                lnZ_ref=0.0, c_o_ref=0.0)          # stage 2 (warm)

    def chem_solve_cold_diag(chem_theta):
        """Primal-only twin of chem_solve_cold returning ``(y, ConvDiag)``.

        ``ConvDiag.accept_count`` is the WORSE of the two stages' (max over stage 1
        T-relax / stage 2 warm-reconverge, or just the one stage's when
        two_stage_z=False) so a count_max-exhausted stage is detectable; the
        convergence fields (longdy / longdydt / count_since_new_min / conv_normal)
        describe STAGE 2 -- the state ``y`` actually is. accept_count alone is NOT
        a convergence test (stall fallback / hybrid phase-flip exits sit well under
        the cap); gate on ``conv_normal`` too. Not on any AD path -- used only by
        the SMC init's likelihood-only phase."""
        if not two_stage:
            return chem.converged_y(chem_theta, return_conv_diag=True)
        th_relax = chem_theta.at[0].set(0.0).at[1].set(0.0)
        y_relaxed, d1 = chem.converged_y(th_relax, return_conv_diag=True)
        y, d2 = chem.converged_y(chem_theta, warm_y=y_relaxed,
                                 lnZ_ref=0.0, c_o_ref=0.0, return_conv_diag=True)
        return y, d2._replace(
            accept_count=jnp.maximum(d1.accept_count, d2.accept_count))

    def chem_solve_warm(chem_theta, y_warm, lnZ_ref, c_o_ref):
        """Converged ABSOLUTE column y (nz, ni) by warm continuation from a
        previously-converged column ``y_warm`` whose inventory corresponds to
        (lnZ_ref, c_o_ref). The lnZ / C-O scalings are applied INCREMENTALLY
        (theta[0]-lnZ_ref, theta[1]-c_o_ref) -- the validated continuation pattern
        (same map the SO2 Hessian campaign marched with). Used by the SMC mutation:
        MCMC proposals move theta a little, so re-converging from the particle's own
        column costs ~count_min steps instead of a full cold two-stage solve.

        Runs under the warm_count_max cap (warm_cap=True): a proposal still not
        converged at warm_count_max accepted steps is cut off there and rejected by the
        pipeline gate -- it must not drag the whole lockstep batch to the cold cap."""
        return chem.converged_y(chem_theta, warm_y=y_warm,
                                lnZ_ref=lnZ_ref, c_o_ref=c_o_ref, warm_cap=True)

    def chem_solve_warm_diag(chem_theta, y_warm, lnZ_ref, c_o_ref):
        """chem_solve_warm + the warm solve's ``ConvDiag``, so the SMC mutation can
        detect a warm proposal that is NOT actually at a certified steady state --
        warm_count_max-exhausted OR stall-fallback/budget-terminated (conv_normal
        False) -- and reject it before trusting its jvp; the warm-side analogue of
        chem_solve_cold_diag. This closes the residual where a non-steady warm
        proposal was fed straight into the tangent/RT-vjp lanes (garbage or
        non-finite gradient -> the NAS job 65200 n_bad_grad raise: 16/864 proposals
        certified by accept_count alone while their tangents had not settled).

        THE warm solve on the SMC gradient path: pipeline._make_batch_eval jvp's
        straight through this (every ConvDiag field rides the runner's primal carry
        for free -- running a second primal-only while_loop just to read it doubled
        the chemistry wall time per sweep). The pipeline stop_gradients + casts the
        diag inside the jvp chain (longdy/longdydt are floats and DO carry a
        tangent; accept_count does not)."""
        return chem.converged_y(chem_theta, warm_y=y_warm,
                                lnZ_ref=lnZ_ref, c_o_ref=c_o_ref,
                                return_conv_diag=True, warm_cap=True)

    def chem_solve_warm_diag_full(chem_theta, y_warm, lnZ_ref, c_o_ref):
        """chem_solve_warm_diag WITHOUT the mutation cap: runs under the cold
        count_max. For the INIT gradient pass only (pipeline._init_state phase 2):
        its inputs are phase-1 SURVIVORS re-certifying from their own converged
        columns -- proven-convergent states, not disposable proposals -- and a
        marginal survivor (slow phase-1 converger, stall-fallback certification) can
        legitimately need more than warm_count_max accepted steps to re-certify.
        Capping them mislabels healthy particles as blown forwards (NAS job 64854:
        5/96 survivors gated at 1500 -> spurious 'RT/AD problem' RuntimeError)."""
        return chem.converged_y(chem_theta, warm_y=y_warm,
                                lnZ_ref=lnZ_ref, c_o_ref=c_o_ref,
                                return_conv_diag=True, warm_cap=False)

    def aux_from_y(y, chem_theta):
        """ART-grid primal profiles aux = (vmr dict, vmr_h2, vmr_he, T_art, mmw_art)
        from an absolute column y (nz, ni). Differentiable and cheap (normalize +
        T-P eval + log-P interpolation); the RT consumes exactly this tuple."""
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        T_art = tpm.eval(chem_theta[3:3 + n_tp], p_art_bar_j)      # (nlayer,)
        mmw_v = ymix @ species_masses                              # (nz,)
        mmw_art = to_art(mmw_v)
        vmr = {key: to_art(ymix[:, col]) for key, col in mol_cols.items()}
        vmr_h2 = to_art(ymix[:, h2_col])
        vmr_he = to_art(ymix[:, he_col])
        return (vmr, vmr_h2, vmr_he, T_art, mmw_art)

    def native_depth_aux(chem_theta, lnR0, cloud=None):
        """Full chain -> (native depth, aux) where aux = (vmr, vmr_h2, vmr_he, T_art,
        mmw_art) are the ART-grid primal profiles. The aux lets a caller take an
        RT-ONLY jvp (e.g. d/dlnR0 or the cloud parameters, which do not touch the
        chemistry) without re-running or re-differentiating the VULCAN while_loop --
        the block-structured likelihood gradient in pipeline.py relies on this split.

        ``cloud`` is None (off) or a (2,) array [log10 kappac0, alphac] for the
        ExoJax powerlaw_clouds term (see exojax_rt / config.CLOUD_NUC0)."""
        chem_theta = jnp.asarray(chem_theta)
        y = chem_solve_cold(chem_theta)                            # (nz, ni) absolute
        aux = aux_from_y(y, chem_theta)                            # ART-grid profiles
        vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
        depth = rt.transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0),
                                        vmr_he=vmr_he, cloud=cloud)
        return depth, aux

    def rt_depth(aux, lnR0, cloud=None):
        """RT-only depth at frozen chemistry/T-P profiles (for the cheap lnR0/cloud jvps)."""
        vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
        return rt.transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, jnp.asarray(lnR0),
                                       vmr_he=vmr_he, cloud=cloud)

    def native_depth(chem_theta, lnR0, cloud=None):
        """(chem+T-P vector, lnR0 scalar, optional cloud) -> native transit depth."""
        return native_depth_aux(chem_theta, lnR0, cloud)[0]

    return SimpleNamespace(
        native_depth=native_depth,
        native_depth_aux=native_depth_aux,
        rt_depth=rt_depth,
        chem_solve_cold=chem_solve_cold,
        chem_solve_cold_diag=chem_solve_cold_diag,
        chem_solve_warm=chem_solve_warm,
        chem_solve_warm_diag=chem_solve_warm_diag,
        chem_solve_warm_diag_full=chem_solve_warm_diag_full,
        aux_from_y=aux_from_y,
        y_baseline=np.asarray(chem.y0, dtype=np.float64),
        nz=int(chem.nz), ni=int(chem.ni),
        wl_um=np.asarray(rt.wl_um, dtype=np.float64),
        nu_grid=np.asarray(rt.nu_grid, dtype=np.float64),
        n_tp=int(n_tp),
        tp_model=tpm,
        chem=chem, rt=rt, to_art=to_art,
        mol_cols=mol_cols, h2_col=h2_col, species_masses=species_masses,
        p_bar_vulcan=np.asarray(chem.p_bar, dtype=np.float64),
        p_art_bar=np.asarray(rt.p_art_bar, dtype=np.float64),
    )
