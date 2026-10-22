"""Pipeline: the theta-space forward as a bounded-prior u-space posterior, and
its batched likelihood evaluators.

Each evaluator runs the chemistry of a whole batch in one call on the lane
queue, the cold two-stage map from the baseline or a warm continuation from a
given column, then the RT, ``lax.map``-chunked. Offsets
and noise inflation (one b, or one per instrument group: make_noise_model) are
analytic. run_nautilus calls them through ``pipe._make_batch_eval``.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, NamedTuple

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
# retrieval_forward (VULCAN-JAX env setup, jax x64) is imported lazily inside
# build_pipeline, so the u-space machinery unit-tests without the chemistry
# stack.

import jax
import jax.numpy as jnp

logger = logging.getLogger("retrieval")


Pipeline = SimpleNamespace   # the attribute bag build_pipeline fills

# A rejected or invalid draw's log-likelihood is this FINITE sentinel on device
# (a -inf would poison the arithmetic on it; run_nautilus maps it to -inf).
# Anything at or below REJECT_BELOW is a rejection, never a likelihood.
REJECT_LOGL = -1.0e30
REJECT_BELOW = -1.0e29

# sample_prior_u draws z = sigmoid(u) inside [eps, 1 - eps], so u stays finite.
_PRIOR_Z_EPS = 1e-6
# T-P window rejection sampling: the floor on candidates per round, and the candidate cap
# (max(factor * n, min)) above which it raises instead of looping on.
_TP_DRAW_MIN = 16
_TP_DRAW_CAP_FACTOR = 64
_TP_DRAW_CAP_MIN = 4096


def save_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class EvalStats(NamedTuple):
    """Per-row solver state of a batch evaluator's exit (device arrays).

    acc      (N,) int32 accept_count at exit
    conv_ok  (N,) bool  canonical-certification bit at exit
    """

    acc: jnp.ndarray
    conv_ok: jnp.ndarray


def make_uspace(specs, dtype):
    """Prior <-> unconstrained u-space (the SWAMPE transform).

    z = sigmoid(u) in (0,1); "uniform" -> lo + (hi-lo) z, "log10_uniform" ->
    10**(log lo + (log hi - log lo) z), "gaussian" -> lo + hi * Phi^-1(z) (lo
    the mean, hi the sigma), so u = logit(z) of a uniform z draws theta exactly
    from its prior.

    Returns (theta_from_u, sample_prior_u). Module-level so it can be
    unit-tested without the VULCAN/ExoJax stack.
    """
    from jax.scipy.special import ndtri

    n_dim = len(specs)
    prior_lo = jnp.asarray([s.lo for s in specs], dtype=dtype)
    prior_hi = jnp.asarray([s.hi for s in specs], dtype=dtype)
    is_log10 = jnp.asarray([1.0 if s.prior_type == "log10_uniform" else 0.0 for s in specs],
                           dtype=dtype)
    is_gauss = jnp.asarray([1.0 if s.prior_type == "gaussian" else 0.0 for s in specs],
                           dtype=dtype)
    lo_lin, span_lin = prior_lo, prior_hi - prior_lo
    lo_log = jnp.log10(jnp.clip(prior_lo, C.UNDERFLOW_DENOM, None))
    span_log = jnp.log10(jnp.clip(prior_hi, C.UNDERFLOW_DENOM, None)) - lo_log

    def theta_from_u(u):
        u = jnp.asarray(u, dtype=dtype)
        z = jax.nn.sigmoid(u)
        theta_lin = lo_lin + span_lin * z
        theta_log = 10.0 ** (lo_log + span_log * z)
        theta_gau = prior_lo + prior_hi * ndtri(z)
        return jnp.where(is_gauss > 0.5, theta_gau,
                         jnp.where(is_log10 > 0.5, theta_log, theta_lin))

    def sample_prior_u(rng_key, n_particles):
        eps = jnp.asarray(_PRIOR_Z_EPS, dtype=dtype)
        z = jax.random.uniform(rng_key, (n_particles, n_dim), dtype=dtype,
                               minval=eps, maxval=1.0 - eps)
        return jnp.log(z) - jnp.log1p(-z)

    return theta_from_u, sample_prior_u


def make_noise_model(names, group_per_bin, sigma0, dtype):
    """The likelihood's per-bin sigma for the run's noise mode, through the one
    map the predictive band also uses (``observations.noise_map``):

      none                         sigma = sigma0
      scalar  noise_inflation      sigma = sigma0 * b                 (every bin)
      group   noise_inflation_<g>  sigma_i = sigma0_i * b_{g(i)}      (one per group,
                                   contiguous in theta)

    ``sigma0`` is a zero-argument callable returning the observed sigma (read at
    trace time: set_observations fills it after the build). Returns
    ``sigma_for(theta) -> (n_bin,)``. Module-level so it is unit-tested without
    the forward stack (tests/test_group_noise.py)."""
    m = OBS.noise_map(names, group_per_bin)
    if m is None:
        return lambda theta: sigma0()
    lo, N = m
    if N is None:
        return lambda theta: sigma0() * theta[lo]
    n = N.shape[1]
    N = jnp.asarray(N, dtype)
    return lambda theta: sigma0() * (N @ theta[lo:lo + n])


def _map_chunks(fn, args, chunk):
    """Apply a BATCH function ``fn`` -- one that takes the whole stacked pytree
    ``args``, here ``jax.vmap`` of a per-particle RT stage -- to padded chunks of
    ``chunk`` particles under ``lax.map``, to bound peak memory. ``chunk<=0``
    (or >= n) is ONE call on the whole batch. Padding rows are dropped, so a
    vmapped per-particle ``fn`` gives the plain vmap's result at any chunk."""
    leaves = jax.tree_util.tree_leaves(args)
    n = int(leaves[0].shape[0])
    if chunk <= 0 or chunk >= n:
        return fn(args)
    n_pad = (-n) % chunk
    if n_pad:
        args = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, x[:n_pad]], axis=0), args)
    args = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, chunk) + x.shape[1:]), args)
    out = jax.lax.map(fn, args)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:])[:n], out)


def _map_chunks_rows(fn, args, chunk, need, fill):
    """``_map_chunks`` on the rows where ``need`` (an (n,) bool) holds; ``fill``
    elsewhere. The rows are reordered needed-first (row order kept within
    each group) into the stacked ``(n_chunks, chunk)`` layout ``_map_chunks``
    builds, and a while loop runs ``fn`` on the first ``ceil(sum(need) /
    chunk)`` chunks only, so a row nobody reads costs no RT. Its body is
    ``lax.map``'s (one chunk sliced out of the stack, the result written back
    into the stacked output), so each chunk is the same computation on the
    same shapes. ``fn`` must return one (chunk,) leaf. ``chunk<=0`` (or >= n)
    is ONE call on the whole batch, as in ``_map_chunks``, masked afterwards."""
    leaves = jax.tree_util.tree_leaves(args)
    n = int(leaves[0].shape[0])
    if chunk <= 0 or chunk >= n:
        return jnp.where(need, fn(args), fill)
    n_pad = (-n) % chunk
    # needed rows first, each group in row order (a stable sort of ~need)
    order = jnp.argsort(jnp.logical_not(need).astype(jnp.int32), stable=True).astype(jnp.int32)
    if n_pad:
        order = jnp.concatenate([order, order[:n_pad]])
    n_chunks = (n + n_pad) // chunk
    xs = jax.tree_util.tree_map(
        lambda x: x[order].reshape((n_chunks, chunk) + x.shape[1:]), args)
    n_need = jnp.sum(need.astype(jnp.int32))
    n_run = (n_need + chunk - 1) // chunk
    fill = jnp.asarray(fill)

    def body(c):
        k, ys = c
        x = jax.tree_util.tree_map(
            lambda a: jax.lax.dynamic_index_in_dim(a, k, 0, keepdims=False), xs)
        return k + 1, jax.lax.dynamic_update_index_in_dim(ys, fn(x), k, 0)

    ys0 = jnp.full((n_chunks, chunk), fill, dtype=fill.dtype)
    _, ys = jax.lax.while_loop(lambda c: c[0] < n_run, body, (jnp.int32(0), ys0))
    # stacked position p holds row order[p]; the first n_need are the needed rows
    pos = jnp.arange(n + n_pad, dtype=jnp.int32)
    dst = jnp.where(pos < n_need, order, jnp.int32(n))
    return jnp.full((n,), fill, dtype=fill.dtype).at[dst].set(ys.reshape(-1), mode="drop")


def build_pipeline(cfg: C.Config) -> Pipeline:
    """Build the forward, observation operators, u-space prior/likelihood and the
    batched evaluators. No inference, no run output written.

    Trace-time baking: the likelihood closes over ``pipe.obs_depth_jax`` /
    ``pipe.obs_sigma_jax``, which the first jitted call bakes in. Call
    ``pipe.set_observations`` once, before any inference call; a second call raises.
    """
    C.validate_config(cfg)
    from retrieval_framework import retrieval_forward as RF   # lazy: pulls in vulcan_chem -> VULCAN-JAX + exojax
    dtype = jnp.float64 if bool(jax.config.jax_enable_x64) else jnp.float32
    npdtype = np.float64

    # ---- forward model (VULCAN chemistry + ExoJax Guillot T-P + RT) ----
    t0 = time.perf_counter()
    fwd = RF.build_retrieval_forward(cfg)
    logger.info(f"Built forward in {time.perf_counter()-t0:.1f}s | native n_nu={fwd.wl_um.size} "
                f"nz={cfg.nz} n_tp={fwd.n_tp} molecules={list(cfg.molecules)}")

    # ---- T-P validity window (NO clipping) ----------------------------------
    # The Guillot profile is drawn raw (tp_profile does not clip). A draw whose T-P
    # leaves the modelable window [tp_model.T_min, tp_model.T_max] anywhere on the
    # ART pressure grid OR the chemistry grid (which reaches below the RT bottom:
    # 7.6 against 7 bar on W39b) is REJECTED, not bent into range:
    # rejection-sampled away at the prior and given the REJECT_LOGL likelihood by
    # every evaluator. A NaN profile (a T1bar draw with no Guillot solution) is
    # rejected the same way. The chem+T-P block of theta is [lnZ, c_o, lnKzz,
    # <chem extras>, <n_tp T-P params>], so the T-P sub-vector is
    # theta[n_chem:n_chem+n_tp].
    n_tp = int(fwd.n_tp)
    n_chem = int(fwd.n_chem)
    p_check_j = jnp.asarray(np.concatenate([np.asarray(fwd.p_art_bar, np.float64),
                                            np.asarray(fwd.p_chem_bar, np.float64)]),
                            dtype=dtype)
    tp_eval = fwd.tp_model.eval
    tp_T_min = jnp.asarray(fwd.tp_model.T_min, dtype)
    tp_T_max = jnp.asarray(fwd.tp_model.T_max, dtype)

    def tp_valid(theta):
        """True iff the drawn T-P lies entirely inside [T_min, T_max] on the ART
        grid and the chemistry grid."""
        T_all = tp_eval(jnp.asarray(theta)[n_chem:n_chem + n_tp], p_check_j)
        return jnp.all(jnp.isfinite(T_all) & (T_all >= tp_T_min) & (T_all <= tp_T_max))

    # ---- observations + linear operators (binning, offsets) ----
    obs, real_bins = OBS.get_observation_grid(cfg, fwd.wl_um)   # bins inside the model band
    B = OBS.build_binning_matrix(fwd.wl_um, obs)
    O_np = OBS.build_offset_design(obs)
    groups = list(obs["groups"])
    n_bin = int(B.shape[0])
    logger.info(f"Observations: {'REAL product bins' if real_bins else 'synthetic grid'} | "
                f"{n_bin} bins | groups={groups} | offset cols={O_np.shape[1]}")
    if n_bin < 2:
        raise RuntimeError(f"only {n_bin} usable observed bins in the model band; widen the band")

    B_jax = jnp.asarray(B, dtype=dtype)
    O_jax = jnp.asarray(O_np, dtype=dtype)

    # ---- parameter layout ----
    specs = C.specs_from_config(cfg, groups=groups)
    names = [s.name for s in specs]
    kinds = [s.kind for s in specs]
    labels = [s.label for s in specs]
    n_dim = len(specs)
    # theta[0:n_chem] = [lnZ, c_o, lnKzz, <extras>], then the T-P entries
    # (specs_from_config and the engine's layout agree: retrieval_forward)
    n_chem_tp = n_chem + fwd.n_tp
    # lnR0 and the cloud parameters are always present (specs_from_config);
    # lnM and s_o only when inferred
    lnR0_idx = names.index("lnR0")
    lnM_idx = names.index("lnM") if "lnM" in names else None
    s_o_idx = names.index("s_o") if "s_o" in names else None
    cloud_idx = [i for i, k in enumerate(kinds) if k == "cloud"]
    off_idx = [i for i, k in enumerate(kinds) if k == "offset"]
    off_lo = off_idx[0] if off_idx else None
    n_off = len(off_idx)
    cloud_lo = cloud_idx[0]
    n_cloud = len(cloud_idx)

    prior_types = [s.prior_type for s in specs]
    param_truth = np.asarray([s.truth for s in specs], dtype=npdtype)

    # ---- u-space reparameterization (bounded prior <-> unconstrained u) ----
    theta_from_u, sample_prior_u = make_uspace(specs, dtype)
    theta_truth = jnp.asarray(param_truth, dtype=dtype)

    # ---- prior draws restricted to the T-P window (no clip -> redraw) ----------
    # Draw from the box prior and REDRAW any particle whose Guillot T-P leaves the
    # modelable window. Cheap (Guillot only, no chemistry). The effective prior is
    # (box) INTERSECT (T-P in window).
    _tp_valid_batch = jax.jit(lambda U: jax.vmap(lambda u: tp_valid(theta_from_u(u)))(U))

    def sample_prior_u_valid(rng_key, n_particles):
        n_particles = int(n_particles)
        key = rng_key
        kept, have, drawn = [], 0, 0
        over = max(n_particles, _TP_DRAW_MIN)
        # loud cap: fail rather than loop forever
        max_draw = max(_TP_DRAW_CAP_FACTOR * n_particles, _TP_DRAW_CAP_MIN)
        while have < n_particles:
            key, sub = jax.random.split(key)
            cand = sample_prior_u(sub, over)
            good = np.asarray(jax.device_get(_tp_valid_batch(cand)))
            drawn += over
            if good.any():
                kept.append(np.asarray(jax.device_get(cand))[good])
                have += int(good.sum())
            if have < n_particles and drawn >= max_draw:
                raise RuntimeError(
                    f"prior T-P rejection: only {have}/{n_particles} valid draws after "
                    f"{drawn} candidates (accept frac {have / max(drawn, 1):.1%}). The "
                    "T-P prior puts most of its mass outside the modelable window "
                    f"[{float(tp_T_min):.0f}, {float(tp_T_max):.0f}] K -- tighten "
                    "prior_Tirr / prior_log10gamma in the run spec so realistic W39b profiles "
                    "dominate instead of being redrawn away.")
        U = np.concatenate(kept, axis=0)[:n_particles]
        n_cand = drawn
        if n_cand > 2 * n_particles:
            logger.info(f"prior T-P rejection: kept {n_particles} valid draws from "
                        f"{n_cand} candidates (accept frac {n_particles / n_cand:.1%}); "
                        "a low fraction means the T-P prior reaches unmodelable corners.")
        return jnp.asarray(U, dtype)

    # ---- forward -> observed (binned + offset) depth ----
    pipe = Pipeline()

    def _cloud_from(theta):
        return theta[cloud_lo:cloud_lo + n_cloud]

    def _lnM_from(theta):
        return theta[lnM_idx] if lnM_idx is not None else jnp.asarray(0.0, dtype)

    def _comp_refs(C_):
        """(N, 3) composition references (lnZ, c_o, s_o) of a chemistry block;
        s_o is 0 when not inferred."""
        s = C_[:, s_o_idx] if s_o_idx is not None else jnp.zeros(C_.shape[0], dtype)
        return jnp.stack([C_[:, 0], C_[:, 1], s], axis=1)

    def _mu_from_depth(depth, theta):
        """Binned model depth: B @ native depth + O @ offsets."""
        mu = B_jax @ depth
        if n_off > 0:
            mu = mu + O_jax @ (theta[off_lo:off_lo + n_off] * OBS.PPM)
        return mu

    def _binned_ok(theta):
        """(binned model depth, ok): ok is the cold certificate bit from
        fwd.native_depth_aux (VULCAN's canonical convergence, conv_normal).
        generate_observations refuses a truth spectrum without it."""
        theta = jnp.asarray(theta, dtype=dtype)
        native, _aux, ok = fwd.native_depth_aux(theta[:n_chem_tp], theta[lnR0_idx],
                                                _cloud_from(theta), _lnM_from(theta))
        return _mu_from_depth(native, theta), ok > 0.5

    binned_ok_jit = jax.jit(_binned_ok)

    # ---- likelihood in u-space (Gaussian, per-bin sigma, finite-guarded) ----
    # sigma_eff for the run's noise mode (scalar b, one b per group, or none)
    _sigma_for = make_noise_model(names, obs["group"], lambda: pipe.obs_sigma_jax, dtype)

    def _gauss_loglik(mu, theta):
        """The (finite-branch) Gaussian log-likelihood formula of the batched
        evaluators."""
        sig = _sigma_for(theta)
        r = (pipe.obs_depth_jax - mu) / sig
        n = mu.size
        return (-0.5 * jnp.sum(r * r) - jnp.sum(jnp.log(sig))
                - 0.5 * n * jnp.log(jnp.asarray(2.0 * math.pi, dtype=dtype)))

    _REJECT = jnp.asarray(REJECT_LOGL, dtype=dtype)

    # Batched evaluators (run_nautilus): the chemistry of the whole batch in one
    # call, then the per-row RT, lax.map-chunked.
    rt_chunk = int(cfg.smc_rt_chunk)
    y_baseline = jnp.asarray(fwd.y_baseline, dtype=dtype)          # (nz, ni)

    def _rt_val(args):
        """Per-particle RT stage, primal only: aux profiles -> loglik value."""
        aux, theta = args
        depth = fwd.rt_depth(aux, theta[lnR0_idx], _cloud_from(theta), _lnM_from(theta))
        mu = _mu_from_depth(depth, theta)
        val = _gauss_loglik(mu, theta)
        finite = jnp.all(jnp.isfinite(depth))
        return jnp.where(finite, val, _REJECT)

    def _mu_from_column(args):
        """Per-particle binned model depth from a converged column, with no
        chemistry solve. UNGATED: a plotting interface (nautilus_report's model
        spectra)."""
        y, theta = args
        aux = fwd.aux_from_y(y, theta[:n_chem_tp])
        return _mu_from_depth(
            fwd.rt_depth(aux, theta[lnR0_idx], _cloud_from(theta), _lnM_from(theta)), theta)

    observed_depth_from_y_jit = jax.jit(
        lambda Y, Theta: _map_chunks(jax.vmap(_mu_from_column), (Y, Theta), rt_chunk))

    def _make_batch_eval(mode: str):
        """Build eval(U, Y, refs, active) -> (L, Y_new, refs_new, EvalStats), all
        (N,)-batched.

        mode="warm": each particle's chemistry re-converges by continuation from its
        carried column Y with incremental (lnZ - refs[0], c_o - refs[1], s_o -
        refs[2]) scaling, under the cold count_max.
        mode="cold": the published solve-from-baseline (two-stage) map; Y/refs are
        still updated from the converged result so cold evals can seed warm ones.

        The EvalStats (final-stage accept_count, VULCAN's certificate) let the
        caller tell a count_max-exhausted from a stall-certified rejection.

        ``active`` ((N,) bool): a False row is padding. On the lane queue its
        chemistry takes no step; it is never ``usable`` (L is the reject
        sentinel, Y the baseline) and is never RT'd. The RT runs only on the
        rows whose L can survive: T-P in window, certified and active
        (``_map_chunks_rows``); the others get the reject sentinel without an
        RT."""
        warm = (mode == "warm")

        def _chem_primal(C_, Y, refs, active):
            """Chemistry for the whole particle batch -> (AUX, Y_new, ConvDiag):
            one batched call (the cold two-stage map, or the warm continuation
            under the cold count_max). Lanes freeze at their own exits;
            agreement with the per-lane map is at the convergence scale.
            ``active`` masks padded rows. The ConvDiag rides the primal carry."""
            if warm:
                Y_new, CD = fwd.chem_solve_warm_diag_full_batch(
                    C_, Y, refs[:, 0], refs[:, 1], refs[:, 2], active=active)
            else:
                Y_new, CD = fwd.chem_solve_cold_diag_batch(C_, active=active)
            return jax.vmap(fwd.aux_from_y)(Y_new, C_), Y_new, CD

        def _rt_vals(AUX, Theta, CD, valid, active):
            """The RT log L of the rows that can be kept: in-window, certified,
            active."""
            need = valid & (CD.conv_normal > 0.5) & active
            return _map_chunks_rows(jax.vmap(_rt_val), (AUX, Theta), rt_chunk,
                                    need, _REJECT)

        def eval_batch(U, Y, refs, active):
            active = jnp.asarray(active, bool)
            U = jnp.asarray(U, dtype)
            Theta = jax.vmap(theta_from_u)(U)                        # (N, n_dim)
            C_ = Theta[:, :n_chem_tp]
            # per-particle T-P window mask (no clip): an out-of-window draw is
            # rejected (L = REJECT_LOGL, state pinned to baseline)
            valid = jax.vmap(tp_valid)(Theta)
            AUX, Ynew, CD = _chem_primal(C_, Y, refs, active)
            vals = _rt_vals(AUX, Theta, CD, valid, active)
            ACC = CD.accept_count.astype(jnp.int32)
            conv_ok = CD.conv_normal > 0.5
            # the solver's verdict alone: the runner stops on accept_count >
            # count_max with convergence first, so a certified solve can exit
            # at cap or cap + 1, and conv_normal is False on every cap exit; a
            # padded row is never usable
            usable = valid & active & conv_ok
            L = jnp.where(jnp.isfinite(vals) & usable, vals, _REJECT)
            # a rejected solve -- non-finite forward, an out-of-window T-P, OR an
            # uncertified one -- must not poison the carried state
            # arithmetic: pin it to the baseline column.
            ok = jnp.all(jnp.isfinite(Ynew), axis=(1, 2)) & usable
            Ynew = jnp.where(ok[:, None, None], Ynew, y_baseline[None])
            comp = _comp_refs(C_)
            refs_new = jnp.where(ok[:, None], comp, jnp.zeros_like(comp))
            return L, Ynew, refs_new, EvalStats(acc=ACC, conv_ok=conv_ok)

        return eval_batch

    # ---- assemble ----
    pipe.__dict__.update(dict(
        cfg=cfg, dtype=dtype, npdtype=npdtype,
        fwd=fwd, obs=obs, real_bins=real_bins, groups=groups,
        B=B, n_bin=n_bin,
        names=names, kinds=kinds, labels=labels, n_dim=n_dim,
        n_chem=n_chem, n_chem_tp=n_chem_tp, lnR0_idx=lnR0_idx, lnM_idx=lnM_idx,
        cloud_idx=cloud_idx, n_cloud=n_cloud,
        param_prior_lo=np.asarray([s.lo for s in specs], npdtype),
        param_prior_hi=np.asarray([s.hi for s in specs], npdtype),
        param_truth=param_truth, prior_types=prior_types,
        # sample_prior_u is the T-P-window-restricted (redraw) sampler; the validity
        # predicate is exposed for diagnostics/calibration.
        theta_from_u=theta_from_u, sample_prior_u=sample_prior_u_valid,
        tp_valid=tp_valid,
        theta_truth=theta_truth,
        observed_depth_from_y_jit=observed_depth_from_y_jit,
        # (binned model depth, certificate bit): the synthetic truth spectrum
        binned_ok_jit=binned_ok_jit,
        y_baseline=y_baseline,
        # the batched evaluators run_nautilus calls (cold, warm)
        _make_batch_eval=_make_batch_eval,
        # observations injected by set_observations
        obs_depth_jax=None, obs_sigma_jax=None, obs_depth=None, obs_sigma=None,
    ))

    def set_observations(depth, sigma):
        # Observations are trace-time constants; a second call would silently
        # keep the old data, so it raises.
        if pipe.obs_depth is not None:
            raise RuntimeError(
                "set_observations was already called on this pipeline. Observations "
                "are trace-time constants baked into the compiled likelihood, so "
                "swapping them in-process would silently keep the old data. Build a "
                "fresh pipeline (build_pipeline) for a different dataset.")
        depth, sigma = validate_observations(depth, sigma, n_bin, npdtype)
        pipe.obs_depth = depth
        pipe.obs_sigma = sigma
        pipe.obs_depth_jax = jnp.asarray(depth, dtype=dtype)
        pipe.obs_sigma_jax = jnp.asarray(sigma, dtype=dtype)
        # The target density is fully defined only now -- observations are its last
        # input. Fingerprint it once here so every checkpoint carries the identity
        # its numbers belong to (run_nautilus refuses a resume across a change).
        from retrieval_framework import provenance
        pipe.target_digest = provenance.target_digest(cfg, pipe)

    pipe.set_observations = set_observations
    return pipe


def validate_observations(depth, sigma, n_bin: int, npdtype):
    """Coerce + VALIDATE an injected (depth, sigma) pair for n_bin spectral bins.

    Fail loud at the API boundary: the Gaussian
    likelihood divides by sigma and logs it, so a non-finite depth or a
    non-positive/non-finite sigma would silently poison every likelihood with
    NaN/Inf (mass rejection) rather than erroring here. Mask
    invalid bins BEFORE injection. Returns the flattened (depth, sigma) arrays."""
    depth = np.asarray(depth, npdtype).reshape(-1)
    sigma = np.asarray(sigma, npdtype).reshape(-1)
    if depth.shape[0] != n_bin or sigma.shape[0] != n_bin:
        raise ValueError(f"obs depth/sigma length must be n_bin={n_bin}")
    if not np.all(np.isfinite(depth)):
        raise ValueError("observed depths must all be finite; got "
                         f"{int((~np.isfinite(depth)).sum())} non-finite "
                         "bin(s) -- drop/mask them before set_observations")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("observed sigmas must all be finite and strictly "
                         f"positive; got {int((~np.isfinite(sigma)).sum())} "
                         f"non-finite and {int((sigma <= 0.0).sum())} "
                         "non-positive bin(s) -- the Gaussian likelihood "
                         "divides by sigma and logs it")
    return depth, sigma


def load_real_into_pipe(pipe: Pipeline) -> Dict[str, np.ndarray]:
    """Inject the real observed depths + sigmas already attached to pipe.obs."""
    obs = pipe.obs
    depth = np.asarray(obs["depth"], pipe.npdtype)
    sigma = np.asarray(obs["sigma"], pipe.npdtype)
    pipe.set_observations(depth, sigma)
    return dict(depth=depth, sigma=sigma)


def generate_observations(pipe: Pipeline, seed: int) -> Dict[str, np.ndarray]:
    """Synthetic injection: model at truth, add Gaussian noise at the (real, if available)
    per-bin sigma. Injects into pipe and returns the arrays. The truth spectrum
    must carry the cold certificate every likelihood row needs: an uncertified
    truth column would inject a spectrum no certified draw can reproduce."""
    sigma = np.asarray(pipe.obs["sigma"], pipe.npdtype)
    mu, ok = pipe.binned_ok_jit(pipe.theta_truth)
    mu_true = np.asarray(mu, pipe.npdtype)
    if not bool(ok):
        raise RuntimeError(
            "the truth chemistry does not certify (VULCAN's certificate under "
            f"count_max={int(pipe.fwd.chem.count_max)}): refusing the synthetic "
            "injection; move truth_* or the solver settings")
    if not np.all(np.isfinite(mu_true)):
        raise RuntimeError("truth forward is non-finite; check truth_* and priors")
    rng = np.random.default_rng(seed)
    depth = mu_true + rng.standard_normal(mu_true.shape) * sigma
    pipe.set_observations(depth, sigma)
    return dict(depth=depth, sigma=sigma, flux_true=mu_true)


def _blank_state(pipe: Pipeline, N: int):
    """(Y0, refs0) placeholders for a fresh batch: the baked baseline column
    (matching refs (lnZ, c_o, s_o) = (0, 0, 0))."""
    dtype = pipe.dtype
    Y0 = jnp.broadcast_to(pipe.y_baseline[None],
                          (N,) + tuple(pipe.y_baseline.shape)).astype(dtype)
    return Y0, jnp.zeros((N, 3), dtype)
