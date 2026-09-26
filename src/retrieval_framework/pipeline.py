"""Pipeline: the theta-space forward as a bounded-prior u-space posterior, and a
self-contained adaptive-tempered SMC with a preconditioned forward-mode-jvp MALA
mutation kernel.

The SMC core is plain JAX (no BlackJAX dependency). The algorithm is the standard
Del Moral (2006) resample-move SMC; each stage (1) picks the next inverse
temperature by ESS bisection, (2) reweights and accumulates the log-evidence
increment, (3) resamples systematically, (4) runs `num_mcmc_steps` preconditioned
MALA sweeps at log_prior_u(u) + beta * loglik(u), (5) Robbins-Monro adapts the step
and refreshes the full-covariance preconditioner, (6) checkpoints atomically.

The VULCAN-JAX runner's `lax.while_loop` supports jvp but not vjp, so the chemistry
gradient is forward mode. The SMC hot path uses staged batched evaluators split at
the chemistry/RT boundary: chemistry jvp lanes for the n_chem_tp dims with every
particle in one batched solve; one reverse-mode RT vjp per particle, `lax.map`-
chunked (the RT vjp holds GiB per lane); offsets and noise inflation
analytic. The per-particle gradient functions are kept for validation.

`smc_chem_mode="cold"` (the default) re-solves every proposal with the two-stage
map, so the target does not depend on sampler history (up to the lane refill
tick); `"warm"` continues each proposal from the particle's carried column
(fewer steps, history-dependent target).
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
from retrieval_framework.certificate import BETA_TOL
# retrieval_forward (VULCAN-JAX env setup, jax x64) is imported lazily inside
# build_pipeline, so the SMC core and u-space machinery unit-test without the
# chemistry stack.

import jax
import jax.numpy as jnp
import jax.scipy.linalg

logger = logging.getLogger("retrieval")


Pipeline = SimpleNamespace   # the attribute bag build_pipeline fills

# A rejected or invalid proposal's log-likelihood is this FINITE sentinel (a
# -inf would poison the evidence and resampling arithmetic). Anything at or
# below REJECT_BELOW is a rejection, never a likelihood.
REJECT_LOGL = -1.0e30
REJECT_BELOW = -1.0e29

# Column of conv_normal in the packed per-particle ConvDiag vector (_pack_cd).
_CD_CONV_NORMAL = 4
# sample_prior_u draws z = sigmoid(u) inside [eps, 1 - eps], so u stays finite.
_PRIOR_Z_EPS = 1e-6
# T-P window rejection sampling: candidates per round, and the candidate cap
# (max(factor * n, min)) above which it raises instead of looping on.
_TP_DRAW_MIN = 16
_TP_DRAW_CAP_FACTOR = 64
_TP_DRAW_CAP_MIN = 4096
# ESS bisection for the next temperature: relative bracket tolerance, max steps.
_DBETA_TOL = 1e-4
_DBETA_BISECT_ITERS = 60
# MALA preconditioner: shrinkage toward the diagonal, the variance floor before
# the correlation matrix is formed, and the Cholesky jitter (relative to the
# largest width squared).
_COV_SHRINK = 0.1
_VAR_FLOOR = 1e-30
_CHOL_JITTER = 1e-12
# Decimals at which two u-space particles count as the same state.
_UNIQ_DECIMALS = 9
# The ladder stops once beta is within this of 1; reached_beta1 uses BETA_TOL.
_BETA_DONE_TOL = 1e-8


def save_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class EvalStats(NamedTuple):
    """Health/diagnostic tail of the batched gradient evaluators (device arrays).

    Uniform across the mutation-capped and init (uncapped) variants, so every
    consumer reads the same structure. Scalars are per-batch tallies; vectors are
    per-particle (N,) and feed the bad-gradient forensics dump.

    n_capped     () int32   valid proposals cut off at the cap (MH-rejected)
    n_stalled    () int32   valid, under-cap proposals whose exit was NOT the
                            runner's canonical certification (e.g. a budget
                            exit) -- MH rejections, not AD pathologies
    acc          (N,) int32 accept_count at exit
    longdy       (N,) f64   runner's convergence metric at exit
    conv_ok      (N,) bool  canonical-certification bit at exit
    bad_grad     (N,) bool  finite-likelihood/non-finite-gradient AD pathology
                            (already masked to usable proposals)
    chem_tan_bad (N,) bool  non-finite chemistry tangent (the jvp DAUX side);
                            bad_grad & ~chem_tan_bad localizes the pathology to
                            the RT vjp instead
    """

    n_capped: jnp.ndarray
    # Uncertified under-cap exits; checkpoints and outputs carry the name.
    n_stalled: jnp.ndarray
    acc: jnp.ndarray
    longdy: jnp.ndarray
    conv_ok: jnp.ndarray
    bad_grad: jnp.ndarray
    chem_tan_bad: jnp.ndarray


def _zero_eval_stats(n: int, dtype) -> EvalStats:
    """All-healthy EvalStats for stub pipelines / cold gradient maps."""
    return EvalStats(
        n_capped=jnp.zeros((), jnp.int32),
        n_stalled=jnp.zeros((), jnp.int32),
        acc=jnp.zeros((n,), jnp.int32),
        longdy=jnp.zeros((n,), dtype),
        conv_ok=jnp.ones((n,), bool),
        bad_grad=jnp.zeros((n,), bool),
        chem_tan_bad=jnp.zeros((n,), bool),
    )


def _proposal_converged(cd_vec):
    """Convergence predicate for a warm MALA proposal's solve, from the packed
    per-particle ConvDiag vector ``[accept_count, longdy, longdydt,
    count_since_new_min, conv_normal]`` (see vulcan_forward.vulcan_chem.ConvDiag).
    The gate that decides whether a proposal's state, and so its jvp tangents, is
    trusted: the runner's canonical two-branch certification recomputed at the
    exit state (``conv_normal``). A stall or budget exit reads False even with
    longdy under yconv_min.
    """
    return cd_vec[:, _CD_CONV_NORMAL] > 0.5


def make_uspace(specs, dtype):
    """Bounded-box prior <-> unconstrained u-space (the SWAMPE transform).

    z = sigmoid(u) in (0,1); "uniform" -> lo + (hi-lo) z, "log10_uniform" ->
    10**(log lo + (log hi - log lo) z). log_prior_u carries the sigmoid Jacobian so
    the induced prior on theta is exactly (log-)uniform on the box.

    Returns (theta_from_u, log_prior_u, sample_prior_u). Module-level so the SMC core
    can be unit-tested without the VULCAN/ExoJax stack.
    """
    n_dim = len(specs)
    prior_lo = jnp.asarray([s.lo for s in specs], dtype=dtype)
    prior_hi = jnp.asarray([s.hi for s in specs], dtype=dtype)
    is_log10 = jnp.asarray([1.0 if s.prior_type == "log10_uniform" else 0.0 for s in specs],
                           dtype=dtype)
    lo_lin, span_lin = prior_lo, prior_hi - prior_lo
    lo_log = jnp.log10(jnp.clip(prior_lo, C.UNDERFLOW_DENOM, None))
    span_log = jnp.log10(jnp.clip(prior_hi, C.UNDERFLOW_DENOM, None)) - lo_log

    def theta_from_u(u):
        u = jnp.asarray(u, dtype=dtype)
        z = jax.nn.sigmoid(u)
        theta_lin = lo_lin + span_lin * z
        theta_log = 10.0 ** (lo_log + span_log * z)
        return jnp.where(is_log10 > 0.5, theta_log, theta_lin)

    def log_prior_u(u):
        u = jnp.asarray(u, dtype=dtype)
        return jnp.sum(jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u))

    def sample_prior_u(rng_key, n_particles):
        eps = jnp.asarray(_PRIOR_Z_EPS, dtype=dtype)
        z = jax.random.uniform(rng_key, (n_particles, n_dim), dtype=dtype,
                               minval=eps, maxval=1.0 - eps)
        return jnp.log(z) - jnp.log1p(-z)

    return theta_from_u, log_prior_u, sample_prior_u


def _tree_dot(a, b):
    """Sum of leaf-wise vdots of two pytrees with identical structure (the tangent /
    cotangent contraction of the split chain rule)."""
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    return sum(jnp.vdot(x, y) for x, y in zip(la, lb))


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


def build_pipeline(cfg: C.Config) -> Pipeline:
    """Build the forward, observation operators, u-space prior/likelihood, and the
    forward-mode-gradient likelihood wrapper. No inference, no file IO.

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
    # leaves the modelable window [tp_model.T_min, tp_model.T_max] on the ART pressure
    # grid (the widest P range; the chemistry grid is a subset) is REJECTED, not bent
    # into range: rejection-sampled away at the prior (init redraw) and given -inf
    # likelihood as a MALA proposal. The chem+T-P block of theta is [lnZ, c_o, lnKzz,
    # <n_tp T-P params>], so the T-P sub-vector is theta[3:3+n_tp].
    n_tp = int(fwd.n_tp)
    p_art_j = jnp.asarray(fwd.p_art_bar, dtype=dtype)
    tp_eval = fwd.tp_model.eval
    tp_T_min = jnp.asarray(fwd.tp_model.T_min, dtype)
    tp_T_max = jnp.asarray(fwd.tp_model.T_max, dtype)

    def tp_valid(theta):
        """True iff the drawn T-P lies entirely inside [T_min, T_max] on the ART grid."""
        T_art = tp_eval(jnp.asarray(theta)[3:3 + n_tp], p_art_j)
        return jnp.all(jnp.isfinite(T_art) & (T_art >= tp_T_min) & (T_art <= tp_T_max))

    # ---- observations + linear operators (binning, offsets) ----
    obs, real_bins = OBS.get_observation_grid(cfg, fwd.wl_um)
    keep, B = OBS.build_binning_matrix(fwd.wl_um, obs)
    # apply keep to the observed arrays so obs bins line up with B's rows
    for k in ("wl", "wl_lo", "wl_hi", "depth", "sigma", "group"):
        obs[k] = np.asarray(obs[k])[keep]
    obs["groups"] = list(dict.fromkeys(np.asarray(obs["group"]).tolist()))
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
    n_chem_tp = 3 + fwd.n_tp
    # The chem+T-P prefix is unpacked by fixed position (theta[0:3]=chem,
    # theta[3:3+n_tp]=T-P). Assert the layout EXACTLY, not just "chem/tp appear in
    # the prefix": dropping a chem toggle shortens the block, and a subset
    # check passes when all nuisances are also off, silently truncating the
    # vector. config_schema.validate_config refuses that config at the boundary;
    # this is the backstop for any path that builds specs without it.
    if (names[:3] != ["lnZ", "c_o", "lnKzz"]
            or kinds[:3] != ["chem", "chem", "chem"]
            or kinds[3:n_chem_tp] != ["tp"] * fwd.n_tp
            or n_dim < n_chem_tp):
        raise RuntimeError(
            "parameter layout error: the vector must start with [lnZ, c_o, lnKzz] "
            f"+ {fwd.n_tp} T-P dims; got names={names[:n_chem_tp]} "
            f"kinds={kinds[:n_chem_tp]} (n_dim={n_dim}). The chem block is "
            "positional -- do not drop infer_lnZ/c_o/lnKzz.")
    # lnR0 and both cloud parameters are always present (specs_from_config)
    lnR0_idx = names.index("lnR0")
    cloud_idx = [i for i, k in enumerate(kinds) if k == "cloud"]
    off_idx = [i for i, k in enumerate(kinds) if k == "offset"]
    noise_idx = names.index("noise_inflation") if "noise_inflation" in names else None
    off_lo = off_idx[0] if off_idx else None
    n_off = len(off_idx)
    cloud_lo = cloud_idx[0]
    n_cloud = len(cloud_idx)

    prior_types = [s.prior_type for s in specs]
    param_truth = np.asarray([s.truth for s in specs], dtype=npdtype)

    # ---- u-space reparameterization (bounded prior <-> unconstrained u) ----
    theta_from_u, log_prior_u, sample_prior_u = make_uspace(specs, dtype)
    theta_truth = jnp.asarray(param_truth, dtype=dtype)

    # ---- prior draws restricted to the T-P window (no clip -> redraw) ----------
    # Draw from the box prior and REDRAW any particle whose Guillot T-P leaves the
    # modelable window. Cheap (Guillot only, no chemistry). The effective prior is
    # (box) INTERSECT (T-P in window); MALA stays inside it because an out-of-window
    # proposal gets -inf likelihood (rejected) at every beta>0.
    _tp_valid_batch = jax.jit(lambda U: jax.vmap(lambda u: tp_valid(theta_from_u(u)))(U))

    # Running tally of the T-P window rejection sampling: n_kept/n_drawn estimates the
    # window's prior mass, one of the two factors in the operational-prior support
    # fraction reported next to the evidence (the other is the init convergence cull).
    tp_prior_stats = {"n_drawn": 0, "n_kept": 0}

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
                    "prior_Tirr / prior_log10gamma in case.py so realistic W39b profiles "
                    "dominate instead of being redrawn away.")
        U = np.concatenate(kept, axis=0)[:n_particles]
        tp_prior_stats["n_drawn"] += int(drawn)
        tp_prior_stats["n_kept"] += int(have)
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

    def _binned_ok(theta):
        """(binned model depth, ok): ok is the cold certificate bit from
        fwd.native_depth_aux (canonical convergence AND under count_max)."""
        theta = jnp.asarray(theta, dtype=dtype)
        chem_theta = theta[:n_chem_tp]
        native, _aux, ok = fwd.native_depth_aux(chem_theta, theta[lnR0_idx],
                                                _cloud_from(theta))
        binned = B_jax @ native                                # (n_bin,)
        if n_off > 0:
            offs = jax.lax.dynamic_slice_in_dim(theta, off_lo, n_off) * OBS.PPM
            binned = binned + O_jax @ offs
        return binned, ok > 0.5

    def observed_depth_model(theta):
        """Binned model depth, UNGATED: a plotting/diagnostic interface (truth
        spectrum, posterior predictive). Every likelihood entry point goes through
        _binned_ok and rejects an uncertified column."""
        return _binned_ok(theta)[0]

    observed_depth_model_jit = jax.jit(observed_depth_model)

    # ---- likelihood in u-space (Gaussian, per-bin sigma, finite-guarded) ----
    def _sigma_for(theta):
        sig = pipe.obs_sigma_jax
        if noise_idx is not None:
            sig = sig * theta[noise_idx]
        return sig

    def _gauss_loglik(mu, theta):
        """The (finite-branch) Gaussian log-likelihood formula -- single source of
        truth shared by log_likelihood_u and the block-structured gradient."""
        sig = _sigma_for(theta)
        r = (pipe.obs_depth_jax - mu) / sig
        n = mu.size
        return (-0.5 * jnp.sum(r * r) - jnp.sum(jnp.log(sig))
                - 0.5 * n * jnp.log(jnp.asarray(2.0 * math.pi, dtype=dtype)))

    _REJECT = jnp.asarray(REJECT_LOGL, dtype=dtype)

    def log_likelihood_u(u):
        theta = theta_from_u(u)

        def _bad():
            return _REJECT

        def _good():
            # only reached for an in-window T-P, so the RT never extrapolates
            mu, ok = _binned_ok(theta)
            return jax.lax.cond(ok & jnp.all(jnp.isfinite(mu)),
                                lambda: _gauss_loglik(mu, theta), _bad)

        # short-circuit an out-of-window T-P to -inf WITHOUT running the forward
        return jax.lax.cond(tp_valid(theta), _good, _bad)

    # ---- forward-mode value-and-grad (no reverse tape through the VULCAN loop) ----
    def _value_and_grad_naive(u):
        """n_dim forward-mode jvps of the scalar likelihood (the SWAMPE pattern).
        Every u-direction -- including the chemistry-free lnR0/offset/noise dims --
        pays a full tangent pass through the VULCAN while_loop."""
        u = jnp.asarray(u)
        eye = jnp.eye(n_dim, dtype=u.dtype)
        y0, dy0 = jax.jvp(log_likelihood_u, (u,), (eye[0],))
        dy_rest = jax.vmap(lambda v: jax.jvp(log_likelihood_u, (u,), (v,))[1])(eye[1:])
        return y0, jnp.concatenate([jnp.atleast_1d(dy0), dy_rest], axis=0)

    def _value_and_grad_block(u):
        """Block-structured exact gradient: only the n_chem_tp chemistry+T-P directions
        take tangents through the VULCAN while_loop -- all in ONE vmapped jvp (a single
        batched device call; the primal + ART-grid aux are read from lane 0, whose primal
        is identical across lanes). lnR0 + cloud dims are one cheap RT-only jacfwd at the
        frozen aux profiles; offsets and noise-inflation are analytic. Exact (the
        parameter blocks enter mu through disjoint sub-graphs); asserted equal to the
        naive gradient in the smoke test."""
        u = jnp.asarray(u)
        theta = theta_from_u(u)
        # diagonal d(theta)/d(u): theta_from_u is elementwise, so J @ 1 == diag(J)
        _, dtheta_du = jax.jvp(theta_from_u, (u,), (jnp.ones_like(u),))
        c = theta[:n_chem_tp]
        r0 = theta[lnR0_idx]
        cloudp = _cloud_from(theta)

        eye_c = jnp.eye(n_chem_tp, dtype=u.dtype)

        def _chain(cc):
            return fwd.native_depth_aux(cc, r0, cloudp)

        (d_all, aux_all, ok_all), (J_chem, _, _) = jax.vmap(
            lambda v: jax.jvp(_chain, (c,), (v,)))(eye_c)
        d0 = d_all[0]                                            # primal native depth
        aux = jax.tree_util.tree_map(lambda x: x[0], aux_all)    # primal ART-grid profiles
        # J_chem: (n_chem_tp, n_native) tangent stack

        # mu = B @ native + O @ offsets  (identical to observed_depth_model)
        mu = B_jax @ d0
        if n_off > 0:
            offs = theta[off_lo:off_lo + n_off] * OBS.PPM
            mu = mu + O_jax @ offs

        sig = _sigma_for(theta)
        resid = pipe.obs_depth_jax - mu
        wres = resid / (sig * sig)                                   # dL/dmu
        val = _gauss_loglik(mu, theta)

        Btw = B_jax.T @ wres                                         # (n_native,)
        g_theta = jnp.zeros((n_dim,), dtype=u.dtype)
        g_theta = g_theta.at[:n_chem_tp].set(J_chem @ Btw)

        # RT-only dims (lnR0 + cloud params): one jacfwd through the RT at frozen aux
        rt_idx = [lnR0_idx] + cloud_idx
        rv0 = jnp.stack([theta[i] for i in rt_idx])

        def _rt(rv):
            return fwd.rt_depth(aux, rv[0], rv[1:])

        J_rt = jax.jacfwd(_rt)(rv0)                              # (n_native, n_rt)
        for j, i in enumerate(rt_idx):
            g_theta = g_theta.at[i].set(jnp.dot(J_rt[:, j], Btw))
        if n_off > 0:
            g_theta = g_theta.at[off_lo:off_lo + n_off].set(OBS.PPM * (O_jax.T @ wres))
        if noise_idx is not None:
            k = theta[noise_idx]
            sig0 = pipe.obs_sigma_jax
            g_theta = g_theta.at[noise_idx].set(
                jnp.sum(resid * resid / (sig0 * sig0 * k ** 3)) - mu.size / k)

        grad_u = g_theta * dtheta_du
        # reject an out-of-window T-P (no clip), a blown forward, or an uncertified solve
        finite = jnp.all(jnp.isfinite(d0)) & tp_valid(theta) & (ok_all[0] > 0.5)
        val = jnp.where(finite, val, _REJECT)
        return val, jnp.where(finite, grad_u, jnp.zeros_like(grad_u))

    # Staged BATCHED likelihood / gradient (the SMC hot path; see module docstring).
    # Exact -- the same chain rule as _value_and_grad_block, regrouped:
    #   dL/dtheta_chem[k] = < d aux / d theta[k]  (fwd jvp through the chemistry),
    #                         d L / d aux          (rev vjp through the RT) >.
    chem_mode = str(cfg.smc_chem_mode).strip().lower()
    if chem_mode not in ("warm", "cold"):
        raise ValueError(f"smc_chem_mode must be 'warm' or 'cold', got {chem_mode!r}")
    rt_chunk = int(cfg.smc_rt_chunk or 0)
    rt_vjp_chunk = int(cfg.smc_rt_vjp_chunk or 0)
    y_baseline = jnp.asarray(fwd.y_baseline, dtype=dtype)          # (nz, ni)
    eye_c = jnp.eye(n_chem_tp, dtype=dtype)

    def _mu_from_depth(depth, theta):
        mu = B_jax @ depth
        if n_off > 0:
            mu = mu + O_jax @ (theta[off_lo:off_lo + n_off] * OBS.PPM)
        return mu

    def _rt_val(args):
        """Per-particle RT stage, primal only: aux profiles -> loglik value."""
        aux, theta = args
        depth = fwd.rt_depth(aux, theta[lnR0_idx], _cloud_from(theta))
        mu = _mu_from_depth(depth, theta)
        val = _gauss_loglik(mu, theta)
        finite = jnp.all(jnp.isfinite(depth))
        return jnp.where(finite, val, _REJECT)

    def _mu_from_column(args):
        """Per-particle binned model depth from a converged column: the RT half of
        observed_depth_model, with no chemistry solve. UNGATED like it; the
        posterior predictive runs it on the final particles' carried columns."""
        y, theta = args
        aux = fwd.aux_from_y(y, theta[:n_chem_tp])
        return _mu_from_depth(
            fwd.rt_depth(aux, theta[lnR0_idx], _cloud_from(theta)), theta)

    observed_depth_from_y_jit = jax.jit(
        lambda Y, Theta: _map_chunks(jax.vmap(_mu_from_column), (Y, Theta), rt_chunk))

    def _rt_val_grad(args):
        """Per-particle RT stage WITH gradient: primal depth + ONE reverse-mode vjp.
        ``daux`` is the (n_chem_tp,)-stacked aux tangent pytree from the chemistry
        jvp lanes; contracting it against the RT cotangent gives the chem+T-P block,
        and the same vjp call yields the lnR0/cloud entries for free."""
        aux, daux, theta = args
        depth, vjp_fn = jax.vjp(fwd.rt_depth, aux, theta[lnR0_idx],
                                _cloud_from(theta))
        mu = _mu_from_depth(depth, theta)
        sig = _sigma_for(theta)
        resid = pipe.obs_depth_jax - mu
        wres = resid / (sig * sig)                                   # dL/dmu
        val = _gauss_loglik(mu, theta)
        Btw = B_jax.T @ wres                                         # (n_native,)
        aux_bar, r_bar, cloud_bar = vjp_fn(Btw)
        g = jnp.zeros((n_dim,), dtype)
        g = g.at[:n_chem_tp].set(jax.vmap(lambda d: _tree_dot(d, aux_bar))(daux))
        g = g.at[lnR0_idx].set(r_bar)
        g = g.at[cloud_lo:cloud_lo + n_cloud].set(cloud_bar)
        if n_off > 0:
            g = g.at[off_lo:off_lo + n_off].set(OBS.PPM * (O_jax.T @ wres))
        if noise_idx is not None:
            k = theta[noise_idx]
            sig0 = pipe.obs_sigma_jax
            g = g.at[noise_idx].set(
                jnp.sum(resid * resid / (sig0 * sig0 * k ** 3)) - mu.size / k)
        finite = jnp.all(jnp.isfinite(depth))
        # A non-finite DEPTH is a rejected proposal (-1e30 sentinel -> -inf MH accept;
        # its gradient is then irrelevant and zeroed only to keep arithmetic clean).
        # A finite depth with a non-finite gradient is an AD pathology: flag it
        # for the host driver's zero-drift handling and backstop.
        bad_grad = finite & ~jnp.all(jnp.isfinite(g))
        val = jnp.where(finite, val, _REJECT)
        g = jnp.where(finite & jnp.isfinite(g), g, jnp.zeros_like(g))
        return val, g, bad_grad

    def _make_batch_eval(mode: str, want_grad: bool, diag: bool = False,
                         mutation_cap: bool = True):
        """Build eval(U, Y, refs) -> (L, G, Y_new, refs_new, n_bad_grad, stats)
        when want_grad (``stats`` an EvalStats), else (L, Y_new, refs_new, stats)
        -- or (L, Y_new, refs_new, per-particle ConvDiag) when diag; all
        (N,)-batched. ``n_bad_grad`` counts finite-likelihood/non-finite-gradient
        AD pathologies (the host driver's backstop reads it).

        mode="warm": each particle's chemistry re-converges by continuation from its
        carried column Y with incremental (lnZ - refs[0], c_o - refs[1]) scaling.
        mode="cold": the published solve-from-baseline (two-stage) map; Y/refs are
        still updated from the converged result so cold evals can seed warm ones.

        ``diag`` (only meaningful for mode="cold", want_grad=False -- the SMC init's
        likelihood-only phase) additionally threads each particle's ConvDiag
        through (final-stage accept_count / longdy / conv_normal), so the
        caller can detect a count_max-exhausted OR stall-certified
        (not-actually-converged) cold solve instead of silently carrying it into L.

        The cold gradient runs the two stages as two explicit jvps, so stage 1
        carries only the lnKzz + T-P directions (the lnZ / c_o tangents are
        identically zero there); tests/test_stage_split.py pins it against the
        single-chain jvp of the same map."""
        warm = (mode == "warm")
        assert not (diag and (warm or want_grad)), "diag is cold+no-grad only"
        # Cap of the convergence gate. mutation_cap=True (MALA proposals): warm
        # solves are capped at warm_count_max and an unconverged proposal is
        # rejected there. mutation_cap=False (init phase 2): survivors re-certify
        # under the cold count_max (a warm_count_max cap here culls healthy
        # survivors). A cold solve is never warm-capped.
        wcmax = (int(fwd.chem.warm_count_max) if (warm and mutation_cap)
                 else int(fwd.chem.count_max))

        if want_grad:
            # A proposal can end finite but unsettled (a non-convergent corner or
            # an uncertified exit), with untrustworthy tangents. Its ConvDiag
            # rides the jvp'd primal carry and is packed into one stop-gradient
            # float vector (keeps the jvp output all-float). eval_batch rejects an
            # exhausted or uncertified proposal; the cold path reads the same diag
            # off its batched stage 2, so both chem modes share one gate.
            if warm:
                _solve_cd_batch = (fwd.chem_solve_warm_diag_batch if mutation_cap
                                   else fwd.chem_solve_warm_diag_full_batch)

            def _pack_cd(cd):
                # [accept_count, longdy, longdydt, count_since_new_min, conv_normal]
                return jax.lax.stop_gradient(jnp.stack([
                    jnp.asarray(cd.accept_count, dtype),
                    jnp.asarray(cd.longdy, dtype),
                    jnp.asarray(cd.longdydt, dtype),
                    jnp.asarray(cd.count_since_new_min, dtype),
                    jnp.asarray(cd.conv_normal, dtype)]))

            # One batched solve per chemistry direction (converged_y_batch, or the
            # lane queue with cfg.cold_lanes): the loop sits above the lane vmap,
            # so lane-dependent conds key to one shared tick. The loop predicate
            # reads the primal only, so a jvp stops where the primal certifies (a
            # tangent-certified stop is converged_y_jvp). Lanes are independent,
            # so one broadcast direction gives each particle its own derivative.
            def _pack_cd_batch(cd):
                return jax.vmap(_pack_cd)(cd)

            def _aux_batch(Y, C_):
                return jax.vmap(fwd.aux_from_y)(Y, C_)

            def _bcast(v, C_):
                return jnp.broadcast_to(v, C_.shape)

            if warm:
                # WARM continuation: each particle re-converges from its OWN
                # carried column at its OWN reference composition, so the
                # carried state and the (N,) reference arrays ride into the
                # batch as constants -- only theta carries a tangent. The
                # mutation cap is per lane on the runner carry (warm_cap).
                def _warm_chain(C_, Y, refs):
                    Y_new, cd = _solve_cd_batch(C_, Y, refs[:, 0], refs[:, 1])
                    return _aux_batch(Y_new, C_), Y_new, _pack_cd_batch(cd)

                def _chem_batch(C_, Y, refs):
                    (AUX_l, Y_l, CD_l), (DAUX_l, _dY, _dCD) = jax.vmap(
                        lambda v: jax.jvp(lambda C: _warm_chain(C, Y, refs),
                                          (C_,), (_bcast(v, C_),)))(eye_c)
                    AUX = jax.tree_util.tree_map(lambda x: x[0], AUX_l)
                    DAUX = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x, 0, 1), DAUX_l)
                    return AUX, DAUX, Y_l[0], CD_l[0]
            else:
                eye_h = eye_c[2:]                      # lnKzz + T-P directions

                def _stage2_chain(C_, Y1):
                    Y, cd = fwd.chem_stage2_diag_batch(C_, Y1)
                    return _aux_batch(Y, C_), Y, _pack_cd_batch(cd)

                def _pad_dy1(dy1):
                    # directions 0,1 (lnZ, c_o): no y_relaxed tangent
                    return jnp.zeros((n_chem_tp,) + dy1.shape[1:],
                                     dy1.dtype).at[2:].set(dy1)

                def _chem_batch(C_, Y, refs):
                    """Cold two-stage chemistry for a whole chunk, stages SPLIT.

                    Directions 0 and 1 carry a zero stage-1 tangent (their theta
                    tangent is killed by the .at[:, 0].set(0).at[:, 1].set(0)
                    inside stage 1); directions 2..n_chem_tp-1 carry the stage-1
                    tangent that flows inline into stage 2's warm_y in the
                    single-chain route. The split feeds the same (e_i, dy1) pair
                    explicitly, so the primal is the same program and the
                    tangents differ at most by XLA fusion. Y/refs are unused on
                    the cold path (as in the single-chain route).
                    """
                    Y1_l, dY1 = jax.vmap(lambda v: jax.jvp(
                        fwd.chem_stage1_batch, (C_,), (_bcast(v, C_),)))(eye_h)
                    Y1 = Y1_l[0]                       # direction-independent primal
                    (AUX_l, Y_l, CD_l), (DAUX_l, _dY, _dCD) = jax.vmap(
                        lambda v, dY: jax.jvp(_stage2_chain, (C_, Y1),
                                              (_bcast(v, C_), dY))
                    )(eye_c, _pad_dy1(dY1))
                    AUX = jax.tree_util.tree_map(lambda x: x[0], AUX_l)
                    # (D, N, ...) -> the (N, D, ...) per-particle layout the RT
                    # stage reads
                    DAUX = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(x, 0, 1), DAUX_l)
                    return AUX, DAUX, Y_l[0], CD_l[0]
        else:
            # Gated like the gradient path: this evaluator is the FD reference
            # (smoke_retrieval), the certificate's cold replay and validate_warm's
            # comparison arm, so it must be the same likelihood. The diag rides
            # the primal carry.
            def _chem_primal(C_, Y, refs):
                """Chemistry for the whole particle batch -> (AUX, Y_new, ConvDiag):
                one batched call (the cold two-stage map, or the warm continuation
                under the mutation cap unless ``mutation_cap=False``). Lanes
                freeze at their own exits; agreement with the per-lane map is at
                the convergence scale."""
                if warm:
                    # mutation_cap=False: the cold count_max, as on the gradient
                    # path (run_nautilus's anchored warm starts)
                    solve = (fwd.chem_solve_warm_diag_batch if mutation_cap
                             else fwd.chem_solve_warm_diag_full_batch)
                    Y_new, CD = solve(C_, Y, refs[:, 0], refs[:, 1])
                else:
                    Y_new, CD = fwd.chem_solve_cold_diag_batch(C_)
                return jax.vmap(fwd.aux_from_y)(Y_new, C_), Y_new, CD

        def eval_batch(U, Y, refs):
            U = jnp.asarray(U, dtype)
            Theta = jax.vmap(theta_from_u)(U)                        # (N, n_dim)
            _, dTh = jax.vmap(
                lambda u: jax.jvp(theta_from_u, (u,), (jnp.ones_like(u),)))(U)
            C_ = Theta[:, :n_chem_tp]
            # per-particle T-P window mask (no clip): an out-of-window proposal is
            # rejected (-inf L, state pinned to baseline) and is NOT flagged as an AD
            # pathology (its gradient is irrelevant once MH rejects it).
            valid = jax.vmap(tp_valid)(Theta)
            usable = valid   # narrowed to (valid & certified) on the warm gradient path
            if want_grad:
                # Chemistry jvp directions: ONE batched solve for the whole
                # cloud in either chem mode. The chemistry tangent lanes are
                # cheap (~20 MB per lane pair); the RT VJP below
                # is the memory wall.
                AUX, DAUX, Ynew, CD = _chem_batch(C_, Y, refs)
                vals, g_th, bads = _map_chunks(jax.vmap(_rt_val_grad),
                                               (AUX, DAUX, Theta), rt_vjp_chunk)
                G = g_th * dTh                                       # chain to u-space
                # Two REJECTION classes (MH rejections, not AD pathologies) whose
                # garbage gradients must NOT trip n_bad_grad, in both chem modes:
                #   capped  -- the solve hit the cap (wcmax above);
                #   stalled -- under the cap but not canonically certified (e.g.
                #              a budget exit): the primal may look settled while
                #              the jvp tangent, with no stopping rule of its own,
                #              has not.
                ACC = CD[:, 0].astype(jnp.int32)
                conv_ok = _proposal_converged(CD)
                under_cap = ACC < wcmax
                usable = valid & under_cap & conv_ok
                n_bad = jnp.sum((bads & usable).astype(jnp.int32))
                # both classes broken out from the generic reject count: the MH
                # correction only knows the Langevin proposal density, so a
                # rejection class that binds often (and possibly state-dependently)
                # is a detailed-balance risk -- it must be visible per sweep/stage,
                # not folded into "rejected".
                n_capped = jnp.sum((valid & ~under_cap).astype(jnp.int32))
                n_stalled = jnp.sum((valid & under_cap & ~conv_ok).astype(jnp.int32))
                # chem-vs-RT attribution for the forensics dump: a non-finite jvp
                # tangent (DAUX) localizes the pathology to the chemistry side;
                # bad_grad with finite DAUX points at the RT vjp.
                chem_bad = jnp.zeros((Theta.shape[0],), bool)
                for _leaf in jax.tree_util.tree_leaves(DAUX):
                    chem_bad = chem_bad | ~jnp.all(
                        jnp.isfinite(_leaf), axis=tuple(range(1, _leaf.ndim)))
                stats = EvalStats(
                    n_capped=n_capped, n_stalled=n_stalled,
                    acc=ACC, longdy=CD[:, 1], conv_ok=conv_ok,
                    bad_grad=bads & usable, chem_tan_bad=chem_bad)
            elif diag:
                AUX, Ynew, CDIAG = _chem_primal(C_, Y, refs)
                vals = _map_chunks(jax.vmap(_rt_val), (AUX, Theta), rt_chunk)
                G = None
            else:
                AUX, Ynew, CDL = _chem_primal(C_, Y, refs)
                vals = _map_chunks(jax.vmap(_rt_val), (AUX, Theta), rt_chunk)
                G = None
                ACC = CDL.accept_count.astype(jnp.int32)
                conv_ok = CDL.conv_normal > 0.5
                under_cap = ACC < wcmax
                usable = valid & under_cap & conv_ok
                # Same two rejection classes the gradient branch tallies above.
                # The gradient-free (rwm) mutation kernel runs THIS branch, and
                # the certificate's late-ladder convergence-cliff gate reads the
                # tallies -- discarding them here would make that gate pass
                # vacuously on a primal-only run.
                stats = _zero_eval_stats(Theta.shape[0], dtype)._replace(
                    n_capped=jnp.sum((valid & ~under_cap).astype(jnp.int32)),
                    n_stalled=jnp.sum((valid & under_cap & ~conv_ok).astype(jnp.int32)),
                    acc=ACC, longdy=CDL.longdy.astype(dtype), conv_ok=conv_ok)
            L = jnp.where(jnp.isfinite(vals) & usable, vals, _REJECT)
            # a blown (-1e30, rejected/culled) solve -- non-finite forward, an
            # out-of-window T-P, OR a non-converged warm proposal -- must not poison the
            # carried state arithmetic: pin it to the baseline column. This is part of the
            # MH rejection mechanics, not an error path -- true failures surface through
            # the -1e30 likelihood (init raises) or n_bad_grad (host raises).
            ok = jnp.all(jnp.isfinite(Ynew), axis=(1, 2)) & usable
            Ynew = jnp.where(ok[:, None, None], Ynew, y_baseline[None])
            refs_new = jnp.where(ok[:, None], C_[:, :2], jnp.zeros_like(C_[:, :2]))
            if want_grad:
                # uniform tail: EvalStats for BOTH the mutation-capped and the init
                # (uncapped) variants -- _init_state reads .acc/.conv_ok to tell a
                # re-certification failure (cull) from a true RT/AD blow-up (raise);
                # the mutation kernel reads .n_capped/.n_stalled and the per-particle
                # vectors for the bad-gradient forensics dump.
                return L, G, Ynew, refs_new, n_bad, stats
            if diag:
                return L, Ynew, refs_new, CDIAG
            return L, Ynew, refs_new, stats

        return eval_batch

    # ---- assemble ----
    pipe.__dict__.update(dict(
        cfg=cfg, dtype=dtype, npdtype=npdtype,
        fwd=fwd, obs=obs, real_bins=real_bins, groups=groups,
        B=B, n_bin=n_bin,
        names=names, kinds=kinds, labels=labels, n_dim=n_dim,
        n_chem_tp=n_chem_tp, lnR0_idx=lnR0_idx,
        cloud_idx=cloud_idx, n_cloud=n_cloud,
        param_prior_lo=np.asarray([s.lo for s in specs], npdtype),
        param_prior_hi=np.asarray([s.hi for s in specs], npdtype),
        param_truth=param_truth, prior_types=prior_types,
        # sample_prior_u is the T-P-window-restricted (redraw) sampler; the validity
        # predicate is exposed for diagnostics/calibration.
        theta_from_u=theta_from_u, log_prior_u=log_prior_u, sample_prior_u=sample_prior_u_valid,
        tp_valid=tp_valid,
        tp_prior_stats=tp_prior_stats,
        theta_truth=theta_truth,
        observed_depth_model_jit=observed_depth_model_jit,
        observed_depth_from_y_jit=observed_depth_from_y_jit,
        log_likelihood_u=log_likelihood_u,
        value_and_grad_naive=_value_and_grad_naive, value_and_grad_block=_value_and_grad_block,
        # staged batched evaluators (the SMC hot path)
        has_chem_state=True, chem_mode=chem_mode, y_baseline=y_baseline,
        # also called directly by run_nautilus (the uncapped warm map) and
        # smoke_retrieval
        _make_batch_eval=_make_batch_eval,
        # the RT half of the gradient evaluators; tests/test_stage_split.py
        # builds its single-chain reference on it
        _rt_val_grad=_rt_val_grad,
        batch_eval_cold_vg=_make_batch_eval("cold", True),
        batch_eval_cold_l=_make_batch_eval("cold", False),
        batch_eval_cold_l_diag=_make_batch_eval("cold", False, diag=True),
        batch_eval_move_vg=_make_batch_eval(chem_mode, True),
        # init phase 2: same evaluator WITHOUT the mutation cap (survivors re-certify
        # under the cold count_max; see _make_batch_eval's mutation_cap note)
        batch_eval_init_vg=_make_batch_eval(chem_mode, True, mutation_cap=False),
        batch_eval_move_l=_make_batch_eval(chem_mode, False),
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
        # its numbers belong to (run_smc_loop refuses a resume across a change).
        from retrieval_framework import certificate as _cert
        pipe.target_digest = _cert.target_digest(cfg, pipe)

    pipe.set_observations = set_observations
    return pipe


def logz_err_lower_bound(ess_hist, n_particles: int) -> float:
    """Optimistic Monte Carlo error on logZ: Var(logZ) ~ sum_t (1/ESS_t - 1/N).

    LOWER bound by construction (ignores resampling/MALA correlation between
    stages). Module-level and jax-free so the formula is unit-testable
    (tests/test_evidence_semantics.py), like evidence_report.
    """
    ess = np.asarray(ess_hist, np.float64)
    ess = ess[np.isfinite(ess) & (ess > 0.0)]
    return (float(np.sqrt(np.sum(1.0 / ess - 1.0 / n_particles)))
            if ess.size else float("nan"))


# Observations
def evidence_report(logZ: float, init_stats: dict) -> dict:
    """Evidence-semantics fields from the SMC ``logZ`` and the init cull
    counters. Module-level and jax-free so the semantics are unit-testable
    (tests/test_evidence_semantics.py), like validate_observations.

    ``logZ`` is the evidence under the OPERATIONAL prior: the declared box
    conditioned on the T-P window A and chemistry convergence C,
    renormalized, Z_oper = E_pi[L | A and C]. Returned fields:

      log_support_physical (+err)   ln f_tp: T-P-window prior mass, solver-INDEPENDENT
      log_conv_attrition (+err)     ln(f_c1 f_c2): convergence success,
                                    solver-DEPENDENT (count_max, tolerances)
      log_support_fraction (+err)   their sum
      logZ_box                      logZ + ln(f_tp f_c1 f_c2): the ZERO-FILLED box
                                    evidence, the integral of pi * L * 1[A and C]
                                    over the box and the only integral-valid box
                                    quantity. Cross-model Bayes factors ONLY at
                                    matched solver settings, with the attrition
                                    shown likelihood-negligible.

    With cold_lanes > 0 a refilled draw's C and L move with its lane tick at
    the convergence scale; "exact" holds up to that.

    No ``logZ_box_physical`` (logZ + ln f_tp) is returned: it is neither the box
    integral over A nor the A-conditioned evidence.

    The ``*_err`` fields are only the binomial error of the support fractions,
    not the seed-to-seed error of ``logZ``, so they are not the total evidence
    uncertainty. For a Bayes factor, add the empirical std of logZ over
    independent seeds in quadrature and require agreement across
    particle-count and tempering settings."""
    def _binom(k, n):
        if n <= 0:
            return 1.0, 0.0
        f = max(k / n, 1.0 / (2.0 * n))          # floor so ln(f) stays finite
        se = math.sqrt(max(f * (1.0 - f), 0.0) / n) / f   # d ln f
        return f, se
    f_tp, se_tp = _binom(init_stats.get("tp_n_kept", 0),
                         init_stats.get("tp_n_drawn", 0))
    f_c1, se_c1 = _binom(init_stats.get("n_alive_phase1", 0),
                         init_stats.get("n_drawn", 0))
    n_p2 = init_stats.get("n_phase2", 0)
    f_c2, se_c2 = _binom(n_p2 - init_stats.get("n_recert_fail", 0), n_p2)
    log_tp = math.log(f_tp)
    log_conv = math.log(f_c1) + math.log(f_c2)
    log_support = log_tp + log_conv
    return dict(
        log_support_fraction=log_support,
        log_support_fraction_err=math.sqrt(se_tp**2 + se_c1**2 + se_c2**2),
        log_support_physical=log_tp, log_support_physical_err=se_tp,
        log_conv_attrition=log_conv,
        log_conv_attrition_err=math.sqrt(se_c1**2 + se_c2**2),
        logZ_box=logZ + log_support,
        f_tp=f_tp, f_conv=math.exp(log_conv),
    )


def validate_observations(depth, sigma, n_bin: int, npdtype):
    """Coerce + VALIDATE an injected (depth, sigma) pair for n_bin spectral bins.

    Fail loud at the API boundary: the Gaussian
    likelihood divides by sigma and logs it, so a non-finite depth or a
    non-positive/non-finite sigma would silently poison every likelihood with
    NaN/Inf (mass rejection / pathological SMC) rather than erroring here. Mask
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
    per-bin sigma. Injects into pipe and returns the arrays."""
    sigma = np.asarray(pipe.obs["sigma"], pipe.npdtype)
    mu_true = np.asarray(pipe.observed_depth_model_jit(pipe.theta_truth), pipe.npdtype)
    if not np.all(np.isfinite(mu_true)):
        raise RuntimeError("truth forward is non-finite; check truth_* and priors")
    rng = np.random.default_rng(seed)
    depth = mu_true + rng.standard_normal(mu_true.shape) * sigma
    pipe.set_observations(depth, sigma)
    return dict(depth=depth, sigma=sigma, flux_true=mu_true)


# SMC core (self-contained, pure JAX)
def _ess_from_incremental(L: np.ndarray, dbeta: float) -> float:
    a = dbeta * (L - L.max())
    w = np.exp(a)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w * w))


def _next_dbeta(L: np.ndarray, beta: float, target_ess: float,
                tol: float = _DBETA_TOL) -> float:
    """Bisection for the temperature increment so ESS(exp(dbeta*L)) = target_ess.
    Returns dbeta in (0, 1-beta]; jumps to 1-beta when even the full step keeps ESS high."""
    dmax = 1.0 - beta
    if dmax <= 0:
        return 0.0
    if _ess_from_incremental(L, dmax) >= target_ess:
        return dmax
    lo, hi = 0.0, dmax
    for _ in range(_DBETA_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if _ess_from_incremental(L, mid) >= target_ess:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * dmax:
            break
    return 0.5 * (lo + hi)


def _systematic_resample_idx(key, weights, N):
    u0 = jax.random.uniform(key, dtype=weights.dtype)
    positions = (u0 + jnp.arange(N, dtype=weights.dtype)) / N
    return jnp.clip(jnp.searchsorted(jnp.cumsum(weights), positions), 0, N - 1)


def _proposal_scale(particles: np.ndarray, cap: float,
                    shrink: float = _COV_SHRINK) -> np.ndarray:
    """Lower-triangular Cholesky factor L of the (resampled, uniformly-weighted)
    cloud covariance, used as the MALA preconditioner (C = L L^T).

    Absolute, not normalized: the proposal narrows with the tempered posterior,
    so the step size only fine-tunes toward the target acceptance (a shape-only,
    unit-geometric-mean preconditioner collapses the late-stage acceptance).

    FULL covariance, not just the diagonal: this posterior's degeneracies are
    between parameters (metallicity against C/O against cloud opacity against
    lnR0 against the inter-instrument offset), and a diagonal preconditioner
    proposes across them instead of along them. The failure is silent -- it shows
    up as particle degeneracy, not as an error.

    ``shrink`` blends toward the diagonal (Ledoit-Wolf style, fixed intensity),
    which keeps L well-conditioned when the cloud is small relative to n_dim or a
    direction has collapsed; shrink=1 is the diagonal preconditioner. Falls back
    to the diagonal preconditioner if the factorization fails."""
    p = np.asarray(particles, np.float64)
    n_dim = p.shape[1]
    sd = np.clip(p.std(axis=0), C.SCALE_FLOOR, float(cap))
    if not np.all(np.isfinite(sd)):
        return np.eye(n_dim)
    if p.shape[0] <= n_dim + 1:      # too few particles for a covariance
        return np.diag(sd)
    cov = np.cov(p, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return np.diag(sd)
    cov = (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))
    # clip the WIDTHS the same way the diagonal preconditioner does, holding the
    # correlations fixed: cov -> D R D with D the clipped std
    d0 = np.sqrt(np.clip(np.diag(cov), _VAR_FLOOR, None))
    corr = cov / np.outer(d0, d0)
    cov = corr * np.outer(sd, sd)
    try:
        return np.linalg.cholesky(cov + _CHOL_JITTER * np.eye(n_dim) * sd.max() ** 2)
    except np.linalg.LinAlgError:
        return np.diag(sd)


def _get_batch_evals(pipe: Pipeline):
    """(cold_vg, cold_l, move_vg, move_l) batched evaluators. Gradient evaluators
    return the 6-tuple (L, G, Y_new, refs_new, n_bad, stats); ``stats`` is an
    EvalStats
    (uniform across the mutation-capped, init-uncapped, cold, and stub variants:
    per-batch n_capped/n_stalled tallies + per-particle acc/longdy/conv_ok/
    bad_grad/chem_tan_bad). Likelihood-only evaluators return
    (L, Y_new, refs_new, stats) with the same EvalStats tail minus the gradient
    fields, so the gradient-free mutation kernel reports the SAME rejection
    classes the certificate gates.
    Real pipelines carry the staged chemistry+RT evaluators; stub pipes (unit tests,
    no chemistry) get a stateless adapter so the SMC/MALA core is exercised through
    the exact same code path."""
    if getattr(pipe, "has_chem_state", False):
        return (pipe.batch_eval_cold_vg, pipe.batch_eval_cold_l,
                pipe.batch_eval_move_vg, pipe.batch_eval_move_l)
    if not hasattr(pipe, "_stub_evals"):
        vg1 = jax.value_and_grad(pipe.log_likelihood_u)

        def eval_vg(U, Y, refs):
            L, G = jax.vmap(vg1)(U)
            bad = jnp.isfinite(L) & ~jnp.all(jnp.isfinite(G), axis=1)
            # mirror the real evaluators' full AD-pathology contract: flag the
            # particle AND zero the non-finite gradient entries (the zeroed
            # drift is what the zero-drift MH handling uses on both sides)
            G = jnp.where(jnp.isfinite(G), G, jnp.zeros_like(G))
            # all-healthy EvalStats (stubs have no warm cap or stall class) except
            # bad_grad -- keeps the 6-tuple contract uniform
            stats = _zero_eval_stats(U.shape[0], U.dtype)._replace(bad_grad=bad)
            return (L, G, Y, refs, jnp.sum(bad.astype(jnp.int32)), stats)

        def eval_l(U, Y, refs):
            return (jax.vmap(pipe.log_likelihood_u)(U), Y, refs,
                    _zero_eval_stats(U.shape[0], U.dtype))

        pipe._stub_evals = (eval_vg, eval_l)
    evg, el = pipe._stub_evals
    return evg, el, evg, el


def _blank_state(pipe: Pipeline, N: int):
    """(Y0, refs0) placeholders for a fresh particle cloud: the baked baseline
    column for real pipelines (matching refs (lnZ, c_o) = (0, 0)), inert zeros for
    stubs."""
    dtype = pipe.dtype
    if getattr(pipe, "has_chem_state", False):
        Y0 = jnp.broadcast_to(pipe.y_baseline[None],
                              (N,) + tuple(pipe.y_baseline.shape)).astype(dtype)
    else:
        Y0 = jnp.zeros((N, 1, 1), dtype)
    return Y0, jnp.zeros((N, 2), dtype)


def _init_draw_count(pipe: Pipeline, n_target: int) -> int:
    """Oversampled cold-init draw count so reject-and-cull still leaves ``n_target``
    healthy particles. Stub pipes (no chemistry) never fail to converge, so they draw
    exactly n_target; real pipelines draw ceil(n_target * cfg.init_oversample)."""
    n_target = int(n_target)
    if not getattr(pipe, "has_chem_state", False):
        return n_target
    over = float(pipe.cfg.init_oversample)
    return max(n_target, int(math.ceil(n_target * over)))


def _init_state(pipe: Pipeline, U, target_n: int):
    """Initialize the SMC particle state -> (U_kept, L, G, Y, refs, init_stats)
    for ``target_n`` healthy particles. ``U`` is the oversampled prior cloud
    (_init_draw_count).

    Phase 1: cold likelihood-only pass over every draw. A draw that exhausts
    count_max, exits without canonical certification or has a non-finite
    forward is rejected. Raises if fewer than target_n survive or the reject
    fraction exceeds init_max_nonconverged_frac.

    Phase 2: gradient pass on the survivors (+ init_phase2_spare) through the
    map the MALA proposals use, uncapped (the cold count_max). Re-certification
    failures are culled and backfilled; a non-finite likelihood on a certified
    solve raises; a non-finite tangent keeps the particle with zeroed gradient
    entries, up to the smc_tangent_bad_max_frac backstop."""
    M = int(U.shape[0])
    target_n = int(target_n)
    if M < target_n:
        raise RuntimeError(f"_init_state got {M} draw(s) but target_n={target_n}: the "
                           "oversampled draw must be at least the target particle count")
    Y0, refs0 = _blank_state(pipe, M)
    _, cold_l, move_vg, _ = _get_batch_evals(pipe)
    # Real (chem-backed) pipelines get the diag-threading cold evaluator so phase 1 can
    # detect a count_max-exhausted (not-actually-converged) particle and REJECT it; stub
    # pipes (unit tests, no chemistry) keep plain cold_l -- there is no while_loop to
    # exhaust, so nothing is ever rejected there.
    has_diag = bool(getattr(pipe, "has_chem_state", False))
    cold_l_init = pipe.batch_eval_cold_l_diag if has_diag else cold_l

    if not hasattr(pipe, "_init_l_jit"):
        pipe._init_l_jit = jax.jit(cold_l_init)
        # phase 2 uses the UNCAPPED move evaluator where the pipeline provides one:
        # survivors re-certify under the cold count_max, not the mutation-proposal cap
        # (stub pipes have no cap distinction and keep move_vg)
        pipe._init_mv_jit = jax.jit(getattr(pipe, "batch_eval_init_vg", None) or move_vg)

    # ---- phase 1: cold likelihood over the full (oversampled) draw ----
    t0 = time.perf_counter()
    lanes = int(pipe.cfg.cold_lanes)
    width = (f"{min(lanes, M)} lanes refilled from the draw queue (wall time = "
             "total work / lanes)" if lanes > 0 else
             "one lockstep batch (wall time = the slowest draw)")
    logger.info(f"init 1/2: batched cold two-stage chemistry over {M} draw(s) on "
                f"{width}; likelihood only, reject non-converged, keep {target_n}, "
                "count_max-bounded")
    if has_diag:
        L0, Y, refs, cd0 = pipe._init_l_jit(U, Y0, refs0)
    else:
        L0, Y, refs, _ = pipe._init_l_jit(U, Y0, refs0)
    jax.block_until_ready(L0)

    # per-particle rejection (real pipes only): non-finite forward, count_max-
    # exhausted, OR stall-certified (the exit was not the runner's canonical
    # certification -- a state whose likelihood/tangents describe an unsettled
    # column; the class behind non-finite mutation gradients)
    L0_np = np.asarray(jax.device_get(L0), np.float64)
    nonfinite = ~np.isfinite(L0_np) | (L0_np <= REJECT_BELOW)
    if has_diag:
        count_max = int(pipe.fwd.chem.count_max)
        wa = np.asarray(jax.device_get(cd0.accept_count), np.int64)
        conv0 = np.asarray(jax.device_get(cd0.conv_normal), bool)
        exhausted = wa >= count_max
        stalled = ~conv0 & ~exhausted & ~nonfinite
    else:
        exhausted = np.zeros(M, bool)
        stalled = np.zeros(M, bool)
    dead = nonfinite | exhausted | stalled
    alive = np.flatnonzero(~dead)
    n_alive, n_dead = int(alive.size), int(dead.sum())

    frac, msg = 0.0, ""
    if n_dead:
        frac = n_dead / M
        n_ex, n_st = int(exhausted.sum()), int(stalled.sum())
        n_nf = int((nonfinite & ~exhausted).sum())
        idx_head = np.flatnonzero(dead)[:12].tolist()
        msg = (f"cold init: rejected {n_dead}/{M} draw(s) ({frac:.0%}: {n_ex} hit "
               f"count_max, {n_st} stall-certified (not a canonical steady state), "
               f"{n_nf} non-finite forward; first indices {idx_head}); "
               f"keeping {target_n} of {n_alive} survivors")
        logger.info(msg)

    # Systemic breakage first: it is the more actionable failure and its remedy
    # (oversample / prior / count_max) is different from the attrition gate's.
    if n_alive < target_n:
        raise RuntimeError(
            f"only {n_alive}/{M} cold draws converged; need {target_n}. The "
            "reject-and-cull ran out of survivors: raise init_oversample (currently "
            f"{float(pipe.cfg.init_oversample):g}), tighten the prior, "
            "or raise count_max. This is a systemic prior/config problem, not a few hard "
            "corners.")

    # Conditioning the posterior on "chemistry converged" removes a region of the
    # declared prior. Surviving computationally is not evidence that the removed
    # region is scientifically negligible, so the tolerance is a GATE, not a hint:
    # a case must declare the attrition it expects.
    if frac > float(pipe.cfg.init_max_nonconverged_frac):
        raise RuntimeError(
            msg + f" -- reject fraction {frac:.0%} exceeds the declared "
            f"init_max_nonconverged_frac "
            f"({float(pipe.cfg.init_max_nonconverged_frac):.0%}). The operational "
            "prior is the declared box conditioned on convergence; an undeclared "
            "attrition rate silently changes the target. Raise the knob in the case "
            "(and justify it) or tighten the prior / raise count_max.")

    # phase 2 evaluates a few SPARE survivors beyond target_n so marginal columns
    # that cannot RE-certify can be culled and backfilled instead of killing the
    # run
    spare = int(pipe.cfg.init_phase2_spare) if has_diag else 0
    n_phase2 = min(n_alive, target_n + spare)
    sel = jnp.asarray(alive[:n_phase2])
    U_keep = jnp.asarray(U)[sel]
    Y, refs = Y[sel], refs[sel]
    logger.info(f"init 1/2 done in {time.perf_counter() - t0:.1f}s "
                f"({n_alive} converged; phase 2 on {n_phase2} = {target_n}"
                f"+{n_phase2 - target_n} spare)")

    # ---- phase 2: gradient on the survivors (+spares) ----
    t0 = time.perf_counter()
    what = ("the production cold two-stage solve per draw, batched, with the "
            "jvp directions on top" if str(getattr(pipe, "chem_mode", "warm")) == "cold"
            else "jvp directions on a warm re-certification from each survivor's "
                 "own converged column")
    logger.info(f"init 2/2: move-map gradient at the kept cloud ({what}; "
                "UNCAPPED -- bounded by the cold count_max)")
    out = pipe._init_mv_jit(U_keep, Y, refs)
    jax.block_until_ready(out[0])
    L, G, Y, refs, n_bad, stats2 = out
    if has_diag:      # real pipelines: EvalStats threads per-particle ACC + conv bit
        acc2_np = np.asarray(jax.device_get(stats2.acc), np.int64)
        conv2_np = np.asarray(jax.device_get(stats2.conv_ok), bool)
    else:             # stub pipelines: the zeroed-EvalStats tail carries no gating info
        acc2_np = None
    n_bad = int(jax.device_get(n_bad))
    if n_bad > 0:
        # The tangent-blown class also occurs on the init phase-2 warm
        # re-certifications, and it is theta-dependent, so culling or raising
        # on it would bias the initial importance sample against that corner
        # (the badgrad class). Consistent with the
        # mutation kernel's zero-drift handling: keep the particle with its
        # certified likelihood and eval-zeroed gradient entries (its first
        # MALA move starts with prior-only drift),
        # and raise only above the systematic-breakage backstop.
        frac_tol = float(pipe.cfg.smc_tangent_bad_max_frac)
        thr_bad = int(math.ceil(frac_tol * n_phase2))
        bad2 = np.asarray(jax.device_get(stats2.bad_grad), bool)
        if n_bad > thr_bad:
            raise RuntimeError(
                f"{n_bad}/{n_phase2} SURVIVING particle(s) produced a finite "
                f"likelihood but a NON-FINITE gradient at initialization, above "
                f"the systematic-breakage backstop ({thr_bad} = "
                f"ceil({frac_tol:g} x {n_phase2})) -- this is not the "
                "theta-dependent tangent corner class but systematic AD "
                "breakage. Indices "
                f"{np.flatnonzero(bad2).tolist()}.")
        logger.warning(
            f"init 2/2: {n_bad}/{n_phase2} survivor(s) have a finite certified "
            f"likelihood but a non-finite forward-mode tangent (indices "
            f"{np.flatnonzero(bad2).tolist()}) -- kept with zeroed gradient "
            "entries (zero-drift first move; expected in the high-Z/low-C-O "
            "corner).")

    # cull re-certification failures; raise on true RT/AD deaths
    L_np = np.asarray(jax.device_get(L), np.float64)
    dead2 = ~np.isfinite(L_np) | (L_np <= REJECT_BELOW)
    if acc2_np is not None:
        cmax2 = int(pipe.fwd.chem.count_max)
        # a dead phase-2 particle is a re-certification failure (cull + backfill)
        # if its solve exhausted count_max OR exited without the canonical
        # certification (an unsettled state the eval gate floors to -1e30);
        # anything else dead is a genuine RT/AD blow-up (raise)
        recert_fail = dead2 & ((acc2_np >= cmax2) | ~conv2_np)
        rt_dead = dead2 & ~recert_fail
    else:
        recert_fail, rt_dead = dead2, np.zeros_like(dead2)
    if np.any(rt_dead):
        raise RuntimeError(
            f"{int(rt_dead.sum())}/{n_phase2} phase-2 particle(s) produced a "
            f"non-finite forward on a certified, NON-exhausted solve (indices "
            f"{np.flatnonzero(rt_dead).tolist()}) -- a genuine RT/AD problem, not a "
            "convergence cull; refusing to start the SMC on a crippled cloud.")
    if np.any(recert_fail):
        logger.warning(
            f"init 2/2: culled {int(recert_fail.sum())}/{n_phase2} marginal "
            f"survivor(s) that certified in phase 1 but cannot RE-certify within "
            f"count_max -- or exit uncertified (indices "
            f"{np.flatnonzero(recert_fail).tolist()}); "
            "backfilling from spares (part of the operational prior; reported "
            "with the phase-1 reject fraction).")
    alive2 = np.flatnonzero(~dead2)
    if alive2.size < target_n:
        raise RuntimeError(
            f"only {int(alive2.size)}/{n_phase2} phase-2 particles are healthy; need "
            f"{target_n}. Spares exhausted -- raise init_phase2_spare (currently "
            f"{spare}) or init_oversample, or investigate why so many survivors "
            "cannot re-certify.")
    sel2 = jnp.asarray(alive2[:target_n])
    U_keep, L, G, Y, refs = U_keep[sel2], L[sel2], G[sel2], Y[sel2], refs[sel2]
    logger.info(f"init 2/2 done in {time.perf_counter() - t0:.1f}s "
                f"(kept {target_n}/{n_phase2})")
    # Structured record of the operational-prior support measurement: these counts
    # define p(theta | forward model evaluates) relative to the declared prior and
    # feed the evidence conditioning report -- they must survive the run (results +
    # checkpoint), not just the log (which rotates).
    init_stats = dict(
        n_drawn=int(M),
        n_alive_phase1=int(n_alive),
        n_exhausted=int(exhausted.sum()),
        n_stalled_init=int(stalled.sum()),
        n_nonfinite=int((nonfinite & ~exhausted).sum()),
        n_phase2=int(n_phase2),
        n_recert_fail=int(np.asarray(recert_fail).sum()),
    )
    return U_keep, L, G, Y, refs, init_stats


def _make_mutation(pipe: Pipeline, n_mcmc: int):
    """Build the state-carrying mutation:

        mutate(key, U, Y, refs, L, G, beta, step, scale,
               where="mutation", dump_dir=None, dump_tag="", cost=None)
            -> (U, Y, refs, L, G, mean_acceptance, n_bad_grad,
                n_warm_capped, n_stalled, cost)

    ``cost`` (N,) is each particle's last accept count; it orders the queue when
    particles outnumber lanes. ``n_warm_capped`` / ``n_stalled`` count MH
    rejections from a capped or uncertified solve; both must stay ~0 in the
    late ladder.

    Runs ``n_mcmc`` sweeps of the configured kernel: "mala" (preconditioned MALA
    on the staged jvp(chem)+vjp(RT) gradient) or "rwm" (full-covariance random
    walk on the same preconditioner and step). Sweeps run as a host loop over a
    jitted single-sweep kernel, so each sweep's health is checked as it
    completes: badgrad events dump forensics to
    ``dump_dir/bad_grad_<dump_tag>_sweep<j>.npz`` and a sweep above the backstop
    raises there. Cold mode re-solves every proposal; warm continues from the
    carried column Y at its refs (lnZ, c_o), capped at warm_count_max.

    L and G are the raw log-likelihood and its u-space gradient; the tempered
    density is assembled per sweep from the analytic prior, so the carried
    state is beta-independent."""
    log_prior_u = pipe.log_prior_u
    _, _move_vg, _move_l = _get_batch_evals(pipe)[1:]
    theta_from_u = pipe.theta_from_u
    kernel = str(pipe.cfg.smc_mcmc_kernel).strip().lower()
    # More particles than lanes: the lane queue starts the first `cold_lanes`
    # proposals and refills in particle order, so a slow proposal that starts
    # late holds the sweep open. Each particle's previous proposal count
    # (`cost`) predicts its next one, so the proposals go to the queue slowest
    # first and come back in particle order. The evaluators are per-particle,
    # so the order changes only which tick a proposal enters its lane at (the
    # queue's convergence-scale contract), never the kernel. With particles <=
    # lanes every proposal starts at tick 0 and the order is not applied.
    ordered = 0 < int(pipe.cfg.cold_lanes) < int(pipe.cfg.smc_num_particles)

    def _slowest_first(ev, cost, *args):
        if not ordered:
            return ev(*args)
        n = cost.shape[0]
        perm = jnp.argsort(-cost)          # stable: ties keep particle order
        inv = jnp.argsort(perm)
        out = ev(*(a[perm] for a in args))
        return jax.tree_util.tree_map(
            lambda x: x[inv] if jnp.ndim(x) and x.shape[0] == n else x, out)

    def move_vg(cost, *args):
        return _slowest_first(_move_vg, cost, *args)

    def move_l(cost, *args):
        return _slowest_first(_move_l, cost, *args)

    def sweep(k, U, Y, refs, L, G, beta, step, scale, cost):
        def dlogprior(U_):
            return 1.0 - 2.0 * jax.nn.sigmoid(U_)

        kp, ka = jax.random.split(k)
        noise = jax.random.normal(kp, U.shape, dtype=U.dtype)
        GT = dlogprior(U) + beta * G
        # preconditioned Langevin with C = scale @ scale.T (scale is the cloud's
        # lower-triangular Cholesky factor): u' ~ N(u + step*C*grad, 2*step*C)
        cov = scale @ scale.T
        U_new = U + step * (GT @ cov) + jnp.sqrt(2.0 * step) * (noise @ scale.T)
        theta_new = jax.vmap(theta_from_u)(U_new)   # forensics; negligible next to the solves
        L_new, G_new, Y_new, refs_new, n_bad, stats = move_vg(cost, U_new, Y, refs)
        # Tangent-blown proposals (finite certified primal, non-finite tangent)
        # are zero-drift MALA moves, not rejections: the zeroed gradient entries
        # enter both proposal densities (GT_new) and, on acceptance, the carried
        # G. Rejecting biases against the theta corner where the class
        # concentrates. Valid only while the zero pattern is a
        # function of theta (cold mode, particles <= lanes). Visible through
        # badgrad= per sweep, the forensics dumps and the backstop.
        GT_new = dlogprior(U_new) + beta * G_new
        # asymmetric MH correction for the preconditioned Langevin proposal.
        # -log q = ||L^-1 (u' - u - step*C*grad)||^2 / (4 step) + const, and the
        # det(C) constant is identical both ways (scale is fixed within a sweep).
        def _whiten(r):
            return jax.scipy.linalg.solve_triangular(scale, r.T, lower=True).T
        df = _whiten(U_new - U - step * (GT @ cov))
        dr = _whiten(U - U_new - step * (GT_new @ cov))
        log_q_fwd = -0.25 / step * jnp.sum(df * df, axis=1)
        log_q_rev = -0.25 / step * jnp.sum(dr * dr, axis=1)
        LP = jax.vmap(log_prior_u)(U) + beta * L
        LP_new = jax.vmap(log_prior_u)(U_new) + beta * L_new
        log_acc = LP_new - LP + log_q_rev - log_q_fwd
        # L_new <= -1e29 is the invalid-forward sentinel, not a likelihood. It is
        # FINITE, so beta*L_new alone only rejects for beta >> 1e-29; force it.
        log_acc = jnp.where(jnp.isfinite(log_acc) & (L_new > REJECT_BELOW),
                            log_acc, -jnp.inf)
        accept = jnp.log(jax.random.uniform(ka, (U.shape[0],), dtype=U.dtype)) < log_acc
        U = jnp.where(accept[:, None], U_new, U)
        Y = jnp.where(accept[:, None, None], Y_new, Y)
        refs = jnp.where(accept[:, None], refs_new, refs)
        L = jnp.where(accept, L_new, L)
        G = jnp.where(accept[:, None], G_new, G)
        acc = jnp.minimum(jnp.exp(jnp.minimum(log_acc, 0.0)), 1.0)
        n_rej = jnp.sum((L_new <= REJECT_BELOW).astype(jnp.int32))
        return (U, Y, refs, L, G, jnp.mean(acc), n_rej, n_bad, stats,
                theta_new, L_new)

    def sweep_rwm(k, U, Y, refs, L, G, beta, step, scale, cost):
        """Gradient-free full-covariance random-walk Metropolis, same 11-tuple.

        u' ~ N(u, 2*step*C) with C = scale @ scale.T -- the MALA proposal with
        the drift dropped, so the proposal is symmetric and log q cancels from
        the MH ratio. Key splitting is the SAME two-way split in the SAME order
        as `sweep`, so the absolute-stage randomness contract is unchanged."""
        kp, ka = jax.random.split(k)
        noise = jax.random.normal(kp, U.shape, dtype=U.dtype)
        U_new = U + jnp.sqrt(2.0 * step) * (noise @ scale.T)
        theta_new = jax.vmap(theta_from_u)(U_new)   # forensics; keeps the tuple shape
        L_new, Y_new, refs_new, stats = move_l(cost, U_new, Y, refs)
        LP = jax.vmap(log_prior_u)(U) + beta * L
        LP_new = jax.vmap(log_prior_u)(U_new) + beta * L_new
        log_acc = LP_new - LP                       # symmetric proposal: log q cancels
        # same -1e29 invalid-forward sentinel gate as the MALA sweep
        log_acc = jnp.where(jnp.isfinite(log_acc) & (L_new > REJECT_BELOW),
                            log_acc, -jnp.inf)
        accept = jnp.log(jax.random.uniform(ka, (U.shape[0],), dtype=U.dtype)) < log_acc
        U = jnp.where(accept[:, None], U_new, U)
        Y = jnp.where(accept[:, None, None], Y_new, Y)
        refs = jnp.where(accept[:, None], refs_new, refs)
        L = jnp.where(accept, L_new, L)
        # G is carried UNTOUCHED, never zeroed: grad_u is a required checkpoint
        # key (the checkpoint writer and validate_warm both read it), and a
        # zeroed array there would read as a converged gradient rather than as
        # "this kernel never computed one".
        acc = jnp.minimum(jnp.exp(jnp.minimum(log_acc, 0.0)), 1.0)
        n_rej = jnp.sum((L_new <= REJECT_BELOW).astype(jnp.int32))
        # structurally no tangent on this path, so no bad-gradient class exists
        n_bad = jnp.zeros((), jnp.int32)
        return (U, Y, refs, L, G, jnp.mean(acc), n_rej, n_bad, stats,
                theta_new, L_new)

    sweep_jit = jax.jit(sweep_rwm if kernel == "rwm" else sweep)

    max_frac = float(pipe.cfg.smc_tangent_bad_max_frac)

    def mutate(key, U, Y, refs, L, G, beta, step, scale,
               where: str = "mutation", dump_dir=None, dump_tag: str = "",
               cost=None):
        keys = jax.random.split(key, n_mcmc)   # one key per sweep
        n_prop = int(U.shape[0])
        if cost is None:
            cost = jnp.zeros((n_prop,), jnp.int32)
        accs: List[float] = []
        n_bad_tot = n_cap_tot = n_stall_tot = 0
        for j in range(n_mcmc):
            (U, Y, refs, L, G, acc, n_rej, n_bad, stats, theta_new,
             L_new) = sweep_jit(keys[j], U, Y, refs, L, G, beta, step, scale,
                                cost)
            cost = stats.acc
            n_bad_j = int(jax.device_get(n_bad))
            n_cap_j = int(jax.device_get(stats.n_capped))
            n_stall_j = int(jax.device_get(stats.n_stalled))
            acc_j = float(jax.device_get(acc))
            # warmcap/stalled/badgrad = state-dependent rejection classes the MH
            # correction cannot see -- all must stay near zero in the
            # converged-ladder stages.
            logger.info(f"    sweep {j + 1}/{n_mcmc}: accept={acc_j:.2f} "
                        f"rejected={int(jax.device_get(n_rej))}/{n_prop} "
                        f"warmcap={n_cap_j} stalled={n_stall_j} "
                        f"badgrad={n_bad_j}")
            if n_bad_j > 0:
                dump_path = None
                if dump_dir is not None:
                    tag = f"{dump_tag}_" if dump_tag else ""
                    dump_path = Path(dump_dir) / f"bad_grad_{tag}sweep{j + 1}.npz"
                _check_mutation_health(
                    n_bad_j, f"{where}, sweep {j + 1}/{n_mcmc}",
                    forensics=dict(
                        bad_grad=stats.bad_grad, chem_tan_bad=stats.chem_tan_bad,
                        acc=stats.acc, longdy=stats.longdy, conv_ok=stats.conv_ok,
                        theta_proposal=theta_new, loglik_proposal=L_new),
                    dump_path=dump_path,
                    n_particles=n_prop, max_frac=max_frac)
            accs.append(acc_j)
            n_bad_tot += n_bad_j
            n_cap_tot += n_cap_j
            n_stall_tot += n_stall_j
        return (U, Y, refs, L, G, float(np.mean(accs)), n_bad_tot,
                n_cap_tot, n_stall_tot, cost)

    return mutate


def _check_mutation_health(n_bad: int, where: str, forensics: Dict[str, Any],
                           dump_path: Optional[Path], n_particles: int,
                           max_frac: float) -> None:
    """Handle flagged tangent pathologies from a mutation sweep (n_bad > 0):
    dump + warn, and raise only above the systematic-breakage backstop.

    A finite-likelihood/non-finite-tangent proposal at a certified state is a
    zero-drift MALA move (see the sweep comment). ``forensics``
    (per-particle device arrays) is dumped to ``dump_path`` and summarized in
    the log. A sweep above ceil(max_frac * n_particles) events is systematic
    AD breakage and raises."""
    f = {k: np.asarray(jax.device_get(v)) for k, v in forensics.items()}
    idx = np.flatnonzero(f["bad_grad"])
    n_chem = int(f["chem_tan_bad"][idx].sum()) if idx.size else 0
    detail = (
        f" Offending particle indices {idx.tolist()}; attribution: {n_chem} "
        f"chemistry-tangent side, {int(idx.size) - n_chem} RT-vjp side; "
        f"accept counts {f['acc'][idx].tolist()}; "
        f"longdy {[f'{v:.3g}' for v in f['longdy'][idx]]}.")
    if dump_path is not None:
        save_npz(Path(dump_path), **f)
        detail += f" Per-particle forensics dumped to {dump_path}."
    threshold = int(math.ceil(max_frac * int(n_particles)))
    if n_bad > threshold:
        raise RuntimeError(
            f"{n_bad} finite-likelihood/non-finite-gradient event(s) during {where} "
            f"exceed the systematic-breakage backstop ({threshold} = "
            f"ceil({max_frac:g} x {n_particles})): systematic AD breakage in "
            "the chemistry tangents or RT vjp." + detail)
    logger.warning(
        f"{n_bad} tangent-blown proposal(s) during {where}: handled as "
        f"zero-drift MALA moves (backstop {threshold} = ceil({max_frac:g} x "
        f"{n_particles})); a theta-independent rate is an anomaly." + detail)


# Reserved fold_in(key, .) namespaces. Stage keys use the absolute stage index
# (0, 1, 2, ...), so anything else must sit far above any reachable stage count.
_DRAW_KEY = 2_000_000
_INIT_KEY = 3_000_000


def _write_checkpoint(checkpoint_path, pipe: Pipeline, *, U, Y, refs, L, G, cost,
                      betas, ess_hist, acc_hist, logz_inc_hist, step_hist,
                      uniq_hist, capped_hist, stalled_hist, badgrad_hist, scale,
                      last_step, logZ, init_stats, log_step) -> None:
    """Atomically write the SMC checkpoint (single writer for the init-level and
    per-stage checkpoints, so their schemas stay in lockstep by construction).
    ``last_step=-1`` marks the INIT-LEVEL checkpoint (written right after
    _init_state, before any tempering stage): betas=[0.0] and empty histories,
    so the resume path enters the ladder at stage 0 like a fresh post-init run
    and a stage-0 death keeps the hours-scale two-phase init."""
    U_np = np.asarray(jax.device_get(U), np.float64)
    theta_ck = np.asarray(jax.device_get(jax.vmap(pipe.theta_from_u)(U)), np.float64)
    tmp = Path(checkpoint_path).with_suffix(".tmp.npz")
    save_npz(tmp, u_particles=U_np, theta_particles=theta_ck,
             betas=np.asarray(betas), ess=np.asarray(ess_hist),
             acceptance_rate=np.asarray(acc_hist),
             logZ_increment=np.asarray(logz_inc_hist),
             step_size_history=np.asarray(step_hist),
             unique_particles=np.asarray(uniq_hist, np.int64),
             warm_capped=np.asarray(capped_hist, np.int64),
             warm_stalled=np.asarray(stalled_hist, np.int64),
             # per-stage badgrad counts (counted, not rejected)
             tangent_rejected=np.asarray(badgrad_hist, np.int64),
             # lower-triangular Cholesky factor of the proposal covariance
             scale_chol=np.asarray(scale),
             # the RM state itself, so resume restores it WITHOUT the
             # exp(log(.)) roundtrip through step_size_history (not bit-exact
             # on every libm; bit-identical resume is the contract)
             mala_log_step=np.asarray(float(log_step)),
             last_step=np.asarray(int(last_step), np.int64),
             logZ=np.asarray(logZ),
             # Target-exactness stamp: warm (likelihood depends on sampler
             # history), cold (exact up to the lane refill tick),
             # or "none" for a stub pipeline. Every artifact carries it.
             chem_mode=np.asarray(str(getattr(pipe, "chem_mode", None) or "none")),
             # Full target identity (certificate.target_digest): resolved config,
             # ordered priors, observation arrays, opacity/network content, code
             # commits and versions.
             target_digest=np.asarray(str(getattr(pipe, "target_digest", "") or "")),
             approximate_history_dependent_target=np.asarray(
                 1 if str(getattr(pipe, "chem_mode", None)) == "warm" else 0,
                 np.int64),
             init_stats_keys=np.asarray(list(init_stats.keys())),
             init_stats_vals=np.asarray(list(init_stats.values()), np.int64),
             # carried per-particle state: resume warm-continues without re-init
             y_state=np.asarray(jax.device_get(Y), np.float64),
             chem_refs=np.asarray(jax.device_get(refs), np.float64),
             loglik=np.asarray(jax.device_get(L), np.float64),
             grad_u=np.asarray(jax.device_get(G), np.float64),
             # each particle's last proposal count: the queue order when
             # particles outnumber lanes, so a resumed run orders as the
             # uninterrupted one did
             move_cost=np.asarray(jax.device_get(cost), np.int64))
    tmp.replace(checkpoint_path)


def run_smc_loop(pipe: Pipeline, key, progress: bool = True,
                 checkpoint_path: Optional[Path] = None,
                 walltime_seconds: float = 0.0,
                 resume_from: Optional[Path] = None) -> Dict[str, Any]:
    """Adaptive-tempered SMC to beta=1. Checkpoints after every stage; stops cleanly if
    the wall-clock budget is exceeded (partial output is always usable -- but flagged:
    a beta<1 stop yields TEMPERED draws, and every export/plot path labels them so).
    Pass ``resume_from=<checkpoint.npz>`` to continue a killed run from its tempered
    cloud (the ladder resumes at the checkpointed beta; completed stages are kept).

    EVIDENCE SEMANTICS (evidence_report): the returned ``logZ`` is the evidence
    under the OPERATIONAL prior (box, T-P window and converged chemistry,
    renormalized); ``logZ_box`` is the ZERO-FILLED box evidence, SOLVER-DEPENDENT
    through the convergence indicator (count_max, warm_count_max, tolerances,
    the canonical conv_normal gate in _proposal_converged, init history). Never
    difference bare ``logZ`` across models with different support fractions."""
    cfg = pipe.cfg
    dtype = pipe.dtype
    N = int(cfg.smc_num_particles)
    n_dim = pipe.n_dim
    target_ess = float(cfg.smc_target_ess_frac) * N
    t_start = time.perf_counter()

    # oversampled cold-init draw: _init_state rejects the non-converged corners and
    # culls back to N healthy particles (resume overwrites U from the checkpoint below).
    # ``key`` is never rebound in this function: every draw is fold_in(key, <namespace>),
    # so a stage's randomness depends only on (seed, absolute stage).
    U = pipe.sample_prior_u(jax.random.fold_in(key, _INIT_KEY),
                            _init_draw_count(pipe, N))

    # _DRAW_KEY / _INIT_KEY sit outside the per-stage fold_in namespace below.
    step = C.MALA_STEP0
    # Both kernels share the step, its clamps and the Robbins-Monro state;
    # only the acceptance they aim at differs (0.234 is the d->inf optimum for a
    # random walk, 0.55 the MALA target).
    target_acc = C.TARGET_ACCEPT[str(cfg.smc_mcmc_kernel).strip().lower()]
    log_step = math.log(min(max(step, C.STEP_MIN), C.STEP_MAX))
    scale = np.eye(n_dim)
    mutate = _make_mutation(pipe, int(cfg.smc_num_mcmc_steps))

    beta = 0.0
    betas: List[float] = [0.0]
    ess_hist, acc_hist, logz_inc_hist, step_hist, uniq_hist = [], [], [], [], []
    capped_hist: List[int] = []
    stalled_hist: List[int] = []
    badgrad_hist: List[int] = []
    logZ = 0.0
    init_stats: Optional[Dict[str, int]] = None
    cost = jnp.zeros((N,), jnp.int32)

    state_loaded = False
    if resume_from is not None and not Path(resume_from).exists():
        raise FileNotFoundError(
            f"resume_from={resume_from} was requested but does not exist. Refusing to "
            "silently start a fresh run (the CLI refuses the same way); pass "
            "resume_from=None to start over.")
    if resume_from is not None:
        from retrieval_framework.certificate import refuse_mismatched_resume
        refuse_mismatched_resume(Path(resume_from), getattr(pipe, "target_digest", ""))
        ck = np.load(resume_from)
        if ck["u_particles"].shape != (N, n_dim):
            raise ValueError(f"checkpoint particles {ck['u_particles'].shape} != ({N},{n_dim}); "
                             "resume requires the same smc_num_particles and parameter set")
        # The checkpoint carries a TARGET-EXACTNESS STAMP (written below).
        # Adopting betas/logZ/loglik across a target change would splice two
        # different densities into one evidence integral, and nothing
        # downstream could detect it -- the certificate reads the last job only.
        ck_mode = str(ck["chem_mode"])
        now_mode = str(getattr(pipe, "chem_mode", None) or "none")
        if ck_mode != now_mode:
            raise ValueError(
                f"checkpoint was produced with chem_mode={ck_mode!r} but this run "
                f"is chem_mode={now_mode!r}; the tempering ladder and logZ are not "
                "transferable across a target change. Start a fresh run or restore "
                "the original chem_mode.")
        U = jnp.asarray(ck["u_particles"], dtype)
        betas = [float(b) for b in ck["betas"]]
        beta = betas[-1]
        ess_hist = [float(x) for x in ck["ess"]]
        acc_hist = [float(x) for x in ck["acceptance_rate"]]
        logz_inc_hist = [float(x) for x in ck["logZ_increment"]]
        step_hist = [float(x) for x in ck["step_size_history"]]
        uniq_hist = [int(x) for x in ck["unique_particles"]]
        logZ = float(ck["logZ"])
        # The digest binds the code commit, so refuse_mismatched_resume has
        # already refused any checkpoint this code did not write: every key the
        # writer above writes is present.
        scale = np.asarray(ck["scale_chol"], np.float64)
        capped_hist = [int(x) for x in ck["warm_capped"]]
        stalled_hist = [int(x) for x in ck["warm_stalled"]]
        badgrad_hist = [int(x) for x in ck["tangent_rejected"]]
        init_stats = {str(k): int(v) for k, v in
                      zip(ck["init_stats_keys"], ck["init_stats_vals"])}
        log_step = float(ck["mala_log_step"])
        Y = jnp.asarray(ck["y_state"], dtype)
        refs = jnp.asarray(ck["chem_refs"], dtype)
        L = jnp.asarray(ck["loglik"], dtype)
        G = jnp.asarray(ck["grad_u"], dtype)
        cost = jnp.asarray(ck["move_cost"], jnp.int32)
        state_loaded = True
        if beta == 0.0:
            logger.info(f"RESUMED from {resume_from}: INIT-LEVEL checkpoint "
                        "(two-phase init recovered; ladder starts at stage 0, beta=0)")
        else:
            logger.info(f"RESUMED from {resume_from}: stage {len(betas)-1}, "
                        f"beta={beta:.4f}, logZ={logZ:.2f}")

    if not state_loaded:
        # one batched cold two-stage solve per particle: the ONLY solve-from-baseline
        # work in the whole run (every mutation proposal warm-continues from here)
        t0 = time.perf_counter()
        U, L, G, Y, refs, init_stats = _init_state(pipe, U, target_n=N)
        jax.block_until_ready(L)
        # fold the T-P-window rejection tally in so init_stats fully describes the
        # operational prior p(theta | window valid AND chemistry converges)
        tp_stats = dict(getattr(pipe, "tp_prior_stats", {}) or {})
        init_stats["tp_n_drawn"] = int(tp_stats.get("n_drawn", 0))
        init_stats["tp_n_kept"] = int(tp_stats.get("n_kept", 0))
        logger.info(f"Initialized particle state (cold likelihood + move-map gradient) "
                    f"in {time.perf_counter()-t0:.1f}s")
        if checkpoint_path is not None:
            # init-level checkpoint (last_step=-1): a stage-0 death -- bad-gradient
            # raise, OOM, preemption -- must not throw away the hours-scale init;
            # RESUME=1 recovers it and enters the ladder at beta=0.
            _write_checkpoint(checkpoint_path, pipe, U=U, Y=Y, refs=refs,
                              L=L, G=G, cost=cost,
                              betas=betas, ess_hist=ess_hist,
                              acc_hist=acc_hist, logz_inc_hist=logz_inc_hist,
                              step_hist=step_hist, uniq_hist=uniq_hist,
                              capped_hist=capped_hist, stalled_hist=stalled_hist,
                              badgrad_hist=badgrad_hist,
                              scale=scale, last_step=-1, logZ=logZ,
                              init_stats=init_stats, log_step=log_step)
            logger.info(f"init-level checkpoint written to {checkpoint_path} "
                        "(RESUME=1 recovers the init from here)")

    logger.info("starting tempering ladder (stage 0 includes the one-time "
                "mutation-kernel compile)")
    it = range(int(cfg.smc_max_steps))
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc="adaptive tempered SMC", leave=True)
        except Exception:
            pass

    for i in it:
        # ABSOLUTE stage index: `i` restarts at 0 on every resume, `betas` does not.
        # Every random draw below is fold_in(seed_key, stage), so (a) a resumed run
        # never REPLAYS the resample offsets and MALA noise of the stages the killed
        # job already ran -- a split chain would restart from the run's own seed --
        # and (b) a chained run is bit-identical to an uninterrupted one. It is also
        # what labels the checkpoints, logs and badgrad dumps, so a second job does
        # not overwrite the first job's stage000 forensics.
        stage = len(betas) - 1
        k_res, k_mut = jax.random.split(jax.random.fold_in(key, stage))
        # (1) carried likelihood at current particles -> (2) next temperature via ESS
        # bisection (L travels with the particles; nothing is re-evaluated here)
        L_np = np.asarray(jax.device_get(L), np.float64)
        if not np.all(np.isfinite(L_np)):
            # rejected particles are floored at -1e30 inside eval_batch, so a
            # non-finite CARRIED likelihood is an invariant violation -- raise, never
            # normalize it away
            raise FloatingPointError(
                f"non-finite carried log-likelihood at SMC stage {stage} "
                f"({int(np.sum(~np.isfinite(L_np)))}/{N} particles)")
        dbeta = _next_dbeta(L_np, beta, target_ess)
        beta_new = min(1.0, beta + dbeta)
        # (2) evidence increment + weights (uniform prior weights each stage post-resample)
        a = dbeta * (L_np - L_np.max())
        w = np.exp(a); w_sum = w.sum()   # >= 1: the max-shifted best particle is exp(0)
        logZ_inc = float(dbeta * L_np.max() + math.log(w_sum) - math.log(N))
        if not math.isfinite(logZ_inc):
            raise FloatingPointError(
                f"non-finite evidence increment at SMC stage {stage} "
                f"(beta {beta:.3e} -> {beta_new:.3e}) -- refusing to corrupt logZ")
        logZ += logZ_inc
        w_norm = w / w_sum
        ess = float(1.0 / np.sum(w_norm * w_norm))
        # (3) systematic resample (the carried state travels with its particle)
        idx = _systematic_resample_idx(k_res, jnp.asarray(w_norm, dtype), N)
        U, Y, refs, L, G = U[idx], Y[idx], refs[idx], L[idx], G[idx]
        cost = cost[idx]
        # (3.5) preconditioner from the freshly RESAMPLED cloud (absolute per-dim
        # width: the proposal tracks the tempered posterior as it narrows)
        scale = _proposal_scale(np.asarray(jax.device_get(U)), cap=C.SCALE_CLIP)
        # (4) mutate at the new temperature -- badgrad events are handled as
        # zero-drift moves and warn+dump per-particle forensics next to the
        # checkpoint; a sweep beyond the systematic-breakage backstop raises
        # INSIDE mutate at the offending sweep
        U, Y, refs, L, G, acc, n_bad, n_capped, n_stalled, cost = mutate(
            k_mut, U, Y, refs, L, G,
            jnp.asarray(beta_new, dtype),
            jnp.asarray(math.exp(log_step), dtype),
            jnp.asarray(scale, dtype),
            where=f"SMC stage {stage} (beta={beta_new:.3e})",
            dump_dir=(Path(checkpoint_path).parent
                      if checkpoint_path is not None else None),
            dump_tag=f"stage{stage:03d}", cost=cost)
        jax.block_until_ready(U)
        acc_f = float(acc)
        n_capped_f = int(n_capped)
        n_stalled_f = int(n_stalled)
        n_bad_f = int(n_bad)
        U_np = np.asarray(jax.device_get(U), np.float64)
        n_uniq = int(np.unique(np.round(U_np, _UNIQ_DECIMALS), axis=0).shape[0])
        # (5) Robbins-Monro step-size trim toward the target acceptance (fine-tuning
        # only -- the width is carried by the absolute preconditioner above)
        if math.isfinite(acc_f):
            log_step += acc_f - target_acc
            log_step = math.log(min(max(math.exp(log_step), C.STEP_MIN), C.STEP_MAX))

        beta = beta_new
        betas.append(beta); ess_hist.append(ess); acc_hist.append(acc_f)
        logz_inc_hist.append(logZ_inc); step_hist.append(math.exp(log_step)); uniq_hist.append(n_uniq)
        capped_hist.append(n_capped_f)
        stalled_hist.append(n_stalled_f)
        badgrad_hist.append(n_bad_f)
        elapsed = time.perf_counter() - t_start
        if hasattr(it, "set_postfix"):
            it.set_postfix(beta=f"{beta:.2e}", ess=f"{ess:.0f}", acc=f"{acc_f:.2f}")
        logger.info(f"SMC {stage:03d}: beta={beta:.3e} ESS={ess:.1f}/{N} accept={acc_f:.3f} "
                    f"unique={n_uniq}/{N} step={math.exp(log_step):.3g} logZ={logZ:.2f} "
                    f"warmcap={n_capped_f} stalled={n_stalled_f} badgrad={n_bad_f} "
                    f"elapsed={elapsed/60:.1f}min")

        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, pipe, U=U, Y=Y, refs=refs,
                              L=L, G=G, cost=cost,
                              betas=betas, ess_hist=ess_hist,
                              acc_hist=acc_hist, logz_inc_hist=logz_inc_hist,
                              step_hist=step_hist, uniq_hist=uniq_hist,
                              capped_hist=capped_hist, stalled_hist=stalled_hist,
                              badgrad_hist=badgrad_hist,
                              scale=scale, last_step=stage, logZ=logZ,
                              init_stats=init_stats, log_step=log_step)

        if beta >= 1.0 - _BETA_DONE_TOL:
            break
        if walltime_seconds and elapsed > walltime_seconds:
            logger.warning(f"walltime budget {walltime_seconds/3600:.1f}h exceeded at stage {stage} "
                           f"(beta={beta:.3f}); stopping cleanly with partial posterior.")
            break

    reached = beta >= 1.0 - BETA_TOL
    # posterior draws: at beta=1 particles are equally weighted; sample with replacement.
    # When the ladder stopped early (walltime) these are TEMPERED (beta<1) draws, NOT
    # posterior samples -- reached_beta1/final_beta travel with every output and the
    # plotting/export paths refuse the "posterior" label without them.
    n_draws = int(cfg.num_chains) * int(cfg.num_samples)
    sub = jax.random.fold_in(key, _DRAW_KEY + len(betas))
    draw_idx = np.asarray(jax.device_get(jax.random.choice(sub, N, (n_draws,), replace=True)))
    theta_draws = np.asarray(jax.device_get(jax.vmap(pipe.theta_from_u)(U)), np.float64)[draw_idx]
    theta_draws = theta_draws.reshape(int(cfg.num_chains), int(cfg.num_samples), n_dim)

    # ---- evidence conditioning report (semantics in evidence_report) --------
    ev = evidence_report(logZ, init_stats)
    logger.info(
        f"evidence conditioning: logZ(conditioned/operational) = {logZ:.2f}; "
        f"ZERO-FILLED box evidence logZ_box = {ev['logZ_box']:.2f} +/- "
        f"{ev['log_support_fraction_err']:.2f} (= logZ + ln(f_tp*f_conv); "
        f"the integral of pi*L*1[T-P valid AND converged] over the "
        f"declared box, exact up to the lane queue's convergence-scale refill "
        f"dependence -- SOLVER-DEPENDENT via the convergence indicator; "
        f"Bayes factors only at matched solver settings AND with the "
        f"attrition shown likelihood-negligible). Supports: T-P window "
        f"f_tp={ev['f_tp']:.3f} (solver-independent), convergence "
        f"f_conv={ev['f_conv']:.3f} (solver-dependent).")

    # Monte Carlo error on logZ: each stage's relative weight variance
    # ~ (N/ESS - 1)/N, summed. An optimistic lower bound (logz_err_lower_bound);
    # the full estimate is the seed-to-seed spread.
    logZ_err_lb = logz_err_lower_bound(ess_hist, N)

    return dict(
        U=np.asarray(jax.device_get(U), np.float64), reached_beta1=reached, final_beta=beta,
        step_size_used=math.exp(log_step), betas=np.asarray(betas),
        ess=np.asarray(ess_hist), acceptance_rate=np.asarray(acc_hist),
        logZ_increment=np.asarray(logz_inc_hist), logZ=logZ,
        logZ_err_lb=logZ_err_lb,
        # evidence-semantics fields from evidence_report
        log_support_fraction=ev["log_support_fraction"],
        log_support_fraction_err=ev["log_support_fraction_err"],
        logZ_box=ev["logZ_box"],
        log_support_physical=ev["log_support_physical"],
        log_support_physical_err=ev["log_support_physical_err"],
        log_conv_attrition=ev["log_conv_attrition"],
        log_conv_attrition_err=ev["log_conv_attrition_err"],
        init_stats=init_stats,
        warm_capped=np.asarray(capped_hist, np.int64),
        warm_stalled=np.asarray(stalled_hist, np.int64),
        # per-stage badgrad counts
        tangent_rejected=np.asarray(badgrad_hist, np.int64),
        step_size_history=np.asarray(step_hist), unique_particles=np.asarray(uniq_hist, np.int64),
        theta_draws=theta_draws,
    )
