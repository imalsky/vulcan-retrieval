"""Pipeline: assemble the theta-space forward into a bounded-prior u-space posterior,
and run a self-contained adaptive-tempered SMC with a preconditioned forward-mode-jvp
MALA mutation kernel.

This mirrors the SWAMPE retrieval (BlackJAX adaptive-tempered SMC + custom
forward-mode-gradient MALA + per-stage step/preconditioner adaptation + per-stage
checkpointing), but the SMC core is implemented directly in JAX so the code has NO
BlackJAX dependency -- the VULCAN-JAX conda env does not ship it, and a pip-install on
the HPC is fragile. The algorithm is the standard Del Moral (2006) resample-move SMC:

  each stage:  (1) pick the next inverse-temperature beta' by ESS bisection,
               (2) reweight + accumulate the log-evidence increment,
               (3) systematic resample,
               (4) mutate with `num_mcmc_steps` preconditioned-MALA sweeps at the
                   tempered target log_prior_u(u) + beta'*loglik(u),
               (5) Robbins-Monro adapt the step size + refresh the diagonal
                   preconditioner from the mutated cloud,
               (6) atomically checkpoint.

The MALA gradient is the crux: the VULCAN-JAX runner's `lax.while_loop` supports jvp but
not vjp, so the likelihood gradient is built from forward-mode jvps (one per u-dimension,
vmapped) and exposed to `jax.value_and_grad` through a `custom_vjp` -- exactly the SWAMPE
trick -- so no reverse-mode tape is ever taped through the chemistry solve.

GH200 batched architecture (2026-07-06 rework -- see README section C):
the per-particle gradient functions above are kept for validation, but the SMC hot path
uses STAGED batched evaluators that split the chain at the chemistry/RT boundary:

  * chemistry: forward-mode jvp lanes for the n_chem_tp dims only, with ALL particles
    batched into ONE vmapped `lax.while_loop` (per-lane state is ~MB, so width is nearly
    free -- wide batches are what keep the GPU busy instead of launch-latency-bound);
  * RT: ONE reverse-mode vjp per particle (legal -- there is no while_loop inside the
    ExoJax RT), `lax.map`-chunked over particles because PreMODIT tangent/tape
    intermediates cost ~GB per lane (this is what OOM'd the old all-in-one design at
    1.5 TiB); a single backward pass replaces the old 6 forward tangents + 3-dim jacfwd;
  * offsets / noise-inflation: analytic (unchanged).

The mutation kernel additionally CARRIES each particle's converged chemistry column and
warm-starts every proposal's solve from it with incremental lnZ/C-O scaling (the
validated continuation pattern) -- ~count_min-step re-converges instead of full cold
two-stage solves. `smc_chem_mode="cold"` restores the published solve-from-baseline map.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
# NOTE: retrieval_forward (-> vulcan_chem -> VULCAN-JAX env setup + chdir) is imported
# LAZILY inside build_pipeline, so the SMC core + u-space machinery in this module can
# be unit-tested (tests/test_smc_gaussian.py) without touching the heavy stack.

import jax
import jax.numpy as jnp

logger = logging.getLogger("retrieval")


# =============================================================================
# Pipeline container
# =============================================================================
class Pipeline:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


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
                            runner's canonical certification (stall fallback /
                            budget exit) -- MH rejections, not AD pathologies
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
    count_since_new_min, conv_normal]`` (see forward.vulcan_chem.ConvDiag).
    THE gate that decides whether a proposal's state -- and therefore its jvp
    tangents -- is trusted; kept in one place so the predicate is swappable.

    Current predicate: the runner's own canonical two-branch certification
    recomputed at the exit state (``conv_normal``). A stall-fallback or budget
    exit reads False even when longdy sits under yconv_min -- the class that
    passed the old accept-count-only gate on NAS job 65200 (16/864 warm
    proposals: primal certified, tangent never settled -> non-finite gradient).
    Measurement backing the choice: validation/diag_warm_stall_tangent.py.
    """
    return cd_vec[:, 4] > 0.5


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
    lo_log = jnp.log10(jnp.clip(prior_lo, 1e-300, None))
    span_log = jnp.log10(jnp.clip(prior_hi, 1e-300, None)) - lo_log

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
        eps = jnp.asarray(1e-6, dtype=dtype)
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


def _map_chunked(fn, args, chunk):
    """vmap ``fn`` over the leading axis of every leaf of ``args`` (a pytree of stacked
    per-particle inputs), running ``lax.map`` over padded chunks of ``chunk`` particles
    to bound peak memory. ``chunk<=0`` (or >= n) is a single all-particles vmap.
    Identical results to the plain vmap for any chunk (padding rows are dropped)."""
    leaves = jax.tree_util.tree_leaves(args)
    n = int(leaves[0].shape[0])
    vfn = jax.vmap(fn)
    if chunk <= 0 or chunk >= n:
        return vfn(args)
    n_pad = (-n) % chunk
    if n_pad:
        args = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x, x[:n_pad]], axis=0), args)
    args = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, chunk) + x.shape[1:]), args)
    out = jax.lax.map(vfn, args)
    return jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:])[:n], out)


def build_pipeline(cfg: C.Config) -> Pipeline:
    """Build the forward, observation operators, u-space prior/likelihood, and the
    forward-mode-gradient likelihood wrapper. No inference, no file IO.

    IMPORTANT (trace-time baking): the likelihood closes over ``pipe.obs_depth_jax`` /
    ``pipe.obs_sigma_jax``; the first jitted call bakes them in as constants. Call
    ``pipe.set_observations`` exactly ONCE, before any inference/tuning call, and never
    swap observations afterwards in the same process (a second call now raises).
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
    # The Guillot profile is drawn raw (tp_profile no longer clips). A draw whose T-P
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
        if n_tp == 0:
            return jnp.asarray(True)
        T_art = tp_eval(jnp.asarray(theta)[3:3 + n_tp], p_art_j)
        return jnp.all(jnp.isfinite(T_art) & (T_art >= tp_T_min) & (T_art <= tp_T_max))

    # ---- observations + linear operators (binning, offsets) ----
    obs, real_bins = OBS.get_observation_grid(cfg, fwd.wl_um)
    keep, B = OBS.build_binning_matrix(fwd.wl_um, obs)
    # apply keep to the observed arrays so obs bins line up with B's rows
    for k in ("wl", "wl_lo", "wl_hi", "depth", "sigma", "group"):
        obs[k] = np.asarray(obs[k])[keep]
    obs["groups"] = list(dict.fromkeys(np.asarray(obs["group"]).tolist()))
    O = OBS.build_offset_design(obs)
    groups = list(obs["groups"])
    n_bin = int(B.shape[0])
    logger.info(f"Observations: {'REAL product bins' if real_bins else 'synthetic grid'} | "
                f"{n_bin} bins | groups={groups} | offset cols={O.shape[1]}")
    if n_bin < 2:
        raise RuntimeError(f"only {n_bin} usable observed bins in the model band; widen the band")

    B_jax = jnp.asarray(B, dtype=dtype)
    O_jax = jnp.asarray(O, dtype=dtype)

    # ---- parameter layout ----
    specs = C.specs_from_config(cfg, groups=groups)
    names = [s.name for s in specs]
    kinds = [s.kind for s in specs]
    labels = [s.label for s in specs]
    n_dim = len(specs)
    n_chem_tp = 3 + fwd.n_tp
    # The chem+T-P prefix is unpacked by fixed position (theta[0:3]=chem,
    # theta[3:3+n_tp]=T-P). Assert the layout EXACTLY, not just "chem/tp appear in
    # the prefix": dropping a chem toggle shortens the block, and the old subset
    # check passed when all nuisances were also off, silently truncating the
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
            "positional and load-bearing -- do not drop infer_lnZ/c_o/lnKzz.")
    lnR0_idx = names.index("lnR0") if "lnR0" in names else None
    cloud_idx = [i for i, k in enumerate(kinds) if k == "cloud"]
    off_idx = [i for i, k in enumerate(kinds) if k == "offset"]
    noise_idx = names.index("noise_inflation") if "noise_inflation" in names else None
    off_lo = off_idx[0] if off_idx else None
    n_off = len(off_idx)
    cloud_lo = cloud_idx[0] if cloud_idx else None
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
        if n_tp == 0:
            tp_prior_stats["n_drawn"] += n_particles
            tp_prior_stats["n_kept"] += n_particles
            return sample_prior_u(rng_key, n_particles)
        key = rng_key
        kept, have, drawn = [], 0, 0
        over = max(n_particles, 16)
        max_draw = max(64 * n_particles, 4096)   # loud cap: fail rather than loop forever
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
        return theta[cloud_lo:cloud_lo + n_cloud] if n_cloud else None

    def observed_depth_model(theta):
        theta = jnp.asarray(theta, dtype=dtype)
        chem_theta = theta[:n_chem_tp]
        lnR0 = theta[lnR0_idx] if lnR0_idx is not None else jnp.asarray(0.0, dtype)
        native = fwd.native_depth(chem_theta, lnR0, _cloud_from(theta))  # (n_native,)
        binned = B_jax @ native                                # (n_bin,)
        if n_off > 0:
            offs = jax.lax.dynamic_slice_in_dim(theta, off_lo, n_off) * OBS.OFFSET_UNIT
            binned = binned + O_jax @ offs
        return binned

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

    _REJECT = jnp.asarray(-1.0e30, dtype=dtype)

    def log_likelihood_u(u):
        theta = theta_from_u(u)

        def _bad():
            return _REJECT

        def _good():
            # only reached for an in-window T-P, so the RT never extrapolates
            mu = observed_depth_model(theta)
            return jax.lax.cond(jnp.all(jnp.isfinite(mu)),
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
        if n_dim == 1:
            return y0, jnp.atleast_1d(dy0)
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
        r0 = theta[lnR0_idx] if lnR0_idx is not None else jnp.asarray(0.0, dtype)
        cloudp = _cloud_from(theta)

        eye_c = jnp.eye(n_chem_tp, dtype=u.dtype)

        def _chain(cc):
            return fwd.native_depth_aux(cc, r0, cloudp)

        (d_all, aux_all), (J_chem, _) = jax.vmap(
            lambda v: jax.jvp(_chain, (c,), (v,)))(eye_c)
        d0 = d_all[0]                                            # primal native depth
        aux = jax.tree_util.tree_map(lambda x: x[0], aux_all)    # primal ART-grid profiles
        # J_chem: (n_chem_tp, n_native) tangent stack

        # mu = B @ native + O @ offsets  (identical to observed_depth_model)
        mu = B_jax @ d0
        if n_off > 0:
            offs = theta[off_lo:off_lo + n_off] * OBS.OFFSET_UNIT
            mu = mu + O_jax @ offs

        sig = _sigma_for(theta)
        resid = pipe.obs_depth_jax - mu
        wres = resid / (sig * sig)                                   # dL/dmu
        val = _gauss_loglik(mu, theta)

        Btw = B_jax.T @ wres                                         # (n_native,)
        g_theta = jnp.zeros((n_dim,), dtype=u.dtype)
        g_theta = g_theta.at[:n_chem_tp].set(J_chem @ Btw)

        # RT-only dims (lnR0 + cloud params): one jacfwd through the RT at frozen aux
        rt_idx = ([lnR0_idx] if lnR0_idx is not None else []) + cloud_idx
        if rt_idx:
            has_r = lnR0_idx is not None
            rv0 = jnp.stack([theta[i] for i in rt_idx])

            def _rt(rv):
                r = rv[0] if has_r else jnp.asarray(0.0, u.dtype)
                if n_cloud:
                    cp = rv[1:] if has_r else rv
                else:
                    cp = None
                return fwd.rt_depth(aux, r, cp)

            J_rt = jax.jacfwd(_rt)(rv0)                          # (n_native, n_rt)
            for j, i in enumerate(rt_idx):
                g_theta = g_theta.at[i].set(jnp.dot(J_rt[:, j], Btw))
        if n_off > 0:
            g_theta = g_theta.at[off_lo:off_lo + n_off].set(OBS.OFFSET_UNIT * (O_jax.T @ wres))
        if noise_idx is not None:
            k = theta[noise_idx]
            sig0 = pipe.obs_sigma_jax
            g_theta = g_theta.at[noise_idx].set(
                jnp.sum(resid * resid / (sig0 * sig0 * k ** 3)) - mu.size / k)

        grad_u = g_theta * dtheta_du
        # reject an out-of-window T-P (no clip) as well as a blown forward
        finite = jnp.all(jnp.isfinite(d0)) & tp_valid(theta)
        val = jnp.where(finite, val, jnp.asarray(-1.0e30, dtype=dtype))
        return val, jnp.where(finite, grad_u, jnp.zeros_like(grad_u))

    grad_mode = str(cfg.gradient_mode).strip().lower()
    if grad_mode not in ("block", "naive"):
        raise ValueError(f"gradient_mode must be 'block' or 'naive', got {grad_mode!r}")
    _vg_impl = _value_and_grad_block if grad_mode == "block" else _value_and_grad_naive

    use_custom = bool(cfg.smc_use_custom_gradients) and (n_dim <= int(cfg.smc_custom_grad_max_dim))
    if use_custom:
        @jax.custom_vjp
        def loglik_fwd(u):
            return log_likelihood_u(u)

        def _fwd(u):
            # Return the gradient RAW. A rejected proposal (non-finite forward) is
            # already handled inside _vg_impl (val -> -1e30, grad -> 0: principled MH
            # rejection). A finite forward with a non-finite gradient is an AD
            # pathology and MUST reach the caller's bad-grad detector so the run
            # raises loudly -- zeroing it here would silently degrade MALA to a random
            # walk (project rule: loud errors, no silent fallbacks).
            return _vg_impl(u)

        def _bwd(grad, g):
            return (g * grad,)

        loglik_fwd.defvjp(_fwd, _bwd)
    else:
        loglik_fwd = log_likelihood_u

    # =========================================================================
    # Staged BATCHED likelihood / gradient (the SMC hot path; see module docstring).
    # Exact -- the same chain rule as _value_and_grad_block, regrouped:
    #   dL/dtheta_chem[k] = < d aux / d theta[k]  (fwd jvp through the chemistry),
    #                         d L / d aux          (rev vjp through the RT) >.
    # =========================================================================
    chem_mode = str(cfg.smc_chem_mode).strip().lower()
    if chem_mode not in ("warm", "cold"):
        raise ValueError(f"smc_chem_mode must be 'warm' or 'cold', got {chem_mode!r}")
    rt_chunk = int(cfg.smc_rt_chunk or 0)
    rt_vjp_chunk = int(cfg.smc_rt_vjp_chunk or 0)
    chem_chunk = int(cfg.smc_chem_chunk or 0)
    y_baseline = jnp.asarray(fwd.y_baseline, dtype=dtype)          # (nz, ni)
    eye_c = jnp.eye(n_chem_tp, dtype=dtype)
    have_cloud = bool(n_cloud)

    def _rt_wrap(aux, r0, cp):
        # cp is a dummy (0,) array when clouds are off, so the vjp signature is fixed
        return fwd.rt_depth(aux, r0, cp if have_cloud else None)

    def _mu_from_depth(depth, theta):
        mu = B_jax @ depth
        if n_off > 0:
            mu = mu + O_jax @ (theta[off_lo:off_lo + n_off] * OBS.OFFSET_UNIT)
        return mu

    def _rt_val(args):
        """Per-particle RT stage, primal only: aux profiles -> loglik value."""
        aux, theta = args
        r0 = theta[lnR0_idx] if lnR0_idx is not None else jnp.asarray(0.0, dtype)
        cp = theta[cloud_lo:cloud_lo + n_cloud] if have_cloud else jnp.zeros((0,), dtype)
        depth = _rt_wrap(aux, r0, cp)
        mu = _mu_from_depth(depth, theta)
        val = _gauss_loglik(mu, theta)
        finite = jnp.all(jnp.isfinite(depth))
        return jnp.where(finite, val, jnp.asarray(-1.0e30, dtype))

    def _rt_val_grad(args):
        """Per-particle RT stage WITH gradient: primal depth + ONE reverse-mode vjp.
        ``daux`` is the (n_chem_tp,)-stacked aux tangent pytree from the chemistry
        jvp lanes; contracting it against the RT cotangent gives the chem+T-P block,
        and the same vjp call yields the lnR0/cloud entries for free."""
        aux, daux, theta = args
        r0 = theta[lnR0_idx] if lnR0_idx is not None else jnp.asarray(0.0, dtype)
        cp = theta[cloud_lo:cloud_lo + n_cloud] if have_cloud else jnp.zeros((0,), dtype)
        depth, vjp_fn = jax.vjp(_rt_wrap, aux, r0, cp)
        mu = _mu_from_depth(depth, theta)
        sig = _sigma_for(theta)
        resid = pipe.obs_depth_jax - mu
        wres = resid / (sig * sig)                                   # dL/dmu
        val = _gauss_loglik(mu, theta)
        Btw = B_jax.T @ wres                                         # (n_native,)
        aux_bar, r_bar, cloud_bar = vjp_fn(Btw)
        g = jnp.zeros((n_dim,), dtype)
        g = g.at[:n_chem_tp].set(jax.vmap(lambda d: _tree_dot(d, aux_bar))(daux))
        if lnR0_idx is not None:
            g = g.at[lnR0_idx].set(r_bar)
        if have_cloud:
            g = g.at[cloud_lo:cloud_lo + n_cloud].set(cloud_bar)
        if n_off > 0:
            g = g.at[off_lo:off_lo + n_off].set(OBS.OFFSET_UNIT * (O_jax.T @ wres))
        if noise_idx is not None:
            k = theta[noise_idx]
            sig0 = pipe.obs_sigma_jax
            g = g.at[noise_idx].set(
                jnp.sum(resid * resid / (sig0 * sig0 * k ** 3)) - mu.size / k)
        finite = jnp.all(jnp.isfinite(depth))
        # A non-finite DEPTH is a rejected proposal (-1e30 sentinel -> -inf MH accept;
        # its gradient is then irrelevant and zeroed only to keep arithmetic clean).
        # A finite depth with a NON-FINITE GRADIENT is an AD pathology: flag it so the
        # host driver raises loudly (project rule: no silent gradient-free fallback).
        bad_grad = finite & ~jnp.all(jnp.isfinite(g))
        val = jnp.where(finite, val, jnp.asarray(-1.0e30, dtype))
        g = jnp.where(finite & jnp.isfinite(g), g, jnp.zeros_like(g))
        return val, g, bad_grad

    def _make_batch_eval(mode: str, want_grad: bool, diag: bool = False,
                         want_dy: bool = False, mutation_cap: bool = True):
        """Build eval(U, Y, refs) -> (L, G, Y_new, refs_new, n_bad_grad, DY, stats)
        when want_grad (``stats`` an EvalStats), else (L, Y_new, refs_new)
        [+ per-particle ConvDiag when diag]; all (N,)-batched. ``n_bad_grad``
        counts finite-likelihood/non-finite-gradient AD pathologies -- the host
        driver raises on it (loud-error rule; no silent random-walk degradation).

        ``DY`` is None unless ``want_dy``: the converged column's parameter tangents
        (N, n_chem_tp, nz, ni), read off the same jvp lanes that produce the gradient
        (zero extra compute), used by the warm_extrapolate mutation kernel to seed
        each proposal's warm solve at a first-order prediction of its own answer.
        With want_dy=False the compiled program is unchanged (None adds no outputs).

        mode="warm": each particle's chemistry re-converges by continuation from its
        carried column Y with incremental (lnZ - refs[0], c_o - refs[1]) scaling.
        mode="cold": the published solve-from-baseline (two-stage) map; Y/refs are
        still updated from the converged result so cold evals can seed warm ones.

        ``diag`` (only meaningful for mode="cold", want_grad=False -- the SMC init's
        likelihood-only phase) additionally threads each particle's ConvDiag
        through (worst-stage accept_count + stage-2 longdy/conv_normal), so the
        caller can detect a count_max-exhausted OR stall-certified
        (not-actually-converged) cold solve instead of silently carrying it into L."""
        warm = (mode == "warm")
        assert not (diag and (warm or want_grad)), "diag is cold+no-grad only"
        assert not (want_dy and not want_grad), "want_dy needs the jvp lanes (want_grad)"
        # Convergence gate for the warm grad. mutation_cap=True (the MALA proposal
        # path): the warm solver is capped at warm_count_max -- a proposal that hasn't
        # converged there is doomed, reject it instead of dragging the lockstep batch
        # to the cold count_max. mutation_cap=False (the INIT phase-2 path): phase-1
        # SURVIVORS re-certify from their own converged columns -- proven-convergent
        # states, not disposable proposals -- and a marginal survivor can need more
        # than warm_count_max steps to re-certify; run them under the cold count_max
        # (NAS job 64854: the cap gated 5/96 healthy survivors -> spurious raise).
        wcmax = (int(fwd.chem.warm_count_max) if mutation_cap
                 else int(fwd.chem.count_max))

        def _solve(c, yw, rf):
            return (fwd.chem_solve_warm(c, yw, rf[0], rf[1]) if warm
                    else fwd.chem_solve_cold(c))

        if want_grad:
            # A warm MALA proposal can continue into a non-convergent corner -- or
            # stall-certify short of a real steady state -- and return a
            # finite-but-unsettled column whose jvp/RT-vjp tangents are garbage. The
            # cold init rejects such draws BEFORE its gradient pass (phase-1 diag);
            # here the warm solve's ConvDiag rides the jvp'd chain itself -- every
            # field is part of the runner's primal carry, so reading it is FREE (an
            # earlier version ran a second primal-only while_loop just for the
            # accept count, doubling the chemistry wall time per sweep). The diag is
            # packed into one stop-gradient'd float vector to keep the jvp output
            # pytree all-float (longdy/longdydt DO carry tangents otherwise).
            # eval_batch rejects an exhausted OR non-certified proposal (-inf L, MH
            # rejection) and drops it from the gradient-health tally. The cold grad
            # path's diag slot is a constant healthy vector (never gates); init
            # phase 2 is warm but UNCAPPED (mutation_cap=False above).
            if warm and mutation_cap:
                def _solve_cd(c, yw, rf):
                    return fwd.chem_solve_warm_diag(c, yw, rf[0], rf[1])
            elif warm:
                def _solve_cd(c, yw, rf):
                    return fwd.chem_solve_warm_diag_full(c, yw, rf[0], rf[1])
            else:
                def _solve_cd(c, yw, rf):
                    return fwd.chem_solve_cold(c), None

            def _pack_cd(cd):
                # [accept_count, longdy, longdydt, count_since_new_min, conv_normal]
                if cd is None:      # cold grad map: constant healthy diag
                    return jnp.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype)
                return jax.lax.stop_gradient(jnp.stack([
                    jnp.asarray(cd.accept_count, dtype),
                    jnp.asarray(cd.longdy, dtype),
                    jnp.asarray(cd.longdydt, dtype),
                    jnp.asarray(cd.count_since_new_min, dtype),
                    jnp.asarray(cd.conv_normal, dtype)]))

            def _chem_one(cc, yw, rf):
                def _chain(c):
                    y, cd = _solve_cd(c, yw, rf)
                    return fwd.aux_from_y(y, c), y, _pack_cd(cd)
                (aux_l, y_l, cd_l), (daux_l, dy_l, _dcd) = jax.vmap(
                    lambda v: jax.jvp(_chain, (cc,), (v,)))(eye_c)
                aux = jax.tree_util.tree_map(lambda x: x[0], aux_l)  # primal (lane 0)
                if want_dy:
                    # dy_l[k] = d(converged column)/d(theta_chem[k]) -- the tangents
                    # relax through the same warm while_loop as the primal
                    return aux, daux_l, y_l[0], cd_l[0], dy_l
                return aux, daux_l, y_l[0], cd_l[0]
        elif diag:
            def _chem_one(cc, yw, rf):
                y, cd = fwd.chem_solve_cold_diag(cc)
                return fwd.aux_from_y(y, cc), y, cd
        else:
            def _chem_one(cc, yw, rf):
                y = _solve(cc, yw, rf)
                return fwd.aux_from_y(y, cc), y

        def eval_batch(U, Y, refs):
            U = jnp.asarray(U, dtype)
            Theta = jax.vmap(theta_from_u)(U)                        # (N, n_dim)
            _, dTh = jax.vmap(
                lambda u: jax.jvp(theta_from_u, (u,), (jnp.ones_like(u),)))(U)
            C_ = Theta[:, :n_chem_tp]
            # per-particle T-P window mask (no clip): an out-of-window proposal is
            # rejected (-inf L, state pinned to baseline) and is NOT flagged as an AD
            # pathology (its gradient is irrelevant once MH rejects it).
            valid = (jax.vmap(tp_valid)(Theta) if n_tp > 0
                     else jnp.ones((Theta.shape[0],), bool))
            usable = valid   # narrowed to (valid & certified) on the warm gradient path
            if want_grad:
                # Chemistry jvp lanes, optionally lax.map-chunked over particles.
                # Probe 2026-07-07: staged chem tangent lanes cost ~20 MB per
                # lane-pair (0.78 GiB at 36 lanes), so full width (chem_chunk=0)
                # is the default -- the old ~1.3 GB/lane figure was the all-in-one
                # architecture's PreMODIT tangents (the 390 GiB OOM), misattributed
                # to photo temporaries. The RT VJP below is the real memory wall
                # (18.4 GiB first lane, ~9.4 GiB per additional at nu_pts=5000).
                if want_dy:
                    AUX, DAUX, Ynew, CD, DY = _map_chunked(lambda a: _chem_one(*a),
                                                           (C_, Y, refs), chem_chunk)
                else:
                    AUX, DAUX, Ynew, CD = _map_chunked(lambda a: _chem_one(*a),
                                                       (C_, Y, refs), chem_chunk)
                    DY = None
                vals, g_th, bads = _map_chunked(_rt_val_grad, (AUX, DAUX, Theta),
                                                rt_vjp_chunk)
                G = g_th * dTh                                       # chain to u-space
                # Two REJECTION classes (MH rejections, not AD pathologies) whose
                # (garbage) gradients must NOT trip n_bad_grad:
                #   capped  -- the warm solve hit the cap (warm_count_max on the
                #              mutation path, count_max on init phase 2);
                #   stalled -- under the cap, but the exit was NOT the runner's
                #              canonical certification (stall fallback / budget
                #              exit): the primal may look settled while the jvp
                #              tangent -- which relaxes through the same
                #              while_loop with no stopping criterion of its own --
                #              has not (NAS job 65200's 16/864 bad gradients).
                # The cold grad path's diag is a constant healthy vector, so both
                # classes are empty there and nothing changes.
                ACC = CD[:, 0].astype(jnp.int32)
                conv_ok = _proposal_converged(CD)
                under_cap = ACC < wcmax
                usable = valid & under_cap & conv_ok
                n_bad = jnp.sum((bads & usable).astype(jnp.int32))
                # both classes broken out from the generic reject count: the MH
                # correction only knows the Langevin proposal density, so a
                # rejection class that binds often (and possibly state-dependently)
                # is a detailed-balance risk -- it must be VISIBLE per sweep/stage,
                # not folded into "rejected". See validate_warm/reversibility notes.
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
                AUX, Ynew, CDIAG = jax.vmap(_chem_one)(C_, Y, refs)
                vals = _map_chunked(_rt_val, (AUX, Theta), rt_chunk)
                G = None
            else:
                AUX, Ynew = jax.vmap(_chem_one)(C_, Y, refs)
                vals = _map_chunked(_rt_val, (AUX, Theta), rt_chunk)
                G = None
            L = jnp.where(jnp.isfinite(vals) & usable, vals, jnp.asarray(-1.0e30, dtype))
            # a blown (-1e30, rejected/culled) solve -- non-finite forward, an
            # out-of-window T-P, OR a non-converged warm proposal -- must not poison the
            # carried state arithmetic: pin it to the baseline column. This is part of the
            # MH rejection mechanics, not an error path -- true failures surface through
            # the -1e30 likelihood (init raises) or n_bad_grad (host raises).
            ok = jnp.all(jnp.isfinite(Ynew), axis=(1, 2)) & usable
            Ynew = jnp.where(ok[:, None, None], Ynew, y_baseline[None])
            refs_new = jnp.where(ok[:, None], C_[:, :2], jnp.zeros_like(refs))
            if want_grad:
                if want_dy:
                    # NaN hygiene only: a pinned/rejected proposal's tangents may be
                    # garbage, and though MH can never accept it into the carried
                    # state (-1e30 L), zeroed is strictly safer than untouched
                    DY = jnp.where(ok[:, None, None, None], DY, jnp.zeros_like(DY))
                # uniform tail: EvalStats for BOTH the mutation-capped and the init
                # (uncapped) variants -- _init_state reads .acc/.conv_ok to tell a
                # re-certification failure (cull) from a true RT/AD blow-up (raise);
                # the mutation kernel reads .n_capped/.n_stalled and the per-particle
                # vectors for the bad-gradient forensics dump.
                return L, G, Ynew, refs_new, n_bad, DY, stats
            if diag:
                return L, Ynew, refs_new, CDIAG
            return L, Ynew, refs_new

        return eval_batch

    # ---- assemble ----
    pipe.__dict__.update(dict(
        cfg=cfg, dtype=dtype, npdtype=npdtype,
        fwd=fwd, obs=obs, real_bins=real_bins, groups=groups,
        B=B, O=O, n_bin=n_bin,
        specs=specs, names=names, kinds=kinds, labels=labels, n_dim=n_dim,
        n_chem_tp=n_chem_tp, lnR0_idx=lnR0_idx, off_idx=off_idx, noise_idx=noise_idx,
        cloud_idx=cloud_idx, n_cloud=n_cloud,
        param_prior_lo=np.asarray([s.lo for s in specs], npdtype),
        param_prior_hi=np.asarray([s.hi for s in specs], npdtype),
        param_truth=param_truth, prior_types=prior_types,
        # sample_prior_u is the T-P-window-restricted (redraw) sampler; the raw box
        # sampler + the validity predicate are exposed for diagnostics/calibration.
        theta_from_u=theta_from_u, log_prior_u=log_prior_u, sample_prior_u=sample_prior_u_valid,
        sample_prior_u_box=sample_prior_u, tp_valid=tp_valid, n_tp=n_tp,
        tp_prior_stats=tp_prior_stats,
        theta_truth=theta_truth,
        observed_depth_model=observed_depth_model, observed_depth_model_jit=observed_depth_model_jit,
        log_likelihood_u=log_likelihood_u, loglik_fwd=loglik_fwd, use_custom_grads=use_custom,
        gradient_mode=grad_mode,
        value_and_grad_naive=_value_and_grad_naive, value_and_grad_block=_value_and_grad_block,
        # staged batched evaluators (the SMC hot path)
        has_chem_state=True, chem_mode=chem_mode, y_baseline=y_baseline,
        warm_extrapolate=bool(cfg.warm_extrapolate) and chem_mode == "warm",
        batch_eval_cold_vg=_make_batch_eval("cold", True),
        batch_eval_cold_l=_make_batch_eval("cold", False),
        batch_eval_cold_l_diag=_make_batch_eval("cold", False, diag=True),
        batch_eval_move_vg=_make_batch_eval(
            chem_mode, True,
            want_dy=bool(cfg.warm_extrapolate) and chem_mode == "warm"),
        # init phase 2: same evaluator WITHOUT the mutation cap (survivors re-certify
        # under the cold count_max; see _make_batch_eval's mutation_cap note)
        batch_eval_init_vg=_make_batch_eval(
            chem_mode, True,
            want_dy=bool(cfg.warm_extrapolate) and chem_mode == "warm",
            mutation_cap=False),
        batch_eval_move_l=_make_batch_eval(chem_mode, False),
        # observations injected by set_observations
        obs_depth_jax=None, obs_sigma_jax=None, obs_depth=None, obs_sigma=None, flux_true=None,
    ))

    def set_observations(depth, sigma):
        # Observations are baked in as trace-time constants at the first jitted
        # likelihood call, so a second (post-compile) swap would silently keep the
        # old data. Enforce the documented call-once contract loudly instead of
        # letting a stale-likelihood run through (standing fail-fast rule).
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

    pipe.set_observations = set_observations
    return pipe


# =============================================================================
# Observations
# =============================================================================
def evidence_report(logZ: float, init_stats: dict | None) -> dict:
    """Evidence-semantics fields from the SMC ``logZ`` and the init cull
    counters. Module-level and jax-free so the semantics are unit-testable
    (tests/test_evidence_semantics.py), like validate_observations.

    ``logZ`` is the evidence under the OPERATIONAL prior (declared box
    conditioned on the T-P window A and chemistry convergence C,
    renormalized): Z_oper = E_pi[L | A and C]. Returned fields:

      log_support_physical (+err)   ln f_tp -- the T-P-window prior mass, a
                                    solver-INDEPENDENT domain restriction;
      log_conv_attrition (+err)     ln(f_c1 f_c2) -- convergence success,
                                    solver-DEPENDENT (count_max, tolerances);
      log_support_fraction (+err)   their sum;
      logZ_box                      logZ + ln(f_tp f_c1 f_c2): the ZERO-FILLED
                                    box evidence, i.e. the exact integral of
                                    pi * L * 1[A and C] over the declared box
                                    (the sampler defines non-convergent draws
                                    as rejected / zero likelihood). The ONLY
                                    integral-valid box quantity here; usable
                                    for cross-model Bayes factors ONLY at
                                    matched solver settings and with the
                                    attrition shown likelihood-negligible.

    There is deliberately NO ``logZ_box_physical`` (= logZ + ln f_tp): that
    construction restores the T-P prior mass while silently keeping the
    convergence conditioning renormalized -- P(A) E[L | A and C] is neither
    the box integral over A nor the A-conditioned evidence, and a support
    fraction cannot reconstruct the unevaluated likelihood on the
    non-converged set (2026-07-12 recheck P0-B; retracted same day it
    shipped).

    UNCERTAINTY CAVEAT (2026-07-13 recheck item 4): the ``*_err`` fields are
    ONLY the binomial uncertainty of the estimated support FRACTIONS. They do
    NOT include the Monte-Carlo (seed-to-seed) uncertainty of the SMC estimate
    of ``logZ`` itself, so the reported +/- is NOT the total evidence
    uncertainty. For a Bayes factor, run several independent SMC seeds, take
    the empirical std of logZ, add it in quadrature with the support-fraction
    error, and require agreement across particle-count and tempering settings.
    A single run's support-fraction error bar underestimates sigma(logZ_box)."""
    def _binom(k, n):
        if n <= 0:
            return 1.0, 0.0
        f = max(k / n, 1.0 / (2.0 * n))          # floor so ln(f) stays finite
        se = math.sqrt(max(f * (1.0 - f), 0.0) / n) / f   # d ln f
        return f, se
    if not init_stats:
        nan = float("nan")
        return dict(log_support_fraction=nan, log_support_fraction_err=nan,
                    log_support_physical=nan, log_support_physical_err=nan,
                    log_conv_attrition=nan, log_conv_attrition_err=nan,
                    logZ_box=nan, f_tp=nan, f_conv=nan)
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

    Fail loud at the API boundary (2026-07-12 re-audit item 4): the Gaussian
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
    pipe.flux_true = np.full_like(depth, np.nan)
    return dict(depth=depth, sigma=sigma)


def generate_observations(pipe: Pipeline, seed: int) -> Dict[str, np.ndarray]:
    """Synthetic injection: model at truth, add Gaussian noise at the (real, if available)
    per-bin sigma. Injects into pipe and returns the arrays."""
    cfg = pipe.cfg
    sigma = np.asarray(pipe.obs["sigma"], pipe.npdtype)
    mu_true = np.asarray(pipe.observed_depth_model_jit(pipe.theta_truth), pipe.npdtype)
    if not np.all(np.isfinite(mu_true)):
        raise RuntimeError("truth forward is non-finite; check truth_* and priors")
    rng = np.random.default_rng(seed)
    depth = mu_true + rng.standard_normal(mu_true.shape) * sigma
    pipe.set_observations(depth, sigma)
    pipe.flux_true = mu_true
    return dict(depth=depth, sigma=sigma, flux_true=mu_true)


# =============================================================================
# SMC core (self-contained, pure JAX)
# =============================================================================
def _ess_from_incremental(L: np.ndarray, dbeta: float) -> float:
    a = dbeta * (L - L.max())
    w = np.exp(a)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return 0.0
    w = w / s
    return float(1.0 / np.sum(w * w))


def _next_dbeta(L: np.ndarray, beta: float, target_ess: float, tol: float = 1e-4) -> float:
    """Bisection for the temperature increment so ESS(exp(dbeta*L)) = target_ess.
    Returns dbeta in (0, 1-beta]; jumps to 1-beta when even the full step keeps ESS high."""
    dmax = 1.0 - beta
    if dmax <= 0:
        return 0.0
    if _ess_from_incremental(L, dmax) >= target_ess:
        return dmax
    lo, hi = 0.0, dmax
    for _ in range(60):
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


def _abs_scale_diag(particles: np.ndarray, cap: float) -> np.ndarray:
    """ABSOLUTE per-dimension std of the (resampled, uniformly-weighted) cloud.

    Used as the diagonal proposal scale: the MALA proposal then narrows in lockstep
    with the tempered posterior, so the scalar step size only fine-tunes toward the
    target acceptance instead of chasing orders of magnitude of width (the SWAMPE
    unit-geometric-mean normalization left the width entirely to the Robbins-Monro
    step, which lags the ladder and collapses acceptance after big beta jumps --
    reproduced by tests/test_smc_gaussian.py before this change)."""
    p = np.asarray(particles, np.float64)
    scale = p.std(axis=0)
    if not np.all(np.isfinite(scale)):
        return np.ones(p.shape[1])
    return np.clip(scale, 1e-3, float(cap))


def _get_batch_evals(pipe: Pipeline):
    """(cold_vg, cold_l, move_vg, move_l) batched evaluators. Gradient evaluators
    return the 7-tuple (L, G, Y_new, refs_new, n_bad, DY, stats) -- DY is None
    unless the pipeline was built with warm_extrapolate; ``stats`` is an EvalStats
    (uniform across the mutation-capped, init-uncapped, cold, and stub variants:
    per-batch n_capped/n_stalled tallies + per-particle acc/longdy/conv_ok/
    bad_grad/chem_tan_bad). Likelihood-only evaluators return (L, Y_new, refs_new).
    Real pipelines carry the staged chemistry+RT evaluators; stub pipes (unit tests,
    no chemistry) get a stateless adapter so the SMC/MALA core is exercised through
    the exact same code path."""
    if getattr(pipe, "has_chem_state", False):
        return (pipe.batch_eval_cold_vg, pipe.batch_eval_cold_l,
                pipe.batch_eval_move_vg, pipe.batch_eval_move_l)
    if not hasattr(pipe, "_stub_evals"):
        vg1 = jax.value_and_grad(pipe.loglik_fwd)

        def eval_vg(U, Y, refs):
            L, G = jax.vmap(vg1)(U)
            bad = jnp.isfinite(L) & ~jnp.all(jnp.isfinite(G), axis=1)
            # all-healthy EvalStats (stubs have no warm cap or stall class) except
            # bad_grad, which mirrors the real evaluators' AD-pathology flag --
            # keeps the 7-tuple contract uniform
            stats = _zero_eval_stats(U.shape[0], U.dtype)._replace(bad_grad=bad)
            return (L, G, Y, refs, jnp.sum(bad.astype(jnp.int32)), None, stats)

        def eval_l(U, Y, refs):
            return jax.vmap(pipe.log_likelihood_u)(U), Y, refs

        pipe._stub_evals = (eval_vg, eval_l)
    evg, el = pipe._stub_evals
    return evg, el, evg, el


def _blank_state(pipe: Pipeline, N: int):
    """(Y0, refs0) placeholders for a fresh particle cloud: the baked baseline column
    for real pipelines (matching refs (lnZ, c_o) = (0, 0)), inert zeros for stubs."""
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
    over = float(getattr(pipe.cfg, "init_oversample", 2.0))  # matches the schema default
    return max(n_target, int(math.ceil(n_target * over)))


def _init_state(pipe: Pipeline, U, target_n: Optional[int] = None):
    """Initialize the SMC particle state, returning (U_kept, L, G, Y, refs, DY) for
    exactly ``target_n`` healthy particles (default: all of U). ``DY`` is the carried
    column tangents for warm_extrapolate pipelines, else None.

    ``U`` is an OVERSAMPLED prior cloud (len(U) = ceil(target_n * init_oversample) for
    real pipelines; see _init_draw_count). The two phases:

    Phase 1 -- cold LIKELIHOOD-ONLY pass over ALL len(U) draws at full width (one primal
    lane per particle, no tangents). Draws whose chemistry doesn't converge within
    count_max, whose exit is not the runner's canonical certification (stall
    fallback / budget exit -- not a certified steady state), or whose forward is
    non-finite are REJECTED, and the first ``target_n`` survivors are kept. This is the best-practice handling of forward-model
    failures: petitRADTRANS / nested-sampling codes discard an invalid forward with -inf
    likelihood, and Herbst-Schorfheide SMC oversamples so the culled cloud still carries
    the target number of particles (ESS preserved). Non-convergence at extreme prior
    corners (hot + extreme-Kzz) is EXPECTED for a full-kinetics forward, not a bug --
    _init_state raises only if fewer than target_n survive (a systemic prior/config
    problem). Wall time is one lockstep max over the draws, count_max-bounded; widening
    the draw to oversample is ~free because the slowest draw dominates regardless.

    Phase 2 -- gradient pass on the target_n SURVIVORS ONLY (the expensive jvp/vjp
    lanes are never paid on a rejected draw): each survivor re-certifies from its own
    phase-1 column and the jvp lanes ride that warm map -- the SAME map every
    subsequent MALA proposal uses, so the carried (L, G) are consistent with the rest
    of the run by construction. Phase 2 runs UNCAPPED (batch_eval_init_vg, cold
    count_max, not warm_count_max): typical survivors re-certify in a few hundred
    steps, but a marginal one (slow phase-1 converger / stall-fallback certification)
    can need more than the mutation cap, and it is a proven-convergent particle, not a
    disposable proposal (NAS job 64854: the cap gated 5/96 healthy survivors).

    Survivors are fully converged, so phase 2 must be SOUND -- there is no MH rejection
    to absorb failures here. A non-finite likelihood or flagged gradient pathology on a
    survivor raises (loud-error rule): that is a real AD/RT problem, NOT a hard prior
    corner (those were already rejected in phase 1)."""
    M = int(U.shape[0])
    if target_n is None:
        target_n = M
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
    logger.info(f"init 1/2: batched cold two-stage chemistry over {M} draw(s) "
                f"(likelihood only; reject non-converged, keep {target_n}; wall time = "
                "the slowest draw, count_max-bounded)")
    if has_diag:
        L0, Y, refs, cd0 = pipe._init_l_jit(U, Y0, refs0)
    else:
        L0, Y, refs = pipe._init_l_jit(U, Y0, refs0)
    jax.block_until_ready(L0)

    # per-particle rejection (real pipes only): non-finite forward, count_max-
    # exhausted, OR stall-certified (the exit was not the runner's canonical
    # certification -- a state whose likelihood/tangents describe an unsettled
    # column; the class behind NAS job 65200's non-finite mutation gradients)
    L0_np = np.asarray(jax.device_get(L0), np.float64)
    nonfinite = ~np.isfinite(L0_np) | (L0_np <= -1.0e29)
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

    if n_dead:
        frac = n_dead / M
        n_ex, n_st = int(exhausted.sum()), int(stalled.sum())
        n_nf = int((nonfinite & ~exhausted).sum())
        idx_head = np.flatnonzero(dead)[:12].tolist()
        msg = (f"cold init: rejected {n_dead}/{M} draw(s) ({frac:.0%}: {n_ex} hit "
               f"count_max, {n_st} stall-certified (not a canonical steady state), "
               f"{n_nf} non-finite forward; first indices {idx_head}); "
               f"keeping {target_n} of {n_alive} survivors")
        if frac > float(pipe.cfg.init_max_nonconverged_frac):
            logger.warning(
                msg + f" -- reject fraction exceeds init_max_nonconverged_frac "
                f"({float(pipe.cfg.init_max_nonconverged_frac):.0%}); the prior reaches "
                "many non-convergent corners (hot / extreme-Kzz). Expected for a "
                "full-kinetics forward and absorbed by reject+oversample, but if it "
                "keeps climbing, tighten the prior or raise count_max.")
        else:
            logger.info(msg)

    if n_alive < target_n:
        raise RuntimeError(
            f"only {n_alive}/{M} cold draws converged; need {target_n}. The "
            "reject-and-cull ran out of survivors: raise init_oversample (currently "
            f"{float(getattr(pipe.cfg, 'init_oversample', 2.0)):g}), tighten the prior, "
            "or raise count_max. This is a systemic prior/config problem, not a few hard "
            "corners.")

    # phase 2 evaluates a few SPARE survivors beyond target_n (width is ~free in the
    # lockstep chemistry) so marginal columns that cannot RE-certify warm can be
    # culled and backfilled instead of killing the run (NAS jobs 64854/64897)
    spare = int(getattr(pipe.cfg, "init_phase2_spare", 8)) if has_diag else 0
    n_phase2 = min(n_alive, target_n + spare)
    sel = jnp.asarray(alive[:n_phase2])
    U_keep = jnp.asarray(U)[sel]
    Y, refs = Y[sel], refs[sel]
    logger.info(f"init 1/2 done in {time.perf_counter() - t0:.1f}s "
                f"({n_alive} converged; phase 2 on {n_phase2} = {target_n}"
                f"+{n_phase2 - target_n} spare)")

    # ---- phase 2: gradient on the survivors (+spares) ----
    t0 = time.perf_counter()
    logger.info("init 2/2: move-map gradient at the kept cloud (jvp lanes on warm "
                "re-certifications from each survivor's own converged column; "
                "UNCAPPED -- bounded by the cold count_max)")
    out = pipe._init_mv_jit(U_keep, Y, refs)
    jax.block_until_ready(out[0])
    L, G, Y, refs, n_bad, DY, stats2 = out
    if has_diag:      # real pipelines: EvalStats threads per-particle ACC + conv bit
        acc2_np = np.asarray(jax.device_get(stats2.acc), np.int64)
        conv2_np = np.asarray(jax.device_get(stats2.conv_ok), bool)
    else:             # stub pipelines: the zeroed-EvalStats tail carries no gating info
        acc2_np = None
    n_bad = int(jax.device_get(n_bad))
    if n_bad > 0:
        raise RuntimeError(
            f"{n_bad} SURVIVING particle(s) produced a finite likelihood but a "
            "NON-FINITE gradient at initialization -- AD pathology in the chemistry "
            "tangents or RT vjp (these already converged in phase 1, so it is not a "
            "hard corner); refusing to continue (no silent gradient-free fallback).")

    # cull re-certification failures; raise on true RT/AD deaths
    L_np = np.asarray(jax.device_get(L), np.float64)
    dead2 = ~np.isfinite(L_np) | (L_np <= -1.0e29)
    if acc2_np is not None:
        cmax2 = int(pipe.fwd.chem.count_max)
        # a dead phase-2 particle is a re-certification failure (cull + backfill)
        # if its warm solve exhausted count_max OR exited without the canonical
        # certification (stall fallback -- an unsettled state the eval gate now
        # floors to -1e30); anything else dead is a genuine RT/AD blow-up (raise)
        recert_fail = dead2 & ((acc2_np >= cmax2) | ~conv2_np)
        rt_dead = dead2 & ~recert_fail
    else:
        recert_fail, rt_dead = dead2, np.zeros_like(dead2)
    if np.any(rt_dead):
        raise RuntimeError(
            f"{int(rt_dead.sum())}/{n_phase2} phase-2 particle(s) produced a "
            f"non-finite forward on a certified, NON-exhausted warm solve (indices "
            f"{np.flatnonzero(rt_dead).tolist()}) -- a genuine RT/AD problem, not a "
            "convergence cull; refusing to start the SMC on a crippled cloud.")
    if np.any(recert_fail):
        logger.warning(
            f"init 2/2: culled {int(recert_fail.sum())}/{n_phase2} marginal "
            f"survivor(s) that certify cold but cannot RE-certify warm within "
            f"count_max -- or only stall-certify (indices "
            f"{np.flatnonzero(recert_fail).tolist()}); "
            "backfilling from spares. A repeatable class (oscillating/stall-fallback "
            "columns), part of the operational prior -- report alongside the phase-1 "
            "reject fraction.")
    alive2 = np.flatnonzero(~dead2)
    if alive2.size < target_n:
        raise RuntimeError(
            f"only {int(alive2.size)}/{n_phase2} phase-2 particles are healthy; need "
            f"{target_n}. Spares exhausted -- raise init_phase2_spare (currently "
            f"{spare}) or init_oversample, or investigate why so many survivors "
            "cannot re-certify warm.")
    sel2 = jnp.asarray(alive2[:target_n])
    U_keep, L, G, Y, refs = U_keep[sel2], L[sel2], G[sel2], Y[sel2], refs[sel2]
    if DY is not None:
        DY = DY[sel2]
    if not np.all(np.isfinite(np.asarray(jax.device_get(G)))):
        raise RuntimeError("non-finite gradient entries at initialization")
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
    return U_keep, L, G, Y, refs, DY, init_stats


def _make_mutation(pipe: Pipeline, n_mcmc: int):
    """Build the state-carrying mutation:

        mutate(key, U, Y, refs, L, G, DY, beta, step, scale,
               where="mutation", dump_dir=None, dump_tag="")
            -> (U, Y, refs, L, G, DY, mean_acceptance, n_bad_grad,
                n_warm_capped, n_stalled)

    ``n_warm_capped`` totals the proposals rejected specifically because their warm
    solve hit warm_count_max; ``n_stalled`` those rejected because the solve exited
    under the cap WITHOUT the runner's canonical certification (stall fallback /
    budget exit -- an unsettled state whose tangents cannot be trusted). Both are
    MH rejections (subsets of "rejected"), surfaced per sweep and per stage because
    a frequently-binding, possibly state-dependent rejection class is a
    detailed-balance risk the MH correction does not see -- keep both ~0 in the
    late ladder.

    Runs `n_mcmc` preconditioned-MALA sweeps over the particle cloud as a HOST
    LOOP over a single-sweep jitted kernel (RNG identical to the former
    lax.scan: the same pre-split keys, consumed in the same order). The host
    loop is what makes the run debuggable: each sweep's health is checked as it
    completes -- a bad-gradient event fails FAST at the offending sweep (not
    after the whole stage; NAS job 65200 burned 2 h of a doomed stage 0) and the
    per-particle forensics (indices, theta, accept counts, longdy, chemistry-vs-RT
    attribution) are dumped to ``dump_dir/bad_grad_<dump_tag>_sweep<j>.npz``
    before the loud raise. The per-sweep device sync costs microseconds against
    ~20-minute GPU sweeps.

    Every proposal's chemistry warm-starts from the particle's carried converged
    column Y (continuation refs = the (lnZ, c_o) that column was converged at), so
    a sweep costs ~count_min chemistry steps instead of a full cold two-stage
    solve -- and the whole cloud's chemistry runs as ONE wide batched while_loop,
    with only the memory-heavy RT lax.map-chunked. The warm solve is
    warm_count_max-capped: a proposal in a non-convergent corner is cut off and
    rejected there instead of dragging the whole lockstep batch to the cold
    count_max (the early-ladder wall-clock killer diagnosed on job 64745).

    ``DY`` is None unless the pipeline was built with ``warm_extrapolate``; then it
    carries each particle's converged-column tangents d y*/d theta_chem, and each
    proposal's warm solve is seeded at the first-order prediction
    Y + DY·(theta_new - theta_cur) instead of at Y itself (measured ~1.65x fewer
    warm steps on MALA-sized moves). The seed's refs are set to the PROPOSAL's
    (lnZ, c_o): the extrapolated column already carries the predicted composition
    shift, so the solver's own refs-rescale must become a no-op (double-scaling
    otherwise). Both seeds relax to the same certified steady state; the
    extrapolation changes wall time, not the target.

    L and G are the raw log-likelihood and its u-space gradient; the tempered
    log-density and its gradient are assembled per sweep from the analytic prior
    (d/du log_prior_u = 1 - 2*sigmoid(u)), so carried state stays beta-independent
    and survives tempering-ladder moves and resampling untouched.

    A finite-likelihood/non-finite-gradient AD pathology raises INSIDE mutate at
    the offending sweep (loud-error rule -- a MALA that silently loses its
    gradient is a different sampler); callers need no separate health check."""
    log_prior_u = pipe.log_prior_u
    _, _, move_vg, _ = _get_batch_evals(pipe)
    extrap = bool(getattr(pipe, "warm_extrapolate", False))
    theta_from_u = pipe.theta_from_u
    n_ct = int(getattr(pipe, "n_chem_tp", 0))

    def sweep(k, U, Y, refs, L, G, DY, beta, step, scale):
        def dlogprior(U_):
            return 1.0 - 2.0 * jax.nn.sigmoid(U_)

        kp, ka = jax.random.split(k)
        noise = jax.random.normal(kp, U.shape, dtype=U.dtype)
        GT = dlogprior(U) + beta * G
        U_new = U + step * (scale * scale) * GT + jnp.sqrt(2.0 * step) * scale * noise
        theta_new = jax.vmap(theta_from_u)(U_new)   # forensics; negligible next to the solves
        if extrap:
            # first-order warm-start extrapolation: seed the proposal's solve at
            # the predicted converged column; refs = the PROPOSAL's (lnZ, c_o) so
            # the solver's refs-rescale is a no-op (no double-scaling)
            C_cur = jax.vmap(theta_from_u)(U)[:, :n_ct]
            C_new = theta_new[:, :n_ct]
            Y_seed = jnp.maximum(
                Y + jnp.einsum("nkij,nk->nij", DY, C_new - C_cur), 0.0)
            L_new, G_new, Y_new, refs_new, n_bad, DY_new, stats = move_vg(
                U_new, Y_seed, C_new[:, :2])
        else:
            L_new, G_new, Y_new, refs_new, n_bad, DY_new, stats = move_vg(
                U_new, Y, refs)
        GT_new = dlogprior(U_new) + beta * G_new
        # asymmetric MH correction for the preconditioned Langevin proposal
        df = (U_new - U - step * (scale * scale) * GT) / scale
        dr = (U - U_new - step * (scale * scale) * GT_new) / scale
        log_q_fwd = -0.25 / step * jnp.sum(df * df, axis=1)
        log_q_rev = -0.25 / step * jnp.sum(dr * dr, axis=1)
        LP = jax.vmap(log_prior_u)(U) + beta * L
        LP_new = jax.vmap(log_prior_u)(U_new) + beta * L_new
        log_acc = LP_new - LP + log_q_rev - log_q_fwd
        log_acc = jnp.where(jnp.isfinite(log_acc), log_acc, -jnp.inf)
        accept = jnp.log(jax.random.uniform(ka, (U.shape[0],), dtype=U.dtype)) < log_acc
        U = jnp.where(accept[:, None], U_new, U)
        Y = jnp.where(accept[:, None, None], Y_new, Y)
        refs = jnp.where(accept[:, None], refs_new, refs)
        L = jnp.where(accept, L_new, L)
        G = jnp.where(accept[:, None], G_new, G)
        if extrap:
            DY = jnp.where(accept[:, None, None, None], DY_new, DY)
        acc = jnp.minimum(jnp.exp(jnp.minimum(log_acc, 0.0)), 1.0)
        n_rej = jnp.sum((L_new <= -1.0e29).astype(jnp.int32))
        return (U, Y, refs, L, G, DY, jnp.mean(acc), n_rej, n_bad, stats,
                theta_new, L_new)

    sweep_jit = jax.jit(sweep)

    def mutate(key, U, Y, refs, L, G, DY, beta, step, scale,
               where: str = "mutation", dump_dir=None, dump_tag: str = ""):
        keys = jax.random.split(key, n_mcmc)   # same stream the lax.scan consumed
        n_prop = int(U.shape[0])
        accs: List[float] = []
        n_bad_tot = n_cap_tot = n_stall_tot = 0
        for j in range(n_mcmc):
            (U, Y, refs, L, G, DY, acc, n_rej, n_bad, stats, theta_new,
             L_new) = sweep_jit(keys[j], U, Y, refs, L, G, DY, beta, step, scale)
            n_bad_j = int(jax.device_get(n_bad))
            n_cap_j = int(jax.device_get(stats.n_capped))
            n_stall_j = int(jax.device_get(stats.n_stalled))
            acc_j = float(jax.device_get(acc))
            # warmcap/stalled = state-dependent rejection classes the MH correction
            # cannot see -- both must stay near zero in the converged-ladder stages.
            logger.info(f"    sweep {j + 1}/{n_mcmc}: accept={acc_j:.2f} "
                        f"rejected={int(jax.device_get(n_rej))}/{n_prop} "
                        f"warmcap={n_cap_j} stalled={n_stall_j} "
                        f"n_bad_grad={n_bad_j}")
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
                    dump_path=dump_path)
            accs.append(acc_j)
            n_bad_tot += n_bad_j
            n_cap_tot += n_cap_j
            n_stall_tot += n_stall_j
        return (U, Y, refs, L, G, DY, float(np.mean(accs)), n_bad_tot,
                n_cap_tot, n_stall_tot)

    return mutate


def _check_mutation_health(n_bad, where: str, forensics: Optional[Dict[str, Any]] = None,
                           dump_path: Optional[Path] = None) -> None:
    """Raise loudly on flagged gradient pathologies from a mutation call.

    ``forensics`` (per-particle device arrays: bad_grad, chem_tan_bad, acc, longdy,
    conv_ok, theta_proposal, loglik_proposal) is dumped to ``dump_path`` (npz) and
    summarized in the exception message BEFORE raising. The raise itself is
    non-negotiable (loud-error rule: zeroing bad gradients would silently degrade
    MALA to a random walk) -- but a multi-hour GPU run must not die without
    recording WHICH proposals failed and WHERE (chemistry tangent vs RT vjp);
    NAS job 65200 left nothing to debug with."""
    n_bad = int(jax.device_get(n_bad))
    if n_bad == 0:
        return
    detail = ""
    if forensics is not None:
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
    raise RuntimeError(
        f"{n_bad} finite-likelihood/non-finite-gradient event(s) during {where} "
        "-- AD pathology in the chemistry tangents or RT vjp. Refusing to "
        "continue: zeroing these would silently degrade MALA to a random walk "
        "(project rule: loud errors, no silent fallbacks)." + detail)


def tune_step_size(pipe: Pipeline, key) -> float:
    """One-shot Robbins-Monro pilot at a low beta (unpreconditioned)."""
    cfg = pipe.cfg
    if not bool(cfg.mcmc_auto_tune):
        return float(cfg.mala_step_size)
    dtype = pipe.dtype
    n_p = int(cfg.mcmc_tune_particles)
    beta = jnp.asarray(float(cfg.mcmc_tune_beta), dtype=dtype)
    scale = jnp.ones((pipe.n_dim,), dtype=dtype)
    key, sub = jax.random.split(key)
    U = pipe.sample_prior_u(sub, _init_draw_count(pipe, n_p))
    U, L, G, Y, refs, DY, _init_stats = _init_state(pipe, U, target_n=n_p)
    mutate = _make_mutation(pipe, int(cfg.mcmc_tune_steps))
    log_step = math.log(min(max(float(cfg.mala_step_size), cfg.mcmc_step_size_min), cfg.mcmc_step_size_max))
    target = float(cfg.mcmc_target_accept_mala)
    for it in range(int(cfg.mcmc_tune_iters)):
        key, sub = jax.random.split(key)
        # a bad-gradient event raises INSIDE mutate (per sweep, with forensics)
        U, Y, refs, L, G, DY, acc, _nbad, _ncap, _nstall = mutate(
            sub, U, Y, refs, L, G, DY, beta,
            jnp.asarray(math.exp(log_step), dtype), scale,
            where=f"step-size tuning iteration {it}")
        acc_f = float(acc)
        log_step += float(cfg.mcmc_tune_gain) * (acc_f - target)
        log_step = math.log(min(max(math.exp(log_step), cfg.mcmc_step_size_min), cfg.mcmc_step_size_max))
    tuned = float(math.exp(log_step))
    logger.info(f"Auto-tuned MALA step (u-space): {tuned:.4g} (target_accept={target:.2f})")
    return tuned


def _write_checkpoint(checkpoint_path, pipe: Pipeline, *, U, Y, refs, L, G, DY,
                      betas, ess_hist, acc_hist, logz_inc_hist, step_hist,
                      uniq_hist, capped_hist, stalled_hist, scale, last_step,
                      logZ, init_stats) -> None:
    """Atomically write the SMC checkpoint (single writer for the init-level and
    per-stage checkpoints, so their schemas stay in lockstep by construction).
    ``last_step=-1`` marks the INIT-LEVEL checkpoint (written right after
    _init_state, before any tempering stage): betas=[0.0] and empty histories,
    so the resume path enters the ladder at stage 0 exactly like a fresh
    post-init run -- a stage-0 death no longer throws away the hours-scale
    two-phase init (NAS job 65200)."""
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
             scale_diag=np.asarray(scale),
             last_step=np.asarray(int(last_step), np.int64),
             init_checkpoint=np.asarray(1 if int(last_step) < 0 else 0, np.int64),
             logZ=np.asarray(logZ),
             **({"init_stats_keys": np.asarray(list(init_stats.keys())),
                 "init_stats_vals": np.asarray(list(init_stats.values()), np.int64)}
                if init_stats else {}),
             # carried per-particle state: resume warm-continues without re-init
             y_state=np.asarray(jax.device_get(Y), np.float64),
             chem_refs=np.asarray(jax.device_get(refs), np.float64),
             loglik=np.asarray(jax.device_get(L), np.float64),
             grad_u=np.asarray(jax.device_get(G), np.float64),
             **({"y_tangents": np.asarray(jax.device_get(DY), np.float64)}
                if DY is not None else {}))
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

    EVIDENCE SEMANTICS (2026-07-12 recheck P0-B -- this REPLACES the retracted
    ``logZ_box_physical``): the returned ``logZ`` is the evidence under the
    OPERATIONAL prior -- the declared box restricted to the modelable T-P window
    (A) and to draws whose chemistry converges (C), renormalized:
    Z_oper = E_pi[L | A and C]. Because the sampler DEFINES a non-convergent
    draw as rejected (zero likelihood), the one integral-valid box quantity is
    the ZERO-FILLED evidence

        logZ_box = logZ + ln(f_tp) + ln(f_c1) + ln(f_c2)
                 = ln( integral_box pi(theta) L(theta) 1[A and C](theta) dtheta ),

    which is SOLVER-DEPENDENT through the convergence indicator (count_max,
    warm_count_max, tolerances, the certification gate -- the canonical
    conv_normal predicate in _proposal_converged, which since 2026-07-15 also
    rejects stall-certified exits -- and init history all move C). Cross-model
    Bayes factors from logZ_box are defensible ONLY when (a) every model is run
    at matched solver settings (including the same certification-gate
    predicate) AND (b) the convergence attrition is shown to be
    likelihood-negligible (f_c near 1, or the rejected region demonstrated to
    carry negligible posterior mass); report both with any comparison. The old
    ``logZ_box_physical = logZ + ln(f_tp)`` is GONE: restoring the T-P prior
    mass while silently keeping the convergence conditioning renormalized is
    P(A) * E[L | A and C] -- neither the box integral over A (needs L on the
    non-converged set) nor the A-conditioned evidence (same reason); a support
    fraction cannot reconstruct an unevaluated likelihood (see
    tests/test_evidence_semantics.py for the numeric counterexample). Never
    difference bare ``logZ`` across models with different support fractions."""
    cfg = pipe.cfg
    dtype = pipe.dtype
    N = int(cfg.smc_num_particles)
    n_dim = pipe.n_dim
    target_ess = float(cfg.smc_target_ess_frac) * N
    t_start = time.perf_counter()

    key, sub = jax.random.split(key)
    # oversampled cold-init draw: _init_state rejects the non-converged corners and
    # culls back to N healthy particles (resume overwrites U from the checkpoint below)
    U = pipe.sample_prior_u(sub, _init_draw_count(pipe, N))

    # fold_in derives an independent stream for the pilot tuner: passing `key` itself
    # would replay the tuner's splits in the main loop (resample/mutation reuse)
    step = (tune_step_size(pipe, jax.random.fold_in(key, 1))
            if (cfg.mcmc_auto_tune and not cfg.mcmc_stage_adapt) else float(cfg.mala_step_size))
    log_step = math.log(min(max(step, cfg.mcmc_step_size_min), cfg.mcmc_step_size_max))
    scale = np.ones(n_dim)
    mutate = _make_mutation(pipe, int(cfg.smc_num_mcmc_steps))

    beta = 0.0
    betas: List[float] = [0.0]
    ess_hist, acc_hist, logz_inc_hist, step_hist, uniq_hist = [], [], [], [], []
    capped_hist: List[int] = []
    stalled_hist: List[int] = []
    logZ = 0.0
    init_stats: Optional[Dict[str, int]] = None

    state_loaded = False
    if resume_from is not None and Path(resume_from).exists():
        ck = np.load(resume_from)
        if ck["u_particles"].shape != (N, n_dim):
            raise ValueError(f"checkpoint particles {ck['u_particles'].shape} != ({N},{n_dim}); "
                             "resume requires the same smc_num_particles and parameter set")
        U = jnp.asarray(ck["u_particles"], dtype)
        betas = [float(b) for b in ck["betas"]]
        beta = betas[-1]
        ess_hist = [float(x) for x in ck["ess"]]
        acc_hist = [float(x) for x in ck["acceptance_rate"]]
        logz_inc_hist = [float(x) for x in ck["logZ_increment"]]
        step_hist = [float(x) for x in ck["step_size_history"]]
        uniq_hist = [int(x) for x in ck["unique_particles"]]
        logZ = float(ck["logZ"])
        scale = np.asarray(ck["scale_diag"], np.float64)
        if "warm_capped" in ck.files:
            capped_hist = [int(x) for x in ck["warm_capped"]]
        if "warm_stalled" in ck.files:
            stalled_hist = [int(x) for x in ck["warm_stalled"]]
        if "init_stats_keys" in ck.files:
            init_stats = {str(k): int(v) for k, v in
                          zip(ck["init_stats_keys"], ck["init_stats_vals"])}
        if step_hist:
            log_step = math.log(min(max(step_hist[-1], cfg.mcmc_step_size_min), cfg.mcmc_step_size_max))
        if all(k in ck.files for k in ("y_state", "chem_refs", "loglik", "grad_u")):
            Y = jnp.asarray(ck["y_state"], dtype)
            refs = jnp.asarray(ck["chem_refs"], dtype)
            L = jnp.asarray(ck["loglik"], dtype)
            G = jnp.asarray(ck["grad_u"], dtype)
            if getattr(pipe, "warm_extrapolate", False):
                if "y_tangents" not in ck.files:
                    raise ValueError(
                        "warm_extrapolate=True but the checkpoint carries no "
                        "y_tangents (it was written with extrapolation off). Resume "
                        "with warm_extrapolate=false, or start a fresh run.")
                DY = jnp.asarray(ck["y_tangents"], dtype)
            else:
                DY = None
            state_loaded = True
        else:
            logger.warning("checkpoint predates the carried chemistry state; "
                           "cold re-initializing at the resumed cloud (warm history "
                           "is NOT recovered -- likelihoods re-anchor to the cold map)")
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
        U, L, G, Y, refs, DY, init_stats = _init_state(pipe, U, target_n=N)
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
            _write_checkpoint(checkpoint_path, pipe, U=U, Y=Y, refs=refs, L=L, G=G,
                              DY=DY, betas=betas, ess_hist=ess_hist,
                              acc_hist=acc_hist, logz_inc_hist=logz_inc_hist,
                              step_hist=step_hist, uniq_hist=uniq_hist,
                              capped_hist=capped_hist, stalled_hist=stalled_hist,
                              scale=scale, last_step=-1, logZ=logZ,
                              init_stats=init_stats)
            logger.info(f"init-level checkpoint written to {checkpoint_path} "
                        "(RESUME=1 now recovers the init on a stage-0 death)")

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
        # (1) carried likelihood at current particles -> (2) next temperature via ESS
        # bisection (L travels with the particles; nothing is re-evaluated here)
        L_np = np.asarray(jax.device_get(L), np.float64)
        if not np.all(np.isfinite(L_np)):
            # rejected particles are floored at -1e30 inside eval_batch, so a
            # non-finite CARRIED likelihood is an invariant violation -- raise, never
            # normalize it away (loud-error rule)
            raise FloatingPointError(
                f"non-finite carried log-likelihood at SMC stage {i} "
                f"({int(np.sum(~np.isfinite(L_np)))}/{N} particles)")
        dbeta = _next_dbeta(L_np, beta, target_ess)
        beta_new = min(1.0, beta + dbeta)
        # (2) evidence increment + weights (uniform prior weights each stage post-resample)
        a = dbeta * (L_np - L_np.max())
        w = np.exp(a); w_sum = w.sum()   # >= 1: the max-shifted best particle is exp(0)
        logZ_inc = float(dbeta * L_np.max() + math.log(w_sum) - math.log(N))
        if not math.isfinite(logZ_inc):
            raise FloatingPointError(
                f"non-finite evidence increment at SMC stage {i} "
                f"(beta {beta:.3e} -> {beta_new:.3e}) -- refusing to corrupt logZ")
        logZ += logZ_inc
        w_norm = w / w_sum
        ess = float(1.0 / np.sum(w_norm * w_norm))
        # (3) systematic resample (the carried state travels with its particle)
        key, sub = jax.random.split(key)
        idx = _systematic_resample_idx(sub, jnp.asarray(w_norm, dtype), N)
        U, Y, refs, L, G = U[idx], Y[idx], refs[idx], L[idx], G[idx]
        if DY is not None:
            DY = DY[idx]
        # (3.5) preconditioner from the freshly RESAMPLED cloud (absolute per-dim
        # width: the proposal tracks the tempered posterior as it narrows)
        if cfg.mcmc_stage_adapt:
            scale = _abs_scale_diag(np.asarray(jax.device_get(U)), cap=float(cfg.mcmc_scale_clip))
        # (4) mutate at the new temperature -- a bad-gradient event raises INSIDE
        # mutate at the offending sweep, after dumping per-particle forensics
        # next to the checkpoint
        key, sub = jax.random.split(key)
        U, Y, refs, L, G, DY, acc, _n_bad, n_capped, n_stalled = mutate(
            sub, U, Y, refs, L, G, DY,
            jnp.asarray(beta_new, dtype),
            jnp.asarray(math.exp(log_step), dtype),
            jnp.asarray(scale, dtype),
            where=f"SMC stage {i} (beta={beta_new:.3e})",
            dump_dir=(Path(checkpoint_path).parent
                      if checkpoint_path is not None else None),
            dump_tag=f"stage{i:03d}")
        jax.block_until_ready(U)
        acc_f = float(acc)
        n_capped_f = int(n_capped)
        n_stalled_f = int(n_stalled)
        U_np = np.asarray(jax.device_get(U), np.float64)
        n_uniq = int(np.unique(np.round(U_np, 9), axis=0).shape[0])
        # (5) Robbins-Monro step-size trim toward the target acceptance (fine-tuning
        # only -- the width is carried by the absolute preconditioner above)
        if cfg.mcmc_stage_adapt and math.isfinite(acc_f):
            log_step += float(cfg.mcmc_stage_adapt_gain) * (acc_f - float(cfg.mcmc_target_accept_mala))
            log_step = math.log(min(max(math.exp(log_step), cfg.mcmc_step_size_min), cfg.mcmc_step_size_max))

        beta = beta_new
        betas.append(beta); ess_hist.append(ess); acc_hist.append(acc_f)
        logz_inc_hist.append(logZ_inc); step_hist.append(math.exp(log_step)); uniq_hist.append(n_uniq)
        capped_hist.append(n_capped_f)
        stalled_hist.append(n_stalled_f)
        elapsed = time.perf_counter() - t_start
        if hasattr(it, "set_postfix"):
            it.set_postfix(beta=f"{beta:.2e}", ess=f"{ess:.0f}", acc=f"{acc_f:.2f}")
        logger.info(f"SMC {i:03d}: beta={beta:.3e} ESS={ess:.1f}/{N} accept={acc_f:.3f} "
                    f"unique={n_uniq}/{N} step={math.exp(log_step):.3g} logZ={logZ:.2f} "
                    f"warmcap={n_capped_f} stalled={n_stalled_f} elapsed={elapsed/60:.1f}min")

        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, pipe, U=U, Y=Y, refs=refs, L=L, G=G,
                              DY=DY, betas=betas, ess_hist=ess_hist,
                              acc_hist=acc_hist, logz_inc_hist=logz_inc_hist,
                              step_hist=step_hist, uniq_hist=uniq_hist,
                              capped_hist=capped_hist, stalled_hist=stalled_hist,
                              scale=scale, last_step=i, logZ=logZ,
                              init_stats=init_stats)

        if beta >= 1.0 - 1e-8:
            break
        if walltime_seconds and elapsed > walltime_seconds:
            logger.warning(f"walltime budget {walltime_seconds/3600:.1f}h exceeded at stage {i} "
                           f"(beta={beta:.3f}); stopping cleanly with partial posterior.")
            break

    reached = beta >= 1.0 - 1e-6
    # posterior draws: at beta=1 particles are equally weighted; sample with replacement.
    # When the ladder stopped early (walltime) these are TEMPERED (beta<1) draws, NOT
    # posterior samples -- reached_beta1/final_beta travel with every output and the
    # plotting/export paths must (and do) refuse the "posterior" label without them.
    n_draws = int(cfg.num_chains) * int(cfg.num_samples)
    key, sub = jax.random.split(key)
    draw_idx = np.asarray(jax.device_get(jax.random.choice(sub, N, (n_draws,), replace=True)))
    theta_draws = np.asarray(jax.device_get(jax.vmap(pipe.theta_from_u)(U)), np.float64)[draw_idx]
    theta_draws = theta_draws.reshape(int(cfg.num_chains), int(cfg.num_samples), n_dim)

    # ---- evidence conditioning report (semantics + retraction rationale in
    # evidence_report's docstring; recheck P0-B) ------------------------------
    ev = evidence_report(logZ, init_stats)
    if init_stats:
        logger.info(
            f"evidence conditioning: logZ(conditioned/operational) = {logZ:.2f}; "
            f"ZERO-FILLED box evidence logZ_box = {ev['logZ_box']:.2f} +/- "
            f"{ev['log_support_fraction_err']:.2f} (= logZ + ln(f_tp*f_conv); "
            f"the exact integral of pi*L*1[T-P valid AND converged] over the "
            f"declared box -- SOLVER-DEPENDENT via the convergence indicator; "
            f"Bayes factors only at matched solver settings AND with the "
            f"attrition shown likelihood-negligible). Supports: T-P window "
            f"f_tp={ev['f_tp']:.3f} (solver-independent), convergence "
            f"f_conv={ev['f_conv']:.3f} (solver-dependent). There is NO "
            f"f_tp-only 'physical' evidence: that arithmetic reconstructs no "
            f"integral (recheck P0-B, retracted).")
    else:
        logger.warning("evidence conditioning: no init_stats available (old resume "
                       "checkpoint) -- the operational-prior support fraction is "
                       "unknown; do NOT quote logZ as a box-prior evidence.")

    return dict(
        U=np.asarray(jax.device_get(U), np.float64), reached_beta1=reached, final_beta=beta,
        step_size_used=math.exp(log_step), betas=np.asarray(betas),
        ess=np.asarray(ess_hist), acceptance_rate=np.asarray(acc_hist),
        logZ_increment=np.asarray(logz_inc_hist), logZ=logZ,
        # evidence-semantics fields from evidence_report (recheck P0-B):
        # logZ_box is the ZERO-FILLED box evidence; the retracted
        # logZ_box_physical is intentionally ABSENT
        log_support_fraction=ev["log_support_fraction"],
        log_support_fraction_err=ev["log_support_fraction_err"],
        logZ_box=ev["logZ_box"],
        log_support_physical=ev["log_support_physical"],
        log_support_physical_err=ev["log_support_physical_err"],
        log_conv_attrition=ev["log_conv_attrition"],
        log_conv_attrition_err=ev["log_conv_attrition_err"],
        init_stats=(init_stats or {}),
        warm_capped=np.asarray(capped_hist, np.int64),
        warm_stalled=np.asarray(stalled_hist, np.int64),
        step_size_history=np.asarray(step_hist), unique_particles=np.asarray(uniq_hist, np.int64),
        scale_diag_final=np.asarray(scale), theta_draws=theta_draws,
    )
