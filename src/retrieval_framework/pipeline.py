"""The theta-space forward as a bounded-prior u-space posterior, and its batched likelihood evaluators.

Each evaluator runs the chemistry of a whole batch in one call on the lane queue, the cold two-stage map or a warm
continuation, then the RT, ``lax.map``-chunked. Offsets and noise inflation are analytic, sampled as parameters or
integrated out of the likelihood. ``run_nautilus`` calls them through ``pipe._make_batch_eval``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from retrieval_framework import config_schema as C, observations as OBS
from retrieval_framework.nuisance import make_nuisance_marginal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from jax import Array
    from jax.typing import ArrayLike

__all__ = [
    "REJECT_BELOW",
    "REJECT_LOGL",
    "EvalStats",
    "build_pipeline",
    "generate_observations",
    "load_real_into_pipe",
    "make_noise_model",
    "make_uspace",
    "save_npz",
    "validate_observations",
]

logger = logging.getLogger("retrieval")


# A rejected draw's log-likelihood is this finite sentinel on device; run_nautilus maps it to -inf. Anything at or below
# REJECT_BELOW is a rejection.
REJECT_LOGL = -1.0e30
REJECT_BELOW = -1.0e29

# vulcan3's TERM_DT_MIN_FROZEN. A local value: the commit requirements-hpc.txt pins does not define it, and has no
# raise_if_frozen. tests/test_run_nautilus_main.py compares it with vulcan3's where vulcan3 has one.
_TERM_DT_MIN_FROZEN = 4

# sample_prior_u draws z = sigmoid(u) inside [eps, 1 - eps], so u stays finite.
_PRIOR_Z_EPS = 1e-6
# T-P window rejection sampling: the floor on candidates per round, and the cap (max(factor * n, min)) where it raises.
_TP_DRAW_MIN = 16
_TP_DRAW_CAP_FACTOR = 64
_TP_DRAW_CAP_MIN = 4096


def save_npz(path: Path, **arrays: Any) -> None:
    """Save arrays to an npz file, creating its directory.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    **arrays : numpy.ndarray
        The arrays, by name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


class EvalStats(NamedTuple):
    """Per-row solver state of a batch evaluator's exit, as device arrays.

    Attributes
    ----------
    acc : jax.Array [N], int32
        accept_count at exit.
    conv_ok : jax.Array [N], bool
        The certification bit at exit.
    reason : jax.Array [N], int32
        Why the solve stopped, a ``vulcan3.state.TERM_*`` code; a freeze in either cold stage reads as frozen.
    last_delta : jax.Array [N]
        The frozen trial's error ratio, to compare with ``rtol``.
    nuis : jax.Array [N, 2 G], optional
        The integrated nuisances' group statistics [R, S] from the RT that gave log L; None when they are sampled.
    """

    acc: jnp.ndarray
    conv_ok: jnp.ndarray
    reason: jnp.ndarray
    last_delta: jnp.ndarray
    nuis: Any = None


def _exit_state(cd: Any) -> tuple[Array, Array]:
    """Return a batch solve's (termination reason, last error ratio) per row, from its ``ConvDiag``.

    The forward that requirements-hpc.txt pins has no such fields, and the vulcan3 pinned with it has no dt_min guard: a
    column there runs to its cap instead of stopping frozen, so reason 0 (never frozen) is what is true of it.
    ``retrieval_forward`` refuses a guarded vulcan3 under a forward that cannot report it.
    """
    if "termination_reason" in cd._fields:
        return cd.termination_reason, cd.last_delta
    return jnp.zeros_like(cd.accept_count), jnp.full(cd.accept_count.shape, jnp.nan)


def _abort_if_frozen(reason: Array, last_delta: Array, z: np.ndarray, rtol: float) -> None:
    """Stop if a row of a batch's ``z`` froze at dt_min: a solver failure, not a verdict on the draw.

    Nothing can raise inside the jitted evaluator, which keeps it in JAX's persistent compile cache, so the check sits
    right after it. The message carries the first frozen row's point, since the job loses the batch's inputs.
    """
    rows = np.flatnonzero(np.asarray(jax.device_get(reason))[: len(z)] == _TERM_DT_MIN_FROZEN)
    if rows.size:
        from vulcan3 import DtMinFrozenError  # only a vulcan3 with the guard writes this reason

        ratio = np.asarray(jax.device_get(last_delta))[: len(z)][rows]
        raise DtMinFrozenError(
            f"{rows.size} of {len(z)} columns froze at dt_min, at rows {rows[:8].tolist()} of this batch: largest last "
            f"error ratio {float(np.max(ratio)):.3g} against rtol {rtol:g}. Point of row {rows[0]}: "
            f"{np.asarray(z[rows[0]]).tolist()}. How to solve such a column is open (todo.md)."
        )


def make_uspace(
    specs: Sequence[C.ParamSpec], dtype: Any
) -> tuple[Callable[[Array], Array], Callable[[Array, int], Array]]:
    """Build the map between the prior and an unconstrained u-space.

    z = sigmoid(u) in (0,1); "uniform" -> lo + (hi-lo) z, "log10_uniform" -> 10**(log lo + (log hi - log lo) z),
    "gaussian" -> lo + hi Phi^-1(z) with lo the mean and hi the sigma, so u = logit(z) of a uniform z draws theta
    exactly from its prior.

    Parameters
    ----------
    specs : sequence of config_schema.ParamSpec
        The parameters.
    dtype : numpy.dtype
        The array type.

    Returns
    -------
    tuple
        ``(theta_from_u, sample_prior_u)``.
    """
    from jax.scipy.special import ndtri

    n_dim = len(specs)
    prior_lo = jnp.asarray([s.lo for s in specs], dtype=dtype)
    prior_hi = jnp.asarray([s.hi for s in specs], dtype=dtype)
    is_log10 = jnp.asarray([1.0 if s.prior_type == "log10_uniform" else 0.0 for s in specs], dtype=dtype)
    is_gauss = jnp.asarray([1.0 if s.prior_type == "gaussian" else 0.0 for s in specs], dtype=dtype)
    lo_lin, span_lin = prior_lo, prior_hi - prior_lo
    lo_log = jnp.log10(jnp.clip(prior_lo, C.UNDERFLOW_DENOM, None))
    span_log = jnp.log10(jnp.clip(prior_hi, C.UNDERFLOW_DENOM, None)) - lo_log

    def theta_from_u(u: Array) -> Array:
        """Map a u vector to theta."""
        u = jnp.asarray(u, dtype=dtype)
        z = jax.nn.sigmoid(u)
        theta_lin = lo_lin + span_lin * z
        theta_log = 10.0 ** (lo_log + span_log * z)
        theta_gau = prior_lo + prior_hi * ndtri(z)
        return jnp.where(is_gauss > 0.5, theta_gau, jnp.where(is_log10 > 0.5, theta_log, theta_lin))

    def sample_prior_u(rng_key: Array, n_particles: int) -> Array:
        """Draw u of shape [n_particles, n_dim] from the box prior."""
        eps = jnp.asarray(_PRIOR_Z_EPS, dtype=dtype)
        z = jax.random.uniform(rng_key, (n_particles, n_dim), dtype=dtype, minval=eps, maxval=1.0 - eps)
        return jnp.log(z) - jnp.log1p(-z)

    return theta_from_u, sample_prior_u


def make_noise_model(
    names: Iterable[str], group_per_bin: ArrayLike, sigma0: Any, dtype: Any
) -> Callable[[Array], Array]:
    """Return the likelihood's per-bin sigma for the run's noise mode, through ``observations.noise_map``.

    No inflation: sigma = sigma0. One ``noise_inflation``: sigma0 b for every bin. One per group: sigma0_i b_{g(i)}.

    Parameters
    ----------
    names : sequence of str
        The parameter names.
    group_per_bin : array-like [n_bin]
        Each bin's group.
    sigma0 : callable
        Returns the observed sigma at trace time, since ``set_observations`` fills it after the build.
    dtype : numpy.dtype
        The array type.

    Returns
    -------
    callable
        ``sigma_for(theta)``, of shape [n_bin].
    """
    m = OBS.noise_map(names, group_per_bin)
    if m is None:
        return lambda _theta: sigma0()
    lo, N = m
    if N is None:
        return lambda theta: sigma0() * theta[lo]
    n = N.shape[1]
    N_j = jnp.asarray(N, dtype)
    return lambda theta: sigma0() * (N_j @ theta[lo : lo + n])


def _map_chunks(fn: Callable[[Any], Any], args: Any, chunk: int) -> Any:
    """Apply a batch function to padded chunks under ``lax.map``, to bound peak memory.

    ``chunk <= 0`` or ``chunk >= n`` is one call on the whole batch. Padding rows are dropped, so a vmapped per-particle
    ``fn`` gives the plain vmap's result at any chunk.
    """
    leaves = jax.tree_util.tree_leaves(args)
    n = int(leaves[0].shape[0])
    if chunk <= 0 or chunk >= n:
        return fn(args)
    n_pad = (-n) % chunk
    if n_pad:
        args = jax.tree_util.tree_map(lambda x: jnp.concatenate([x, x[:n_pad]], axis=0), args)
    args = jax.tree_util.tree_map(lambda x: x.reshape((-1, chunk, *x.shape[1:])), args)
    out = jax.lax.map(fn, args)
    return jax.tree_util.tree_map(lambda x: x.reshape((-1, *x.shape[2:]))[:n], out)


def _map_chunks_rows(fn: Callable[[Any], Any], args: Any, chunk: int, need: Array, fill: Any) -> Array:
    """Run ``_map_chunks`` on the rows where ``need`` holds, and ``fill`` elsewhere.

    The rows are reordered needed-first into the stacked [n_chunks, chunk] layout, and a while loop runs ``fn`` on the
    first ceil(sum(need) / chunk) chunks only, so a row nobody reads costs no RT. Its body is ``lax.map``'s, so each
    chunk is the same computation on the same shapes. ``fn`` must return one [chunk, ...] leaf.
    """
    leaves = jax.tree_util.tree_leaves(args)
    n = int(leaves[0].shape[0])
    if chunk <= 0 or chunk >= n:
        y = fn(args)
        return jnp.where(need.reshape(need.shape + (1,) * (y.ndim - 1)), y, fill)
    n_pad = (-n) % chunk
    # needed rows first, each group in row order (a stable sort of ~need)
    order = jnp.argsort(jnp.logical_not(need).astype(jnp.int32), stable=True).astype(jnp.int32)
    if n_pad:
        order = jnp.concatenate([order, order[:n_pad]])
    n_chunks = (n + n_pad) // chunk
    xs = jax.tree_util.tree_map(lambda x: x[order].reshape((n_chunks, chunk, *x.shape[1:])), args)
    n_need = jnp.sum(need.astype(jnp.int32))
    n_run = (n_need + chunk - 1) // chunk
    fill = jnp.asarray(fill)
    trail = jax.eval_shape(fn, jax.tree_util.tree_map(lambda a: a[0], xs)).shape[1:]

    def body(c: Any) -> tuple[Array, Array]:
        """Run one chunk and write it back into the stacked output."""
        k, ys = c
        x = jax.tree_util.tree_map(lambda a: jax.lax.dynamic_index_in_dim(a, k, 0, keepdims=False), xs)
        return k + 1, jax.lax.dynamic_update_index_in_dim(ys, fn(x), k, 0)

    ys0 = jnp.full((n_chunks, chunk, *trail), fill, dtype=fill.dtype)
    _, ys = jax.lax.while_loop(lambda c: c[0] < n_run, body, (jnp.int32(0), ys0))
    # stacked position p holds row order[p]; the first n_need are the needed rows
    pos = jnp.arange(n + n_pad, dtype=jnp.int32)
    dst = jnp.where(pos < n_need, order, jnp.int32(n))
    return jnp.full((n, *trail), fill, dtype=fill.dtype).at[dst].set(ys.reshape((-1, *trail)), mode="drop")


# size-limit-ok: the closures of one pipeline share its operators and its parameter layout
def build_pipeline(cfg: C.Config) -> SimpleNamespace:  # noqa: C901  # counts its closures
    """Build the forward, observation operators, u-space prior and likelihood and the batched evaluators.

    The likelihood closes over ``pipe.obs_depth_jax`` and ``pipe.obs_sigma_jax``, which the first jitted call bakes in,
    so call ``pipe.set_observations`` once, before any inference call; a second call raises.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    types.SimpleNamespace
        The pipeline: forward model, observations, parameter layout, prior maps, evaluators.

    Raises
    ------
    RuntimeError
        With fewer than two usable observed bins in the model band.
    """
    C.validate_config(cfg)
    from retrieval_framework import retrieval_forward as RF  # lazy: pulls in vulcan_chem -> VULCAN-JAX + exojax

    dtype = jnp.float64  # the import above set x64; chemistry stays f64

    t0 = time.perf_counter()
    fwd = RF.build_retrieval_forward(cfg)
    logger.info(
        "Built forward in %.1fs | native n_nu=%s nz=%s n_tp=%s molecules=%s",
        time.perf_counter() - t0,
        fwd.wl_um.size,
        cfg.nz,
        fwd.n_tp,
        list(cfg.molecules),
    )

    # A draw whose raw T-P leaves [T_min, T_max] anywhere on the ART grid or the chemistry grid (which reaches below the
    # RT bottom) is rejected, not bent into range: redrawn at the prior and given REJECT_LOGL by every evaluator. A NaN
    # profile (a T1bar draw with no Guillot solution) goes the same way.
    n_tp = int(fwd.n_tp)
    n_chem = int(fwd.n_chem)
    p_check_j = jnp.asarray(
        np.concatenate([np.asarray(fwd.p_art_bar, np.float64), np.asarray(fwd.p_chem_bar, np.float64)]), dtype=dtype
    )
    tp_eval = fwd.tp_model.eval
    tp_T_min = jnp.asarray(fwd.tp_model.T_min, dtype)
    tp_T_max = jnp.asarray(fwd.tp_model.T_max, dtype)

    def tp_valid(theta: Array) -> Array:
        """Say whether the drawn T-P lies inside [T_min, T_max] on the ART grid and the chemistry grid."""
        T_all = tp_eval(jnp.asarray(theta)[n_chem : n_chem + n_tp], p_check_j)
        return jnp.all(jnp.isfinite(T_all) & (T_all >= tp_T_min) & (T_all <= tp_T_max))

    obs, real_bins = OBS.get_observation_grid(cfg, fwd.wl_um)  # bins inside the model band
    B = OBS.build_binning_matrix(fwd.wl_um, obs)
    O_np = OBS.build_offset_design(obs)
    groups = list(obs["groups"])
    n_bin = int(B.shape[0])
    logger.info(
        "Observations: %s | %s bins | groups=%s | offset cols=%s",
        "REAL product bins" if real_bins else "synthetic grid",
        n_bin,
        groups,
        O_np.shape[1],
    )
    if n_bin < 2:
        raise RuntimeError(f"only {n_bin} usable observed bins in the model band; widen the band")

    B_jax = jnp.asarray(B, dtype=dtype)
    O_jax = jnp.asarray(O_np, dtype=dtype)

    nuisance = make_nuisance_marginal(cfg, groups, obs["group"]) if cfg.integrate_nuisance else None

    specs = C.specs_from_config(cfg, groups=groups)
    names = [s.name for s in specs]
    kinds = [s.kind for s in specs]
    labels = [s.label for s in specs]
    n_dim = len(specs)
    n_chem_tp = n_chem + fwd.n_tp
    lnR0_idx = names.index("lnR0")
    lnM_idx = names.index("lnM") if "lnM" in names else None
    s_o_idx = names.index("s_o") if "s_o" in names else None
    cloud_idx = [i for i, k in enumerate(kinds) if k == "cloud"]
    off_idx = [i for i, k in enumerate(kinds) if k == "offset"]
    off_lo = off_idx[0] if off_idx else 0
    n_off = len(off_idx)
    cloud_lo = cloud_idx[0]
    n_cloud = len(cloud_idx)

    prior_types = [s.prior_type for s in specs]
    param_truth = np.asarray([s.truth for s in specs], dtype=np.float64)

    theta_from_u, sample_prior_u = make_uspace(specs, dtype)
    theta_truth = jnp.asarray(param_truth, dtype=dtype)

    # The effective prior is the box intersected with the T-P window.
    _tp_valid_batch = jax.jit(lambda U: jax.vmap(lambda u: tp_valid(theta_from_u(u)))(U))  # noqa: PLW0108  # a fresh trace

    def sample_prior_u_valid(rng_key: Array, n_particles: int) -> Array:
        """Draw u from the box prior, redrawing any particle whose T-P leaves the window; raises at a candidate cap."""
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
                    "dominate instead of being redrawn away."
                )
        U = np.concatenate(kept, axis=0)[:n_particles]
        n_cand = drawn
        if n_cand > 2 * n_particles:
            logger.info(
                "prior T-P rejection: kept %s valid draws from %s candidates (accept frac %s); a low fraction means "
                "the T-P prior reaches unmodelable corners.",
                n_particles,
                n_cand,
                format(n_particles / n_cand, ".1%"),
            )
        return jnp.asarray(U, dtype)

    pipe = SimpleNamespace()

    def _cloud_from(theta: Array) -> Array:
        """Return the cloud block of theta."""
        return theta[cloud_lo : cloud_lo + n_cloud]

    def _lnM_from(theta: Array) -> Array:
        """Return lnM of theta, or 0."""
        return theta[lnM_idx] if lnM_idx is not None else jnp.asarray(0.0, dtype)

    def _comp_refs(C_: Array) -> Array:
        """Return the composition references (lnZ, c_o, s_o) of a chemistry block, [N, 3]; s_o is 0 if not inferred."""
        s = C_[:, s_o_idx] if s_o_idx is not None else jnp.zeros(C_.shape[0], dtype)
        return jnp.stack([C_[:, 0], C_[:, 1], s], axis=1)

    def _mu_from_depth(depth: Any, theta: Array) -> Array:
        """Return the binned model depth: B @ native depth + O @ offsets."""
        mu = B_jax @ depth
        if n_off > 0:
            mu = mu + O_jax @ (theta[off_lo : off_lo + n_off] * OBS.PPM)
        return mu

    def _binned_ok(theta: Array) -> tuple[Array, Array]:
        """Return (binned model depth, the cold certificate bit); a truth without it is refused."""
        theta = jnp.asarray(theta, dtype=dtype)
        native, _aux, ok = fwd.native_depth_aux(
            theta[:n_chem_tp], theta[lnR0_idx], _cloud_from(theta), _lnM_from(theta)
        )
        return _mu_from_depth(native, theta), ok > 0.5

    binned_ok_jit = jax.jit(_binned_ok)

    _sigma_for = make_noise_model(names, obs["group"], lambda: pipe.obs_sigma_jax, dtype)

    def _gauss_loglik(mu: Any, theta: Array) -> Array:
        """Return the finite-branch Gaussian log-likelihood of the batched evaluators."""
        sig = _sigma_for(theta)
        r = (pipe.obs_depth_jax - mu) / sig
        n = mu.size
        return (
            -0.5 * jnp.sum(r * r) - jnp.sum(jnp.log(sig)) - 0.5 * n * jnp.log(jnp.asarray(2.0 * math.pi, dtype=dtype))
        )

    _REJECT = jnp.asarray(REJECT_LOGL, dtype=dtype)

    rt_chunk = int(cfg.smc_rt_chunk)
    y_baseline = jnp.asarray(fwd.y_baseline, dtype=dtype)  # (nz, ni)

    def _rt_val(args: Any) -> Array:
        """Return one particle's log L from its profiles; with integrated nuisances [log L, R, S]."""
        aux, theta = args
        depth = fwd.rt_depth(aux, theta[lnR0_idx], _cloud_from(theta), _lnM_from(theta))
        mu = _mu_from_depth(depth, theta)
        finite = jnp.all(jnp.isfinite(depth))
        if nuisance is None:
            return jnp.where(finite, _gauss_loglik(mu, theta), _REJECT)
        st = nuisance.stats(mu, pipe.obs_depth_jax, pipe.obs_sigma_jax)
        val = jnp.where(finite, nuisance.log_marginal(st, pipe.obs_sigma_jax), _REJECT)
        return jnp.concatenate([val[None], st])

    def _mu_from_column(args: Any) -> Array:
        """Return one particle's binned depth from a converged column, ungated: a plotting interface."""
        y, theta = args
        aux = fwd.aux_from_y(y, theta[:n_chem_tp])
        return _mu_from_depth(fwd.rt_depth(aux, theta[lnR0_idx], _cloud_from(theta), _lnM_from(theta)), theta)

    observed_depth_from_y_jit = jax.jit(lambda Y, Theta: _map_chunks(jax.vmap(_mu_from_column), (Y, Theta), rt_chunk))

    def _make_batch_eval(mode: str) -> Callable[..., tuple[Array, Array, Array, EvalStats]]:
        """Build ``eval(U, Y, refs, active) -> (L, Y_new, refs_new, EvalStats)``, all batched over N.

        "warm": each particle re-converges by continuation from its carried column Y with incremental composition
        scaling, under the cold count_max. "cold": the two-stage map from the baseline; Y and refs still update, so cold
        evals can seed warm ones. A False row of ``active`` is padding: its chemistry takes no step, L is the reject
        sentinel and it gets no RT. The RT runs only on the rows whose L can survive: T-P in window, certified and
        active.
        """
        warm = mode == "warm"

        def _chem_primal(C_: Array, Y: Array, refs: Array, active: ArrayLike) -> tuple[Any, Array, Any]:
            """Return (AUX, Y_new, ConvDiag) for the whole batch, in one call on the lane queue."""
            if warm:
                Y_new, CD = fwd.chem_solve_warm_diag_full_batch(
                    C_, Y, refs[:, 0], refs[:, 1], refs[:, 2], active=active
                )
            else:
                Y_new, CD = fwd.chem_solve_cold_diag_batch(C_, active=active)
            return jax.vmap(fwd.aux_from_y)(Y_new, C_), Y_new, CD

        def eval_batch(U: Array, Y: Array, refs: Array, active: ArrayLike) -> tuple[Array, Array, Array, EvalStats]:
            """Evaluate one batch."""
            active = jnp.asarray(active, bool)
            U = jnp.asarray(U, dtype)
            Theta = jax.vmap(theta_from_u)(U)
            C_ = Theta[:, :n_chem_tp]
            valid = jax.vmap(tp_valid)(Theta)
            AUX, Ynew, CD = _chem_primal(C_, Y, refs, active)
            conv_ok = CD.conv_normal > 0.5
            # The solver's verdict alone: a certified solve can exit at cap or cap + 1, and conv_normal is False on
            # every cap exit. Only these rows get an RT; the rest hold the reject sentinel.
            usable = valid & active & conv_ok
            vals = _map_chunks_rows(jax.vmap(_rt_val), (AUX, Theta), rt_chunk, usable, _REJECT)
            nuis = None
            if nuisance is not None:
                vals, nuis = vals[:, 0], vals[:, 1:]
            ACC = CD.accept_count.astype(jnp.int32)
            L = jnp.where(jnp.isfinite(vals), vals, _REJECT)
            # A rejected solve must not poison the carried state: pin it to the baseline column.
            ok = jnp.all(jnp.isfinite(Ynew), axis=(1, 2)) & usable
            Ynew = jnp.where(ok[:, None, None], Ynew, y_baseline[None])
            comp = _comp_refs(C_)
            refs_new = jnp.where(ok[:, None], comp, jnp.zeros_like(comp))
            reason, last_delta = _exit_state(CD)
            return (
                L,
                Ynew,
                refs_new,
                EvalStats(acc=ACC, conv_ok=conv_ok, reason=reason, last_delta=last_delta, nuis=nuis),
            )

        return eval_batch

    pipe.__dict__.update(
        {
            "cfg": cfg,
            "dtype": dtype,
            "fwd": fwd,
            "obs": obs,
            "real_bins": real_bins,
            "groups": groups,
            "B": B,
            "n_bin": n_bin,
            "names": names,
            "kinds": kinds,
            "labels": labels,
            "n_dim": n_dim,
            "n_chem": n_chem,
            "n_chem_tp": n_chem_tp,
            "lnR0_idx": lnR0_idx,
            "lnM_idx": lnM_idx,
            "cloud_idx": cloud_idx,
            "n_cloud": n_cloud,
            "nuisance": nuisance,
            "param_prior_lo": np.asarray([s.lo for s in specs], np.float64),
            "param_prior_hi": np.asarray([s.hi for s in specs], np.float64),
            "param_truth": param_truth,
            "prior_types": prior_types,
            # sample_prior_u is the T-P-window-restricted (redraw) sampler; the validity
            # predicate is exposed for diagnostics/calibration.
            "theta_from_u": theta_from_u,
            "sample_prior_u": sample_prior_u_valid,
            "tp_valid": tp_valid,
            "theta_truth": theta_truth,
            "observed_depth_from_y_jit": observed_depth_from_y_jit,
            "binned_ok_jit": binned_ok_jit,
            "y_baseline": y_baseline,
            "_make_batch_eval": _make_batch_eval,
            "obs_depth_jax": None,
            "obs_sigma_jax": None,
            "obs_depth": None,
            "obs_sigma": None,
        }
    )

    def set_observations(depth: Any, sigma: ArrayLike) -> None:
        """Bake the observed depth and sigma into the pipeline, once, and fingerprint the target."""
        if pipe.obs_depth is not None:
            raise RuntimeError(
                "set_observations was already called on this pipeline. Observations "
                "are trace-time constants baked into the compiled likelihood, so "
                "swapping them in-process would silently keep the old data. Build a "
                "fresh pipeline (build_pipeline) for a different dataset."
            )
        depth, sigma = validate_observations(depth, sigma, n_bin)
        pipe.obs_depth = depth
        pipe.obs_sigma = sigma
        pipe.obs_depth_jax = jnp.asarray(depth, dtype=dtype)
        pipe.obs_sigma_jax = jnp.asarray(sigma, dtype=dtype)
        # Observations are the target's last input: fingerprint it here, so every checkpoint carries its identity.
        from retrieval_framework import provenance

        pipe.target_digest = provenance.target_digest(cfg, pipe)

    pipe.set_observations = set_observations
    return pipe


def validate_observations(depth: Any, sigma: ArrayLike, n_bin: int) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and check an injected (depth, sigma) pair for n_bin spectral bins.

    The Gaussian likelihood divides by sigma and logs it, so a bad value would poison every likelihood. Mask invalid
    bins before injection.

    Parameters
    ----------
    depth : array-like [n_bin]
        Observed depths.
    sigma : array-like [n_bin]
        Observed sigmas.
    n_bin : int
        The expected length.

    Returns
    -------
    tuple of numpy.ndarray
        The flattened (depth, sigma).

    Raises
    ------
    ValueError
        On a wrong length, a non-finite depth or a non-positive or non-finite sigma.
    """
    depth = np.asarray(depth, np.float64).reshape(-1)
    sigma = np.asarray(sigma, np.float64).reshape(-1)
    if depth.shape[0] != n_bin or sigma.shape[0] != n_bin:
        raise ValueError(f"obs depth/sigma length must be n_bin={n_bin}")
    if not np.all(np.isfinite(depth)):
        raise ValueError(
            "observed depths must all be finite; got "
            f"{int((~np.isfinite(depth)).sum())} non-finite "
            "bin(s) -- drop/mask them before set_observations"
        )
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            "observed sigmas must all be finite and strictly "
            f"positive; got {int((~np.isfinite(sigma)).sum())} "
            f"non-finite and {int((sigma <= 0.0).sum())} "
            "non-positive bin(s) -- the Gaussian likelihood "
            "divides by sigma and logs it"
        )
    return depth, sigma


def load_real_into_pipe(pipe: SimpleNamespace) -> dict[str, np.ndarray]:
    """Inject the real observed depths and sigmas already attached to ``pipe.obs``.

    Parameters
    ----------
    pipe : types.SimpleNamespace
        The pipeline.

    Returns
    -------
    dict
        ``depth`` and ``sigma``.
    """
    obs = pipe.obs
    depth = np.asarray(obs["depth"], np.float64)
    sigma = np.asarray(obs["sigma"], np.float64)
    pipe.set_observations(depth, sigma)
    return {"depth": depth, "sigma": sigma}


def generate_observations(pipe: SimpleNamespace, seed: int) -> dict[str, np.ndarray]:
    """Inject a synthetic spectrum: the model at truth plus Gaussian noise at the per-bin sigma.

    An uncertified truth column would inject a spectrum no certified draw can reproduce, so it is refused.

    Parameters
    ----------
    pipe : types.SimpleNamespace
        The pipeline.
    seed : int
        The noise seed.

    Returns
    -------
    dict
        ``depth``, ``sigma`` and ``flux_true``.

    Raises
    ------
    RuntimeError
        If the truth chemistry does not certify or the truth forward is non-finite.
    """
    sigma = np.asarray(pipe.obs["sigma"], np.float64)
    mu, ok = pipe.binned_ok_jit(pipe.theta_truth)
    mu_true = np.asarray(mu, np.float64)
    if pipe.nuisance is not None and pipe.cfg.infer_offsets:
        off = OBS.build_offset_design(pipe.obs)
        mu_true = mu_true + off @ np.full(off.shape[1], pipe.cfg.truth_offset_ppm * OBS.PPM)
    if not bool(ok):
        raise RuntimeError(
            "the truth chemistry does not certify or froze at dt_min (VULCAN's certificate under "
            f"count_max={int(pipe.fwd.chem.count_max)}): refusing the synthetic "
            "injection; move truth_* or the solver settings"
        )
    if not np.all(np.isfinite(mu_true)):
        raise RuntimeError("truth forward is non-finite; check truth_* and priors")
    rng = np.random.default_rng(seed)
    depth = mu_true + rng.standard_normal(mu_true.shape) * sigma
    pipe.set_observations(depth, sigma)
    return {"depth": depth, "sigma": sigma, "flux_true": mu_true}


def _blank_state(pipe: SimpleNamespace, N: int) -> tuple[Array, Array]:
    """Return (Y0, refs0) placeholders for a fresh batch: the baseline column, with references 0."""
    dtype = pipe.dtype
    Y0 = jnp.broadcast_to(pipe.y_baseline[None], (N, *tuple(pipe.y_baseline.shape))).astype(dtype)
    return Y0, jnp.zeros((N, 3), dtype)
