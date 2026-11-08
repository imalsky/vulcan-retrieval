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

    from retrieval_framework.retrieval_forward import RetrievalForward

__all__ = [
    "REJECT_BELOW",
    "REJECT_LOGL",
    "EvalStats",
    "Pipeline",
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

# vulcan3.state.TERM_DT_MIN_FROZEN, held here because this module imports no engine: vulcan3's first import must follow
# the network export. tests/test_run_nautilus_main.py compares the two.
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


def _abort_if_frozen(reason: Array, last_delta: Array, z: np.ndarray, rtol: float) -> None:
    """Stop if a row of a batch's ``z`` froze at dt_min: a solver failure, not a verdict on the draw.

    Nothing can raise inside the jitted evaluator, which keeps it in JAX's persistent compile cache, so the check sits
    right after it. The message carries the first frozen row's point, since the job loses the batch's inputs.
    """
    rows = np.flatnonzero(np.asarray(jax.device_get(reason))[: len(z)] == _TERM_DT_MIN_FROZEN)
    if rows.size:
        from vulcan3 import DtMinFrozenError

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


class Pipeline:
    """The forward, observation operators, u-space prior and likelihood of one run, and its batched evaluators.

    The likelihood reads ``obs_depth_jax`` and ``obs_sigma_jax`` at trace time, and the first jitted call bakes them in,
    so call ``set_observations`` once, before any inference call; a second call raises.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.
    fwd : retrieval_forward.RetrievalForward
        Its forward model.

    Attributes
    ----------
    cfg, fwd, dtype, nuisance
        The configuration, the forward, the array type and the integrated-nuisance marginal or None.
    obs, real_bins, groups, B, n_bin
        The observation grid, whether its bins are real, its groups, the binning matrix and its row count.
    names, kinds, labels, n_dim, n_chem, n_chem_tp, lnR0_idx, lnM_idx, cloud_idx, n_cloud
        The parameter layout.
    prior_types, param_prior_lo, param_prior_hi, param_truth, theta_truth, theta_from_u, theta_batch
        The priors, the truth, the map from u to theta and its jitted batch [n, n_dim].
    tp_valid_batch
        Whether each u of a batch [n, n_dim] has its T-P inside the window, jitted.
    y_baseline, observed_depth_from_y_jit, binned_ok_jit
        The column a rejected solve is pinned to, the ungated binned depth of columns, the certified one of a theta.
    obs_depth, obs_sigma, obs_depth_jax, obs_sigma_jax, target_manifest, target_digest
        The observations and the target's manifest and fingerprint: None, the last two absent, until
        ``set_observations``.

    Raises
    ------
    RuntimeError
        With fewer than two usable observed bins in the model band.
    """

    # set by set_observations, since observations are the target's last input
    target_manifest: dict[str, Any]
    target_digest: str

    def __init__(self, cfg: C.Config, fwd: RetrievalForward) -> None:
        self.cfg = cfg
        self.fwd = fwd
        self.dtype = dtype = jnp.float64  # the forward's import set x64; chemistry stays f64

        # A draw whose raw T-P leaves [T_min, T_max] anywhere on the ART grid or the chemistry grid (which reaches below
        # the RT bottom) is rejected, not bent into range: redrawn at the prior and given REJECT_LOGL by every
        # evaluator. A NaN profile (a T1bar draw with no Guillot solution) goes the same way.
        self.n_chem = int(fwd.n_chem)
        self._n_tp = int(fwd.n_tp)
        self._p_check_j = jnp.asarray(
            np.concatenate([np.asarray(fwd.p_art_bar, np.float64), np.asarray(fwd.p_chem_bar, np.float64)]), dtype=dtype
        )
        self._tp_eval = fwd.tp_model.eval
        self._tp_T_min = jnp.asarray(fwd.tp_model.T_min, dtype)
        self._tp_T_max = jnp.asarray(fwd.tp_model.T_max, dtype)

        self.obs, self.real_bins = OBS.get_observation_grid(cfg, fwd.wl_um)  # bins inside the model band
        self.B = OBS.build_binning_matrix(fwd.wl_um, self.obs)
        O_np = OBS.build_offset_design(self.obs)
        self.groups = list(self.obs["groups"])
        self.n_bin = int(self.B.shape[0])
        logger.info(
            "Observations: %s | %s bins | groups=%s | offset cols=%s",
            "REAL product bins" if self.real_bins else "synthetic grid",
            self.n_bin,
            self.groups,
            O_np.shape[1],
        )
        if self.n_bin < 2:
            raise RuntimeError(f"only {self.n_bin} usable observed bins in the model band; widen the band")

        self._B_jax = jnp.asarray(self.B, dtype=dtype)
        self._O_jax = jnp.asarray(O_np, dtype=dtype)

        # Any: consumers read it only on a run with integrate_nuisance.
        self.nuisance: Any = None
        if cfg.integrate_nuisance:
            self.nuisance = make_nuisance_marginal(cfg, self.groups, self.obs["group"])

        specs = C.specs_from_config(cfg, groups=self.groups)
        self.names = [s.name for s in specs]
        self.kinds = [s.kind for s in specs]
        self.labels = [s.label for s in specs]
        self.n_dim = len(specs)
        self.n_chem_tp = self.n_chem + fwd.n_tp
        self.lnR0_idx = self.names.index("lnR0")
        self.lnM_idx = self.names.index("lnM") if "lnM" in self.names else None
        self._s_o_idx = self.names.index("s_o") if "s_o" in self.names else None
        self.cloud_idx = [i for i, k in enumerate(self.kinds) if k == "cloud"]
        off_idx = [i for i, k in enumerate(self.kinds) if k == "offset"]
        self._off_lo = off_idx[0] if off_idx else 0
        self._n_off = len(off_idx)
        self._cloud_lo = self.cloud_idx[0]
        self.n_cloud = len(self.cloud_idx)

        self.prior_types = [s.prior_type for s in specs]
        self.param_prior_lo = np.asarray([s.lo for s in specs], np.float64)
        self.param_prior_hi = np.asarray([s.hi for s in specs], np.float64)
        self.param_truth = np.asarray([s.truth for s in specs], dtype=np.float64)

        self.theta_from_u, self._sample_box_u = make_uspace(specs, dtype)
        self.theta_truth = jnp.asarray(self.param_truth, dtype=dtype)

        # The effective prior is the box intersected with the T-P window.
        self.tp_valid_batch = jax.jit(lambda U: jax.vmap(lambda u: self.tp_valid(self.theta_from_u(u)))(U))  # noqa: PLW0108  # a fresh trace
        self.theta_batch = jax.jit(jax.vmap(self.theta_from_u))

        self.binned_ok_jit = jax.jit(self._binned_ok)

        self._sigma_for = make_noise_model(self.names, self.obs["group"], lambda: self.obs_sigma_jax, dtype)
        self._reject = jnp.asarray(REJECT_LOGL, dtype=dtype)

        self._rt_chunk = int(cfg.smc_rt_chunk)
        self.y_baseline = jnp.asarray(fwd.y_baseline, dtype=dtype)  # (nz, ni)

        self.observed_depth_from_y_jit = jax.jit(
            lambda Y, Theta: _map_chunks(jax.vmap(self._mu_from_column), (Y, Theta), self._rt_chunk)
        )
        self._solve_cold_batch = jax.jit(fwd.chem_solve_cold_diag_batch)

        self.obs_depth_jax: Any = None
        self.obs_sigma_jax: Any = None
        self.obs_depth: np.ndarray | None = None
        self.obs_sigma: np.ndarray | None = None

    def tp_valid(self, theta: Array) -> Array:
        """Say whether the drawn T-P lies inside [T_min, T_max] on the ART grid and the chemistry grid."""
        T_all = self._tp_eval(jnp.asarray(theta)[self.n_chem : self.n_chem + self._n_tp], self._p_check_j)
        return jnp.all(jnp.isfinite(T_all) & (T_all >= self._tp_T_min) & (T_all <= self._tp_T_max))

    def sample_prior_u(self, rng_key: Array, n_particles: int) -> Array:
        """Draw u from the box prior, redrawing any particle whose T-P leaves the window; raises at a candidate cap."""
        n_particles = int(n_particles)
        key = rng_key
        kept, have, drawn = [], 0, 0
        over = max(n_particles, _TP_DRAW_MIN)
        # loud cap: fail rather than loop forever
        max_draw = max(_TP_DRAW_CAP_FACTOR * n_particles, _TP_DRAW_CAP_MIN)
        while have < n_particles:
            key, sub = jax.random.split(key)
            cand = self._sample_box_u(sub, over)
            good = np.asarray(jax.device_get(self.tp_valid_batch(cand)))
            drawn += over
            if good.any():
                kept.append(np.asarray(jax.device_get(cand))[good])
                have += int(good.sum())
            if have < n_particles and drawn >= max_draw:
                raise RuntimeError(
                    f"prior T-P rejection: only {have}/{n_particles} valid draws after "
                    f"{drawn} candidates (accept frac {have / max(drawn, 1):.1%}). The "
                    "T-P prior puts most of its mass outside the modelable window "
                    f"[{float(self._tp_T_min):.0f}, {float(self._tp_T_max):.0f}] K -- tighten "
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
        return jnp.asarray(U, self.dtype)

    def rt_args(self, theta: Array) -> tuple[Array, Array, Array]:
        """Return the RT's (lnR0, cloud block, lnM) of one theta; lnM is 0 where it is not sampled."""
        lnM = theta[self.lnM_idx] if self.lnM_idx is not None else jnp.asarray(0.0, self.dtype)
        return theta[self.lnR0_idx], theta[self._cloud_lo : self._cloud_lo + self.n_cloud], lnM

    def solve_cold(self, theta: ArrayLike) -> tuple[Array, np.ndarray, np.ndarray]:
        """Re-solve rows cold and certify them as a run does; stop if a column froze at dt_min.

        Parameters
        ----------
        theta : array-like [n, n_dim]
            The rows.

        Returns
        -------
        Y : jax.Array [n, nz, ni]
            The two-stage cold columns.
        mu : numpy.ndarray [n, n_bin]
            Their binned depths, ungated.
        ok : numpy.ndarray [n]
            The certificate: conv_normal (a solve can certify at cap + 1) and a finite depth.
        """
        theta = np.asarray(theta, np.float64)
        th = jnp.asarray(theta, self.dtype)
        Y, cd = self._solve_cold_batch(th[:, : self.n_chem_tp])
        _abort_if_frozen(cd.termination_reason, cd.last_delta, theta, self.fwd.chem_rtol)
        mu = np.asarray(self.observed_depth_from_y_jit(Y, th), np.float64)
        ok = (np.asarray(cd.conv_normal) > 0.5) & np.all(np.isfinite(mu), axis=1)
        return Y, mu, ok

    def _comp_refs(self, C_: Array) -> Array:
        """Return the composition references (lnZ, c_o, s_o) of a chemistry block, [N, 3]; s_o is 0 if not inferred."""
        s = C_[:, self._s_o_idx] if self._s_o_idx is not None else jnp.zeros(C_.shape[0], self.dtype)
        return jnp.stack([C_[:, 0], C_[:, 1], s], axis=1)

    def _mu_from_depth(self, depth: Any, theta: Array) -> Array:
        """Return the binned model depth: B @ native depth + O @ offsets."""
        mu = self._B_jax @ depth
        if self._n_off > 0:
            mu = mu + self._O_jax @ (theta[self._off_lo : self._off_lo + self._n_off] * OBS.PPM)
        return mu

    def _binned_ok(self, theta: Array) -> tuple[Array, Array]:
        """Return (binned model depth, the cold certificate bit); a truth without it is refused."""
        theta = jnp.asarray(theta, dtype=self.dtype)
        native, _aux, ok = self.fwd.native_depth_aux(theta[: self.n_chem_tp], *self.rt_args(theta))
        return self._mu_from_depth(native, theta), ok > 0.5

    def _gauss_loglik(self, mu: Any, theta: Array) -> Array:
        """Return the finite-branch Gaussian log-likelihood of the batched evaluators."""
        sig = self._sigma_for(theta)
        r = (self.obs_depth_jax - mu) / sig
        n = mu.size
        return (
            -0.5 * jnp.sum(r * r)
            - jnp.sum(jnp.log(sig))
            - 0.5 * n * jnp.log(jnp.asarray(2.0 * math.pi, dtype=self.dtype))
        )

    def _rt_val(self, args: Any) -> Array:
        """Return one particle's log L from its profiles; with integrated nuisances [log L, R, S]."""
        aux, theta = args
        depth = self.fwd.rt_depth(aux, *self.rt_args(theta))
        mu = self._mu_from_depth(depth, theta)
        finite = jnp.all(jnp.isfinite(depth))
        if self.nuisance is None:
            return jnp.where(finite, self._gauss_loglik(mu, theta), self._reject)
        st = self.nuisance.stats(mu, self.obs_depth_jax, self.obs_sigma_jax)
        val = jnp.where(finite, self.nuisance.log_marginal(st, self.obs_sigma_jax), self._reject)
        return jnp.concatenate([val[None], st])

    def _mu_from_column(self, args: Any) -> Array:
        """Return one particle's binned depth from a converged column, ungated: a plotting interface."""
        y, theta = args
        aux = self.fwd.aux_from_y(y, theta[: self.n_chem_tp])
        depth = self.fwd.rt_depth(aux, *self.rt_args(theta))
        return self._mu_from_depth(depth, theta)

    def _make_batch_eval(self, mode: str) -> Callable[..., tuple[Array, Array, Array, EvalStats]]:
        """Build ``eval(U, Y, refs, active) -> (L, Y_new, refs_new, EvalStats)``, all batched over N.

        "warm": each particle re-converges by continuation from its carried column Y with incremental composition
        scaling, under the cold count_max. "cold": the two-stage map from the baseline; Y and refs still update, so cold
        evals can seed warm ones. A False row of ``active`` is padding: its chemistry takes no step, L is the reject
        sentinel and it gets no RT. The RT runs only on the rows whose L can survive: T-P in window, certified and
        active.
        """
        warm = mode == "warm"
        fwd = self.fwd

        def eval_batch(U: Array, Y: Array, refs: Array, active: ArrayLike) -> tuple[Array, Array, Array, EvalStats]:
            """Evaluate one batch."""
            active = jnp.asarray(active, bool)
            U = jnp.asarray(U, self.dtype)
            Theta = jax.vmap(self.theta_from_u)(U)
            C_ = Theta[:, : self.n_chem_tp]
            valid = jax.vmap(self.tp_valid)(Theta)
            # The whole batch, in one call on the lane queue.
            if warm:
                Ynew, CD = fwd.chem_solve_warm_diag_full_batch(C_, Y, refs[:, 0], refs[:, 1], refs[:, 2], active=active)
            else:
                Ynew, CD = fwd.chem_solve_cold_diag_batch(C_, active=active)
            AUX = jax.vmap(fwd.aux_from_y)(Ynew, C_)
            conv_ok = CD.conv_normal > 0.5
            # The solver's verdict alone: a certified solve can exit at cap or cap + 1, and conv_normal is False on
            # every cap exit. Only these rows get an RT; the rest hold the reject sentinel.
            usable = valid & active & conv_ok
            vals = _map_chunks_rows(jax.vmap(self._rt_val), (AUX, Theta), self._rt_chunk, usable, self._reject)
            nuis = None
            if self.nuisance is not None:
                vals, nuis = vals[:, 0], vals[:, 1:]
            ACC = CD.accept_count.astype(jnp.int32)
            L = jnp.where(jnp.isfinite(vals), vals, self._reject)
            # A rejected solve must not poison the carried state: pin it to the baseline column.
            ok = jnp.all(jnp.isfinite(Ynew), axis=(1, 2)) & usable
            Ynew = jnp.where(ok[:, None, None], Ynew, self.y_baseline[None])
            comp = self._comp_refs(C_)
            refs_new = jnp.where(ok[:, None], comp, jnp.zeros_like(comp))
            return (
                L,
                Ynew,
                refs_new,
                EvalStats(acc=ACC, conv_ok=conv_ok, reason=CD.termination_reason, last_delta=CD.last_delta, nuis=nuis),
            )

        return eval_batch

    def set_observations(self, depth: Any, sigma: ArrayLike) -> None:
        """Bake the observed depth and sigma into the pipeline, once, and fingerprint the target.

        Parameters
        ----------
        depth : array-like [n_bin]
            Observed depths.
        sigma : array-like [n_bin]
            Observed sigmas.

        Raises
        ------
        RuntimeError
            On a second call.
        """
        if self.obs_depth is not None:
            raise RuntimeError(
                "set_observations was already called on this pipeline. Observations "
                "are trace-time constants baked into the compiled likelihood, so "
                "swapping them in-process would silently keep the old data. Build a "
                "fresh pipeline (build_pipeline) for a different dataset."
            )
        depth, sigma = validate_observations(depth, sigma, self.n_bin)
        self.obs_depth = depth
        self.obs_sigma = sigma
        self.obs_depth_jax = jnp.asarray(depth, dtype=self.dtype)
        self.obs_sigma_jax = jnp.asarray(sigma, dtype=self.dtype)
        # Observations are the target's last input: fingerprint it here, so every checkpoint carries its identity.
        from retrieval_framework import provenance

        self.target_manifest = provenance.target_manifest(self.cfg, self)
        self.target_digest = provenance.manifest_digest(self.target_manifest)


def build_pipeline(cfg: C.Config) -> Pipeline:
    """Build the pipeline of one run, on its forward model.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    Pipeline
        The pipeline, with no observations yet.
    """
    C.validate_config(cfg)
    from retrieval_framework import retrieval_forward as RF  # lazy: pulls in vulcan_chem -> VULCAN-JAX + exojax

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
    return Pipeline(cfg, fwd)


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


def load_real_into_pipe(pipe: Pipeline) -> dict[str, np.ndarray]:
    """Inject the real observed depths and sigmas already attached to ``pipe.obs``.

    Parameters
    ----------
    pipe : Pipeline
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


def generate_observations(pipe: Pipeline, seed: int) -> dict[str, np.ndarray]:
    """Inject a synthetic spectrum: the model at truth plus Gaussian noise at the per-bin sigma.

    An uncertified truth column would inject a spectrum no certified draw can reproduce, so it is refused.

    Parameters
    ----------
    pipe : Pipeline
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


def _blank_state(pipe: Pipeline, N: int) -> tuple[Array, Array]:
    """Return (Y0, refs0) placeholders for a fresh batch: the baseline column, with references 0."""
    dtype = pipe.dtype
    Y0 = jnp.broadcast_to(pipe.y_baseline[None], (N, *tuple(pipe.y_baseline.shape))).astype(dtype)
    return Y0, jnp.zeros((N, 3), dtype)
