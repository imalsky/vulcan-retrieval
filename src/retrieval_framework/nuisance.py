"""The Gaussian likelihood with the offsets and noise inflation(s) integrated out over their priors.

docs/method.md states the integrals.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from retrieval_framework import observations as OBS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from jax import Array
    from jax.typing import ArrayLike

    from retrieval_framework import config_schema as C

__all__ = ["NUISANCE_NODES", "make_nuisance_marginal"]

# Trapezoid nodes in log10 b for an integrated noise inflation. On the real v5 bins
# the error against brute force falls as h^2: 3.6e-3 nats at 513 nodes in the worst
# case tried (b piled at its floor), <= 5e-5 on realistic spectra (notes "Design
# records", the integrated nuisances).
NUISANCE_NODES = 2049
_LOG2PI = math.log(2.0 * math.pi)


def _log_phi_diff(a: Any, c: Any, xp: ModuleType, log_ndtr: Callable[[Any], Any]) -> Any:
    """Return log(Phi(a) - Phi(c)) for a > c, stable in both tails."""
    flip = c > 0  # both in the upper tail: 1 - Phi
    hi = xp.where(flip, -c, a)
    lo = xp.where(flip, -a, c)
    lh, ll = log_ndtr(hi), log_ndtr(lo)
    return lh + xp.log1p(-xp.exp(ll - lh))


# size-limit-ok: the closures share one node grid and one group design
def make_nuisance_marginal(cfg: C.Config, groups: Iterable[str], group_per_bin: ArrayLike) -> SimpleNamespace:
    """Build the Gaussian likelihood with the offsets and noise inflation(s) integrated out over their priors.

    Group g's bins see the offset o_g, uniform on ``prior_offset_ppm`` (none for the reference group or with
    infer_offsets off), and sigma0 b, with b log10-uniform on ``prior_noise_inflation`` (per group, for every bin, or
    b = 1). From R_g = sum r / s0^2 and S_g = sum r^2 / s0^2, the o integral is closed form and the b integral a
    normalized trapezoid over ``NUISANCE_NODES`` nodes in log10 b; the groups multiply.

    Parameters
    ----------
    cfg : config_schema.Config
        Reads the nuisance settings.
    groups : sequence of str
        The ordered groups.
    group_per_bin : array-like [n_bin]
        Each bin's group.

    Returns
    -------
    types.SimpleNamespace
        ``names`` (the integrated parameters), ``groups``, ``stats(mu, depth, sigma0)``, ``log_marginal(stats,
        sigma0)``, ``draw(stats, sigma0, rng)`` (numpy; each row's b on the node grid, then o_g from its truncated
        normal; offsets in ppm) and ``apply(mu, nuis)`` (mu with the offsets, the sigma multiplier per bin).
    """
    from jax.scipy.special import log_ndtr as jlog_ndtr, logsumexp as jlse

    groups = list(groups)
    G = len(groups)
    Nm = OBS.build_group_design({"groups": groups, "group": group_per_bin})
    has_off = np.array([bool(cfg.infer_offsets) and k > 0 for k in range(G)])
    a_lo, a_hi = (float(v) * OBS.PPM for v in cfg.prior_offset_ppm)
    infl = bool(cfg.infer_noise_inflation)
    per_group = infl and bool(cfg.noise_inflation_per_group)
    if infl:
        t = np.linspace(*np.log10(cfg.prior_noise_inflation), NUISANCE_NODES)  # type: ignore[call-overload]  # two bounds
        w = np.full(NUISANCE_NODES, 1.0 / (NUISANCE_NODES - 1))
        w[[0, -1]] *= 0.5
    else:
        t, w = np.zeros(1), np.ones(1)
    b, logw = 10.0**t, np.log(w)
    names = [f"offset_{g}" for g, h in zip(groups, has_off, strict=True) if h]
    names += [f"noise_inflation_{g}" for g in groups] if per_group else ["noise_inflation"] if infl else []
    n_off = int(has_off.sum())

    def _consts(sigma0: Any, xp: ModuleType) -> tuple[Any, Any, Any]:
        """Return the per-group weight, bin count and summed log sigma."""
        s0 = xp.asarray(sigma0)
        N = xp.asarray(Nm)
        return N.T @ (1.0 / s0**2), N.sum(axis=0), N.T @ xp.log(s0)

    def _terms(R: Any, S: Any, consts: tuple[Any, Any, Any], xp: ModuleType, log_ndtr: Callable[[Any], Any]) -> Any:
        """Return log of group g's likelihood at node b_k, [..., G, K], integrated over its offset when it has one."""
        W, n, slog = (c[:, None] for c in consts)
        bb = xp.asarray(b)
        R, S = R[..., None], S[..., None]
        base = -slog - 0.5 * n * _LOG2PI - n * xp.log(bb)
        oh = R / W
        sd = bb / xp.sqrt(W)
        off = (
            base
            - 0.5 * (S - R * oh) / bb**2
            + 0.5 * _LOG2PI
            + xp.log(sd)
            + _log_phi_diff((a_hi - oh) / sd, (a_lo - oh) / sd, xp, log_ndtr)
            - math.log(a_hi - a_lo)
        )
        return xp.where(xp.asarray(has_off)[:, None], off, base - 0.5 * S / bb**2)

    def stats(mu: Any, depth: Any, sigma0: Any) -> Array:
        """Return the group statistics [R, S], of shape [2 G]."""
        N = jnp.asarray(Nm)
        r = (depth - mu) / sigma0**2
        return jnp.concatenate([N.T @ r, N.T @ (r * (depth - mu))])

    def log_marginal(st: Any, sigma0: Any) -> Array:
        """Return the log marginal likelihood from the group statistics."""
        T = _terms(st[:G], st[G:], _consts(sigma0, jnp), jnp, jlog_ndtr)  # (G, K)
        lw = jnp.asarray(logw)
        if per_group or not infl:  # without an inflation K = 1
            return jnp.sum(jlse(T + lw, axis=-1))
        return jlse(jnp.sum(T, axis=0) + lw)  # one b: the groups multiply at each node

    def draw(st: Any, sigma0: Any, rng: np.random.Generator, chunk: int = 1024) -> np.ndarray:
        """Draw the integrated parameters from their conditional posterior, [n, len(names)]."""
        from scipy.special import log_ndtr, logsumexp
        from scipy.stats import truncnorm

        st = np.asarray(st, np.float64)
        consts = _consts(np.asarray(sigma0, np.float64), np)
        out = np.empty((len(st), len(names)))
        for i in range(0, len(st), chunk):
            R, S = st[i : i + chunk, :G], st[i : i + chunk, G:]
            T = _terms(R, S, consts, np, log_ndtr)  # (n, G, K)
            if not per_group:  # one b: its conditional is the product's
                T = np.broadcast_to(T.sum(axis=1, keepdims=True), T.shape)
            T = T + logw
            P = np.exp(T - logsumexp(T, axis=-1, keepdims=True)).cumsum(axis=-1)
            k = (P < rng.random((*P.shape[:-1], 1))).sum(axis=-1).clip(0, len(b) - 1)
            if not per_group:
                k = np.broadcast_to(k[:, :1], k.shape)
            bk = b[k]  # (n, G)
            cols = []
            for g in np.flatnonzero(has_off):
                oh, sd = R[:, g] / consts[0][g], bk[:, g] / np.sqrt(consts[0][g])
                # atleast_1d: truncnorm returns a scalar for one-element arguments, as a one-row chunk has
                rvs = truncnorm.rvs((a_lo - oh) / sd, (a_hi - oh) / sd, loc=oh, scale=sd, random_state=rng)
                cols.append(np.atleast_1d(rvs) / OBS.PPM)
            cols += list(bk.T) if per_group else [bk[:, 0]] if infl else []
            out[i : i + chunk] = np.stack(cols, axis=1)
        return out

    def apply(mu: Any, nuis: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return mu with the drawn offsets and the sigma multiplier per bin."""
        nuis = np.asarray(nuis, np.float64)
        mu = np.asarray(mu, np.float64) + (nuis[:, :n_off] * OBS.PPM) @ Nm[:, has_off].T
        mult = (
            nuis[:, n_off:] @ Nm.T
            if per_group
            else nuis[:, n_off : n_off + 1] * np.ones(Nm.shape[0])
            if infl
            else np.ones_like(mu)
        )
        return mu, mult

    return SimpleNamespace(names=names, groups=groups, stats=stats, log_marginal=log_marginal, draw=draw, apply=apply)
