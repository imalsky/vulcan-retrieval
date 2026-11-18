"""One chunk of unit-cube points through a batched chemistry evaluator, as ``run_nautilus`` evaluates it.

The padding to the evaluator's width and the active mask, the host-side verdict with its rejection classes, and the cold
solve of extra rows that the end-of-run spot-check uses. docs/method.md "Sampler" states the padding rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from retrieval_framework import constants

if TYPE_CHECKING:
    from collections.abc import Callable

    from retrieval_framework.pipeline import Pipeline

__all__ = ["Rejections", "classify", "cold_log_l", "pad", "verdict"]


class Rejections(NamedTuple):
    """A chunk's rejected rows, one mask per class."""

    tp_out: np.ndarray
    exhausted: np.ndarray
    stalled: np.ndarray
    nonfinite: np.ndarray

    @property
    def dead(self) -> np.ndarray:
        """Return the rows of any class."""
        return self.tp_out | self.exhausted | self.stalled | self.nonfinite

    def code(self) -> np.ndarray:
        """Return each row's class: 0 T-P out, 1 exhausted, 2 stalled, 3 otherwise."""
        return np.where(self.tp_out, 0, np.where(self.exhausted, 1, np.where(self.stalled, 2, 3)))


def classify(
    tp_in: np.ndarray, conv: np.ndarray, acc: np.ndarray, L: np.ndarray, count_max: int, reject_below: float
) -> Rejections:
    """Sort a chunk's rows into the rejection classes.

    Solver state first: the warm evaluator already floors L on a stalled or exhausted exit, which is not a non-finite
    forward. A solve certified at cap or cap + 1 is kept.

    Parameters
    ----------
    tp_in : numpy.ndarray of bool [n]
        The rows whose T-P profile is in the window.
    conv : numpy.ndarray of bool [n]
        The rows the solver certified.
    acc : numpy.ndarray of int [n]
        Accepted solver steps.
    L : numpy.ndarray [n]
        log L.
    count_max : int
        The step cap.
    reject_below : float
        log L at or below this is a rejection.

    Returns
    -------
    Rejections
        One mask per class.
    """
    tp_out = ~tp_in
    exhausted = ~tp_out & ~conv & (acc >= count_max)
    stalled = ~tp_out & ~exhausted & ~conv
    nonfinite = ~tp_out & ~exhausted & conv & (~np.isfinite(L) | (L <= reject_below))
    return Rejections(tp_out, exhausted, stalled, nonfinite)


def pad(pipe: Pipeline, Zc: np.ndarray, n_batch: int) -> tuple[np.ndarray, Any, np.ndarray, Any]:
    """Pad a chunk with copies of its last row to the evaluator's width.

    Parameters
    ----------
    pipe : pipeline.Pipeline
        The pipeline.
    Zc : numpy.ndarray [m, n_dim]
        Unit-cube points, m <= ``n_batch``.
    n_batch : int
        The evaluator's width.

    Returns
    -------
    tuple
        The padded points [n_batch, n_dim], their u on the device, the T-P window mask [n_batch] and the active mask
        on the device [n_batch]: the real rows in the window.
    """
    import jax
    import jax.numpy as jnp

    m = len(Zc)
    if m < n_batch:
        Zc = np.concatenate([Zc, np.repeat(Zc[-1:], n_batch - m, axis=0)])
    Uj = jnp.asarray(np.log(Zc) - np.log1p(-Zc), pipe.dtype)
    # Not bitwise for the kept rows: fewer jobs on the lane queue shift the shared refresh cadence.
    tp_in = np.asarray(jax.device_get(pipe.tp_valid_batch(Uj)), bool)
    return Zc, Uj, tp_in, jnp.asarray((np.arange(n_batch) < m) & tp_in)


def verdict(
    pipe: Pipeline, Zc: np.ndarray, m: int, tp_in: np.ndarray, L: Any, st: Any
) -> tuple[np.ndarray, np.ndarray, Rejections]:
    """Stop on a column frozen at dt_min, then bring a chunk's first ``m`` rows to the host and classify them.

    Parameters
    ----------
    pipe : pipeline.Pipeline
        The pipeline.
    Zc : numpy.ndarray [n_batch, n_dim]
        The padded points.
    m : int
        The real rows.
    tp_in : numpy.ndarray of bool [n_batch]
        The T-P window mask.
    L : array [n_batch]
        The evaluator's log L.
    st : stats
        The evaluator's solver statistics (``acc``, ``conv_ok``, ``reason``, ``last_delta``).

    Returns
    -------
    tuple
        log L [m], accepted steps [m] and the rejection classes.
    """
    import jax

    from retrieval_framework import pipeline as P

    P._abort_if_frozen(st.reason, st.last_delta, Zc[:m], pipe.fwd.chem_rtol)
    L = np.asarray(jax.device_get(L), np.float64)[:m]
    acc = np.asarray(jax.device_get(st.acc), np.int64)[:m]
    conv = np.asarray(jax.device_get(st.conv_ok), bool)[:m]
    return L, acc, classify(tp_in[:m], conv, acc, L, int(pipe.fwd.chem.count_max), constants.REJECT_BELOW)


def cold_log_l(pipe: Pipeline, cold: Callable[..., Any], n_batch: int, Z: np.ndarray) -> np.ndarray:
    """Return the log-likelihood of unit-cube points solved cold, as the run's cold chunks are; nothing is logged.

    Parameters
    ----------
    pipe : pipeline.Pipeline
        The run's pipeline.
    cold : callable
        The run's jitted cold evaluator (``run_nautilus.make_loglike``'s ``cold``), so nothing compiles again.
    n_batch : int
        The evaluator's width: the points go through in padded chunks of it.
    Z : numpy.ndarray [n, n_dim]
        Unit-cube points.

    Returns
    -------
    numpy.ndarray [n]
        log L, -inf on every rejection class.
    """
    from retrieval_framework import pipeline as P

    Y0, refs0 = P._blank_state(pipe, n_batch)
    out = np.empty(len(Z))
    for i in range(0, len(Z), n_batch):
        m = min(n_batch, len(Z) - i)
        Zc, Uj, tp_in, act = pad(pipe, Z[i : i + m], n_batch)
        L, _, _, st = cold(Uj, Y0, refs0, act)
        L, _, rej = verdict(pipe, Zc, m, tp_in, L, st)
        out[i : i + m] = np.where(rej.dead, -np.inf, L)
    return out
