"""The warm-start anchors of a run: every certified column, keyed by its unit-cube chemistry and T-P coordinates.

``run_nautilus.make_loglike`` starts each solve from the nearest anchor that gives it a physical start; docs/method.md
"Warm starts".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

from retrieval_framework import constants

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = ["Anchors", "atomic_savez"]


def atomic_savez(path: Path, **arrays: Any) -> None:
    """Write an npz under a temporary name and rename it, so a kill leaves no partial file."""
    tmp = path.with_name(f".{path.name}.part")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
    tmp.replace(path)


class Anchors:
    """The certified columns so far: unit-cube coordinates, the column and its reference composition.

    One npz per added batch in ``folder``; an existing folder is reloaded on resume. A file with two-wide references
    reads as s_o = 0.
    """

    N_REF = 3

    def __init__(self, folder: Path, n_key: int) -> None:
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self.z = np.empty((0, n_key))
        self.refs = np.empty((0, self.N_REF))
        self.parts: list[np.ndarray] = []
        self.part, self.row = np.empty(0, int), np.empty(0, int)
        self._tree: cKDTree | None = None
        for f in sorted(folder.glob("anchors_*.npz")):
            with np.load(f) as d:
                self._append(d["z"], d["Y"], d["refs"])

    def __len__(self) -> int:
        """Return the number of anchors."""
        return len(self.z)

    def _append(self, z: np.ndarray, Y: np.ndarray, refs: np.ndarray) -> None:
        """Append one batch in memory."""
        refs = np.asarray(refs, np.float64)
        if refs.shape[1] < self.N_REF:
            refs = np.pad(refs, ((0, 0), (0, self.N_REF - refs.shape[1])))
        self.part = np.concatenate([self.part, np.full(len(z), len(self.parts))])
        self.row = np.concatenate([self.row, np.arange(len(z))])
        self.parts.append(np.asarray(Y))
        self.z = np.concatenate([self.z, z])
        self.refs = np.concatenate([self.refs, refs])
        self._tree = None

    def add(self, z: np.ndarray, Y: np.ndarray, refs: np.ndarray) -> None:
        """Append certified columns as one new file.

        Parameters
        ----------
        z : numpy.ndarray [n, n_key]
            Unit-cube coordinates.
        Y : numpy.ndarray [n, nz, ni]
            The columns.
        refs : numpy.ndarray [n, 3]
            The reference compositions.
        """
        if len(z):
            atomic_savez(self.folder / f"anchors_{len(self.parts):05d}.npz", z=z, Y=Y, refs=refs)
            self._append(z, Y, refs)

    def nearest(self, zq: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return each query's ``k`` nearest anchors, nearest first, Euclidean in the unit cube.

        Parameters
        ----------
        zq : numpy.ndarray [n, n_key]
            The queries.
        k : int
            Anchors per query; at most the number held.

        Returns
        -------
        tuple of numpy.ndarray [n, k]
            ``(index, distance)``.
        """
        n = len(self.z)
        k = min(k, n)
        tree = self._tree
        if tree is None:
            tree = self._tree = cKDTree(self.z)
        idx = np.empty((len(zq), k), int)
        dist = np.empty((len(zq), k))
        for r, q in enumerate(zq):
            m = min(n, constants.ANCHOR_SHORTLIST_FACTOR * k)
            while True:
                tdist, cand = tree.query(q, m)
                cand = np.atleast_1d(cand)
                d2 = ((q[None, :] - self.z[cand]) ** 2).sum(-1)
                kth = np.partition(d2, k - 1)[k - 1]
                if m == n or np.atleast_1d(tdist)[-1] ** 2 > kth * (1.0 + constants.ANCHOR_SHORTLIST_MARGIN):
                    break
                m = min(n, 2 * m)
            # Every anchor at or inside the k-th distance is ordered by distance, a tie to the older anchor, so a tie
            # at the k-th place cannot drop the older one.
            inside = cand[d2 <= kth]
            d2_in = d2[d2 <= kth]
            order = np.lexsort((inside, d2_in))[:k]
            idx[r] = inside[order]
            dist[r] = np.sqrt(d2_in[order])
        return idx, dist

    def nearest_physical(
        self, zq: np.ndarray, ok: Callable[[np.ndarray, np.ndarray], np.ndarray], k: int, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return each query's nearest anchor whose warm start is physical, among its ``k`` nearest.

        A guess built from another draw's column can be no physical state for this draw while the next nearest column
        gives one (notes "Warm starts").

        Parameters
        ----------
        zq : numpy.ndarray [n, n_key]
            The queries.
        ok : callable
            From the columns [n, nz, ni] and references [n, 3] of one candidate per query to whether each start is
            physical, [n].
        k : int
            Candidates tried per query.
        active : numpy.ndarray of bool [n]
            The queries searched; the others keep their nearest anchor, rank 0, and never prolong the search.

        Returns
        -------
        tuple of numpy.ndarray [n]
            ``(index, distance, rank)``: the anchor taken, its distance and its place among the nearest (0 the
            nearest); rank -1 marks a query with no physical start among the ``k``, which keeps its nearest anchor.
        """
        cand, dist = self.nearest(zq, k)
        rows = np.arange(len(zq))
        rank = np.zeros(len(zq), int)
        left = rows[np.asarray(active, bool)]
        for j in range(cand.shape[1]):
            rank[left] = j
            left = left[~np.asarray(ok(*self.columns(cand[rows, rank])), bool)[left]]
            if not len(left):
                break
        rank[left] = 0
        out = rank.copy()
        out[left] = -1
        return cand[rows, rank], dist[rows, rank], out

    def columns(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the columns and references of anchors ``idx``.

        Parameters
        ----------
        idx : numpy.ndarray [n]
            The anchor indices.

        Returns
        -------
        tuple of numpy.ndarray
            ``(Y, refs)``.
        """
        return (
            np.stack([self.parts[p][r] for p, r in zip(self.part[idx], self.row[idx], strict=True)]),
            self.refs[idx],
        )
