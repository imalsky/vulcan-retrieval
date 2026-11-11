"""The warm-start anchors of a run: every certified column, keyed by its unit-cube chemistry and T-P coordinates.

``run_nautilus.make_loglike`` starts each solve from the nearest anchor; docs/method.md "Warm starts".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
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

    def nearest(self, zq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return each query's nearest anchor, Euclidean in the unit cube.

        Parameters
        ----------
        zq : numpy.ndarray [n, n_key]
            The queries.

        Returns
        -------
        tuple of numpy.ndarray
            ``(index, distance)``.
        """
        idx = np.empty(len(zq), int)
        dist = np.empty(len(zq))
        for j in range(0, len(zq), 64):
            d2 = ((zq[j : j + 64, None, :] - self.z[None]) ** 2).sum(-1)
            idx[j : j + 64] = d2.argmin(1)
            dist[j : j + 64] = np.sqrt(d2.min(1))
        return idx, dist

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
