"""Deep H2S boundary lookup: ln x_H2S(T_bottom, lnZ, c_o) at the engine P_b.

The Route B production boundary condition (B0A decision record item 3,
round-4 amendment): a checksum-gated trilinear interpolation of the
FastChem equilibrium table built by docs/route_b/h2s_boundary_table.py.
The interpolant is C0 — values continuous, first derivatives piecewise
constant with jumps at lookup-cell boundaries — so Fisher use additionally
requires the knot-distance guard below (or a future validated C1
replacement).

Parity vs accuracy (keep the two metrics separate in any artifact): the
tests certify that this packaged interpolant reproduces the validated
prototype RULE (numpy-reference parity to 1e-12). The table's SCIENTIFIC
accuracy against FastChem holdouts is a separate measured quantity -- max
value error 0.11%, per-point derivative errors archived in the table
JSON's `validation` block -- and the full-chain AD-vs-FD gate (B0C G6)
remains pending. Neither implies the other.

Contract (fail fast and loud, standing rule):
- `load_h2s_boundary_table` REFUSES a table whose lnx block does not hash
  to the value pinned in the decision record, whose grids are not strictly
  ascending, whose values are non-finite, or whose provenance lacks the
  lnZ metallicity baseline (`baseline_X_H`).
- Domain enforcement is a BUILD-TIME duty: a model constructor must declare
  the full permitted (T_bottom, lnZ, c_o) box and prove it sits inside the
  grid via `validate_domain_box` (a per-point check cannot protect a model
  whose theta varies at run time); `validate_domain` is the per-point form
  for host-side evaluation-boundary checks (harness/gate scripts validate
  every actually-visited point, e.g. FD endpoints). The pure interpolant
  itself clamps to the boundary cell and must never be fed unvalidated
  points.
- `knot_report` / `assert_knot_safe` implement the round-4 Fisher
  precondition for axis-aligned uncertainty boxes: measure the fiducial
  point's and the region's distance to the nearest interior knot per axis,
  and refuse when the region approaches or crosses a cell boundary.
  `assert_points_same_cell` is the point-set form for everything a box
  cannot certify -- FD endpoints at h and h/2, covariance-eigenvector
  points, 1-sigma ellipsoid samples: trilinear derivatives are constant
  inside a cell and jump only at knots, so "all points share the
  fiducial's cell" is exactly "no derivative jump anywhere between them".
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# float64 is non-negotiable (VULCAN-JAX standing rule; ln x spans many
# decades and float32 costs ~1e-7 relative on the pin value). vulcan_jax
# sets this at ITS import, but this module is importable standalone and
# must not depend on import order.
jax.config.update("jax_enable_x64", True)

# sha256 of json.dumps(lnx_table-as-nested-lists) — the exact gate value in
# the B0A decision record (docs/route_b/b0a_decision_record.txt, item 3).
EXPECTED_LNX_TABLE_SHA256 = (
    "11e2b3795ad1731108673553450e9dfd7a04076c3630959116120ac662c82ca4"
)


class H2SBoundaryTable(NamedTuple):
    """Loaded, checksum-verified boundary table.

    `lnx` has shape (n_T, n_lnZ, n_co); the grids are strictly ascending
    float64 arrays. `P_bar` is the single validated bottom pressure.
    """

    T_K: np.ndarray
    lnZ: np.ndarray
    c_o: np.ndarray
    lnx: jnp.ndarray
    P_bar: float
    sha256: str
    source: str
    # lnZ = 0 metallicity basis: element-to-H number ratios of the FastChem
    # abundance file the table was built from (provenance block). A consumer
    # whose own lnZ = 0 baseline differs is using a shifted axis -- the model
    # constructor must verify its baseline ratios against these.
    baseline_X_H: dict


def load_h2s_boundary_table(path: str | Path) -> H2SBoundaryTable:
    """Load + verify the equilibrium boundary table (loud on any mismatch)."""
    path = Path(path)
    data = json.loads(path.read_text())
    digest = hashlib.sha256(json.dumps(data["lnx_table"]).encode()).hexdigest()
    if digest != EXPECTED_LNX_TABLE_SHA256:
        raise RuntimeError(
            f"H2S boundary table at {path} hashes to {digest}, expected "
            f"{EXPECTED_LNX_TABLE_SHA256} (the value pinned in the B0A "
            "decision record). Refusing: a changed table needs a re-derived "
            "gate value and re-validation, not silent acceptance."
        )
    grids = {
        "T_K": np.asarray(data["T_K"], dtype=np.float64),
        "lnZ": np.asarray(data["lnZ"], dtype=np.float64),
        "c_o": np.asarray(data["c_o"], dtype=np.float64),
    }
    for name, g in grids.items():
        if g.ndim != 1 or g.size < 2 or not np.all(np.diff(g) > 0):
            raise RuntimeError(
                f"H2S boundary table axis {name} is not strictly ascending "
                f"1-D (shape {g.shape})."
            )
    lnx = np.asarray(data["lnx_table"], dtype=np.float64)
    expect_shape = (grids["T_K"].size, grids["lnZ"].size, grids["c_o"].size)
    if lnx.shape != expect_shape:
        raise RuntimeError(
            f"H2S boundary table lnx shape {lnx.shape} != grids {expect_shape}."
        )
    if not np.all(np.isfinite(lnx)):
        raise RuntimeError("H2S boundary table carries non-finite lnx values.")
    baseline = (data.get("provenance") or {}).get("baseline_X_H")
    if not isinstance(baseline, dict) or not baseline:
        raise RuntimeError(
            f"H2S boundary table at {path} carries no provenance.baseline_X_H "
            "block: the lnZ axis is meaningless without its metallicity "
            "baseline. Refusing."
        )
    return H2SBoundaryTable(
        T_K=grids["T_K"],
        lnZ=grids["lnZ"],
        c_o=grids["c_o"],
        lnx=jnp.asarray(lnx),
        P_bar=float(data["P_bar"]),
        sha256=digest,
        source=str(path),
        baseline_X_H={k: float(v) for k, v in baseline.items()},
    )


def validate_domain(
    table: H2SBoundaryTable, T_bottom: float, lnZ: float, c_o: float
) -> None:
    """Loud raise when a point sits outside the validated grid (host-side,
    build time — the jitted interpolant never extrapolates silently only
    because every fed point passed here first)."""
    for name, g, v in (
        ("T_bottom", table.T_K, float(T_bottom)),
        ("lnZ", table.lnZ, float(lnZ)),
        ("c_o", table.c_o, float(c_o)),
    ):
        if not (g[0] <= v <= g[-1]):
            raise ValueError(
                f"H2S boundary point {name}={v} outside the validated table "
                f"domain [{g[0]}, {g[-1]}]; extrapolation is refused (B0A "
                "record item 3 / G0)."
            )


def validate_domain_box(
    table: H2SBoundaryTable, T_bounds, lnZ_bounds, c_o_bounds
) -> None:
    """Loud raise unless the whole axis-aligned parameter box is inside the
    validated grid.

    The build-time duty of any model constructor whose theta varies at run
    time (round-5 review): declare the full permitted (T_bottom, lnZ, c_o)
    box up front and prove it fits -- a per-point check at construction
    cannot protect later evaluations. Bounds are (lo, hi) per axis."""
    for name, g, bounds in (
        ("T_bottom", table.T_K, T_bounds),
        ("lnZ", table.lnZ, lnZ_bounds),
        ("c_o", table.c_o, c_o_bounds),
    ):
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo <= hi:
            raise ValueError(
                f"H2S boundary {name} box ({lo}, {hi}) has lo > hi."
            )
        if lo < g[0] or hi > g[-1]:
            raise ValueError(
                f"H2S boundary: declared {name} box [{lo}, {hi}] exceeds the "
                f"validated table domain [{g[0]}, {g[-1]}]; extrapolation is "
                "refused (B0A record item 3 / G0). Shrink the permitted "
                "parameter range or extend and re-validate the table."
            )


def cell_of(table: H2SBoundaryTable, T_bottom, lnZ, c_o) -> tuple:
    """Host-side lookup-cell index of a point, using the interpolant's own
    clamped indexing rule (so classification matches evaluation exactly)."""
    idx = []
    for g, v in (
        (table.T_K, T_bottom),
        (table.lnZ, lnZ),
        (table.c_o, c_o),
    ):
        idx.append(int(np.clip(np.searchsorted(g, float(v)) - 1, 0, g.size - 2)))
    return tuple(idx)


def assert_points_same_cell(table: H2SBoundaryTable, points) -> tuple:
    """Refuse unless every point lies in ONE lookup cell.

    The point-set form of the knot guard, for everything an axis-aligned box
    cannot certify (round-4 amendment + round-5 review): FD endpoints at h
    and h/2, covariance-eigenvector points, 1-sigma ellipsoid samples.
    Trilinear derivatives are constant inside a cell and jump only at knots,
    so "all points share the fiducial's cell" is exactly "no derivative jump
    anywhere between them". `points` is (N, 3) array-like of
    (T_bottom, lnZ, c_o) rows, fiducial first; every point is also
    domain-validated. Returns the shared cell index for archiving."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 1:
        raise ValueError(
            f"assert_points_same_cell expects (N, 3) rows of "
            f"(T_bottom, lnZ, c_o); got shape {pts.shape}."
        )
    for row in pts:
        validate_domain(table, row[0], row[1], row[2])
    cells = [cell_of(table, row[0], row[1], row[2]) for row in pts]
    bad = {i: c for i, c in enumerate(cells) if c != cells[0]}
    if bad:
        raise ValueError(
            f"H2S boundary lookup: points at rows {sorted(bad)} sit in "
            f"different lookup cells ({bad}) than the fiducial (row 0, cell "
            f"{cells[0]}); the trilinear derivative jumps between them. "
            "Shrink the FD steps / uncertainty region or move to the C1 "
            "interpolant."
        )
    return cells[0]


def _axis_cell(grid: jnp.ndarray, v):
    i = jnp.clip(jnp.searchsorted(grid, v) - 1, 0, grid.shape[0] - 2)
    w = (v - grid[i]) / (grid[i + 1] - grid[i])
    return i, w


def h2s_pin_mix(table: H2SBoundaryTable, T_bottom, lnZ, c_o):
    """x_H2S at the bottom node — pure jnp, jit/jvp-compatible.

    Trilinear interpolation of ln x over (T, lnZ, c_o), matching the
    validated prototype rule (docs/route_b/h2s_boundary_table.py) exactly;
    returns exp(ln x). Inputs MUST have passed `validate_domain` at build
    time (indices clamp; they never extrapolate honestly).
    """
    tg = jnp.asarray(table.T_K)
    zg = jnp.asarray(table.lnZ)
    cg = jnp.asarray(table.c_o)
    i, u = _axis_cell(tg, T_bottom)
    j, v = _axis_cell(zg, lnZ)
    k, w = _axis_cell(cg, c_o)
    c = jax_dynamic_cell(table.lnx, i, j, k)
    wu = jnp.stack([1.0 - u, u])
    wv = jnp.stack([1.0 - v, v])
    ww = jnp.stack([1.0 - w, w])
    lnx = jnp.einsum("a,b,c,abc->", wu, wv, ww, c)
    return jnp.exp(lnx)


def jax_dynamic_cell(lnx: jnp.ndarray, i, j, k) -> jnp.ndarray:
    """The (2,2,2) corner block at cell (i, j, k), traceable indices."""
    return jax.lax.dynamic_slice(lnx, (i, j, k), (2, 2, 2))


def knot_report(
    table: H2SBoundaryTable,
    T_bottom: float,
    lnZ: float,
    c_o: float,
    half_T: float = 0.0,
    half_lnZ: float = 0.0,
    half_co: float = 0.0,
) -> dict:
    """Knot-distance guard data for the C0 interpolant (round-4 amendment).

    For the fiducial point and an axis-aligned uncertainty box (half-widths
    per axis, e.g. Fisher 1-sigma), report per axis: the distance from the
    point to the nearest INTERIOR knot in units of the local cell width,
    and whether the box crosses any interior knot (where the trilinear
    derivative jumps).
    """
    out = {}
    for name, g, v, h in (
        ("T_bottom", table.T_K, float(T_bottom), float(half_T)),
        ("lnZ", table.lnZ, float(lnZ), float(half_lnZ)),
        ("c_o", table.c_o, float(c_o), float(half_co)),
    ):
        interior = g[1:-1]
        cell = np.diff(g).min()
        if interior.size:
            dist = float(np.min(np.abs(interior - v)))
            crosses = bool(np.any((interior > v - h) & (interior < v + h)))
        else:
            dist = float("inf")
            crosses = False
        out[name] = {
            "distance_to_nearest_interior_knot": dist,
            "distance_in_min_cell_widths": dist / cell,
            "box_halfwidth": h,
            "box_crosses_knot": crosses,
        }
    out["any_crossing"] = any(a["box_crosses_knot"] for a in out.values()
                              if isinstance(a, dict))
    return out


def assert_knot_safe(
    table: H2SBoundaryTable,
    T_bottom: float,
    lnZ: float,
    c_o: float,
    half_T: float = 0.0,
    half_lnZ: float = 0.0,
    half_co: float = 0.0,
    min_cell_frac: float = 0.1,
) -> dict:
    """Refuse when the point/region approaches or crosses a lookup knot.

    A Fisher consumer must call this (round-4 amendment): raises when the
    uncertainty box crosses an interior knot or the fiducial point sits
    within `min_cell_frac` of one (derivative-jump territory). Returns the
    report on success so callers can archive it.
    """
    rep = knot_report(table, T_bottom, lnZ, c_o, half_T, half_lnZ, half_co)
    if rep["any_crossing"]:
        raise ValueError(
            "H2S boundary lookup: the uncertainty region crosses a lookup-"
            f"cell boundary ({ {k: v for k, v in rep.items() if isinstance(v, dict) and v['box_crosses_knot']} }); "
            "the trilinear derivative jumps there. Refuse, shrink the "
            "region, or move to the C1 interpolant."
        )
    for name in ("T_bottom", "lnZ", "c_o"):
        if rep[name]["distance_in_min_cell_widths"] < min_cell_frac:
            raise ValueError(
                f"H2S boundary lookup: fiducial {name} sits within "
                f"{min_cell_frac} cell widths of an interior knot "
                f"({rep[name]}); derivative-jump territory. Refuse or move "
                "to the C1 interpolant."
            )
    return rep
