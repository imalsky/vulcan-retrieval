"""Deep H2S boundary lookup: ln x_H2S(T_bottom, lnZ, c_o) at the engine P_b.

The Route B production boundary condition (B0A decision record item 3,
round-4 amendment): a checksum-gated trilinear interpolation of the
FastChem equilibrium table built by docs/route_b/h2s_boundary_table.py.
The interpolant is C0 — values continuous, first derivatives piecewise
constant with jumps at lookup-cell boundaries — so Fisher use additionally
requires the knot-distance guard below (or a future validated C1
replacement).

Contract (fail fast and loud, standing rule):
- `load_h2s_boundary_table` REFUSES a table whose lnx block does not hash
  to the value pinned in the decision record, whose grids are not strictly
  ascending, or whose values are non-finite.
- Domain enforcement is a BUILD-TIME duty: call `validate_domain` (loud
  raise on any point outside the validated grid) before entering jit; the
  pure interpolant itself clamps to the boundary cell and must never be
  fed unvalidated points.
- `knot_report` / `assert_knot_safe` implement the round-4 Fisher
  precondition: measure the fiducial point's and the uncertainty region's
  distance to the nearest interior knot per axis, and refuse/warn when the
  region approaches or crosses a cell boundary.
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
    return H2SBoundaryTable(
        T_K=grids["T_K"],
        lnZ=grids["lnZ"],
        c_o=grids["c_o"],
        lnx=jnp.asarray(lnx),
        P_bar=float(data["P_bar"]),
        sha256=digest,
        source=str(path),
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
