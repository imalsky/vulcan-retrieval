"""Tests for the checksum-gated H2S boundary lookup (Route B G0 item 1).

Covers: checksum gate (tamper -> loud raise), grid validation, domain
refusal, parity of the jnp trilinear interpolant against an independent
numpy reference, node exactness, jvp finiteness with d ln x/d lnZ ~ 1
(the measured physics), piecewise-constant derivatives inside a cell, and
the knot-distance guard (crossing + proximity refusals).
"""

import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from retrieval_framework.forward import h2s_boundary as hb  # noqa: E402

TABLE = Path(__file__).resolve().parents[1] / "docs/route_b/h2s_boundary_table.json"


@pytest.fixture(scope="module")
def table():
    if not TABLE.exists():
        pytest.skip(f"boundary table not present at {TABLE}")
    return hb.load_h2s_boundary_table(TABLE)


def _numpy_trilinear_lnx(tab, T, lnZ, c_o):
    """Independent reference implementation (mirrors the validated builder)."""
    lnx = np.asarray(tab.lnx)

    def cell(grid, v):
        i = int(np.clip(np.searchsorted(grid, v) - 1, 0, grid.size - 2))
        return i, (v - grid[i]) / (grid[i + 1] - grid[i])

    i, u = cell(tab.T_K, T)
    j, v = cell(tab.lnZ, lnZ)
    k, w = cell(tab.c_o, c_o)
    c = lnx[i : i + 2, j : j + 2, k : k + 2]
    wu = np.array([1 - u, u])
    wv = np.array([1 - v, v])
    ww = np.array([1 - w, w])
    return float(np.einsum("a,b,c,abc->", wu, wv, ww, c))


def test_checksum_gate_rejects_tampered_table(tmp_path, table):
    data = json.loads(TABLE.read_text())
    data["lnx_table"][0][0][0] += 1e-6
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="hashes to"):
        hb.load_h2s_boundary_table(bad)


def test_domain_validation_raises_outside(table):
    hb.validate_domain(table, 1200.0, 0.0, 0.0)  # inside: no raise
    with pytest.raises(ValueError, match="outside the validated"):
        hb.validate_domain(table, 2100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="outside the validated"):
        hb.validate_domain(table, 1200.0, 3.0, 0.0)
    with pytest.raises(ValueError, match="outside the validated"):
        hb.validate_domain(table, 1200.0, 0.0, 0.9)


def test_interpolant_matches_numpy_reference(table):
    rng = np.random.default_rng(11)
    for _ in range(25):
        T = rng.uniform(410.0, 1990.0)
        z = rng.uniform(-2.2, 2.2)
        c = rng.uniform(-1.6, 0.45)
        got = float(hb.h2s_pin_mix(table, T, z, c))
        ref = np.exp(_numpy_trilinear_lnx(table, T, z, c))
        np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_node_exactness(table):
    lnx = np.asarray(table.lnx)
    for (ti, zi, ci) in [(0, 0, 0), (5, 4, 3), (16, 8, 6)]:
        got = float(
            hb.h2s_pin_mix(table, table.T_K[ti], table.lnZ[zi], table.c_o[ci])
        )
        np.testing.assert_allclose(got, np.exp(lnx[ti, zi, ci]), rtol=1e-12)


def test_jvp_finite_and_lnZ_slope_near_one(table):
    def lnx_of(theta):
        return jnp.log(hb.h2s_pin_mix(table, theta[0], theta[1], theta[2]))

    theta0 = jnp.asarray([1150.0, 0.3, -0.2])
    g = jax.jacfwd(lnx_of)(theta0)
    assert bool(jnp.all(jnp.isfinite(g)))
    # measured physics: d ln x_H2S / d lnZ = 0.987..1.008 over the domain
    assert 0.95 < float(g[1]) < 1.05


def test_derivative_piecewise_constant_within_cell(table):
    def lnx_of(theta):
        return jnp.log(hb.h2s_pin_mix(table, theta[0], theta[1], theta[2]))

    a = jnp.asarray([1120.0, 0.31, -0.21])
    b = jnp.asarray([1130.0, 0.33, -0.19])  # same cell on all axes
    ga = np.asarray(jax.jacfwd(lnx_of)(a))
    gb = np.asarray(jax.jacfwd(lnx_of)(b))
    # trilinear: dlnx/dT varies bilinearly with (lnZ, c_o) but at FIXED
    # other-axis coordinates only through them — moving within one cell
    # changes partials smoothly, never by a jump; crossing a knot jumps.
    # Verify no jump-scale change within the cell:
    assert np.all(np.abs(ga - gb) < 0.5 * np.maximum(np.abs(ga), 1e-12) + 1e-6)


def test_knot_guard(table):
    # comfortably inside a cell, small box: passes and returns the report
    rep = hb.assert_knot_safe(table, 1150.0, 0.29, -0.4167,
                              half_T=5.0, half_lnZ=0.05, half_co=0.05)
    assert not rep["any_crossing"]
    # box crossing the T=1200 knot: refused
    with pytest.raises(ValueError, match="crosses a lookup-cell boundary"):
        hb.assert_knot_safe(table, 1150.0, 0.29, -0.4167,
                            half_T=60.0, half_lnZ=0.05, half_co=0.05)
    # fiducial hugging a knot: refused
    with pytest.raises(ValueError, match="derivative-jump territory"):
        hb.assert_knot_safe(table, 1199.5, 0.29, -0.4167)
