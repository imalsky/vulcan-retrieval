"""build_retrieval_forward on stub engines, with no chemistry: stage 1 runs at baseline composition, stage 2 from its
column at the proposal's, and the condensed reservoirs are not gas.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

# Import order: vulcan_chem must precede exojax, so check availability with find_spec.
if importlib.util.find_spec("exojax") is None:
    pytest.skip("chemistry/RT stack (exojax) not installed", allow_module_level=True)

import jax.numpy as jnp

from retrieval_framework import config_schema as C
from retrieval_framework import retrieval_forward as RF

LANES = 2
SIDX = {"H2": 0, "He": 1, "H2O": 2, "S8_l_s": 3}
MASSES = np.array([2.0, 4.0, 18.0, 256.0])
P_GRID = np.geomspace(1e-5, 1.0, 5)


def _diag(n, **fields):
    """A real ConvDiag of ``n`` rows, zero but for ``fields``; each tree's own field list."""
    zero = {f: jnp.zeros(n) for f in RF.vulcan_chem.ConvDiag._fields}
    return RF.vulcan_chem.ConvDiag(**{**zero, **fields})


def _forward(monkeypatch, stage_diag=None, **over):
    """Return the forward on stub engines and the list of its calls; ``stage_diag(n, k)`` is call k's diag."""
    calls = []
    diags = stage_diag or (lambda n, _k: _diag(n))

    def converged_y(theta, **kw):
        calls.append(("solo", np.asarray(theta), kw))
        y = jnp.ones((5, 4)) * (float(len(calls)))
        return (y, _diag(1)) if kw.get("return_conv_diag") else y

    def converged_y_queue(thetas, n_lanes, **kw):
        calls.append(("queue", np.asarray(thetas), {"n_lanes": n_lanes, **kw}))
        return jnp.ones((thetas.shape[0], 5, 4)) * float(len(calls)), diags(thetas.shape[0], len(calls))

    chem = SimpleNamespace(
        converged_y=converged_y,
        converged_y_queue=converged_y_queue,
        conden_spec=None,
        co_bz_bound=10.0,
        p_bar=P_GRID,
        sidx=SIDX,
        species_masses=jnp.asarray(MASSES),
        y0=np.ones((5, 4)),
        _integ=SimpleNamespace(statics=SimpleNamespace(rtol_accept=0.05)),
    )
    rt = SimpleNamespace(molecules=("H2O",), p_art_bar=P_GRID, wl_um=np.array([3.0, 4.0]))
    monkeypatch.setattr(RF, "_check_frozen_network", lambda _cfg: None)
    monkeypatch.setattr(RF.vulcan_chem, "build_chem_model", lambda *_a, **_k: chem)
    monkeypatch.setattr(RF.exojax_rt, "build_rt_model", lambda _profile: rt)
    monkeypatch.setattr(
        RF.tp_profile,
        "build_tp_model",
        lambda _cfg: SimpleNamespace(n_params=1, eval=lambda tp, p: jnp.full_like(p, tp[0])),
    )
    cfg = C.Config(
        vulcan_cfg_name="W39b",
        network="thermo/SNCHO_photo_network_2025.txt",
        rp_cm=1.0,
        rstar_cm=10.0,
        infer_s_o=True,
        cold_lanes=LANES,
        **over,
    )
    return RF.build_retrieval_forward(cfg), calls


THETA = jnp.asarray([0.4, -0.3, 1.2, 0.7, 1500.0])  # lnZ, c_o, lnKzz, s_o, the stub's one T-P entry


def test_cold_solve_runs_stage_1_at_baseline_composition(monkeypatch):
    fwd, calls = _forward(monkeypatch)
    assert fwd.chem_rtol == 0.05
    y, _ = fwd.chem_solve_cold_diag(THETA)
    assert [c[0] for c in calls] == ["solo", "solo"]
    # stage 1: every composition knob at 0, lnKzz and the T-P as drawn
    assert calls[0][1].tolist() == [0.0, 0.0, 1.2, 0.0, 1500.0]
    assert "warm_y" not in calls[0][2]
    # stage 2: the proposal, continuing stage 1's column from the baseline references
    assert calls[1][1].tolist() == pytest.approx(np.asarray(THETA).tolist())
    stage1_y = np.ones((5, 4))
    np.testing.assert_array_equal(np.asarray(calls[1][2]["warm_y"]), stage1_y)
    assert (calls[1][2]["lnZ_ref"], calls[1][2]["c_o_ref"], calls[1][2]["s_o_ref"]) == (0.0, 0.0, 0.0)
    assert float(y[0, 0]) == 2.0


def test_batched_solves_share_one_route_at_the_lane_cap(monkeypatch):
    fwd, calls = _forward(monkeypatch)
    stack = jnp.stack([THETA, THETA, THETA])
    fwd.chem_solve_cold_diag_batch(stack)
    assert [c[0] for c in calls] == ["queue", "queue"]
    assert np.all(calls[0][1][:, [0, 1, 3]] == 0.0)
    assert calls[0][1][0, 2] == pytest.approx(1.2)
    assert calls[1][1][0, 3] == pytest.approx(0.7)
    assert calls[0][2]["n_lanes"] == LANES
    calls.clear()
    fwd.chem_solve_cold_diag_batch(stack[:1])  # narrower than the lanes: every draw has its own
    assert calls[0][2]["n_lanes"] == 1


def test_condensed_reservoirs_are_not_gas(monkeypatch):
    """S8_l_s carries 256 g/mol: counted as gas it would dilute every VMR and inflate the mean molecular weight."""
    fwd, _ = _forward(monkeypatch)
    y = jnp.ones((5, 4)) * jnp.asarray([8.0, 1.0, 1.0, 90.0])
    vmr, vmr_h2, vmr_he, T_art, mmw = fwd.aux_from_y(y, THETA)
    np.testing.assert_allclose(np.asarray(vmr["H2O"]), 0.1, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(vmr_h2), 0.8, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(vmr_he), 0.1, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mmw), 0.8 * 2.0 + 0.1 * 4.0 + 0.1 * 18.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(T_art), 1500.0)


@pytest.mark.parametrize(("over", "key"), [({"cloud_model": "deck"}, "deck"), ({}, "cloud")])
def test_the_rt_keywords_route_the_cloud_block_by_the_cloud_model(monkeypatch, over, key):
    """What the likelihood's RT and the leave-one-out screen both pass."""
    fwd, _ = _forward(monkeypatch, **over)
    kw = fwd.rt_kwargs(jnp.asarray([-2.0, 0.5]), 0.3)
    assert set(kw) == {key, "lnM"}
    assert float(kw["lnM"]) == 0.3
    assert float(jnp.ravel(kw[key])[0]) == -2.0


def test_a_freeze_in_either_stage_reaches_the_batch_diag(monkeypatch):
    """Stage 1 is not gated, and its diag would be dropped: a freeze there must still read as frozen."""
    frozen = RF.TERM_DT_MIN_FROZEN

    def stage_diag(n, k):
        if k == 1:  # stage 1 freezes row 0
            return _diag(n, termination_reason=jnp.array([frozen, 1, 1]), last_delta=jnp.array([0.08, 0.1, 0.1]))
        return _diag(n, termination_reason=jnp.array([1, frozen, 1]), last_delta=jnp.array([0.01, 0.09, 0.01]))

    fwd, _ = _forward(monkeypatch, stage_diag=stage_diag)
    _, cd = fwd.chem_solve_cold_diag_batch(jnp.stack([THETA, THETA, THETA]))
    assert np.asarray(cd.termination_reason).tolist() == [frozen, frozen, 1]
    np.testing.assert_allclose(np.asarray(cd.last_delta), [0.08, 0.09, 0.01])
