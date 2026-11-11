"""build_pipeline's gates on a stub forward, with no chemistry: the certificate, the T-P window on both grids, the
rejected row's baseline pin, the exact Gaussian log L and the refusal of an uncertified synthetic truth.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import retrieval_framework
from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
from retrieval_framework import pipeline as P
from retrieval_framework import provenance

NZ, NI = 4, 3
P_ART = np.geomspace(1e-5, 1.0, 6)
P_CHEM = np.geomspace(1e-5, 10.0, NZ)  # reaches below the RT bottom, as the real chemistry grid does
T_MIN, T_MAX = 100.0, 2000.0
DT_PER_DEX = 100.0  # the stub T-P: Tirr + DT_PER_DEX log10(p / bar)
LNZ_UNCERTIFIED = 0.5  # the stub solver certifies below this lnZ
LNZ_NAN_DEPTH = -0.5  # and the stub RT is non-finite below this one
BASE_DEPTH = 0.02
Y_BASELINE = np.full((NZ, NI), 7.0)


class _Diag(NamedTuple):
    conv_normal: jnp.ndarray
    accept_count: jnp.ndarray
    termination_reason: jnp.ndarray
    last_delta: jnp.ndarray


def _diag(C_):
    n = C_.shape[0]
    return _Diag(
        (C_[:, 0] < LNZ_UNCERTIFIED).astype(jnp.float64), jnp.full(n, 11), jnp.zeros(n, int), jnp.full(n, 0.01)
    )


def _column(C_):
    return jnp.broadcast_to((C_[:, 0] + 10.0)[:, None, None], (C_.shape[0], NZ, NI))


def _stub_forward(n_tp):
    def rt_depth(aux, lnR0, _cloud, _lnM):
        wl = 1e4 / jnp.linspace(1923.0, 3450.0, 400)
        depth = BASE_DEPTH * jnp.exp(2.0 * lnR0) * (1.0 + 0.01 * wl)
        return jnp.where(aux > 10.0 + LNZ_NAN_DEPTH, depth, jnp.nan)

    def native_depth_aux(chem_theta, lnR0, cloud, lnM):
        aux = chem_theta[0] + 10.0
        return rt_depth(aux, lnR0, cloud, lnM), aux, (chem_theta[0] < LNZ_UNCERTIFIED).astype(jnp.float64)

    return SimpleNamespace(
        wl_um=1e4 / np.linspace(1923.0, 3450.0, 400),
        n_tp=n_tp,
        n_chem=3,
        p_art_bar=P_ART,
        p_chem_bar=P_CHEM,
        tp_model=SimpleNamespace(
            eval=lambda tp, p: tp[0] + DT_PER_DEX * jnp.log10(p),
            T_min=T_MIN,
            T_max=T_MAX,
        ),
        y_baseline=Y_BASELINE,
        rt_depth=rt_depth,
        aux_from_y=lambda y, _theta: y[0, 0],
        native_depth_aux=native_depth_aux,
        chem_solve_cold_diag_batch=lambda C_, **_kw: (_column(C_), _diag(C_)),
        chem_solve_warm_diag_full_batch=lambda C_, Y, r0, *_refs, **_kw: (Y + r0[:, None, None], _diag(C_)),
        chem=SimpleNamespace(count_max=50),
    )


def _build(monkeypatch, **over):
    cfg = C.Config(
        vulcan_cfg_name="W39b",
        network="thermo/SNCHO_photo_network_2025.txt",
        rp_cm=1.0,
        rstar_cm=10.0,
        generate_synthetic_data=True,
        infer_offsets=True,
        infer_noise_inflation=True,
        prior_lnZ=(-1.0, 1.0),
        prior_Tirr=(1000.0, 2500.0),
        smc_rt_chunk=2,  # the needed-first reorder of the RT chunks
        **over,
    )
    n_tp = sum(s.kind == "tp" for s in C.specs_from_config(cfg))
    stub = SimpleNamespace(build_retrieval_forward=lambda _cfg: _stub_forward(n_tp))
    monkeypatch.setitem(sys.modules, "retrieval_framework.retrieval_forward", stub)
    monkeypatch.setattr(retrieval_framework, "retrieval_forward", stub, raising=False)
    monkeypatch.setattr(provenance, "target_manifest", lambda _cfg, _pipe: {"stub": True})
    grid = OBS.get_observation_grid

    def two_groups(cfg_, wl_um):
        """The synthetic grid split in two instrument groups: the second gets an offset parameter."""
        obs, real = grid(cfg_, wl_um)
        n = obs["wl"].size
        return {**obs, "group": np.asarray(["A"] * (n // 2) + ["B"] * (n - n // 2)), "groups": ["A", "B"]}, real

    monkeypatch.setattr(OBS, "get_observation_grid", two_groups)
    return P.build_pipeline(cfg)


def _u(pipe, **theta):
    """Return u with each named parameter at its value and every other at its prior's middle."""
    u = np.zeros(pipe.n_dim)
    for name, value in theta.items():
        i = pipe.names.index(name)
        z = (value - pipe.param_prior_lo[i]) / (pipe.param_prior_hi[i] - pipe.param_prior_lo[i])
        u[i] = np.log(z) - np.log1p(-z)
    return u


def test_tp_window_holds_on_both_grids(monkeypatch):
    pipe = _build(monkeypatch)
    in_art_only = pipe.theta_from_u(jnp.asarray(_u(pipe, Tirr=1950.0)))  # 2050 K only below the RT bottom
    assert not bool(pipe.tp_valid(in_art_only))
    assert bool(pipe.tp_valid(pipe.theta_from_u(jnp.asarray(_u(pipe, Tirr=1850.0)))))
    # the sampled prior is the box inside the window: an off-window draw is redrawn, never kept
    tirr = jax.vmap(pipe.theta_from_u)(pipe.sample_prior_u(jax.random.PRNGKey(0), 32))[:, pipe.names.index("Tirr")]
    assert float(tirr.max()) <= T_MAX - DT_PER_DEX


def test_cold_evaluator_gates_and_exact_log_l(monkeypatch):
    pipe = _build(monkeypatch)
    n_bin = pipe.n_bin
    depth_obs = np.full(n_bin, BASE_DEPTH) + 1e-4 * np.sin(np.arange(n_bin))
    sigma = np.full(n_bin, 2e-4) * (1.0 + 0.1 * np.cos(np.arange(n_bin)))
    pipe.set_observations(depth_obs, sigma)

    offsets = {n: 30.0 * (i + 1) for i, n in enumerate(n for n in pipe.names if n.startswith("offset_"))}
    kept = _u(pipe, lnZ=-0.2, Tirr=1500.0, noise_inflation=1.3, lnR0=0.01, **offsets)
    uncertified = _u(pipe, lnZ=0.8, Tirr=1500.0)
    out_of_window = _u(pipe, lnZ=-0.2, Tirr=2200.0)
    padding = _u(pipe, lnZ=-0.2, Tirr=1500.0)
    nan_depth = _u(pipe, lnZ=-0.8, Tirr=1500.0)
    U = jnp.asarray(np.stack([kept, uncertified, out_of_window, padding, nan_depth]))
    active = jnp.asarray([True, True, True, False, True])
    Y0, refs0 = P._blank_state(pipe, 5)
    L, Ynew, refs, st = pipe._make_batch_eval("cold")(U, Y0, refs0, active)
    L, Ynew, refs = np.asarray(L), np.asarray(Ynew), np.asarray(refs)

    # the kept row's log L, by hand from the data, the design matrices and the Gaussian
    theta = np.asarray(pipe.theta_from_u(U[0]))
    native = BASE_DEPTH * np.exp(2.0 * theta[pipe.names.index("lnR0")]) * (1.0 + 0.01 * np.asarray(pipe.fwd.wl_um))
    design = OBS.build_offset_design(pipe.obs)
    off = np.array([theta[pipe.names.index(n)] for n in offsets])
    assert off.size > 0, "the synthetic grid must carry an offset group"
    mu = pipe.B @ native + design @ (off * 1e-6)
    sig = sigma * theta[pipe.names.index("noise_inflation")]
    want = -0.5 * np.sum(((depth_obs - mu) / sig) ** 2) - np.sum(np.log(sig)) - 0.5 * n_bin * math.log(2.0 * math.pi)
    assert L[0] == pytest.approx(want, rel=1e-12)

    # every other row is the reject sentinel, for its own reason
    assert np.all(L[1:] == P.REJECT_LOGL)
    assert np.asarray(st.conv_ok).tolist() == [True, False, True, True, True]
    assert np.all(np.asarray(st.acc) == 11)

    # a kept column is carried with its composition; a rejected one is pinned to the baseline with references 0
    np.testing.assert_allclose(Ynew[0], -0.2 + 10.0, rtol=1e-12)
    assert refs[0].tolist() == pytest.approx([-0.2, theta[1], 0.0], rel=1e-12)
    for row in (1, 2, 3):
        assert np.array_equal(Ynew[row], Y_BASELINE), row
        assert np.array_equal(refs[row], np.zeros(3)), row


def test_warm_evaluator_hands_each_row_its_own_references(monkeypatch):
    pipe = _build(monkeypatch)
    pipe.set_observations(np.full(pipe.n_bin, BASE_DEPTH), np.full(pipe.n_bin, 2e-4))
    U = jnp.asarray(np.stack([_u(pipe, lnZ=-0.2, Tirr=1500.0), _u(pipe, lnZ=-0.1, Tirr=1500.0)]))
    Y = jnp.full((2, NZ, NI), 20.0)
    refs = jnp.asarray([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    _, Ynew, _, _ = pipe._make_batch_eval("warm")(U, Y, refs, jnp.ones(2, bool))
    assert np.asarray(Ynew)[:, 0, 0].tolist() == [20.25, 20.5]


@pytest.mark.parametrize(("truth_lnZ", "refused"), [(0.8, True), (0.0, False)])
def test_synthetic_truth_must_certify(monkeypatch, truth_lnZ, refused):
    pipe = _build(monkeypatch, truth_lnZ=truth_lnZ, truth_Tirr=1500.0)
    if refused:
        with pytest.raises(RuntimeError, match="does not certify"):
            P.generate_observations(pipe, seed=1)
        assert pipe.obs_depth is None
    else:
        P.generate_observations(pipe, seed=1)
        assert np.all(np.isfinite(pipe.obs_depth))


def test_a_data_bin_outside_the_model_band_is_refused(monkeypatch):
    """Every bin of the data window is fitted or the run stops: the band cut drops none."""
    from pathlib import Path

    lo = 3.0 + 0.1 * np.arange(20)
    obs = {
        "wl": lo + 0.05,
        "wl_lo": lo,
        "wl_hi": lo + 0.1,
        "depth": np.full(20, 0.02),
        "sigma": np.full(20, 1.0e-4),
        "group": np.array(["A"] * 20),
    }
    monkeypatch.setattr(OBS, "load_real_observations", lambda _cfg: dict(obs))
    cfg = C.Config(obs_dir=Path("data"), obs_products={"A": ("a.csv",)})
    real, is_real = OBS.get_observation_grid(cfg, np.linspace(2.9, 5.1, 400))
    assert is_real
    assert real["wl"].size == 20
    with pytest.raises(ValueError, match="3 of the 20 data bins"):
        OBS.get_observation_grid(cfg, np.linspace(2.9, 4.75, 400))
