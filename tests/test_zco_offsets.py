"""Offset-model regression tests for the Z-C/O Fisher machinery (zco_lib).

Pins the 2026-07-20 offset correction (propagated from vulcan-jwst-tool's
2026-07-12 P0-A audit fix): under offset_model="all_groups" EVERY instrument
group carries a depth-offset nuisance, the first included, because lnR0 is a
physical radiative-transfer derivative (not spectrally constant) and cannot
absorb an absolute depth-calibration error. Three contracts:

  1. a single-group analysis includes one constant offset;
  2. relabeling the group order does not change the science constraints
     (it does under the retained "reference_fixed" reproduction mode, which is
     exactly why that mode is not the default);
  3. zco_lib's rank-aware marginal sigmas agree with vulcan-jwst-tool's
     fisher.mode_forecast on an equivalent synthetic design.

Pure numpy; the synthetic design goes through the real build_design binning
path (no cached Jacobians needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ZCO_DIR = Path(__file__).resolve().parents[1] / "scripts" / "zco_information"
sys.path.insert(0, str(_ZCO_DIR))
import zco_lib  # noqa: E402


# --------------------------------------------------------------------------- #
#  Synthetic tier payload + observations                                      #
# --------------------------------------------------------------------------- #
def _payload(n_model=1200):
    """Synthetic per-tier Jacobian payload on a fine model grid.

    J_lnR0 is deliberately close to (but not exactly) constant in wavelength:
    near-constant lnR0 is the regime where the offset-model choice matters most
    (a constant lnR0 would be exactly degenerate with the sum of all offsets).
    """
    wl = np.linspace(1.0, 5.0, n_model)
    J_chem = np.column_stack([
        200e-6 * np.exp(-((wl - 2.7) / 0.30) ** 2) + 30e-6 * np.sin(3.0 * wl),
        150e-6 * np.exp(-((wl - 4.3) / 0.20) ** 2) + 20e-6 * np.cos(2.0 * wl),
        60e-6 * np.sin(1.3 * wl + 0.4),
        25e-6 * (wl / 5.0) + 10e-6 * np.sin(5.0 * wl),
    ])
    J_lnR0 = 600e-6 * (1.0 + 0.03 * np.sin(2.0 * wl))
    depth = 0.0210 + 400e-6 * np.exp(-((wl - 2.7) / 0.30) ** 2)
    return wl, dict(depth=depth, J_chem=J_chem, J_lnR0=J_lnR0)


def _obs(groups=("A", "B"), n_per=30):
    """Synthetic observed bins: contiguous wavelength blocks, one per group,
    different per-group noise levels (so the anchoring choice is not symmetric)."""
    spans = {"A": (1.10, 2.90), "B": (3.00, 4.90)}
    sig = {"A": 80e-6, "B": 120e-6}
    W, LO, HI, D, S, G = [], [], [], [], [], []
    for g in groups:
        lo, hi = spans[g]
        edges = np.linspace(lo, hi, n_per + 1)
        W.append(0.5 * (edges[:-1] + edges[1:]))
        LO.append(edges[:-1]); HI.append(edges[1:])
        D.append(np.full(n_per, 0.021)); S.append(np.full(n_per, sig[g]))
        G.append(np.array([g] * n_per))
    wl = np.concatenate(W)
    o = np.argsort(wl)
    group = np.concatenate(G)[o]
    return dict(wl=wl[o], wl_lo=np.concatenate(LO)[o], wl_hi=np.concatenate(HI)[o],
                depth=np.concatenate(D)[o], sigma=np.concatenate(S)[o],
                group=group, groups=list(groups))


def _marg_sigmas_chem(des):
    F, C = zco_lib.fisher(des["J"], des["sigma"])
    return np.sqrt(np.diag(C)[:4])


# --------------------------------------------------------------------------- #
#  1. single-group analyses include one constant offset                       #
# --------------------------------------------------------------------------- #
def test_single_group_all_groups_has_one_offset():
    wl, pay = _payload()
    obs = _obs(groups=("A",))
    des = zco_lib.build_design(pay, wl, obs, offset_model="all_groups")
    off = [k for k in des["keys"] if k.startswith("offset_")]
    assert off == ["offset_A"]
    j = des["keys"].index("offset_A")
    assert np.allclose(des["J"][:, j], zco_lib.OFFSET_UNIT)


def test_single_group_reference_fixed_has_no_offset():
    # The retained reproduction mode: one group == the assumed-calibrated
    # reference, so no offset at all. Documented, not default.
    wl, pay = _payload()
    obs = _obs(groups=("A",))
    des = zco_lib.build_design(pay, wl, obs, offset_model="reference_fixed")
    assert not any(k.startswith("offset_") for k in des["keys"])


def test_unknown_offset_model_raises():
    wl, pay = _payload()
    obs = _obs(groups=("A",))
    with pytest.raises(ValueError):
        zco_lib.build_design(pay, wl, obs, offset_model="anchored")


# --------------------------------------------------------------------------- #
#  2. group-order invariance                                                  #
# --------------------------------------------------------------------------- #
def test_group_order_invariance_all_groups():
    wl, pay = _payload()
    obs_ab = _obs(groups=("A", "B"))
    obs_ba = dict(obs_ab, groups=["B", "A"])   # same data, relabeled order
    s_ab = _marg_sigmas_chem(zco_lib.build_design(pay, wl, obs_ab,
                                                  offset_model="all_groups"))
    s_ba = _marg_sigmas_chem(zco_lib.build_design(pay, wl, obs_ba,
                                                  offset_model="all_groups"))
    assert np.allclose(s_ab, s_ba, rtol=1e-9)


def test_group_order_changes_reference_fixed():
    # The demonstration of why the fix is needed: under the old parameterization
    # the first group is treated as perfectly depth-calibrated, so relabeling
    # which instrument comes first changes the forecast.
    wl, pay = _payload()
    obs_ab = _obs(groups=("A", "B"))
    obs_ba = dict(obs_ab, groups=["B", "A"])
    s_ab = _marg_sigmas_chem(zco_lib.build_design(pay, wl, obs_ab,
                                                  offset_model="reference_fixed"))
    s_ba = _marg_sigmas_chem(zco_lib.build_design(pay, wl, obs_ba,
                                                  offset_model="reference_fixed"))
    assert np.max(np.abs(s_ab - s_ba) / s_ab) > 1e-4


def test_all_groups_never_tighter_than_reference_fixed():
    # Freeing one more calibration direction can only lose information.
    wl, pay = _payload()
    obs = _obs(groups=("A", "B"))
    s_all = _marg_sigmas_chem(zco_lib.build_design(pay, wl, obs,
                                                   offset_model="all_groups"))
    s_ref = _marg_sigmas_chem(zco_lib.build_design(pay, wl, obs,
                                                   offset_model="reference_fixed"))
    assert np.all(s_all >= s_ref * (1 - 1e-12))


# --------------------------------------------------------------------------- #
#  3. agreement with vulcan-jwst-tool's rank-aware Fisher                     #
# --------------------------------------------------------------------------- #
def test_matches_jwst_tool_mode_forecast():
    jf = pytest.importorskip(
        "jwst_tool.fisher",
        reason="vulcan-jwst-tool not installed; cross-tool agreement not checkable")
    wl, pay = _payload()
    obs = _obs(groups=("A", "B"))
    des = zco_lib.build_design(pay, wl, obs, offset_model="all_groups")
    assert des["keys"][:5] == zco_lib.CHEM_KEYS + ["lnR0"]

    # Same binned rows, jwst-tool's own offset construction (one per segment,
    # segment 0 included). Offset column units differ (1.0 vs OFFSET_UNIT); the
    # free-parameter marginals are invariant under nuisance rescaling.
    seg = np.array([0 if g == des["groups"][0] else 1 for g in des["group"]])
    result = dict(jac_bins=des["J"][:, :5].T, sigma=des["sigma"], seg=seg)
    sig_jwst = jf.mode_forecast(result, zco_lib.CHEM_KEYS)

    sig_zco = dict(zip(des["keys"][:4], _marg_sigmas_chem(des)))
    for k in zco_lib.CHEM_KEYS:
        assert sig_jwst[k] == pytest.approx(sig_zco[k], rel=1e-8)


# --------------------------------------------------------------------------- #
#  rank-aware covariance unit checks                                          #
# --------------------------------------------------------------------------- #
def test_rank_aware_cov_full_rank_matches_inv():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(40, 6))
    F = A.T @ A
    dg = {}
    C = zco_lib.rank_aware_cov(F, diag=dg)
    assert dg["rank"] == 6
    assert np.allclose(C, np.linalg.inv(F), rtol=1e-8, atol=0)


def test_rank_aware_cov_flags_exact_degeneracy():
    # Duplicate column -> exact null direction; the loaded parameters must read
    # inf variance, never a falsely precise finite number.
    rng = np.random.default_rng(4)
    A = rng.normal(size=(40, 5))
    A[:, 4] = A[:, 3]
    F = A.T @ A
    dg = {}
    C = zco_lib.rank_aware_cov(F, diag=dg)
    assert dg["rank"] == 4
    assert np.isinf(C[3, 3]) and np.isinf(C[4, 4])
    assert np.all(np.isfinite(np.diag(C)[:3]))
