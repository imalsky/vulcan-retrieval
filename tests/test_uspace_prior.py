"""u-space transform: bounds, midpoints, induced-prior uniformity.
Light: imports pipeline (jax) but never builds the VULCAN forward."""
import importlib.util

import numpy as np
import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework.config_schema import ParamSpec  # noqa: E402

EDGE_ATOL = 1e-8   # theta at |u| = 40 sits within this of the box edge

# The second spec is the config's own noise-inflation prior (log10-uniform
# on [0.5, 3.0]), so the tests below pin what a run actually samples.
SPECS = [
    ParamSpec("a", "a", "uniform", -2.0, 3.0, 0.0, "chem"),
    next(s for s in C.specs_from_config(C.Config(infer_noise_inflation=True))
         if s.name == "noise_inflation"),
]


def test_bounds_and_midpoint():
    f, _ = P.make_uspace(SPECS, jnp.float64)
    th0 = np.asarray(f(jnp.zeros(2)))
    assert np.isclose(th0[0], 0.5)                       # uniform midpoint
    assert np.isclose(th0[1], np.sqrt(0.5 * 3.0))        # log10 midpoint = geometric mean
    th_lo = np.asarray(f(jnp.full(2, -40.0)))
    th_hi = np.asarray(f(jnp.full(2, 40.0)))
    assert np.allclose(th_lo, [-2.0, 0.5], atol=EDGE_ATOL)
    assert np.allclose(th_hi, [3.0, 3.0], atol=EDGE_ATOL)


def test_prior_samples_uniform_in_theta():
    f, s = P.make_uspace(SPECS, jnp.float64)
    U = s(jax.random.PRNGKey(0), 40000)
    TH = np.asarray(jax.vmap(f)(U))
    # uniform dim: mean = center, var = span^2/12
    assert abs(TH[:, 0].mean() - 0.5) < 0.03
    assert abs(TH[:, 0].var() - 25.0 / 12.0) < 0.06
    # log10_uniform dim: log10(theta) uniform on [log10 lo, log10 hi]
    lg = np.log10(TH[:, 1])
    lo, hi = np.log10(0.5), np.log10(3.0)
    assert abs(lg.mean() - 0.5 * (lo + hi)) < 0.01
    assert abs(lg.var() - (hi - lo) ** 2 / 12.0) < 0.01


def test_tp_window_is_inset_inside_the_opacity_range():
    """The T-P reject window sits strictly inside the k-tables' range."""
    from retrieval_framework import tp_profile as tp
    assert tp.T_OPA_MIN_K < tp._T_MIN < tp._T_MAX < tp.T_OPA_MAX_K


def test_gaussian_prior_is_mean_plus_sigma_times_the_normal_quantile():
    """A gaussian spec (lo = mean, hi = sigma) maps u = 0 to the mean and a
    uniform z to N(mean, sigma); the uniform dimension beside it is untouched."""
    specs = [ParamSpec("a", "a", "uniform", -2.0, 3.0, 0.0, "chem"),
             ParamSpec("lnM", "lnM", "gaussian", 0.0, 0.11, 0.0, "lnM")]
    f, s = P.make_uspace(specs, jnp.float64)
    th0 = np.asarray(f(jnp.zeros(2)))
    assert np.allclose(th0, [0.5, 0.0])
    TH = np.asarray(jax.vmap(f)(s(jax.random.PRNGKey(2), 40000)))
    assert abs(TH[:, 1].mean()) < 0.002 and abs(TH[:, 1].std() - 0.11) < 0.002
    assert abs(TH[:, 0].mean() - 0.5) < 0.03
    # the layout: the gaussian lnM sits after lnR0, the extras after lnKzz
    cfg = C.Config(infer_lnM=True, infer_s_o=True, kzz_mode="powerlaw", cloud_model="deck")
    names = [sp.name for sp in C.specs_from_config(cfg)]
    assert names == ["lnZ", "c_o", "lnKzz", "s_o", "kzz_slope", "Tirr", "log10kappa",
                     "log10gamma", "lnR0", "lnM", "log10P_deck"]
    assert C.chem_extras(cfg) == ("s_o", "kzz_slope")
    assert next(sp for sp in C.specs_from_config(cfg) if sp.name == "lnM").prior_type == "gaussian"



# find_spec, not importorskip: importing exojax before vulcan_chem breaks the
# import-order guard for every later test in the worker.
@pytest.mark.skipif(importlib.util.find_spec("exojax") is None,
                    reason="needs exojax (the Guillot profile); CI is stack-free")
def test_free_tint_at_the_fixed_value_is_the_fixed_profile():
    """A retrieved Tint is the last T-P entry and, at the fixed value, gives the
    fixed-Tint profile exactly; the spec order matches the T-P sub-vector."""
    from vulcan_forward import vulcan_chem  # noqa: F401  (before exojax)
    from retrieval_framework import tp_profile as tp
    fixed_cfg, free_cfg = C.Config(), C.Config(tp_infer_Tint=True)
    fixed, free = tp.build_tp_model(fixed_cfg), tp.build_tp_model(free_cfg)
    names = [s.name for s in C.specs_from_config(free_cfg) if s.kind == "tp"]
    assert names == ["Tirr", "log10kappa", "log10gamma", "Tint"]
    assert (fixed.n_params, free.n_params) == (3, 4)
    p = jnp.logspace(-9.0, np.log10(7.0), 40)
    th = jnp.array([1100.0, -0.5, -1.0])
    assert np.array_equal(
        np.asarray(fixed.eval(th, p)),
        np.asarray(free.eval(jnp.append(th, fixed_cfg.tp_Tint_K), p)))
    assert np.all(np.asarray(free.eval(jnp.append(th, 400.0), p))
                  >= np.asarray(fixed.eval(th, p)))


@pytest.mark.skipif(importlib.util.find_spec("exojax") is None,
                    reason="needs exojax (the Guillot profile); CI is stack-free")
def test_deep_temperature_parametrization_and_the_adiabat():
    """With tp_deep_param="T1bar" the fourth T-P entry is the temperature at
    tp_deep_p_bar: the profile passes through it exactly, equals the Tint
    parametrization at the reconstructed Tint, and a value the irradiation
    alone exceeds has no profile (NaN, which the pipeline rejects). With
    tp_adiabat the profile follows the adiabat below the retrieved boundary and
    the Guillot profile above it."""
    from vulcan_forward import vulcan_chem  # noqa: F401  (before exojax)
    from retrieval_framework import tp_profile as tp
    deep_cfg = C.Config(tp_infer_Tint=True, tp_deep_param="T1bar", tp_deep_p_bar=1.0)
    deep = tp.build_tp_model(deep_cfg)
    tint = tp.build_tp_model(C.Config(tp_infer_Tint=True))
    assert [s.name for s in C.specs_from_config(deep_cfg) if s.kind == "tp"] == [
        "Tirr", "log10kappa", "log10gamma", "T1bar"]
    assert (deep.n_params, deep.deep_param, deep.p_deep_bar) == (4, "T1bar", 1.0)
    th = jnp.array([1000.0, -0.5, -1.5, 1700.0])
    assert abs(float(deep.eval(th, jnp.array([1.0]))[0]) - 1700.0) < 1e-6
    Tint_rec = float(deep.tint_of(th))
    assert 0.0 < Tint_rec < 1700.0
    p = jnp.logspace(-9.0, np.log10(7.6), 60)
    np.testing.assert_allclose(np.asarray(deep.eval(th, p)),
                               np.asarray(tint.eval(th.at[3].set(Tint_rec), p)), rtol=1e-12)
    # the irradiation term alone sets T(1 bar) with Tint = 0; below it no profile exists
    floor = float(tint.eval(th.at[3].set(0.0), jnp.array([1.0]))[0])
    assert np.all(np.isnan(np.asarray(deep.eval(th.at[3].set(0.9 * floor), p))))
    # the adiabat: Guillot above P_rcb, T_rcb (P / P_rcb)^nabla below
    ad = tp.build_tp_model(C.Config(tp_infer_Tint=True, tp_adiabat=True, tp_nabla_ad=0.29))
    assert ad.n_params == 5
    th_ad = jnp.array([1000.0, -0.5, -1.5, 300.0, -1.0])
    T_g = np.asarray(tint.eval(th_ad[:4], p))
    T_a = np.asarray(ad.eval(th_ad, p))
    above, below = np.asarray(p) <= 0.1, np.asarray(p) > 0.1
    np.testing.assert_allclose(T_a[above], T_g[above], rtol=0, atol=0)
    T_rcb = float(tint.eval(th_ad[:4], jnp.array([0.1]))[0])
    np.testing.assert_allclose(T_a[below], T_rcb * (np.asarray(p)[below] / 0.1) ** 0.29,
                               rtol=1e-12)
    assert np.all(T_a[below] >= T_g[below])
    # T1bar with the adiabat: T(1 bar) is still the retrieved value when the
    # boundary sits above 1 bar (the deep level on the adiabat) and when it sits
    # below; a boundary temperature under the irradiation floor has no profile
    both = tp.build_tp_model(C.Config(tp_infer_Tint=True, tp_deep_param="T1bar",
                                      tp_adiabat=True, tp_nabla_ad=0.29))
    for lg_rcb in (-1.0, 0.5):
        th_b = jnp.array([1000.0, -0.5, -0.5, 1700.0, lg_rcb])
        assert abs(float(both.eval(th_b, jnp.array([1.0]))[0]) - 1700.0) < 1e-6, lg_rcb
        T_b = np.asarray(both.eval(th_b, p))
        deep = np.asarray(p) > 10.0 ** lg_rcb
        ratio = T_b[deep] / (np.asarray(p)[deep] / 10.0 ** lg_rcb) ** 0.29
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)     # one adiabat below P_rcb
    # gamma 0.03 lifts the deep irradiation floor above the 1700 K adiabat's 872 K at
    # 0.1 bar: no Guillot part reaches it, so the draw is NaN (rejected upstream)
    assert np.all(np.isnan(np.asarray(both.eval(
        jnp.array([1000.0, -0.5, -1.5, 1700.0, -1.0]), p))))
