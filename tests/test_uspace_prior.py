"""u-space transform: bounds, midpoints, induced-prior uniformity, finite gradients.
Light: imports pipeline (jax) but never builds the VULCAN forward."""
import numpy as np
import jax

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
    f, _, _ = P.make_uspace(SPECS, jnp.float64)
    th0 = np.asarray(f(jnp.zeros(2)))
    assert np.isclose(th0[0], 0.5)                       # uniform midpoint
    assert np.isclose(th0[1], np.sqrt(0.5 * 3.0))        # log10 midpoint = geometric mean
    th_lo = np.asarray(f(jnp.full(2, -40.0)))
    th_hi = np.asarray(f(jnp.full(2, 40.0)))
    assert np.allclose(th_lo, [-2.0, 0.5], atol=EDGE_ATOL)
    assert np.allclose(th_hi, [3.0, 3.0], atol=EDGE_ATOL)


def test_prior_samples_uniform_in_theta():
    f, _, s = P.make_uspace(SPECS, jnp.float64)
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


def test_log_prior_gradient_finite():
    _, lp, _ = P.make_uspace(SPECS, jnp.float64)
    g = jax.grad(lp)(jnp.asarray([0.3, -20.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    # log-prior maximized at u=0 (theta mid-box)
    assert float(lp(jnp.zeros(2))) > float(lp(jnp.asarray([3.0, -3.0])))


def test_tp_window_is_inset_inside_the_opacity_range():
    """The T-P reject window sits strictly inside the k-tables' range."""
    from retrieval_framework import tp_profile as tp
    assert tp.T_OPA_MIN_K < tp._T_MIN < tp._T_MAX < tp.T_OPA_MAX_K

