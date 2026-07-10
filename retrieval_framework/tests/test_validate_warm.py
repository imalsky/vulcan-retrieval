"""validate_warm.compare must (1) measure the warm-vs-cold bias on healthy particles
only, (2) route cold-nonconverged and dead-warm particles into separate counts rather
than polluting the statistics, and (3) mirror _init_state's rejection condition
(count_max-exhausted OR -1e30 sentinel OR non-finite)."""
import numpy as np

from retrieval_framework.validate_warm import compare


def test_compare_healthy_cloud():
    rng = np.random.default_rng(3)
    L_warm = rng.normal(-100.0, 3.0, size=32)
    bias = rng.normal(0.0, 1e-3, size=32)
    s = compare(L_warm, L_warm + bias, np.full(32, 500), count_max=5000)
    assert s["n"] == 32 and s["n_ok"] == 32
    assert s["n_cold_nonconverged"] == 0 and s["n_dead_warm"] == 0
    assert np.isclose(s["abs_max"], np.max(np.abs(bias)))
    assert np.isclose(s["abs_median"], np.median(np.abs(bias)))
    assert np.isclose(s["logl_spread"], L_warm.max() - L_warm.min())
    assert np.allclose(s["dlogl"], bias)


def test_compare_excludes_failures():
    L_warm = np.array([-10.0, -11.0, -12.0, -1.0e30, -13.0])
    L_cold = np.array([-10.5, -11.0, -1.0e30, -4.0, -13.2])
    wa = np.array([100, 5000, 200, 100, 100])  # particle 1 exhausted count_max
    s = compare(L_warm, L_cold, wa, count_max=5000)
    # 2 cold-nonconverged (sentinel + exhausted), 1 dead warm, 2 healthy
    assert s["n_cold_nonconverged"] == 2 and s["n_dead_warm"] == 1 and s["n_ok"] == 2
    assert np.isclose(s["abs_max"], 0.5)
    assert np.isnan(s["dlogl"][1]) and np.isnan(s["dlogl"][2]) and np.isnan(s["dlogl"][3])


def test_compare_all_excluded_is_nan_not_crash():
    s = compare(np.array([-1.0e30]), np.array([-5.0]), np.array([0]), count_max=5000)
    assert s["n_ok"] == 0 and np.isnan(s["abs_max"])
