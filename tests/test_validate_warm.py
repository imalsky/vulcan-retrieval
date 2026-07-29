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


def test_compare_grad_identical_and_scaled():
    from retrieval_framework.validate_warm import compare_grad
    rng = np.random.default_rng(7)
    G = rng.normal(size=(16, 5))
    ok = np.ones(16, bool)
    s = compare_grad(G, G, ok)
    assert s["n_ok"] == 16
    assert s["rel_max"] < 1e-15 and np.isclose(s["cos_min"], 1.0)
    # a uniformly 10%-longer cold gradient: rel = 0.1/1.1 (symmetric denominator),
    # direction identical
    s2 = compare_grad(G, 1.1 * G, ok)
    assert np.isclose(s2["rel_max"], 0.1 / 1.1) and np.isclose(s2["cos_min"], 1.0)


def test_compare_grad_flags_zeroed_and_flipped_rows():
    from retrieval_framework.validate_warm import compare_grad
    rng = np.random.default_rng(11)
    Gc = rng.normal(size=(4, 3))
    Gw = Gc.copy()
    Gw[1] = 0.0          # badgrad zero-drift row: rel=1, cos undefined (nan)
    Gw[2] = -Gc[2]       # sign-flipped drift: rel = ||2 Gc||/||Gc|| = 2, cos=-1
    s = compare_grad(Gw, Gc, np.ones(4, bool))
    assert np.isclose(s["rel"][0], 0.0)
    assert np.isclose(s["rel"][1], 1.0) and np.isnan(s["cos"][1])
    assert np.isclose(s["rel"][2], 2.0) and np.isclose(s["cos"][2], -1.0)
    assert np.isclose(s["cos_min"], -1.0) and np.isclose(s["rel_max"], 2.0)


def test_compare_grad_respects_ok_mask():
    from retrieval_framework.validate_warm import compare_grad
    G = np.ones((3, 2))
    Gw = G.copy()
    Gw[2] = 100.0        # a wild row that MUST be excluded by the mask
    s = compare_grad(Gw, G, np.array([True, True, False]))
    assert s["n_ok"] == 2 and s["rel_max"] < 1e-15
    assert np.isnan(s["rel"][2]) and np.isnan(s["cos"][2])
