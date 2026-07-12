"""Observation-injection validation (2026-07-12 re-audit item 4).

pipeline.validate_observations is the API-boundary guard behind
pipe.set_observations: the Gaussian likelihood divides by sigma and logs it, so
a non-finite depth or a non-positive/non-finite sigma must RAISE here rather than
silently produce NaN/Inf likelihoods. Pure-numpy helper -- no forward model.
"""
import numpy as np
import pytest

from retrieval_framework import pipeline as P


N = 5
GOOD_DEPTH = np.linspace(0.01, 0.02, N)
GOOD_SIGMA = np.full(N, 1e-4)


def test_valid_vector_is_preserved_exactly():
    d, s = P.validate_observations(GOOD_DEPTH, GOOD_SIGMA, N, np.float64)
    assert np.array_equal(d, GOOD_DEPTH)
    assert np.array_equal(s, GOOD_SIGMA)
    assert d.shape == (N,) and s.shape == (N,)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length must be n_bin"):
        P.validate_observations(GOOD_DEPTH[:-1], GOOD_SIGMA[:-1], N, np.float64)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_depth_raises(bad):
    d = GOOD_DEPTH.copy()
    d[2] = bad
    with pytest.raises(ValueError, match="depths must all be finite"):
        P.validate_observations(d, GOOD_SIGMA, N, np.float64)


@pytest.mark.parametrize("bad", [0.0, -1e-4, np.nan, np.inf, -np.inf])
def test_bad_sigma_raises(bad):
    s = GOOD_SIGMA.copy()
    s[1] = bad
    with pytest.raises(ValueError, match="sigmas must all be finite and strictly"):
        P.validate_observations(GOOD_DEPTH, s, N, np.float64)


def test_all_positive_finite_sigma_passes_at_tiny_values():
    s = np.full(N, 1e-30)
    d, out = P.validate_observations(GOOD_DEPTH, s, N, np.float64)
    assert np.all(out > 0.0)
