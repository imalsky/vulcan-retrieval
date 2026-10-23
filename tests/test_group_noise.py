"""Per-group noise inflation (cfg.noise_inflation_per_group; off = the one b).

Stack-free:
* layout: off keeps the scalar 'noise_inflation' exactly; on gives one
  'noise_inflation_<group>' per observation group, in group order, after the
  offsets, each log10-uniform on prior_noise_inflation; it needs the groups and
  infer_noise_inflation;
* observations.build_group_design / noise_scale_per_bin in all three modes;
* pipeline.make_noise_model, the likelihood's sigma, is sigma0 times
  noise_scale_per_bin, the inflation the predictive band uses, in every mode
  (both apply observations.noise_map);
* nautilus_report.predictive_replicates with one b and with a per-group
  inflation.

Opt-in production preflight (RUN_PRODUCTION_PREFLIGHT=1): the real three-group
spectrum gets three inflations, in the pipeline's group order.
"""
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import observations as OBS  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402

GROUPS = ["NIRISS", "G395H_NRS1", "G395H_NRS2"]


def _cfg(**kw):
    return C.Config(infer_noise_inflation=True, **kw)


def _layout(cfg, groups=GROUPS):
    specs = C.specs_from_config(cfg, groups=groups)
    return specs, [s.name for s in specs], [s.kind for s in specs]


def test_off_keeps_the_scalar_parameter():
    specs, names, _ = _layout(_cfg())
    noise = [s for s in specs if s.kind == "noise"]
    assert [(s.name, s.label, s.prior_type, s.lo, s.hi, s.truth) for s in noise] == [
        ("noise_inflation", "$b$", "log10_uniform", 0.5, 3.0, 1.0)]
    assert names[-1] == "noise_inflation"
    assert C.Config().noise_inflation_per_group is False


def test_per_group_layout():
    specs, names, kinds = _layout(_cfg(noise_inflation_per_group=True))
    noise = [s for s in specs if s.kind == "noise"]
    assert [s.name for s in noise] == [f"noise_inflation_{g}" for g in GROUPS]
    assert all((s.prior_type, s.lo, s.hi, s.truth) == ("log10_uniform", 0.5, 3.0, 1.0)
               for s in noise)
    # contiguous, last, after the offsets; one more dimension per extra group
    assert names[-len(GROUPS):] == [s.name for s in noise]
    assert names.index("offset_G395H_NRS2") < names.index("noise_inflation_NIRISS")
    assert len(specs) == len(_layout(_cfg())[0]) + len(GROUPS) - 1
    assert noise[1].label == r"$b_{\rm G395H\_NRS1}$"
    pytest.importorskip("matplotlib")                         # labels render in corner
    from matplotlib.mathtext import MathTextParser
    for s in noise:
        MathTextParser("path").parse(s.label)


def test_per_group_needs_the_groups():
    # the inference flag is validate_config's (test_run_nautilus)
    with pytest.raises(ValueError, match="groups"):
        C.specs_from_config(_cfg(noise_inflation_per_group=True))


def test_group_design_and_scale_per_bin():
    gpb = np.array(["NIRISS", "NIRISS", "G395H_NRS1", "G395H_NRS2", "G395H_NRS2"])
    N = OBS.build_group_design({"groups": GROUPS, "group": gpb})
    assert N.tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    with pytest.raises(ValueError, match="G395H_NRS2"):
        OBS.build_group_design({"groups": GROUPS[:2], "group": gpb})
    names = ["x"] + [f"noise_inflation_{g}" for g in GROUPS]
    th = np.array([[0.0, 1.21, 0.94, 1.13], [0.0, 2.0, 1.0, 0.5]])
    np.testing.assert_array_equal(OBS.noise_scale_per_bin(th, names, gpb),
                                  [[1.21, 1.21, 0.94, 1.13, 1.13], [2, 2, 1, .5, .5]])
    np.testing.assert_array_equal(OBS.noise_scale_per_bin(th[0], names, gpb),
                                  [1.21, 1.21, 0.94, 1.13, 1.13])
    np.testing.assert_array_equal(
        OBS.noise_scale_per_bin(np.array([0.0, 1.5]), ["x", "noise_inflation"], gpb),
        np.full(5, 1.5))
    np.testing.assert_array_equal(OBS.noise_scale_per_bin(np.zeros(1), ["x"], gpb),
                                  np.ones(5))
    with pytest.raises(ValueError, match="G395H_NRS2"):
        OBS.noise_scale_per_bin(th[0, :3], names[:3], gpb)
    with pytest.raises(ValueError, match="contiguous"):
        OBS.noise_map([names[1], "x", names[2], names[3]], gpb)


def _toy(n_bin=40, seed=0):
    rng = np.random.default_rng(seed)
    gpb = np.asarray(GROUPS)[rng.integers(0, len(GROUPS), n_bin)]
    gpb[:len(GROUPS)] = GROUPS                                # every group present
    return gpb, jnp.asarray(rng.uniform(50e-6, 150e-6, n_bin))


@pytest.mark.parametrize("per_group", [False, True])
def test_likelihood_sigma_is_the_predictive_inflation(per_group):
    """The likelihood's sigma_eff (make_noise_model) and the predictive band's
    inflation (observations.noise_scale_per_bin) are one map, in both modes."""
    gpb, sig0 = _toy()
    specs, names, _ = _layout(_cfg(noise_inflation_per_group=per_group))
    sigma_for = P.make_noise_model(names, gpb, lambda: sig0, jnp.float64)
    theta_from_u, _ = P.make_uspace(specs, jnp.float64)
    theta = theta_from_u(jnp.asarray(np.random.default_rng(1).normal(size=len(specs))))
    np.testing.assert_allclose(
        np.asarray(sigma_for(theta)),
        np.asarray(sig0) * OBS.noise_scale_per_bin(np.asarray(theta), names, gpb),
        rtol=1e-15)
    s = P.make_noise_model(["x"], gpb, lambda: sig0, jnp.float64)
    assert s(theta) is sig0                                    # no noise parameter


@pytest.mark.parametrize("names, b, want_b", [
    (["noise_inflation"], [2.5], [2.5, 2.5, 2.5]),                    # one b
    (["noise_inflation_A", "noise_inflation_B"], [1.5, 0.8], [1.5, 1.5, 0.8]),
])
def test_replicates_carry_the_likelihood_sigma(names, b, want_b):
    """The predictive band is replicated data, mu + N(0, (sigma b)^2) with the
    run's own inflation, centred on the latent model."""
    from retrieval_framework.nautilus_report import predictive_replicates
    rng, n = np.random.default_rng(0), 20000
    gpb = np.array(["A", "A", "B"])
    sigma = np.array([1.0e-4, 2.0e-4, 5.0e-4])
    infl = OBS.noise_scale_per_bin(np.tile(b, (n, 1)), names, gpb)
    mu = np.tile(np.array([1e-2, 2e-2, 3e-2]), (n, 1))
    y_rep, sig_eff = predictive_replicates(mu, sigma, infl, rng)
    want = sigma * np.asarray(want_b)
    np.testing.assert_allclose(sig_eff[0], want, rtol=1e-15)
    np.testing.assert_allclose(y_rep.std(axis=0), want, rtol=0.05)
    assert np.allclose(y_rep.mean(axis=0), mu[0], atol=4 * want / np.sqrt(n))


@pytest.mark.skipif(os.environ.get("RUN_PRODUCTION_PREFLIGHT") != "1",
                    reason="opt-in: builds the exact production forward")
def test_production_case_gets_one_inflation_per_group():
    from conftest import RUN_DIR
    from retrieval_framework import run_setup as R
    cfg = R.load_spec(RUN_DIR / "wasp39_test1.yaml").cfg      # per-group noise inflation is v5's
    pipe = P.build_pipeline(cfg)
    P.load_real_into_pipe(pipe)
    assert list(pipe.groups) == GROUPS
    idx = [i for i, k in enumerate(pipe.kinds) if k == "noise"]
    assert [pipe.names[i] for i in idx] == [f"noise_inflation_{g}" for g in GROUPS]
    assert pipe.n_dim == 17
