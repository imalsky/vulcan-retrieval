"""Per-group noise inflation (cfg.noise_inflation_per_group; off = the one b).

Stack-free:
* layout: off keeps the scalar 'noise_inflation' exactly; on gives one
  'noise_inflation_<group>' per observation group, in group order, after the
  offsets, each log10-uniform on prior_noise_inflation; it needs the groups and
  infer_noise_inflation;
* observations.build_group_design / noise_scale_per_bin in all three modes;
* pipeline.make_noise_model, the one formula BOTH gradient paths call: its
  analytic d logL / d b equals jax.grad of the Gaussian log-likelihood (theta
  and u space), and its scalar branch is the pre-patch expression bit for bit;
* build_pipeline's two gradient paths route the noise entries through it;
* run_smc.predictive_replicates with a per-bin inflation.

Slow (real smoke pipeline, one synthetic group): the batched RT stage's noise
gradient (_rt_val_grad) equals the forward-mode derivative of its own value.
Opt-in production preflight (RUN_PRODUCTION_PREFLIGHT=1): the real three-group
spectrum gets three inflations, in the pipeline's group order, with the same
check.
"""
import ast
import inspect
import os
from pathlib import Path

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


def test_per_group_needs_groups_and_the_inference_flag():
    with pytest.raises(ValueError, match="groups"):
        C.specs_from_config(_cfg(noise_inflation_per_group=True))
    with pytest.raises(ValueError, match="infer_noise_inflation"):
        C.validate_config(C.Config(noise_inflation_per_group=True))


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


def _toy(n_bin=40, seed=0):
    rng = np.random.default_rng(seed)
    gpb = np.asarray(GROUPS)[rng.integers(0, len(GROUPS), n_bin)]
    gpb[:len(GROUPS)] = GROUPS                                # every group present
    sig0 = rng.uniform(50e-6, 150e-6, n_bin)
    resid = rng.normal(size=n_bin) * sig0 * 1.3
    return gpb, jnp.asarray(sig0), jnp.asarray(resid)


def _gauss(resid, sig):
    """pipeline._gauss_loglik's formula on a residual."""
    return (-0.5 * jnp.sum((resid / sig) ** 2) - jnp.sum(jnp.log(sig))
            - 0.5 * resid.size * jnp.log(2.0 * jnp.pi))


@pytest.mark.parametrize("per_group", [False, True])
def test_noise_gradient_is_the_exact_derivative(per_group):
    gpb, sig0, resid = _toy()
    specs, names, kinds = _layout(_cfg(noise_inflation_per_group=per_group))
    sigma_for, add_grad = P.make_noise_model(names, kinds, GROUPS, gpb, lambda: sig0,
                                             jnp.float64)
    noise = np.array([k == "noise" for k in kinds])
    theta_from_u, _, _ = P.make_uspace(specs, jnp.float64)
    u = jnp.asarray(np.random.default_rng(1).normal(size=len(specs)))
    theta = theta_from_u(u)

    # the two terms of dL/db nearly cancel near b ~ |r|/sigma0, so the
    # tolerance is absolute, on the scale of the terms (n_bin / b)
    atol = 1e-10 * resid.size / 0.5
    g_ad = np.asarray(jax.grad(lambda th: _gauss(resid, sigma_for(th)))(theta))
    g_an = np.asarray(add_grad(jnp.zeros(len(specs)), resid, theta))
    np.testing.assert_allclose(g_an[noise], g_ad[noise], rtol=1e-12, atol=atol)
    assert np.all(g_an[~noise] == 0.0)                        # writes only its entries
    # and through the u-space chain the MALA drift uses (log10-uniform Jacobian)
    _, dth_du = jax.jvp(theta_from_u, (u,), (jnp.ones_like(u),))
    gu_ad = np.asarray(jax.grad(lambda uu: _gauss(resid, sigma_for(theta_from_u(uu))))(u))
    np.testing.assert_allclose((g_an * np.asarray(dth_du))[noise], gu_ad[noise],
                               rtol=1e-12, atol=atol)
    if per_group:
        # each b sees only its own group's bins
        b = np.asarray(theta)[noise]
        s0, r = np.asarray(sig0), np.asarray(resid)
        for j, g in enumerate(GROUPS):
            m = gpb == g
            want = np.sum(r[m] ** 2 / (s0[m] ** 2 * b[j] ** 3)) - m.sum() / b[j]
            assert g_an[noise][j] == pytest.approx(want, rel=1e-12, abs=atol)


def test_scalar_branch_is_the_pre_patch_expression():
    gpb, sig0, resid = _toy()
    sigma_for, add_grad = P.make_noise_model(["x", "noise_inflation"], ["chem", "noise"],
                                             GROUPS, gpb, lambda: sig0, jnp.float64)
    theta = jnp.asarray([0.3, 1.37])

    def old(theta):          # pipeline.py at 40339fd, _sigma_for and _rt_val_grad
        k = theta[1]
        g = jnp.zeros(2).at[1].set(jnp.sum(resid * resid / (sig0 * sig0 * k ** 3))
                                   - resid.size / k)
        return sig0 * theta[1], g

    def new(theta):
        return sigma_for(theta), add_grad(jnp.zeros(2), resid, theta)

    for f_old, f_new in ((old, new), (jax.jit(old), jax.jit(new))):
        (s_o, g_o), (s_n, g_n) = f_old(theta), f_new(theta)
        assert np.array_equal(np.asarray(s_o), np.asarray(s_n))
        assert np.array_equal(np.asarray(g_o), np.asarray(g_n))
    s, a = P.make_noise_model(["x"], ["chem"], GROUPS, gpb, lambda: sig0, jnp.float64)
    assert s(theta) is sig0 and a("g", resid, theta) == "g"   # no noise parameter


def test_both_gradient_paths_route_the_noise_entries_through_one_function():
    tree = ast.parse(Path(inspect.getfile(P)).read_text())
    build = next(n for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name == "build_pipeline")
    fns = {n.name: n for n in ast.walk(build) if isinstance(n, ast.FunctionDef)}
    for name in ("_value_and_grad_block", "_rt_val_grad", "_gauss_loglik"):
        calls = {c.func.id for c in ast.walk(fns[name])
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        assert "_sigma_for" in calls, name
        if name != "_gauss_loglik":
            assert "_add_noise_grad" in calls, name
    assert "noise_idx" not in ast.unparse(build)               # no second, hand-rolled path


def test_replicates_with_a_per_group_inflation():
    from retrieval_framework.run_smc import predictive_replicates
    rng = np.random.default_rng(0)
    gpb = np.array(["A", "A", "B"])
    sigma = np.array([1.0e-4, 2.0e-4, 5.0e-4])
    th = np.tile([1.5, 0.8], (20000, 1))
    infl = OBS.noise_scale_per_bin(th, ["noise_inflation_A", "noise_inflation_B"], gpb)
    mu = np.tile(np.array([1e-2, 2e-2, 3e-2]), (20000, 1))
    y_rep, sig_eff = predictive_replicates(mu, sigma, infl, rng)
    want = sigma * np.array([1.5, 1.5, 0.8])
    np.testing.assert_allclose(sig_eff[0], want, rtol=1e-15)
    np.testing.assert_allclose(y_rep.std(axis=0), want, rtol=0.05)


def _rt_stage_noise_check(pipe):
    """The batched RT stage's noise entries against forward-mode AD of its own
    value, at the baseline column (RT only, no chemistry solve)."""
    idx = [i for i, k in enumerate(pipe.kinds) if k == "noise"]
    assert idx
    th = pipe.theta_from_u(jnp.asarray(np.linspace(-0.35, 0.4, pipe.n_dim)))
    aux = pipe.fwd.aux_from_y(jnp.asarray(pipe.y_baseline), th[:pipe.n_chem_tp])
    daux = jax.tree_util.tree_map(
        lambda x: jnp.zeros((pipe.n_chem_tp,) + x.shape, x.dtype), aux)
    val, g, _bad = pipe._rt_val_grad((aux, daux, th))
    assert float(val) > P.REJECT_BELOW
    for i in idx:
        e = jnp.zeros_like(th).at[i].set(1.0)
        _, dv = jax.jvp(lambda t: pipe._rt_val_grad((aux, daux, t))[0], (th,), (e,))
        assert float(g[i]) == pytest.approx(float(dv), rel=1e-9), pipe.names[i]
    return idx


@pytest.mark.slow
def test_smoke_pipeline_rt_stage_noise_gradient():
    from conftest import build_smoke_pipe
    pipe = build_smoke_pipe(infer_noise_inflation=True, noise_inflation_per_group=True)
    idx = _rt_stage_noise_check(pipe)
    assert [pipe.names[i] for i in idx] == [f"noise_inflation_{g}" for g in pipe.groups]


@pytest.mark.skipif(os.environ.get("RUN_PRODUCTION_PREFLIGHT") != "1",
                    reason="opt-in: builds the exact production forward")
def test_production_case_gets_one_inflation_per_group(monkeypatch):
    from dataclasses import replace

    from conftest import RUN_DIR
    from retrieval_framework import run_smc as R
    monkeypatch.setenv("SMC_RETRIEVAL_PRESET", "gpu")
    cfg, _ = R.make_config(RUN_DIR)
    pipe = P.build_pipeline(replace(cfg, noise_inflation_per_group=True))
    P.load_real_into_pipe(pipe)
    assert list(pipe.groups) == GROUPS
    idx = _rt_stage_noise_check(pipe)
    assert [pipe.names[i] for i in idx] == [f"noise_inflation_{g}" for g in GROUPS]
    assert pipe.n_dim == 15
