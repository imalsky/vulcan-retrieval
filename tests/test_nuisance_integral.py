"""The integrated nuisances (cfg.integrate_nuisance, pipeline.make_nuisance_marginal):
the marginal likelihood equals a brute-force integral of the sampled model over the
offsets and the inflation(s) in every mode, tails included; the conditional draws
have the brute-force posterior's moments; the layout drops exactly the five
nuisance parameters; and run_nautilus logs each certified row's statistics and joins
them back to the posterior by (z, log L). Last, on the real smoke pipeline (slow), the
batched evaluator's log L is the marginal of its own spectrum."""
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import observations as OBS  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402
from retrieval_framework.run_nautilus import make_loglike, nuisance_posterior  # noqa: E402

GROUPS = ["NIRISS", "G395H_NRS1", "G395H_NRS2"]
SIZES = (91, 30, 29)                                  # v5's bins per group
BASE = C.Config(molecules=("H2O",), vulcan_cfg_name="W39b",
                network="thermo/SNCHO_photo_network_2025.txt", infer_noise_inflation=True,
                noise_inflation_per_group=True, integrate_nuisance=True)


def _data(case, rng):
    grp = np.repeat(GROUPS, SIZES)
    s0 = rng.uniform(80e-6, 300e-6, grp.size)
    if case == "realistic":                      # inflated noise, offsets inside the prior
        r = 1.3 * s0 * rng.standard_normal(grp.size) + np.select(
            [grp == GROUPS[1], grp == GROUPS[2]], [150e-6, -60e-6], 0.0)
    else:                                        # overfit (b on its floor), an offset past its edge
        r = 0.3 * s0 * rng.standard_normal(grp.size) - 950e-6 * (grp == GROUPS[1])
    mu = np.full(grp.size, 0.02)
    return grp, s0, mu + r, mu


def _brute(cfg, grp, s0, d, mu, n_t=4001, n_o=8001):
    """log of the sampled likelihood integrated over the offset and inflation priors
    on dense trapezoid grids, from the bins (no closed form)."""
    infl, per = cfg.infer_noise_inflation, cfg.noise_inflation_per_group
    t = np.linspace(*np.log10(cfg.prior_noise_inflation), n_t) if infl else np.zeros(1)
    wt = np.full(t.size, 1.0 / max(t.size - 1, 1))
    wt[[0, -1]] *= 0.5 if infl else 1.0
    lo, hi = (v * OBS.PPM for v in cfg.prior_offset_ppm)
    o = np.linspace(lo, hi, n_o)
    wo = np.full(n_o, (hi - lo) / (n_o - 1) / (hi - lo))
    wo[[0, -1]] *= 0.5
    per_g = []                                   # (G, n_t): each group integrated over o
    for k, g in enumerate(GROUPS):
        m = grp == g
        r, s = d[m] - mu[m], s0[m]
        rows = []
        for b in 10.0 ** t:
            if cfg.infer_offsets and k > 0:
                z = (r[None, :] - o[:, None]) / (s * b)
                ll = -0.5 * (z * z).sum(1) - np.log(s * b).sum() - 0.5 * r.size * np.log(2 * np.pi)
                rows.append(logsumexp(ll + np.log(wo)))
            else:
                z = r / (s * b)
                rows.append(-0.5 * (z * z).sum() - np.log(s * b).sum() - 0.5 * r.size * np.log(2 * np.pi))
        per_g.append(np.asarray(rows))
    per_g = np.asarray(per_g)
    if per or not infl:
        return float(sum(logsumexp(x + np.log(wt)) for x in per_g))
    return float(logsumexp(per_g.sum(0) + np.log(wt)))


@pytest.mark.parametrize("case", ["realistic", "edges"])
@pytest.mark.parametrize("mode", [dict(), dict(noise_inflation_per_group=False),
                                  dict(infer_noise_inflation=False, noise_inflation_per_group=False),
                                  dict(infer_offsets=False)])
def test_marginal_matches_brute_force(case, mode):
    cfg = replace(BASE, **mode)
    grp, s0, d, mu = _data(case, np.random.default_rng(0))
    nm = P.make_nuisance_marginal(cfg, GROUPS, grp)
    st = nm.stats(jnp.asarray(mu), jnp.asarray(d), jnp.asarray(s0))
    got = float(nm.log_marginal(st, jnp.asarray(s0)))
    tol = 1e-4 if case == "realistic" else 2e-3     # the floor case is the quadrature's worst
    assert abs(got - _brute(cfg, grp, s0, d, mu)) < tol


def test_layout_and_draws():
    grp, s0, d, mu = _data("realistic", np.random.default_rng(1))
    nm = P.make_nuisance_marginal(BASE, GROUPS, grp)
    assert nm.names == ["offset_G395H_NRS1", "offset_G395H_NRS2", "noise_inflation_NIRISS",
                        "noise_inflation_G395H_NRS1", "noise_inflation_G395H_NRS2"]
    sampled = [s.name for s in C.specs_from_config(replace(BASE, integrate_nuisance=False), GROUPS)]
    kept = [s.name for s in C.specs_from_config(BASE, GROUPS)]
    assert sampled == kept + nm.names                # exactly the five leave the layout
    # the draws against the brute-force conditional moments of NRS1's (o, b)
    st = np.asarray(nm.stats(jnp.asarray(mu), jnp.asarray(d), jnp.asarray(s0)))
    n = 40000
    dr = nm.draw(np.tile(st, (n, 1)), s0, np.random.default_rng(2))
    m = grp == GROUPS[1]
    r, s = d[m] - mu[m], s0[m]
    t = np.linspace(np.log10(0.5), np.log10(3.0), 1501)
    o = np.linspace(-800e-6, 800e-6, 4001)
    z = (r[None, None, :] - o[None, :, None]) / (s[None, None, :] * 10.0 ** t[:, None, None])
    ll = -0.5 * (z * z).sum(-1) - r.size * t[:, None] * np.log(10.0)
    p = np.exp(ll - ll.max())
    p /= p.sum()
    for j, x in ((0, np.broadcast_to(o[None, :] / OBS.PPM, p.shape)),
                 (3, np.broadcast_to(10.0 ** t[:, None], p.shape))):
        mean, sd = (p * x).sum(), np.sqrt((p * (x - (p * x).sum()) ** 2).sum())
        assert abs(dr[:, j].mean() - mean) < 4 * sd / np.sqrt(n)
        assert abs(dr[:, j].std() / sd - 1) < 0.03
    # apply: each bin gets its group's offset once and its group's inflation
    mu2, mult = nm.apply(mu[None], dr[:1])
    np.testing.assert_allclose(mu2[0] - mu, dr[0, 0] * OBS.PPM * m + dr[0, 1] * OBS.PPM * (grp == GROUPS[2]))
    np.testing.assert_allclose(mult[0], np.select([grp == g for g in GROUPS], dr[0, 2:]))


@pytest.mark.parametrize("over, msg", [
    (dict(infer_offsets=False, infer_noise_inflation=False, noise_inflation_per_group=False),
     "nothing to integrate"),
    (dict(prior_noise_inflation=(0.0, 3.0)), "0 < lo < hi"),
    (dict(prior_offset_ppm=(800.0, -800.0)), "prior_offset_ppm")])
def test_refusals(over, msg):
    with pytest.raises(ValueError, match=msg):
        C.validate_config(replace(BASE, **over))


class _Stats(NamedTuple):
    acc: Any
    conv_ok: Any
    nuis: Any


def test_run_logs_and_joins_the_statistics(tmp_path):
    """Certified rows' statistics are logged with the log L nautilus sees; the join
    takes the entry whose log L matches (a row logged twice after a resume), NaN on
    a rejected row, and raises on a finite row it cannot find."""
    def cold(U, Y, refs, active):
        z = jax.nn.sigmoid(U)
        L = jnp.where(z[:, 0] < 0.5, -jnp.sum(z, axis=1), P.REJECT_LOGL)
        st = jnp.stack([z[:, 0], 2 * z[:, 0]], axis=1)                # G = 1
        return L, Y, refs, _Stats(jnp.full(z.shape[0], 5), z[:, 0] < 0.5, st)

    nuis = SimpleNamespace(groups=["A"], names=["noise_inflation"],
                           draw=lambda st, sigma0, rng: st[:, :1] * 10)
    pipe = SimpleNamespace(_make_batch_eval=lambda mode: cold, n_dim=2, dtype=jnp.float64,
                           theta_from_u=lambda u: u, tp_valid=lambda th: True, n_chem_tp=2,
                           y_baseline=jnp.zeros((1, 1)), nuisance=nuis, obs_sigma=None,
                           fwd=SimpleNamespace(chem=SimpleNamespace(count_max=10)))
    like = make_loglike(pipe, 8, {}, nuisance=tmp_path / "nuisance")
    z = np.array([[0.2, 0.3], [0.7, 0.1], [0.4, 0.4]])
    L = like(z)
    assert np.isneginf(L[1]) and np.isfinite(L[[0, 2]]).all()
    with np.load(tmp_path / "nuisance" / "nuisance_00000.npz") as d:
        assert np.array_equal(d["z"], z[[0, 2]]) and np.array_equal(d["log_l"], L[[0, 2]])
    # a resume's replay logs row 0 again with another log L: the join takes the match
    np.savez(tmp_path / "nuisance" / "nuisance_00001.npz", z=z[:1], log_l=L[:1] - 1.0,
             stats=np.array([[9.0, 9.0]]))
    (nu,) = nuisance_posterior(pipe, tmp_path / "nuisance", [(z, L)], None)
    np.testing.assert_allclose(nu[[0, 2], 0], 10 * z[[0, 2], 0])
    assert np.isnan(nu[1]).all()
    with pytest.raises(RuntimeError, match="no logged nuisance statistics"):
        nuisance_posterior(pipe, tmp_path / "nuisance", [(z[:1] + 0.01, L[:1])], None)


@pytest.mark.slow
def test_smoke_pipeline_scores_the_marginal_of_its_own_spectrum():
    """Smoke (one group, no offsets) with one integrated inflation: the batched
    evaluator's log L is the marginal of the spectrum its RT produced, and that
    marginal is the brute-force integral of the sampled likelihood over b."""
    from conftest import build_smoke_pipe
    pipe = build_smoke_pipe(infer_noise_inflation=True, integrate_nuisance=True)
    assert pipe.nuisance.names == ["noise_inflation"] and "noise_inflation" not in pipe.names
    U = pipe.sample_prior_u(jax.random.PRNGKey(0), 2)
    Y0, refs0 = P._blank_state(pipe, 2)
    L, Y, _, st = jax.jit(pipe._make_batch_eval("cold"))(U, Y0, refs0, np.ones(2, bool))
    keep = np.asarray(L) > P.REJECT_BELOW
    assert keep.any()
    th = jax.vmap(pipe.theta_from_u)(U)
    mu = np.asarray(pipe.observed_depth_from_y_jit(Y, th))
    d, s0 = pipe.obs_depth, pipe.obs_sigma
    for i in np.flatnonzero(keep):
        np.testing.assert_allclose(np.asarray(st.nuis)[i], np.asarray(
            pipe.nuisance.stats(jnp.asarray(mu[i]), jnp.asarray(d), jnp.asarray(s0))), rtol=1e-10)
        t = np.linspace(np.log10(0.5), np.log10(3.0), 20001)
        z = (d - mu[i])[None, :] / (s0[None, :] * 10.0 ** t[:, None])
        ll = -0.5 * (z * z).sum(1) - np.log(s0[None, :] * 10.0 ** t[:, None]).sum(1) \
            - 0.5 * d.size * np.log(2 * np.pi)
        w = np.full(t.size, 1.0 / (t.size - 1))
        w[[0, -1]] *= 0.5
        assert abs(float(L[i]) - logsumexp(ll + np.log(w))) < 1e-4
