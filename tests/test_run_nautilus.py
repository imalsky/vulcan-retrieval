"""run_nautilus.make_loglike on a stub pipeline: every rejection class the SMC
init culls (T-P window, count_max exhausted, stall-certified, non-finite) is
log L = -inf, so nautilus's evidence is the zero-filled box integral (pinned
against the exact value on a 3-D Gaussian with three cut regions); with
anchors, each warm solve starts from the nearest certified column and only
certified columns become anchors."""
import math
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import nautilus
import numpy as np
from scipy.special import ndtr

jax.config.update("jax_enable_x64", True)

from retrieval_framework import config_schema as C
from retrieval_framework import pipeline as P
from retrieval_framework.run_nautilus import Anchors, make_loglike

D, MU, SD, COUNT_MAX = 3, 0.5, 0.1, 50


class _Diag(NamedTuple):
    accept_count: jnp.ndarray
    conv_normal: jnp.ndarray


class _Stats(NamedTuple):
    acc: jnp.ndarray
    conv_ok: jnp.ndarray


def _stub_pipe():
    specs = [C.ParamSpec(f"p{i}", f"p{i}", "uniform", 0.0, 1.0, MU, "chem") for i in range(D)]
    theta_from_u, _, _ = P.make_uspace(specs, jnp.float64)

    def common(U):
        th = jax.vmap(theta_from_u)(U)
        L = jnp.sum(-0.5 * ((th - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
        acc = jnp.where(th[:, 0] > 0.6, COUNT_MAX, 10)            # exhausted
        conv = th[:, 1] >= 0.35                                     # stalled below
        return th, jnp.where(th[:, 2] > 0.7, -jnp.inf, L), acc, conv   # T-P window

    def cold(U, Y, refs):          # the column is a code of the point's own chemistry
        th, L, acc, conv = common(U)
        return L, (th[:, 0] + 1000 * th[:, 1])[:, None, None], th[:, :2], _Diag(acc, conv)

    def warm(U, Y, refs):          # hands back the column it started from
        _, L, acc, conv = common(U)
        return L, Y, refs, _Stats(acc, conv)

    return SimpleNamespace(batch_eval_cold_l_diag=cold,
                           _make_batch_eval=lambda mode, grad, mutation_cap: warm,
                           theta_from_u=theta_from_u, tp_valid=lambda th: th[2] <= 0.7,
                           dtype=jnp.float64, n_chem_tp=2, has_chem_state=False,
                           fwd=SimpleNamespace(chem=SimpleNamespace(count_max=COUNT_MAX)))


def test_zero_filled_evidence_and_rejection_classes():
    tally = {}
    like = make_loglike(_stub_pipe(), 64, tally)
    z = np.array([[0.5, 0.5, 0.5], [0.65, 0.5, 0.5], [0.5, 0.3, 0.5], [0.5, 0.5, 0.75]])
    L = like(z)                                  # 4 rows through one padded chunk of 64
    assert np.isfinite(L[0]) and np.all(np.isneginf(L[1:]))
    assert {k: tally[k] for k in ("n_eval", "tp_out", "exhausted", "stalled", "nonfinite")} == {
        "n_eval": 4, "tp_out": 1, "exhausted": 1, "stalled": 1, "nonfinite": 0}

    s = nautilus.Sampler(lambda u: u, like, n_dim=D, n_live=300, n_batch=64,
                         vectorized=True, pass_dict=False, seed=0)
    s.run(discard_exploration=True, verbose=False)
    exact = (math.log(ndtr(1.0) - ndtr(-5.0)) + math.log(ndtr(5.0) - ndtr(-1.5))
             + math.log(ndtr(2.0) - ndtr(-5.0)))
    assert abs(s.log_z - exact) < 0.05


def test_warm_starts_from_the_nearest_certified_column(tmp_path):
    rng = np.random.default_rng(0)
    anchors, tally = Anchors(tmp_path, 2), {}
    like = make_loglike(_stub_pipe(), 16, tally, anchors)
    z1 = rng.uniform(0.02, 0.98, (16, D))
    L1 = like(z1)                                # no anchors yet: cold
    assert len(anchors) == np.isfinite(L1).sum() > 0 and tally["n_cold"] == 16
    z_a, code_a = anchors.z.copy(), anchors.columns(np.arange(len(anchors)))[0][:, 0, 0]
    L2 = like(rng.uniform(0.02, 0.98, (16, D)))  # warm from the first batch's anchors
    new = np.arange(len(z_a), len(anchors))
    assert tally["n_warm"] == 16 and len(new) == np.isfinite(L2).sum()
    want = code_a[((anchors.z[new, None, :] - z_a[None]) ** 2).sum(-1).argmin(1)]
    assert np.array_equal(anchors.columns(new)[0][:, 0, 0], want)
    assert len(Anchors(tmp_path, 2)) == len(anchors)   # reloads on resume
