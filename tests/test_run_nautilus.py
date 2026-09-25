"""run_nautilus.make_loglike on a stub pipeline: every rejection class the SMC
init culls (T-P window, count_max exhausted, stall-certified, non-finite) is
log L = -inf, so nautilus's evidence is the zero-filled box integral. Pinned
against the exact value on a 3-D Gaussian with three cut regions."""
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
from retrieval_framework.run_nautilus import make_loglike

D, MU, SD, COUNT_MAX = 3, 0.5, 0.1, 50


class _Diag(NamedTuple):
    accept_count: jnp.ndarray
    conv_normal: jnp.ndarray


def _stub_pipe():
    specs = [C.ParamSpec(f"p{i}", f"p{i}", "uniform", 0.0, 1.0, MU, "chem") for i in range(D)]
    theta_from_u, _, _ = P.make_uspace(specs, jnp.float64)

    def ev(U, Y, refs):
        th = jax.vmap(theta_from_u)(U)
        L = jnp.sum(-0.5 * ((th - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
        acc = jnp.where(th[:, 0] > 0.6, COUNT_MAX, 10)            # exhausted
        conv = th[:, 1] >= 0.35                                     # stalled below
        L = jnp.where(th[:, 2] > 0.7, -jnp.inf, L)                  # T-P window
        return L, Y, refs, _Diag(acc, conv)

    return SimpleNamespace(batch_eval_cold_l_diag=ev, theta_from_u=theta_from_u,
                           tp_valid=lambda th: th[2] <= 0.7, dtype=jnp.float64,
                           fwd=SimpleNamespace(chem=SimpleNamespace(count_max=COUNT_MAX)),
                           has_chem_state=False)


def test_zero_filled_evidence_and_rejection_classes():
    tally = {}
    like = make_loglike(_stub_pipe(), 64, tally)
    z = np.array([[0.5, 0.5, 0.5], [0.65, 0.5, 0.5], [0.5, 0.3, 0.5], [0.5, 0.5, 0.75]])
    L = like(z)                                  # 4 rows through one padded chunk of 64
    assert np.isfinite(L[0]) and np.all(np.isneginf(L[1:]))
    assert tally == {"n_eval": 4, "tp_out": 1, "exhausted": 1, "stalled": 1, "nonfinite": 0}

    s = nautilus.Sampler(lambda u: u, like, n_dim=D, n_live=300, n_batch=64,
                         vectorized=True, pass_dict=False, seed=0)
    s.run(discard_exploration=True, verbose=False)
    exact = (math.log(ndtr(1.0) - ndtr(-5.0)) + math.log(ndtr(5.0) - ndtr(-1.5))
             + math.log(ndtr(2.0) - ndtr(-5.0)))
    assert abs(s.log_z - exact) < 0.05
