"""run_nautilus.make_loglike on a stub pipeline: every rejection class (T-P
window, count_max exhausted, stall-certified, non-finite) is log L = -inf and a
solve the solver certified at the cap is kept, so the stock sampler's evidence
is the zero-filled box integral (pinned against
the exact value on a 3-D Gaussian with three cut regions); with
anchors, each warm solve starts from the nearest certified column and only
certified columns become anchors. Padding and T-P-invalid rows are masked: a
short chunk's filler rows and its out-of-window rows reach the evaluator as
``active`` False. Last, every config refusal of validate_config."""
import json
import math
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import nautilus
import numpy as np
import pytest
from scipy.special import ndtr

jax.config.update("jax_enable_x64", True)

from retrieval_framework import config_schema as C
from retrieval_framework import pipeline as P
from retrieval_framework.run_nautilus import (DISCARD_EXPLORATION, F_LIVE, N_NETWORKS,
                                              Anchors, make_loglike, make_sampler)

D, MU, SD, COUNT_MAX = 3, 0.5, 0.1, 50
# log Z of the stub likelihood over the unit box with its three cut regions
EXACT_LOGZ = (math.log(ndtr(1.0) - ndtr(-5.0)) + math.log(ndtr(5.0) - ndtr(-1.5))
              + math.log(ndtr(2.0) - ndtr(-5.0)))


class _Stats(NamedTuple):
    acc: jnp.ndarray
    conv_ok: jnp.ndarray


def _masked(name, seen, active, L, acc, conv):
    """The pipeline's padding contract: an inactive row takes no step (acc 0,
    uncertified) and scores the reject sentinel. Records (name, active) per call."""
    if seen is not None:
        jax.debug.callback(lambda a: seen.append((name, np.asarray(a).tolist())), active,
                           ordered=True)
    return (jnp.where(active, L, P.REJECT_LOGL), jnp.where(active, acc, 0),
            active & conv)


def _stub_pipe(seen=None):
    specs = [C.ParamSpec(f"p{i}", f"p{i}", "uniform", 0.0, 1.0, MU, "chem") for i in range(D)]
    theta_from_u, _ = P.make_uspace(specs, jnp.float64)

    def common(U):
        th = jax.vmap(theta_from_u)(U)
        L = jnp.sum(-0.5 * ((th - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
        conv = (th[:, 1] >= 0.35) & (th[:, 0] <= 0.6)              # stalled below p1 0.35
        # at the cap: exhausted above p0 = 0.6 (a cap exit never certifies),
        # certified AT the cap in (0.55, 0.6] (kept, as the solver says)
        acc = jnp.where((th[:, 0] > 0.6) | (conv & (th[:, 0] > 0.55)), COUNT_MAX, 10)
        return th, jnp.where(th[:, 2] > 0.7, -jnp.inf, L), acc, conv   # T-P window

    def refs_of(th):   # (lnZ, c_o, s_o): the pipeline's three-wide references
        return jnp.concatenate([th[:, :2], jnp.zeros((th.shape[0], 1))], axis=1)

    def cold(U, Y, refs, active):  # the column is a code of the point's own chemistry
        th, L, acc, conv = common(U)
        L, acc, conv = _masked("cold", seen, active, L, acc, conv)
        return L, (th[:, 0] + 1000 * th[:, 1])[:, None, None], refs_of(th), _Stats(acc, conv)

    def warm(U, Y, refs, active):  # hands back the column it started from; like the
        _, L, acc, conv = common(U)   # pipeline's, it floors L on a stalled/exhausted exit
        L, acc, conv = _masked("warm", seen, active, L, acc, conv)
        L = jnp.where(conv, L, P.REJECT_LOGL)
        return L, Y, refs, _Stats(acc, conv)

    def make(mode):
        return cold if mode == "cold" else warm

    return SimpleNamespace(_make_batch_eval=make, n_dim=D, refs_of=refs_of,
                           theta_from_u=theta_from_u, tp_valid=lambda th: th[2] <= 0.7,
                           dtype=jnp.float64, n_chem_tp=2, y_baseline=jnp.zeros((1, 1)),
                           fwd=SimpleNamespace(chem=SimpleNamespace(count_max=COUNT_MAX)))


def test_rejection_classes(tmp_path):
    tally = {}
    like = make_loglike(_stub_pipe(), 64, tally, rejects=tmp_path / "rejected")
    z = np.array([[0.5, 0.5, 0.5], [0.65, 0.5, 0.5], [0.5, 0.3, 0.5], [0.5, 0.5, 0.75],
                  [0.58, 0.5, 0.5]])             # the last certifies at the cap
    L = like(z)                                  # 5 rows through one padded chunk of 64
    assert np.all(np.isfinite(L[[0, 4]])) and np.all(np.isneginf(L[1:4]))
    assert {k: tally[k] for k in ("n_eval", "tp_out", "exhausted", "stalled", "nonfinite")} == {
        "n_eval": 5, "tp_out": 1, "exhausted": 1, "stalled": 1, "nonfinite": 0}
    # the rejected draws are kept with their class: exhausted, stalled, T-P out (a
    # T-P-out row is masked, so its lane took no step)
    with np.load(tmp_path / "rejected" / "rejected_00000.npz") as d:
        assert np.array_equal(d["z"], z[1:4]) and d["cls"].tolist() == [1, 2, 0]
        assert d["acc"].tolist() == [COUNT_MAX, 10, 0] and int(d["warm"]) == 0
    like(z[:1])                                  # a chunk with nothing rejected writes nothing
    assert sorted(p.name for p in (tmp_path / "rejected").iterdir()) == ["rejected_00000.npz"]


def test_sampler_is_stock_nautilus_with_default_relu_networks(tmp_path):
    """run_nautilus builds a STOCK nautilus.Sampler (PyPI 1.0.6, not the former
    SiLU / spread_batch fork) with nautilus's default networks, discards the
    exploration phase, and on the stub pipeline gets the exact zero-filled evidence;
    its checkpoint resumes."""
    assert nautilus.__version__ == "1.0.6"
    assert not hasattr(nautilus.Sampler, "add_samples_spread")
    like = make_loglike(_stub_pipe(), 64, {})
    f = tmp_path / "ckpt.hdf5"
    s = make_sampler(like, D, n_live=300, n_batch=64, seed=0, filepath=str(f))
    assert type(s) is nautilus.Sampler
    assert s.n_networks == N_NETWORKS == 4 and s.neural_network_kwargs == {}
    assert DISCARD_EXPLORATION is True
    s.run(f_live=F_LIVE, n_eff=2000, discard_exploration=DISCARD_EXPLORATION, verbose=False)
    assert s.n_eff >= 2000 and abs(s.log_z - EXACT_LOGZ) < 0.05
    r = make_sampler(like, D, n_live=300, n_batch=64, seed=0, filepath=str(f), resume=True)
    assert r.n_like == s.n_like and r.explored and abs(r.log_z - s.log_z) < 1e-12


def test_warm_starts_from_the_nearest_certified_column(tmp_path):
    rng = np.random.default_rng(0)
    anchors, tally = Anchors(tmp_path, 2), {}
    like = make_loglike(_stub_pipe(), 16, tally, anchors, tmp_path / "batches.jsonl")
    z1 = rng.uniform(0.02, 0.98, (16, D))
    L1 = like(z1)                                # no anchors yet: cold
    assert len(anchors) == np.isfinite(L1).sum() > 0 and tally["n_cold"] == 16
    z_a, code_a = anchors.z.copy(), anchors.columns(np.arange(len(anchors)))[0][:, 0, 0]
    z2 = rng.uniform(0.02, 0.98, (16, D))
    L2 = like(z2)                                # warm from the first batch's anchors
    new = np.arange(len(z_a), len(anchors))
    assert tally["n_warm"] == 16 and len(new) == np.isfinite(L2).sum()
    # a stalled warm exit is tallied as stalled, not as a non-finite forward
    z = np.concatenate([z1, z2])                 # theta = z on the unit box
    live = (z[:, 2] <= 0.7) & (z[:, 0] <= 0.6)
    assert (live & (z[:, 1] < 0.35))[16:].any()
    assert tally["stalled"] == (live & (z[:, 1] < 0.35)).sum() and tally["nonfinite"] == 0
    want = code_a[((anchors.z[new, None, :] - z_a[None]) ** 2).sum(-1).argmin(1)]
    assert np.array_equal(anchors.columns(new)[0][:, 0, 0], want)
    assert len(Anchors(tmp_path, 2)) == len(anchors)   # reloads on resume
    rows = [json.loads(r) for r in (tmp_path / "batches.jsonl").read_text().splitlines()]
    assert [(r["mode"], len(r["acc"])) for r in rows] == [("cold", 16), ("warm", 16)]


VALID = dict(vulcan_cfg_name="W39b", rp_cm=1.0, rstar_cm=10.0)


@pytest.mark.parametrize("over, msg", [
    (dict(cold_lanes=0), "cold_lanes"),       # every batch runs on the lane queue
    (dict(cold_lanes=-1), "cold_lanes"),
    (dict(cold_refill_chunk=0), "cold_refill_chunk"),
    (dict(noise_inflation_per_group=True), "infer_noise_inflation"),
    (dict(cfg_overrides={"use_condense": True}), "use_condense"),
    (dict(tp_deep_param="T_1bar", tp_infer_Tint=True), "tp_deep_param"),
    (dict(tp_deep_param="T1bar"), "tp_infer_Tint"),       # a fixed Tint would ignore it
    (dict(tp_deep_p_bar=0.0, tp_infer_Tint=True), "tp_deep_p_bar"),
    (dict(kzz_mode="const"), "kzz_mode"),
    (dict(cloud_model="patchy"), "cloud_model"),
    (dict(infer_lnM=True, prior_lnM_sigma=0.0), "prior_lnM_sigma"),
    # accepted: condensation with its explicit opt-in, and the v5 model choices
    (dict(cfg_overrides={"use_condense": True}, allow_condense_inference=True), None),
    (dict(tp_infer_Tint=True, tp_deep_param="T1bar", tp_adiabat=True, infer_s_o=True,
          kzz_mode="powerlaw", infer_lnM=True, cloud_model="deck"), None),
])
def test_validate_config_refusals(over, msg):
    """Every refusal a case preset or an override file can hit (msg None:
    accepted)."""
    cfg = C.Config(**{**VALID, **over})
    if msg is None:
        C.validate_config(cfg)
    else:
        with pytest.raises(ValueError, match=msg):
            C.validate_config(cfg)
