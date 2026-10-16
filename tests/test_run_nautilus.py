"""run_nautilus.make_loglike on a stub pipeline: every rejection class the SMC
init culls (T-P window, count_max exhausted, stall-certified, non-finite) is
log L = -inf, so nautilus's evidence is the zero-filled box integral (pinned
against the exact value on a 3-D Gaussian with three cut regions); with
anchors, each warm solve starts from the nearest certified column and only
certified columns become anchors. With chem_retry, exactly the in-window rows
the first solve left uncertified are re-solved once, from the same start, on
the retry evaluator; a row is kept only if that solve certifies, and the
tally, the batch log and the anchors' rtol record the second pass. Off, none
of that appears."""
import json
import math
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import nautilus
import numpy as np
from scipy.special import ndtr

jax.config.update("jax_enable_x64", True)

from conftest import StubConvDiag
from retrieval_framework import config_schema as C
from retrieval_framework import pipeline as P
from retrieval_framework.run_nautilus import Anchors, make_loglike, retry_width

D, MU, SD, COUNT_MAX = 3, 0.5, 0.1, 50


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
        return L, (th[:, 0] + 1000 * th[:, 1])[:, None, None], th[:, :2], StubConvDiag(acc, conv)

    def warm(U, Y, refs):          # hands back the column it started from; like the
        _, L, acc, conv = common(U)   # pipeline's, it floors L on a stalled/exhausted exit
        L = jnp.where(conv & (acc < COUNT_MAX), L, P.REJECT_LOGL)
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


# ---- chem_retry ---------------------------------------------------------------
# First solve as _stub_pipe (exhausted above p0 = 0.6, stalled below p1 = 0.35,
# T-P out above p2 = 0.7). The retry program caps at RETRY_MAX: it exhausts above
# p0 = 0.8 and stalls below p1 = 0.2; a row it certifies scores L + DL_RETRY and
# hands back the column it started from (cold: the column code + 0.5).
RETRY_MAX, DL_RETRY, RTOL0, RTOL1 = 80, 0.5, 0.2, 0.05


def _retry_stub_pipe():
    base = _stub_pipe()
    theta_from_u = base.theta_from_u

    def second(U):
        th = jax.vmap(theta_from_u)(U)
        L = jnp.sum(-0.5 * ((th - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
        acc = jnp.where(th[:, 0] > 0.8, RETRY_MAX, 60)
        conv = th[:, 1] >= 0.2
        return th, jnp.where(th[:, 2] > 0.7, -jnp.inf, L + DL_RETRY), acc, conv

    def cold2(U, Y, refs):
        th, L, acc, conv = second(U)
        code = (th[:, 0] + 1000 * th[:, 1] + 0.5)[:, None, None]
        return L, code, th[:, :2], StubConvDiag(acc, conv)

    def warm2(U, Y, refs):
        _, L, acc, conv = second(U)
        L = jnp.where(conv & (acc < RETRY_MAX), L, P.REJECT_LOGL)
        return L, Y, refs, _Stats(acc, conv)

    evals = {("warm", False): base._make_batch_eval("warm", False, False),
             ("cold", True): cold2, ("warm", True): warm2}
    base._make_batch_eval = (lambda mode, grad, diag=False, mutation_cap=True, retry=False:
                             evals[(mode, retry)])
    base.chem_retry = SimpleNamespace(rtol_first=RTOL0, count_max_first=COUNT_MAX,
                                      rtol=RTOL1, count_max=RETRY_MAX)
    return base


def _expected(z):
    """Per row (theta = z): first-pass failure, final kept, final classes, L."""
    tp_out = z[:, 2] > 0.7
    fail1 = ~tp_out & ((z[:, 0] > 0.6) | (z[:, 1] < 0.35))
    exh2, conv2 = z[:, 0] > 0.8, z[:, 1] >= 0.2
    kept = ~tp_out & (~fail1 | (~exh2 & conv2))
    L = np.sum(-0.5 * ((z - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
    L = np.where(kept, L + np.where(fail1, DL_RETRY, 0.0), -np.inf)
    return dict(tp_out=tp_out, fail1=fail1, kept=kept, L=L,
                exhausted=fail1 & exh2, stalled=fail1 & ~exh2 & ~conv2)


def test_retry_width_pads_to_the_bucket():
    assert [retry_width(n) for n in (0, 1, 31, 32, 33, 64, 65, 396)] == [
        0, 32, 32, 32, 64, 64, 96, 416]


def test_retry_resolves_exactly_the_uncertified_rows_from_their_start(tmp_path):
    rng = np.random.default_rng(3)
    anchors, tally, log = Anchors(tmp_path, 2), {}, tmp_path / "batches.jsonl"
    like = make_loglike(_retry_stub_pipe(), 16, tally, anchors, log)
    z1 = rng.uniform(0.02, 0.98, (16, D))
    # exhausted -> kept; exhausted -> exhausted again; stalled -> kept; stalled -> stalled
    z1[:4] = [[0.7, 0.5, 0.3], [0.9, 0.5, 0.3], [0.5, 0.3, 0.3], [0.5, 0.1, 0.3]]
    L1 = like(z1)                                  # cold, both passes
    e1 = _expected(z1)
    assert e1["fail1"][:4].all() and e1["kept"][[0, 2]].all() and not e1["kept"][[1, 3]].any()
    np.testing.assert_allclose(L1[e1["kept"]], e1["L"][e1["kept"]], rtol=0, atol=1e-12)
    assert np.all(np.isneginf(L1[~e1["kept"]]))
    z_a, code_a = anchors.z.copy(), anchors.columns(np.arange(len(anchors)))[0][:, 0, 0]
    with np.load(tmp_path / "anchors_00000.npz") as d:
        assert np.array_equal(d["rtol"], np.where(e1["fail1"], RTOL1, RTOL0)[e1["kept"]])
    # a cold retried row's column is the RETRY program's (code + 0.5)
    assert np.allclose(code_a, z1[e1["kept"], 0] + 1000 * z1[e1["kept"], 1]
                       + np.where(e1["fail1"], 0.5, 0.0)[e1["kept"]])

    z2 = rng.uniform(0.02, 0.98, (16, D))
    z2[:3] = [[0.75, 0.5, 0.3], [0.5, 0.25, 0.3], [0.95, 0.5, 0.3]]
    L2 = like(z2)                                  # warm, both passes
    e2 = _expected(z2)
    assert e2["kept"][[0, 1]].all() and e2["exhausted"][2]
    np.testing.assert_allclose(L2[e2["kept"]], e2["L"][e2["kept"]], rtol=0, atol=1e-12)
    assert np.all(np.isneginf(L2[~e2["kept"]]))
    # every kept warm row, retried or not, carries the column it started from:
    # the retry started from the same nearest anchor as the first solve
    new = np.arange(len(z_a), len(anchors))
    want = code_a[((anchors.z[new, None, :] - z_a[None]) ** 2).sum(-1).argmin(1)]
    assert np.array_equal(anchors.columns(new)[0][:, 0, 0], want)
    with np.load(tmp_path / "anchors_00001.npz") as d:
        assert np.array_equal(d["rtol"], np.where(e2["fail1"], RTOL1, RTOL0)[e2["kept"]])

    z = np.concatenate([z1, z2])
    e = {k: np.concatenate([e1[k], e2[k]]) for k in e1}
    assert {k: tally[k] for k in ("n_eval", "tp_out", "exhausted", "stalled", "nonfinite",
                                  "retried", "retry_certified", "retry_exhausted",
                                  "retry_stalled", "retry_nonfinite")} == {
        "n_eval": 32, "tp_out": int(e["tp_out"].sum()),
        "exhausted": int(e["exhausted"].sum()), "stalled": int(e["stalled"].sum()),
        "nonfinite": 0, "retried": int(e["fail1"].sum()),
        "retry_certified": int((e["fail1"] & e["kept"]).sum()),
        "retry_exhausted": int(e["exhausted"].sum()),
        "retry_stalled": int(e["stalled"].sum()), "retry_nonfinite": 0}
    assert tally["cold_retried"] == e1["fail1"].sum() and tally["warm_retried"] == e2["fail1"].sum()
    assert tally["cold_certified"] + tally["warm_certified"] == e["kept"].sum() == len(anchors)
    assert tally["retry_steps_certified"] == 60 * int((e["fail1"] & e["kept"]).sum())
    rows = [json.loads(r) for r in log.read_text().splitlines()]
    assert [r["mode"] for r in rows] == ["cold", "warm"]
    for r, ek, zz in zip(rows, (e1, e2), (z1, z2)):
        rr = np.flatnonzero(ek["fail1"])
        assert r["retry"]["rows"] == rr.tolist()
        assert r["retry"]["width"] == retry_width(len(rr)) == 32
        assert r["acc"] == np.where(zz[:, 0] > 0.6, COUNT_MAX, 10).tolist()   # first pass
        assert r["retry"]["acc"] == np.where(zz[rr, 0] > 0.8, RETRY_MAX, 60).tolist()
        assert (r["retry"]["rtol"], r["retry"]["count_max"]) == (RTOL1, RETRY_MAX)


def test_retry_off_writes_nothing_new(tmp_path):
    rng = np.random.default_rng(0)
    anchors, tally, log = Anchors(tmp_path, 2), {}, tmp_path / "batches.jsonl"
    like = make_loglike(_stub_pipe(), 16, tally, anchors, log)
    like(rng.uniform(0.02, 0.98, (16, D)))
    like(rng.uniform(0.02, 0.98, (16, D)))
    assert not any(k.startswith("retry") or k.endswith("retried") for k in tally)
    assert all(set(json.loads(r)) == {"t", "mode", "acc"}
               for r in log.read_text().splitlines())
    for f in sorted(tmp_path.glob("anchors_*.npz")):
        with np.load(f) as d:
            assert set(d.files) == {"z", "Y", "refs"}
