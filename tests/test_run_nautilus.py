"""run_nautilus.make_loglike on a stub pipeline: every rejection class (T-P
window, count_max exhausted, stall-certified, non-finite) is log L = -inf, so
nautilus's evidence is the zero-filled box integral (pinned against the exact
value on a 3-D Gaussian with three cut regions); with
anchors, each warm solve starts from the nearest certified column and only
certified columns become anchors. With chem_retry, exactly the in-window rows
the first solve left uncertified are re-solved once, from the same start, on
the retry evaluator; a row is kept only if that solve certifies, and the
tally, the batch log and the anchors' rtol record the second pass. Off, none
of that appears. Padding is masked: a short chunk's filler rows and the retry
bucket's filler rows reach the evaluator as ``active`` False, and the warm
evaluators can be compiled ahead on a host thread."""
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

from conftest import StubConvDiag
from retrieval_framework import config_schema as C
from retrieval_framework import pipeline as P
from retrieval_framework.run_nautilus import (DISCARD_EXPLORATION, F_LIVE, N_NETWORKS,
                                              RETRY_PRECOMPILE, Anchors, make_loglike,
                                              make_sampler, retry_width)

D, MU, SD, COUNT_MAX = 3, 0.5, 0.1, 50


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
    theta_from_u, _, _ = P.make_uspace(specs, jnp.float64)

    def common(U):
        th = jax.vmap(theta_from_u)(U)
        L = jnp.sum(-0.5 * ((th - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
        acc = jnp.where(th[:, 0] > 0.6, COUNT_MAX, 10)            # exhausted
        conv = th[:, 1] >= 0.35                                     # stalled below
        return th, jnp.where(th[:, 2] > 0.7, -jnp.inf, L), acc, conv   # T-P window

    def cold(U, Y, refs, active):  # the column is a code of the point's own chemistry
        th, L, acc, conv = common(U)
        L, acc, conv = _masked("cold", seen, active, L, acc, conv)
        return L, (th[:, 0] + 1000 * th[:, 1])[:, None, None], th[:, :2], StubConvDiag(acc, conv)

    def warm(U, Y, refs, active):  # hands back the column it started from; like the
        _, L, acc, conv = common(U)   # pipeline's, it floors L on a stalled/exhausted exit
        L, acc, conv = _masked("warm", seen, active, L, acc, conv)
        L = jnp.where(conv & (acc < COUNT_MAX), L, P.REJECT_LOGL)
        return L, Y, refs, _Stats(acc, conv)

    def make(mode, diag=False, rt_kept_only=False):
        assert rt_kept_only and diag == (mode == "cold")
        return cold if mode == "cold" else warm

    return SimpleNamespace(_make_batch_eval=make, n_dim=D,
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
    exact = (math.log(ndtr(1.0) - ndtr(-5.0)) + math.log(ndtr(5.0) - ndtr(-1.5))
             + math.log(ndtr(2.0) - ndtr(-5.0)))
    assert s.n_eff >= 2000 and abs(s.log_z - exact) < 0.05
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


# ---- chem_retry ---------------------------------------------------------------
# First solve as _stub_pipe (exhausted above p0 = 0.6, stalled below p1 = 0.35,
# T-P out above p2 = 0.7). The retry program caps at RETRY_MAX: it exhausts above
# p0 = 0.8 and stalls below p1 = 0.2; a row it certifies scores L + DL_RETRY and
# hands back the column it started from (cold: the column code + 0.5).
RETRY_MAX, DL_RETRY, RTOL0, RTOL1 = 80, 0.5, 0.2, 0.05


def _retry_stub_pipe(seen=None):
    base = _stub_pipe(seen)
    theta_from_u = base.theta_from_u

    def second(U):
        th = jax.vmap(theta_from_u)(U)
        L = jnp.sum(-0.5 * ((th - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
        acc = jnp.where(th[:, 0] > 0.8, RETRY_MAX, 60)
        conv = th[:, 1] >= 0.2
        return th, jnp.where(th[:, 2] > 0.7, -jnp.inf, L + DL_RETRY), acc, conv

    def cold2(U, Y, refs, active):
        th, L, acc, conv = second(U)
        L, acc, conv = _masked("cold2", seen, active, L, acc, conv)
        code = (th[:, 0] + 1000 * th[:, 1] + 0.5)[:, None, None]
        return L, code, th[:, :2], StubConvDiag(acc, conv)

    def warm2(U, Y, refs, active):
        _, L, acc, conv = second(U)
        L, acc, conv = _masked("warm2", seen, active, L, acc, conv)
        L = jnp.where(conv & (acc < RETRY_MAX), L, P.REJECT_LOGL)
        return L, Y, refs, _Stats(acc, conv)

    first = base._make_batch_eval

    def make(mode, diag=False, retry=False, rt_kept_only=False):
        if not retry:
            return first(mode, diag, rt_kept_only)
        assert rt_kept_only and diag == (mode == "cold")
        return cold2 if mode == "cold" else warm2

    base._make_batch_eval = make
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
    assert [retry_width(n) for n in (0, 1, 131, 132, 133, 264, 265, 660, 661)] == [
        0, 132, 132, 132, 264, 264, 396, 660, 792]
    assert [retry_width(n, 2) for n in (1, 2, 3)] == [2, 2, 4]
    assert all(b == retry_width(b) for b in RETRY_PRECOMPILE)


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
        assert r["retry"]["width"] == retry_width(len(rr)) == 132
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


def test_padding_rows_are_masked(tmp_path):
    """A chunk shorter than n_batch and the retry bucket are filled with copies
    of their last row that reach the evaluators as active=False: the real rows
    lead, in order, and only they are active."""
    rng = np.random.default_rng(5)
    seen, tally = [], {}
    anchors, log = Anchors(tmp_path, 2), tmp_path / "batches.jsonl"
    like = make_loglike(_retry_stub_pipe(seen), 16, tally, anchors, log)
    z1 = rng.uniform(0.02, 0.98, (5, D))
    z1[:2] = [[0.7, 0.5, 0.3], [0.5, 0.3, 0.3]]        # both fail the first pass
    L1 = like(z1)                                       # cold: 5 real rows of 16
    z2 = rng.uniform(0.02, 0.98, (16, D))
    z2[:3] = [[0.75, 0.5, 0.3], [0.5, 0.25, 0.3], [0.95, 0.5, 0.3]]
    L2 = like(z2)                                       # warm: a full chunk
    jax.effects_barrier()
    e1, e2 = _expected(z1), _expected(z2)
    for L, e in ((L1, e1), (L2, e2)):
        np.testing.assert_allclose(L[e["kept"]], e["L"][e["kept"]], rtol=0, atol=1e-12)
        assert np.all(np.isneginf(L[~e["kept"]]))
    calls = [name for name, _ in seen]
    assert calls == ["cold", "cold2", "warm", "warm2"], calls
    act = dict(seen)
    n1, n2 = int(e1["fail1"].sum()), int(e2["fail1"].sum())
    assert act["cold"] == [True] * 5 + [False] * 11
    assert act["cold2"] == [True] * n1 + [False] * (retry_width(n1) - n1)
    assert act["warm"] == [True] * 16
    assert act["warm2"] == [True] * n2 + [False] * (retry_width(n2) - n2)
    assert tally["n_eval"] == 21 and tally["retry_width_sum"] == retry_width(n1) + retry_width(n2)


def test_precompile_builds_the_warm_programs_ahead(tmp_path):
    """precompile=True: once the cold first chunk is dispatched, the warm
    first-pass program and every RETRY_PRECOMPILE retry bucket compile on a
    host thread; the warm chunks then run on them, and the results are the
    ones compiled at first use."""
    rng = np.random.default_rng(3)
    z1, z2 = rng.uniform(0.02, 0.98, (16, D)), rng.uniform(0.02, 0.98, (16, D))
    z1[:2] = [[0.7, 0.5, 0.3], [0.5, 0.3, 0.3]]
    z2[:2] = [[0.75, 0.5, 0.3], [0.5, 0.25, 0.3]]
    out = {}
    for pre in (False, True):
        tally, anchors = {}, Anchors(tmp_path / f"a{int(pre)}", 2)
        like = make_loglike(_retry_stub_pipe(), 16, tally, anchors, precompile=pre)
        L1 = like(z1)
        progs = like.programs
        ahead = {("warm", 16)} | {("retry_warm", b) for b in RETRY_PRECOMPILE}
        if pre:
            assert ahead <= set(progs.pending) | set(progs.exe)
            for f in list(progs.pending.values()):
                f.result()
        else:
            assert not progs.pending and not (ahead & set(progs.exe))
        L2 = like(z2)
        out[pre] = (L1, L2, tally)
        assert ("warm", 16) in progs.exe
    for a, b in zip(out[False][:2], out[True][:2]):
        assert np.array_equal(a, b)
    counts = [{k: v for k, v in t.items() if not k.endswith("wall_s")} for t in
              (out[False][2], out[True][2])]
    assert counts[0] == counts[1]


# ---- chem_presolve -------------------------------------------------------------
# The retry stub plus a presolve that "certifies" where the stub's first pass
# does (p0 <= 0.6, p1 >= 0.35) and hands back the column code + 0.25. Every
# in-window row then goes through the retry program (warm2 hands back the
# column it started from), so a kept row's anchor column tells which start the
# second pass used.
PRE_CODE = 0.25


def _presolve_stub_pipe(seen=None):
    base = _retry_stub_pipe(seen)
    theta_from_u = base.theta_from_u

    def make_pre(mode):
        name = f"pre_{mode}"

        def ev(U, Y, refs, active):
            th = jax.vmap(theta_from_u)(U)
            acc = jnp.where(th[:, 0] > 0.6, COUNT_MAX, 10)
            conv = th[:, 1] >= 0.35
            _, acc, conv = _masked(name, seen, active, jnp.zeros(th.shape[0]), acc, conv)
            code = (th[:, 0] + 1000 * th[:, 1] + PRE_CODE)[:, None, None]
            return code, th[:, :2], acc, conv
        return ev

    base._make_presolve_eval = make_pre
    base.chem_retry.presolve = True
    return base


def _expected_presolve(z):
    tp_out = z[:, 2] > 0.7
    pre_ok = ~tp_out & (z[:, 0] <= 0.6) & (z[:, 1] >= 0.35)
    kept = ~tp_out & (z[:, 0] <= 0.8) & (z[:, 1] >= 0.2)
    L = np.sum(-0.5 * ((z - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
    return dict(tp_out=tp_out, pre_ok=pre_ok, kept=kept, L=np.where(kept, L + DL_RETRY, -np.inf),
                exhausted=~tp_out & (z[:, 0] > 0.8), stalled=~tp_out & (z[:, 0] <= 0.8) & (z[:, 1] < 0.2))


def test_presolve_solves_every_in_window_row_from_its_presolve_column(tmp_path):
    rng = np.random.default_rng(7)
    seen, tally = [], {}
    anchors, log = Anchors(tmp_path, 2), tmp_path / "batches.jsonl"
    like = make_loglike(_presolve_stub_pipe(seen), 16, tally, anchors, log)
    z1 = rng.uniform(0.02, 0.98, (16, D))
    z1[:4] = [[0.7, 0.5, 0.3], [0.9, 0.5, 0.3], [0.5, 0.3, 0.3], [0.5, 0.1, 0.3]]
    L1 = like(z1)                                   # cold chunk
    e1 = _expected_presolve(z1)
    np.testing.assert_allclose(L1[e1["kept"]], e1["L"][e1["kept"]], rtol=0, atol=1e-12)
    assert np.all(np.isneginf(L1[~e1["kept"]]))
    # cold chunk: every second pass starts from the presolve's exit column
    code1 = anchors.columns(np.arange(len(anchors)))[0][:, 0, 0]
    assert np.allclose(code1, (z1[:, 0] + 1000 * z1[:, 1] + PRE_CODE)[e1["kept"]])
    z_a = anchors.z.copy()
    with np.load(tmp_path / "anchors_00000.npz") as d:
        assert np.all(d["rtol"] == RTOL1)              # every kept column is the retry's

    z2 = rng.uniform(0.02, 0.98, (16, D))
    z2[:3] = [[0.75, 0.5, 0.3], [0.5, 0.25, 0.3], [0.5, 0.5, 0.3]]
    L2 = like(z2)                                   # warm chunk
    e2 = _expected_presolve(z2)
    assert e2["kept"][:3].all() and not e2["pre_ok"][:2].any() and e2["pre_ok"][2]
    np.testing.assert_allclose(L2[e2["kept"]], e2["L"][e2["kept"]], rtol=0, atol=1e-12)
    assert np.all(np.isneginf(L2[~e2["kept"]]))
    # warm chunk: from the presolve column where the presolve certified, else
    # from the nearest anchor (the row's original start)
    new = np.arange(len(z_a), len(anchors))
    near = code1[((anchors.z[new, None, :] - z_a[None]) ** 2).sum(-1).argmin(1)]
    zk = z2[e2["kept"]]
    want = np.where(e2["pre_ok"][e2["kept"]], zk[:, 0] + 1000 * zk[:, 1] + PRE_CODE, near)
    assert np.allclose(anchors.columns(new)[0][:, 0, 0], want)

    jax.effects_barrier()
    act = {}
    for name, a in seen:
        act.setdefault(name, []).append(a)
    assert [n for n, _ in seen] == ["pre_cold", "warm2", "pre_warm", "warm2"]
    for z, e, a2 in ((z1, e1, act["warm2"][0]), (z2, e2, act["warm2"][1])):
        n_in = int((~e["tp_out"]).sum())
        assert a2 == [True] * n_in + [False] * (16 - n_in)     # fixed width n_batch
    e = {k: np.concatenate([e1[k], e2[k]]) for k in e1}
    assert {k: tally[k] for k in ("n_eval", "tp_out", "exhausted", "stalled", "nonfinite",
                                  "retried", "retry_certified", "presolve_certified")} == {
        "n_eval": 32, "tp_out": int(e["tp_out"].sum()), "exhausted": int(e["exhausted"].sum()),
        "stalled": int(e["stalled"].sum()), "nonfinite": 0, "retried": int((~e["tp_out"]).sum()),
        "retry_certified": int(e["kept"].sum()), "presolve_certified": int(e["pre_ok"].sum())}
    assert tally["retry_width_sum"] == 32 and "presolve_wall_s" in tally
    rows = [json.loads(r) for r in log.read_text().splitlines()]
    for r, ek in zip(rows, (e1, e2)):
        assert r["retry"]["rows"] == np.flatnonzero(~ek["tp_out"]).tolist()
        assert r["retry"]["width"] == 16
        assert r["presolve"] == {"certified": int(ek["pre_ok"].sum()), "rtol": RTOL0,
                                 "count_max": COUNT_MAX}


def test_presolve_off_is_the_stock_retry(tmp_path):
    """A chem_retry namespace with presolve False is the stock retry, row for row."""
    rng = np.random.default_rng(3)
    z1, z2 = rng.uniform(0.02, 0.98, (16, D)), rng.uniform(0.02, 0.98, (16, D))
    out = []
    for flag in (None, False):
        pipe = _retry_stub_pipe()
        if flag is not None:
            pipe.chem_retry.presolve = flag
        tally, anchors = {}, Anchors(tmp_path / f"a{flag}", 2)
        log = tmp_path / f"b{flag}.jsonl"
        like = make_loglike(pipe, 16, tally, anchors, log)
        L = np.concatenate([like(z1), like(z2)])
        rows = [{k: v for k, v in json.loads(r).items() if k not in ("t", "t_first")}
                for r in log.read_text().splitlines()]
        for r in rows:
            r["retry"].pop("t")
        out.append((L, {k: v for k, v in tally.items() if not k.endswith("wall_s")}, rows,
                    anchors.columns(np.arange(len(anchors)))[0]))
        assert set(like.programs.fns) == {"cold", "warm", "retry_cold", "retry_warm"}
    assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1]
    assert out[0][2] == out[1][2] and np.array_equal(out[0][3], out[1][3])
    assert not any(k.startswith("presolve") for k in out[1][1])


def test_presolve_precompiles_only_the_n_batch_programs(tmp_path):
    """precompile with the presolve: the retry program and the warm presolve at
    n_batch, no retry bucket; the results equal those compiled at first use."""
    rng = np.random.default_rng(11)
    z1, z2 = rng.uniform(0.02, 0.98, (16, D)), rng.uniform(0.02, 0.98, (16, D))
    out = {}
    for pre in (False, True):
        tally, anchors = {}, Anchors(tmp_path / f"p{int(pre)}", 2)
        like = make_loglike(_presolve_stub_pipe(), 16, tally, anchors, precompile=pre)
        L1 = like(z1)
        progs = like.programs
        if pre:
            assert set(progs.pending) | set(progs.exe) >= {("retry_warm", 16), ("pre_warm", 16)}
            for f in list(progs.pending.values()):
                f.result()
        L2 = like(z2)
        assert all(w == 16 for _, w in progs.exe)       # nothing at a bucket width
        assert not any(name in ("warm", "retry_cold") for name, _ in progs.exe)
        out[pre] = (L1, L2, {k: v for k, v in tally.items() if not k.endswith("wall_s")})
    for a, b in zip(out[False][:2], out[True][:2]):
        assert np.array_equal(a, b)
    assert out[False][2] == out[True][2]


@pytest.mark.parametrize("knob,bad", [("cold_lanes", -1), ("cold_refill_chunk", 0)])
def test_validate_config_refuses_broken_knobs(knob, bad):
    """A negative lane count or a zero refill chunk is a broken run, not a
    setting (cold_lanes=0 is the single lockstep batch; the default is the
    production lane count)."""
    with pytest.raises(ValueError, match=knob):
        C.validate_config(C.Config(**{knob: bad}))
