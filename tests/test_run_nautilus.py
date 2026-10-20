"""run_nautilus.make_loglike on a stub pipeline: every rejection class (T-P
window, count_max exhausted, stall-certified, non-finite) is log L = -inf, so
the stock sampler's evidence is the zero-filled box integral (pinned against
the exact value on a 3-D Gaussian with three cut regions); with
anchors, each warm solve starts from the nearest certified column and only
certified columns become anchors. With the presolve chain, every in-window
row is solved again on the retry evaluator from its presolve column (or its
original start); a row is kept only if that solve certifies, and the tally,
the batch log and the anchors' rtol record the second pass. Off, none of that
appears. Padding is masked: a short chunk's filler rows and the second pass's
unused rows reach the evaluator as ``active`` False, and the warm evaluators
can be compiled ahead on a host thread. Last, every config
refusal of validate_config."""
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

    def make(mode, diag=False):
        assert diag == (mode == "cold")
        return cold if mode == "cold" else warm

    return SimpleNamespace(_make_batch_eval=make, n_dim=D,
                           theta_from_u=theta_from_u, tp_valid=lambda th: th[2] <= 0.7,
                           dtype=jnp.float64, n_chem_tp=2, y_baseline=jnp.zeros((1, 1)),
                           fwd=SimpleNamespace(chem=SimpleNamespace(count_max=COUNT_MAX)))


def test_rejection_classes():
    tally = {}
    like = make_loglike(_stub_pipe(), 64, tally)
    z = np.array([[0.5, 0.5, 0.5], [0.65, 0.5, 0.5], [0.5, 0.3, 0.5], [0.5, 0.5, 0.75]])
    L = like(z)                                  # 4 rows through one padded chunk of 64
    assert np.isfinite(L[0]) and np.all(np.isneginf(L[1:]))
    assert {k: tally[k] for k in ("n_eval", "tp_out", "exhausted", "stalled", "nonfinite")} == {
        "n_eval": 4, "tp_out": 1, "exhausted": 1, "stalled": 1, "nonfinite": 0}


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


# ---- the presolve chain -------------------------------------------------------
# The presolve "certifies" where the stub's first pass does (p0 <= 0.6,
# p1 >= 0.35) and hands back the column code + PRE_CODE. The retry program caps
# at RETRY_MAX: it exhausts above p0 = 0.8 and stalls below p1 = 0.2; a row it
# certifies scores L + DL_RETRY and hands back the column it started from, so a
# kept row's anchor column tells which start the second pass used.
RETRY_MAX, DL_RETRY, RTOL0, RTOL1, PRE_CODE = 80, 0.5, 0.2, 0.05, 0.25


def _presolve_stub_pipe(seen=None):
    base = _stub_pipe(seen)
    theta_from_u = base.theta_from_u

    def warm2(U, Y, refs, active):
        th = jax.vmap(theta_from_u)(U)
        L = jnp.sum(-0.5 * ((th - MU) / SD) ** 2 - math.log(SD * math.sqrt(2 * math.pi)), axis=1)
        L = jnp.where(th[:, 2] > 0.7, -jnp.inf, L + DL_RETRY)
        acc = jnp.where(th[:, 0] > 0.8, RETRY_MAX, 60)
        L, acc, conv = _masked("warm2", seen, active, L, acc, th[:, 1] >= 0.2)
        L = jnp.where(conv & (acc < RETRY_MAX), L, P.REJECT_LOGL)
        return L, Y, refs, _Stats(acc, conv)

    first = base._make_batch_eval

    def make(mode, diag=False, retry=False):
        if not retry:
            return first(mode, diag)
        assert mode == "warm" and not diag
        return warm2

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

    base._make_batch_eval = make
    base._make_presolve_eval = make_pre
    base.chem_retry = SimpleNamespace(rtol_first=RTOL0, count_max_first=COUNT_MAX,
                                      rtol=RTOL1, count_max=RETRY_MAX, presolve=True)
    return base


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
    z1 = rng.uniform(0.02, 0.98, (5, D))
    z1[:4] = [[0.7, 0.5, 0.3], [0.9, 0.5, 0.3], [0.5, 0.3, 0.3], [0.5, 0.1, 0.3]]
    L1 = like(z1)                                   # cold chunk: 5 real rows of 16
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
    assert act["pre_cold"] == [[True] * 5 + [False] * 11]      # the padded rows
    for z, e, a2 in ((z1, e1, act["warm2"][0]), (z2, e2, act["warm2"][1])):
        n_in = int((~e["tp_out"]).sum())
        assert a2 == [True] * n_in + [False] * (16 - n_in)     # fixed width n_batch
    e = {k: np.concatenate([e1[k], e2[k]]) for k in e1}
    assert {k: tally[k] for k in ("n_eval", "tp_out", "exhausted", "stalled", "nonfinite",
                                  "retried", "retry_certified", "presolve_certified")} == {
        "n_eval": 21, "tp_out": int(e["tp_out"].sum()), "exhausted": int(e["exhausted"].sum()),
        "stalled": int(e["stalled"].sum()), "nonfinite": 0, "retried": int((~e["tp_out"]).sum()),
        "retry_certified": int(e["kept"].sum()), "presolve_certified": int(e["pre_ok"].sum())}
    assert tally["retry_width_sum"] == 32 and "presolve_wall_s" in tally
    rows = [json.loads(r) for r in log.read_text().splitlines()]
    for r, ek in zip(rows, (e1, e2)):
        assert r["retry"]["rows"] == np.flatnonzero(~ek["tp_out"]).tolist()
        assert r["retry"]["width"] == 16
        assert r["presolve"] == {"certified": int(ek["pre_ok"].sum()), "rtol": RTOL0,
                                 "count_max": COUNT_MAX}


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
        assert not any(name in ("warm", "cold") for name, _ in progs.exe)
        out[pre] = (L1, L2, {k: v for k, v in tally.items() if not k.endswith("wall_s")})
    for a, b in zip(out[False][:2], out[True][:2]):
        assert np.array_equal(a, b)
    assert out[False][2] == out[True][2]


VALID = dict(vulcan_cfg_name="W39b", rp_cm=1.0, rstar_cm=10.0)
CHAIN = dict(chem_retry=True, chem_presolve=True)


@pytest.mark.parametrize("over, msg", [
    (dict(cold_lanes=-1), "cold_lanes"),      # 0 is the single lockstep batch
    (dict(cold_refill_chunk=0), "cold_refill_chunk"),
    (dict(chem_retry=1, chem_presolve=True), "chem_retry must be"),   # a JSON 1, not true
    (dict(CHAIN, chem_retry_rtol=0.0), "chem_retry_rtol"),
    (dict(CHAIN, chem_retry_rtol=1.0), "chem_retry_rtol"),
    (dict(CHAIN, chem_retry_rtol=math.nan), "chem_retry_rtol"),
    (dict(CHAIN, chem_retry_count_max=0), "chem_retry_count_max"),
    (dict(chem_presolve=True), "go together"),
    (dict(chem_retry=True), "go together"),      # the removed retry-only mode
    (dict(chem_retry=True, chem_presolve=1), "chem_presolve must be"),
    (dict(noise_inflation_per_group=True), "infer_noise_inflation"),
    (dict(cfg_overrides={"use_condense": True}), "use_condense"),
    # accepted: the retry settings are inert while it is off, the presolve
    # chain, and condensation with its explicit opt-in
    (dict(chem_retry_rtol=0.0, chem_retry_count_max=0), None),
    (CHAIN, None),
    (dict(cfg_overrides={"use_condense": True}, allow_condense_inference=True), None),
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
