"""cfg.chem_retry, the certifying program of the presolve chain.

The retry profile changes only rtol and the cap. Its manifest binding is in
test_provenance.py, its config refusals and the make_loglike logic (on a stub
pipeline) in test_run_nautilus.py.
The slow test builds the REAL smoke pipeline with the chain on
(conftest.build_smoke_pipe at count_max=50 < count_min, so no presolve
certifies) and checks the retry program differs from the main one only in rtol
and the cap, and that make_loglike sends every in-window draw through it.
"""
import json

import numpy as np
import pytest
import jax

jax.config.update("jax_enable_x64", True)

from retrieval_framework import config_schema as C  # noqa: E402
from retrieval_framework import pipeline as P  # noqa: E402

VALID = dict(vulcan_cfg_name="W39b", rp_cm=1.0, rstar_cm=10.0)


def test_retry_profile_changes_only_rtol_and_the_cap():
    cfg = C.Config(**VALID, count_max=3000, chem_retry=True, cfg_overrides={"Rp": 1.0})
    p0, p1 = cfg.profile(), C.retry_profile(cfg)
    assert p1.pop("cfg_overrides") == {"Rp": 1.0, "rtol": 0.05}
    assert p1.pop("count_max") == 5000
    p0.pop("cfg_overrides")
    p0.pop("count_max")
    assert p1 == p0
    assert cfg.profile()["cfg_overrides"] == {"Rp": 1.0}            # untouched


@pytest.mark.slow
def test_real_pipeline_presolve_sends_every_draw_through_the_retry(tmp_path):
    """chem_presolve on the REAL smoke pipeline: the presolve evaluator is the
    main program's chemistry (no RT), and every in-window draw is solved again
    on the retry program at the fixed width n_batch (here no solve certifies:
    both caps sit below count_min)."""
    from conftest import CAPPED_COLD_CMAX, build_smoke_pipe
    from retrieval_framework import retrieval_forward as RF
    from retrieval_framework import run_nautilus as RN
    retry_cmax = CAPPED_COLD_CMAX + 10            # still < count_min: cannot certify
    pipe = build_smoke_pipe(count_max=CAPPED_COLD_CMAX, chem_retry=True,
                            chem_retry_rtol=0.05, chem_retry_count_max=retry_cmax,
                            chem_presolve=True)
    r = pipe.chem_retry
    assert r.presolve is True
    assert (r.count_max_first, r.count_max, r.rtol) == (CAPPED_COLD_CMAX, retry_cmax, 0.05)
    st0, st1 = pipe.fwd.chem._integ._statics, pipe.fwd.chem_retry._integ._statics
    diff = set(RF.statics_diff(st0, st1))
    assert "count_max" in diff and diff <= RF.RETRY_STATICS_FREE, diff
    assert float(st1.rtol_accept) == 0.05 and r.rtol_first == float(st0.rtol_accept)
    U = np.asarray(pipe.sample_prior_u(jax.random.PRNGKey(0), 2))   # T-P in window
    Y, refs = P._blank_state(pipe, 2)
    Yp, rp, acc, conv = jax.jit(pipe._make_presolve_eval("cold"))(U, Y, refs, np.ones(2, bool))
    th = np.asarray(jax.vmap(pipe.theta_from_u)(U))
    assert np.allclose(np.asarray(rp), th[:, :2]) and np.all(np.isfinite(np.asarray(Yp)))
    assert np.all(np.asarray(acc) >= CAPPED_COLD_CMAX) and not np.any(np.asarray(conv))
    Z = 1.0 / (1.0 + np.exp(-U))
    tally, log = {}, tmp_path / "batches.jsonl"
    L = RN.make_loglike(pipe, 2, tally, None, log)(Z)
    assert np.all(np.isneginf(L))
    assert tally["retried"] == tally["retry_exhausted"] == tally["exhausted"] == 2
    assert tally["presolve_certified"] == tally["retry_certified"] == tally["tp_out"] == 0
    row = json.loads(log.read_text().splitlines()[0])
    assert row["mode"] == "cold" and min(row["acc"]) >= CAPPED_COLD_CMAX
    assert (row["presolve"]["certified"], row["presolve"]["count_max"]) == (0, CAPPED_COLD_CMAX)
    assert row["retry"]["rows"] == [0, 1] and row["retry"]["width"] == 2
    assert min(row["retry"]["acc"]) >= retry_cmax and not any(row["retry"]["conv"])
