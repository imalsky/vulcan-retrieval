"""Cold is the default target, and the warm gradient gate is implementable.

Two things pinned here:

1. `smc_chem_mode` defaults to "cold", so a likelihood evaluation is a fixed
   deterministic function of theta -- what MALA, SMC tempering, and a quoted
   evidence all assume. Warm continuation stays available by explicit opt-in.

2. The warm-vs-cold GRADIENT comparison is a FAIL gate, but it EXCLUDES rows
   whose warm drift was zeroed by the badgrad handling. That exclusion is what
   makes the gate mean anything: a zeroed warm row against a finite cold row
   reads rel exactly 1.0 by construction, so a naive hard gate at 0.1 would fail
   every run containing a single badgrad particle -- which is most runs, since
   the class is posterior-concentrated (6.5% of certified proposals at job
   65815). The gate would then be re-measuring "did badgrad occur" rather than
   "does warm continuation reproduce the cold drift". The zeroed fraction gets
   its own separate ceiling instead.

Pure numpy; no jax, no chemistry stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from retrieval_framework import config_schema as C
from retrieval_framework.validate_warm import (
    GRAD_REL_FAIL, GRAD_ZEROED_FRAC_FAIL, compare_grad,
)


# --- the default target ------------------------------------------------------

def test_default_chem_mode_is_cold():
    assert C.Config.smc_chem_mode == "cold"


def test_warm_extrapolate_is_off_by_default_and_requires_warm():
    """warm_extrapolate has no meaning without a carried column to extrapolate."""
    assert C.Config.warm_extrapolate is False
    cfg = C.Config(smc_chem_mode="cold", warm_extrapolate=True)
    with pytest.raises(ValueError, match="warm_extrapolate"):
        C.validate_config(cfg)


def test_w39b_production_preset_resolves_to_cold():
    """The case the paper reports must not quietly stay on the warm target."""
    import importlib.util
    from pathlib import Path

    case_py = (Path(__file__).resolve().parents[1] / "runs"
               / "w39b_smc_retrieval" / "case.py")
    spec = importlib.util.spec_from_file_location("w39b_case", case_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.PRESETS["gpu"]()
    assert cfg.smc_chem_mode == "cold"
    assert cfg.warm_extrapolate is False


# --- the gradient gate -------------------------------------------------------

def _grads(n=10, d=4, seed=0):
    rng = np.random.default_rng(seed)
    Gc = rng.normal(size=(n, d))
    return Gc.copy(), Gc


def test_identical_gradients_pass_the_gate():
    Gw, Gc = _grads()
    gs = compare_grad(Gw, Gc, np.ones(len(Gc), bool))
    assert gs["rel_max_gated"] < GRAD_REL_FAIL
    assert gs["n_zeroed"] == 0
    assert gs["n_gated"] == len(Gc)


def test_zeroed_row_reads_rel_one_but_is_excluded_from_the_gate():
    """The exact situation that made a naive hard gate unimplementable."""
    Gw, Gc = _grads()
    Gw[3] = 0.0                                   # badgrad zero-drift handling
    gs = compare_grad(Gw, Gc, np.ones(len(Gc), bool))

    # the raw statistic sees it, and reads exactly 1.0 ...
    assert gs["rel"][3] == pytest.approx(1.0)
    assert gs["rel_max"] == pytest.approx(1.0)
    # ... which would have failed a naive gate at 0.1
    assert gs["rel_max"] >= GRAD_REL_FAIL

    # ... but the GATE subset excludes it and passes
    assert gs["n_zeroed"] == 1
    assert gs["n_gated"] == len(Gc) - 1
    assert gs["rel_max_gated"] < GRAD_REL_FAIL


def test_a_real_disagreement_still_fails_the_gate():
    """Excluding zeroed rows must not blind the gate to genuine drift error."""
    Gw, Gc = _grads()
    Gw[5] = -Gc[5]                                # sign-flipped drift
    gs = compare_grad(Gw, Gc, np.ones(len(Gc), bool))
    assert gs["n_zeroed"] == 0
    assert gs["rel_max_gated"] >= GRAD_REL_FAIL
    assert gs["cos_min"] < 0.0                    # drift points the wrong way


def test_zeroed_fraction_has_its_own_ceiling():
    """Mass zeroing is a real problem, just not the one rel_max_gated measures."""
    Gw, Gc = _grads(n=20)
    Gw[:8] = 0.0                                  # 40% of the cloud
    gs = compare_grad(Gw, Gc, np.ones(len(Gc), bool))
    assert gs["zeroed_frac"] == pytest.approx(0.4)
    assert gs["zeroed_frac"] >= GRAD_ZEROED_FRAC_FAIL
    # the surviving rows still agree, so the two conditions are independent
    assert gs["rel_max_gated"] < GRAD_REL_FAIL


def test_a_zero_cold_gradient_is_not_counted_as_zeroed():
    """Only a zeroed WARM row against a finite COLD row is the badgrad case."""
    Gw, Gc = _grads()
    Gc[2] = 0.0
    Gw[2] = 0.0
    gs = compare_grad(Gw, Gc, np.ones(len(Gc), bool))
    assert gs["n_zeroed"] == 0        # both zero: nothing was discarded


def test_excluded_particles_are_ignored_entirely():
    Gw, Gc = _grads()
    Gw[1] = -Gc[1]
    ok = np.ones(len(Gc), bool)
    ok[1] = False                                  # cold-nonconverged particle
    gs = compare_grad(Gw, Gc, ok)
    assert gs["n_ok"] == len(Gc) - 1
    assert gs["rel_max_gated"] < GRAD_REL_FAIL
