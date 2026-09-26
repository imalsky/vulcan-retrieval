"""Evidence-semantics regression.

The SMC samples the OPERATIONAL prior: the declared box conditioned on the
T-P window (A) and chemistry convergence (C), renormalized, so its evidence
is Z_oper = E_pi[L | A and C]. These tests pin the identity that makes the
ZERO-FILLED logZ_box a real integral, the counterexample showing that
P(A) * Z_oper (an f_tp-only "physical" evidence) is not one, and
evidence_report's fields.
"""
import math

import numpy as np
import pytest

from retrieval_framework.pipeline import evidence_report

IDENTITY_REL = 1e-12   # exact identities, up to float round-off

# toy measure: P(A)=0.5, P(C|A)=0.5, L=10 on A and C
Z_OPER = 10.0                                        # E[L | A and C]
Z_BOX_ZEROFILL = 0.5 * 0.5 * Z_OPER                  # int pi L 1[A&C] = 2.5


def test_zero_filled_box_evidence_is_the_exact_masked_integral():
    """logZ_box = logZ + ln(f_tp f_conv) is exactly the integral of
    pi * L * 1[A and C] over the declared box -- verified by direct
    quadrature on a discrete toy measure with a likelihood-correlated
    convergence mask (the dangerous case)."""
    rng = np.random.default_rng(0)
    n = 200_000
    theta = rng.uniform(0.0, 1.0, n)                 # pi = U[0,1]
    in_A = theta < 0.6
    # convergence correlated with likelihood: fails preferentially where L big
    L = np.where(theta < 0.3, 12.0, 2.0)
    in_C = rng.uniform(0.0, 1.0, n) < np.where(theta < 0.3, 0.4, 0.9)
    mask = in_A & in_C
    z_oper = L[mask].mean()                          # what SMC estimates
    f_tp = in_A.mean()
    f_conv = mask.mean() / in_A.mean()               # P(C | A)
    z_box_via_report = z_oper * f_tp * f_conv
    z_box_direct = (L * mask).mean()                 # direct masked quadrature
    assert z_box_via_report == pytest.approx(z_box_direct, rel=IDENTITY_REL)
    # ...whereas the f_tp-only product misses the true A-integral: a support
    # fraction cannot reconstruct the likelihood on the non-converged set
    z_box_true_A = (L * in_A).mean()
    assert abs(z_oper * f_tp - z_box_true_A) / z_box_true_A > 0.15


def test_evidence_report_fields_and_identity():
    stats = dict(tp_n_kept=500, tp_n_drawn=1000,     # f_tp = 0.5
                 n_alive_phase1=400, n_drawn=800,    # f_c1 = 0.5
                 n_phase2=100, n_recert_fail=0)      # f_c2 = 1.0
    logZ = math.log(Z_OPER)
    ev = evidence_report(logZ, stats)
    assert ev["f_tp"] == pytest.approx(0.5)
    assert ev["f_conv"] == pytest.approx(0.5)
    # zero-filled identity on the toy numbers: 10 * 0.5 * 0.5 = 2.5
    assert math.exp(ev["logZ_box"]) == pytest.approx(Z_BOX_ZEROFILL, rel=IDENTITY_REL)
    # the support split is additive in logs
    assert ev["log_support_fraction"] == pytest.approx(
        ev["log_support_physical"] + ev["log_conv_attrition"], rel=IDENTITY_REL)


def test_logZ_error_lower_bound_tracks_ess_collapse():
    """The ESS-based logZ error bound is optimistic but must exist and grow as the
    ladder degrades.

    Exercises the PRODUCTION formula (pipeline.logz_err_lower_bound), not a
    local restatement of it."""
    from retrieval_framework.pipeline import logz_err_lower_bound as lb
    N = 144

    healthy = [0.9 * N] * 20
    degraded = [0.25 * N] * 20
    assert lb(healthy, N) < lb(degraded, N)
    # a perfectly efficient ladder (ESS == N every stage) has zero variance here
    assert lb([float(N)] * 20, N) == 0.0
    # no usable stages -> NaN, never a silent 0
    assert math.isnan(lb([], N))
