"""Evidence-semantics regression (2026-07-12 recheck P0-B).

The SMC samples the OPERATIONAL prior: the declared box conditioned on the
T-P window (A) and chemistry convergence (C), renormalized, so its evidence
is Z_oper = E_pi[L | A and C]. The retracted ``logZ_box_physical`` multiplied
Z_oper by P(A) alone -- restoring the T-P prior mass while silently keeping
the convergence conditioning renormalized. The audit's toy numbers show that
quantity is NOT any valid evidence; these tests pin (a) the counterexample,
(b) the identity that makes the ZERO-FILLED logZ_box a real integral, and
(c) that evidence_report exposes no f_tp-only "physical" evidence.
"""
import math

import numpy as np
import pytest

from retrieval_framework.pipeline import evidence_report


# ---- the audit's toy measure: P(A)=0.5, P(C|A)=0.5, L=10 on A&C, 1 on A&~C
P_A = 0.5
P_C_GIVEN_A = 0.5
L_ON_AC = 10.0
L_ON_A_NOT_C = 1.0

Z_OPER = L_ON_AC                                     # E[L | A and C]
Z_BOX_ZEROFILL = P_A * P_C_GIVEN_A * L_ON_AC         # int pi L 1[A&C] = 2.5
Z_BOX_TRUE_A = P_A * (P_C_GIVEN_A * L_ON_AC
                      + (1 - P_C_GIVEN_A) * L_ON_A_NOT_C)   # int_A pi L = 2.75
Z_COND_A = Z_BOX_TRUE_A / P_A                        # E[L | A] = 5.5
Z_RETRACTED = P_A * Z_OPER                           # the old "physical" = 5.0


def test_retracted_physical_correction_is_no_valid_evidence():
    """P(A) * E[L | A and C] equals none of the well-defined quantities: not
    the zero-filled box integral, not the true box integral over A, not the
    A-conditioned evidence. A support fraction cannot reconstruct the
    likelihood on the unevaluated (non-converged) set."""
    for valid in (Z_BOX_ZEROFILL, Z_BOX_TRUE_A, Z_COND_A, Z_OPER):
        assert abs(Z_RETRACTED - valid) > 0.4


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
    assert z_box_via_report == pytest.approx(z_box_direct, rel=1e-12)
    # ...whereas the retracted f_tp-only product misses the true A-integral
    z_box_true_A = (L * in_A).mean()
    assert abs(z_oper * f_tp - z_box_true_A) / z_box_true_A > 0.15


def test_evidence_report_fields_and_identity():
    stats = dict(tp_n_kept=500, tp_n_drawn=1000,     # f_tp = 0.5
                 n_alive_phase1=400, n_drawn=800,    # f_c1 = 0.5
                 n_phase2=100, n_recert_fail=0)      # f_c2 = 1.0
    logZ = math.log(Z_OPER)
    ev = evidence_report(logZ, stats)
    assert "logZ_box_physical" not in ev             # retracted, absent
    assert ev["f_tp"] == pytest.approx(0.5)
    assert ev["f_conv"] == pytest.approx(0.5)
    # zero-filled identity on the toy numbers: 10 * 0.5 * 0.5 = 2.5
    assert math.exp(ev["logZ_box"]) == pytest.approx(Z_BOX_ZEROFILL, rel=1e-12)
    # the support split is additive in logs
    assert ev["log_support_fraction"] == pytest.approx(
        ev["log_support_physical"] + ev["log_conv_attrition"], rel=1e-12)
    # no init stats -> NaNs, never silent numbers
    ev0 = evidence_report(logZ, None)
    assert all(math.isnan(ev0[k]) for k in
               ("logZ_box", "log_support_fraction", "log_conv_attrition"))
