"""eta_C Fisher-linearity harness (plan section 9; directive L).

The multidimensional Fisher validity guard: coordinate-wise checks alone
are insufficient under correlated posteriors, so the probe set is

  1. each coordinate direction  theta +/- sigma_i,
  2. the principal covariance eigenvector directions (+/-),
  3. `n_ellipsoid` additional points on the nominal 1-sigma ellipsoid
     (deterministic low-discrepancy directions -- no RNG, reproducible),

and at each probe point the NOISE-WHITENED nonlinearity metric, in the same
metric the Fisher matrix uses (||x||_Cinv = sqrt(x^T C^-1 x)):

    eta_C = ||d(theta + dtheta) - d(theta) - J dtheta||_Cinv
            / max(||J dtheta||_Cinv, eta_floor)

with a documented eta_floor guarding the small-response limit. The harness
is MODEL-AGNOSTIC: it takes `forward(theta) -> (n_d,) data vector`,
`J (n_d, n_p)` (the derivative the Fisher forecast will actually use --
the G6 solver-map route once it exists), the noise covariance C (diagonal
sigma or full matrix), and the Fisher covariance whose sigma/eigenvectors
define the probe set. Every forward() call at a probe point is a FULL
reconverged solve on the caller's side.

Built now (directive L), exercised at Fisher-enablement time: Fisher with
condensation stays DISABLED until every essential B0C gate passes AND this
guard passes at the enabled preset with a documented threshold.

Also provided: `noise_floor_from_twins(d_a, d_b, C)` -- the measured
twin-solve floor (plan section 8: identical-theta re-solves) expressed in
the same whitened norm, the natural basis for choosing eta_floor.
"""
from __future__ import annotations

import numpy as np

DEFAULT_ETA_THRESHOLD = 0.1   # documented default; the record may tighten it


def _whitener(C):
    """x -> ||x||_Cinv for diagonal (1-D sigma^2 or sigma) or full C."""
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        sig = np.sqrt(C) if np.all(C > 0) else None
        if sig is None:
            raise ValueError("diagonal C must be positive variances")

        def norm(x):
            return float(np.linalg.norm(np.asarray(x) / sig))

        return norm
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be (n,) variances or (n, n); got {C.shape}")
    L = np.linalg.cholesky(C)

    def norm(x):
        return float(np.linalg.norm(np.linalg.solve(L, np.asarray(x))))

    return norm


def probe_directions(cov, n_ellipsoid: int = 4):
    """The plan's probe set as unit-sigma displacement vectors dtheta.

    cov : (n_p, n_p) Fisher parameter covariance. Returns a list of
    (label, dtheta) with dtheta ON the 1-sigma surface:
    coordinate +/- sigma_i, eigenvector +/- sqrt(lambda_k) v_k, and
    n_ellipsoid deterministic mixed points (normalized so
    dtheta^T cov^-1 dtheta = 1).
    """
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError(f"cov must be square; got {cov.shape}")
    evals, evecs = np.linalg.eigh(cov)
    if np.any(evals <= 0):
        raise ValueError(f"cov not positive definite (eigenvalues {evals})")
    out = []
    sig = np.sqrt(np.diag(cov))
    for i in range(n):
        e = np.zeros(n)
        e[i] = sig[i]
        out.append((f"coord+{i}", e.copy()))
        out.append((f"coord-{i}", -e))
    for k in range(n):
        v = np.sqrt(evals[k]) * evecs[:, k]
        out.append((f"eig+{k}", v.copy()))
        out.append((f"eig-{k}", -v))
    # deterministic mixed ellipsoid points: equal-weight +-1 sign patterns
    # (Hadamard-like, no RNG), scaled onto the 1-sigma surface.
    cinv = np.linalg.inv(cov)
    for j in range(int(n_ellipsoid)):
        signs = np.array([1.0 if (j >> b) & 1 == 0 else -1.0
                          for b in range(n)])
        d = evecs @ (signs * np.sqrt(evals) / np.sqrt(n))
        d = d / np.sqrt(d @ cinv @ d)
        out.append((f"ellipsoid{j}", d))
    return out


def noise_floor_from_twins(d_a, d_b, C) -> float:
    """Measured twin-solve noise floor in the whitened norm (plan section
    8): two independent full solves at IDENTICAL theta; their whitened
    data-vector distance is the floor eta_floor should sit above."""
    return _whitener(C)(np.asarray(d_a) - np.asarray(d_b))


def eta_c_report(forward, theta0, d0, J, C, cov, *,
                 eta_floor: float,
                 threshold: float = DEFAULT_ETA_THRESHOLD,
                 n_ellipsoid: int = 4) -> dict:
    """Run the full probe set; return the measured eta_C table + verdict.

    forward(theta) -> (n_d,) must be a FULL reconverged forward solve.
    d0 = forward(theta0) (caller supplies to avoid re-solving), J the
    (n_d, n_p) derivative the forecast uses, C the noise covariance,
    cov the Fisher parameter covariance, eta_floor the documented
    small-response floor (use noise_floor_from_twins), threshold the
    documented refusal level.
    """
    theta0 = np.asarray(theta0, dtype=np.float64)
    d0 = np.asarray(d0, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    norm = _whitener(C)
    if not (eta_floor > 0.0):
        raise ValueError("eta_floor must be positive and documented "
                         "(measure it with noise_floor_from_twins)")
    rows = []
    worst = 0.0
    for label, dth in probe_directions(cov, n_ellipsoid=n_ellipsoid):
        d1 = np.asarray(forward(theta0 + dth), dtype=np.float64)
        lin = J @ dth
        num = norm(d1 - d0 - lin)
        den = max(norm(lin), eta_floor)
        eta = num / den
        worst = max(worst, eta)
        rows.append({"direction": label,
                     "dtheta": dth.tolist(),
                     "response_whitened": norm(lin),
                     "residual_whitened": num,
                     "eta_C": eta})
    verdict = ("PASS" if worst <= threshold else
               f"FAIL (worst eta_C {worst:.3e} > threshold {threshold})")
    return {
        "harness": "eta_c_linearity (plan section 9)",
        "eta_floor": float(eta_floor),
        "threshold": float(threshold),
        "n_probes": len(rows),
        "worst_eta_C": worst,
        "rows": rows,
        "verdict": verdict,
    }
