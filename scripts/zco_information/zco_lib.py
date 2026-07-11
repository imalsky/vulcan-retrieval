"""Fisher / Laplace machinery for the WASP-39b metallicity + C/O information figures.

Pure numpy. Reads (i) a cached per-tier Jacobian J = d(transit depth)/d(theta) built by
``scripts/zco_information/build_zco_jacobians.py`` (forward-mode AD through VULCAN-JAX kinetics ->
ExoJax transmission), and (ii) the real Carter & May (2024) combined WASP-39b JWST
spectrum (Zenodo 10161743, Fixed_LimbDarkening products) for the per-bin error bars.

The parameter vector for the reported figures is

    theta = [ lnZ, dln(C/O), lnKzz, dT,  lnR0,  offset_1 .. offset_{G-1} ]
            |------ chemistry (4) ------|  radius  inter-instrument offsets (G-1 groups)

    * lnZ       natural-log metallicity scale (scales C/N/O/S element totals)
    * dln(C/O)  log carbon-to-oxygen ratio at FIXED oxygen (fixed-O knob in vulcan_chem)
    * lnKzz     eddy diffusion scale
    * dT        uniform temperature shift (K; historically mislabeled T_int)
    * lnR0      reference-radius scaling at the bottom pressure (the standard xR_p
                transmission normalization nuisance; Batalha & Line 2017)
    * offset_g  a flat depth offset (ppm) for instrument group g, relative to the
                reference group -- the JWST inter-instrument offsets Carter & May measured
                (NIRISS vs G395H ~ -370 ppm). lnR0 already carries the global level, so
                only G-1 relative offsets are free (else lnR0+offsets are degenerate).

We build the full Fisher F = J^T diag(1/sigma^2) J, invert it, and REPORT the
(lnZ, dlnCO) 2x2 sub-block of C = F^-1 -- i.e. everything else is MARGINALIZED, not
fixed. Marginalizing (not fixing) Kzz/dT/R0/offsets is what makes the error bars
honest; see docs.

NOT modeled (documented as a toy limitation, per Isaac's scope decision): clouds/hazes,
a free T-P profile beyond the uniform shift, individual molecular abundances, stellar
contamination, and off-diagonal (wavelength-correlated) noise. Absolute sigma are
therefore best-case lower bounds; the RELATIVE statements (which wavelengths, which
chemistry tier, which parameter combination is degenerate) are the robust content.
"""
from __future__ import annotations

import numpy as np

from retrieval_framework.forward import config  # import-light (os+pathlib only); owns the root/env logic

# np.trapz was renamed np.trapezoid in NumPy 2.0 (and trapz removed); support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

DATA = config.OUTPUTS         # generated caches (zco_jacobians/zco_walk) in the repo output/
FIGS = config.FIGS            # manuscript figures stay in jax_paper/figures
CM24 = config.DATA_DIR / "cm24_wasp39b"

# Model wavelength reach (the ExoJax H2-H2 CIA short edge sits at 1 um; the demo band
# tops out ~5.3 um). Observed bins outside this are dropped.
WL_LO, WL_HI = 1.00, 5.28

# Molecular band centers (um) annotated on the figures.
BANDS = {"H2O": 2.70, "CH4": 3.30, "SO2": 4.05, "CO2": 4.30, "CO": 4.66}

# Chemistry parameter labels (columns 0..3 of the cached J).
CHEM_LABELS = [r"$\ln Z$", r"$\ln(\mathrm{C/O})$", r"$\ln K_{zz}$", r"$\Delta T$"]
CHEM_KEYS = ["lnZ", "lnCO", "lnKzz", "dT"]

# Carter & May (2024) Fixed-limb-darkening recommended products. Each entry:
#   (instrument_group, csv_filename). NRS1/NRS2 share the G395H group (one offset).
CM24_PRODUCTS = {
    "PRISM":  [("PRISM", "PRISM_native.csv")],
    "NIRISS": [("NIRISS", "NIRISS_O1_R100.csv"), ("NIRISS", "NIRISS_O2_R100.csv")],
    "G395H":  [("G395H", "G395H_NRS1_R100.csv"), ("G395H", "G395H_NRS2_R100.csv")],
    "NIRCam": [("NIRCam", "NIRCam_R100.csv")],
}
# Default combined spectrum: NIRISS (water bands) + G395H (SO2/CO2/CO), the highest-
# resolution full-range JWST pairing; two offset groups.
DEFAULT_COMBO = ["NIRISS", "G395H"]


# --------------------------------------------------------------------------- #
#  Data: the real Carter & May 2024 combined spectrum                         #
# --------------------------------------------------------------------------- #
def _read_cm24_csv(path):
    """Read one C&M product CSV -> (wl, wl_lo, wl_hi, depth_frac, sigma_frac).

    Columns (with a leading unnamed index col): wave, wave_low, wave_hig, rp/rs,
    rp/rs_err_low, rp/rs_err_hih. depth = (rp/rs)^2; sigma_depth = 2 rp/rs * sigma_rprs
    with the near-symmetric low/high errors averaged.
    """
    a = np.genfromtxt(path, delimiter=",", skip_header=1)
    wl, wlo, whi, rprs, el, eh = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    sig_rprs = 0.5 * (np.abs(el) + np.abs(eh))
    depth = rprs ** 2
    sigma = 2.0 * rprs * sig_rprs
    lo = np.minimum(wlo, whi)
    hi = np.maximum(wlo, whi)
    good = np.isfinite(wl) & np.isfinite(sigma) & (sigma > 0) & (hi > lo)
    return wl[good], lo[good], hi[good], depth[good], sigma[good]


def load_combined(combo=DEFAULT_COMBO, wl_lo=WL_LO, wl_hi=WL_HI):
    """Load and concatenate C&M products into one combined WASP-39b spectrum.

    Returns a dict with per-bin arrays (sorted by wavelength), clipped to [wl_lo, wl_hi]:
        wl, wl_lo, wl_hi : bin center + bounds (um)
        depth, sigma     : transit depth + 1-sigma (FRACTIONAL, (Rp/Rs)^2)
        group            : instrument-group label per bin (str)
        groups           : ordered unique group labels (reference = groups[0])
    """
    W, LO, HI, D, S, G = [], [], [], [], [], []
    for grp_name in combo:
        for grp, fname in CM24_PRODUCTS[grp_name]:
            wl, lo, hi, d, s = _read_cm24_csv(CM24 / fname)
            W.append(wl); LO.append(lo); HI.append(hi); D.append(d); S.append(s)
            G.append(np.array([grp] * len(wl)))
    wl = np.concatenate(W); lo = np.concatenate(LO); hi = np.concatenate(HI)
    depth = np.concatenate(D); sigma = np.concatenate(S); group = np.concatenate(G)
    sel = (wl >= wl_lo) & (wl <= wl_hi)
    wl, lo, hi, depth, sigma, group = wl[sel], lo[sel], hi[sel], depth[sel], sigma[sel], group[sel]
    o = np.argsort(wl)
    groups = list(dict.fromkeys(group[o].tolist()))   # ordered-unique; ref = groups[0]
    return dict(wl=wl[o], wl_lo=lo[o], wl_hi=hi[o], depth=depth[o], sigma=sigma[o],
                group=group[o], groups=groups)


# --------------------------------------------------------------------------- #
#  Cached model Jacobian -> observed bins                                      #
# --------------------------------------------------------------------------- #
def load_jacobians(npz=DATA / "zco_jacobians.npz"):
    """Load the per-tier cached Jacobians. Returns (wl_um, dict(tier -> payload)).

    Each payload: depth (n,), J_chem (n,4) [lnZ,lnCO,lnKzz,dT], J_lnR0 (n,).
    """
    d = np.load(npz, allow_pickle=True)
    wl = np.asarray(d["wl_um"], float)
    tiers = [str(t) for t in d["tiers"]]
    out = {}
    for t in tiers:
        out[t] = dict(depth=np.asarray(d[f"depth_{t}"], float),
                      J_chem=np.asarray(d[f"Jchem_{t}"], float),
                      J_lnR0=np.asarray(d[f"JlnR0_{t}"], float))
    meta = dict(tiers=tiers,
                theta0=np.asarray(d["theta0"], float) if "theta0" in d.files else None,
                molecules=[str(m) for m in d["molecules"]] if "molecules" in d.files else None)
    return wl, out, meta


def bin_to_obs(wl_model, cols, obs):
    """Bin model columns onto the observed bins by a d(lambda)-weighted trapezoidal
    average (interp to each bin's [lo,hi], integrate, divide by width). `cols` is
    (n_model, k); returns (keep_mask, binned (n_keep, k)). The derivative of a
    bin-integrated depth IS the bin-average of the derivative, so this is the correct
    operator for both the depth and the J columns.
    """
    wl = np.asarray(wl_model, float)
    order = np.argsort(wl)
    wl = wl[order]
    Y = np.asarray(cols, float)[order]
    lo_all, hi_all = obs["wl_lo"], obs["wl_hi"]
    nb = len(obs["wl"])
    out = np.full((nb, Y.shape[1]), np.nan)
    for b in range(nb):
        lo, hi = lo_all[b], hi_all[b]
        if not (wl[0] <= lo < hi <= wl[-1]):
            continue
        inside = (wl > lo) & (wl < hi)
        x = np.concatenate([[lo], wl[inside], [hi]])
        y = np.column_stack([np.interp(x, wl, Y[:, k]) for k in range(Y.shape[1])])
        out[b] = _trapezoid(y, x, axis=0) / (hi - lo)
    keep = np.all(np.isfinite(out), axis=1)
    return keep, out


# --------------------------------------------------------------------------- #
#  Fisher assembly                                                            #
# --------------------------------------------------------------------------- #
OFFSET_UNIT = 1.0e-6   # offset parameter is in ppm: d(depth)/d(offset_ppm) = 1e-6

def build_design(tier_payload, wl_model, obs, use_lnR0=True, use_offsets=True):
    """Assemble the binned design matrix J_full and per-bin sigma for one tier.

    Columns: [lnZ, lnCO, lnKzz, dT] (chemistry) + [lnR0] + [offset_g for g in
    groups[1:]] (relative to the reference group groups[0]).

    Returns dict(J, sigma, labels, keys, interest=(0,1), wl, group, depthM).
    """
    cols = np.column_stack([tier_payload["J_chem"], tier_payload["J_lnR0"],
                            tier_payload["depth"]])
    keep, binned = bin_to_obs(wl_model, cols, obs)
    Jchem = binned[keep, 0:4]
    JlnR0 = binned[keep, 4:5]
    depthM = binned[keep, 5]
    sigma = obs["sigma"][keep]
    depthO = obs["depth"][keep]
    group = obs["group"][keep]
    wl = obs["wl"][keep]

    blocks = [Jchem]
    labels = list(CHEM_LABELS)
    keys = list(CHEM_KEYS)
    if use_lnR0:
        blocks.append(JlnR0)
        labels.append(r"$\ln R_0$"); keys.append("lnR0")
    if use_offsets and len(obs["groups"]) > 1:
        for g in obs["groups"][1:]:
            ind = (group == g).astype(float)[:, None] * OFFSET_UNIT
            blocks.append(ind)
            labels.append(rf"$\delta_{{{g}}}$"); keys.append(f"offset_{g}")
    J = np.column_stack(blocks)
    # Drop any NUISANCE column that carries no information (whitened norm ~ 0) -- e.g. the
    # lnKzz column in the equilibrium tier, where Kzz is zeroed so d(depth)/dlnKzz == 0
    # identically. A zero column makes F singular; an uninformative parameter contributes
    # nothing to the (lnZ, lnCO) marginal anyway, so dropping it is exact, not an approximation.
    Jw_norm = np.linalg.norm(J / sigma[:, None], axis=0)
    thr = 1e-8 * Jw_norm[:2].max()
    keep_col = [c for c in range(J.shape[1]) if c < 2 or Jw_norm[c] > thr]
    if len(keep_col) < J.shape[1]:
        dropped = [keys[c] for c in range(J.shape[1]) if c not in keep_col]
        J = J[:, keep_col]
        labels = [labels[c] for c in keep_col]
        keys = [keys[c] for c in keep_col]
        print(f"[build_design] dropped uninformative column(s): {dropped}")
    return dict(J=J, sigma=sigma, labels=labels, keys=keys, interest=(0, 1),
                wl=wl, group=group, depthM=depthM, depthO=depthO, groups=obs["groups"],
                keep=keep)


def spd_inv(F):
    """Symmetric-positive-definite inverse with diagonal (Jacobi) preconditioning.

    The design mixes parameters whose natural units differ by ~1e3 (lnR0's column is
    ~2*depth/sigma ~ 600, an offset column ~1e-6/sigma ~ 0.01), so F's raw condition
    number is ~1e10 -- numerically fine in float64 but ugly. Scaling F by d=1/sqrt(diag F)
    on both sides makes the inverted matrix unit-diagonal (condition number = the INTRINSIC
    physical conditioning), then we unscale. The result is identical to np.linalg.inv up to
    round-off; only the numerics improve.
    """
    d = 1.0 / np.sqrt(np.diag(F))
    Fs = F * np.outer(d, d)
    Cs = np.linalg.inv(Fs)
    return Cs * np.outer(d, d)


def fisher(J, sigma):
    Ninv = 1.0 / np.asarray(sigma, float) ** 2
    F = J.T @ (Ninv[:, None] * J)
    return F, spd_inv(F)


def marginal_cov(F, interest=(0, 1)):
    """(lnZ, lnCO) marginal covariance = sub-block of F^-1 (all others marginalized)."""
    C = spd_inv(F)
    ix = np.ix_(interest, interest)
    return C[ix], C


def marginal_info_per_bin(J, sigma, p, others):
    """Per-bin marginal Fisher information for parameter column `p`: the noise-whitened
    p-sensitivity with its OLS projection onto ALL `others` columns removed (squared
    residual). Residuals sum to 1/C_pp -- the honest 'where does the spectrum UNIQUELY
    constrain p, after every degeneracy is projected out' map.
    """
    Jw = np.asarray(J, float) / np.asarray(sigma, float)[:, None]
    a = Jw[:, p].copy()
    if len(others):
        B = Jw[:, list(others)]
        coeff, *_ = np.linalg.lstsq(B, a, rcond=None)
        a = a - B @ coeff
    return a ** 2


def eigendecompose_marginal(F, interest=(0, 1)):
    """Eigen-decomposition of the (lnZ, lnCO) MARGINAL Fisher (= inverse of the 2x2
    marginal covariance). Small eigenvalue -> poorly-constrained (degenerate) direction.
    Returns (evals ascending, evecs columns, C2 marginal covariance).
    """
    C2, _ = marginal_cov(F, interest)
    Fmarg = np.linalg.inv(C2)
    evals, evecs = np.linalg.eigh(Fmarg)
    return evals, evecs, C2


def ellipse_xy(C2, center=(0.0, 0.0), dchi2=2.30, n=240):
    """Joint 2D confidence ellipse for a 2x2 covariance (default 68%, 2 dof: dchi2=2.30)."""
    vals, vecs = np.linalg.eigh(C2)
    t = np.linspace(0, 2 * np.pi, n)
    pts = (vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0) * dchi2))) @ np.stack([np.cos(t), np.sin(t)])
    return center[0] + pts[0], center[1] + pts[1]


def scale_sigma(obs, n_transits=1):
    """Return a copy of obs with per-bin sigma scaled as random noise: sigma/sqrt(N)."""
    o = dict(obs)
    o["sigma"] = obs["sigma"] / np.sqrt(n_transits)
    return o


# --------------------------------------------------------------------------- #
#  Self-verification (Monte-Carlo GLS recovery of the covariance)             #
# --------------------------------------------------------------------------- #
def verify(combo=DEFAULT_COMBO, tier=None):
    wl_model, tiers, meta = load_jacobians()
    tier = tier or (meta["tiers"][-1])   # default: the richest (photochem) tier
    obs = load_combined(combo)
    des = build_design(tiers[tier], wl_model, obs)
    J, sigma = des["J"], des["sigma"]
    F, C = fisher(J, sigma)
    p = J.shape[1]
    print(f"=== zco_lib verify: tier={tier} combo={combo} bins={len(sigma)} params={p} ===")
    print("  columns:", ", ".join(des["keys"]))

    sym = np.max(np.abs(F - F.T)) / np.max(np.abs(F))
    ev = np.linalg.eigvalsh(F)
    print(f"  (1) F symmetric={sym:.1e}  min_eig={ev.min():.3e}  cond={ev.max()/ev.min():.2e}")

    ratio = np.diag(C) / (1.0 / np.diag(F))
    print("  (2) marginal/conditional variance ratio >= 1: "
          + ", ".join(f"{des['keys'][i]}:{ratio[i]:.1f}" for i in range(p)))
    marg_ok = np.all(ratio >= 1 - 1e-9)

    rng = np.random.default_rng(0)
    M = C @ (J.T * (1.0 / sigma ** 2))
    n_mc = 60000
    theta_true = np.zeros(p)
    data = (J @ theta_true)[None, :] + rng.normal(size=(n_mc, sigma.size)) * sigma
    Cemp = np.cov((data @ M.T).T)
    rel = np.max(np.abs(Cemp - C) / np.sqrt(np.outer(np.diag(C), np.diag(C))))
    print(f"  (3) Monte-Carlo GLS cov vs F^-1: max rel err {rel:.4f} (expect ~{1/np.sqrt(n_mc):.4f})")

    C2, _ = marginal_cov(F)
    ln10 = np.log(10.0)
    sZ = np.sqrt(C2[0, 0]) / ln10
    sCO = np.sqrt(C2[1, 1]) / ln10
    rho = C2[0, 1] / np.sqrt(C2[0, 0] * C2[1, 1])
    print(f"  marginal: sigma(log10 Z)={sZ:.3f} dex  sigma(log10 C/O)={sCO:.3f} dex  rho={rho:+.3f}")
    ok = (sym < 1e-9) and (ev.min() > 0) and marg_ok and (rel < 0.05)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(verify())
