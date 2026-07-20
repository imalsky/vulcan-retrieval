"""Fisher / Laplace machinery for the WASP-39b metallicity + C/O information figures.

Pure numpy. Reads (i) a cached per-tier Jacobian J = d(transit depth)/d(theta) built by
``scripts/zco_information/build_zco_jacobians.py`` (forward-mode AD through VULCAN-JAX kinetics ->
ExoJax transmission), and (ii) the real Carter & May (2024) combined WASP-39b JWST
spectrum (Zenodo 10161743, Fixed_LimbDarkening products) for the per-bin error bars.

The parameter vector for the reported figures is

    theta = [ lnZ, dln(C/O), lnKzz, dT,  lnR0,  offset_1 .. offset_G ]
            |------ chemistry (4) ------|  radius  per-instrument depth offsets (G groups)

    * lnZ       natural-log metallicity scale (scales C/N/O/S element totals)
    * dln(C/O)  log carbon-to-oxygen ratio at FIXED oxygen (fixed-O knob in vulcan_chem)
    * lnKzz     eddy diffusion scale
    * dT        uniform temperature shift (K; historically mislabeled T_int)
    * lnR0      reference-radius scaling at the bottom pressure (the standard xR_p
                transmission normalization nuisance; Batalha & Line 2017)
    * offset_g  a flat depth offset (ppm) for instrument group g. Default
                (offset_model="all_groups"): EVERY group gets one, the first included --
                no instrument is assumed absolutely depth-calibrated. lnR0 cannot stand
                in for the first group's offset: it is a physical radiative-transfer
                derivative, generally NOT constant in wavelength, so the calibration
                offset and lnR0 are distinct nuisance directions (the same P0-A
                distinction vulcan-jwst-tool fisher.py documents; 2026-07-12 recheck).
                Any exact or near redundancy this introduces is the job of the
                rank-aware whitened inversion below -- never of manually dropping a
                column. offset_model="reference_fixed" reproduces the pre-2026-07-20
                figures: the FIRST instrument group is assumed to define the absolute
                transit-depth baseline (G-1 relative offsets). Figure reproduction
                only -- under it, relabeling which instrument comes first changes the
                forecast.

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

OFFSET_MODELS = ("all_groups", "reference_fixed")


def build_design(tier_payload, wl_model, obs, use_lnR0=True, use_offsets=True,
                 offset_model="all_groups"):
    """Assemble the binned design matrix J_full and per-bin sigma for one tier.

    Columns: [lnZ, lnCO, lnKzz, dT] (chemistry) + [lnR0] + depth offsets per
    ``offset_model``:

      "all_groups" (default)  one offset for EVERY instrument group, the first
          included (a single-group analysis gets one global constant). No
          instrument is assumed absolutely calibrated; atmospheric information
          comes from wavelength-dependent structure only. Near-degeneracy with
          lnR0 is handled by the rank-aware inversion (rank_aware_cov), not by
          dropping a column.
      "reference_fixed"       the pre-2026-07-20 parameterization: the FIRST
          instrument group is assumed to define the absolute transit-depth
          baseline (offsets for groups[1:] only). Figure reproduction only.

    Returns dict(J, sigma, labels, keys, interest=(0,1), wl, group, depthM, ...).
    """
    if offset_model not in OFFSET_MODELS:
        raise ValueError(f"offset_model must be one of {OFFSET_MODELS}, got {offset_model!r}")
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
    if use_offsets:
        if offset_model == "all_groups":
            offset_groups = list(obs["groups"])
        else:
            offset_groups = list(obs["groups"][1:])
            print(f"[build_design] offset_model='reference_fixed': group "
                  f"'{obs['groups'][0]}' is assumed to define the absolute "
                  f"transit-depth baseline (figure-reproduction mode only)")
        for g in offset_groups:
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
                keep=keep, offset_model=(offset_model if use_offsets else None))


# Rank thresholds, ported one-to-one from vulcan-jwst-tool fisher.py (2026-07-12
# scale-invariance audit): the rank decision runs on the Jacobi-whitened
# (unit-diagonal, correlation-form) matrix, whose eigen-spectrum is invariant
# under per-parameter unit changes -- thresholding the raw mixed-unit matrix
# flipped finite constraints under a pure rescaling. eigh's noise floor is
# ~1e-16 x wmax; 1e-10 keeps margin either way.
REL_EIG_TOL = 1e-10
# A parameter whose L2 projection onto the null eigenvectors (whitened
# coordinates, basis-invariant) exceeds this lives partly in an unconstrained
# direction: its variance reads inf.
NULL_LOAD_TOL = 1e-6


def rank_aware_cov(F, diag=None):
    """Rank-aware (pseudo-)inverse of a Fisher matrix, in whitened coordinates.

    Full-rank F reproduces np.linalg.inv(F) up to round-off (the Jacobi
    whitening also preconditions the ~1e10 raw condition number this design
    mixes: lnR0's whitened column norm ~600 vs an offset's ~0.01). Whitened
    eigenvalues <= REL_EIG_TOL x the largest are treated as numerically
    unconstrained (null space): the covariance is built from the constrained
    eigenspace only, and any parameter loading on the null subspace (projection
    > NULL_LOAD_TOL) gets variance inf / covariance nan -- never a falsely
    precise finite number. Pass a dict as ``diag`` to receive
    rank / dimension / condition_number / eigenvalues (the whitened spectrum,
    scale-free) / null_load.
    """
    F = np.asarray(F, float)
    F = 0.5 * (F + F.T)
    n = F.shape[0]
    d = np.sqrt(np.clip(np.diag(F), 0.0, None))
    nz = d > 0.0
    if not nz.any():
        if diag is not None:
            diag.update(dimension=n, rank=0, condition_number=float("inf"),
                        eigenvalues=np.zeros(0), rel_eig_tol=REL_EIG_TOL,
                        null_load=np.ones(n))
        C = np.full((n, n), np.nan)
        np.fill_diagonal(C, np.inf)
        return C
    Fw = F[np.ix_(nz, nz)] / np.outer(d[nz], d[nz])
    w, V = np.linalg.eigh(0.5 * (Fw + Fw.T))
    wmax = float(w[-1]) if w.size else 0.0
    good = w > REL_EIG_TOL * max(wmax, 1e-300)
    load_nz = (np.sqrt(np.sum(V[:, ~good] ** 2, axis=1)) if (~good).any()
               else np.zeros(int(nz.sum())))
    if diag is not None:
        null_load = np.ones(n)
        null_load[nz] = load_nz
        diag.update(dimension=n, rank=int(good.sum()),
                    condition_number=(wmax / float(w[good].min()) if good.any()
                                      else float("inf")),
                    eigenvalues=w.copy(), rel_eig_tol=REL_EIG_TOL,
                    null_load=null_load)
    C = np.full((n, n), np.nan)
    np.fill_diagonal(C, np.inf)
    if good.any():
        Cw = (V[:, good] / w[good]) @ V[:, good].T
        Cnz = Cw / np.outer(d[nz], d[nz])
        bad_nz = load_nz > NULL_LOAD_TOL
        Cnz[bad_nz, :] = np.nan
        Cnz[:, bad_nz] = np.nan
        Cnz[np.diag_indices_from(Cnz)] = np.where(bad_nz, np.inf, np.diag(Cnz))
        C[np.ix_(nz, nz)] = Cnz
    return C


def fisher(J, sigma, diag=None):
    Ninv = 1.0 / np.asarray(sigma, float) ** 2
    F = J.T @ (Ninv[:, None] * J)
    return F, rank_aware_cov(F, diag=diag)


def marginal_cov(F, interest=(0, 1), diag=None):
    """(lnZ, lnCO) marginal covariance = sub-block of the rank-aware F^-1
    (all others marginalized; an interest parameter loading on a null
    direction reads inf variance, never a falsely precise finite number)."""
    C = rank_aware_cov(F, diag=diag)
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
    ln10 = np.log(10.0)
    ok_all = True
    for om in OFFSET_MODELS:
        des = build_design(tiers[tier], wl_model, obs, offset_model=om)
        J, sigma = des["J"], des["sigma"]
        dg = {}
        F, C = fisher(J, sigma, diag=dg)
        p = J.shape[1]
        print(f"=== zco_lib verify: tier={tier} combo={combo} bins={len(sigma)} "
              f"params={p} offset_model={om} ===")
        print("  columns:", ", ".join(des["keys"]))

        sym = np.max(np.abs(F - F.T)) / np.max(np.abs(F))
        print(f"  (1) F symmetric={sym:.1e}")
        w = dg["eigenvalues"]
        full_rank = dg["rank"] == dg["dimension"]
        print(f"  (2) whitened rank={dg['rank']}/{dg['dimension']}  "
              f"eigenvalues [{w.min():.3e}, {w.max():.3e}]  cond={dg['condition_number']:.2e}  "
              f"max null_load={dg['null_load'].max():.1e}")

        ratio = np.diag(C) / (1.0 / np.diag(F))
        print("  (3) marginal/conditional variance ratio >= 1: "
              + ", ".join(f"{des['keys'][i]}:{ratio[i]:.1f}" for i in range(p)))
        marg_ok = np.all(ratio >= 1 - 1e-9)

        rel = None
        if full_rank:
            rng = np.random.default_rng(0)
            M = C @ (J.T * (1.0 / sigma ** 2))
            n_mc = 60000
            theta_true = np.zeros(p)
            data = (J @ theta_true)[None, :] + rng.normal(size=(n_mc, sigma.size)) * sigma
            Cemp = np.cov((data @ M.T).T)
            rel = np.max(np.abs(Cemp - C) / np.sqrt(np.outer(np.diag(C), np.diag(C))))
            print(f"  (4) Monte-Carlo GLS cov vs F^-1: max rel err {rel:.4f} "
                  f"(expect ~{1/np.sqrt(n_mc):.4f})")
        else:
            print("  (4) Monte-Carlo GLS check SKIPPED: Fisher is rank-deficient "
                  "(the rank-aware covariance is a pseudo-inverse; quote the whitened "
                  "spectrum above)")

        C2, _ = marginal_cov(F)
        if np.all(np.isfinite(C2)):
            sZ = np.sqrt(C2[0, 0]) / ln10
            sCO = np.sqrt(C2[1, 1]) / ln10
            rho = C2[0, 1] / np.sqrt(C2[0, 0] * C2[1, 1])
            print(f"  marginal: sigma(log10 Z)={sZ:.3f} dex  sigma(log10 C/O)={sCO:.3f} dex  "
                  f"rho={rho:+.3f}")
        else:
            print("  marginal: (lnZ, lnCO) loads on a null direction -> unconstrained (inf)")
        ok = (sym < 1e-9) and marg_ok and (rel is None or rel < 0.05) \
            and np.all(np.isfinite(C2))
        ok_all = ok_all and ok
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    import sys
    sys.exit(verify())
