#!/usr/bin/env python3
"""Numerical convergence of the BINNED transit depth, on the production RT path.

Transmission is nonlinear in optical depth (T = e^-tau), so a grid that does not
resolve the physics does not average into the correct R~100 bin -- narrow
saturated lines and low-opacity windows must be resolved, or treated with a
validated k-distribution.

WHICH KNOB THIS SWEEPS: the production RT is correlated-k on the tables' own
R=1000 band grid and their own 16-point quadrature. There is no spectral
resolution knob -- both come from the files -- so the remaining free numerical
axis is the ART VERTICAL grid, ``art_nlayer``, and that is what the ladder
sweeps. (The retired sampled-line-by-line alternative was measured NOT to
converge on this band; the numbers are archived in notes.md.)

Method: converge the W39b chemistry ONCE (baseline theta), then rebuild the RT at
each rung, bin every native spectrum onto the SAME R=100 bins, and compare
adjacent rungs. Optionally convolve with a Gaussian LSF (--lsf-r) before binning
to show LSF insensitivity at R=100 products, and optionally check a chemistry
Jacobian column (--jacobian: d(binned depth)/d lnZ via a warm-started jvp per
rung).

Run on the GPU node (primal RT only -- the vjp memory wall does not apply):

    python validation/resolution_ladder.py

PASS gates (from the review): the production rung must change by < 5 ppm against
the next rung, and its Jacobian direction by < 1% where the depth response is
significant. Testing only the two finest rungs says nothing about whether the
much coarser production rung is adequate.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# PYTHONSAFEPATH strips the script's own directory from sys.path in some
# sandboxes, so be explicit rather than relying on it.
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import _artifact  # noqa: E402

GATE_PPM = 5.0
GATE_JAC_REL = 0.01
BIN_R = 100.0
BAND = (1900.0, 9900.0)          # production retrieval band (cm^-1)


def gaussian_lsf(wl, y, R_lsf):
    """Gaussian LSF of resolving power R_lsf applied in ln-lambda (host-side)."""
    ln = np.log(wl)
    s = 1.0 / (R_lsf * 2.3548200450309493)      # FWHM = 1/R in ln-lambda
    out = np.empty_like(y)
    for i, l0 in enumerate(ln):
        w = np.exp(-0.5 * ((ln - l0) / s) ** 2)
        out[i] = np.sum(w * y) / np.sum(w)
    return out


def production_pair(rungs, production_value, knob="art_nlayer"):
    """Return the production/next-rung pair or fail a mis-specified ladder."""
    ordered = sorted({int(r) for r in rungs})
    if len(ordered) < 2:
        raise ValueError("resolution ladder needs at least two distinct rungs")
    if ordered[0] != int(production_value):
        raise ValueError(
            f"lowest rung must be production {knob}={production_value}, "
            f"got {ordered[0]}")
    return ordered, (ordered[0], ordered[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=int, nargs="+", default=[67, 101, 151],
                    help="art_nlayer rungs; the lowest must be the production "
                         "value (default: 67 101 151)")
    ap.add_argument("--lsf-r", type=float, default=0.0,
                    help="optional Gaussian LSF resolving power before binning")
    ap.add_argument("--jacobian", action="store_true",
                    help="also compare d(binned)/dlnZ per rung (jvp; expensive)")
    ap.add_argument("--no-artifact", action="store_true",
                    help="skip writing the provenance-bearing result under "
                         "validation/results/ (exploration only)")
    args = ap.parse_args()
    knob = "art_nlayer"

    from retrieval_framework.forward import config
    from vulcan_forward import interp_map
    # import order is load-bearing: vulcan_chem before exojax
    from vulcan_forward import vulcan_chem
    from vulcan_forward import exojax_rt
    import jax
    import jax.numpy as jnp

    # The production case IS the specification: nz, art_nlayer, art_ptop_bar,
    # molecules, yconv_cri, count_max and dt_max all come from it, so this ladder
    # can never certify a model production does not run.
    profile = _artifact.production_profile(nu_min=BAND[0], nu_max=BAND[1])
    chem = vulcan_chem.build_chem_model(profile)
    theta0 = jnp.zeros(4, dtype=jnp.float64)
    y0 = chem.converged_y(theta0)
    he, h2 = chem.sidx["He"], chem.sidx[config.BULK_H2_VULCAN]

    edges = _artifact.make_r_bins(1e4 / BAND[1], 1e4 / BAND[0], BIN_R)

    def run_rung(prof, want_jac):
        """Build the RT for one profile and return its R=100 binned depth (and,
        optionally, the binned d(depth)/dlnZ from a warm-started jvp)."""
        rt = exojax_rt.build_rt_model(prof)
        to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)

        def depth_of(y):
            ymix = y / jnp.sum(y, axis=1, keepdims=True)
            mmw = to_art(ymix @ chem.species_masses)
            vmr = {k: to_art(ymix[:, chem.sidx[config.MOLECULES[k]["vulcan"]]])
                   for k in rt.molecules}
            T_art = to_art(jnp.asarray(chem.T_base))
            return rt.transmission_depth(vmr, to_art(ymix[:, h2]), T_art, mmw,
                                         vmr_he=to_art(ymix[:, he]))

        d = np.asarray(depth_of(y0), np.float64)
        wl = np.asarray(rt.wl_um, np.float64)
        o = np.argsort(wl); wl, d = wl[o], d[o]
        if args.lsf_r > 0:
            d = gaussian_lsf(wl, d, args.lsf_r)
        entry = dict(binned=_artifact.bin_trapz(wl, d, edges))
        if want_jac:
            def f(th):
                return depth_of(chem.converged_y(th, warm_y=y0,
                                                 lnZ_ref=0.0, c_o_ref=0.0))
            _, jd = jax.jvp(f, (theta0,), (jnp.array([1.0, 0.0, 0.0, 0.0]),))
            entry["jac_lnZ"] = _artifact.bin_trapz(wl, np.asarray(jd, np.float64)[o],
                                                   edges)
        return entry

    results = {}
    for rung in sorted(args.ladder):
        t0 = time.time()
        prof = dict(profile); prof[knob] = int(rung)
        results[rung] = run_rung(prof, args.jacobian)
        print(f"[ladder] {knob}={rung}: {time.time()-t0:.0f}s", flush=True)

    try:
        rungs, decisive_pair = production_pair(results, profile[knob], knob)
    except ValueError as exc:
        raise SystemExit(f"resolution_ladder: {exc}") from exc
    ok = True
    measurements = []
    print("\n==== binned-depth convergence (vs next rung) ====")
    for a, b in zip(rungs[:-1], rungs[1:]):
        da, db = results[a]["binned"], results[b]["binned"]
        m = np.isfinite(da) & np.isfinite(db)
        dppm = 1e6 * np.max(np.abs(da[m] - db[m]))
        print(f"{knob} {a} -> {b}: max |Delta binned depth| = {dppm:.2f} ppm")
        decisive = (a, b) == decisive_pair
        measurements.append({
            "name": f"max |Delta binned depth|, {knob} {a} -> {b}",
            "value": f"{dppm:.2f} ppm",
            "value_raw": float(dppm), "unit": "ppm",
            "gate": f"< {GATE_PPM} ppm" if decisive else "(informational)",
            "decisive": decisive,
            "passed": bool(dppm < GATE_PPM) if decisive else None,
        })
        if decisive:
            ok &= dppm < GATE_PPM
    if args.jacobian:
        print("\n==== Jacobian (dlnZ) convergence ====")
        a, b = decisive_pair
        ja, jb = results[a]["jac_lnZ"], results[b]["jac_lnZ"]
        m = np.isfinite(ja) & np.isfinite(jb) & (np.abs(jb) > 0.01 * np.nanmax(np.abs(jb)))
        jrel = float(np.max(np.abs(ja[m] - jb[m]) / np.abs(jb[m])))
        print(f"{knob} {a} -> {b}: max rel Jacobian change (significant bins) = {jrel:.3%}")
        measurements.append({
            "name": f"max rel Jacobian (dlnZ) change, {knob} {a} -> {b}",
            "value": f"{jrel:.3%}", "value_raw": float(jrel), "unit": "relative",
            "gate": f"< {GATE_JAC_REL:.0%}", "decisive": True,
            "passed": bool(jrel < GATE_JAC_REL),
        })
        ok &= jrel < GATE_JAC_REL
    # Without --jacobian the depth gate alone cannot certify gradient convergence,
    # so a clean run is REPORT, not PASS. certificate.validate refuses REPORT.
    status = ("PASS" if args.jacobian else "REPORT") if ok else "FAIL"
    print(f"\nVERDICT: {status} (production-rung gate {GATE_PPM} ppm"
          + (f", Jacobian {GATE_JAC_REL:.0%}" if args.jacobian else "") + ")")

    # A verdict printed to a terminal and lost is not evidence. Archive it with
    # enough provenance to tie the number to an exact code and data state.
    if not args.no_artifact:
        _artifact.emit(
            name="resolution_ladder",
            title="Vertical-grid (art_nlayer) convergence of the binned depth",
            measurements=measurements,
            status=status,
            summary=(
                f"opacity_mode=exomolop; {knob} ladder {rungs}; "
                f"production is {profile[knob]}. "
                + ("The production resolution is converged at the declared "
                   "gates." if ok else
                   f"NOT converged at the declared gates -- adopt the lowest "
                   f"tested passing rung as the production {knob}.")
                + (" Jacobian axis included." if args.jacobian else
                   " Jacobian axis NOT run (--jacobian); the depth gate alone "
                   "does not certify gradient convergence.")),
            resolved_config=profile,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
