"""Numerical convergence of the binned transit depth in ``art_nlayer``, on the production RT path.

    python validation/resolution_ladder.py [--jacobian]

Transmission is nonlinear in optical depth, so a grid that does not resolve the physics does not average into the
correct R=100 bin. The band grid and quadrature come from the tables, so the free axis is the ART vertical grid. The
chemistry converges once, then the RT is rebuilt at each rung and every spectrum binned on the same bins.

Gates: the production rung changes by < 5 ppm against the next rung, and its Jacobian direction by < 1% where the depth
response is significant.
"""

from __future__ import annotations

import itertools
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# PYTHONSAFEPATH strips the script's own directory from sys.path in some
# sandboxes, so be explicit rather than relying on it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifact

GATE_PPM = 5.0
GATE_JAC_REL = 0.01
BIN_R = 100.0
BAND = (1900.0, 9900.0)  # production retrieval band (cm^-1)
LADDER = (67, 101, 151)  # art_nlayer rungs; the lowest must be production's


def production_pair(rungs, production_value, knob="art_nlayer"):
    """Return the production/next-rung pair or fail a mis-specified ladder."""
    ordered = sorted({int(r) for r in rungs})
    if len(ordered) < 2:
        raise ValueError("resolution ladder needs at least two distinct rungs")
    if ordered[0] != int(production_value):
        raise ValueError(f"lowest rung must be production {knob}={production_value}, got {ordered[0]}")
    return ordered, (ordered[0], ordered[1])


def _run_rung(prof, want_jac, chem, y0, theta0, edges):
    """Build the RT for one profile and return its R=100 binned depth and, optionally, the binned d(depth)/dlnZ."""
    from vulcan_forward import constants, exojax_rt, interp_map
    import jax
    import jax.numpy as jnp

    rt = exojax_rt.build_rt_model(prof)
    to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)
    he, h2 = chem.sidx["He"], chem.sidx[constants.BULK_H2_VULCAN]

    def depth_of(y):
        ymix = y / jnp.sum(y, axis=1, keepdims=True)
        mmw = to_art(ymix @ chem.species_masses)
        vmr = {k: to_art(ymix[:, chem.sidx[constants.MOLECULES[k]["vulcan"]]]) for k in rt.molecules}
        T_art = to_art(jnp.asarray(chem.T_base))
        return rt.transmission_depth(vmr, to_art(ymix[:, h2]), T_art, mmw, vmr_he=to_art(ymix[:, he]))

    d = np.asarray(depth_of(y0), np.float64)
    wl = np.asarray(rt.wl_um, np.float64)
    o = np.argsort(wl)
    wl, d = wl[o], d[o]
    entry = {"binned": _artifact.bin_trapz(wl, d, edges)}
    if want_jac:

        def f(th):
            return depth_of(chem.converged_y(th, warm_y=y0, lnZ_ref=0.0, c_o_ref=0.0))

        _, jd = jax.jvp(f, (theta0,), (jnp.array([1.0, 0.0, 0.0, 0.0]),))
        entry["jac_lnZ"] = _artifact.bin_trapz(wl, np.asarray(jd, np.float64)[o], edges)
    return entry


def _depth_gates(results, rungs, decisive_pair, knob):
    """(measurements, ok): each rung against the next; only the production pair is a gate."""
    ok = True
    measurements = []
    print("\n==== binned-depth convergence (vs next rung) ====")
    for a, b in itertools.pairwise(rungs):
        dppm = _artifact.worst_ppm(results[a]["binned"], results[b]["binned"])
        print(f"{knob} {a} -> {b}: max |Delta binned depth| = {dppm:.2f} ppm")
        decisive = (a, b) == decisive_pair
        measurements.append(
            {
                "name": f"max |Delta binned depth|, {knob} {a} -> {b}",
                "value": f"{dppm:.2f} ppm",
                "value_raw": float(dppm),
                "unit": "ppm",
                "gate": f"< {GATE_PPM} ppm" if decisive else "(informational)",
                "decisive": decisive,
                "passed": bool(dppm < GATE_PPM) if decisive else None,
            }
        )
        if decisive:
            ok &= dppm < GATE_PPM
    return measurements, ok


def _jacobian_gate(results, decisive_pair, knob):
    """(measurement, ok): the production pair's dlnZ Jacobian, in the bins where the response is significant."""
    print("\n==== Jacobian (dlnZ) convergence ====")
    a, b = decisive_pair
    ja, jb = results[a]["jac_lnZ"], results[b]["jac_lnZ"]
    if np.any(np.isfinite(ja) != np.isfinite(jb)):
        jrel = float("inf")
    else:
        m = np.isfinite(jb) & (np.abs(jb) > 0.01 * np.nanmax(np.abs(jb)))
        jrel = float(np.max(np.abs(ja[m] - jb[m]) / np.abs(jb[m])))
    print(f"{knob} {a} -> {b}: max rel Jacobian change (significant bins) = {jrel:.3%}")
    return (
        {
            "name": f"max rel Jacobian (dlnZ) change, {knob} {a} -> {b}",
            "value": f"{jrel:.3%}",
            "value_raw": float(jrel),
            "unit": "relative",
            "gate": f"< {GATE_JAC_REL:.0%}",
            "decisive": True,
            "passed": bool(jrel < GATE_JAC_REL),
        },
        jrel < GATE_JAC_REL,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("--jacobian", action="store_true", help="also compare d(binned)/dlnZ per rung (jvp; expensive)")
    args = ap.parse_args()
    knob = "art_nlayer"

    # import order matters: vulcan_chem before exojax
    from vulcan_forward import vulcan_chem
    import jax.numpy as jnp

    # The production case IS the specification: nz, art_nlayer, art_ptop_bar,
    # molecules, yconv_cri, count_max and dt_max all come from it. The state is
    # the baseline column at the VULCAN baseline T-P (no Guillot hook).
    profile = _artifact.production_profile(nu_min=BAND[0], nu_max=BAND[1])
    chem = vulcan_chem.build_chem_model(profile)
    theta0 = jnp.zeros(4, dtype=jnp.float64)
    y0 = chem.converged_y(theta0)

    edges = _artifact.make_r_bins(1e4 / BAND[1], 1e4 / BAND[0], BIN_R)

    results = {}
    for rung in LADDER:
        t0 = time.time()
        prof = dict(profile)
        prof[knob] = int(rung)
        results[rung] = _run_rung(prof, args.jacobian, chem, y0, theta0, edges)
        print(f"[ladder] {knob}={rung}: {time.time() - t0:.0f}s", flush=True)

    try:
        rungs, decisive_pair = production_pair(results, profile[knob], knob)
    except ValueError as exc:
        raise SystemExit(f"resolution_ladder: {exc}") from exc
    measurements, ok = _depth_gates(results, rungs, decisive_pair, knob)
    if args.jacobian:
        measurement, jac_ok = _jacobian_gate(results, decisive_pair, knob)
        measurements.append(measurement)
        ok &= jac_ok
    # Without --jacobian the depth gate alone cannot certify gradient convergence,
    # so a clean run is REPORT, not PASS.
    status = ("PASS" if args.jacobian else "REPORT") if ok else "FAIL"
    print(
        f"\nVERDICT: {status} (production-rung gate {GATE_PPM} ppm"
        + (f", Jacobian {GATE_JAC_REL:.0%}" if args.jacobian else "")
        + ")"
    )

    # Archive the verdict with its provenance.
    _artifact.emit(
        name="resolution_ladder",
        title="Vertical-grid (art_nlayer) convergence of the binned depth",
        measurements=measurements,
        status=status,
        summary=(
            f"opacity_mode=exomolop; {knob} ladder {rungs}; "
            f"production is {profile[knob]}. "
            + (
                "The production resolution is converged at the declared gates."
                if ok
                else f"NOT converged at the declared gates -- adopt the lowest "
                f"tested passing rung as the production {knob}."
            )
            + (
                " Jacobian axis included."
                if args.jacobian
                else " Jacobian axis NOT run (--jacobian); the depth gate alone does not certify gradient convergence."
            )
        ),
        resolved_config=profile,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
