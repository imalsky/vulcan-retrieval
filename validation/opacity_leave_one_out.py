#!/usr/bin/env python3
"""Which absorbers does the production spectrum actually need?

The chemistry solves ~90 species; 25 of them have an installed ExoMolOP k-table.
Production radiates only some of them, and an absorber the chemistry PRODUCES but
the RT omits does not vanish -- its opacity gets absorbed by whatever else can
fit it (metallicity, C/O, the cloud deck, the instrument offsets).

Leave-one-out on the PRODUCTION forward (`build_pipeline`, so the parametric T-P
and the two-stage cold solve are the real ones): for each state, solve the
chemistry ONCE, then take the engine's own leave-one-out observable
(`transmission_depth_r(..., wo_mols=...)`, bit-identical to a from-scratch solve
with that VMR zeroed but sharing the correlated-k fold prefix) and bin every row
onto the real observation grid with the run's own binning matrix. Chemistry and
continuum are untouched between rungs, so the difference is opacity alone.

    python validation/opacity_leave_one_out.py                  # every unscreened table
    python validation/opacity_leave_one_out.py --molecules NH3 SO
    python validation/opacity_leave_one_out.py --states 8       # + 8 prior draws
    python validation/opacity_leave_one_out.py --from-posterior out/posterior_samples.npz

GATE (per bin, per state): |Delta binned depth| < min(5 ppm, 0.1 * quoted sigma).
A species may be dropped from production only with a recorded bound BELOW the gate
across the screened state set -- not by chemical intuition, and not from one state:
a thick cloud deck suppresses molecular features, so a single cloud-free state
reads optimistic.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifact

GATE_PPM = 5.0
GATE_SIGMA_FRAC = 0.1

def unscreened_candidates(prod_mols):
    """Every molecule with an INSTALLED k-table that production does not radiate.

    Inventoried at run time, not listed here: RC-02 asks for the omitted set to
    be discovered, and a hard-coded list silently stops testing a species the
    day someone installs its table. A table whose species the network does not
    solve makes build_pipeline raise, which is the intended loud failure.
    """
    from vulcan_forward import constants, exomolop
    return [m for m in constants.MOLECULES
            if m not in prod_mols and exomolop.table_path(m).is_file()]


def states_for(pipe, n_prior=0, posterior_path=None, seed=0):
    """(K, n_dim) theta states to screen at.

    Always includes the nominal state (u=0, the centre of every declared box).
    ``n_prior`` adds that many T-P-valid prior draws; ``posterior_path`` adds the
    SAME number again from a finished run's posterior_samples.npz (4 when
    n_prior is 0). RC-02 requires the screen over a prior/posterior state set,
    not one state.
    """
    import jax
    import jax.numpy as jnp

    out = [np.asarray(pipe.theta_from_u(jnp.zeros(pipe.n_dim)), np.float64)]
    if n_prior:
        u = pipe.sample_prior_u(jax.random.PRNGKey(int(seed)), int(n_prior))
        out += list(np.asarray(jax.vmap(pipe.theta_from_u)(u), np.float64))
    if posterior_path:
        z = np.load(Path(posterior_path))
        names = [str(s) for s in z["param_names"]]
        if names != list(pipe.names):
            raise RuntimeError(
                f"posterior parameter order {names} does not match this "
                f"configuration {list(pipe.names)}: it is a different target")
        draws = np.asarray(z["samples"], np.float64).reshape(-1, len(names))
        rng = np.random.default_rng(int(seed))
        k = min(int(n_prior) or 4, draws.shape[0])
        out += list(draws[rng.choice(draws.shape[0], k, replace=False)])
    return np.asarray(out, np.float64)


def _aux_for(pipe, theta):
    """(aux, lnR0, cloud) at one state -- the chemistry solve, done ONCE."""
    import jax.numpy as jnp
    th = jnp.asarray(theta)
    chem_theta = th[:pipe.n_chem_tp]
    lnR0 = th[pipe.lnR0_idx] if pipe.lnR0_idx is not None else jnp.asarray(0.0)
    cloud = (th[pipe.cloud_idx[0]:pipe.cloud_idx[0] + pipe.n_cloud]
             if pipe.n_cloud else None)
    y = pipe.fwd.chem_solve_cold(chem_theta)
    return pipe.fwd.aux_from_y(y, chem_theta), lnR0, cloud


def deltas_at(pipe, theta, candidates):
    """(n_cand, n_bin) |binned depth with each candidate removed - full|."""
    aux, lnR0, cloud = _aux_for(pipe, theta)
    vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
    full, wo = pipe.fwd.rt.transmission_depth_r(
        vmr, vmr_h2, T_art, mmw_art, lnR0, vmr_he=vmr_he, cloud=cloud,
        wo_mols=list(candidates))
    B = np.asarray(pipe.B, np.float64)
    return np.abs(np.asarray(wo, np.float64) @ B.T
                  - (B @ np.asarray(full, np.float64))[None, :])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="*", default=None,
                    help="candidates to test (default: every installed table "
                         "production does not radiate)")
    ap.add_argument("--all-candidates", action="store_true",
                    help="test every molecule in the baseline, production included")
    ap.add_argument("--states", type=int, default=0,
                    help="prior draws on top of the nominal state (and the same "
                         "number from --from-posterior)")
    ap.add_argument("--from-posterior", default=None,
                    help="posterior_samples.npz to draw representative states from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-artifact", action="store_true")
    args = ap.parse_args()

    # import order is load-bearing: vulcan_chem sets the import-frozen env vars
    from vulcan_forward import vulcan_chem  # noqa: F401
    from retrieval_framework.pipeline import build_pipeline

    cfg = _artifact.production_config()
    prod_mols = list(cfg.molecules)
    named = args.molecules
    extra = ([m for m in named if m not in prod_mols] if named is not None
             else unscreened_candidates(prod_mols))
    baseline_mols = prod_mols + extra
    # An OMITTED candidate over the gate is a defect; a PRODUCTION one over the
    # gate is the list working as intended. Screening both is useful, but they
    # are opposite verdicts and must not share one.
    candidates = (baseline_mols if args.all_candidates
                  else list(named) if named is not None
                  else (extra or prod_mols))
    omitted = set(extra)

    # The PRODUCTION forward at the augmented molecule list: parametric T-P,
    # two-stage cold solve, the run's own binning matrix and sigmas.
    pipe = build_pipeline(replace(cfg, molecules=tuple(baseline_mols)))
    profile = pipe.cfg.profile()
    sigma = np.asarray(pipe.obs["sigma"], np.float64)
    wl_bin = np.asarray(pipe.obs["wl"], np.float64)
    gate = np.minimum(GATE_PPM * 1e-6, GATE_SIGMA_FRAC * sigma)

    states = states_for(pipe, args.states, args.from_posterior, args.seed)
    print(f"[loo] baseline {len(baseline_mols)} absorbers: {baseline_mols}", flush=True)
    print(f"[loo] testing {len(candidates)} over {len(states)} state(s); "
          f"{len(omitted & set(candidates))} of them omitted from production",
          flush=True)

    # one chemistry solve + one folded RT pass per state gives every candidate
    per_state = []
    for i, th in enumerate(states):
        t0 = time.time()
        per_state.append(deltas_at(pipe, th, candidates))
        print(f"[loo] state {i}: {len(candidates)} rungs "
              f"({time.time() - t0:.0f}s)", flush=True)
    worst_over_states = np.max(per_state, axis=0)          # (n_cand, n_bin)

    rows, measurements, worst = [], [], []
    for c, m in enumerate(candidates):
        d = worst_over_states[c]
        i = int(np.argmax(d))
        over = int(np.sum(d > gate))
        rows.append((m, 1e6 * d[i], wl_bin[i], d[i] / sigma[i],
                     1e6 * float(np.sqrt(np.mean(d ** 2))), over))
        print(f"[loo] {m:5s} max {1e6*d[i]:8.3f} ppm @ {wl_bin[i]:.5f} um  "
              f"{d[i]/sigma[i]:.3f} sigma  rms {1e6*np.sqrt(np.mean(d**2)):7.3f} ppm  "
              f"{over}/{d.size} bins over gate", flush=True)
        measurements.append({
            "name": f"omit {m}: max |Delta binned depth| over {len(states)} state(s)",
            "value": f"{1e6*d[i]:.3f} ppm", "value_raw": float(1e6 * d[i]),
            "unit": "ppm", "gate": "< min(5 ppm, 0.1 sigma) in every bin",
            # only an OMITTED absorber over the gate is a failure
            "decisive": bool(m in omitted),
            "passed": not bool(over and m in omitted),
        })
        if over and m in omitted:
            worst.append(m)

    print(f"\n{'species':>8} {'max ppm':>10} {'at um':>10} {'/sigma':>8} "
          f"{'rms ppm':>9} {'bins over':>10}")
    for m, mx, w, s, rms, over in sorted(rows, key=lambda r: -r[1]):
        print(f"{m:>8} {mx:10.3f} {w:10.5f} {s:8.3f} {rms:9.3f} {over:10d}")

    idle = [m for m, _, _, _, _, over in rows if not over and m not in omitted]
    ok = not worst
    msg = (f"Every OMITTED absorber tested is below the per-bin gate across "
           f"{len(states)} state(s); none is required beyond the production list."
           if ok else
           f"REQUIRED but omitted: {', '.join(worst)}. Add them to the case's "
           "molecule list and rerun the posterior -- their opacity is currently "
           "absorbed by the fitted parameters.")
    if idle:
        msg += (f" Production absorbers under the gate at this state set "
                f"({', '.join(idle)}) are carried without measurable effect -- "
                "informational, not a failure; the gate only refuses OMISSIONS.")
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    if not args.no_artifact:
        _artifact.emit(
            name="opacity_leave_one_out",
            title="Leave-one-out absorber screen on the production observation grid",
            measurements=measurements, status="PASS" if ok else "FAIL",
            summary=msg, resolved_config=profile)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
