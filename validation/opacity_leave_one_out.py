"""Which absorbers does a case's spectrum need: leave-one-out on the case's own forward.

    python validation/opacity_leave_one_out.py --spec <run spec> [--molecules NH3 SO] [--states 8]
        [--from-posterior <npz>]

The screen belongs to a planet: ``--spec`` names its case.

An absorber the chemistry produces but the RT omits does not vanish: its opacity gets absorbed by whatever else can fit
it. For each state this solves the chemistry once, takes ``transmission_depth_r(..., wo_mols=...)`` and bins every row
on the real observation grid, so the difference is opacity alone.

Gate, per bin and state: |Delta binned depth| < min(5 ppm, 0.1 sigma). A species may be dropped only with a recorded
bound below the gate across the screened states: a thick cloud deck suppresses molecular features, so one cloud-free
state reads optimistic.
"""

from __future__ import annotations

import argparse
import logging
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
    """Return every molecule with an installed k-table that the case does not radiate.

    Inventoried at run time: a hard-coded list would stop testing a species once its table is installed. A table whose
    species the network does not solve makes build_pipeline raise.
    """
    from vulcan_forward import constants, exomolop

    return [m for m in constants.MOLECULES if m not in prod_mols and exomolop.table_path(m).is_file()]


def states_for(pipe, n_prior=0, posterior_path=None, seed=0):
    """Return the (K, n_dim) theta states to screen at.

    The nominal state (u = 0, the centre of every box) when its T-P profile is in the window (the WASP-39b case's is
    not). ``n_prior`` adds that many T-P-valid prior draws; ``posterior_path`` adds as many again from a finished run's
    nautilus_posterior.npz (4 when n_prior is 0).
    """
    import jax
    import jax.numpy as jnp

    nominal = np.asarray(pipe.theta_from_u(jnp.zeros(pipe.n_dim)), np.float64)
    out = [nominal] if bool(pipe.tp_valid(jnp.asarray(nominal))) else []
    if not out:
        print("[loo] the nominal state leaves the T-P window: not screened", flush=True)
    if n_prior:
        u = pipe.sample_prior_u(jax.random.PRNGKey(int(seed)), int(n_prior))
        out += list(np.asarray(jax.vmap(pipe.theta_from_u)(u), np.float64))
    if posterior_path:
        z = np.load(Path(posterior_path))
        names = [str(s) for s in z["param_names"]]
        if names != list(pipe.names):
            raise RuntimeError(
                f"posterior parameter order {names} does not match this "
                f"configuration {list(pipe.names)}: it is a different target"
            )
        draws = np.asarray(z["samples"], np.float64).reshape(-1, len(names))
        rng = np.random.default_rng(int(seed))
        k = min(int(n_prior) or 4, draws.shape[0])
        out += list(draws[rng.choice(draws.shape[0], k, replace=False)])
    return np.asarray(out, np.float64)


def _aux_for(pipe, theta):
    """Return (aux, lnR0, cloud, lnM, certified) at one state, from its one chemistry solve."""
    import jax.numpy as jnp

    th = jnp.asarray(theta, pipe.dtype)
    chem_theta = th[: pipe.n_chem_tp]
    y, diag = pipe.fwd.chem_solve_cold_diag(chem_theta)
    certified = bool(diag.conv_normal)
    if not certified:
        # vulcan3.state.TERM_*: 3 is the step cap, 4 a column frozen at dt_min, 5 a non-finite state.
        print(
            f"[loo] stop reason {int(diag.termination_reason)} after {int(diag.accept_count)} accepted steps",
            flush=True,
        )
    return pipe.fwd.aux_from_y(y, chem_theta), *pipe.rt_args(th), certified


def deltas_at(pipe, theta, candidates):
    """(n_cand, n_bin) |binned depth with each candidate removed - full|, or None
    when the state's chemistry does not certify cold (not a model state).
    """
    aux, lnR0, cloud, lnM, certified = _aux_for(pipe, theta)
    if not certified:
        return None
    vmr, vmr_h2, vmr_he, T_art, mmw_art = aux
    full, wo = pipe.fwd.rt.transmission_depth_r(
        vmr, vmr_h2, T_art, mmw_art, lnR0, vmr_he=vmr_he, wo_mols=list(candidates), **pipe.fwd.rt_kwargs(cloud, lnM)
    )
    B = np.asarray(pipe.B, np.float64)
    return np.abs(np.asarray(wo, np.float64) @ B.T - (B @ np.asarray(full, np.float64))[None, :])


def _screen(pipe, states, candidates):
    """(worst over the certified states, per candidate and bin; how many certified): one solve and RT per state."""
    per_state = []
    for i, th in enumerate(states):
        t0 = time.time()
        d = deltas_at(pipe, th, candidates)
        if d is None:
            print(f"[loo] state {i}: chemistry not certified cold, skipped", flush=True)
            continue
        if not np.all(np.isfinite(d)):
            # max() would carry the NaN and `NaN > gate` reads as a pass
            raise RuntimeError(f"state {i} {th.tolist()}: certified but a non-finite leave-one-out spectrum")
        per_state.append(d)
        print(f"[loo] state {i}: {len(candidates)} rungs ({time.time() - t0:.0f}s)", flush=True)
    if not per_state:
        raise RuntimeError("no screened state certified: nothing to gate")
    return np.max(per_state, axis=0), len(per_state)


def _measure(worst_over_states, n_screened, candidates, omitted, gate, sigma, wl_bin):
    """(table rows, artifact measurements, omitted candidates over the gate), printing one line per candidate."""
    rows, measurements, worst = [], [], []
    for c, m in enumerate(candidates):
        d = worst_over_states[c]
        i = int(np.argmax(d))
        over = int(np.sum(d > gate))
        rows.append((m, 1e6 * d[i], wl_bin[i], d[i] / sigma[i], 1e6 * float(np.sqrt(np.mean(d**2))), over))
        print(
            f"[loo] {m:5s} max {1e6 * d[i]:8.3f} ppm @ {wl_bin[i]:.5f} um  "
            f"{d[i] / sigma[i]:.3f} sigma  rms {1e6 * np.sqrt(np.mean(d**2)):7.3f} ppm  "
            f"{over}/{d.size} bins over gate",
            flush=True,
        )
        measurements.append(
            {
                "name": f"omit {m}: max |Delta binned depth| over {n_screened} state(s)",
                "value": f"{1e6 * d[i]:.3f} ppm",
                "value_raw": float(1e6 * d[i]),
                "unit": "ppm",
                "gate": "< min(5 ppm, 0.1 sigma) in every bin",
                # only an omitted absorber over the gate is a failure
                "decisive": bool(m in omitted),
                "passed": not bool(over and m in omitted),
            }
        )
        if over and m in omitted:
            worst.append(m)
    return rows, measurements, worst


def _verdict(rows, worst, omitted, n_screened):
    """The summary: which omitted absorbers are required, and which of the case's carry no measurable effect."""
    idle = [m for m, _, _, _, _, over in rows if not over and m not in omitted]
    msg = (
        f"Every OMITTED absorber tested is below the per-bin gate across "
        f"{n_screened} state(s); none is required beyond the case's list."
        if not worst
        else f"REQUIRED but omitted: {', '.join(worst)}. Add them to the case's "
        "molecule list and rerun the posterior -- their opacity is currently "
        "absorbed by the fitted parameters."
    )
    if idle:
        msg += (
            f" The case's absorbers under the gate at this state set "
            f"({', '.join(idle)}) are carried without measurable effect -- "
            "informational, not a failure; the gate only refuses OMISSIONS."
        )
    return msg


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--molecules",
        nargs="*",
        default=None,
        help="candidates to test (default: every installed table the case does not radiate)",
    )
    ap.add_argument(
        "--states",
        type=int,
        default=0,
        help="prior draws on top of the nominal state (and the same number from --from-posterior)",
    )
    ap.add_argument("--from-posterior", default=None, help="nautilus_posterior.npz to draw representative states from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec", required=True, help="the run spec to screen")
    args = ap.parse_args()

    from retrieval_framework import config_schema as C

    spec = Path(args.spec).resolve()
    cfg, results = _artifact.spec_case(spec)
    C.export_network(cfg)  # before vulcan3 is imported
    from vulcan_forward import vulcan_chem  # noqa: F401  # before exojax
    from retrieval_framework.pipeline import build_pipeline

    prod_mols = list(cfg.molecules)
    named = args.molecules
    extra = [m for m in named if m not in prod_mols] if named is not None else unscreened_candidates(prod_mols)
    baseline_mols = prod_mols + extra
    # An omitted candidate over the gate is a defect; one of the case's over the
    # gate is the list working as intended. Screening both is useful, but they
    # are opposite verdicts and must not share one.
    candidates = list(named) if named is not None else (extra or prod_mols)
    omitted = set(extra)

    # The case's forward at the augmented molecule list: parametric T-P,
    # two-stage cold solve, the run's own binning matrix and sigmas.
    pipe = build_pipeline(replace(cfg, molecules=tuple(baseline_mols)))
    profile = pipe.cfg.profile()
    sigma = np.asarray(pipe.obs["sigma"], np.float64)
    wl_bin = np.asarray(pipe.obs["wl"], np.float64)
    gate = np.minimum(GATE_PPM * 1e-6, GATE_SIGMA_FRAC * sigma)

    states = states_for(pipe, args.states, args.from_posterior, args.seed)
    print(f"[loo] baseline {len(baseline_mols)} absorbers: {baseline_mols}", flush=True)
    print(
        f"[loo] testing {len(candidates)} over {len(states)} state(s); "
        f"{len(omitted & set(candidates))} of them omitted from the case",
        flush=True,
    )

    # one chemistry solve + one folded RT pass per state gives every candidate
    worst_over_states, n_screened = _screen(pipe, states, candidates)
    rows, measurements, worst = _measure(worst_over_states, n_screened, candidates, omitted, gate, sigma, wl_bin)

    print(f"\n{'species':>8} {'max ppm':>10} {'at um':>10} {'/sigma':>8} {'rms ppm':>9} {'bins over':>10}")
    for m, mx, w, s, rms, over in sorted(rows, key=lambda r: -r[1]):
        print(f"{m:>8} {mx:10.3f} {w:10.5f} {s:8.3f} {rms:9.3f} {over:10d}")

    ok = not worst
    msg = _verdict(rows, worst, omitted, n_screened)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} -- {msg}")
    _artifact.emit(
        name=f"opacity_leave_one_out_{spec.stem}",
        title="Leave-one-out absorber screen on the case's observation grid",
        measurements=measurements,
        status="PASS" if ok else "FAIL",
        summary=msg,
        resolved_config=profile,
        out_dir=results,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
