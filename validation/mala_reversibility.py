#!/usr/bin/env python3
"""Warm-cap reversibility probe for the MALA-within-SMC mutation kernel.

The warm mutation solve is cut off at warm_count_max. Rejecting a capped
proposal is a valid MH move ONLY if the cap does not make the transition
support state-dependent: a proposal B that converges when approached from
particle A's column but caps out when approached from B's own neighborhood (or
vice versa) breaks the symmetry the Langevin MH correction assumes. This probe
measures that directly on a finished run's checkpointed cloud:

  * pick K nearest-neighbor particle pairs (i, j) in u-space;
  * warm-solve theta_j FROM particle i's carried column, and theta_i FROM
    particle j's (the two directions of one virtual move);
  * classify each direction (converged-within-cap / capped) and compare the
    reached likelihoods.

PASS: no asymmetric convergence classification (one direction capped, the other
not) and |L(fwd) - L(carried)| consistent with validate_warm's gate. Any
asymmetric pair is listed -- if they appear at production settings, either raise
warm_count_max or run the final ladder stages with smc_chem_mode="cold".

    SMC_RETRIEVAL_PRESET=gpu python validation/mala_reversibility.py runs/w39b_smc_retrieval --pairs 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="runs/w39b_smc_retrieval")
    ap.add_argument("--pairs", type=int, default=24)
    args = ap.parse_args()

    from retrieval_framework.run_smc import make_config
    from retrieval_framework import pipeline as P
    import jax
    import jax.numpy as jnp

    cfg, preset = make_config(Path(args.run_dir))
    out = cfg.out_dir
    ck = np.load(out / "smc_checkpoint.npz")
    for k in ("u_particles", "y_state", "chem_refs", "loglik"):
        if k not in ck.files:
            raise KeyError(f"checkpoint lacks {k}")
    pipe = P.build_pipeline(cfg)
    d = np.load(out / "observations.npz")
    pipe.set_observations(d["depth"], d["sigma"])

    U = np.asarray(ck["u_particles"], np.float64)
    Y = np.asarray(ck["y_state"], np.float64)
    refs = np.asarray(ck["chem_refs"], np.float64)
    L = np.asarray(ck["loglik"], np.float64)
    N = U.shape[0]
    healthy = np.isfinite(L) & (L > -1e29)

    # nearest-neighbor pairs among healthy particles (unique, closest first)
    idx = np.flatnonzero(healthy)
    D = np.linalg.norm(U[idx][:, None, :] - U[idx][None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    pairs, used = [], set()
    for a, b in order:
        if a in used or b in used or a == b:
            continue
        pairs.append((int(idx[a]), int(idx[b]))); used.update((a, b))
        if len(pairs) >= args.pairs:
            break
    print(f"probing {len(pairs)} nearest-neighbor pairs of {int(healthy.sum())} "
          f"healthy particles (warm_count_max={pipe.fwd.chem.warm_count_max})")

    theta = np.asarray(jax.device_get(jax.vmap(pipe.theta_from_u)(jnp.asarray(U))))
    n_ct = pipe.n_chem_tp
    wcmax = int(pipe.fwd.chem.warm_count_max)

    @jax.jit
    def solve_dir(chem_theta, y_from, refs_from):
        _y, cd = pipe.fwd.chem_solve_warm_diag(chem_theta, y_from,
                                               refs_from[0], refs_from[1])
        return jnp.asarray(cd.accept_count, jnp.int32), cd.conv_normal

    asym = 0
    for i, j in pairs:
        ac_ij, ok_ij = solve_dir(jnp.asarray(theta[j, :n_ct]), jnp.asarray(Y[i]),
                                 jnp.asarray(refs[i]))
        ac_ji, ok_ji = solve_dir(jnp.asarray(theta[i, :n_ct]), jnp.asarray(Y[j]),
                                 jnp.asarray(refs[j]))
        ac_ij, ac_ji = int(ac_ij), int(ac_ji)
        # rejected = warm-capped OR stall-certified (both are MH rejection
        # classes the correction cannot see; asymmetry in EITHER is the risk)
        rej_ij = (ac_ij >= wcmax) or (not bool(ok_ij))
        rej_ji = (ac_ji >= wcmax) or (not bool(ok_ji))
        tag = ""
        if rej_ij != rej_ji:
            asym += 1
            tag = "  <-- ASYMMETRIC (detailed-balance risk)"
        print(f"pair ({i:3d},{j:3d}): i->j accept={ac_ij:5d} capped={ac_ij >= wcmax} "
              f"stalled={not bool(ok_ij)} | j->i accept={ac_ji:5d} "
              f"capped={ac_ji >= wcmax} stalled={not bool(ok_ji)}{tag}", flush=True)

    frac = asym / max(1, len(pairs))
    print(f"\n==== reversibility summary ====\nasymmetric pairs: {asym}/{len(pairs)} "
          f"({frac:.0%})")
    ok = asym == 0
    print(f"VERDICT: {'PASS -- no state-dependent cap/stall events at this cloud' if ok else 'FAIL -- the warm cap or stall gate binds asymmetrically; raise warm_count_max or finish the ladder with smc_chem_mode=cold'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
