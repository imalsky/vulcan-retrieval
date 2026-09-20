#!/usr/bin/env python3
"""Which ExoGibbs profile-solve method is cheapest for the cold equilibrium seed.

Every cold draw calls ``vulcan_jax.ini_abun.eq_seed`` -> ``exogibbs.api.gas.
solve_profile`` once per lane, so the method is a per-draw cost on the GPU. This
times the three methods on the W39b column at the production lane width, with
the seed's own solver controls (epsilon_crit 1e-12, max_iter 450).

The second-call wall is the number that matters; the first call carries the
compile. Method names describe EXOGIBBS's array order, not VULCAN's: on VULCAN's
grid index 0 is the BOTTOM (highest pressure), and ``scan_hot_from_bottom``
flips the arrays, so it actually starts at the top of this column. The bench
decides on numbers, not names.

    python tools/bench_seed_methods.py            # 144 lanes (the gpu preset's N)
    python tools/bench_seed_methods.py --lanes 4  # local smoke run

Runs anywhere JAX runs; it is a GPU deliverable (tools/gpu_kernel_test.pbs).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

# Import-order contract (see CLAUDE.md): vulcan_chem freezes the VULCAN_JAX_*
# env vars and jax x64 before anything else touches jax or exojax.
from vulcan_forward import vulcan_chem  # noqa: F401
from vulcan_forward import constants

import jax  # noqa: E402
import vulcan_jax  # noqa: E402
from vulcan_jax import ini_abun  # noqa: E402
from vulcan_jax.state import RunState, legacy_view  # noqa: E402
from exogibbs.api.gas import EquilibriumOptions, solve_profile  # noqa: E402

NZ = 150                       # VULCAN's default column depth (gpu preset: 62)
T_OFFSETS_K = (-385.0, 385.0)  # the limb-T span the case's Tirr prior (1100-2200 K,
#                                ~0.7x at the terminator) puts on the baseline column
METHODS = ("vmap_cold", "scan_hot_from_top", "scan_hot_from_bottom")


def w39b_column(nz: int):
    """Baseline (T, p_bar) of the W39b column on the retrieval's grid."""
    cfg = vulcan_jax.load_config("W39b")
    cfg.use_live_plot = cfg.use_live_flux = cfg.use_print_prog = False
    cfg.use_photo = False
    cfg.nz = nz
    cfg.P_t = constants.ART_PTOP_BAR * 1.0e6   # as vulcan_chem sets it
    _var, atm, _para = legacy_view(RunState.with_pre_loop_setup(cfg))
    return (np.asarray(atm.Tco, np.float64),
            np.asarray(atm.pco, np.float64) / 1.0e6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=144,
                    help="profiles solved in one vmap (default: the gpu preset's N)")
    args = ap.parse_args()

    T0, p_bar = w39b_column(NZ)
    T = T0[None, :] + np.linspace(*T_OFFSETS_K, args.lanes)[:, None]
    b = ini_abun._element_vector()
    setup = ini_abun._seed()[0]
    print(f"lanes={args.lanes} nz={NZ} T {T.min():.0f}-{T.max():.0f} K, "
          f"p {p_bar.min():.3g}-{p_bar.max():.3g} bar, device {jax.devices()[0]}",
          flush=True)

    def make_run(method):
        opts = EquilibriumOptions(epsilon_crit=1e-12, max_iter=450, method=method)

        @jax.jit
        @jax.vmap
        def run(T_lane):
            return solve_profile(setup, T_lane, p_bar, b, Pref=1.0,
                                 options=opts, return_diagnostics=True)
        return run

    for method in METHODS:
        run = make_run(method)
        t0 = time.perf_counter()
        res, diag = jax.block_until_ready(run(T))
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        res, diag = jax.block_until_ready(run(T))
        t_steady = time.perf_counter() - t0
        conv = np.asarray(diag["converged"])
        n_it = np.asarray(diag["n_iter"])
        print(f"{method:>21}: first {t_first:8.3f} s (compile+run), steady "
              f"{t_steady:8.3f} s, max_iter {int(n_it.max()):4d}, "
              f"converged {int(conv.sum())}/{conv.size} lane-layers, "
              f"finite {bool(np.all(np.isfinite(res.x)))}", flush=True)


if __name__ == "__main__":
    main()
