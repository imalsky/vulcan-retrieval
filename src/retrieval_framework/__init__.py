"""Reusable VULCAN-JAX -> ExoJax transmission-retrieval framework (SMC + fwd-jvp MALA).

Planet-agnostic machinery only. A concrete retrieval lives in a CASE DIRECTORY
(see ``runs/w39b_smc_retrieval/``) containing:

    case.py            PRESETS = {"smoke": fn, "gpu": fn, ...} -> config_schema.Config
    overrides/*.json   optional Config-override files
    data/<preset>/     run outputs (created by the driver)
    <run>.pbs          the HPC submit script for that run

Entry points (run from anywhere; <run_dir> defaults to the cwd):

    python -m retrieval_framework.run_smc              <run_dir> [--calibrate]
    python -m retrieval_framework.calibrate_count_max  <run_dir> [--n-draws N ...]
    python -m retrieval_framework.probe_memory         <run_dir>
    python -m retrieval_framework.smoke_retrieval      <run_dir>
    python -m retrieval_framework.plot_smc             <out_dir>

Module map (the import chain is heavy-import-safe top to bottom):

    config_schema      Config dataclass + ParamSpec + specs_from_config (no jax)
    observations       observed-spectrum loading + exact linear binning/offset operators
    tp_profile         differentiable ExoJax Guillot / power-law T-P evaluators
    retrieval_forward  theta -> native transit depth (live VULCAN-JAX chemistry + ExoJax RT)
    pipeline           u-space posterior, staged batched evaluators, SMC core, MALA kernel
    run_smc            case-directory driver (presets, overrides, outputs, PPC)
    plot_smc           post-run figures from the .npz bundles (numpy+matplotlib only)
    forward/           the shared forward-model engine (config, vulcan_chem, exojax_rt,
                       interp_map, sensitivity) -- import order is load-bearing there:
                       forward.vulcan_chem before anything exojax (guard-enforced)

This ``__init__`` stays import-light (no jax): run_smc and config_schema must be
importable without pulling the chemistry/RT stack.
"""
from __future__ import annotations

from retrieval_framework._version import __version__  # noqa: F401
