"""Reusable VULCAN-JAX -> ExoJax transmission-retrieval framework (nautilus nested sampling).

Planet-agnostic machinery only. A concrete run is one YAML file in a CASE DIRECTORY
(see ``runs/w39b_smc_retrieval/``): its ``config`` block sets config_schema.Config,
``sampler`` the nautilus settings and ``runtime`` the engine's environment
(run_setup.load_spec). Outputs go to ``output/<case>/<name>_nautilus``; on NAS
``tools/submit.py <spec>`` submits ``tools/run_nas.pbs``.

Entry points (run from anywhere):

    python -m retrieval_framework.run_nautilus         runs/<case>/<name>.yaml [--resume] [--dry-run]
    python -m retrieval_framework.nautilus_report      <out_dir> [--spectra N]
    python -m retrieval_framework.validate_env         <project_root> [--require-gpu]

Module map (the import chain is heavy-import-safe top to bottom):

    config_schema      Config dataclass + ParamSpec + specs_from_config (no jax)
    observations       observed-spectrum loading + exact linear binning/offset operators
    tp_profile         differentiable ExoJax Guillot T-P evaluator
    retrieval_forward  theta -> native transit depth (live VULCAN-JAX chemistry + ExoJax RT)
    pipeline           u-space posterior and the batched likelihood evaluators
    provenance         code/data identity and the target manifest and digest
    run_setup          the run specification (load_spec), logging, identity files
    run_nautilus       the sampler driver
    nautilus_report    certificate and figures of a finished run
    forward/           this repo's case constants and data paths (config.py only); the
                       engine lives in the vulcan-forward distribution -- import order
                       matters there: vulcan_forward.vulcan_chem before anything
                       exojax (guard-enforced)

This ``__init__`` stays import-light (no jax): run_setup and config_schema must be
importable without pulling the chemistry/RT stack.
"""
from __future__ import annotations

from retrieval_framework._version import __version__  # noqa: F401
