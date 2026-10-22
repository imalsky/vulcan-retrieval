"""run_setup.py -- case-directory setup shared by the retrieval entry points.

The planet lives in the case directory's ``case.py`` (PRESETS dict of Config
factories); this module resolves the case + preset + env overrides into a
Config, sets up logging, loads real (or generates synthetic) observations into
a built pipeline, and writes the run's identity files (config.json,
observations.npz, target_manifest.json).

Presets / overrides (env vars)
------------------------------
- ``SMC_RETRIEVAL_PRESET``        : key into the case's PRESETS dict (default: the
                                    case's DEFAULT_PRESET, else "smoke").
- ``SMC_RETRIEVAL_OVERRIDES``     : JSON object of Config field overrides,
                                    e.g. '{"nz": 50, "cold_lanes": 32}'.
- ``SMC_RETRIEVAL_OVERRIDES_FILE``: JSON file of Config field overrides (relative
                                    paths resolve against the run dir).
- ``SMC_RETRIEVAL_OUT_DIR``       : output directory (default <run_dir>/data/<preset>).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from retrieval_framework import config_schema as C  # light import (no jax)

log = logging.getLogger("retrieval")


def load_case(run_dir: Path):
    """Import <run_dir>/case.py as a module. It must define PRESETS (dict preset-name
    -> zero-arg Config factory) and may define DEFAULT_PRESET."""
    run_dir = Path(run_dir).resolve()
    case_py = run_dir / "case.py"
    if not case_py.is_file():
        raise FileNotFoundError(
            f"{case_py} not found -- a retrieval run dir must contain case.py "
            "(see runs/w39b_smc_retrieval/case.py for the template)")
    spec = importlib.util.spec_from_file_location(f"retrieval_case_{run_dir.name}", case_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PRESETS"):
        raise AttributeError(f"{case_py} defines no PRESETS dict")
    return mod


def resolve_preset(case_mod) -> str:
    preset = os.environ.get("SMC_RETRIEVAL_PRESET", "").strip().lower()
    if not preset:
        preset = str(getattr(case_mod, "DEFAULT_PRESET", "smoke")).lower()
    if preset not in case_mod.PRESETS:
        raise ValueError(f"Unknown SMC_RETRIEVAL_PRESET={preset!r} "
                         f"(this case has: {sorted(case_mod.PRESETS)})")
    return preset


def make_config(run_dir: Path) -> Tuple[C.Config, str]:
    """Resolve (case, preset, env overrides) -> (Config, preset_name). Relative
    override-file paths and out_dir resolve against the RUN DIR, not the cwd."""
    run_dir = Path(run_dir).resolve()
    case_mod = load_case(run_dir)
    preset = resolve_preset(case_mod)
    cfg = case_mod.PRESETS[preset]()

    overrides: Dict[str, Any] = {}
    ov_file = os.environ.get("SMC_RETRIEVAL_OVERRIDES_FILE", "").strip()
    if ov_file:
        ov_path = Path(ov_file)
        if not ov_path.is_absolute():
            ov_path = run_dir / ov_path
        overrides.update(json.loads(ov_path.read_text()))
    ov = os.environ.get("SMC_RETRIEVAL_OVERRIDES", "").strip()
    if ov:
        overrides.update(json.loads(ov))
    overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}

    out_env = os.environ.get("SMC_RETRIEVAL_OUT_DIR", "").strip()
    if out_env:
        overrides["out_dir"] = out_env
    if "out_dir" in overrides:
        od = Path(overrides["out_dir"])
        overrides["out_dir"] = od if od.is_absolute() else (run_dir / od).resolve()
    elif cfg.out_dir is None:
        cfg = replace(cfg, out_dir=run_dir / "data" / preset)

    # tuples in the dataclass may arrive as JSON lists
    for k, v in list(overrides.items()):
        if isinstance(v, list):
            overrides[k] = tuple(v)
    if overrides:
        cfg = replace(cfg, **overrides)
    # VULCAN3_SOLVER is import-frozen: choose it before anything imports
    # vulcan3 (every entry point resolves its config here first). Printed, not
    # logged: the callers set up logging after this.
    print(f"[retrieval] solver: {C.choose_solver()}", file=sys.stderr, flush=True)
    return cfg, preset


def setup_logging(log_file: Path | None = None, mode: str = "a") -> None:
    """INFO-level logging to stderr, and to ``log_file`` when given."""
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode=mode))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=handlers, force=True)


def log_hardware() -> None:
    """Log the hardware profile."""
    log.info(f"hardware: {C.hardware_profile()}")


def write_run_identity(cfg: C.Config, pipe, preset: str, obs_save: dict) -> None:
    """Write config.json, observations.npz and the canonical target manifest
    into ``cfg.out_dir``. Call only after the resume-identity check."""
    from retrieval_framework import pipeline as P
    from retrieval_framework import provenance
    write_config_json(cfg, pipe, preset)
    P.save_npz(cfg.out_dir / "observations.npz", **obs_save)
    # The canonical manifest, not only its hash: a refused resume names the
    # differing class only if the two manifests can be diffed. The digest that
    # binds the checkpoint is the sha256 of this manifest's canonical form
    # (provenance._canonical: compact, sorted keys), not of the indented file.
    (cfg.out_dir / provenance.MANIFEST_FILE).write_text(json.dumps(
        provenance.target_manifest(cfg, pipe), indent=2, sort_keys=True,
        default=str) + "\n")


def write_config_json(cfg: C.Config, pipe, preset: str) -> None:
    d = asdict(cfg)
    d.update(dict(
        preset=preset,
        inferred_param_names=pipe.names,
        inferred_param_labels=pipe.labels,
        inferred_param_kinds=pipe.kinds,
        inferred_param_prior_types=pipe.prior_types,
        inferred_param_prior_lo=pipe.param_prior_lo.tolist(),
        inferred_param_prior_hi=pipe.param_prior_hi.tolist(),
        inferred_param_truth=output_truth(cfg, pipe).tolist(),
        instrument_groups=pipe.groups,
        n_obs_bins=int(pipe.n_bin),
        real_bins=bool(pipe.real_bins),
        n_chem_tp=int(pipe.n_chem_tp),
    ))
    d["hardware"] = C.hardware_profile()
    (cfg.out_dir / "config.json").write_text(json.dumps(d, indent=2, default=str))


def set_observations(cfg: C.Config, pipe, P):
    """Load real (or generate synthetic) observations into ``pipe`` exactly
    once, before any jitted likelihood call. Returns the bundle to save as
    observations.npz. Writes nothing: the caller saves it after its
    resume-identity check."""
    if cfg.generate_synthetic_data:
        log.info("Generating synthetic observations (injection at truth_*)...")
        o = P.generate_observations(pipe, seed=int(cfg.seed))
        obs_save = dict(
            wl=pipe.obs["wl"], wl_lo=pipe.obs["wl_lo"], wl_hi=pipe.obs["wl_hi"],
            depth=o["depth"], sigma=o["sigma"], flux_true=o["flux_true"],
            group=np.asarray(pipe.obs["group"], dtype="<U16"),
            synthetic=np.asarray(1, np.int32),
            inferred_param_names=np.asarray(pipe.names, dtype="<U64"),
            inferred_param_truth=np.asarray(pipe.param_truth))
    else:
        o = P.load_real_into_pipe(pipe)
        obs_save = dict(
            wl=pipe.obs["wl"], wl_lo=pipe.obs["wl_lo"], wl_hi=pipe.obs["wl_hi"],
            depth=o["depth"], sigma=o["sigma"],
            group=np.asarray(pipe.obs["group"], dtype="<U16"),
            synthetic=np.asarray(0, np.int32))
        log.info(f"Using REAL observed spectrum: {pipe.n_bin} bins "
                 f"({', '.join(pipe.groups)})")
    return obs_save


def output_truth(cfg: C.Config, pipe) -> np.ndarray:
    """Truth vector for outputs: real values for synthetic injections, NaN for real
    data (where the cfg truth_* fields are just initialization placeholders)."""
    if cfg.generate_synthetic_data:
        return np.asarray(pipe.param_truth, dtype=np.float64)
    return np.full(pipe.n_dim, np.nan, dtype=np.float64)
