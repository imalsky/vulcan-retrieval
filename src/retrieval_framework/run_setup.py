"""run_setup.py -- the run specification and a run's identity files.

One YAML file describes a nautilus run, ``runs/<case>/<name>.yaml``:

    base:     optional, one file beside it whose sections it refines
    runtime:  the environment the engine is imported under (Runtime below)
    sampler:  the nautilus settings (Sampler below)
    config:   fields of config_schema.Config, flat; a key not listed keeps the default

The run's name is the file stem and its output directory is
``<repo>/output/<case>/<name>_nautilus`` (never a key). The PBS job is not part of
the run: tools/submit.py sets it. A variant names one ``base``
and restates the keys it changes: within each section a key's value replaces the
base's value whole, lists and dicts included; a base has no base of its own.
Loading is pure (no environment write, no jax import); ``export_runtime`` writes
the runtime block into the environment before the engine is imported and refuses
an inherited value that disagrees with the file.

    python -m retrieval_framework.run_nautilus runs/w39b_smc_retrieval/wasp39_test1.yaml [--resume] [--dry-run]
    python -m retrieval_framework.run_setup runs/w39b_smc_retrieval/wasp39_test1.yaml   # the runtime as `export` lines
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import sys
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from retrieval_framework import config_schema as C  # light import (no jax)

log = logging.getLogger("retrieval")

SECTIONS = ("base", "runtime", "sampler", "config")


@dataclass(frozen=True)
class Runtime:
    """Import-frozen engine settings; one environment variable each. ``None``
    means the variable must be unset."""
    solver: str = "fast"                       # VULCAN3_SOLVER: fast | ffi | reference
    chem_rhs: str = "codegen"                  # VULCAN3_CHEM_RHS: codegen | grouped
    bt_buffers: Optional[int] = None           # VULCAN3_BT_BUFFERS: 1 | 2 | None (per call)
    gpu_memory_fraction: Optional[float] = None   # XLA_PYTHON_CLIENT_MEM_FRACTION
    xla_flags: Optional[str] = None            # XLA_FLAGS

    def env(self) -> Dict[str, Optional[str]]:
        return {"VULCAN3_SOLVER": self.solver,
                "VULCAN3_CHEM_RHS": self.chem_rhs,
                "VULCAN3_BT_BUFFERS": None if self.bt_buffers is None else str(self.bt_buffers),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": (None if self.gpu_memory_fraction is None
                                                   else repr(float(self.gpu_memory_fraction))),
                "XLA_FLAGS": self.xla_flags}


@dataclass(frozen=True)
class Sampler:
    n_live: int = 1000
    n_eff: int = 10_000          # effective posterior samples at which nautilus stops
    n_batch: Optional[int] = None    # None: run_nautilus.BATCH_PER_LANE x the lanes
    warm_starts: bool = True


@dataclass(frozen=True)
class Spec:
    path: Path
    base: Optional[Path]
    name: str
    cfg: C.Config
    sampler: Sampler
    runtime: Runtime


class _Loader(yaml.SafeLoader):
    pass


def _no_duplicates(loader, node):
    seen = set()
    for k, _ in node.value:
        key = loader.construct_object(k)
        if key in seen:
            raise ValueError(f"duplicate key {key!r} at {node.start_mark}")
        seen.add(key)
    return loader.construct_mapping(node)


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def _read(path: Path) -> Dict[str, Any]:
    doc = yaml.load(path.read_text(), Loader=_Loader)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: the document must be a mapping")
    bad = set(doc) - set(SECTIONS)
    if bad:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(bad)}; allowed: {SECTIONS}")
    for s in SECTIONS[1:]:
        if s in doc and not isinstance(doc[s], dict):
            raise ValueError(f"{path}: `{s}:` must be a mapping")
    return doc


def _tuples(v):
    if isinstance(v, list):
        return tuple(_tuples(x) for x in v)
    if isinstance(v, dict):
        return {k: _tuples(x) for k, x in v.items()}
    return v


_ORIGIN = {tuple: (tuple,), dict: (dict,)}


def _check(where: str, key: str, value, hint) -> None:
    """``value`` against the field annotation ``hint`` (int, float, bool, str, Path,
    Tuple[...], Dict[...], Optional[...]). A str where a number is expected names
    the YAML exponent trap."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is typing.Union:                       # Optional[X]
        if value is None and type(None) in args:
            return
        return _check(where, key, value, [a for a in args if a is not type(None)][0])
    if hint is Any or hint is object:
        return
    if origin in (tuple, dict):
        if not isinstance(value, origin):
            raise TypeError(f"{where}: `{key}` must be a {origin.__name__}, got {value!r}")
        if origin is tuple and args:
            elem = args[0] if (len(args) == 2 and args[1] is Ellipsis) else None
            for i, x in enumerate(value):
                _check(where, f"{key}[{i}]", x, elem if elem is not None else args[i] if i < len(args) else args[-1])
        return
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where}: `{key}` must be true or false, got {value!r}")
        return
    if hint in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            extra = (" (YAML reads 1e11 as text: write 1.0e+11)" if isinstance(value, str)
                     else "")
            raise TypeError(f"{where}: `{key}` must be a number, got {value!r}{extra}")
        if hint is int and not isinstance(value, int):
            raise TypeError(f"{where}: `{key}` must be an integer, got {value!r}")
        return
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: `{key}` must be text, got {value!r}")
        return
    if hint is Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{where}: `{key}` must be a path, got {value!r}")
        return
    if is_dataclass(hint):
        return


def _build(cls, mapping: Dict[str, Any], where: str):
    """A frozen dataclass from a YAML mapping: unknown keys refused, every value
    type-checked against the field annotation."""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    bad = set(mapping) - set(names)
    if bad:
        raise ValueError(f"{where}: unknown key(s) {sorted(bad)}; the fields are {names}")
    for k, v in mapping.items():
        _check(where, k, v, hints[k])
    return cls(**mapping)


def output_base(run_dir: Path) -> Path:
    """Where a case's generated artifacts go: ``<repo>/output/<case>/`` for a case
    under ``<repo>/runs/``, else ``<run_dir>/output``. Inputs (the spec files, the
    job scripts) stay in the run dir; outputs never do."""
    run_dir = Path(run_dir).resolve()
    if run_dir.parent.name == "runs":
        return run_dir.parents[1] / "output" / run_dir.name
    return run_dir / "output"


def load_spec(path: str | Path) -> Spec:
    """The run specification at ``path`` (module docstring), resolved against its
    ``base`` and validated. Pure: writes no environment variable, imports no jax."""
    path = Path(path).resolve()
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"{path}: a run specification is a .yaml file")
    doc = _read(path)
    base_path = None
    sections: Dict[str, Dict[str, Any]] = {s: {} for s in SECTIONS[1:]}
    if "base" in doc:
        base_path = (path.parent / str(doc["base"])).resolve()
        base = _read(base_path)
        if "base" in base:
            raise ValueError(f"{path}: its base {base_path.name} has a base of its own; "
                             "a spec resolves from at most two files")
        for s in sections:
            sections[s].update(base.get(s) or {})
    for s in sections:
        sections[s].update(doc.get(s) or {})       # the variant's value wins, whole

    runtime = _build(Runtime, sections["runtime"], f"{path.name} runtime")
    sampler = _build(Sampler, sections["sampler"], f"{path.name} sampler")
    if runtime.solver not in ("fast", "ffi", "reference"):
        raise ValueError(f"{path.name} runtime: solver {runtime.solver!r} is not fast, ffi or reference")
    if runtime.chem_rhs not in ("codegen", "grouped"):
        raise ValueError(f"{path.name} runtime: chem_rhs {runtime.chem_rhs!r} is not codegen or grouped")
    if runtime.bt_buffers not in (None, 1, 2):
        raise ValueError(f"{path.name} runtime: bt_buffers {runtime.bt_buffers!r} is not 1, 2 or null")
    if sampler.n_live < 2 or sampler.n_eff < 1 or (sampler.n_batch is not None and sampler.n_batch < 1):
        raise ValueError(f"{path.name} sampler: n_live, n_eff and n_batch must be positive")

    # lists become tuples (the Config annotations), except inside cfg_overrides, which
    # vulcan3 reads as its own YAML would give it
    cfgd = {k: (v if k == "cfg_overrides" else _tuples(v)) for k, v in sections["config"].items()}
    where = f"{path.name} config"
    if "out_dir" in cfgd:
        raise ValueError(f"{where}: `out_dir` is derived from the file name, never a key")
    names = {f.name for f in fields(C.Config)}
    bad = set(cfgd) - names
    if bad:
        raise ValueError(f"{where}: unknown key(s) {sorted(bad)}: not fields of config_schema.Config")
    hints = typing.get_type_hints(C.Config)
    for k, v in cfgd.items():
        _check(where, k, v, hints[k])
    if cfgd.get("obs_dir") is not None:
        od = Path(cfgd["obs_dir"])
        if not od.is_absolute():
            from retrieval_framework.forward import config as fwd_config   # the repo root (no jax)
            od = fwd_config.REPO_DIR / od
        cfgd["obs_dir"] = od
    name = path.stem
    cfg = C.Config(**cfgd, out_dir=output_base(path.parent) / f"{name}_nautilus")
    C.validate_config(cfg)
    return Spec(path=path, base=base_path, name=name, cfg=cfg, sampler=sampler,
                runtime=runtime)


def spec_digest(spec: Spec) -> str:
    """sha256 over the bytes of the spec file and of its base: submit.py records it and
    the job refuses a spec that no longer has it (edited while the job was queued)."""
    h = hashlib.sha256()
    for f in (spec.path, spec.base):
        if f is not None:
            h.update(Path(f).read_bytes())
    return h.hexdigest()


def check_run_dir(out: Path, resume: bool) -> None:
    """Refuse a resume with no checkpoint, and a fresh run into a directory that holds
    a previous run: a checkpoint or anchor files (an empty ``anchors/`` is a run that
    stopped before its first chunk, e.g. at the startup check, and is not refused).
    submit.py and run_nautilus call this one check."""
    ckpt = Path(out) / "nautilus_checkpoint.hdf5"
    if resume and not ckpt.exists():
        raise FileNotFoundError(f"--resume but no checkpoint at {ckpt}; drop --resume "
                                "to start a fresh run.")
    if not resume and (ckpt.exists() or any((Path(out) / "anchors").glob("anchors_*.npz"))):
        raise FileExistsError(f"{out} holds a previous nautilus run (checkpoint or anchors/): "
                              "pass --resume to continue it, or copy the spec to a new file "
                              "name for a fresh run.")


def check_inputs(cfg: C.Config) -> None:
    """Refuse, before anything is queued or built, a config whose input files would
    only fail minutes into a job: the reaction network, the case's vulcan3 config and
    the atm and stellar-flux files it names (resolved from the working directory as
    the run resolves them, ``provenance.case_config``), and with an ``obs_dir`` every
    product of ``combo`` read as the run reads it; a real-data run also needs
    observations.MIN_REAL_BINS bins in the fit window and one in every group. No jax."""
    from retrieval_framework import observations as OBS
    from retrieval_framework import provenance
    net = provenance.case_data_file(cfg.network)
    if not net.is_file():
        raise FileNotFoundError(f"network {cfg.network!r} not found (resolved to {net})")
    path, case = provenance.case_config(cfg.vulcan_cfg_name)
    for key in ("atm_file", "sflux_file"):
        f = provenance.case_data_file(cfg.cfg_overrides.get(key, case[key]))
        if not f.is_file():
            raise FileNotFoundError(f"{path}: its {key} {f} does not exist")
    if not cfg.obs_dir:
        return
    obs = OBS.load_real_observations(cfg)
    empty = [g for g in cfg.combo if g not in obs["groups"]]
    if not cfg.generate_synthetic_data and (len(obs["wl"]) < OBS.MIN_REAL_BINS or empty):
        raise ValueError(
            f"{cfg.obs_dir}: {len(obs['wl'])} bins of {list(cfg.combo)} in the fit window "
            f"[{cfg.obs_wl_lo}, {cfg.obs_wl_hi}] um (groups with none: {empty}); a real-data "
            f"run needs {OBS.MIN_REAL_BINS} and one in every group")


def runtime_env_updates(rt: Runtime, environ) -> Dict[str, Optional[str]]:
    """The runtime block as environment values, checked against ``environ``: a
    variable already set to a different value is refused (the file is the single
    source), and a value of None requires the variable to be unset or empty."""
    for var, want in rt.env().items():
        have = environ.get(var)
        if want is None and have:
            raise RuntimeError(f"{var}={have!r} is set in the environment but the spec's "
                               "runtime block leaves it unset: unset it, or set it in the spec")
        if want is not None and have is not None and have != want:
            raise RuntimeError(f"{var}={have!r} in the environment disagrees with the spec's "
                               f"runtime block ({want!r}): unset it, the spec decides")
    return rt.env()


def export_runtime(rt: Runtime) -> None:
    """Write the runtime block into the environment, before the engine is imported."""
    for mod in ("vulcan3", "jax"):
        if mod in sys.modules:
            raise RuntimeError(f"export_runtime: {mod} is already imported; the runtime block "
                               "must be exported first")
    for var, want in runtime_env_updates(rt, os.environ).items():
        if want is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = want


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


def write_run_identity(cfg: C.Config, pipe, spec: Spec, obs_save: dict) -> None:
    """Write config.json, observations.npz and the canonical target manifest
    into ``cfg.out_dir``. Call only after the resume-identity check."""
    from retrieval_framework import pipeline as P
    from retrieval_framework import provenance
    write_config_json(cfg, pipe, spec)
    P.save_npz(cfg.out_dir / "observations.npz", **obs_save)
    # The canonical manifest, not only its hash: a refused resume names the
    # differing class only if the two manifests can be diffed. The digest that
    # binds the checkpoint is the sha256 of this manifest's canonical form
    # (provenance._canonical: compact, sorted keys), not of the indented file.
    (cfg.out_dir / provenance.MANIFEST_FILE).write_text(json.dumps(
        provenance.target_manifest(cfg, pipe), indent=2, sort_keys=True,
        default=str) + "\n")


def write_config_json(cfg: C.Config, pipe, spec: Spec) -> None:
    d = asdict(cfg)
    d.update(dict(
        spec=spec.name,
        spec_file=str(spec.path),
        sampler=asdict(spec.sampler),
        runtime=asdict(spec.runtime),
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


def main(argv=None) -> None:
    """Print the spec's runtime block as shell ``export`` / ``unset`` lines, so a job
    script can put the kernel build and the preflight under the same environment
    as the run: ``eval "$(python -m retrieval_framework.run_setup <spec>)"``."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("spec", help="runs/<case>/<name>.yaml")
    ap.add_argument("--digest", default=None,
                    help="refuse unless the spec still has this spec_digest (submit.py's)")
    args = ap.parse_args(argv)
    spec = load_spec(args.spec)
    if args.digest is not None and spec_digest(spec) != args.digest:
        sys.exit(f"{spec.path}: the spec or its base changed after it was submitted (digest "
                 f"{spec_digest(spec)}, submitted {args.digest}): resubmit it")
    for var, val in spec.runtime.env().items():
        print(f"unset {var}" if val is None else f"export {var}={shlex.quote(val)}")


if __name__ == "__main__":
    main()
