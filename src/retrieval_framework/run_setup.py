"""The run specification, ``runs/<case>/<name>.yaml``, and a run's identity files.

Sections: ``base`` (one file beside it whose sections it refines), ``runtime`` (the environment the engine is imported
under), ``sampler`` (nautilus) and ``config`` (fields of ``config_schema.Config``, flat). The run's name is the file
stem and its output directory ``<repo>/output/<case>/<name>_nautilus``. A variant restates the keys it changes: a key's
value replaces the base's whole, and a base has no base of its own. Loading writes no environment and imports no jax;
``export_runtime`` writes the runtime block before the engine is imported and refuses a value that disagrees.

    python -m retrieval_framework.run_setup runs/<case>/<name>.yaml   # the runtime as export lines
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import sys
import types
import typing
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from retrieval_framework import config_schema as C, constants

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from retrieval_framework.pipeline import Pipeline

__all__ = [
    "Runtime",
    "Sampler",
    "Spec",
    "check_inputs",
    "check_run_dir",
    "export_runtime",
    "load_spec",
    "main",
    "output_base",
    "runtime_env_updates",
    "set_observations",
    "setup_logging",
    "spec_digest",
    "write_run_identity",
]

log = logging.getLogger("retrieval")

SECTIONS = ("base", "runtime", "sampler", "config")

# Keys the run reads in one mode only: (keys, whether the Config's mode reads them, that mode in words). A stated key
# whose mode is off is refused, since it would change nothing; a key also in _MODE_REQUIRED must be stated when its mode
# is on. A key in two rows needs both modes.
_MODE_KEYS: tuple[tuple[tuple[str, ...], Callable[[C.Config], bool], str], ...] = (
    (("tp_Tint_K",), lambda c: not c.tp_infer_Tint, "tp_infer_Tint: false"),
    (("tp_deep_param",), lambda c: c.tp_infer_Tint, "tp_infer_Tint: true"),
    (
        ("prior_T1bar", "truth_T1bar"),  # tp_deep_p_bar: the report's deep level in every mode
        lambda c: c.tp_infer_Tint and c.tp_deep_param == "T1bar",
        "tp_infer_Tint: true and tp_deep_param: T1bar",
    ),
    (
        ("prior_Tint", "truth_Tint"),
        lambda c: c.tp_infer_Tint and c.tp_deep_param == "Tint",
        "tp_infer_Tint: true and tp_deep_param: Tint",
    ),
    (("tp_gamma_fixed",), lambda c: not c.tp_infer_gamma, "tp_infer_gamma: false"),
    (("prior_log10gamma", "truth_log10gamma"), lambda c: c.tp_infer_gamma, "tp_infer_gamma: true"),
    (("tp_nabla_ad", "prior_log10P_rcb", "truth_log10P_rcb"), lambda c: c.tp_adiabat, "tp_adiabat: true"),
    (
        ("prior_log10kappa_cloud", "truth_log10kappa_cloud", "prior_cloud_alpha", "truth_cloud_alpha"),
        lambda c: c.cloud_model == "powerlaw",
        "cloud_model: powerlaw",
    ),
    (("prior_log10P_deck", "truth_log10P_deck"), lambda c: c.cloud_model == "deck", "cloud_model: deck"),
    (("prior_s_o", "truth_s_o"), lambda c: c.infer_s_o, "infer_s_o: true"),
    (
        ("kzz_p_ref_bar", "prior_kzz_slope", "truth_kzz_slope"),
        lambda c: c.kzz_mode == "powerlaw",
        "kzz_mode: powerlaw",
    ),
    (("truth_lnM",), lambda c: c.infer_lnM, "infer_lnM: true"),
    (("prior_offset_ppm", "truth_offset_ppm"), lambda c: c.infer_offsets, "infer_offsets: true"),
    (
        ("prior_noise_inflation", "truth_noise_inflation", "noise_inflation_per_group"),
        lambda c: c.infer_noise_inflation,
        "infer_noise_inflation: true",
    ),
    (
        ("combo", "obs_wl_lo", "obs_wl_hi", "obs_max_bin_R"),
        lambda c: bool(c.obs_dir),
        "a data source (obs_dir and obs_products)",
    ),
    (
        tuple(f.name for f in fields(C.Config) if f.name.startswith("truth_")),
        lambda c: c.generate_synthetic_data,
        "generate_synthetic_data: true",
    ),
)
# Keys of _MODE_KEYS that have no default for a case whose mode reads them.
_MODE_REQUIRED = frozenset({"tp_Tint_K", "prior_log10P_rcb", "prior_log10P_deck", "combo", "obs_wl_lo", "obs_wl_hi"})


@dataclass(frozen=True)
class Runtime:
    """Import-frozen engine settings, one environment variable each; None means the variable must be unset.

    Attributes
    ----------
    solver : str
        VULCAN3_SOLVER: fast, ffi or reference.
    chem_rhs : str
        VULCAN3_CHEM_RHS: codegen or grouped.
    bt_buffers : int, optional
        VULCAN3_BT_BUFFERS: 1, 2 or None (per call).
    gpu_memory_fraction : float, optional
        XLA_PYTHON_CLIENT_MEM_FRACTION.
    xla_flags : str, optional
        XLA_FLAGS.
    """

    solver: str = "fast"
    chem_rhs: str = "codegen"
    bt_buffers: int | None = None
    gpu_memory_fraction: float | None = None
    xla_flags: str | None = None

    def env(self) -> dict[str, str | None]:
        """Return the block as environment values.

        Returns
        -------
        dict
            Variable name to value, None for unset.
        """
        return {
            "VULCAN3_SOLVER": self.solver,
            "VULCAN3_CHEM_RHS": self.chem_rhs,
            "VULCAN3_BT_BUFFERS": None if self.bt_buffers is None else str(self.bt_buffers),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": (
                None if self.gpu_memory_fraction is None else repr(float(self.gpu_memory_fraction))
            ),
            "XLA_FLAGS": self.xla_flags,
        }


@dataclass(frozen=True)
class Sampler:
    """The nautilus settings.

    Attributes
    ----------
    n_live : int
        Live points.
    n_eff : int
        Effective posterior samples at which nautilus stops.
    n_batch : int, optional
        Points per batch; None is ``constants.BATCH_PER_LANE`` times the lanes.
    warm_starts : bool
        Continue each solve from the nearest certified column.
    """

    n_live: int = 1000
    n_eff: int = 10_000
    n_batch: int | None = None
    warm_starts: bool = True


@dataclass(frozen=True)
class Spec:
    """One resolved run specification.

    Attributes
    ----------
    path : pathlib.Path
        The spec file.
    base : pathlib.Path, optional
        Its base file.
    name : str
        The run's name, the file stem.
    cfg : config_schema.Config
        The configuration.
    sampler : Sampler
        The nautilus settings.
    runtime : Runtime
        The engine's environment.
    """

    path: Path
    base: Path | None
    name: str
    cfg: C.Config
    sampler: Sampler
    runtime: Runtime


class _Loader(yaml.SafeLoader):
    """A YAML loader that refuses duplicate keys."""

    pass


def _no_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    """Construct a mapping, refusing a duplicate key."""
    seen = set()
    for k, _ in node.value:
        key = loader.construct_object(k)
        if key in seen:
            raise ValueError(f"duplicate key {key!r} at {node.start_mark}")
        seen.add(key)
    return loader.construct_mapping(node)


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


def _read(path: Path) -> dict[str, Any]:
    """Read one spec file and check its top-level keys."""
    doc = yaml.load(path.read_text(), Loader=_Loader)
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: the document must be a mapping")  # noqa: TRY004  # every refusal at load is a ValueError
    bad = set(doc) - set(SECTIONS)
    if bad:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(bad)}; allowed: {SECTIONS}")
    for s in SECTIONS[1:]:
        if s in doc and not isinstance(doc[s], dict):
            raise ValueError(f"{path}: `{s}:` must be a mapping")
    return doc


def _tuples(v: Any) -> Any:
    """Return a YAML value with its lists as tuples, the Config annotations."""
    if isinstance(v, list):
        return tuple(_tuples(x) for x in v)
    if isinstance(v, dict):
        return {k: _tuples(x) for k, x in v.items()}
    return v


def _check(where: str, key: str, value: Any, hint: Any) -> None:  # noqa: C901  # one branch per annotation form
    """Check ``value`` against the field annotation ``hint``; a str for a number names the YAML exponent trap."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):  # X | None
        if value is None and type(None) in args:
            return
        _check(where, key, value, next(a for a in args if a is not type(None)))
        return
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
            extra = " (YAML reads 1e11 as text: write 1.0e+11)" if isinstance(value, str) else ""
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


def _build(cls: Any, mapping: dict[str, Any], where: str) -> Any:
    """Build a frozen dataclass from a YAML mapping, refusing unknown keys and type-checking every value."""
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    bad = set(mapping) - set(names)
    if bad:
        raise ValueError(f"{where}: unknown key(s) {sorted(bad)}; the fields are {names}")
    for k, v in mapping.items():
        _check(where, k, v, hints[k])
    return cls(**mapping)


def _check_mode_keys(cfg: C.Config, stated: set[str], where: str) -> None:
    """Refuse a stated key its mode does not read, and an unstated one its mode needs (``_MODE_KEYS``)."""
    for keys, reads, mode in _MODE_KEYS:
        on = bool(reads(cfg))
        dead = sorted(set(keys) & stated) if not on else []
        if dead:
            raise ValueError(
                f"{where}: {dead} are read only with {mode}, which this spec does not set, so they would "
                "change nothing: drop them"
            )
        missing = sorted(set(keys) & _MODE_REQUIRED - stated) if on else []
        if missing:
            raise ValueError(
                f"{where}: {missing} unset: {mode} reads them, and a schema default would not be this "
                "case's: state them"
            )


def output_base(run_dir: Path) -> Path:
    """Return where a case's generated artifacts go.

    Parameters
    ----------
    run_dir : pathlib.Path
        The directory holding the spec.

    Returns
    -------
    pathlib.Path
        ``<repo>/output/<case>/`` for a case under ``<repo>/runs/``, else ``<run_dir>/output``.
    """
    run_dir = Path(run_dir).resolve()
    if run_dir.parent.name == "runs":
        return run_dir.parents[1] / "output" / run_dir.name
    return run_dir / "output"


def load_spec(path: str | Path) -> Spec:  # noqa: C901  # one refusal per way a spec can be malformed
    """Return the run specification at ``path``, resolved against its ``base`` and validated.

    Parameters
    ----------
    path : str or pathlib.Path
        The spec file.

    Returns
    -------
    Spec
        The resolved specification.

    Raises
    ------
    ValueError
        On a bad file, key or value.
    """
    path = Path(path).resolve()
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"{path}: a run specification is a .yaml file")
    doc = _read(path)
    base_path = None
    sections: dict[str, dict[str, Any]] = {s: {} for s in SECTIONS[1:]}
    if "base" in doc:
        base_path = (path.parent / str(doc["base"])).resolve()
        base = _read(base_path)
        if "base" in base:
            raise ValueError(
                f"{path}: its base {base_path.name} has a base of its own; a spec resolves from at most two files"
            )
        for s, values in sections.items():
            values.update(base.get(s) or {})
    for s, values in sections.items():
        values.update(doc.get(s) or {})  # the variant's value wins, whole

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
    unset = sorted(C.CASE_KEYS - set(cfgd)) or sorted(k for k in C.CASE_KEYS if cfgd[k] is None)
    if unset:
        raise ValueError(
            f"{where}: {unset} unset. Each belongs to the planet and its data, so the spec (or its base) states it: "
            "the schema's default is not this planet's."
        )
    hints = typing.get_type_hints(C.Config)
    for k, v in cfgd.items():
        _check(where, k, v, hints[k])
    if cfgd.get("obs_dir") is not None:
        od = Path(cfgd["obs_dir"])
        if not od.is_absolute():
            from retrieval_framework.forward import config as fwd_config  # the repo root (no jax)

            od = fwd_config.REPO_DIR / od
        cfgd["obs_dir"] = od
    name = path.stem
    cfg = C.Config(**cfgd, out_dir=output_base(path.parent) / f"{name}_nautilus")
    C.validate_config(cfg)
    _check_mode_keys(cfg, set(cfgd), where)
    return Spec(path=path, base=base_path, name=name, cfg=cfg, sampler=sampler, runtime=runtime)


def spec_digest(spec: Spec) -> str:
    """Return the sha256 over the bytes of the spec file and its base.

    ``submit.py`` records it and the job refuses a spec edited while the job was queued.

    Parameters
    ----------
    spec : Spec
        The specification.

    Returns
    -------
    str
        The digest.
    """
    h = hashlib.sha256()
    for f in (spec.path, spec.base):
        if f is not None:
            h.update(Path(f).read_bytes())
    return h.hexdigest()


def check_run_dir(out: Path, resume: bool) -> None:
    """Refuse a resume with no checkpoint, and a fresh run into a directory that holds a previous run.

    An empty ``anchors/`` is a run stopped before its first chunk, and is not refused.

    Parameters
    ----------
    out : pathlib.Path
        The run's output directory.
    resume : bool
        Whether this is a resume.

    Raises
    ------
    FileNotFoundError
        On a resume with no checkpoint.
    FileExistsError
        On a fresh run into a previous run's directory.
    """
    ckpt = Path(out) / "nautilus_checkpoint.hdf5"
    if resume and not ckpt.exists():
        raise FileNotFoundError(f"--resume but no checkpoint at {ckpt}; drop --resume to start a fresh run.")
    if not resume and (ckpt.exists() or any((Path(out) / "anchors").glob("anchors_*.npz"))):
        raise FileExistsError(
            f"{out} holds a previous nautilus run (checkpoint or anchors/): "
            "pass --resume to continue it, or copy the spec to a new file "
            "name for a fresh run."
        )


def check_inputs(cfg: C.Config) -> None:
    """Refuse a config whose input files would only fail minutes into a job.

    Checks the network, the case's vulcan3 config and its atm and stellar-flux files, and every product of ``combo``. A
    named data source also needs ``constants.MIN_REAL_BINS`` bins in the fit window and one in every group, in a
    synthetic run too. Imports no jax.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Raises
    ------
    FileNotFoundError
        If an input file is missing.
    ValueError
        If a named data source has too few bins.
    """
    from retrieval_framework import observations as OBS, provenance

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
    if len(obs["wl"]) < constants.MIN_REAL_BINS or empty:
        raise ValueError(
            f"{cfg.obs_dir}: {len(obs['wl'])} bins of {list(cfg.combo)} in the fit window "
            f"[{cfg.obs_wl_lo}, {cfg.obs_wl_hi}] um (groups with none: {empty}); a named data "
            f"source needs {constants.MIN_REAL_BINS} and one in every group. A synthetic run on the synthetic grid "
            "names no data source"
        )


def runtime_env_updates(rt: Runtime, environ: Mapping[str, str]) -> dict[str, str | None]:
    """Return the runtime block as environment values, checked against ``environ``.

    Parameters
    ----------
    rt : Runtime
        The block.
    environ : mapping
        The environment.

    Returns
    -------
    dict
        Variable name to value, None for unset.

    Raises
    ------
    RuntimeError
        If a variable is set to a different value, or set when the block leaves it unset.
    """
    for var, want in rt.env().items():
        have = environ.get(var)
        if want is None and have:
            raise RuntimeError(
                f"{var}={have!r} is set in the environment but the spec's "
                "runtime block leaves it unset: unset it, or set it in the spec"
            )
        if want is not None and have is not None and have != want:
            raise RuntimeError(
                f"{var}={have!r} in the environment disagrees with the spec's "
                f"runtime block ({want!r}): unset it, the spec decides"
            )
    return rt.env()


def export_runtime(rt: Runtime) -> None:
    """Write the runtime block into the environment, before the engine is imported.

    Parameters
    ----------
    rt : Runtime
        The block.

    Raises
    ------
    RuntimeError
        If vulcan3 or jax is already imported.
    """
    for mod in ("vulcan3", "jax"):
        if mod in sys.modules:
            raise RuntimeError(f"export_runtime: {mod} is already imported; the runtime block must be exported first")
    for var, want in runtime_env_updates(rt, os.environ).items():
        if want is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = want


def setup_logging(log_file: Path | None = None, mode: str = "a") -> None:
    """Log at INFO to stderr, and to ``log_file`` when given.

    Parameters
    ----------
    log_file : pathlib.Path, optional
        The log file.
    mode : str, optional
        Its open mode.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode=mode))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True
    )


def write_run_identity(cfg: C.Config, pipe: Pipeline, spec: Spec, obs_save: dict) -> None:
    """Write config.json, observations.npz and the target manifest; call only after the resume-identity check.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.
    pipe : pipeline.Pipeline
        The pipeline.
    spec : Spec
        The specification.
    obs_save : dict
        The arrays of observations.npz.
    """
    from retrieval_framework import pipeline as P, provenance

    _write_config_json(cfg, pipe, spec)
    out: Path = cfg.out_dir  # type: ignore[assignment]  # load_spec sets it
    P.save_npz(out / "observations.npz", **obs_save)
    # The canonical manifest, not only its hash: a refused resume names the
    # differing class only if the two manifests can be diffed. The digest that
    # binds the checkpoint is the sha256 of this manifest's canonical form
    # (provenance._canonical: compact, sorted keys), not of the indented file.
    (out / provenance.MANIFEST_FILE).write_text(
        json.dumps(pipe.target_manifest, indent=2, sort_keys=True, default=str) + "\n"
    )


def _write_config_json(cfg: C.Config, pipe: Pipeline, spec: Spec) -> None:
    """Write config.json: the Config, the spec, the parameter table and the hardware."""
    d = asdict(cfg)
    d.update(
        {
            "spec": spec.name,
            "spec_file": str(spec.path),
            "sampler": asdict(spec.sampler),
            "runtime": asdict(spec.runtime),
            "inferred_param_names": pipe.names,
            "inferred_param_labels": pipe.labels,
            "inferred_param_kinds": pipe.kinds,
            "inferred_param_prior_types": pipe.prior_types,
            "inferred_param_prior_lo": pipe.param_prior_lo.tolist(),
            "inferred_param_prior_hi": pipe.param_prior_hi.tolist(),
            "inferred_param_truth": _output_truth(cfg, pipe).tolist(),
            "instrument_groups": pipe.groups,
            "n_obs_bins": int(pipe.n_bin),
            "real_bins": bool(pipe.real_bins),
            "n_chem_tp": int(pipe.n_chem_tp),
        }
    )
    d["hardware"] = C.hardware_profile()
    out: Path = cfg.out_dir  # type: ignore[assignment]  # load_spec sets it
    (out / "config.json").write_text(json.dumps(d, indent=2, default=str))


def set_observations(cfg: C.Config, pipe: Pipeline, P: ModuleType) -> dict[str, Any]:
    """Load real or synthetic observations into ``pipe``, once, and return the bundle to save; writes nothing.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.
    pipe : pipeline.Pipeline
        The pipeline.
    P : module
        ``retrieval_framework.pipeline``, which the caller has imported after the runtime export.

    Returns
    -------
    dict
        The arrays of observations.npz.
    """
    if cfg.generate_synthetic_data:
        log.info("Generating synthetic observations (injection at truth_*)...")
        o = P.generate_observations(pipe, seed=int(cfg.seed))
        obs_save = {
            "wl": pipe.obs["wl"],
            "wl_lo": pipe.obs["wl_lo"],
            "wl_hi": pipe.obs["wl_hi"],
            "depth": o["depth"],
            "sigma": o["sigma"],
            "flux_true": o["flux_true"],
            "group": np.asarray(pipe.obs["group"], dtype="<U16"),
            "synthetic": np.asarray(1, np.int32),
            "inferred_param_names": np.asarray(pipe.names, dtype="<U64"),
            "inferred_param_truth": np.asarray(pipe.param_truth),
        }
    else:
        o = P.load_real_into_pipe(pipe)
        obs_save = {
            "wl": pipe.obs["wl"],
            "wl_lo": pipe.obs["wl_lo"],
            "wl_hi": pipe.obs["wl_hi"],
            "depth": o["depth"],
            "sigma": o["sigma"],
            "group": np.asarray(pipe.obs["group"], dtype="<U16"),
            "synthetic": np.asarray(0, np.int32),
        }
        log.info("Using REAL observed spectrum: %s bins (%s)", pipe.n_bin, ", ".join(pipe.groups))
    return obs_save


def _output_truth(cfg: C.Config, pipe: Pipeline) -> np.ndarray:
    """Return the truth vector [n_dim] for outputs: the injected values, or NaN for real data."""
    if cfg.generate_synthetic_data:
        return np.asarray(pipe.param_truth, dtype=np.float64)
    return np.full(pipe.n_dim, np.nan, dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    """Print the spec's runtime block as shell ``export`` and ``unset`` lines.

    So a job script can put the kernel build and the preflight under the run's environment:
    ``eval "$(python -m retrieval_framework.run_setup <spec>)"``.

    Parameters
    ----------
    argv : list of str, optional
        The arguments; default ``sys.argv``.
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("spec", help="runs/<case>/<name>.yaml")
    ap.add_argument("--digest", default=None, help="refuse unless the spec still has this spec_digest (submit.py's)")
    args = ap.parse_args(argv)
    spec = load_spec(args.spec)
    if args.digest is not None and spec_digest(spec) != args.digest:
        sys.exit(
            f"{spec.path}: the spec or its base changed after it was submitted (digest "
            f"{spec_digest(spec)}, submitted {args.digest}): resubmit it"
        )
    for var, val in spec.runtime.env().items():
        print(f"unset {var}" if val is None else f"export {var}={shlex.quote(val)}")


if __name__ == "__main__":
    main()
