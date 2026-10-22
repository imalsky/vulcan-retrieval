"""The run specification (run_setup.load_spec): every shipped spec loads to the
parameter count it is documented with and to its derived output directory; a variant
replaces its base's values one level deep, whole; a malformed file is refused by name;
the runtime block is exported only when the environment does not contradict it."""
from pathlib import Path

import pytest
import yaml

from conftest import RUN_DIR
from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
from retrieval_framework import run_setup as R

MINIMAL = {"vulcan_cfg_name": "W39b", "rp_cm": 1.0, "rstar_cm": 1.0}


def _write(path: Path, doc) -> Path:
    path.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc))
    return path


@pytest.mark.parametrize("name,n_dim", [("v5", 17), ("smoke", 8)])
def test_shipped_spec_loads_to_its_parameter_count(name, n_dim):
    spec = R.load_spec(RUN_DIR / f"{name}.yaml")
    groups = list(OBS.load_real_observations(spec.cfg)["groups"])
    assert len(C.specs_from_config(spec.cfg, groups=groups)) == n_dim
    assert spec.cfg.out_dir == R.output_base(RUN_DIR) / f"{name}_nautilus"
    assert (spec.job is not None) == name.startswith("v5")


def test_variant_replaces_a_base_value_whole(tmp_path):
    _write(tmp_path / "base.yaml", {"runtime": {"solver": "fast"},
                                    "config": dict(MINIMAL, cfg_overrides={"Rp": 1.0, "rtol": 0.2},
                                                   nz=40)})
    spec = R.load_spec(_write(tmp_path / "v.yaml",
                              {"base": "base.yaml", "config": {"cfg_overrides": {"rtol": 0.1}}}))
    assert spec.cfg.cfg_overrides == {"rtol": 0.1}     # the whole dict, not a deep merge
    assert spec.cfg.nz == 40 and spec.runtime.solver == "fast" and spec.name == "v"
    assert spec.base == (tmp_path / "base.yaml").resolve()


@pytest.mark.parametrize("doc,match", [
    ({"config": {"nope": 1}}, "unknown key"),
    ({"config": dict(MINIMAL, dt_max="1e11")}, "1.0e[+]11"),
    ({"config": dict(MINIMAL, out_dir="x")}, "derived"),
    ({"config": dict(MINIMAL, nz=True)}, "must be a number"),
    ({"config": dict(MINIMAL, nz=2.5)}, "must be an integer"),
    ({"sampler": {"n_live": 0}, "config": MINIMAL}, "positive"),
    ({"runtime": {"solver": "cuda"}, "config": MINIMAL}, "fast, ffi or reference"),
    ({"extra": {}, "config": MINIMAL}, "unknown top-level"),
    ({"job": {"queue": "q", "group_list": "g", "select": "s", "walltime": "48:00:00"},
      "config": dict(MINIMAL, walltime_seconds=172800.0)}, "must end at least"),
    ("config:\n  nz: 1\n  nz: 2\n", "duplicate key"),
])
def test_malformed_spec_is_refused_by_name(tmp_path, doc, match):
    with pytest.raises((ValueError, TypeError), match=match):
        R.load_spec(_write(tmp_path / "bad.yaml", doc))


def test_base_with_a_base_is_refused(tmp_path):
    _write(tmp_path / "a.yaml", {"config": MINIMAL})
    _write(tmp_path / "b.yaml", {"base": "a.yaml"})
    with pytest.raises(ValueError, match="base of its own"):
        R.load_spec(_write(tmp_path / "c.yaml", {"base": "b.yaml"}))


def test_runtime_env_refuses_a_contradicting_environment():
    rt = R.Runtime(solver="ffi", chem_rhs="grouped")
    assert R.runtime_env_updates(rt, {"VULCAN3_SOLVER": "ffi"}) == rt.env()
    with pytest.raises(RuntimeError, match="disagrees"):
        R.runtime_env_updates(rt, {"VULCAN3_SOLVER": "fast"})
    with pytest.raises(RuntimeError, match="leaves it unset"):
        R.runtime_env_updates(rt, {"XLA_FLAGS": "--x"})
