"""The run specification (run_setup.load_spec): every shipped spec loads to the
parameter count it is documented with and to its derived output directory; a variant
replaces its base's values one level deep, whole; a malformed file is refused by name;
the runtime block is exported only when the environment does not contradict it;
the forward build refuses an engine frozen on another network than the config's;
check_inputs refuses a missing input file before anything is queued or built;
tools/submit.py charges $PBS_GROUP and sets the wall from the spec's governor."""
import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import RUN_DIR
from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
from retrieval_framework import run_setup as R

MINIMAL = {"vulcan_cfg_name": "W39b", "network": "thermo/SNCHO_photo_network_2025.txt",
           "rp_cm": 1.0, "rstar_cm": 1.0}


def _write(path: Path, doc) -> Path:
    path.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc))
    return path


@pytest.mark.parametrize("case,name,n_dim", [      # v5 and w107b integrate their nuisances out
    *[(RUN_DIR.name, n, d) for n, d in (
        ("wasp39_test1", 12), ("wasp39_test1_kzz1000", 12), ("wasp39_test1_deck", 11),
        ("wasp39_test1_kzz_powerlaw", 13), ("wasp39_test1_adiabat", 13), ("smoke", 8))],
    *[("w107b", n, 12) for n in ("w107b_joint", "w107b_g395h", "w107b_welbanks")]])
def test_shipped_spec_loads_to_its_parameter_count(case, name, n_dim):
    run_dir = RUN_DIR.parent / case
    spec = R.load_spec(run_dir / f"{name}.yaml")
    groups = list(OBS.load_real_observations(spec.cfg)["groups"])
    assert len(C.specs_from_config(spec.cfg, groups=groups)) == n_dim
    assert spec.cfg.out_dir == R.output_base(run_dir) / f"{name}_nautilus"


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
    ({"job": {}, "config": MINIMAL}, "unknown top-level"),    # the PBS job is submit.py's
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


def test_submit_charges_pbs_group_and_walls_past_the_governor(tmp_path, monkeypatch, capsys):
    """The qsub line charges $PBS_GROUP, walls 2 h past the governor on gpu_long, and
    carries the spec digest the job checks (run_setup.main refuses an edited spec)."""
    monkeypatch.chdir(tmp_path)                  # submit.py moves to the repo; restored after
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "submit.py"))["main"]
    spec = str(_write(tmp_path / "s.yaml", {"config": dict(MINIMAL, walltime_seconds=165600.0)}))
    monkeypatch.delenv("PBS_GROUP", raising=False)
    with pytest.raises(SystemExit, match="PBS_GROUP"):
        main([spec, "--dry-run"])
    monkeypatch.setenv("PBS_GROUP", "g0")
    main([spec, "--dry-run"])
    line = capsys.readouterr().out
    assert "walltime=48:00:00" in line and "group_list=g0" in line and "-q gpu_long" in line
    digest = R.spec_digest(R.load_spec(spec))
    assert f"VR_SPEC_DIGEST={digest}" in line
    R.main([spec, "--digest", digest])
    Path(spec).write_text(Path(spec).read_text() + "# edited after submission\n")
    with pytest.raises(SystemExit, match="changed after it was submitted"):
        R.main([spec, "--digest", digest])
    with pytest.raises(SystemExit, match="past gpu_long"):       # 120 h + 2 h > the 120 h queue
        main([str(_write(tmp_path / "u.yaml", {"config": dict(MINIMAL, walltime_seconds=432000.0)})),
              "--dry-run"])
    with pytest.raises(SystemExit, match="walltime_seconds"):
        main([str(_write(tmp_path / "t.yaml", {"config": MINIMAL})), "--dry-run"])


# find_spec, not importorskip: importing exojax before vulcan_chem breaks the
# import-order guard for every later test in the worker.
@pytest.mark.skipif(importlib.util.find_spec("exojax") is None,
                    reason="needs exojax (retrieval_forward); CI is stack-free")
def test_forward_build_refuses_an_engine_frozen_on_another_network():
    """A run's network is its config's: a tool that rebuilt the run without
    C.export_network would solve other chemistry, so the build refuses."""
    from retrieval_framework import retrieval_forward as RF
    RF._check_frozen_network(C.Config(**MINIMAL))          # what the test session froze
    with pytest.raises(RuntimeError, match="export_network"):
        RF._check_frozen_network(C.Config(**{**MINIMAL, "network": "thermo/SNCHO_photo_network.txt"}))


@pytest.mark.parametrize("broken, msg", [
    (None, None), ("network", "network"), ("atm", "atm_file"), ("obs", "obs.csv"),
    ("window", "bins of"),
])
def test_check_inputs_refuses_a_missing_input_before_any_build(broken, msg, tmp_path,
                                                               monkeypatch):
    """submit.py and validate_env refuse a spec whose input files would fail only
    minutes into a job: the network, the case's atm and stellar-flux files, the
    observation products and, for real data, bins in the fit window."""
    monkeypatch.chdir(tmp_path)
    case = tmp_path / "data" / "c"
    case.mkdir(parents=True)
    for f in ("atm.txt", "sflux.txt"):
        (case / f).write_text("x\n")
    (case / "C.yaml").write_text("atm_file: data/c/atm.txt\nsflux_file: data/c/sflux.txt\n")
    rows = [f"{i},{w},{w - 0.01},{w + 0.01},0.1,0.001,0.001"
            for i, w in enumerate(np.linspace(3.0, 3.4, 5))]
    (case / "obs.csv").write_text("i,wave,lo,hi,rprs,el,eh\n" + "\n".join(rows) + "\n")
    over = dict(vulcan_cfg_name="data/c/C.yaml", obs_dir="data/c",
                obs_products={"G": ["obs.csv"]}, combo=("G",), obs_wl_lo=2.9, obs_wl_hi=3.5)
    if broken == "network":
        over["network"] = "thermo/no_such_network.txt"
    elif broken in ("atm", "obs"):
        (case / {"atm": "atm.txt", "obs": "obs.csv"}[broken]).unlink()
    elif broken == "window":
        over.update(obs_wl_lo=4.0, obs_wl_hi=5.0)
    cfg = C.Config(**{**MINIMAL, **over})
    if msg is None:
        R.check_inputs(cfg)
    else:
        with pytest.raises((OSError, ValueError), match=msg):
            R.check_inputs(cfg)


def test_one_run_directory_check_for_submit_and_the_runner(tmp_path):
    """submit.py and run_nautilus refuse with the same check: a resume needs the
    checkpoint, a fresh run refuses a checkpoint or anchor files, and the empty
    anchors/ a run stopped at its startup check leaves does not block a resubmit."""
    (tmp_path / "anchors").mkdir()
    R.check_run_dir(tmp_path, resume=False)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        R.check_run_dir(tmp_path, resume=True)
    (tmp_path / "anchors" / "anchors_00000.npz").write_bytes(b"")
    with pytest.raises(FileExistsError, match="--resume"):
        R.check_run_dir(tmp_path, resume=False)
    (tmp_path / "nautilus_checkpoint.hdf5").write_bytes(b"")
    R.check_run_dir(tmp_path, resume=True)
