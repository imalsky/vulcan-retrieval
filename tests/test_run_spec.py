"""The run specification (run_setup.load_spec): every shipped spec loads to the
parameter count it is documented with and to its derived output directory; a variant
replaces its base's values one level deep, whole; a malformed file is refused by name;
the runtime block is exported only when the environment does not contradict it;
a network other than the one vulcan3 froze is refused;
check_inputs refuses a missing input file before anything is queued or built;
tools/submit.py charges $PBS_GROUP and sets the wall from the spec's governor.
"""

import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import RUN_DIR
from retrieval_framework import config_schema as C
from retrieval_framework import observations as OBS
from retrieval_framework import run_setup as R

MINIMAL = {
    "vulcan_cfg_name": "W39b",
    "network": "thermo/SNCHO_photo_network_2025.txt",
    "atom_list": ["H", "O", "C", "N", "S"],
    "rp_cm": 1.0,
    "rstar_cm": 1.0,
    # the keys every spec states (case_keys.CASE_KEYS)
    "tp_gravity_cgs": 422.0,
    "p_ref_bar": 1.0e-3,
    "art_pbtm_bar": 7.0,
    "molecules": ["H2O", "CO2", "CO", "CH4", "NH3", "HCN", "H2S", "SO2"],
    "nu_min": 1923.0,
    "nu_max": 3450.0,
    "cloud_model": "powerlaw",
    "count_max": 5000,
    "cold_lanes": 8,
    "prior_Tirr": [800.0, 2600.0],
    "prior_lnM_sigma": 0.11,
    "baseline_metallicity_solar": 10.0,
    "seed": 1,
    "tp_deep_p_bar": 1.0,
    "tp_Tint_K": 150.0,  # tp_infer_Tint is off (case_keys.MODE_KEYS)
    "prior_lnZ": [-2.3, 2.3],
    "prior_c_o": [-1.7, 0.24],
    "prior_lnKzz": [-2.3, 6.9],
    "prior_log10kappa": [-3.5, 0.5],
    "prior_lnR0": [-0.05, 0.15],
    # what the schema's modes read here: gamma inferred, the power-law cloud, offsets inferred
    "prior_log10gamma": [-2.0, 0.7],
    "prior_log10kappa_cloud": [-7.0, 1.0],
    "prior_cloud_alpha": [0.0, 6.0],
    "prior_offset_ppm": [-800.0, 800.0],
}


# The truths of MINIMAL's sampled parameters, at the schema's values, without truth_noise_inflation.
TRUTHS = {
    "truth_lnZ": 0.0,
    "truth_c_o": 0.0,
    "truth_lnKzz": 0.0,
    "truth_Tirr": 1650.0,
    "truth_log10kappa": -2.0,
    "truth_log10gamma": -0.4,
    "truth_lnR0": 0.0,
    "truth_log10kappa_cloud": -6.5,
    "truth_cloud_alpha": 1.0,
}


def _drop(*keys):
    return {k: v for k, v in MINIMAL.items() if k not in keys}


def _write(path: Path, doc) -> Path:
    path.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc))
    return path


@pytest.mark.parametrize(
    ("case", "name", "n_dim"),
    [  # the WASP-39b case and the survey integrate their nuisances out
        *[
            (RUN_DIR.name, n, d)
            for n, d in (
                ("wasp39_test1", 12),
                ("wasp39_test1_kzz1000", 12),
                ("wasp39_test1_deck", 11),
                ("wasp39_test1_kzz_powerlaw", 13),
                ("wasp39_test1_adiabat", 13),
                ("smoke", 8),
            )
        ],
        *[("survey", n, 12) for n in ("wasp39b", "wasp107b", "wasp96b", "wasp166b", "hd189733b", "hd209458b")],
    ],
)
def test_shipped_spec_loads_to_its_parameter_count(case, name, n_dim):
    run_dir = RUN_DIR.parent / case
    spec = R.load_spec(run_dir / f"{name}.yaml")
    # the smoke case names no data source: one synthetic group
    groups = list(OBS.load_real_observations(spec.cfg)["groups"]) if spec.cfg.obs_dir else ["SYNTH"]
    assert len(C.specs_from_config(spec.cfg, groups=groups)) == n_dim
    assert spec.cfg.out_dir == R.output_base(run_dir) / f"{name}_nautilus"


def test_variant_replaces_a_base_value_whole(tmp_path):
    _write(
        tmp_path / "base.yaml",
        {"runtime": {"solver": "fast"}, "config": dict(MINIMAL, cfg_overrides={"Rp": 1.0, "rtol": 0.2}, nz=40)},
    )
    spec = R.load_spec(_write(tmp_path / "v.yaml", {"base": "base.yaml", "config": {"cfg_overrides": {"rtol": 0.1}}}))
    assert spec.cfg.cfg_overrides == {"rtol": 0.1}  # the whole dict, not a deep merge
    assert spec.cfg.nz == 40
    assert spec.runtime.solver == "fast"
    assert spec.name == "v"
    assert spec.base == (tmp_path / "base.yaml").resolve()


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"config": {"nope": 1}}, "unknown key"),
        ({"config": dict(MINIMAL, dt_max="1e11")}, "1.0e[+]11"),
        ({"config": dict(MINIMAL, out_dir="x")}, "derived"),
        ({"config": dict(MINIMAL, nz=True)}, "must be a number"),
        ({"config": dict(MINIMAL, nz=2.5)}, "must be an integer"),
        ({"sampler": {"n_live": 0}, "config": MINIMAL}, "positive"),
        ({"runtime": {"solver": "cuda"}, "config": MINIMAL}, "fast, ffi or reference"),
        ({"job": {}, "config": MINIMAL}, "unknown top-level"),  # the PBS job is submit.py's
        ("config:\n  nz: 1\n  nz: 2\n", "duplicate key"),
        # a spec states its planet: the schema's default is another planet's
        ({"config": {k: v for k, v in MINIMAL.items() if k != "tp_gravity_cgs"}}, r"\['tp_gravity_cgs'\] unset"),
        ({"config": {k: v for k, v in MINIMAL.items() if k not in ("molecules", "cloud_model")}}, "unset"),
        # the run's own numbers, no schema default (case_keys.CASE_KEYS)
        *[
            ({"config": _drop(k)}, rf"\['{k}'\] unset")
            for k in ("count_max", "cold_lanes", "prior_Tirr", "prior_lnM_sigma", "baseline_metallicity_solar")
        ],
        ({"config": dict(MINIMAL, count_max=None)}, r"\['count_max'\] unset"),
        # a key the case's mode reads, with no default (case_keys.required_keys)
        ({"config": _drop("tp_Tint_K")}, r"\['tp_Tint_K'\] unset"),
        (
            {"config": dict(_drop("prior_log10kappa_cloud", "prior_cloud_alpha"), cloud_model="deck")},
            r"\['prior_log10P_deck'\] unset",
        ),
        ({"config": dict(MINIMAL, tp_adiabat=True)}, r"\['prior_log10P_rcb', 'tp_nabla_ad'\] unset"),
        ({"config": _drop("prior_log10kappa")}, r"\['prior_log10kappa'\] unset"),
        ({"config": _drop("prior_offset_ppm")}, r"\['prior_offset_ppm'\] unset"),
        ({"config": dict(MINIMAL, infer_noise_inflation=True)}, r"\['prior_noise_inflation'\] unset"),
        (
            {"config": dict(MINIMAL, tp_infer_gamma=False)},
            r"\['prior_log10gamma'\] would change nothing: prior_log10gamma: no parameter",
        ),
        ({"config": dict(_drop("prior_log10gamma"), tp_infer_gamma=False)}, r"\['tp_gamma_fixed'\] unset"),
        (
            {"config": dict(MINIMAL, tp_infer_Tint=True, tp_deep_param="T1bar", prior_T1bar=[320.0, 2980.0])},
            r"\['tp_Tint_K'\] would change nothing",
        ),
        (
            {
                "config": dict(
                    _drop("tp_Tint_K", "tp_deep_p_bar"),
                    tp_infer_Tint=True,
                    tp_deep_param="T1bar",
                    prior_T1bar=[320.0, 2980.0],
                )
            },
            r"\['tp_deep_p_bar'\] unset",
        ),
        # a synthetic run states every truth it injects, truth_noise_inflation included
        (
            {"config": dict(_drop("prior_offset_ppm"), generate_synthetic_data=True, infer_offsets=False)},
            r"\['truth_.*\] unset",
        ),
        (  # the injection reads truth_noise_inflation whether or not the inflation is inferred
            {"config": dict(_drop("prior_offset_ppm"), generate_synthetic_data=True, infer_offsets=False, **TRUTHS)},
            r"\['truth_noise_inflation'\] unset",
        ),
        (
            {"config": dict(MINIMAL, truth_noise_inflation=1.0)},
            r"truth_noise_inflation: it is read only with generate_synthetic_data",
        ),
        (
            {"config": dict(MINIMAL, obs_dir="data/x", obs_products={"G": ["a.csv"]})},
            r"\['combo', 'obs_wl_hi', 'obs_wl_lo'\] unset",
        ),
        # a key the case's mode does not read (case_keys.refused_keys)
        ({"config": dict(MINIMAL, tp_gamma_fixed=0.3)}, r"tp_gamma_fixed: it is read only with tp_infer_gamma: false"),
        ({"config": dict(MINIMAL, tp_infer_Tint=True)}, r"\['tp_Tint_K'\] would change nothing"),
        (
            {"config": dict(MINIMAL, prior_log10P_deck=[-6.0, 0.8])},
            r"prior_log10P_deck: no parameter this spec samples reads it",
        ),
        ({"config": dict(MINIMAL, cloud_model="none")}, "prior_cloud_alpha: no parameter this spec samples reads it"),
        # what validate_config no longer repeats: the table refuses a deep parameter or a per-group split nobody reads
        ({"config": dict(MINIMAL, tp_deep_param="T1bar")}, "tp_deep_param: it is read only with tp_infer_Tint: true"),
        ({"config": dict(MINIMAL, noise_inflation_per_group=True)}, "noise_inflation_per_group: it is read only"),
        ({"config": dict(MINIMAL, kzz_p_ref_bar=1.0e-3)}, "only with kzz_mode: powerlaw"),
        ({"config": dict(MINIMAL, tp_nabla_ad=0.3)}, "only with tp_adiabat"),
        ({"config": dict(MINIMAL, truth_lnZ=0.5)}, "only with generate_synthetic_data"),
        ({"config": dict(MINIMAL, obs_wl_lo=2.0)}, "only with a data source"),
    ],
)
def test_malformed_spec_is_refused_by_name(tmp_path, doc, match):
    with pytest.raises((ValueError, TypeError), match=match):
        R.load_spec(_write(tmp_path / "bad.yaml", doc))


def test_a_variant_unsets_keys_its_base_states(tmp_path):
    """A clear-sky variant of a power-law base drops the base's cloud priors with `unset:`; a key the base does not
    state, a key the variant restates, and `unset:` without a base are refused.
    """
    _write(tmp_path / "base.yaml", {"config": MINIMAL})
    cloud = ["prior_log10kappa_cloud", "prior_cloud_alpha"]
    clear = {"base": "base.yaml", "unset": cloud, "config": {"cloud_model": "none"}}
    spec = R.load_spec(_write(tmp_path / "clear.yaml", clear))
    assert spec.cfg.cloud_model == "none"
    with pytest.raises(ValueError, match="prior_cloud_alpha: no parameter"):  # without unset
        R.load_spec(_write(tmp_path / "c0.yaml", {"base": "base.yaml", "config": {"cloud_model": "none"}}))
    for name, doc in (
        ("c1", {**clear, "unset": [*cloud, "prior_log10P_deck"]}),
        ("c2", {**clear, "config": {"cloud_model": "none", "prior_cloud_alpha": [0.0, 1.0]}}),
    ):
        with pytest.raises(ValueError, match="`unset:` names"):
            R.load_spec(_write(tmp_path / f"{name}.yaml", doc))
    with pytest.raises(ValueError, match=r"names \['prior_cloud_alpha'\] more than once"):
        R.load_spec(_write(tmp_path / "c4.yaml", {**clear, "unset": [*cloud, "prior_cloud_alpha"]}))
    with pytest.raises(ValueError, match="needs a base"):
        R.load_spec(_write(tmp_path / "c3.yaml", {"unset": cloud, "config": MINIMAL}))


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
    carries the spec digest the job checks (run_setup.main refuses an edited spec).
    """
    monkeypatch.chdir(tmp_path)  # submit.py moves to the repo; restored after
    main = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "submit.py"))["main"]
    spec = str(_write(tmp_path / "s.yaml", {"config": dict(MINIMAL, walltime_seconds=165600.0)}))
    monkeypatch.delenv("PBS_GROUP", raising=False)
    with pytest.raises(SystemExit, match="PBS_GROUP"):
        main([spec, "--dry-run"])
    monkeypatch.setenv("PBS_GROUP", "g0")
    main([spec, "--dry-run"])
    line = capsys.readouterr().out
    assert "walltime=48:00:00" in line
    assert "group_list=g0" in line
    assert "-q gpu_long" in line
    digest = R.spec_digest(R.load_spec(spec))
    assert f"VR_SPEC_DIGEST={digest}" in line
    R.main([spec, "--digest", digest])
    Path(spec).write_text(Path(spec).read_text() + "# edited after submission\n")
    with pytest.raises(SystemExit, match="changed after it was submitted"):
        R.main([spec, "--digest", digest])
    with pytest.raises(SystemExit, match="past gpu_long"):  # 120 h + 2 h > the 120 h queue
        main([str(_write(tmp_path / "u.yaml", {"config": dict(MINIMAL, walltime_seconds=432000.0)})), "--dry-run"])
    with pytest.raises(SystemExit, match="walltime_seconds"):
        main([str(_write(tmp_path / "t.yaml", {"config": MINIMAL})), "--dry-run"])


# find_spec, not importorskip: importing exojax before vulcan_chem breaks the
# import-order guard for every later test in the worker.
@pytest.mark.skipif(
    importlib.util.find_spec("exojax") is None, reason="needs exojax (the engine's vulcan_chem); CI is stack-free"
)
def test_the_network_is_refused_once_vulcan3_froze_another():
    """A run's network is its config's: once vulcan3 froze the session's network, a tool that rebuilds a run on another
    would solve other chemistry, so export_network refuses it (the engine's select_network).
    """
    import vulcan3  # noqa: F401  # frozen on the session's network (conftest)

    C.export_network(C.Config(**MINIMAL))
    with pytest.raises(RuntimeError, match="already imported"):
        C.export_network(C.Config(**{**MINIMAL, "network": "thermo/SNCHO_photo_network.txt"}))
    with pytest.raises(RuntimeError, match="already imported"):
        C.export_network(C.Config(**{**MINIMAL, "atom_list": ["H", "O", "C", "N"]}))


@pytest.mark.parametrize(
    ("broken", "msg"),
    [
        (None, None),
        ("network", "network"),
        ("atm", "atm_file"),
        ("obs", "obs.csv"),
        ("window", "bins of"),
        ("window_synthetic", "bins of"),  # a named source is fitted or refused, in a synthetic run too
    ],
)
def test_check_inputs_refuses_a_missing_input_before_any_build(broken, msg, tmp_path, monkeypatch):
    """submit.py and validate_env refuse a spec whose input files would fail only
    minutes into a job: the network, the case's atm and stellar-flux files, the
    observation products and, for real data, bins in the fit window.
    """
    monkeypatch.chdir(tmp_path)
    case = tmp_path / "data" / "c"
    case.mkdir(parents=True)
    for f in ("atm.txt", "sflux.txt"):
        (case / f).write_text("x\n")
    (case / "C.yaml").write_text("atm_file: data/c/atm.txt\nsflux_file: data/c/sflux.txt\n")
    rows = [f"{i},{w},{w - 0.01},{w + 0.01},0.1,0.001,0.001" for i, w in enumerate(np.linspace(3.0, 3.4, 5))]
    (case / "obs.csv").write_text("i,wave,lo,hi,rprs,el,eh\n" + "\n".join(rows) + "\n")
    over = {
        "vulcan_cfg_name": "data/c/C.yaml",
        "obs_dir": "data/c",
        "obs_products": {"G": ["obs.csv"]},
        "combo": ("G",),
        "obs_wl_lo": 2.9,
        "obs_wl_hi": 3.5,
    }
    if broken == "network":
        over["network"] = "thermo/no_such_network.txt"
    elif broken in ("atm", "obs"):
        (case / {"atm": "atm.txt", "obs": "obs.csv"}[broken]).unlink()
    elif broken in ("window", "window_synthetic"):
        over.update(obs_wl_lo=4.0, obs_wl_hi=5.0, generate_synthetic_data=broken == "window_synthetic")
    cfg = C.Config(**{**MINIMAL, **over})
    if msg is None:
        R.check_inputs(cfg)
    else:
        with pytest.raises((OSError, ValueError), match=msg):
            R.check_inputs(cfg)


@pytest.mark.parametrize(("state", "held"), [("F", None), ("R", "7.srv")])
def test_resume_chains_only_after_a_job_pbs_still_holds(tmp_path, monkeypatch, state, held):
    """PBS keeps a finished job in its history, where a plain qstat exits with an error: the resume chain asks with -x
    and waits only for a job still queued or running.
    """
    qstat = tmp_path / "bin" / "qstat"
    qstat.parent.mkdir()
    qstat.write_text(
        "#!/bin/sh\n"
        f'case "{state}$*" in R*|*-x*) echo \'{{"Jobs": {{"7.srv": {{"job_state": "{state}"}}}}}}\';;\n'
        '*) echo "qstat: 7.srv Job has finished, use -x or -H to obtain historical job information" >&2; exit 35;;\n'
        "esac\n"
    )
    qstat.chmod(0o755)
    monkeypatch.setenv("PATH", str(qstat.parent), prepend=":")
    submit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools" / "submit.py"))
    assert submit["_last_active_job"](_write(tmp_path / "jobs.txt", "7.srv resume=0\n")) == held


def test_one_run_directory_check_for_submit_and_the_runner(tmp_path):
    """submit.py and run_nautilus refuse with the same check: a resume needs the
    checkpoint, a fresh run refuses a checkpoint or anchor files, and the empty
    anchors/ a run stopped at its startup check leaves does not block a resubmit.
    """
    (tmp_path / "anchors").mkdir()
    R.check_run_dir(tmp_path, resume=False)
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        R.check_run_dir(tmp_path, resume=True)
    (tmp_path / "anchors" / "anchors_00000.npz").write_bytes(b"")
    with pytest.raises(FileExistsError, match="--resume"):
        R.check_run_dir(tmp_path, resume=False)
    (tmp_path / "nautilus_checkpoint.hdf5").write_bytes(b"")
    R.check_run_dir(tmp_path, resume=True)


@pytest.mark.parametrize("solver", ["fast", "ffi"])
@pytest.mark.parametrize(("n_batch", "want_batch"), [(None, 24), (10, 10)])
def test_the_lanes_that_run_are_cold_lanes_under_every_solver(
    tmp_path, monkeypatch, capsys, solver, n_batch, want_batch
):
    """cold_lanes is the lane count that runs, under ffi too; the batch is n_batch, else three points per lane; and the
    dry run prints both.
    """
    from retrieval_framework.run_nautilus import lanes_and_batch, main

    doc = {"runtime": {"solver": solver}, "sampler": {"n_batch": n_batch}, "config": MINIMAL}
    spec = R.load_spec(_write(tmp_path / "lanes.yaml", doc))
    assert lanes_and_batch(spec) == (MINIMAL["cold_lanes"], want_batch)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    main([str(spec.path), "--dry-run"])
    assert f"lanes: {MINIMAL['cold_lanes']}   batch: {want_batch} points" in capsys.readouterr().out


@pytest.mark.slow
def test_a_fresh_process_builds_the_smoke_case_with_no_network_in_the_environment():
    """build_pipeline selects the case's network itself: no VULCAN3_* variable and no prior vulcan3 import needed."""
    repo = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith("VULCAN3_")}
    job = (
        "import sys\n"
        "assert 'vulcan3' not in sys.modules\n"
        "from retrieval_framework import pipeline as P, run_setup as R\n"
        "pipe = P.build_pipeline(R.load_spec('runs/w39b_smc_retrieval/smoke.yaml').cfg)\n"
        "from vulcan3.config import default_config\n"
        "print('FROZEN', default_config().network, ','.join(default_config().atom_list), pipe.n_dim)\n"
    )
    r = subprocess.run([sys.executable, "-c", job], cwd=repo, env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "FROZEN thermo/SNCHO_photo_network_2025.txt H,O,C,N,S 8" in r.stdout


def test_dry_run_prints_the_resolved_settings_and_touches_nothing(capsys, monkeypatch):
    """What a user checks before submitting: the production spec's parameters, priors and output directory, with no
    run directory written.
    """
    from retrieval_framework.run_nautilus import main

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    spec = R.load_spec(RUN_DIR / "wasp39_test1.yaml")
    existed = spec.cfg.out_dir.exists()
    main([str(spec.path), "--dry-run"])
    out = capsys.readouterr().out
    assert f"nz={spec.cfg.nz}" in out
    assert f"count_max={spec.cfg.count_max}" in out
    assert f"out_dir: {spec.cfg.out_dir}" in out
    assert "REAL observed spectrum" in out
    specs = C.specs_from_config(spec.cfg, groups=list(spec.cfg.combo))
    assert f"parameters ({len(specs)})" in out
    for s in specs:
        assert f"    {s.name:<17s} " in out, s.name
    assert spec.cfg.out_dir.exists() == existed
