"""Run provenance: the target digest binds every class that defines the target
and nothing that churns, and the data identity says why when it cannot resolve.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from retrieval_framework import provenance
from conftest import case_config

# digest_pipe stubs this, so that a test without the data tree can build a manifest.
SCIENCE_DATA_IDENTITY = provenance.science_data_identity


def test_data_identity_never_reports_null_opacity_without_saying_why(tmp_path, monkeypatch):
    """The data identity must not silently claim a run read no opacity data.

    This repo never sets $VULCAN_FORWARD_DATA -- it hands the engine its tree
    through paths.set_data_root -- so resolving the trees from the environment
    yields null for a run that read gigabytes of them. Holds in any
    environment: either the root resolves, or the identity says why not.
    """
    monkeypatch.delenv("VULCAN_FORWARD_DATA", raising=False)
    ident = provenance._data_identity(tmp_path, {})
    assert not (ident["data_root_resolved"] is None and "engine_data_error" not in ident), (
        "data identity is null with no stated reason -- the environment-only resolution regressed"
    )
    # the variable's own field reports the variable, never the resolved root
    assert ident["VULCAN_FORWARD_DATA"] is None


# --- target identity: what a checkpoint's numbers belong to --------------------


@pytest.mark.parametrize(
    ("field", "value", "moves"),
    [
        ("yconv_cri", 0.001, True),  # convergence tolerance
        ("molecules", ("H2O",), True),  # opacity list
        ("nz", 80, True),  # chemistry grid
        ("art_ptop_bar", 1e-8, True),  # pressure domain
        ("prior_lnZ", (-1.0, 1.0), True),  # prior definition
        ("seed", 999, True),  # RNG identity (bit-identical resume)
        ("count_max", 4000, True),  # solver-defined support
        ("art_pbtm_bar", 9.0, True),  # the RT grid
        ("report_kzz_ceiling_x", 10.0, False),  # read by the report only
        ("walltime_seconds", 3600.0, False),  # per-JOB governor
        ("smc_rt_chunk", 24, False),  # batch split, numerically identical
        ("out_dir", Path("/tmp/elsewhere"), False),
    ],
)
def test_target_digest_moves_with_every_bound_class(field, value, moves, digest_pipe):
    """Each manifest class must change the digest, or a resume could carry numbers from a different density.

    A chained RESUME job legitimately changes the per-job settings; binding them would refuse the documented NAS
    chaining workflow.
    """
    base = case_config(molecules=("H2O", "CO2"))
    assert getattr(base, field) != value, f"{field} probe equals the default"
    changed = replace(base, **{field: value})
    assert (provenance.target_digest(base, digest_pipe) != provenance.target_digest(changed, digest_pipe)) is moves


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("VULCAN3_SOLVER", "ffi"),
        ("VULCAN3_CHEM_RHS", "grouped"),
        ("VULCAN3_BT_BUFFERS", "1"),
    ],
)
def test_target_digest_binds_the_solver_env(var, value, digest_pipe, monkeypatch):
    """Each env var is a numerical choice of the solve, so a resume must not switch it."""
    cfg = case_config(molecules=("H2O",))
    monkeypatch.delenv(var, raising=False)
    d0 = provenance.target_digest(cfg, digest_pipe)
    monkeypatch.setenv(var, value)
    assert provenance.target_digest(cfg, digest_pipe) != d0, f"{var} not bound"


def test_target_digest_binds_each_runtime_input_and_fixed_number_alone(digest_pipe, monkeypatch, tmp_path):
    """The composition table's content at one path, XLA's flags and memory fraction, the backend, and a fixed number of
    either package: each, changed alone with everything else held, must move the digest, and restoring it restores it.
    """
    import jax
    from vulcan_forward import constants as engine_constants

    from retrieval_framework import constants

    cfg = case_config(molecules=("H2O",))
    table = tmp_path / "compose.txt"
    table.write_text("a\n")
    # the solver's loaded table, as once vulcan3 is imported
    monkeypatch.setitem(sys.modules, "vulcan3.composition", SimpleNamespace(COM_FILE_PATH=str(table)))
    for var in ("XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
        monkeypatch.delenv(var, raising=False)
    m = provenance.target_manifest(cfg, digest_pipe)
    assert m["manifest_version"] == 8
    assert m["composition_file"]["path"] == str(table.resolve())
    d0 = provenance.manifest_digest(m)

    def digest() -> str:
        return provenance.target_digest(cfg, digest_pipe)

    table.write_text("b\n")
    assert digest() != d0, "composition table content not bound"
    table.write_text("a\n")
    assert digest() == d0
    for var, value in (("XLA_FLAGS", "--xla_gpu_enable_command_buffer="), ("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")):
        with monkeypatch.context() as mp:
            mp.setenv(var, value)
            assert digest() != d0, f"{var} not bound"
        assert digest() == d0
    for module, name, value in (
        (jax, "default_backend", lambda: "gpu"),
        (constants, "T_EDGE_INSET_K", 30.0),
        (engine_constants, "GM_MATCH_RTOL", 0.03),
    ):
        with monkeypatch.context() as mp:
            mp.setattr(module, name, value)
            assert digest() != d0, f"{name} not bound"
        assert digest() == d0


def test_target_digest_binds_a_network_its_thermo_data_and_a_preset_outside_the_package(
    digest_pipe, monkeypatch, tmp_path
):
    """A network, its NASA9 folder and an abundance preset that resolve outside vulcan3's package are outside every
    code state, so their content must move the digest.
    """
    monkeypatch.chdir(tmp_path)
    case = tmp_path / "data" / "c"
    (case / "thermo" / "NASA9").mkdir(parents=True)
    files = {
        "atm.txt": "atm\n",
        "sflux.txt": "flux\n",
        "thermo/net.txt": "net\n",
        "thermo/NASA9/H2.txt": "nasa\n",
        "preset.dat": "preset\n",
    }
    for name, text in files.items():
        (case / name).write_text(text)
    (case / "C.yaml").write_text("atm_file: data/c/atm.txt\nsflux_file: data/c/sflux.txt\n")
    cfg = case_config(
        molecules=("H2O",),
        vulcan_cfg_name="data/c/C.yaml",
        network=str(case / "thermo" / "net.txt"),
        cfg_overrides={"fastchem_solar_abundance_file": "data/c/preset.dat"},
    )
    m = provenance.target_manifest(cfg, digest_pipe)
    assert set(m["external_chemistry_inputs"]) == {"network", "nasa9", "abundance_preset"}
    d0 = provenance.manifest_digest(m)
    for name in ("thermo/net.txt", "thermo/NASA9/H2.txt", "preset.dat"):
        (case / name).write_text(files[name] + "# edited\n")
        assert provenance.target_digest(cfg, digest_pipe) != d0, f"{name} not bound"
        (case / name).write_text(files[name])
    assert provenance.target_digest(cfg, digest_pipe) == d0


def test_a_run_record_refuses_a_config_that_names_no_chemistry(digest_pipe):
    """The external-input binding needs the network and the case config: an unset one is refused, never skipped."""
    with pytest.raises(ValueError, match="needs both to name its chemistry inputs"):
        provenance.target_manifest(case_config(network=None), digest_pipe)


def test_the_baseline_c_to_o_is_the_case_s_own():
    """The report's C/O multiplies the case's C_H / O_H, after its overrides; a solar-preset case is refused."""
    assert provenance.baseline_c_to_o("W39b", {}) == 0.00295 / 0.00537
    assert provenance.baseline_c_to_o("W39b", {"C_H": 0.00537}) == 1.0
    with pytest.raises(ValueError, match="use_solar"):
        provenance.baseline_c_to_o("W39b", {"use_solar": True})
    with pytest.raises(ValueError, match="ini_mix"):  # another start column, another C/O
        provenance.baseline_c_to_o("W39b", {"ini_mix": "const_mix"})


def test_target_digest_moves_with_the_observations(digest_pipe):
    cfg = case_config(molecules=("H2O",))
    a = provenance.target_digest(cfg, digest_pipe)
    digest_pipe.obs_sigma = digest_pipe.obs_sigma * 2.0
    assert provenance.target_digest(cfg, digest_pipe) != a, "observations not bound"


def _repo(**over):
    st = {"commit": "a" * 40, "dirty": True, "dirty_files": ["a.py"], "src_diff": "d" * 64}
    st.update(over)
    return {"r": st}


@pytest.mark.parametrize(
    ("state", "moves"),
    [
        ({"dirty_files": ["a.py", "b.py"]}, False),  # the LIST churns; not an identity
        ({"dirty": False}, False),  # a PBS edit: the same commit and the same source
        ({"dirty": True}, False),
        ({"dirty": False, "dirty_files": [], "src_diff": None}, True),
        ({"src_diff": "e" * 64}, True),  # a DIFFERENT uncommitted src edit
        ({"commit": "b" * 40}, True),  # a different commit
    ],
)
def test_target_digest_binds_code_state_not_file_churn(state, moves, digest_pipe, monkeypatch):
    """A scratch edit must not refuse a resume, but a different source diff must.

    `dirty_files` churns with any untracked file; `src_diff` is the content hash
    of `git diff HEAD -- src`, so it moves only when the code that defines the
    target changed.
    """
    cfg = case_config(molecules=("H2O",))
    monkeypatch.setattr(provenance, "_repo_states", lambda *_a, **_k: _repo())
    d0 = provenance.target_digest(cfg, digest_pipe)
    monkeypatch.setattr(provenance, "_repo_states", lambda *_a, **_k: _repo(**state))
    assert (provenance.target_digest(cfg, digest_pipe) != d0) is moves


def test_target_digest_binds_the_cia_tables(digest_pipe, monkeypatch, tmp_path):
    """The CIA tables are outside git, so nothing else can notice a swapped continuum."""
    from vulcan_forward import paths

    monkeypatch.setattr(provenance, "science_data_identity", SCIENCE_DATA_IDENTITY)
    a, b = tmp_path / "a.cia", tmp_path / "b.cia"
    a.write_text("a")
    b.write_text("b")
    monkeypatch.setattr(paths, "cia_h2h2_file", lambda: a)
    monkeypatch.setattr(paths, "cia_h2he_file", lambda: a)
    cfg = case_config(molecules=())
    d0 = provenance.target_digest(cfg, digest_pipe)
    monkeypatch.setattr(paths, "cia_h2he_file", lambda: b)
    assert provenance.target_digest(cfg, digest_pipe) != d0


def test_untracked_source_moves_the_code_state(tmp_path, monkeypatch):
    """`git diff HEAD` cannot see an untracked module, but importing one changes the code that defines the target."""
    import subprocess
    from retrieval_framework.provenance import _src_state

    env = {**os.environ, "HOME": str(tmp_path), "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x", "--allow-empty"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    clean = _src_state(repo)
    (repo / "src" / "new_module.py").write_text("x = 1\n")
    one = _src_state(repo)
    (repo / "src" / "new_module.py").write_text("x = 2\n")
    two = _src_state(repo)
    assert len({clean, one, two}) == 3, "untracked src content unbound"
    (repo / "src" / "new_module.py").unlink()
    assert _src_state(repo) == clean


def test_target_digest_binds_a_path_style_case_and_its_files(digest_pipe, tmp_path, monkeypatch):
    """A path-style case (`vulcan_cfg_name: data/<case>/X.yaml`) and the atm and
    stellar-flux files it names live outside every tree the code state binds, so
    an edit to any of them must refuse a resume; the shipped cases resolve all
    three files as vulcan3 does, from the repo root.
    """
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    case = tmp_path / "data" / "c"
    case.mkdir(parents=True)
    (case / "atm.txt").write_text("atm\n")
    (case / "sflux.txt").write_text("flux\n")
    (case / "C.yaml").write_text("atm_file: data/c/atm.txt\nsflux_file: data/c/sflux.txt\n")
    cfg = case_config(molecules=("H2O",), vulcan_cfg_name="data/c/C.yaml")
    d0 = provenance.target_digest(cfg, digest_pipe)
    for f in ("atm.txt", "sflux.txt", "C.yaml"):
        (case / f).write_text((case / f).read_text() + "# edited\n")
        d1 = provenance.target_digest(cfg, digest_pipe)
        assert d1 != d0, f"{f} not bound"
        d0 = d1
    monkeypatch.chdir(repo)
    for name in ("W39b", "data/survey/w107b/W107b.yaml"):
        got = provenance.case_inputs(name, {})
        assert all(got[k] for k in ("config", "atm_file", "sflux_file")), (name, got)


@pytest.mark.parametrize("absent", ["cia", "opacity"])
def test_a_target_manifest_refuses_a_table_it_cannot_hash(digest_pipe, monkeypatch, tmp_path, absent):
    """A run read these tables: a manifest that says nothing about one would bind nothing."""
    from vulcan_forward import exomolop, paths

    monkeypatch.setattr(provenance, "science_data_identity", SCIENCE_DATA_IDENTITY)
    table = tmp_path / "table"
    table.write_text("x")
    missing = tmp_path / "absent.dat"
    monkeypatch.setattr(paths, "cia_h2h2_file", lambda: table)
    monkeypatch.setattr(paths, "cia_h2he_file", lambda: missing if absent == "cia" else table)
    monkeypatch.setattr(exomolop, "table_path", lambda _m: missing if absent == "opacity" else table)
    with pytest.raises(FileNotFoundError, match=r"absent\.dat"):
        provenance.target_manifest(case_config(molecules=("H2O",)), digest_pipe)


def test_an_archived_manifest_hashes_as_it_was_written(tmp_path):
    """Version 5 hashed `dirty`; its stored digest must still match, or every old certificate calls itself foreign."""
    import hashlib
    import json

    old = {"manifest_version": 5, "code": {"r": {"commit": "c", "dirty": True, "src_diff": None}}}
    new = {"manifest_version": 6, "code": {"r": {"commit": "c", "dirty": True, "src_diff": None}, "q": None}}
    (tmp_path / provenance.MANIFEST_FILE).write_text(json.dumps(old))
    want = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    assert provenance.archived_manifest_digest(tmp_path) == want
    (tmp_path / provenance.MANIFEST_FILE).write_text(json.dumps(new))
    clean = {"manifest_version": 6, "code": {"r": {"commit": "c", "dirty": False, "src_diff": None}, "q": None}}
    d_new = provenance.archived_manifest_digest(tmp_path)
    (tmp_path / provenance.MANIFEST_FILE).write_text(json.dumps(clean))
    assert provenance.archived_manifest_digest(tmp_path) == d_new


@pytest.mark.parametrize("manifest", ['{"manifest_version": 6}', '{"manifest_version": 6, "code": {"r": 3}}'])
def test_a_damaged_manifest_reads_as_unreadable(tmp_path, manifest):
    (tmp_path / provenance.MANIFEST_FILE).write_text(manifest)
    assert provenance.archived_manifest_digest(tmp_path).startswith("unreadable: ")
