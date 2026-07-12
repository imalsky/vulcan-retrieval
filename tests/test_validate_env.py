"""validate_env: pure-helper and data-tree checks (no chemistry/RT stack needed).

The interpreter/jax/editable checks are exercised for real by the PBS preflight
and the bootstrap; here we pin the environment-independent logic: version-file
parsing, the data-tree fingerprints, and the loud aggregate verdict on a
missing project root.
"""
from __future__ import annotations

from pathlib import Path

from retrieval_framework import validate_env as V


def _reset():
    V._ERRORS.clear()
    V._WARNINGS.clear()


def test_repo_version_parses_version_file(tmp_path: Path):
    pkg = tmp_path / "src" / "somepkg"
    pkg.mkdir(parents=True)
    (pkg / "_version.py").write_text('__version__ = "1.2.3"\n')
    assert V._repo_version(tmp_path, "somepkg") == "1.2.3"
    assert V._repo_version(tmp_path, "otherpkg") is None


def test_data_tree_checks_flag_missing_and_pass_when_seeded(tmp_path: Path):
    _reset()
    data = tmp_path / "vulcan-retrieval" / "data"
    (data / "cm24_wasp39b").mkdir(parents=True)
    (data / "exojax_linelists").mkdir()
    (data / "opacity_cache").mkdir()
    V._check_data_tree(tmp_path)
    # missing CSVs, CO dir, and both CIA files are ERRORS; line lists a WARNING
    assert len(V._ERRORS) == 4
    assert len(V._WARNINGS) == 1

    _reset()
    (data / "cm24_wasp39b" / "obs.csv").write_text("wl,depth\n")
    (data / "opacity_cache" / "CO" / "12C-16O" / "Li2015").mkdir(parents=True)
    (data / "opacity_cache" / "H2-H2_2011.cia").write_text("")
    (data / "opacity_cache" / "H2-He_2011.cia").write_text("")
    for m in ("H2O", "CO2", "CH4", "SO2", "HCN", "C2H2", "H2S"):
        (data / "exojax_linelists" / f"{m}.h5").write_text("")
    V._check_data_tree(tmp_path)
    assert V._ERRORS == []
    assert V._WARNINGS == []


def test_main_fails_loudly_without_checkouts(tmp_path: Path, capsys):
    _reset()
    rc = V.main([str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "must contain the vulcan-retrieval and" in err
    assert "bootstrap_nas_env.pbs" in err


def test_fastchem_probe_rejects_missing_binary(tmp_path: Path):
    assert V._fastchem_runnable(tmp_path) is False
