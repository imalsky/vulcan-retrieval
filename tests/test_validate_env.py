"""validate_env: pure-helper and data-tree checks (no chemistry/RT stack needed).

The interpreter/jax/editable checks are exercised for real by the PBS preflight
and the bootstrap; here we pin the environment-independent logic: version-file
parsing, the data-tree fingerprints, the exogibbs floor and the nautilus pin.
"""
from __future__ import annotations

from pathlib import Path

import pytest

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


# the real list is resolved from the run spec (V.production_molecules); this
# fixture only needs A list, so the check is exercised without a spec
PROD = ("H2O", "CO2", "SO2")


def test_data_tree_checks_flag_missing_and_pass_when_seeded(tmp_path: Path, monkeypatch):
    from vulcan_forward import paths
    # _check_data_tree sets the process-wide engine data root; restore it after
    monkeypatch.setattr(paths, "_root_override", paths._root_override)
    monkeypatch.delenv(paths.ENV_OPACITY_CACHE, raising=False)
    _reset()
    data = tmp_path / "vulcan-retrieval" / "data"
    (data / "opacity_cache").mkdir(parents=True)
    (data / "exomolop").mkdir()
    V._check_data_tree(tmp_path, PROD)
    # ERRORS: missing ExoMolOP k-tables (the only opacity path) and both CIA
    # files. Nothing else is read, so nothing else may warn.
    assert len(V._ERRORS) == 3
    assert any("exomolop" in e or "ExoMolOP" in e for e in V._ERRORS), V._ERRORS
    assert V._WARNINGS == []

    _reset()
    (data / "opacity_cache" / "H2-H2_2011.cia").write_text("")
    (data / "opacity_cache" / "H2-He_2011.cia").write_text("")
    for m in PROD:
        (data / "exomolop" / f"{m}.ktable.h5").write_text("")
    V._check_data_tree(tmp_path, PROD)
    assert V._ERRORS == []
    assert V._WARNINGS == []


def test_exogibbs_floor_rejects_an_old_release(monkeypatch):
    """The equilibrium cold seed needs exogibbs >= EXOGIBBS_MIN; an older one
    imports fine and then returns a seed the solver cannot use, so the preflight
    must fail on the VERSION, not only on the import."""
    _reset()
    import exogibbs
    monkeypatch.setattr(exogibbs, "__version__", "0.5.9", raising=False)
    V._check_exogibbs()
    assert len(V._ERRORS) == 1 and V.EXOGIBBS_MIN in V._ERRORS[0]

    _reset()
    monkeypatch.setattr(exogibbs, "__version__", V.EXOGIBBS_MIN, raising=False)
    V._check_exogibbs()
    assert V._ERRORS == [] and V._WARNINGS == []


def test_nautilus_check_wants_the_stock_release_and_refuses_the_former_fork(monkeypatch):
    """run_nautilus is written for stock nautilus-sampler NAUTILUS_PIN; the former
    fork (SiLU networks, spread_batch) carried the same version string, so the
    check looks for its additions, not only at the version."""
    pytest.importorskip("nautilus")
    import nautilus
    import nautilus.neural
    _reset()
    V._check_nautilus()                       # the installed release (stock in CI)
    assert V._ERRORS == [] and V._WARNINGS == []

    for owner, attr in ((nautilus.Sampler, "add_samples_spread"),
                        (nautilus.neural, "SiLUMLPRegressor")):
        _reset()
        with monkeypatch.context() as m:
            m.setattr(owner, attr, object, raising=False)
            V._check_nautilus()
        assert len(V._ERRORS) == 1 and "former fork" in V._ERRORS[0]

    _reset()
    with monkeypatch.context() as m:
        m.setattr(nautilus, "__version__", "1.0.5")
        V._check_nautilus()
    assert len(V._ERRORS) == 1 and V.NAUTILUS_PIN in V._ERRORS[0]
