"""validate_env: pure-helper and data-tree checks (no chemistry/RT stack needed).

Each check on fake modules and files, with no chemistry or RT stack, and main's order, spec selection and exit status.
The PBS preflight runs the same checks on the real environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple
from types import SimpleNamespace

import pytest

from retrieval_framework import validate_env as V


def _reset():
    V._ERRORS.clear()


def test_repo_version_parses_version_file(tmp_path: Path):
    pkg = tmp_path / "src" / "somepkg"
    pkg.mkdir(parents=True)
    (pkg / "_version.py").write_text('__version__ = "1.2.3"\n')
    assert V._repo_version(tmp_path, "somepkg") == "1.2.3"
    assert V._repo_version(tmp_path, "otherpkg") is None


# the real list is resolved from the run spec (the spec's `molecules`); this
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
    # files. Nothing else may warn.
    assert len(V._ERRORS) == 3
    assert any("exomolop" in e or "ExoMolOP" in e for e in V._ERRORS), V._ERRORS

    _reset()
    (data / "opacity_cache" / "H2-H2_2011.cia").write_text("")
    (data / "opacity_cache" / "H2-He_2011.cia").write_text("")
    for m in PROD:
        (data / "exomolop" / f"{m}.ktable.h5").write_text("")
    V._check_data_tree(tmp_path, PROD)
    assert V._ERRORS == []


def test_exogibbs_floor_rejects_an_old_release(monkeypatch):
    """The equilibrium cold seed needs exogibbs >= EXOGIBBS_MIN; an older one
    imports fine and then returns a seed the solver cannot use, so the preflight
    must fail on the VERSION, not only on the import.
    """
    _reset()
    import exogibbs

    monkeypatch.setattr(exogibbs, "__version__", "0.5.9", raising=False)
    V._check_exogibbs()
    assert len(V._ERRORS) == 1
    assert V.EXOGIBBS_MIN in V._ERRORS[0]

    _reset()
    monkeypatch.setattr(exogibbs, "__version__", V.EXOGIBBS_MIN, raising=False)
    V._check_exogibbs()
    assert V._ERRORS == []


def test_nautilus_check_wants_the_stock_release_and_refuses_the_former_fork(monkeypatch):
    """run_nautilus is written for stock nautilus-sampler NAUTILUS_PIN; the former
    fork (SiLU networks, spread_batch) carried the same version string, so the
    check looks for its additions, not only at the version.
    """
    pytest.importorskip("nautilus")
    import nautilus
    import nautilus.neural

    _reset()
    V._check_nautilus()  # the installed release (stock in CI)
    assert V._ERRORS == []

    for owner, attr in ((nautilus.Sampler, "add_samples_spread"), (nautilus.neural, "SiLUMLPRegressor")):
        _reset()
        with monkeypatch.context() as m:
            m.setattr(owner, attr, object, raising=False)
            V._check_nautilus()
        assert len(V._ERRORS) == 1
        assert "former fork" in V._ERRORS[0]

    _reset()
    with monkeypatch.context() as m:
        m.setattr(nautilus, "__version__", "1.0.5")
        V._check_nautilus()
    assert len(V._ERRORS) == 1
    assert V.NAUTILUS_PIN in V._ERRORS[0]


def test_check_editable_catches_a_shadowing_install_and_stale_metadata(tmp_path: Path, monkeypatch, capsys):
    """A stale non-editable install would run other code than the checkout's, so each form of it is an error."""
    from importlib import metadata

    pkg = tmp_path / "src" / "fakepkg_editable"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "1.0"\n')
    (pkg / "_version.py").write_text('__version__ = "1.0"\n')
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    monkeypatch.delitem(sys.modules, "fakepkg_editable", raising=False)
    monkeypatch.setattr(metadata, "version", lambda _dist: "1.0")

    def errors(repo):
        _reset()
        V._check_editable("fakepkg_editable", "fake-dist", repo, "fakepkg_editable")
        return list(V._ERRORS)

    assert errors(tmp_path) == []
    assert "shadowing" in errors(tmp_path / "another_checkout")[0]
    monkeypatch.setattr(metadata, "version", lambda _dist: "0.9")
    assert "predates" in errors(tmp_path)[0]
    monkeypatch.setattr(metadata, "version", lambda _dist: "1.0")
    (pkg / "_version.py").write_text('__version__ = "2.0"\n')
    assert "!= checkout" in errors(tmp_path)[0]
    _reset()
    V._check_editable("no_such_package_at_all", "fake-dist", tmp_path, "x")
    assert "failed to import" in V._ERRORS[0]
    _reset()
    capsys.readouterr()


def test_cross_repo_pin_refuses_when_it_cannot_read_the_requirements(monkeypatch, capsys):
    """A check that cannot run is a failed check, not a passed one."""
    from importlib import metadata

    def absent(_name):
        raise metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(metadata, "requires", absent)
    _reset()
    V._check_cross_repo_pin()
    assert len(V._ERRORS) == 1
    assert "vulcan-retrieval has no installed metadata" in V._ERRORS[0]
    _reset()
    capsys.readouterr()


def test_cross_repo_pin_refuses_a_stale_sibling(monkeypatch, capsys):
    """The coupling contract: a sibling engine below this distribution's floor is an error, naming both versions."""
    from importlib import metadata

    monkeypatch.setattr(metadata, "requires", lambda _d: ["vulcan3>=0.18.0", "vulcan-forward>=0.30.0"])
    monkeypatch.setattr(metadata, "version", lambda d: {"vulcan3": "0.18.1", "vulcan-forward": "0.29.9"}[d])
    _reset()
    V._check_cross_repo_pin()
    assert len(V._ERRORS) == 1
    assert "vulcan-forward 0.29.9" in V._ERRORS[0]
    assert ">=0.30.0" in V._ERRORS[0]
    _reset()
    capsys.readouterr()


def _fake_jax(backend="cpu", raises=None):
    def default_backend():
        if raises is not None:
            raise raises
        return backend

    return SimpleNamespace(__version__="0.6.2", default_backend=default_backend, devices=lambda: ["d0"])


@pytest.mark.parametrize(
    ("backend", "require_gpu", "n_errors"),
    [("cpu", False, 0), ("cpu", True, 1), ("gpu", True, 0), ("cuda", True, 0)],
)
def test_check_jax_refuses_a_cpu_backend_when_a_gpu_is_required(monkeypatch, backend, require_gpu, n_errors):
    monkeypatch.setitem(sys.modules, "jax", _fake_jax(backend))
    _reset()
    V._check_jax(require_gpu)
    assert len(V._ERRORS) == n_errors
    _reset()


def test_check_jax_reports_a_backend_that_cannot_start(monkeypatch):
    """A broken GPU plugin raises at backend start, after the import has worked; it must join the aggregated report."""
    monkeypatch.setitem(sys.modules, "jax", _fake_jax(raises=RuntimeError("no plugin")))
    _reset()
    V._check_jax(False)
    assert "no plugin" in V._ERRORS[0]
    _reset()


@pytest.mark.parametrize(("version", "n_errors"), [(V.EXOJAX_PIN, 0), ("2.5.0", 1)])
def test_check_exojax_holds_the_pin(monkeypatch, version, n_errors):
    monkeypatch.setitem(sys.modules, "exojax", SimpleNamespace(__version__=version))
    _reset()
    V._check_exojax()
    assert len(V._ERRORS) == n_errors
    _reset()


def test_check_python_refuses_below_the_floor(monkeypatch):
    _reset()
    V._check_python()
    assert V._ERRORS == []
    monkeypatch.setattr(V, "SUPPORTED_PYTHON", (99, 0))
    V._check_python()
    assert "older than the supported" in V._ERRORS[0]
    _reset()


def test_an_unreadable_checkout_version_is_an_error_not_an_ok(tmp_path, monkeypatch, capsys):
    from importlib import metadata

    pkg = tmp_path / "src" / "fakepkg_noversion"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('__version__ = "1.0"\n')
    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    monkeypatch.delitem(sys.modules, "fakepkg_noversion", raising=False)
    monkeypatch.setattr(metadata, "version", lambda _dist: "1.0")
    _reset()
    V._check_editable("fakepkg_noversion", "fake-dist", tmp_path, "fakepkg_noversion")
    assert V._ERRORS[0].startswith("no version readable in")
    assert "_version.py" in V._ERRORS[0]
    assert "ok:" not in capsys.readouterr().out
    _reset()


def _record_checks(monkeypatch, calls):
    """Replace every check with a recorder, and the spec load with a stand-in naming its file."""
    from retrieval_framework import run_setup

    def rec(name):
        return lambda *a: calls.append((name, *a))

    for name in ("python", "jax", "editable", "cross_repo_pin", "engine_features", "exojax", "exogibbs", "nautilus"):
        monkeypatch.setattr(V, f"_check_{name}", rec(name))
    monkeypatch.setattr(V, "_check_spec_inputs", lambda cfg: calls.append(("spec_inputs", cfg.path)))
    monkeypatch.setattr(V, "_check_data_tree", lambda root, prod: calls.append(("data_tree", root, prod)))
    monkeypatch.setattr(
        run_setup, "load_spec", lambda p: SimpleNamespace(cfg=SimpleNamespace(path=p, molecules=("H2O", "CO")))
    )


def test_main_runs_the_checks_in_import_safe_order_on_the_requested_spec(tmp_path, monkeypatch):
    for d in ("vulcan-retrieval", "VULCAN-JAX"):
        (tmp_path / d).mkdir()
    monkeypatch.chdir(tmp_path)
    calls = []
    _record_checks(monkeypatch, calls)
    _reset()
    spec = tmp_path / "other.yaml"
    assert V.main([str(tmp_path), "--require-gpu", "--spec", str(spec)]) == 0
    names = [c[0] for c in calls]
    # vulcan_chem's import-order guard raises if exojax comes before the editable checks
    assert names == [
        "python",
        "jax",
        "editable",
        "editable",
        "editable",
        "cross_repo_pin",
        "engine_features",
        "exojax",
        "spec_inputs",
        "data_tree",
        "exogibbs",
        "nautilus",
    ]
    assert calls[1] == ("jax", True)
    assert calls[2] == ("editable", "vulcan3", "vulcan3", tmp_path / "VULCAN-JAX", "vulcan3")
    assert ("spec_inputs", spec.resolve()) in calls
    assert ("data_tree", tmp_path.resolve(), ("H2O", "CO")) in calls
    # the checks run from the job's working directory
    assert Path.cwd() == (tmp_path / "vulcan-retrieval").resolve()


def test_main_defaults_to_the_production_spec_and_fails_on_any_error(tmp_path, monkeypatch, capsys):
    for d in ("vulcan-retrieval", "VULCAN-JAX"):
        (tmp_path / d).mkdir()
    monkeypatch.chdir(tmp_path)
    calls = []
    _record_checks(monkeypatch, calls)
    monkeypatch.setattr(V, "_check_exojax", lambda: V._err("exojax broken"))
    _reset()
    assert V.main([str(tmp_path)]) == 1
    default = (tmp_path / "vulcan-retrieval" / V.PRODUCTION_SPEC).resolve()
    assert ("spec_inputs", default) in calls
    err = capsys.readouterr().err
    assert "FAIL (1 error(s))" in err
    assert "bootstrap_nas_env.pbs" in err
    _reset()


def test_main_refuses_a_root_without_the_checkouts_before_any_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    _record_checks(monkeypatch, calls)
    _reset()
    assert V.main([str(tmp_path)]) == 1
    assert calls == []
    assert "must contain" in V._ERRORS[0]
    _reset()


def test_an_engine_without_the_dt_min_guard_is_refused(monkeypatch):
    """vulcan3 0.18.0 also names commits without the guard, so the version floor cannot see it."""
    vulcan_chem = pytest.importorskip("vulcan_forward.vulcan_chem")

    _reset()
    V._check_engine_features()
    assert V._ERRORS == []

    class Old(NamedTuple):
        conv_normal: int

    monkeypatch.setattr(vulcan_chem, "ConvDiag", Old)
    V._check_engine_features()
    assert len(V._ERRORS) == 1
    assert "git pull" in V._ERRORS[0]
