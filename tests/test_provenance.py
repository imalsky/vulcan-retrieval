"""Run provenance: the target digest binds every class that defines the target
and nothing that churns, and the data identity says why when it cannot resolve.
"""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from retrieval_framework import config_schema as C
from retrieval_framework import provenance


def test_data_identity_never_reports_null_opacity_without_saying_why(
        tmp_path, monkeypatch):
    """The data identity must not silently claim a run read no opacity data.

    This repo never sets $VULCAN_FORWARD_DATA -- it hands the engine its tree
    through paths.set_data_root -- so resolving the trees from the environment
    yields null for a run that read gigabytes of them. Holds in any
    environment: either the root resolves, or the identity says why not.
    """
    monkeypatch.delenv("VULCAN_FORWARD_DATA", raising=False)
    ident = provenance._data_identity(tmp_path, {})
    assert not (ident["data_root_resolved"] is None
                and "engine_data_error" not in ident), (
        "data identity is null with no stated reason -- the environment-only "
        "resolution regressed")
    # the variable's own field reports the variable, never the resolved root
    assert ident["VULCAN_FORWARD_DATA"] is None


# --- target identity: what a checkpoint's numbers belong to --------------------

CHAIN = dict(chem_retry=True, chem_presolve=True)


@pytest.mark.parametrize("base_kw, field, value, moves", [
    ({}, "yconv_cri", 0.001, True),            # convergence tolerance
    ({}, "molecules", ("H2O",), True),         # opacity list
    ({}, "nz", 80, True),                      # chemistry grid
    ({}, "art_ptop_bar", 1e-8, True),          # pressure domain
    ({}, "prior_lnZ", (-1.0, 1.0), True),      # prior definition
    ({}, "seed", 999, True),                   # RNG identity (bit-identical resume)
    ({}, "count_max", 4000, True),             # solver-defined support
    ({}, "walltime_seconds", 3600.0, False),   # per-JOB governor
    ({}, "smc_rt_chunk", 24, False),           # batch split, numerically identical
    ({}, "out_dir", Path("/tmp/elsewhere"), False),
    # a switched key binds only while its switch is on
    ({}, "chem_retry", True, True),
    ({}, "chem_presolve", True, True),
    ({}, "chem_retry_rtol", 0.01, False),
    ({}, "chem_retry_count_max", 9000, False),
    (CHAIN, "chem_retry_rtol", 0.1, True),
    (CHAIN, "chem_retry_count_max", 8000, True),
])
def test_target_digest_moves_with_every_bound_class(base_kw, field, value, moves,
                                                    digest_pipe):
    """Each manifest class must change the digest, or a resume could carry
    numbers from a different density. A chained RESUME job legitimately changes
    the per-job settings; binding them would refuse the documented NAS chaining
    workflow."""
    base = C.Config(molecules=("H2O", "CO2"), **base_kw)
    assert getattr(base, field) != value, f"{field} probe equals the default"
    changed = replace(base, **{field: value})
    assert (provenance.target_digest(base, digest_pipe)
            != provenance.target_digest(changed, digest_pipe)) is moves


def test_switches_off_leave_the_manifest_as_before(digest_pipe):
    """Off, the switched keys stay out of the manifest: it is exactly the
    manifest of a Config that never had them, so every earlier run keeps its
    digest."""
    switched = {k for keys in provenance.SWITCHED_KEYS.values() for k in keys}
    base = C.Config(molecules=("H2O",))
    assert provenance.target_manifest(base, digest_pipe)["config"] == {
        k: v for k, v in asdict(base).items()
        if k not in provenance.TARGET_FREE_KEYS and k not in switched}


@pytest.mark.parametrize("var, value", [
    ("VULCAN3_SOLVER", "ffi"),
    ("VULCAN3_CHEM_RHS", "grouped"),
    ("VULCAN3_BT_FACTOR", "lu"),
    ("VULCAN3_BT_BUFFERS", "1"),
])
def test_target_digest_binds_the_solver_env(var, value, digest_pipe, monkeypatch):
    """Each env var is a numerical choice of the solve, so a resume must not
    switch it."""
    cfg = C.Config(molecules=("H2O",))
    monkeypatch.delenv(var, raising=False)
    d0 = provenance.target_digest(cfg, digest_pipe)
    monkeypatch.setenv(var, value)
    assert provenance.target_digest(cfg, digest_pipe) != d0, f"{var} not bound"


def test_target_digest_moves_with_the_observations(digest_pipe):
    cfg = C.Config(molecules=("H2O",))
    a = provenance.target_digest(cfg, digest_pipe)
    digest_pipe.obs_sigma = digest_pipe.obs_sigma * 2.0
    assert provenance.target_digest(cfg, digest_pipe) != a, "observations not bound"


def _repo(**over):
    st = {"commit": "a" * 40, "dirty": True, "dirty_files": ["a.py"],
          "src_diff": "d" * 64}
    st.update(over)
    return {"r": st}


@pytest.mark.parametrize("state, moves", [
    ({"dirty_files": ["a.py", "b.py"]}, False),   # the LIST churns; not an identity
    ({"dirty": False, "dirty_files": [], "src_diff": None}, True),
    ({"src_diff": "e" * 64}, True),               # a DIFFERENT uncommitted src edit
    ({"commit": "b" * 40}, True),                 # a different commit
])
def test_target_digest_binds_code_state_not_file_churn(state, moves, digest_pipe,
                                                       monkeypatch):
    """A scratch edit must not refuse a resume, but a different source diff must.

    `dirty_files` churns with any untracked file; `src_diff` is the content hash
    of `git diff HEAD -- src`, so it moves only when the code that defines the
    target changed.
    """
    cfg = C.Config(molecules=("H2O",))
    monkeypatch.setattr(provenance, "_repo_states", lambda *a, **k: _repo())
    d0 = provenance.target_digest(cfg, digest_pipe)
    monkeypatch.setattr(provenance, "_repo_states", lambda *a, **k: _repo(**state))
    assert (provenance.target_digest(cfg, digest_pipe) != d0) is moves


def test_target_digest_binds_the_cia_tables(digest_pipe, monkeypatch):
    """The CIA tables are the only target-affecting input outside git, so nothing
    else can notice a swapped continuum."""
    cfg = C.Config(molecules=("H2O",))
    monkeypatch.setattr(provenance, "_cia_identity",
                        lambda *a, **k: {n: {"sha256": "1" * 64}
                                         for n in provenance.CIA_TABLES})
    d0 = provenance.target_digest(cfg, digest_pipe)
    monkeypatch.setattr(provenance, "_cia_identity",
                        lambda *a, **k: {n: {"sha256": "2" * 64}
                                         for n in provenance.CIA_TABLES})
    assert provenance.target_digest(cfg, digest_pipe) != d0


# --- run-directory identity: a refused resume must not rewrite it -------------

def test_untracked_source_moves_the_code_state(tmp_path, monkeypatch):
    """`git diff HEAD` cannot see an untracked module, but importing one changes
    the code that defines the target."""
    import subprocess
    from retrieval_framework.provenance import _src_state
    env = {**os.environ, "HOME": str(tmp_path), "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_CONFIG_NOSYSTEM": "1"}
    repo = tmp_path / "r"
    (repo / "src").mkdir(parents=True)
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t",
                  "commit", "-qm", "x", "--allow-empty"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=env,
                       capture_output=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    clean = _src_state(repo)
    (repo / "src" / "new_module.py").write_text("x = 1\n")
    one = _src_state(repo)
    (repo / "src" / "new_module.py").write_text("x = 2\n")
    two = _src_state(repo)
    assert clean != one != two and clean != two, "untracked src content unbound"
    (repo / "src" / "new_module.py").unlink()
    assert _src_state(repo) == clean
