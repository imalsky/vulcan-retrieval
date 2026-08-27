#!/usr/bin/env python3
"""Production certificate for a finished retrieval run.

    python -m retrieval_framework.certificate runs/w39b_smc_retrieval

Collects, from a completed run's own outputs, everything needed to decide
whether its numbers may be reported -- and says so with one PASS/FAIL verdict.

WHY. "An old run failed for reasons since fixed" and "the current code produces
a posterior" are different claims, and only the second one licenses reporting
numbers.

WHAT IT GATES. Each check answers "would a reader be misled?":

  * code and data identity: all four repository commits, package versions, and
    the identity of the observation / opacity / CIA / network / config inputs;
  * the fully resolved config, hashed;
  * `reached_beta1` and a final beta of exactly 1 within tolerance -- a
    beta < 1 cloud is TEMPERED, not a posterior;
  * an EXACT target: cold chemistry, or an explicit warm run whose artifacts
    carry `approximate_history_dependent_target` and whose two validators
    (validate_warm, mala_reversibility) both passed;
  * evidence reported with its operational-prior / box-prior semantics and the
    already-implemented support-fraction uncertainties;
  * the run-health diagnostics (bad-gradient, cap, stall, rejection, ESS,
    uniqueness);
  * the three production-fidelity artifacts from `validation/results/`;
  * a cold replay of a small deterministic subset, which catches an
    environment or provenance mistake that every internal check would miss.

This module READS a finished run. It never re-runs the sampler, and it never
writes into the run's own outputs beyond the certificate itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
# Canonical repository labels plus accepted checkout directory names.  The
# public repository is jax-vulcan; the NAS deployment historically clones it
# as VULCAN-JAX.  Provenance must work in either layout without misnaming the
# repository in the certificate.
REPOSITORIES = {
    "jax-vulcan": ("jax-vulcan", "VULCAN-JAX"),
    "vulcan-forward": ("vulcan-forward",),
    "vulcan-retrieval": ("vulcan-retrieval",),
    "vulcan-jwst-tool": ("vulcan-jwst-tool",),
}

# beta must be 1 to within this; the ladder bisects, so it lands on 1 exactly
# or not at all, and a loose tolerance here would let a nearly-tempered cloud
# through as a posterior.
BETA_TOL = 1e-6

# --- run-health floors -------------------------------------------------------
# A diagnostic that is merely PRESENT says nothing, and two of these read 0 in a
# healthy run and 0 in a broken one for opposite reasons. Particle degeneracy is
# the characteristic SMC failure and it is invisible in a corner plot: the cloud
# still has N rows, they are just copies of a handful of states.
UNIQUE_FRAC_FAIL = 0.25        # distinct particles at the final stage, over N
ESS_FRAC_FAIL = 0.20           # smallest per-stage ESS over N
ACCEPT_LO, ACCEPT_HI = 0.05, 0.95
# warmcap + stalled are chemistry-convergence REJECTIONS the MH correction cannot
# see. Some are expected early, while the cloud is still prior-wide; in the late
# ladder they mean the posterior itself sits on the convergence cliff, which makes
# the sampled target solver-defined rather than physical.
LATE_LADDER_FRAC = 1.0 / 3.0
LATE_REJECT_FRAC_FAIL = 0.02   # per-proposal rate over the late stages
# A posterior median on a prior edge is set by the prior, not measured.
PRIOR_RAIL_FRAC = 0.02

# The two production-fidelity artifacts. Their absence is a FAIL for
# a few-ppm or evidence claim: a check that was never run at production settings
# is not a check that passed.
# the resolved-config keys a validation artifact certifies; a run whose values
# differ from the artifact's is refused (validate())
GRID_KEYS = ("art_ptop_bar", "nz", "art_nlayer")

REQUIRED_VALIDATION_ARTIFACTS = (
    "resolution_ladder",
    "top_pressure_ladder",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_GIT_FAILED = object()   # git did not run (or exited non-zero): NOT "clean"


def _git_raw(repo: Path, *args: str):
    """Stdout on a zero exit -- possibly empty -- or `_GIT_FAILED`.

    `git status --porcelain` on a clean tree exits 0 with zero bytes, which is
    indistinguishable from a failed call if both collapse to None. Provenance
    must not record a repo as clean because git was missing.
    """
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, timeout=15)
    except Exception:
        return _GIT_FAILED
    return r.stdout.strip() if r.returncode == 0 else _GIT_FAILED


def _git(repo: Path, *args: str) -> str | None:
    out = _git_raw(repo, *args)
    return out or None if out is not _GIT_FAILED else None


def _repo_states(workspace: Path | None = None) -> dict:
    """Commit + dirty state per repository. `dirty` is None when git could not
    be run, so "unknown" never reads as "clean"."""
    workspace = WORKSPACE if workspace is None else Path(workspace)
    out = {}
    for name, directory_names in REPOSITORIES.items():
        repo = next((workspace / directory for directory in directory_names
                     if _git(workspace / directory, "rev-parse", "HEAD")),
                    None)
        head = _git(repo, "rev-parse", "HEAD") if repo is not None else None
        if head is None:
            out[name] = None
            continue
        dirty = _git_raw(repo, "status", "--porcelain")
        failed = dirty is _GIT_FAILED
        out[name] = {"commit": head,
                     "dirty": None if failed else bool(dirty),
                     "dirty_files": [] if failed else dirty.splitlines()[:20],
                     "checkout": repo.name}
    return out


def _versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("jax", "jaxlib", "numpy", "scipy", "exojax"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    for dist in ("vulcan-jax", "vulcan-forward", "vulcan-retrieval",
                 "vulcan-jwst-tool"):
        try:
            from importlib.metadata import version
            out[dist] = version(dist)
        except Exception:
            out[dist] = None
    return out


def _load_npz(path: Path):
    return np.load(path, allow_pickle=False) if path.is_file() else None


def _scalar(z, key, default=None):
    if z is None or key not in z.files:
        return default
    v = z[key]
    try:
        return v.item()
    except Exception:
        return v


def _data_identity(out_dir: Path, cfg_dict: dict) -> dict:
    """Hash the observation file; fingerprint the big opacity/line-list trees."""
    ident = {}
    obs = out_dir / "observations.npz"
    ident["observations"] = ({"path": str(obs), "sha256": _sha256(obs),
                              "bytes": obs.stat().st_size}
                             if obs.is_file() else None)

    # Resolve through the engine, never from $VULCAN_FORWARD_DATA: this repo
    # hands the engine its tree programmatically and paths.set_data_root takes
    # precedence over the variable, so the environment says nothing about what
    # a run actually read. Both are recorded, under names that say which is
    # which -- a resolved path filed under the variable's name would assert
    # the environment was set when it was not.
    ident["VULCAN_FORWARD_DATA"] = os.environ.get("VULCAN_FORWARD_DATA")
    root = None
    tree_dirs = {}
    try:
        # importing this module is what hands the engine this repo's data tree
        from retrieval_framework.forward import config as _fwd_config  # noqa: F401
        from vulcan_forward import paths as _fwd_paths
        root = Path(_fwd_paths.data_root())
        tree_dirs = {"opacity_cache": Path(_fwd_paths.opacity_cache_dir()),
                     "exomolop": Path(_fwd_paths.exomolop_dir())}
    except Exception as exc:                                # pragma: no cover
        # Never silently null: a check that cannot run says why it could not.
        ident["engine_data_error"] = f"{type(exc).__name__}: {exc}"
    ident["data_root_resolved"] = str(root) if root is not None else None

    for sub in ("opacity_cache", "exomolop"):
        p = tree_dirs.get(sub)
        if p is None or not p.is_dir():
            ident[sub] = None
            continue
        n = tot = 0
        newest = 0.0
        for f in p.rglob("*"):
            if f.is_file():
                st = f.stat()
                n += 1
                tot += st.st_size
                newest = max(newest, st.st_mtime)
        ident[sub] = {"path": str(p.resolve()), "files": n, "bytes": tot,
                      "newest_mtime_utc": time.strftime(
                          "%Y-%m-%dT%H:%M:%SZ", time.gmtime(newest))}

    # The two CIA tables are direct radiative-transfer inputs, small enough to
    # hash exactly and too important to hide inside only a directory count.
    # A swapped H2-He table can leave the tree summary unchanged while changing
    # the continuum fitted by the retrieval.
    opacity_dir = tree_dirs.get("opacity_cache")
    for name in ("H2-H2_2011.cia", "H2-He_2011.cia"):
        p = opacity_dir / name if opacity_dir is not None else None
        ident[name] = (
            {"path": str(p.resolve()), "sha256": _sha256(p),
             "bytes": p.stat().st_size}
            if p is not None and p.is_file() else None)

    # the reaction network is a vendored VULCAN-JAX input; hash the exact file
    net = cfg_dict.get("vulcan_cfg_name")
    ident["vulcan_cfg_name"] = net
    try:
        # find_spec, NOT import: importing vulcan_jax parses and import-locks
        # its reaction network, and this identity collector only needs the
        # package path. The bare import here locked the process to the default
        # network and broke every later SNCHO chem build in the same session.
        import importlib.util
        spec = importlib.util.find_spec("vulcan_jax")
        if spec is None or not spec.origin:
            raise ModuleNotFoundError("vulcan_jax is not installed")
        pkg = Path(spec.origin).resolve().parent
        cfg_yaml = pkg / "configs" / f"{net}.yaml"
        if cfg_yaml.is_file():
            ident["vulcan_config_yaml"] = {"path": str(cfg_yaml),
                                           "sha256": _sha256(cfg_yaml)}
            import yaml
            netfile = yaml.safe_load(cfg_yaml.read_text()).get("network")
            npath = pkg / netfile if netfile else None
            if npath and npath.is_file():
                ident["network_file"] = {"path": str(npath),
                                         "sha256": _sha256(npath)}
    except Exception as exc:                                # pragma: no cover
        ident["network_file_error"] = str(exc)
    return ident


def _validation_artifacts() -> dict:
    """Read the three production-fidelity results, if they were archived."""
    results = REPO / "validation" / "results"
    out = {}
    for name in REQUIRED_VALIDATION_ARTIFACTS:
        p = results / f"{name}.json"
        if not p.is_file():
            out[name] = None
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as exc:                            # pragma: no cover
            out[name] = {"error": str(exc)}
            continue
        # the grid the artifact was measured on (top_pressure_ladder nests the
        # production profile under "production"); validate() refuses a run
        # whose grid differs, so a changed constant cannot ride on a stale PASS
        rc = d.get("provenance", {}).get("resolved_config") or {}
        rc = rc.get("production", rc)
        out[name] = {
            "status": d.get("verdict", {}).get("status"),
            "summary": d.get("verdict", {}).get("summary"),
            "generated_utc": d.get("provenance", {}).get("generated_utc"),
            "sha256": _sha256(p),
            "grid": {k: rc.get(k) for k in GRID_KEYS},
        }
    return out


def _mala_reversibility_artifact(out_dir: Path) -> dict | None:
    """Read the warm-kernel artifact and bind it to the current checkpoint."""
    path = out_dir / "mala_reversibility.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:                                # pragma: no cover
        return {"status": "ERROR", "error": str(exc), "sha256": _sha256(path)}
    checkpoint = out_dir / "smc_checkpoint.npz"
    current_sha = _sha256(checkpoint) if checkpoint.is_file() else None
    recorded_sha = (payload.get("checkpoint") or {}).get("sha256")
    return {
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "pairs_requested": payload.get("pairs_requested"),
        "pairs_tested": payload.get("pairs_tested"),
        "asymmetric_pairs": payload.get("asymmetric_pairs"),
        "checkpoint_sha256": recorded_sha,
        "checkpoint_matches": bool(current_sha and recorded_sha == current_sha),
        "sha256": _sha256(path),
    }


def _posterior_summary(samples, cfg_dict: dict) -> list | None:
    """Per-parameter median and where it sits inside its own prior box.

    ``prior_position`` is the median's fractional position in the prior, measured
    in the space the prior is FLAT in (log10 for a log10_uniform parameter), so 0
    and 1 are the declared edges. ``q05_position`` / ``q95_position`` are the
    same measure for the 5th and 95th percentiles: a median-only test is blind
    to a bimodal marginal with a mode pinned to an edge.
    """
    if samples is None or "samples" not in samples.files:
        return None
    names = list(cfg_dict.get("inferred_param_names") or [])
    lo = np.asarray(cfg_dict.get("inferred_param_prior_lo") or [], float)
    hi = np.asarray(cfg_dict.get("inferred_param_prior_hi") or [], float)
    types = list(cfg_dict.get("inferred_param_prior_types") or [])
    th = np.asarray(samples["samples"], float).reshape(-1, len(names) or 1)
    if not names or lo.size != th.shape[1] or hi.size != th.shape[1]:
        return None
    out = []
    for i, name in enumerate(names):
        q05, q50, q95 = (float(v) for v in np.percentile(th[:, i], [5, 50, 95]))
        is_log = i < len(types) and types[i] == "log10_uniform" and lo[i] > 0

        def _pos(x, i=i, is_log=is_log):
            a, b = lo[i], hi[i]
            if is_log and x > 0:
                a, b, x = math.log10(a), math.log10(b), math.log10(x)
            return float((x - a) / (b - a)) if b > a else None

        out.append({"name": name, "median": q50, "q05": q05, "q95": q95,
                    "prior_lo": float(lo[i]), "prior_hi": float(hi[i]),
                    "prior_position": _pos(q50),
                    "q05_position": _pos(q05), "q95_position": _pos(q95)})
    return out


def collect(out_dir: Path) -> dict:
    """Assemble the certificate payload from a finished run's outputs."""
    out_dir = Path(out_dir).resolve()
    cfg_path = out_dir / "config.json"
    cfg_dict = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    cfg_blob = json.dumps(cfg_dict, sort_keys=True, default=str)

    # These are the filenames written by run_smc.py.
    samples = _load_npz(out_dir / "posterior_samples.npz")
    extra = _load_npz(out_dir / "smc_extra_fields.npz")
    ckpt = _load_npz(out_dir / "smc_checkpoint.npz")
    vwarm = _load_npz(out_dir / "validate_warm.npz")

    chem_mode = str(cfg_dict.get("smc_chem_mode", "")).strip().lower() or None
    final_beta = _scalar(samples, "final_beta")
    reached = _scalar(samples, "reached_beta1")

    def _arr(z, key):
        if z is None or key not in z.files:
            return None
        return np.asarray(z[key]).tolist()

    return {
        "certificate_version": 2,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": str(out_dir),
        "code": {"repos": _repo_states(), "versions": _versions()},
        "data": _data_identity(out_dir, cfg_dict),
        "resolved_config": cfg_dict,
        "resolved_config_sha256": hashlib.sha256(cfg_blob.encode()).hexdigest(),
        "target": {
            "smc_chem_mode": chem_mode,
            "approximate_history_dependent_target": bool(
                _scalar(samples, "approximate_history_dependent_target", 0)),
            "warm_extrapolate": cfg_dict.get("warm_extrapolate"),
        },
        "convergence": {
            "reached_beta1": (None if reached is None else bool(reached)),
            "final_beta": (None if final_beta is None else float(final_beta)),
            "n_stages": (len(_arr(extra, "smc_betas") or []) - 1
                         if extra is not None else None),
        },
        "evidence": {
            # semantics preserved verbatim: smc_logZ is conditional on the
            # OPERATIONAL prior support; logZ_box is the zero-filled declared-box
            # quantity. They are not interchangeable and must not be differenced
            # across models with different support fractions.
            "smc_logZ": _scalar(extra, "smc_logZ"),
            "smc_logZ_box": _scalar(extra, "smc_logZ_box"),
            "log_support_fraction": _scalar(extra, "smc_log_support_fraction"),
            "log_support_fraction_err": _scalar(
                extra, "smc_log_support_fraction_err"),
            "log_support_physical": _scalar(extra, "smc_log_support_physical"),
            "log_support_physical_err": _scalar(
                extra, "smc_log_support_physical_err"),
            "log_conv_attrition": _scalar(extra, "smc_log_conv_attrition"),
            "log_conv_attrition_err": _scalar(
                extra, "smc_log_conv_attrition_err"),
        },
        "diagnostics": {
            "ess": _arr(extra, "smc_ess"),
            "acceptance_rate": _arr(extra, "smc_acceptance_rate"),
            "unique_particles": _arr(extra, "smc_unique_particles"),
            "warm_capped": _arr(extra, "smc_warm_capped"),
            "warm_stalled": _arr(extra, "smc_warm_stalled"),
            "badgrad": _arr(extra, "smc_tangent_rejected"),
            "step_size_history": _arr(extra, "smc_step_size_history"),
            "n_particles": _scalar(extra, "smc_num_particles"),
            "n_mcmc_steps": _scalar(extra, "smc_num_mcmc_steps"),
        },
        "warm_validation": (None if vwarm is None else {
            "dlogl_max": float(np.nanmax(np.abs(np.asarray(vwarm["dlogl"]))))
                         if "dlogl" in vwarm.files else None,
            "spectrum_dppm_max": _scalar(vwarm, "spectrum_dppm_max"),
            "atom_ratio_rel_max": _scalar(vwarm, "atom_ratio_rel_max"),
            "grad_rel_max_gated": _scalar(vwarm, "grad_rel_max_gated"),
            "grad_zeroed_frac": _scalar(vwarm, "grad_zeroed_frac"),
        }),
        "posterior": _posterior_summary(samples, cfg_dict),
        "mala_reversibility": _mala_reversibility_artifact(out_dir),
        "validation_artifacts": _validation_artifacts(),
        "checkpoint_present": ckpt is not None,
    }


def validate(cert: dict, replay: dict | None = None) -> list[str]:
    """Return gate failures; empty means the run may be reported."""
    problems = []

    # --- identity ----------------------------------------------------------
    for name, state in cert["code"]["repos"].items():
        if state is None:
            problems.append(f"{name}: commit not recorded (not a git checkout?)")
        elif state["dirty"]:
            problems.append(
                f"{name}: working tree DIRTY at {state['commit'][:12]} -- the "
                "run cannot be attributed to a committed state")
    if not cert["resolved_config"]:
        problems.append("no config.json: the resolved configuration is unknown")
    if cert["data"].get("observations") is None:
        problems.append("no observations.npz: the fitted data is unidentified")

    # --- the cloud is a posterior, not a tempered intermediate --------------
    conv = cert["convergence"]
    if not conv["reached_beta1"]:
        problems.append(
            f"reached_beta1 is {conv['reached_beta1']!r} (final beta "
            f"{conv['final_beta']!r}): the cloud is TEMPERED, not a posterior")
    elif conv["final_beta"] is None or abs(conv["final_beta"] - 1.0) > BETA_TOL:
        problems.append(
            f"final beta {conv['final_beta']!r} is not 1 within {BETA_TOL}")

    # --- the target is exact, or explicitly approximate AND validated -------
    tgt = cert["target"]
    if tgt["smc_chem_mode"] == "warm":
        if not tgt["approximate_history_dependent_target"]:
            problems.append(
                "warm run whose artifacts are NOT stamped "
                "approximate_history_dependent_target")
        wv = cert["warm_validation"]
        if wv is None:
            problems.append(
                "warm run with no validate_warm.npz: the history-dependent "
                "bias is UNMEASURED, which is not the same as small")
        else:
            from retrieval_framework.validate_warm import (
                DLOGL_MAX_PASS, GRAD_REL_FAIL, GRAD_ZEROED_FRAC_FAIL,
                SPEC_PPM_MAX_PASS,
            )
            checks = [
                ("dlogl_max", DLOGL_MAX_PASS, "warm-vs-cold likelihood bias"),
                ("spectrum_dppm_max", SPEC_PPM_MAX_PASS, "spectrum difference"),
                ("grad_rel_max_gated", GRAD_REL_FAIL, "MALA drift agreement"),
                ("grad_zeroed_frac", GRAD_ZEROED_FRAC_FAIL, "zeroed-drift fraction"),
            ]
            for key, limit, label in checks:
                v = wv.get(key)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    problems.append(f"warm run: {label} ({key}) not measured")
                elif float(v) >= limit:
                    problems.append(
                        f"warm run: {label} ({key}) {float(v):.3e} >= {limit}")
        mala = cert.get("mala_reversibility")
        if mala is None:
            problems.append(
                "warm run without mala_reversibility.json: the mandatory "
                "warm-kernel reversibility probe was not archived")
        elif mala.get("status") != "PASS":
            problems.append(
                "warm run whose mala_reversibility probe did not PASS: "
                f"{mala.get('summary') or mala.get('error') or 'no detail'}")
        elif not mala.get("checkpoint_matches"):
            problems.append(
                "warm run whose mala_reversibility artifact does not match "
                "the current smc_checkpoint.npz")
    elif tgt["smc_chem_mode"] != "cold":
        problems.append(
            f"smc_chem_mode is {tgt['smc_chem_mode']!r}; expected 'cold' "
            "(the default) or an explicit, validated 'warm'")

    # --- evidence semantics -------------------------------------------------
    ev = cert["evidence"]
    for key in ("smc_logZ", "smc_logZ_box", "log_support_fraction",
                "log_support_fraction_err"):
        if ev.get(key) is None:
            problems.append(f"no {key} recorded")

    # --- run health ---------------------------------------------------------
    diag = cert["diagnostics"]
    for key in ("ess", "acceptance_rate", "unique_particles", "warm_capped",
                "warm_stalled", "badgrad"):
        if not diag.get(key):
            problems.append(f"diagnostic '{key}' missing from the run outputs")
    problems += health_problems(diag)
    problems += rail_problems(cert.get("posterior"))

    # --- production-fidelity artifacts --------------------------------------
    required = set(REQUIRED_VALIDATION_ARTIFACTS)
    opa = str(cert["resolved_config"].get("opacity_mode", "")) or None
    if opa is None:
        problems.append(
            "resolved config records no opacity_mode: which opacity the run "
            "used is unknown")
    elif opa != "exomolop":
        problems.append(
            f"resolved config records opacity_mode={opa!r}: the sampled "
            "line-by-line path was removed with vulcan-forward 0.11.0 and is "
            "measurably biased on this band; only correlated-k ('exomolop') "
            "runs are certifiable")
    for name, art in cert["validation_artifacts"].items():
        if name not in required:
            continue
        if art is None:
            problems.append(
                f"validation artifact '{name}' is missing: the production "
                "choice it measures has not been checked at production "
                "settings, so no few-ppm or converged-evidence claim is "
                "supported")
        elif art.get("status") == "FAIL":
            problems.append(f"validation artifact '{name}' FAILED: "
                            f"{art.get('summary', '')[:160]}")
        elif art.get("status") == "REPORT":
            problems.append(
                f"validation artifact '{name}' is REPORT, not PASS -- its "
                "decisive test was not run (see its summary)")
        elif art.get("status") != "PASS":
            problems.append(
                f"validation artifact '{name}' has status "
                f"{art.get('status')!r}, expected PASS: "
                f"{art.get('summary', '')[:160]}")
        else:
            run = {k: cert["resolved_config"].get(k) for k in GRID_KEYS}
            got = art.get("grid") or {}
            same = all(
                got.get(k) is not None and run[k] is not None
                and math.isclose(float(got[k]), float(run[k]), rel_tol=1e-9)
                for k in GRID_KEYS)
            if not same:
                problems.append(
                    f"validation artifact '{name}' was measured on a different "
                    f"grid (artifact {got} vs run {run}): re-run it at the "
                    "production settings")

    # --- cold replay --------------------------------------------------------
    if replay is not None and replay.get("ran"):
        if not replay.get("passed"):
            problems.append(
                f"cold replay MISMATCH: {replay.get('detail', '')}")
    else:
        problems.append(
            "cold replay not run: a deterministic re-solve of a small subset "
            "is what catches an environment or provenance mistake that every "
            "internal consistency check would pass")

    return problems


def health_problems(diag: dict) -> list[str]:
    """Value gates on the run-health diagnostics (module-level so they are
    testable without a run)."""
    out: list[str] = []
    n = int(diag.get("n_particles") or 0)
    uniq = list(diag.get("unique_particles") or [])
    ess = list(diag.get("ess") or [])
    acc = [float(a) for a in (diag.get("acceptance_rate") or [])
           if a is not None and math.isfinite(float(a))]
    if n and uniq:
        frac = float(uniq[-1]) / n
        if frac < UNIQUE_FRAC_FAIL:
            out.append(
                f"particle degeneracy: {int(uniq[-1])}/{n} distinct particles at "
                f"the final stage ({frac:.0%} < {UNIQUE_FRAC_FAIL:.0%}). The "
                "cloud still has N rows, so every marginal looks smooth while "
                "resting on a handful of distinct states")
    if n and ess:
        i = int(np.argmin(ess))
        frac = float(ess[i]) / n
        if frac < ESS_FRAC_FAIL:
            out.append(
                f"ESS collapsed to {float(ess[i]):.1f}/{n} ({frac:.0%} < "
                f"{ESS_FRAC_FAIL:.0%}) at stage {i}: the ladder took a "
                "temperature step the cloud could not absorb")
    if acc:
        k = max(1, int(round(LATE_LADDER_FRAC * len(acc))))
        a = float(np.mean(acc[-k:]))
        if not (ACCEPT_LO < a < ACCEPT_HI):
            out.append(
                f"late-ladder MALA acceptance {a:.2f} outside "
                f"({ACCEPT_LO}, {ACCEPT_HI}): the mutation kernel is not "
                "moving particles at the target rate, so the cloud is not "
                "mixing at beta=1")
    cap = list(diag.get("warm_capped") or [])
    stall = list(diag.get("warm_stalled") or [])
    sweeps = int(diag.get("n_mcmc_steps") or 0)
    if n and sweeps and cap and stall and len(cap) == len(stall):
        k = max(1, int(round(LATE_LADDER_FRAC * len(cap))))
        late = sum(cap[-k:]) + sum(stall[-k:])
        rate = late / float(n * sweeps * k)
        if rate > LATE_REJECT_FRAC_FAIL:
            out.append(
                f"late-ladder chemistry-convergence rejections {rate:.1%} of "
                f"proposals (> {LATE_REJECT_FRAC_FAIL:.0%}; {late} over the last "
                f"{k} stage(s)): the posterior sits on the convergence cliff, so "
                "the sampled target is defined by count_max rather than by the "
                "physics")
    return out


def rail_problems(post: list | None) -> list[str]:
    """A posterior quantile on a prior edge is prior-determined, not measured.

    Both tails are tested, not only the median: a bimodal marginal can put a
    mode hard on an edge while the median sits comfortably mid-box.
    """
    if not post:
        return ["no posterior summary: the parameter medians could not be read "
                "from posterior_samples.npz, so a prior-railed parameter would "
                "go unnoticed"]
    out = []
    for row in post:
        for label, key in (("median", "prior_position"),
                           ("5th percentile", "q05_position"),
                           ("95th percentile", "q95_position")):
            f = row.get(key)
            if f is None:
                continue
            if f < PRIOR_RAIL_FRAC or f > 1.0 - PRIOR_RAIL_FRAC:
                out.append(
                    f"{row['name']}: posterior {label} sits at {f:.1%} of its "
                    f"prior range [{row['prior_lo']:g}, {row['prior_hi']:g}] -- "
                    "the value is set by the prior edge, not by the data. Widen "
                    "the prior, or report the parameter as a limit rather than "
                    "a measurement")
    return out


def render(cert: dict, problems: list[str]) -> str:
    ok = not problems
    L = [
        "# Retrieval production certificate",
        "",
        (f"**VERDICT: {'PASS' if ok else 'FAIL'}**"
         + ("" if ok else f" -- {len(problems)} problem(s)")),
        "",
        f"Run: `{cert['out_dir']}`  ",
        f"Generated {cert['generated_utc']}",
        "",
    ]
    if problems:
        L += ["## Why this run may not be reported", ""]
        L += [f"- {p}" for p in problems]
        L += [""]
    conv, tgt, ev = cert["convergence"], cert["target"], cert["evidence"]
    L += [
        "## Result", "",
        "| field | value |", "|---|---|",
        f"| chem mode (target) | {tgt['smc_chem_mode']} |",
        f"| approximate history-dependent target | "
        f"{tgt['approximate_history_dependent_target']} |",
        f"| reached beta=1 | {conv['reached_beta1']} |",
        f"| final beta | {conv['final_beta']} |",
        f"| stages | {conv['n_stages']} |",
        f"| smc_logZ (operational prior) | {ev['smc_logZ']} |",
        f"| smc_logZ_box (zero-filled box) | {ev['smc_logZ_box']} |",
        f"| ln f_support +- err | {ev['log_support_fraction']} +- "
        f"{ev['log_support_fraction_err']} |",
        "",
        "`smc_logZ` is conditional on the OPERATIONAL prior support "
        "(box AND T-P window AND converged, renormalized). `smc_logZ_box` is "
        "the zero-filled declared-box quantity. They are different numbers; "
        "never difference either across models with different support "
        "fractions.",
        "",
        "## Code and data", "", "| key | value |", "|---|---|",
    ]
    for name, st in cert["code"]["repos"].items():
        L.append(f"| {name} | "
                 + ("not a git checkout" if st is None else
                    st["commit"][:12] + (" (DIRTY)" if st["dirty"] else ""))
                 + " |")
    for k in ("jax", "numpy", "exojax", "python"):
        L.append(f"| {k} | {cert['code']['versions'].get(k)} |")
    obs = cert["data"].get("observations")
    L.append(f"| observations sha256 | "
             f"{obs['sha256'][:16] + '...' if obs else 'MISSING'} |")
    net = cert["data"].get("network_file")
    L.append(f"| network sha256 | "
             f"{net['sha256'][:16] + '...' if net else 'not resolved'} |")
    L.append(f"| resolved config sha256 | "
             f"{cert['resolved_config_sha256'][:16]}... |")
    diag = cert["diagnostics"]
    n = diag.get("n_particles")
    uniq = diag.get("unique_particles") or []
    ess = diag.get("ess") or []
    acc = diag.get("acceptance_rate") or []
    L += ["", "## Run health", "", "| field | value |", "|---|---|",
          f"| particles | {n} |",
          f"| distinct particles, final stage | {uniq[-1] if uniq else None} |",
          f"| min ESS over the ladder | "
          f"{min(ess):.1f} |" if ess else "| min ESS over the ladder | None |",
          f"| final acceptance | {acc[-1]:.2f} |" if acc
          else "| final acceptance | None |",
          f"| late warmcap + stalled | "
          f"{sum((diag.get('warm_capped') or [])[-3:]) + sum((diag.get('warm_stalled') or [])[-3:])} |",
          f"| badgrad total | {sum(diag.get('badgrad') or [])} |",
          "", "## Production-fidelity artifacts", "",
          "| artifact | status |", "|---|---|"]
    for name, art in cert["validation_artifacts"].items():
        L.append(f"| {name} | {'MISSING' if art is None else art['status']} |")
    L += ["", "Machine-readable: `certificate.json`", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=".",
                    help="retrieval case directory containing case.py")
    ap.add_argument("--replay-n", type=int, default=4,
                    help="particles to cold-replay (0 disables; the replay is "
                         "what catches an environment/provenance mistake)")
    args = ap.parse_args(argv)

    from retrieval_framework.run_smc import make_config

    cfg, _preset = make_config(Path(args.run_dir))
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"certificate: no output directory at {out_dir}")

    cert = collect(out_dir)

    replay = {"ran": False, "reason": "disabled (--replay-n 0)"}
    if args.replay_n > 0:
        replay = cold_replay(Path(args.run_dir), cfg, out_dir, args.replay_n)
    cert["cold_replay"] = replay

    problems = validate(cert, replay)
    cert["verdict"] = {"passed": not problems, "problems": problems}

    (out_dir / "certificate.json").write_text(
        json.dumps(cert, indent=2, default=str) + "\n")
    (out_dir / "certificate.md").write_text(render(cert, problems))
    print(f"wrote {out_dir / 'certificate.json'}")
    print(f"wrote {out_dir / 'certificate.md'}")

    if problems:
        print(f"\nCERTIFICATE: FAIL ({len(problems)} problem(s))",
              file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nThis run's numbers are NOT certified for reporting.",
              file=sys.stderr)
        return 1
    print("\nCERTIFICATE: PASS")
    return 0


def cold_replay(run_dir: Path, cfg, out_dir: Path, n: int) -> dict:
    """Re-evaluate a few checkpointed particles COLD and compare likelihoods.

    Run even when the production run was already cold: the point is not to
    re-test the sampler but to prove that THIS environment, with THIS data and
    THIS config, reproduces the recorded numbers. An environment or provenance
    mistake (a swapped line list, a stale editable install pointing at another
    checkout, a different network file) passes every internal consistency check
    and fails here.
    """
    ck = out_dir / "smc_checkpoint.npz"
    if not ck.is_file():
        return {"ran": False, "reason": f"no checkpoint at {ck}"}
    obs_path = out_dir / "observations.npz"
    if not obs_path.is_file():
        return {"ran": False, "reason": f"no observations at {obs_path}"}
    try:
        import jax
        import jax.numpy as jnp

        from retrieval_framework import pipeline as P
    except Exception as exc:
        return {"ran": False, "reason": f"import failed: {exc}"}

    try:
        z = np.load(ck, allow_pickle=False)
        U = np.asarray(z["u_particles"], np.float64)
        L_rec = np.asarray(z["loglik"], np.float64)
        # Deterministic subset, chosen by recorded likelihood rank so the choice
        # does not depend on RNG state or on file ordering. Take the healthiest
        # particles: a posterior-edge particle sitting on the convergence cliff
        # would make this test about count_max, not about the environment.
        finite = np.isfinite(L_rec) & (L_rec > -1.0e29)
        if not finite.any():
            return {"ran": False, "reason": "no finite likelihoods in checkpoint"}
        idx = np.argsort(-np.where(finite, L_rec, -np.inf))[:max(1, int(n))]
        idx = np.sort(idx)

        pipe = P.build_pipeline(cfg)
        o = np.load(obs_path, allow_pickle=False)
        pipe.set_observations(o["depth"], o["sigma"])   # the run's OWN obs

        Y0, refs0 = P._blank_state(pipe, len(idx))
        cold_l = jax.jit(pipe.batch_eval_cold_l)
        L_new, _Y, _refs = cold_l(jnp.asarray(U[idx]), Y0, refs0)
        L_new = np.asarray(jax.device_get(L_new), np.float64)

        d = np.abs(L_new - L_rec[idx])
        worst = float(np.nanmax(d)) if d.size else float("nan")
        from retrieval_framework.validate_warm import DLOGL_MAX_PASS
        passed = bool(np.isfinite(worst) and worst < DLOGL_MAX_PASS)
        return {
            "ran": True, "n": int(len(idx)),
            "particle_indices": [int(i) for i in idx],
            "loglik_recorded": L_rec[idx].tolist(),
            "loglik_replayed": L_new.tolist(),
            "max_abs_dlogl": worst, "gate": DLOGL_MAX_PASS,
            "passed": passed,
            "detail": ("" if passed else
                       f"max |dlogL| {worst:.3e} vs gate {DLOGL_MAX_PASS} -- "
                       "this environment does not reproduce the recorded "
                       "likelihoods, so the run's provenance is in doubt "
                       "(swapped line list? stale editable install? different "
                       "network file?)"),
        }
    except Exception as exc:
        return {"ran": False, "reason": f"replay error: {exc!r}"}


if __name__ == "__main__":
    sys.exit(main())
