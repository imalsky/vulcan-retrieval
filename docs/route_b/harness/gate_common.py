"""Shared machinery for the B0C gate runners (G1-G5).

Provenance capture, artifact I/O, solve/termination/ledger/residual
reporting. Every gate artifact must carry: repo SHAs (both trees), network
sha256, lookup-table sha256, package versions, platform, float64 statement,
the full effective configuration, and MEASURED numbers -- never adjectives
(plan work rules; directive M).
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HARNESS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = HARNESS_DIR / "results"

# Single-profile runs never set state.termination_reason (batched-runner
# field); derive it from the carry exactly like the B0-6 probe did.
TERMINATION_ORDER = ("non-finite", "step-count-exhausted", "runtime-cap",
                     "converged-gate")


def _git_sha(repo: Path) -> str:
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git rev-parse failed for {repo}: {out.stderr}")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + ("+dirty" if dirty else "")


def provenance(extra: dict | None = None) -> dict:
    """Repo/table/network/version provenance for a gate artifact."""
    import jax
    import vulcan_jax
    from retrieval_framework.forward import config, h2s_boundary as hb
    from vulcan_jax._paths import resolve_data_path

    repo_retrieval = Path(config.REPO_DIR)
    repo_jax = Path(vulcan_jax.__file__).resolve().parents[2]
    net_path = resolve_data_path(config.VULCAN_NETWORK)
    table = hb.load_h2s_boundary_table(config.H2S_BOUNDARY_TABLE)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("jax_enable_x64 is False; gate runs are f64-only")
    prov = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vulcan_retrieval_sha": _git_sha(repo_retrieval),
        "vulcan_jax_sha": _git_sha(repo_jax),
        "network": str(config.VULCAN_NETWORK),
        "network_sha256": hashlib.sha256(net_path.read_bytes()).hexdigest(),
        "h2s_table_sha256": table.sha256,
        "float64": True,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {m: __import__(m).__version__
                     for m in ("jax", "jaxlib", "numpy", "exojax")},
    }
    if extra:
        prov.update(extra)
    return prov


def save_artifact(name: str, payload: dict, arrays: dict | None = None) -> Path:
    """Write <name>_<stamp>.json (+ optional .npz of arrays) under results/.

    Returns the JSON path. Arrays (converged states, ledgers) go to the npz
    so downstream gates (G3 reads G1's state) never re-solve.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    jpath = RESULTS_DIR / f"{name}_{stamp}.json"
    if arrays:
        npath = RESULTS_DIR / f"{name}_{stamp}.npz"
        np.savez_compressed(npath, **{k: np.asarray(v)
                                      for k, v in arrays.items()})
        payload = dict(payload, arrays_npz=npath.name)
    jpath.write_text(json.dumps(payload, indent=1, default=_jsonify))
    print(f"[gate] artifact written: {jpath}")
    return jpath


def _jsonify(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON-serializable: {type(o)}")


def termination_of(final, count_max: int, runtime: float) -> str:
    """Derive the single-profile termination reason from the final carry."""
    y = np.asarray(final.y)
    if not np.all(np.isfinite(y)):
        return "non-finite"
    if int(final.accept_count) >= int(count_max):
        return "step-count-exhausted"
    if float(final.t) >= float(runtime):
        return "runtime-cap"
    return "converged-gate"


def solve_report(chem, final, count_max: int, runtime: float) -> dict:
    """Measured solve summary: termination, steps, time, longdy, dt."""
    return {
        "termination": termination_of(final, count_max, runtime),
        "accept_count": int(final.accept_count),
        "count_max": int(count_max),
        "t_end_s": float(final.t),
        "runtime_cap_s": float(runtime),
        "dt_end_s": float(final.dt),
        "longdy": float(final.longdy),
        "longdy_seen_min": float(getattr(final, "longdy_seen_min", np.nan)),
    }


def ledger_report(chem, final) -> dict:
    """Per-operator elemental ledger at the last accepted step (D6/G3 inputs).

    Entries are column-inventory deltas [atoms cm^-2] per element in the
    runner's atom order; led_rain is the instantaneous elemental rainout
    RATE [atoms cm^-2 s^-1]; telescoping residual = |led_step + led_renorm
    + led_bc| relative to the column inventory (must sit at float64 noise).
    """
    atoms = list(chem._integ._atom_order)
    led = {k: np.asarray(getattr(final, k), dtype=np.float64)
           for k in ("led_step", "led_renorm", "led_bc", "led_rain")}
    dt = float(final.led_dt)
    y = np.asarray(final.y, dtype=np.float64)
    dz = np.asarray(chem.dz, dtype=np.float64)
    compo = np.asarray(chem._integ._compo_arr, dtype=np.float64)
    inventory = (y * dz[:, None]).sum(axis=0) @ compo  # atoms cm^-2 per element
    tele = led["led_step"] + led["led_renorm"] + led["led_bc"]
    return {
        "atom_order": atoms,
        "led_dt_s": dt,
        **{k: dict(zip(atoms, v.tolist())) for k, v in led.items()},
        "column_inventory": dict(zip(atoms, inventory.tolist())),
        "telescoping_rel": dict(zip(
            atoms, (np.abs(tele) / np.maximum(np.abs(inventory), 1e-300)).tolist())),
    }


def species_of(chem) -> list:
    """Species names indexed by column (sidx is name -> column)."""
    names = [""] * chem.ni
    for name, idx in chem.sidx.items():
        names[int(idx)] = name
    return names


def residual_report(chem, final, atm_T, top_n: int = 10) -> dict:
    """Direct-residual report at the finished state (G1's second half)."""
    from vulcan_jax.steady_residual import residual_from_state

    rep = residual_from_state(chem._integ, final, atm_T)
    R = np.asarray(rep.R, dtype=np.float64)
    mask = np.asarray(rep.mask)
    live = ~mask
    absR = np.where(live, np.abs(R), 0.0)
    flat = np.argsort(absR, axis=None)[::-1][:top_n]
    species = species_of(chem)
    worst = [{"z": int(z), "species": species[i], "R_s^-1": float(R[z, i])}
             for z, i in (np.unravel_index(f, R.shape) for f in flat)]
    return {
        "max_R_s^-1": float(rep.max_R),
        "argmax_z": int(rep.argmax_z),
        "argmax_species": species[int(rep.argmax_i)],
        "n_live_cells": int(live.sum()),
        "n_excluded_cells": int(mask.sum()),
        "worst_cells": worst,
    }


def species_delta_report(chem, y_a, y_b, floor_vmr: float = 1e-15) -> dict:
    """Max |delta ln n| between two converged states over species above the
    VMR floor in EITHER state (G2/G4/G5 agreement metric)."""
    y_a = np.asarray(y_a, dtype=np.float64)
    y_b = np.asarray(y_b, dtype=np.float64)
    mix_a = y_a / y_a.sum(axis=1, keepdims=True)
    mix_b = y_b / y_b.sum(axis=1, keepdims=True)
    live = (mix_a > floor_vmr) | (mix_b > floor_vmr)
    with np.errstate(divide="ignore", invalid="ignore"):
        dln = np.abs(np.log(y_a) - np.log(y_b))
    dln = np.where(live & (y_a > 0) & (y_b > 0), dln, 0.0)
    z, i = np.unravel_index(np.argmax(dln), dln.shape)
    species = species_of(chem)
    return {
        "floor_vmr": floor_vmr,
        "max_abs_dln_n": float(dln.max()),
        "argmax_z": int(z),
        "argmax_species": species[int(i)],
        "n_live_cells": int(live.sum()),
    }
