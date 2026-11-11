"""The device a process runs on, for a run's record, and the solver an analysis tool picks for it.

A run's own solver is its specification's ``runtime`` block (``run_setup.export_runtime``); ``choose_solver`` serves the
tools that re-solve a finished run. Imports jax only inside its functions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["choose_solver", "hardware_profile"]

# The CUDA compute capability the `ffi` block-Thomas kernel runs on.
_FFI_COMPUTE_CAPABILITY = "9.0"


def hardware_profile() -> dict:
    """Return the hardware this process runs on, read from JAX and the OS, for the log and config.json.

    Returns
    -------
    dict
        Backend, device kind, host cores and RAM, the solver environment and, on a CUDA device, its SM count, compute
        capability and memory.
    """
    import jax

    dev = jax.devices()[0]
    prof = {
        "backend": jax.default_backend(),
        "device_kind": dev.device_kind,
        "n_devices": len(jax.devices()),
        # sched_getaffinity is Linux-only (absent on macOS)
        "host_cores": (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()),
        "host_ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
        "vulcan3_solver": os.environ.get("VULCAN3_SOLVER", "fast"),
        "vulcan3_chem_rhs": os.environ.get("VULCAN3_CHEM_RHS", ""),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
    }
    if prof["backend"] in ("gpu", "cuda"):
        cc = str(getattr(dev, "compute_capability", ""))
        stats = dev.memory_stats() or {}
        prof.update(
            sm_count=getattr(dev, "core_count", None),
            compute_capability=cc,
            device_mem_gib=round(stats.get("bytes_limit", 0) / 2**30, 1),
        )
    return prof


def choose_solver() -> str:
    """Set VULCAN3_SOLVER for an analysis tool from the device, unless the user set it.

    ``ffi`` (the block-Thomas CUDA kernel) on compute capability 9.0, the only card it is validated on, when its library
    is built; ``fast`` everywhere else. A run's own solver is its spec's ``runtime`` block. Import-frozen, so this runs
    before the first vulcan3 import.

    Returns
    -------
    str
        The choice and why, for the log.
    """
    import importlib.util

    import jax

    if "VULCAN3_SOLVER" in os.environ:
        return f"{os.environ['VULCAN3_SOLVER']} (set by the user)"
    step = sys.modules.get("vulcan3.jax_step")
    if step is not None:  # already frozen by an earlier import: report it
        os.environ["VULCAN3_SOLVER"] = step._SOLVER
        return f"{step._SOLVER} (vulcan3 was imported before the choice)"
    dev = jax.devices()[0]
    cc = str(getattr(dev, "compute_capability", ""))
    spec = importlib.util.find_spec("vulcan3")
    lib = Path(spec.submodule_search_locations[0]) / "csrc" / "libblock_thomas_cuda.so"  # type: ignore[union-attr, index]  # a dependency
    if jax.default_backend() in ("gpu", "cuda") and cc == _FFI_COMPUTE_CAPABILITY and lib.is_file():
        os.environ["VULCAN3_SOLVER"] = "ffi"
        return f"ffi (compute capability {cc}, {lib.name} built)"
    os.environ["VULCAN3_SOLVER"] = "fast"
    why = (
        "not a CUDA device"
        if jax.default_backend() not in ("gpu", "cuda")
        else f"compute capability {cc} is not {_FFI_COMPUTE_CAPABILITY}"
        if cc != _FFI_COMPUTE_CAPABILITY
        else f"{lib} is not built (python -c 'from vulcan3 import solver_fast; "
        "solver_fast.build(cuda=True)' on the GPU host)"
    )
    return f"fast ({why})"
