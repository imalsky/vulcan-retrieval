"""The environment-gated CUDA profiling hook of the likelihood batches, inert unless VULCAN_PROFILE_BATCH is set."""

from __future__ import annotations

import atexit
import ctypes
import logging
import os
from pathlib import Path

__all__ = ["BatchProfiler"]

log = logging.getLogger("retrieval")


class BatchProfiler:
    """Environment-gated profiling hook, inert unless VULCAN_PROFILE_BATCH is set.

    VULCAN_PROFILE_BATCH=k calls cudaProfilerStart just before likelihood batch k of this process and cudaProfilerStop
    when batch k+1 starts or at exit. Batch 1 compiles the cold evaluator and the first warm batch compiles the warm
    one, so k = 3 is the first steady warm batch of a fresh run. Meant for nsys --capture-range=cudaProfilerApi
    --capture-range-end=stop.
    """

    def __init__(self) -> None:
        self.k = int(os.environ.get("VULCAN_PROFILE_BATCH", "0") or 0)
        self.n = 0
        self.active = False
        if self.k <= 0:
            return
        self._rt = self._cudart()
        atexit.register(self.stop)
        log.info("profile: capture batch %s (cudaProfilerStart/Stop)", self.k)

    @staticmethod
    def _cudart() -> ctypes.CDLL:
        """Load the CUDA runtime of the jax wheels, or the system's."""
        paths = []
        try:
            import nvidia.cuda_runtime as m  # the CUDA runtime of the jax wheels

            paths = [str(Path(p) / "lib" / "libcudart.so.12") for p in m.__path__]
        except ImportError:
            pass
        for p in [*paths, "libcudart.so.12", "libcudart.so"]:
            try:
                return ctypes.CDLL(p)
            except OSError:
                continue
        raise OSError("VULCAN_PROFILE_BATCH: no libcudart to call cudaProfilerStart from")

    def stop(self) -> None:
        """Stop the profiler if it is running."""
        if self.active:
            rc = int(self._rt.cudaProfilerStop())
            self.active = False
            log.info("profile: cudaProfilerStop after batch %s (rc %s)", self.k, rc)

    def batch_begin(self) -> None:
        """Count one evaluator call; the previous batch's device work is complete here."""
        self.n += 1
        if self.k and self.n == self.k + 1:
            self.stop()
        if self.k and self.n == self.k:
            rc = int(self._rt.cudaProfilerStart())
            self.active = True
            log.info("profile: cudaProfilerStart before batch %s (rc %s)", self.k, rc)
