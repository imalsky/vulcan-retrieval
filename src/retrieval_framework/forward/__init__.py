"""Retrieval-side configuration and composition for the shared forward engine.

The forward-model ENGINE moved to the ``vulcan-forward`` distribution
(2026-07-29): ``vulcan_chem``, ``exojax_rt`` and ``interp_map`` now live at
``vulcan_forward.*`` and are shared with vulcan-jwst-tool, so neither
application depends on the other. What remains here is this repo's own:

- ``config``      -- this repo's paths + WASP-39 b case constants + run
                     profiles + parameter-vector labels; re-exports the shared
                     physics constants from ``vulcan_forward.constants`` so the
                     two distributions cannot drift, and hands the engine its
                     data root. No heavy imports; always safe to import first.
- ``sensitivity`` -- the 4-parameter theta -> spectrum chain composer used by
                     ``examples/run_demo.py`` and ``validation/smoke_test.py``.
                     Production uses ``retrieval_forward.py`` instead, which
                     carries lnR0, clouds and the two-stage solve.

IMPORT ORDER IS STILL LOAD-BEARING: ``vulcan_forward.vulcan_chem`` sets the
VULCAN_JAX_* import-frozen env vars and jax x64 at import and must come before
anything from exojax (it raises if it arrives late).

This ``__init__`` deliberately imports NOTHING, so importing the subpackage stays
free of jax/vulcan_jax/exojax side effects.
"""
