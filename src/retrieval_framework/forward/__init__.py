"""Retrieval-side configuration and composition for the shared forward engine.

The forward-model ENGINE lives in the ``vulcan-forward`` distribution:
``vulcan_chem``, ``exojax_rt`` and ``interp_map`` are at ``vulcan_forward.*``
and are shared with jwst-transit-authority, so neither application depends on the
other. What remains here is this repo's own:

- ``config``      -- this repo's paths + WASP-39 b case constants + the
                     SMOKE profile, and hands the engine its data root. No
                     heavy imports; always safe to import first. The shared
                     physics constants are ``vulcan_forward.constants``.

Import order: ``vulcan_forward.vulcan_chem`` sets the VULCAN_JAX_* import-frozen
env vars and jax x64 at import and must precede exojax (it raises otherwise).

This ``__init__`` imports nothing, so importing the subpackage stays free of
jax/vulcan_jax/exojax side effects.
"""
