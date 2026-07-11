"""The shared VULCAN-JAX -> ExoJax forward-model engine.

Modules (import order is load-bearing where noted):

- ``config``      -- pure constants + path resolution (no heavy imports; always safe first)
- ``vulcan_chem`` -- VULCAN-JAX chemistry wrapper. Sets the VULCAN_JAX_* import-frozen
                     env vars and jax x64 at import; MUST be imported before exojax
                     (enforced with a loud RuntimeError guard).
- ``exojax_rt``   -- ExoJax ArtTransPure/ArtEmisPure radiative transfer
- ``interp_map``  -- differentiable log-P bridge (VULCAN grid -> ART grid)
- ``sensitivity`` -- theta -> converged VMR -> transit spectrum chain composer for jvps

This ``__init__`` deliberately imports NOTHING: importing the subpackage must stay
free of jax/vulcan_jax/exojax side effects so light consumers (config readers, the
jwst-tool GUI's cache face) stay fast.
"""
