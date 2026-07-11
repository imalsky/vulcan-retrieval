"""Compose the full differentiable chain:  theta -> converged VMR -> transit spectrum.

    forward(theta) = transmission_depth( bridge( converged_ymix(theta), T(theta) ) )

theta = [lnZ, c_o, lnKzz, dT] (dT = uniform T offset). The returned ``forward`` is one pure-JAX function
through which ``jax.jvp`` / ``jax.jacfwd`` push forward-mode tangents end-to-end -- from
the physics parameters all the way to ``(R_p(lambda)/R_star)^2``.

Importing this module triggers the env-ordered VULCAN-JAX setup (via vulcan_chem) before
any exojax import, which is the required order.
"""
from __future__ import annotations

from types import SimpleNamespace

# import order is load-bearing: vulcan_chem (env + jax x64) before anything exojax
from retrieval_framework.forward import config        # pure constants, no heavy imports
from retrieval_framework.forward import vulcan_chem   # sets env + jax x64; must import before exojax
import jax.numpy as jnp

from retrieval_framework.forward import exojax_rt
from retrieval_framework.forward import interp_map


def build_forward(profile: dict) -> SimpleNamespace:
    """Build ``forward(theta)`` and return it alongside the chem/rt sub-models.

    Returns SimpleNamespace with: forward, chem, rt, mol_cols, h2_col.
    """
    chem = vulcan_chem.build_chem_model(profile)
    rt = exojax_rt.build_rt_model(profile)
    to_art = interp_map.make_to_art(chem.p_bar, rt.p_art_bar)

    mol_cols = {key: chem.sidx[config.MOLECULES[key]["vulcan"]] for key in rt.molecules}
    h2_col = chem.sidx[config.BULK_H2_VULCAN]
    he_col = chem.sidx["He"]        # H2-He CIA partner (He is inert in the network)
    T_base_j = jnp.asarray(chem.T_base)
    species_masses = chem.species_masses

    def forward(theta):
        ymix = chem.converged_ymix(theta)                 # (nz, ni)
        T_v = T_base_j + theta[3]                          # same perturbed T as chemistry
        mmw_v = ymix @ species_masses                      # (nz,)

        T_art = to_art(T_v)
        mmw_art = to_art(mmw_v)
        vmr = {key: to_art(ymix[:, col]) for key, col in mol_cols.items()}
        vmr_h2 = to_art(ymix[:, h2_col])
        # He CIA was silently omitted before 2026-07-10 (vmr_he defaulted to None);
        # the RT now REQUIRES it -- regenerate sensitivity.npz/wide_sensitivity.npz
        vmr_he = to_art(ymix[:, he_col])
        return rt.transmission_depth(vmr, vmr_h2, T_art, mmw_art, vmr_he=vmr_he)

    return SimpleNamespace(forward=forward, chem=chem, rt=rt,
                           mol_cols=mol_cols, h2_col=h2_col)
