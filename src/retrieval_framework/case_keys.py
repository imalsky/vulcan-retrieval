"""Which ``Config`` keys a run specification states: those every case states, and those its mode reads.

A key the run reads with no default is required: a case states it, and ``config_schema.validate_config`` refuses it
unset. A key the mode does not read is refused when a specification states it (``run_setup.load_spec``), since it would
change nothing. The prior and truth keys follow from ``config_schema.active_parameters``, the one place a mode decides
what is inferred; the other mode-dependent keys are ``MODE_KEYS``. Imports no jax.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING, NamedTuple

from retrieval_framework import config_schema as C

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["CASE_KEYS", "MODE_KEYS", "ModeKey", "read_parameter_keys", "refused_keys", "required_keys"]

# Keys every case states: each belongs to the planet, its data or its run, and a schema default would be another case's.
# The priors of the parameters every case samples come from active_parameters; prior_lnM_sigma is read with lnM fixed
# too (the forward checks' mass variant), and tp_deep_p_bar by the report's deep temperature in every mode.
CASE_KEYS = frozenset(
    {
        *("vulcan_cfg_name", "network", "atom_list", "rp_cm", "rstar_cm", "tp_gravity_cgs", "p_ref_bar"),
        *("art_pbtm_bar", "molecules", "nu_min", "nu_max", "cloud_model", "count_max", "cold_lanes", "seed"),
        *("prior_lnM_sigma", "baseline_metallicity_solar", "tp_deep_p_bar"),
    }
)


class ModeKey(NamedTuple):
    """Keys that are not a parameter's prior or truth, read in one mode only.

    Attributes
    ----------
    keys : tuple of str
        The Config keys.
    reads : callable
        Whether a Config's mode reads them.
    mode : str
        That mode, in the words of a specification.
    required : bool
        Whether a case whose mode reads them states them (no default).
    """

    keys: tuple[str, ...]
    reads: Callable[[C.Config], bool]
    mode: str
    required: bool


MODE_KEYS = (
    ModeKey(("tp_Tint_K",), lambda c: not c.tp_infer_Tint, "tp_infer_Tint: false", True),
    ModeKey(("tp_deep_param",), lambda c: c.tp_infer_Tint, "tp_infer_Tint: true", False),
    ModeKey(("tp_gamma_fixed",), lambda c: not c.tp_infer_gamma, "tp_infer_gamma: false", True),
    ModeKey(("tp_nabla_ad",), lambda c: c.tp_adiabat, "tp_adiabat: true", True),
    ModeKey(("kzz_p_ref_bar",), lambda c: c.kzz_mode == "powerlaw", "kzz_mode: powerlaw", True),
    ModeKey(("noise_inflation_per_group",), lambda c: c.infer_noise_inflation, "infer_noise_inflation: true", False),
    ModeKey(("combo", "obs_wl_lo", "obs_wl_hi"), lambda c: bool(c.obs_dir), "a data source (obs_dir)", True),
    ModeKey(("obs_max_bin_R",), lambda c: bool(c.obs_dir), "a data source (obs_dir)", False),
)

# Two placeholder groups, so the offsets and the per-group inflations count as read whenever they are inferred.
_GROUPS = ["group_a", "group_b"]
_PARAMETER_KEYS = frozenset(f.name for f in fields(C.Config) if f.name.startswith(("prior_", "truth_")))


def read_parameter_keys(cfg: C.Config) -> frozenset[str]:
    """Return the prior and truth keys this Config reads: those of every sampled or integrated parameter.

    A synthetic run also reads each one's truth, and ``truth_noise_inflation``, which scales the injected noise.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    frozenset of str
        The keys.
    """
    params = C.active_parameters(replace(cfg, integrate_nuisance=False), groups=_GROUPS)
    keys = {p.prior_key for p in params}
    if cfg.generate_synthetic_data:
        keys |= {p.truth_key for p in params} | {"truth_noise_inflation"}
    return frozenset(keys)


def required_keys(cfg: C.Config) -> frozenset[str]:
    """Return the keys this Config must have set: ``CASE_KEYS``, the parameters' keys, and its mode's required keys.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    frozenset of str
        The keys.
    """
    mode = {k for m in MODE_KEYS if m.required and m.reads(cfg) for k in m.keys}
    return CASE_KEYS | read_parameter_keys(cfg) | mode


def refused_keys(cfg: C.Config) -> dict[str, str]:
    """Return the keys this Config's mode does not read, each with why, for a specification that states one.

    Parameters
    ----------
    cfg : config_schema.Config
        The run's configuration.

    Returns
    -------
    dict
        Key to the reason, in a specification's words.
    """
    unread = _PARAMETER_KEYS - read_parameter_keys(cfg) - CASE_KEYS
    out = {
        k: "it is read only with generate_synthetic_data: true"
        if k.startswith("truth_") and not cfg.generate_synthetic_data
        else "no parameter this spec samples reads it"
        for k in unread
    }
    out.update({k: f"it is read only with {m.mode}" for m in MODE_KEYS if not m.reads(cfg) for k in m.keys})
    return out
