"""The keys a specification states follow the parameters its mode samples, in every mode combination.

A parameter ``config_schema.specs_from_config`` samples (or integrates out) has its prior required and never refused;
a parameter it does not sample has its prior refused when stated. A synthetic run also states every truth it injects.
The prior and truth names are spelled out here by hand, so a table that drifts from the parameter list fails.
"""

from __future__ import annotations

import itertools
from dataclasses import fields, replace

from conftest import case_config

from retrieval_framework import case_keys as K
from retrieval_framework import config_schema as C

GROUPS = ["group_a", "group_b"]
PRIORS = {f.name for f in fields(C.Config) if f.name.startswith("prior_")}
TRUTHS = {f.name for f in fields(C.Config) if f.name.startswith("truth_")}


def _keys(name: str) -> tuple[str, str]:
    """The prior and truth keys of a parameter, by its name."""
    if name.startswith("offset_"):
        return "prior_offset_ppm", "truth_offset_ppm"
    if name.startswith("noise_inflation"):
        return "prior_noise_inflation", "truth_noise_inflation"
    if name == "lnM":
        return "prior_lnM_sigma", "truth_lnM"
    return f"prior_{name}", f"truth_{name}"


def _modes():
    for gamma, deep, adiabat, cloud, s_o, kzz, lnm, off, noise, integrate, synth in itertools.product(
        (True, False),
        ((False, "Tint"), (True, "Tint"), (True, "T1bar")),
        (True, False),
        ("powerlaw", "deck", "none"),
        (True, False),
        ("scale", "powerlaw"),
        (True, False),
        (True, False),
        ((False, False), (True, False), (True, True)),
        (True, False),
        (True, False),
    ):
        if integrate and not (off or noise[0]):
            continue
        yield case_config(
            tp_infer_gamma=gamma,
            tp_infer_Tint=deep[0],
            tp_deep_param=deep[1],
            tp_adiabat=adiabat,
            cloud_model=cloud,
            infer_s_o=s_o,
            kzz_mode=kzz,
            infer_lnM=lnm,
            infer_offsets=off,
            infer_noise_inflation=noise[0],
            noise_inflation_per_group=noise[1],
            integrate_nuisance=integrate,
            generate_synthetic_data=synth,
        )


def test_required_and_refused_keys_follow_the_sampled_parameters_in_every_mode():
    n = 0
    for cfg in _modes():
        n += 1
        sampled = [s.name for s in C.specs_from_config(replace(cfg, integrate_nuisance=False), groups=GROUPS)]
        priors = {_keys(name)[0] for name in sampled}
        truths = {_keys(name)[1] for name in sampled} | {"truth_noise_inflation"}  # the injection scales its noise
        required, refused = K.required_keys(cfg), set(K.refused_keys(cfg))
        read = priors | (truths if cfg.generate_synthetic_data else set())
        assert read <= required, (cfg, sorted(read - required))
        assert not read & refused, (cfg, sorted(read & refused))
        # prior_lnM_sigma is read with lnM fixed too (the forward checks' mass variant)
        assert PRIORS - priors - {"prior_lnM_sigma"} <= refused, (cfg, sorted(PRIORS - priors - refused))
        unread_truths = TRUTHS - (truths if cfg.generate_synthetic_data else set())
        assert unread_truths <= refused, (cfg, sorted(unread_truths - refused))
        assert not (PRIORS | TRUTHS) & required - read - {"prior_lnM_sigma"}, cfg
    assert n > 1000  # every combination ran
