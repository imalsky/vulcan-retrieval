"""The forward build's gates on the resolved engine config: condensation, the C/O prior bound and the baked rtol.

``config_schema.validate_config`` sees only ``cfg_overrides``, but a base VULCAN config can default
``use_condense=True`` (e.g. ``Earth.yaml``), so ``retrieval_forward._refuse_condense_inference`` reads the resolved
``chem.conden_spec``.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

# Import order: vulcan_chem must precede exojax, so check availability with
# find_spec; importorskip("exojax") would import it first and break collection.
if importlib.util.find_spec("exojax") is None:
    pytest.skip(
        "chemistry/RT stack (exojax) not installed; the CI here is stack-free. Run this integration test locally.",
        allow_module_level=True,
    )

from retrieval_framework.retrieval_forward import _refuse_c_o_bound, _refuse_condense_inference, _rtol_accept


@pytest.mark.parametrize(
    ("conden", "allow", "refused"),
    [
        (True, False, True),  # a retrieval through a resolved condensing state
        (True, True, False),  # the explicit expert opt-in
        (False, False, False),  # conden_spec is None: condensation off
    ],
)
def test_gate_reads_the_resolved_conden_spec(conden, allow, refused):
    chem = SimpleNamespace(conden_spec=object() if conden else None)
    cfg = SimpleNamespace(allow_condense_inference=allow, vulcan_cfg_name="Earth")
    if refused:
        with pytest.raises(ValueError, match="RESOLVED VULCAN config"):
            _refuse_condense_inference(chem, cfg)
    else:
        _refuse_condense_inference(chem, cfg)


@pytest.mark.parametrize(
    ("co_mode", "hi", "refused"),
    [
        ("fixed_O", 0.9, False),
        ("fixed_O", 1.0, True),  # at the b_z positivity bound
        ("fixed_O", 1.1, True),
        ("proxy", 1.1, False),  # no b_z compensation, so no bound
    ],
)
def test_c_o_prior_reaching_the_positivity_bound_is_refused(co_mode, hi, refused):
    chem = SimpleNamespace(co_bz_bound=1.0)
    cfg = SimpleNamespace(co_mode=co_mode, prior_c_o=(-1.0, hi))
    if refused:
        with pytest.raises(ValueError, match="positivity"):
            _refuse_c_o_bound(chem, cfg)
    else:
        _refuse_c_o_bound(chem, cfg)


def test_chem_rtol_reads_the_baked_acceptance_tolerance():
    chem = SimpleNamespace(_integ=SimpleNamespace(statics=SimpleNamespace(rtol_accept=0.05)))
    assert _rtol_accept(chem) == 0.05
