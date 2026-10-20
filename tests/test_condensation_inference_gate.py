"""Resolved-config condensation inference gate.

Condensation is an engine (forward-model) capability only: every retrieval
pipeline evaluates a likelihood, so it refuses condensation unless
``allow_condense_inference``. The early ``cfg_overrides`` gate
in ``config_schema.validate_config`` catches the common case, but a base VULCAN
config can default ``use_condense=True`` (e.g. ``Earth.yaml``) without the flag
appearing in ``cfg_overrides``. ``retrieval_forward._refuse_condense_inference``
gates on the RESOLVED ``chem.conden_spec`` instead, closing that bypass.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

# Import order: vulcan_chem must precede exojax, so check availability with
# find_spec; importorskip("exojax") would import it first and break collection.
if importlib.util.find_spec("exojax") is None:                  # pragma: no cover
    pytest.skip(
        "chemistry/RT stack (exojax) not installed; the CI here is "
        "stack-free. Run this integration test locally.",
        allow_module_level=True,
    )

from retrieval_framework.retrieval_forward import _refuse_condense_inference  # noqa: E402


@pytest.mark.parametrize("conden, allow, refused", [
    (True, False, True),     # a retrieval through a resolved condensing state
    (True, True, False),     # the explicit expert opt-in
    (False, False, False),   # conden_spec is None: condensation off
])
def test_gate_reads_the_resolved_conden_spec(conden, allow, refused):
    chem = SimpleNamespace(conden_spec=object() if conden else None)
    cfg = SimpleNamespace(allow_condense_inference=allow, vulcan_cfg_name="Earth")
    if refused:
        with pytest.raises(ValueError, match="RESOLVED VULCAN config"):
            _refuse_condense_inference(chem, cfg)
    else:
        _refuse_condense_inference(chem, cfg)
