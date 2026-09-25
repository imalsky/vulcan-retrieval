"""Resolved-config condensation inference gate (F5).

Condensation is a forward-model capability only. The early ``cfg_overrides`` gate
in ``config_schema.validate_config`` catches the common case, but a base VULCAN
config can default ``use_condense=True`` (e.g. ``Earth.yaml``) without the flag
appearing in ``cfg_overrides``. ``retrieval_forward._refuse_condense_inference``
gates on the RESOLVED ``chem.conden_spec`` instead, closing that bypass.

Record: VULCAN-JAX notes §2.5-2.6.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

# `retrieval_forward` imports vulcan_chem and exojax_rt at module scope, and that
# order is LOAD-BEARING: `vulcan_forward.vulcan_chem` raises if exojax reached
# sys.modules first (it has to own the first jax import to set x64 and the
# VULCAN_JAX_* import-frozen env vars). So availability must be checked WITHOUT
# importing -- `pytest.importorskip("exojax")` imports it and would break the
# very contract this file tests, failing the whole suite at COLLECTION time.
# `find_spec` answers the same question without executing the module.
if importlib.util.find_spec("exojax") is None:                  # pragma: no cover
    pytest.skip(
        "chemistry/RT stack (exojax) not installed; the CI here is deliberately "
        "stack-free. Run this integration test locally.",
        allow_module_level=True,
    )

from retrieval_framework.retrieval_forward import _refuse_condense_inference  # noqa: E402


@pytest.mark.parametrize("conden, run_inference, allow, refused", [
    (True, True, False, True),     # inference through a resolved condensing state
    (True, False, False, False),   # a forward / synthetic solve is always allowed
    (True, True, True, False),     # the explicit expert opt-in
    (False, True, False, False),   # conden_spec is None: condensation off
])
def test_inference_gate_reads_the_resolved_conden_spec(conden, run_inference, allow, refused):
    chem = SimpleNamespace(conden_spec=object() if conden else None)
    cfg = SimpleNamespace(run_inference=run_inference, allow_condense_inference=allow,
                          vulcan_cfg_name="Earth")
    if refused:
        with pytest.raises(ValueError, match="RESOLVED VULCAN config"):
            _refuse_condense_inference(chem, cfg)
    else:
        _refuse_condense_inference(chem, cfg)
