"""Resolved-config condensation inference gate (F5).

Condensation is a forward-model capability only. The early ``cfg_overrides`` gate
in ``config_schema.validate_config`` catches the common case, but a base VULCAN
config can default ``use_condense=True`` (e.g. ``Earth.yaml``) without the flag
appearing in ``cfg_overrides``. ``retrieval_forward._refuse_condense_inference``
gates on the RESOLVED ``chem.conden_spec`` instead, closing that bypass.

See ../../docs/condensation_differentiation.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from retrieval_framework.retrieval_forward import _refuse_condense_inference


def _chem(conden_spec):
    return SimpleNamespace(conden_spec=conden_spec)


def _cfg(run_inference, allow=False, name="Earth"):
    return SimpleNamespace(
        run_inference=run_inference,
        allow_condense_inference=allow,
        vulcan_cfg_name=name,
    )


def test_refuses_inference_when_conden_resolved():
    with pytest.raises(ValueError, match="RESOLVED VULCAN config"):
        _refuse_condense_inference(_chem(object()), _cfg(run_inference=True))


def test_allows_forward_solve_with_conden():
    # run_inference=False (forward / synthetic) is always allowed.
    _refuse_condense_inference(_chem(object()), _cfg(run_inference=False))


def test_allows_explicit_optin():
    _refuse_condense_inference(_chem(object()), _cfg(run_inference=True, allow=True))


def test_noop_without_condensation():
    # conden_spec is None when condensation is off in the resolved config.
    _refuse_condense_inference(_chem(None), _cfg(run_inference=True, name="W39b"))
