"""Editable-install guard: the suite must exercise THIS checkout's code.

Because of the src layout, pytest imports the *installed* retrieval_framework. A
non-editable install (pip install . or a wheel) shadows the checkout, so the suite
would silently test stale code. Fail collection loudly instead (same convention as
VULCAN-JAX's conftest). Fix: pip install -e ./vulcan-retrieval --no-deps
"""
from pathlib import Path

import retrieval_framework

_SRC = Path(__file__).resolve().parent.parent / "src" / "retrieval_framework"
_IMPORTED = Path(retrieval_framework.__file__).resolve().parent

if _SRC.is_dir() and _IMPORTED != _SRC:
    raise RuntimeError(
        f"retrieval_framework imports from {_IMPORTED}, not this checkout's {_SRC}. "
        "A non-editable install is shadowing the checkout -- the suite would test stale "
        "code. Fix: pip install -e ./vulcan-retrieval --no-deps (from the repo root)."
    )
