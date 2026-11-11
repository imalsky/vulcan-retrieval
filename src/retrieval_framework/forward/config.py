"""The repository's filesystem layout; importing it hands the engine its data root.

Pure constants and paths, with no jax, vulcan3 or exojax, so it is safe before the environment-order-sensitive setup.
"""

from __future__ import annotations

import os
from pathlib import Path

from vulcan_forward import paths as _fwd_paths

__all__ = ["DATA_DIR", "PROJECT_ROOT", "REPO_DIR"]

# VULCAN_PROJECT_ROOT is the directory containing the vulcan-retrieval/ checkout. Without it the root is inferred from
# this file inside an editable checkout; a site-packages install must set it.
_env_root = os.environ.get("VULCAN_PROJECT_ROOT")
if _env_root:
    PROJECT_ROOT = Path(_env_root).expanduser()
    REPO_DIR = PROJECT_ROOT / "vulcan-retrieval"
else:
    REPO_DIR = Path(__file__).resolve().parents[3]
    PROJECT_ROOT = REPO_DIR.parent
if not (REPO_DIR / "data" / "cm24_wasp39b").is_dir():  # tracked, so in every clone
    raise RuntimeError(
        f"vulcan-retrieval data tree not found at {REPO_DIR}/data. Set VULCAN_PROJECT_ROOT "
        "to the directory that contains the vulcan-retrieval/ checkout (a site-packages "
        "install cannot infer it). Large caches (data/opacity_cache, data/exomolop) "
        "are seeded separately -- see the repo README data policy."
    )

DATA_DIR = REPO_DIR / "data"

# Takes precedence over $VULCAN_FORWARD_DATA. The engine owns every path inside that tree: ask paths.
_fwd_paths.set_data_root(DATA_DIR)
