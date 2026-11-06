"""Nautilus nested-sampling retrieval on a VULCAN 3 and ExoJAX forward model.

A concrete run is one YAML file, ``runs/<case>/<name>.yaml``; outputs go to ``output/<case>/<name>_nautilus``::

    python -m retrieval_framework.run_nautilus    runs/<case>/<name>.yaml [--resume] [--dry-run]
    python -m retrieval_framework.nautilus_report <out_dir> [--spectra N]
    python -m retrieval_framework.validate_env    <project_root> [--require-gpu]

Importing the package loads no jax. Modules, methods and sources are in docs/method.md.
"""

from __future__ import annotations

from retrieval_framework._version import __version__

__all__ = ["__version__"]
