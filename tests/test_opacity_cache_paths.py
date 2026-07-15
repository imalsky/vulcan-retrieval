"""Opacity cache-path regressions (no downloads, no heavy stack).

Pins the contract between config.MOLECULES and exojax/radis: every HITRAN
cache path handed to MdbHitran must have a path STEM that radis can parse as
a molecule -- exojax >= 2.x derives the HITRAN molecule id from the stem. The
old h2he layout ("<db>_h2he") violated this and every h2he run died in
MdbHitran before downloading anything (2026-07-15 data-dependency audit);
the layout is now "h2he/<db>".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from retrieval_framework.forward import config

radis_classes = pytest.importorskip(
    "radis.db.classes", reason="radis (exojax dependency) not installed")


def _hitran_cache_paths():
    """Every (molecule, cache path) pair the RT can hand to MdbHitran."""
    pairs = []
    for key, spec in config.MOLECULES.items():
        if spec["source"] != "hitran":
            continue
        pairs.append((key, config.DEMO_DATABASE / spec["db"]))          # air
        pairs.append((key, config.DEMO_DATABASE / "h2he" / spec["db"]))  # h2he
    return pairs


def test_every_hitran_cache_stem_is_a_parseable_molecule():
    pairs = _hitran_cache_paths()
    assert pairs, "no HITRAN molecules configured?"
    for key, path in pairs:
        stem = Path(path).stem
        # raises NotImplementedError for an unparseable stem (the old bug)
        mol_id = radis_classes.get_molecule_identifier(stem)
        assert isinstance(mol_id, int) and mol_id > 0, (key, path)


def test_h2he_layout_is_a_subdirectory_not_a_suffix():
    # the suffix layout is the pinned regression: it must never come back
    for key, spec in config.MOLECULES.items():
        if spec["source"] != "hitran":
            continue
        bad = config.DEMO_DATABASE / f"{spec['db']}_h2he"
        with pytest.raises(Exception):
            radis_classes.get_molecule_identifier(bad.stem)
