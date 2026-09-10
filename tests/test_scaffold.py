# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Phase 0 smoke tests: package skeleton imports, parity fixture loads, and the
golden-array helper round-trips. These guard the scaffold itself; real
per-module goldens/invariants arrive in later phases.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import tests.golden as golden_mod
from tests.golden import assert_matches_golden

SUBPACKAGES = [
    "geosigma.core",
    "geosigma.config",
    "geosigma.preprocess",
    "geosigma.themes",
    "geosigma.kriging",
    "geosigma.assembly",
]


@pytest.mark.parametrize("modname", SUBPACKAGES)
def test_subpackage_importable(modname):
    """Every scaffolded subpackage imports and carries a docstring."""
    mod = importlib.import_module(modname)
    assert mod.__doc__ and mod.__doc__.strip()


def test_single_layer_fixture_aligned(single_layer):
    """Top and bottom of the parity fixture load and share a grid."""
    top, bottom = single_layer["top"], single_layer["bottom"]
    assert top.ndim == 2 and bottom.ndim == 2
    assert top.shape == bottom.shape

    tm, bm = single_layer["top_meta"], single_layer["bot_meta"]
    for key in ("ncols", "nrows", "xllcorner", "yllcorner"):
        assert tm[key] == bm[key], f"grid metadata '{key}' mismatch"
    assert (tm["nrows"], tm["ncols"]) == top.shape


def test_golden_helper_roundtrip(tmp_path, monkeypatch):
    """assert_matches_golden writes then matches; a perturbation fails."""
    monkeypatch.setattr(golden_mod, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(golden_mod, "REGEN", True)
    arr = np.arange(12, dtype=float).reshape(3, 4)
    assert_matches_golden("roundtrip", values=arr)  # writes golden

    monkeypatch.setattr(golden_mod, "REGEN", False)
    assert_matches_golden("roundtrip", values=arr)  # matches

    with pytest.raises(AssertionError):
        assert_matches_golden("roundtrip", values=arr + 1.0)  # perturbed -> fail


def test_golden_helper_missing_is_instructive(tmp_path, monkeypatch):
    """A missing golden fails with a regen instruction rather than silently."""
    monkeypatch.setattr(golden_mod, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(golden_mod, "REGEN", False)
    with pytest.raises(AssertionError, match="GEOSIGMA_REGEN_GOLDEN=1"):
        assert_matches_golden("does_not_exist", values=np.zeros(3))
