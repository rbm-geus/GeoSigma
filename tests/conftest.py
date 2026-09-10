# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Shared pytest fixtures for the GeoSigma test harness.

The parity fixture is intentionally tiny — a single layer of the Jutland sample
data (``top_surface`` as the layer top, ``layer_1_bottom`` as its bottom) — so
tests stay fast and reviewable while still exercising real grid I/O and the
grid-metadata contract used downstream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import load_geotiff_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
JUTLAND_DIR = REPO_ROOT / "examples" / "data" / "Jutland_hydrostrat_model"


@pytest.fixture(scope="session")
def jutland_dir() -> Path:
    """Directory of the Jutland sample GeoTIFFs."""
    if not JUTLAND_DIR.is_dir():
        pytest.skip(f"Jutland sample data not found at {JUTLAND_DIR}")
    return JUTLAND_DIR


@pytest.fixture(scope="session")
def single_layer(jutland_dir: Path) -> dict:
    """One-layer parity fixture: top surface + layer-1 bottom, with metadata.

    Returns a dict::

        {
            "top":      2D ndarray (layer top elevation),
            "bottom":   2D ndarray (layer-1 bottom elevation),
            "top_meta": grid-metadata dict,
            "bot_meta": grid-metadata dict,
        }
    """
    top_path = jutland_dir / "top_surface.tif"
    bot_path = jutland_dir / "layer_1_bottom.tif"
    for p in (top_path, bot_path):
        if not p.is_file():
            pytest.skip(f"Missing sample raster {p}")

    top, top_meta = load_geotiff_grid(str(top_path))
    bottom, bot_meta = load_geotiff_grid(str(bot_path))
    return {
        "top": top,
        "bottom": bottom,
        "top_meta": top_meta,
        "bot_meta": bot_meta,
    }
