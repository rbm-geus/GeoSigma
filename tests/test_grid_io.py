# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Phase 2 tests: the grid-metadata contract and the grid loaders.

Covers the canonical contract emitted by the loaders, ESRI<->GeoTIFF geometry
round-trips, the alignment gate, and backward-compatible normalization of the
deprecated ``cellsize``/``nodata_value`` aliases.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils import (
    REQUIRED_KEYS,
    grid_coordinates_from_esri_meta,
    grids_aligned,
    load_esri_ascii_grid,
    load_geotiff_grid,
    make_grid_meta,
    normalize_grid_meta,
    validate_grid_meta,
)
from utils.ascii_to_geotiff import ascii_to_geotiff


# --------------------------------------------------------------------------
# Contract helpers
# --------------------------------------------------------------------------
def test_make_grid_meta_defaults_and_aliases():
    meta = make_grid_meta(ncols=10, nrows=5, dx=100.0, xllcorner=0.0, yllcorner=0.0)
    # All required keys present, dy defaults to dx, transform derived.
    validate_grid_meta(meta)
    assert meta["dy"] == 100.0
    assert meta["transform"].a == 100.0
    assert meta["transform"].e == -100.0
    # Upper-left y = yllcorner + nrows*dy.
    assert meta["transform"].f == pytest.approx(500.0)
    # Legacy aliases mirror canonical values.
    assert meta["cellsize"] == meta["dx"]
    assert meta["nodata_value"] == meta["nodata"]


def test_normalize_accepts_legacy_aliases():
    legacy = {
        "ncols": 3,
        "nrows": 2,
        "cellsize": 50.0,
        "xllcorner": 10.0,
        "yllcorner": 20.0,
        "nodata_value": -9999.0,
    }
    meta = normalize_grid_meta(legacy)
    validate_grid_meta(meta)
    assert meta["dx"] == 50.0 and meta["dy"] == 50.0
    assert meta["nodata"] == -9999.0
    assert meta["transform"] is not None


def test_validate_rejects_incomplete_meta():
    with pytest.raises(ValueError):
        validate_grid_meta({"ncols": 3, "nrows": 2})


def test_normalize_requires_some_spacing():
    with pytest.raises(KeyError):
        normalize_grid_meta({"ncols": 3, "nrows": 2, "xllcorner": 0, "yllcorner": 0})


# --------------------------------------------------------------------------
# Loaders on the sample data
# --------------------------------------------------------------------------
def test_geotiff_loader_contract(single_layer):
    meta = single_layer["top_meta"]
    for key in REQUIRED_KEYS:
        assert key in meta
    assert "crs" in meta  # may be None: the sample GeoTIFFs carry no CRS
    assert meta["transform"] is not None
    # yllcorner must be *below* the top surface (regression: it was computed
    # above the raster before Phase 2).
    top_y = meta["transform"].f
    assert meta["yllcorner"] < top_y
    assert meta["yllcorner"] == pytest.approx(top_y - meta["nrows"] * meta["dy"])
    # Shape matches declared geometry.
    assert single_layer["top"].shape == (meta["nrows"], meta["ncols"])


def test_two_sample_grids_aligned(single_layer):
    assert grids_aligned(single_layer["top_meta"], single_layer["bot_meta"])


def test_misaligned_grids_detected(single_layer):
    shifted = dict(single_layer["bot_meta"])
    shifted["xllcorner"] = shifted["xllcorner"] + 12345.0
    assert not grids_aligned(single_layer["top_meta"], shifted)


# --------------------------------------------------------------------------
# ESRI ASCII round-trip
# --------------------------------------------------------------------------
def _write_ascii(path, *, corner=True):
    ref = "xllcorner" if corner else "xllcenter"
    lines = [
        "ncols 3",
        "nrows 2",
        f"{ref} 100.0",
        f"{'yllcorner' if corner else 'yllcenter'} 200.0",
        "cellsize 50.0",
        "NODATA_value -9999",
        "1 2 3",
        "4 5 -9999",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_esri_loader_contract_and_nodata(tmp_path):
    p = tmp_path / "grid.asc"
    _write_ascii(p, corner=True)
    data, meta = load_esri_ascii_grid(str(p))
    validate_grid_meta(meta)
    assert data.shape == (2, 3)
    assert np.isnan(data[1, 2])  # nodata -> NaN
    assert meta["dx"] == 50.0 and meta["xllcorner"] == 100.0
    assert meta["crs"] is None


def test_esri_centre_referenced_header(tmp_path):
    p = tmp_path / "grid_centre.asc"
    _write_ascii(p, corner=False)
    _, meta = load_esri_ascii_grid(str(p))
    # centre 100 -> corner 100 - cellsize/2.
    assert meta["xllcorner"] == pytest.approx(100.0 - 25.0)


def test_ascii_to_geotiff_geometry_roundtrip(tmp_path):
    asc = tmp_path / "grid.asc"
    _write_ascii(asc, corner=True)
    tif = tmp_path / "grid.tif"
    ascii_to_geotiff(str(asc), str(tif), crs="EPSG:25832")

    _, am = load_esri_ascii_grid(str(asc))
    _, gm = load_geotiff_grid(str(tif))
    assert grids_aligned(am, gm)


def test_grid_coordinates_centres(tmp_path):
    p = tmp_path / "grid.asc"
    _write_ascii(p, corner=True)
    _, meta = load_esri_ascii_grid(str(p))
    x, y, xx, yy = grid_coordinates_from_esri_meta(meta)
    # First centre = corner + dx/2.
    assert x[0] == pytest.approx(100.0 + 25.0)
    assert y[0] == pytest.approx(200.0 + 25.0)
    assert xx.shape == (meta["ncols"], meta["nrows"])  # MATLAB-style (ij)
