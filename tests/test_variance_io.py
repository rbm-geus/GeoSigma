# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Variance-map export: the handoff GeoTIFFs and the GIS-ready std GeoTIFFs.

The handoff files (``variance_NN.tif``) must store the variance exactly as
computed, sentinel included. The display files (``std_NN.tif``) hold the
standard deviation with no-information cells as GeoTIFF no-data.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from geosigma.themes import (
    DISPLAY_NODATA,
    NODATA_VARIANCE,
    combine_variances,
    informed_mask,
    read_variance_stack,
    write_display_stack,
    write_variance_stack,
)
from geosigma.themes.__main__ import main as themes_cli
from utils import make_grid_meta

NY, NX, NL = 6, 5, 2


def _meta(nodata=None):
    return make_grid_meta(
        ncols=NX,
        nrows=NY,
        dx=100.0,
        xllcorner=500000.0,
        yllcorner=6200000.0,
        nodata=nodata,
        crs="EPSG:25832",
    )


def _themes():
    """Two themes covering the combine cases that matter for export."""
    a = np.full((NY, NX, NL), NODATA_VARIANCE)
    b = np.full((NY, NX, NL), NODATA_VARIANCE)
    a[0:2] = 4.0  # informed by a only
    b[2:3] = 9.0  # informed by b only
    b[3:4] = np.inf  # a sentinel + b +inf -> combines to the sentinel itself
    # rows 4-5: both sentinel -> combine to NODATA_VARIANCE / 2
    return [a, b]


def _read_raw(path):
    with rasterio.open(path) as src:
        return src.read(1), src.nodata


# ---------------------------------------------------------------------------
# informed_mask
# ---------------------------------------------------------------------------


def test_informed_mask_marks_only_cells_with_information():
    mask = informed_mask(_themes())
    expected = np.zeros((NY, NX, NL), dtype=bool)
    expected[0:3] = True
    np.testing.assert_array_equal(mask, expected)


def test_informed_mask_treats_nan_inf_and_sentinel_as_no_information():
    t = np.array([np.nan, np.inf, NODATA_VARIANCE, 2 * NODATA_VARIANCE, 1.0])
    t = t.reshape(1, 5, 1)
    np.testing.assert_array_equal(
        informed_mask([t])[0, :, 0], [False, False, False, False, True]
    )


def test_informed_mask_requires_a_theme():
    with pytest.raises(ValueError):
        informed_mask([])


# ---------------------------------------------------------------------------
# Handoff files stay bit-identical
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nodata", [None, -9999.0])
def test_variance_stack_stores_values_exactly(tmp_path, nodata):
    combined = combine_variances(_themes())
    # The both-sentinel rows combine to NODATA_VARIANCE / 2 (to round-off).
    np.testing.assert_allclose(combined[4:], NODATA_VARIANCE / 2, rtol=1e-12)

    paths = write_variance_stack(combined, _meta(nodata), tmp_path)
    for s, path in enumerate(paths):
        band, file_nodata = _read_raw(path)
        assert file_nodata == nodata
        np.testing.assert_array_equal(band, combined[:, :, s].astype(np.float32))


def test_display_stack_does_not_touch_variance_files(tmp_path):
    combined = combine_variances(_themes())
    meta = _meta()
    paths = write_variance_stack(combined, meta, tmp_path)
    before = [p.read_bytes() for p in paths]
    write_display_stack(combined, meta, tmp_path, informed=informed_mask(_themes()))
    assert [p.read_bytes() for p in paths] == before
    stack, _ = read_variance_stack(tmp_path)
    np.testing.assert_array_equal(stack, combined.astype(np.float32))


# ---------------------------------------------------------------------------
# Display (std) files
# ---------------------------------------------------------------------------


def test_display_stack_writes_std_and_nodata(tmp_path):
    themes = _themes()
    combined = combine_variances(themes)
    informed = informed_mask(themes)

    paths = write_display_stack(combined, _meta(), tmp_path, informed=informed)
    assert [p.name for p in paths] == ["std_01.tif", "std_02.tif"]
    for s, path in enumerate(paths):
        band, file_nodata = _read_raw(path)
        assert file_nodata == DISPLAY_NODATA
        inf_s = informed[:, :, s]
        np.testing.assert_allclose(band[inf_s], np.sqrt(combined[:, :, s][inf_s]))
        assert np.all(band[~inf_s] == DISPLAY_NODATA)
    # Spot values: 4 m^2 and 9 m^2 themes give ~2 m and ~3 m. Not exact, because
    # the other theme's sentinel adds a small precision 1 / NODATA_VARIANCE.
    band, _ = _read_raw(paths[0])
    assert band[0, 0] == pytest.approx(2.0, rel=1e-4)
    assert band[2, 0] == pytest.approx(3.0, rel=1e-4)


def test_display_stack_without_mask_only_drops_non_finite(tmp_path):
    stack = np.full((NY, NX, 1), 16.0)
    stack[0, 0, 0] = np.inf
    stack[0, 1, 0] = np.nan
    (path,) = write_display_stack(stack, _meta(), tmp_path)
    band, _ = _read_raw(path)
    assert band[0, 0] == DISPLAY_NODATA and band[0, 1] == DISPLAY_NODATA
    assert np.all(band.ravel()[2:] == 4.0)


def test_display_stack_rejects_mismatched_mask(tmp_path):
    with pytest.raises(ValueError):
        write_display_stack(
            np.ones((NY, NX, NL)), _meta(), tmp_path, informed=np.ones((NY, NX, 1))
        )


# ---------------------------------------------------------------------------
# CLI writes both versions
# ---------------------------------------------------------------------------


def test_cli_combine_writes_variance_and_std(tmp_path):
    meta = _meta()
    themes = _themes()
    theme_dirs = []
    for i, t in enumerate(themes):
        d = tmp_path / f"theme{i}"
        write_variance_stack(t, meta, d)
        theme_dirs.append(d)
    # Flat terrain at 0 m with bottoms at -10/-20 m: min_map = 0.01 / 0.04 m^2,
    # below every theme value here, so the floor does not change the result.
    surfaces = tmp_path / "surfaces"
    terrain = np.zeros((NY, NX, 1))
    bottoms = np.stack([np.full((NY, NX), -10.0), np.full((NY, NX), -20.0)], -1)
    (terrain_path,) = write_variance_stack(terrain, meta, surfaces, prefix="terrain")
    bottom_paths = write_variance_stack(bottoms, meta, surfaces, prefix="bottom")

    out = tmp_path / "out"
    argv = ["combine", "--terrain", str(terrain_path), "--out", str(out)]
    for d in theme_dirs:
        argv += ["--theme", str(d)]
    for b in bottom_paths:
        argv += ["--bottom", str(b)]
    assert themes_cli(argv) == 0

    combined = combine_variances(themes)
    informed = informed_mask(themes)
    for s in range(NL):
        var, _ = _read_raw(out / f"variance_{s + 1:02d}.tif")
        np.testing.assert_array_equal(var, combined[:, :, s].astype(np.float32))
        std, nodata = _read_raw(out / f"std_{s + 1:02d}.tif")
        assert nodata == DISPLAY_NODATA
        assert np.all(std[~informed[:, :, s]] == DISPLAY_NODATA)
        np.testing.assert_allclose(
            std[informed[:, :, s]], np.sqrt(combined[:, :, s][informed[:, :, s]])
        )
