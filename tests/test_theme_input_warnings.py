# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""build_theme warns on thick <= 0 and var0 <= 0 without changing the grid."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from geosigma.themes import GeoSigmaWarning, ThemeData, build_theme
from geosigma.themes import themes as theme_specs

GX = np.arange(6) * 100.0
GY = np.arange(5) * 100.0
N_LAYERS = 3


def _skytem_data(thick):
    """Two soundings; depth 20/40/60 m, DOI deep enough that nothing is masked."""
    return ThemeData(
        xs=np.array([150.0, 350.0]),
        ys=np.array([150.0, 250.0]),
        attributes={
            "depth": np.array([[20.0, 40.0, 60.0], [20.0, 40.0, 60.0]]),
            "thick": np.asarray(thick, dtype=float),
            "doi": np.array([200.0, 200.0]),
        },
        complexity=np.full((GY.size, GX.size), 2.0),
    )


def _build(thick):
    spec = theme_specs.skytem_spec(n_prereq=2)
    return build_theme(_skytem_data(thick), spec, GX, GY, N_LAYERS)


def test_no_warning_for_positive_thickness():
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoSigmaWarning)
        _build([[5.0, 8.0, 12.0], [5.0, 8.0, 12.0]])


def test_zero_thickness_warns_for_points_and_cells():
    thick = [[5.0, 0.0, 12.0], [5.0, 8.0, 12.0]]
    with pytest.warns(GeoSigmaWarning) as rec:
        _build(thick)
    msgs = [str(w.message) for w in rec]
    assert any("1 data point(s) have thick <= 0" in m and "L02: 1" in m for m in msgs)
    assert any("var0 <= 0" in m and "L02" in m for m in msgs)


def test_negative_var0_warns_even_without_thick():
    spec = theme_specs.fewtem_spec(n_prereq=2)
    spec = dataclasses.replace(spec, var0_fn=lambda a: -np.ones_like(a["depth"]))
    data = ThemeData(
        xs=np.array([150.0]),
        ys=np.array([150.0]),
        attributes={"depth": np.array([[20.0, 40.0, 60.0]]), "doi": np.array([200.0])},
        complexity=np.full((GY.size, GX.size), 2.0),
    )
    with pytest.warns(GeoSigmaWarning, match="var0 <= 0"):
        build_theme(data, spec, GX, GY, N_LAYERS)


def test_masked_cells_do_not_warn():
    """thick <= 0 only below the DOI: the cells are masked, so no var0 warning."""
    data = _skytem_data([[5.0, 8.0, 0.0], [5.0, 8.0, 0.0]])
    data.attributes["doi"] = np.array([50.0, 50.0])  # layer 3 (60 m) is masked
    spec = theme_specs.skytem_spec(n_prereq=2)
    with pytest.warns(GeoSigmaWarning) as rec:
        build_theme(data, spec, GX, GY, N_LAYERS)
    msgs = [str(w.message) for w in rec]
    assert any("thick <= 0" in m for m in msgs)  # the point-level check still fires
    assert not any("var0 <= 0" in m for m in msgs)


def test_warning_does_not_change_the_grid():
    thick = [[5.0, 0.0, 12.0], [5.0, 8.0, 12.0]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeoSigmaWarning)
        g = _build(thick)
    spec = theme_specs.skytem_spec(n_prereq=2)
    from geosigma.themes import base

    orig = base._warn_nonpositive_inputs
    base._warn_nonpositive_inputs = lambda *a, **k: None
    try:
        g_silent = build_theme(_skytem_data(thick), spec, GX, GY, N_LAYERS)
    finally:
        base._warn_nonpositive_inputs = orig
    np.testing.assert_array_equal(g, g_silent)
