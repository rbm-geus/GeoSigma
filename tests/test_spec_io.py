# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for the declarative theme-spec front-end (``geosigma.themes.spec_io``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geosigma.themes import (
    NODATA_VARIANCE,
    RangeGroup,
    ThemeData,
    build_theme,
    compile_expr,
    dump_spec,
    dump_spec_yaml,
    load_spec,
    load_spec_yaml,
)
from geosigma.themes import themes as theme_specs
from geosigma.themes.base import _range_map

SPECS = Path(__file__).resolve().parents[1] / "examples" / "theme_specs"

_PACES_19 = {
    "name": "PACES",
    "search_radius": 1,
    "cert_fun": "FRAFA_apr2023",
    "combine": "nearest",
    "attributes": {"depth": "mean"},
    "range_model": [
        {"layers": "L01-L13", "range_by_complexity": [100, 500, 400, 250, 100]},
        {"layers": "L14-L19", "range_by_complexity": [500, 500, 500, 500, 500]},
    ],
    "var0": "(0.5 * maximum(2, 0.25*depth))**2",
    "kernel": "75",
    "mask": "depth > 18",
}

_SINGLE = {
    "name": "single",
    "search_radius": 4,
    "cert_fun": "ILM_sep2023",
    "attributes": {"depth": "mean", "thick": "min", "doi": "mean"},
    "range_model": [
        {"layers": "all", "range_by_complexity": [50, 300, 250, 150, 80]},
    ],
    "var0": "(0.5 * 1.2 * thick)**2",
    "kernel": "maximum(depth, 60)",
    "mask": "doi < depth",
}


def test_two_group_range_model_applies_correct_table_per_side():
    spec = load_spec(_PACES_19)
    n_layers = 19
    rmap = _range_map(np.full((2, 3), 2.0), spec.range_model, n_layers)
    assert np.all(rmap[:, :, 0:13] == 400.0)
    assert np.all(rmap[:, :, 13:19] == 500.0)
    rmap3 = _range_map(np.full((2, 3), 3.0), spec.range_model, n_layers)
    assert np.all(rmap3[:, :, 0:13] == 250.0)
    assert np.all(rmap3[:, :, 13:19] == 500.0)


def test_single_group_range_model_is_uniform_across_layers():
    spec = load_spec(_SINGLE)
    rmap = _range_map(np.full((2, 2), 3.0), spec.range_model, 8)
    assert rmap.shape == (2, 2, 8)
    assert np.all(rmap == 150.0)


def test_range_model_gap_and_overlap_raise():
    gap = [RangeGroup((0, 0), [100, 500, 400, 250, 100])]
    with pytest.raises(ValueError, match="uncovered"):
        _range_map(np.full((1, 1), 1.0), gap, 3)
    overlap = [
        RangeGroup("all", [100, 500, 400, 250, 100]),
        RangeGroup((0, 0), [100, 500, 400, 250, 100]),
    ]
    with pytest.raises(ValueError, match="more than one"):
        _range_map(np.full((1, 1), 1.0), overlap, 3)


def test_dict_round_trip_preserves_spec():
    spec = load_spec(_PACES_19)
    d2 = dump_spec(spec)
    spec2 = load_spec(d2)
    assert spec2.name == spec.name
    assert spec2.search_radius == spec.search_radius
    assert spec2.cert_fun_name == spec.cert_fun_name
    assert tuple(spec2.attributes) == tuple(spec.attributes)
    assert spec2.var0_fn.expr == spec.var0_fn.expr
    assert spec2.kernel_fn.expr == spec.kernel_fn.expr
    assert spec2.mask_fn.expr == spec.mask_fn.expr
    assert d2["range_model"][0]["layers"] == "L01-L13"
    assert d2["range_model"][1]["layers"] == "L14-L19"
    cx = np.full((2, 2), 2.0)
    np.testing.assert_array_equal(
        _range_map(cx, spec.range_model, 19),
        _range_map(cx, spec2.range_model, 19),
    )


def test_yaml_file_round_trip(tmp_path):
    spec = load_spec_yaml(SPECS / "paces.spec.yaml")
    out = tmp_path / "paces_out.yaml"
    text = dump_spec_yaml(spec, out)
    assert "range_model" in text
    spec2 = load_spec_yaml(out)
    assert spec2.mask_fn.expr == spec.mask_fn.expr
    np.testing.assert_array_equal(
        _range_map(np.full((2, 2), 2.0), spec.range_model, 19),
        _range_map(np.full((2, 2), 2.0), spec2.range_model, 19),
    )


@pytest.mark.parametrize("fname,n_layers", [("paces.spec.yaml", 19),
                                            ("dronetem.spec.yaml", 8)])
def test_example_specs_load_and_build(fname, n_layers):
    spec = load_spec_yaml(SPECS / fname)
    grid_x = np.arange(4) * 100.0
    grid_y = np.arange(3) * 100.0
    xs = np.array([150.0])
    ys = np.array([110.0])
    attributes = {
        "depth": np.full((1, n_layers), 8.0),
        "thick": np.full((1, n_layers), 20.0),
        "doi": np.array([50.0]),
    }
    data = ThemeData(xs=xs, ys=ys, attributes=attributes)
    grid = build_theme(data, spec, grid_x, grid_y, n_layers)
    assert grid.shape == (grid_y.size, grid_x.size, n_layers)
    assert not np.isnan(grid).any()


def _paces_dict_4layer():
    d = dict(_PACES_19)
    d["range_model"] = [
        {"layers": "L01-L02", "range_by_complexity": [100, 500, 400, 250, 100]},
        {"layers": "L03-L04", "range_by_complexity": [500, 500, 500, 500, 500]},
    ]
    return d


def test_yaml_paces_matches_danish_helper():
    n_layers = 4
    grid_x = np.array([0.0, 100.0, 200.0])
    grid_y = np.array([0.0, 100.0])
    xs = np.array([100.0])
    ys = np.array([50.0])
    attributes = {"depth": np.array([[5.0, 12.0, 16.0, 30.0]])}
    complexity = np.full((grid_y.size, grid_x.size), 2.0)
    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=complexity)
    yaml_spec = load_spec(_paces_dict_4layer())
    danish = theme_specs.paces_spec(n_prereq=2)
    g_yaml = build_theme(data, yaml_spec, grid_x, grid_y, n_layers)
    g_danish = build_theme(data, danish, grid_x, grid_y, n_layers)
    np.testing.assert_array_equal(g_yaml, g_danish)


def test_load_spec_rejects_unknown_cert_fun():
    with pytest.raises(ValueError, match="cert_fun"):
        load_spec(dict(_PACES_19, cert_fun="not_a_kernel"))


def test_load_spec_rejects_missing_key():
    bad = {k: v for k, v in _PACES_19.items() if k != "var0"}
    with pytest.raises(ValueError, match="missing required"):
        load_spec(bad)


def test_load_spec_rejects_unsupported_combine():
    with pytest.raises(ValueError, match="combine"):
        load_spec(dict(_PACES_19, combine="sum_n_nearest"))


def test_compile_expr_evaluates_over_attributes():
    fn = compile_expr("(0.5 * maximum(2, 0.25*depth))**2", ["depth"])
    out = fn({"depth": np.array([0.0, 40.0])})
    np.testing.assert_allclose(out, [(0.5 * 2) ** 2, (0.5 * 10.0) ** 2])
    assert fn.expr == "(0.5 * maximum(2, 0.25*depth))**2"


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "depth.real",
        "depth[0]",
        "foo(depth)",
        "unknown_attr + 1",
        "'a string'",
    ],
)
def test_compile_expr_rejects_unsafe(expr):
    with pytest.raises(ValueError):
        compile_expr(expr, ["depth"])


def test_mask_expression_drives_nodata():
    spec = load_spec(_PACES_19)
    n_layers = 19
    grid_x = np.array([0.0, 100.0, 200.0])
    grid_y = np.array([0.0, 100.0])
    xs = np.array([100.0])
    ys = np.array([50.0])
    depth = np.full((1, n_layers), 5.0)
    depth[0, 1] = 25.0
    data = ThemeData(xs=xs, ys=ys, attributes={"depth": depth},
                     complexity=np.full((2, 3), 2.0))
    grid = build_theme(data, spec, grid_x, grid_y, n_layers)
    assert grid[1, 1, 0] < NODATA_VARIANCE
    assert grid[1, 1, 1] == NODATA_VARIANCE
