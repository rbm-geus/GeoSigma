# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for the three-file ThemeData loader (``geosigma.themes.data_io``).

Fixtures are tiny and fully synthetic — no Danish sample data — following the
pattern in ``test_spec_io.py``/``test_themes.py``: hand-built arrays written to a
manifest + CSV in ``tmp_path``, round-tripped back through the loader.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from geosigma.themes import (
    LoadedTheme,
    ThemeData,
    build_theme,
    load_theme_data,
    resolve_theme_spec,
)

# A minimal synthetic theme: 3 points, 3 modelled layers, one per-layer block
# (``depth``) and one per-point scalar (``doi``).
_XS = np.array([150.0, 250.0, 90.0])
_YS = np.array([110.0, 60.0, 210.0])
_DEPTH = np.array(
    [
        [4.0, 9.0, 14.0],
        [5.0, 11.0, 20.0],
        [3.0, 7.0, 12.0],
    ]
)
_DOI = np.array([50.0, np.nan, 33.0])
_N_LAYERS = 3


def _write_theme(
    tmp_path,
    *,
    complexity="external",
    spec="paces",
    attributes=None,
    n_layers=_N_LAYERS,
    x_column="x",
    y_column="y",
    spec_params=None,
    drop_key=None,
    csv_overrides=None,
    depth_cols=("depth__L01", "depth__L02", "depth__L03"),
):
    """Write a synthetic manifest + points CSV; return the manifest path."""
    import pandas as pd

    if attributes is None:
        attributes = {
            "depth": {"per_layer": True, "prefix": "depth__L"},
            "doi": {"per_layer": False, "column": "doi"},
        }

    cols = {x_column: _XS, y_column: _YS, "doi": _DOI}
    for j, name in enumerate(depth_cols):
        # Tile the 3-column fixture so callers can request any number of
        # per-layer columns (width/count tests need up to 10); values are
        # irrelevant to those structural checks.
        cols[name] = _DEPTH[:, j % _DEPTH.shape[1]]
    if csv_overrides:
        cols.update(csv_overrides)
    csv_path = tmp_path / "points.csv"
    pd.DataFrame(cols).to_csv(csv_path, index=False)

    man = {
        "name": spec.upper(),
        "spec": spec,
        "points_csv": "points.csv",
        "n_layers": n_layers,
        "crs": "EPSG:25832",
        "x_column": x_column,
        "y_column": y_column,
        "attributes": attributes,
        "complexity": complexity,
    }
    if spec_params is not None:
        man["spec_params"] = spec_params
    if drop_key is not None:
        man.pop(drop_key)

    man_path = tmp_path / "theme.manifest.yaml"
    with open(man_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(man, fh, sort_keys=False)
    return man_path


def _write_esri_grid(path, *, ncols, nrows, cellsize, xll, yll, value=2.0):
    """Write a tiny ESRI ASCII grid of constant ``value``."""
    lines = [
        f"ncols {ncols}",
        f"nrows {nrows}",
        f"xllcorner {xll}",
        f"yllcorner {yll}",
        f"cellsize {cellsize}",
        "NODATA_value -9999",
    ]
    for _ in range(nrows):
        lines.append(" ".join(str(float(value)) for _ in range(ncols)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Happy-path round trip
# --------------------------------------------------------------------------- #


def test_round_trip_reassembles_arrays(tmp_path):
    man = _write_theme(tmp_path)
    complexity = np.full((2, 3), 2.0)
    loaded = load_theme_data(man, complexity=complexity)

    assert isinstance(loaded, LoadedTheme)
    assert isinstance(loaded.data, ThemeData)
    assert loaded.spec_name == "paces"
    assert loaded.n_layers == _N_LAYERS
    assert loaded.name == "PACES"
    assert loaded.crs == "EPSG:25832"
    assert loaded.spec_params == {}

    np.testing.assert_array_equal(loaded.data.xs, _XS)
    np.testing.assert_array_equal(loaded.data.ys, _YS)
    np.testing.assert_array_equal(loaded.data.attributes["depth"], _DEPTH)
    # Per-point scalar: NaN preserved (not filled at load time).
    np.testing.assert_array_equal(loaded.data.attributes["doi"], _DOI)
    assert np.isnan(loaded.data.attributes["doi"][1])
    np.testing.assert_array_equal(loaded.data.complexity, complexity)


def test_loaded_theme_builds_a_grid(tmp_path):
    man = _write_theme(tmp_path)
    loaded = load_theme_data(man, complexity=np.full((2, 3), 2.0))
    spec = resolve_theme_spec(loaded.spec_name, n_prereq=2)
    grid_x = np.array([0.0, 100.0, 200.0])
    grid_y = np.array([0.0, 100.0])
    grid = build_theme(loaded.data, spec, grid_x, grid_y, loaded.n_layers)
    assert grid.shape == (grid_y.size, grid_x.size, loaded.n_layers)


# --------------------------------------------------------------------------- #
# Complexity contract
# --------------------------------------------------------------------------- #


def test_external_requires_complexity_argument(tmp_path):
    man = _write_theme(tmp_path, complexity="external")
    with pytest.raises(ValueError, match="external"):
        load_theme_data(man)  # no complexity= supplied


def test_none_declares_uniform_and_keeps_complexity_none(tmp_path):
    man = _write_theme(tmp_path, complexity="none")
    loaded = load_theme_data(man)
    assert loaded.data.complexity is None


def test_none_rejects_supplied_complexity(tmp_path):
    man = _write_theme(tmp_path, complexity="none")
    with pytest.raises(ValueError, match="no grid"):
        load_theme_data(man, complexity=np.zeros((2, 3)))


def test_null_complexity_is_an_error(tmp_path):
    man = _write_theme(tmp_path, complexity=None)
    with pytest.raises(ValueError, match="unset .missing or null."):
        load_theme_data(man)


def test_missing_complexity_key_gets_the_options_message(tmp_path):
    # A manifest with no complexity key at all must get the explicit three-option
    # guidance, not the generic "missing required key" message.
    man = _write_theme(tmp_path, drop_key="complexity")
    with pytest.raises(ValueError, match="grid path.*external.*none"):
        load_theme_data(man)


def test_complexity_path_loads_and_aligns(tmp_path):
    from utils import make_grid_meta

    grid = tmp_path / "complexity.asc"
    _write_esri_grid(grid, ncols=3, nrows=2, cellsize=100.0, xll=0.0, yll=0.0)
    man = _write_theme(tmp_path, complexity={"path": "complexity.asc"})
    grid_meta = make_grid_meta(
        ncols=3, nrows=2, dx=100.0, dy=100.0, xllcorner=0.0, yllcorner=0.0
    )
    loaded = load_theme_data(man, grid_meta=grid_meta)
    assert loaded.data.complexity.shape == (2, 3)
    assert np.all(loaded.data.complexity == 2.0)


def test_complexity_path_misaligned_raises(tmp_path):
    from utils import make_grid_meta

    grid = tmp_path / "complexity.asc"
    _write_esri_grid(grid, ncols=3, nrows=2, cellsize=100.0, xll=0.0, yll=0.0)
    man = _write_theme(tmp_path, complexity="complexity.asc")
    misaligned = make_grid_meta(
        ncols=3, nrows=2, dx=100.0, dy=100.0, xllcorner=999.0, yllcorner=0.0
    )
    with pytest.raises(ValueError, match="not aligned"):
        load_theme_data(man, grid_meta=misaligned)


def test_complexity_path_rejects_supplied_array(tmp_path):
    grid = tmp_path / "complexity.asc"
    _write_esri_grid(grid, ncols=3, nrows=2, cellsize=100.0, xll=0.0, yll=0.0)
    man = _write_theme(tmp_path, complexity="complexity.asc")
    with pytest.raises(ValueError, match="also supplied"):
        load_theme_data(man, complexity=np.zeros((2, 3)))


# --------------------------------------------------------------------------- #
# Manifest / CSV validation errors
# --------------------------------------------------------------------------- #


def test_missing_required_key_raises(tmp_path):
    man = _write_theme(tmp_path, drop_key="spec")
    with pytest.raises(ValueError, match="missing required key"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


def test_missing_points_csv_raises(tmp_path):
    man = _write_theme(tmp_path)
    (tmp_path / "points.csv").unlink()
    with pytest.raises(FileNotFoundError, match="points CSV not found"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


def test_missing_coordinate_column_raises(tmp_path):
    # CSV keeps its 'x' column; point the manifest at a non-existent 'easting'.
    man = _write_theme(tmp_path)
    doc = yaml.safe_load(man.read_text(encoding="utf-8"))
    doc["x_column"] = "easting"
    man.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="x_column"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


def test_per_layer_count_mismatch_raises(tmp_path):
    # CSV carries only 2 depth columns but the manifest declares n_layers=3.
    man = _write_theme(tmp_path, depth_cols=("depth__L01", "depth__L02"))
    with pytest.raises(ValueError, match="n_layers=3"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


def test_missing_per_point_column_raises(tmp_path):
    attrs = {
        "depth": {"per_layer": True, "prefix": "depth__L"},
        "doi": {"per_layer": False, "column": "not_a_column"},
    }
    man = _write_theme(tmp_path, attributes=attrs)
    with pytest.raises(ValueError, match="absent from the points CSV"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


def test_per_layer_prefix_matches_nothing_raises(tmp_path):
    attrs = {"depth": {"per_layer": True, "prefix": "nope__L"}}
    man = _write_theme(tmp_path, attributes=attrs)
    with pytest.raises(ValueError, match="matched no CSV columns"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


# --------------------------------------------------------------------------- #
# Per-layer label-width rule (zero-padding enforced, not just documented)
# --------------------------------------------------------------------------- #


def test_unpadded_labels_raise_with_guidance(tmp_path):
    # L1..L10 (10 columns) pass the count check but sort L1, L10, L2, ... —
    # silent mis-assignment. The loader must catch the mixed width.
    depth_cols = tuple(f"depth__L{i}" for i in range(1, 11))
    man = _write_theme(
        tmp_path,
        n_layers=10,
        depth_cols=depth_cols,
        attributes={"depth": {"per_layer": True, "prefix": "depth__L"}},
    )
    with pytest.raises(ValueError, match="zero-padded to a constant width"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


def test_padded_labels_load(tmp_path):
    # L01..L10, constant width 2 — sorts in layer order, loads.
    depth_cols = tuple(f"depth__L{i:02d}" for i in range(1, 11))
    man = _write_theme(
        tmp_path,
        n_layers=10,
        depth_cols=depth_cols,
        attributes={"depth": {"per_layer": True, "prefix": "depth__L"}},
    )
    loaded = load_theme_data(man, complexity=np.full((2, 3), 2.0))
    assert loaded.data.attributes["depth"].shape == (3, 10)


def test_mixed_width_labels_raise(tmp_path):
    # A deliberately mixed set (widths 2 and 3) — inconsistent, must raise.
    depth_cols = ("depth__L01", "depth__L02", "depth__L003")
    man = _write_theme(
        tmp_path,
        n_layers=3,
        depth_cols=depth_cols,
        attributes={"depth": {"per_layer": True, "prefix": "depth__L"}},
    )
    with pytest.raises(ValueError, match="inconsistent width"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


# --------------------------------------------------------------------------- #
# format_version
# --------------------------------------------------------------------------- #


def test_format_version_explicit_one_accepted(tmp_path):
    man = _write_theme(tmp_path)
    doc = yaml.safe_load(man.read_text(encoding="utf-8"))
    doc["format_version"] = 1
    man.write_text(yaml.safe_dump(doc), encoding="utf-8")
    loaded = load_theme_data(man, complexity=np.full((2, 3), 2.0))
    assert loaded.spec_name == "paces"


def test_format_version_unsupported_raises(tmp_path):
    man = _write_theme(tmp_path)
    doc = yaml.safe_load(man.read_text(encoding="utf-8"))
    doc["format_version"] = 2
    man.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported format_version"):
        load_theme_data(man, complexity=np.full((2, 3), 2.0))


def test_format_version_absent_defaults_to_one(tmp_path):
    # The synthetic manifest carries no format_version; it must load as v1.
    man = _write_theme(tmp_path)
    doc = yaml.safe_load(man.read_text(encoding="utf-8"))
    assert "format_version" not in doc
    loaded = load_theme_data(man, complexity=np.full((2, 3), 2.0))
    assert loaded.spec_name == "paces"


# --------------------------------------------------------------------------- #
# Spec resolution
# --------------------------------------------------------------------------- #


def test_resolve_unknown_spec_raises():
    with pytest.raises(ValueError, match="unknown theme spec"):
        resolve_theme_spec("not_a_theme", n_prereq=2)


def test_resolve_paces_no_params():
    spec = resolve_theme_spec("paces", n_prereq=2)
    assert spec.name == "PACES"


def test_resolve_refseis_active_layers_from_spec_params(tmp_path):
    man = _write_theme(
        tmp_path,
        spec="refseis",
        attributes={"thick": {"per_layer": True, "prefix": "depth__L"}},
        spec_params={"active_layers": [0, 2]},
    )
    loaded = load_theme_data(man, complexity=np.full((2, 3), 2.0))
    assert loaded.spec_params == {"active_layers": [0, 2]}
    spec = resolve_theme_spec(loaded.spec_name, 2, loaded.spec_params)
    assert list(spec.active_layers) == [0, 2]


def test_resolve_mep_doi_fill_from_spec_params():
    spec = resolve_theme_spec("mep", n_prereq=2, spec_params={"doi_fill": 42.0})
    # doi_fill is baked into the post_reduce_fn; exercise it on a NaN doi grid.
    out = spec.post_reduce_fn({"doi": np.array([[np.nan, 1.0]])})
    np.testing.assert_array_equal(out["doi"], [[42.0, 1.0]])


def test_resolve_bad_spec_params_raises():
    with pytest.raises(ValueError, match="rejected spec_params"):
        resolve_theme_spec("paces", n_prereq=2, spec_params={"nope": 1})
