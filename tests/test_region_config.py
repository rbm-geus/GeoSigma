# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Phase 2 tests: the model-agnostic RegionConfig registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from geosigma.config import RegionConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = (
    REPO_ROOT / "examples" / "data" / "Jutland_hydrostrat_model" / "region_config.json"
)


def _names(n):
    return ["top"] + [f"b{i}" for i in range(1, n)]


def test_default_modelled_layers_excludes_top_and_bottom():
    cfg = RegionConfig(name="x", layer_names=_names(5))  # top + 4 boundaries
    # has_bottom=True -> exclude index 0 (top) and last (bottom).
    assert cfg.modelled_layer_indices == [1, 2, 3]
    assert cfg.n_modelled_layers == 3
    assert cfg.modelled_layer_names == ["b1", "b2", "b3"]


def test_no_bottom_keeps_last_layer():
    cfg = RegionConfig(name="x", layer_names=_names(5), has_bottom=False)
    assert cfg.modelled_layer_indices == [1, 2, 3, 4]


def test_explicit_modelled_layers_override():
    cfg = RegionConfig(name="x", layer_names=_names(6), modelled_layers=[2, 4])
    assert cfg.modelled_layer_indices == [2, 4]


def test_explicit_modelled_layers_range_checked():
    with pytest.raises(ValueError):
        RegionConfig(name="x", layer_names=_names(4), modelled_layers=[9])


def test_duplicate_layer_names_rejected():
    with pytest.raises(ValueError):
        RegionConfig(name="x", layer_names=["top", "b1", "b1"])


def test_too_few_layers_rejected():
    with pytest.raises(ValueError):
        RegionConfig(name="x", layer_names=["only_top"])


def test_surface_path_by_index_and_name():
    cfg = RegionConfig(
        name="x", layer_names=_names(3), surface_dir="/data", file_ext=".tif"
    )
    assert cfg.surface_path(0) == Path("/data/top.tif")
    assert cfg.surface_path("b1") == Path("/data/b1.tif")


def test_surface_path_unknown_name_rejected():
    cfg = RegionConfig(name="x", layer_names=_names(3), surface_dir="/data")
    with pytest.raises(KeyError):
        cfg.surface_path("nope")


def test_surface_path_requires_dir():
    cfg = RegionConfig(name="x", layer_names=_names(3))
    with pytest.raises(ValueError):
        cfg.surface_path(0)


def test_dict_roundtrip():
    cfg = RegionConfig(
        name="x",
        layer_names=_names(4),
        well_files={"interpreted": "a.mat"},
        metadata={"preq_index": 2},
    )
    again = RegionConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError):
        RegionConfig.from_dict({"name": "x", "layer_names": _names(3), "bogus": 1})


def test_json_file_roundtrip(tmp_path):
    cfg = RegionConfig(name="x", layer_names=_names(4), crs="EPSG:25832")
    p = tmp_path / "cfg.json"
    cfg.save(p)
    loaded = RegionConfig.from_file(p)
    assert loaded.to_dict() == cfg.to_dict()


def test_example_config_loads():
    if not EXAMPLE_CONFIG.is_file():
        pytest.skip(f"example config not found at {EXAMPLE_CONFIG}")
    cfg = RegionConfig.from_file(EXAMPLE_CONFIG)
    assert cfg.layer_names[0] == "top_surface"
    assert cfg.n_interfaces == 16
    # has_bottom=False -> all 15 boundaries modelled.
    assert cfg.n_modelled_layers == 15
