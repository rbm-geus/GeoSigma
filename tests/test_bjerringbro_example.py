"""The Bjerringbro example's Quaternary / pre-Quaternary split (``n_prereq``).

Model layers 1-10 are Quaternary and 11-15 pre-Quaternary
(``examples/bjerringbro/data/FohmLayers.csv``). The surfaces are layer
bottoms, so the bottom of layer 10 (base of the Quaternary) must get the
Quaternary ranges and the bottom of layer 11 the pre-Quaternary ranges.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geosigma.config import RegionConfig
from geosigma.themes import resolve_theme_spec
from geosigma.themes.base import _range_map
from geosigma.themes.themes import _COMP2RANGE, _COMP2RANGE_PREQ

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "bjerringbro"


def _workflow():
    spec = importlib.util.spec_from_file_location(
        "bjerringbro_workflow", EXAMPLE / "geosigma_dk_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow_config():
    return _workflow().CONFIG


def _stack_index(layer_no):
    """0-based variance-stack index of the bottom of model layer ``layer_no``."""
    cfg = RegionConfig.from_file(EXAMPLE / "region_config.json")
    return cfg.modelled_layer_names.index(f"layer_{layer_no}_bottom")


def test_layer_table_puts_the_split_between_layers_10_and_11():
    layers = pd.read_csv(EXAMPLE / "data" / "FohmLayers.csv", skipinitialspace=True)
    quaternary = layers["Navn"].str.startswith("Kvartaer")
    assert layers.loc[quaternary, "Modellag"].tolist() == list(range(2, 11))
    assert layers.loc[~quaternary, "Modellag"].tolist() == [1, 11, 12, 13, 14, 15]
    # Layer 1 is the soil layer (Jordbund), above the Quaternary deposits.
    assert layers.loc[layers["Modellag"] == 1, "Navn"].item() == "Jordbund"


def test_n_prereq_is_the_stack_index_of_the_bottom_of_layer_11():
    assert _stack_index(10) == 9
    assert _stack_index(11) == 10
    assert _workflow_config()["n_prereq"] == _stack_index(11)


@pytest.mark.parametrize("spec_name", ["skytem", "fewtem", "paces"])
def test_bottom_of_layer_10_quaternary_bottom_of_layer_11_pre_quaternary(spec_name):
    n_prereq = _workflow_config()["n_prereq"]
    spec = resolve_theme_spec(spec_name, n_prereq)
    complexity = np.array([[1, 2, 3, 4]])  # every complexity class once
    ranges = _range_map(complexity, spec.range_model, n_layers=15)

    quaternary = np.asarray(_COMP2RANGE, dtype=float)[1:]
    pre_quaternary = np.asarray(_COMP2RANGE_PREQ, dtype=float)[1:]
    assert not np.array_equal(quaternary, pre_quaternary)  # the test can tell

    np.testing.assert_array_equal(ranges[0, :, _stack_index(10)], quaternary)
    np.testing.assert_array_equal(ranges[0, :, _stack_index(11)], pre_quaternary)
    # And the regimes hold over the whole stack.
    for i in range(_stack_index(10) + 1):
        np.testing.assert_array_equal(ranges[0, :, i], quaternary)
    for i in range(_stack_index(11), 15):
        np.testing.assert_array_equal(ranges[0, :, i], pre_quaternary)


# --------------------------------------------------------------------------- #
# Complexity class order
# --------------------------------------------------------------------------- #
def test_config_declares_the_danish_map_as_higher_is_more_complex():
    conf = _workflow_config()
    assert conf["complexity_grid"]
    assert conf["complexity_order"] == "higher_is_more_complex"


def test_missing_complexity_order_is_an_error():
    wf = _workflow()
    conf = wf.resolve_paths(wf.CONFIG)
    conf.pop("complexity_order")
    with pytest.raises(ValueError, match="complexity_order"):
        wf.load_support_grids(conf, np.zeros((1, 1)), None)


def _crop_module():
    spec = importlib.util.spec_from_file_location(
        "crop_complexity", EXAMPLE / "data_prep" / "crop_complexity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _meta(x0, y0, nx, ny, cell):
    return {
        "xllcorner": x0,
        "yllcorner": y0,
        "ncols": nx,
        "nrows": ny,
        "dx": cell,
        "dy": cell,
    }


def test_shipped_complexity_map_lies_on_the_model_grid():
    from utils import grids_aligned, load_geotiff_grid

    wf = _workflow()
    conf = wf.resolve_paths(wf.CONFIG)
    _, meta, *_ = wf.load_model(conf["region_config"])
    _, m = load_geotiff_grid(conf["complexity_grid"])
    assert grids_aligned(m, meta)


def test_workflow_rejects_a_map_that_is_not_on_the_model_grid(tmp_path):
    # GeoSigma does not adapt the map: a misaligned one stops the run.
    import rasterio
    from rasterio.transform import from_origin

    wf = _workflow()
    conf = wf.resolve_paths(wf.CONFIG)
    _, meta, _, _, terrain, _ = wf.load_model(conf["region_config"])
    path = tmp_path / "shifted.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=meta["nrows"],
        width=meta["ncols"],
        count=1,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_origin(
            meta["xllcorner"] - 50,
            meta["yllcorner"] + meta["nrows"] * meta["dy"] - 50,
            meta["dx"],
            meta["dy"],
        ),
        nodata=0,
    ) as ds:
        ds.write(np.full((meta["nrows"], meta["ncols"]), 3, dtype="uint8"), 1)
    conf["complexity_grid"] = str(path)
    with pytest.raises(ValueError, match="not aligned.*crop_complexity"):
        wf.load_support_grids(conf, terrain, meta)


def test_crop_same_cell_size_shifts_half_a_cell_like_matlab():
    # Map edges 50 m off the model's: each model cell gets the map cell 50 m
    # to its south-west (MATLAB import_complexitymap: +50 m), copied as is.
    crop = _crop_module()
    model = _meta(1000.0, 2000.0, 3, 2, 100.0)
    src_meta = _meta(950.0, 1950.0, 4, 3, 100.0)
    src = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2]])  # north-up
    out, how = crop.to_model_grid(src, src_meta, model, "higher_is_more_complex")
    # Model row 0 (top, 2100-2200) <- map row 1 (2050-2150), columns 0..2.
    np.testing.assert_array_equal(out, [[2, 3, 4], [3, 4, 1]])
    assert "+50 m east, +50 m north" in how


def test_crop_reverses_lower_is_more_complex():
    crop = _crop_module()
    model = _meta(0.0, 0.0, 3, 1, 100.0)
    out, _ = crop.to_model_grid(
        np.array([[0, 1, 4]]), model, model, "lower_is_more_complex", n_classes=4
    )
    np.testing.assert_array_equal(out, [[0, 4, 1]])


def test_crop_vote_ties_go_to_the_most_complex_class_in_either_order():
    # A 50 m map under a 100 m model: each model cell covers two columns of
    # classes 2 | 3, a tie that must give the more complex class (3) whichever
    # way the file is coded.
    crop = _crop_module()
    model = _meta(0.0, 0.0, 2, 2, 100.0)
    fine = _meta(0.0, 0.0, 4, 4, 50.0)
    z = np.tile([2, 3, 2, 3], (4, 1))
    out, how = crop.to_model_grid(z, fine, model, "higher_is_more_complex")
    np.testing.assert_array_equal(out, np.full((2, 2), 3))
    assert "majority vote" in how
    out, _ = crop.to_model_grid(
        5 - z, fine, model, "lower_is_more_complex", n_classes=4
    )
    np.testing.assert_array_equal(out, np.full((2, 2), 3))


def test_crop_fills_model_cells_outside_the_map():
    crop = _crop_module()
    model = _meta(0.0, 0.0, 3, 1, 100.0)
    src_meta = _meta(0.0, 0.0, 2, 1, 100.0)
    out, _ = crop.to_model_grid(
        np.array([[2, 0]]), src_meta, model, "higher_is_more_complex", fill_class=1
    )
    np.testing.assert_array_equal(out, [[2, 1, 1]])


def test_gerda_converter_requires_an_order_for_a_grid_path():
    spec = importlib.util.spec_from_file_location(
        "gerda_to_themedata", EXAMPLE / "data_prep" / "gerda_to_themedata.py"
    )
    g2t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g2t)
    decl = g2t.complexity_declaration
    assert decl("none") == "none"
    assert decl("external") == "external"
    assert (
        decl("c.tif", "higher_is_more_complex")
        == "{path: c.tif, order: higher_is_more_complex}"
    )
    assert (
        decl("c.tif", "lower_is_more_complex", 4)
        == "{path: c.tif, order: lower_is_more_complex, n_classes: 4}"
    )
    with pytest.raises(ValueError, match="complexity-order"):
        decl("c.tif")
    with pytest.raises(ValueError, match="n-classes"):
        decl("c.tif", "lower_is_more_complex")
    with pytest.raises(ValueError, match="only apply"):
        decl("none", "higher_is_more_complex")
