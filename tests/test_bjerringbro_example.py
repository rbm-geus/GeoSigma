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


def _write_asc(path, z, meta):
    """ESRI ASCII grid (north-up rows) on the model grid's corner and cell size."""
    lines = [
        f"ncols {z.shape[1]}",
        f"nrows {z.shape[0]}",
        f"xllcorner {meta['xllcorner']}",
        f"yllcorner {meta['yllcorner']}",
        f"cellsize {meta['dx']}",
        "NODATA_value -9999",
    ]
    lines += [" ".join(str(int(v)) for v in row) for row in z]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_lower_is_more_complex_mirrors_classes_on_an_aligned_grid(tmp_path):
    # A grid whose cells coincide with the model cells has no tied votes, so
    # reversing the order must mirror every class (0 = unknown stays 0).
    wf = _workflow()
    conf = wf.resolve_paths(wf.CONFIG)
    _, meta, *_ = wf.load_model(conf["region_config"])
    rng = np.random.default_rng(0)
    z = rng.integers(0, 5, size=(meta["nrows"], meta["ncols"]))
    path = tmp_path / "classes.asc"
    _write_asc(path, z, meta)
    higher = wf.class_grid_to_model(path, meta, "c", order="higher_is_more_complex")
    lower = wf.class_grid_to_model(
        path, meta, "c", order="lower_is_more_complex", n_classes=4
    )
    np.testing.assert_array_equal(higher, np.flipud(z))
    np.testing.assert_array_equal(lower, np.where(higher > 0, 5 - higher, 0))


def test_tied_votes_go_to_the_more_complex_class_in_either_order(tmp_path):
    # Shift a two-class grid by half a cell so every model cell straddles a
    # 2 | 3 boundary: the tie must give the more complex class (internal 3)
    # whichever way the file is coded.
    wf = _workflow()
    conf = wf.resolve_paths(wf.CONFIG)
    _, meta, *_ = wf.load_model(conf["region_config"])
    shifted = dict(meta, xllcorner=meta["xllcorner"] - meta["dx"] / 2)
    ncols = meta["ncols"] + 1
    row = np.where(np.arange(ncols) % 2 == 0, 2, 3)
    z = np.tile(row, (meta["nrows"], 1))
    path = tmp_path / "ties.asc"
    _write_asc(path, z, shifted)
    higher = wf.class_grid_to_model(path, meta, "c", order="higher_is_more_complex")
    assert np.all(higher == 3)
    # Same geology coded the other way round (file 3 = less complex).
    _write_asc(path, 5 - z, shifted)
    lower = wf.class_grid_to_model(
        path, meta, "c", order="lower_is_more_complex", n_classes=4
    )
    assert np.all(lower == 3)


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
