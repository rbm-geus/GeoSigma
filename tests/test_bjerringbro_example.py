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


def _workflow_config():
    spec = importlib.util.spec_from_file_location(
        "bjerringbro_workflow", EXAMPLE / "geosigma_dk_workflow.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CONFIG


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
