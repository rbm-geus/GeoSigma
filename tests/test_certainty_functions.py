# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""The recommended certainty kernel and the legacy MATLAB kernels."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from geosigma.themes import (
    LEGACY_CERTAINTY_FUNCTIONS,
    ThemeData,
    build_theme,
    certainty_function,
)
from geosigma.themes import themes as theme_specs

WIDTHS = (0.0, 30.0, 75.0, 150.0, 400.0)
RANGES = (50.0, 100.0, 250.0, 500.0)
SILL = 2.5


def _kwpt():
    return certainty_function("kernel_with_point_tail")


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("range_", RANGES)
def test_kernel_with_point_tail_is_sill_at_the_data_point(width, range_):
    assert _kwpt()(np.array([0.0]), range_, width, SILL)[0] == pytest.approx(SILL)


@pytest.mark.parametrize("width", WIDTHS[1:])
@pytest.mark.parametrize("range_", RANGES)
def test_kernel_with_point_tail_is_continuous_at_the_edge(width, range_):
    f = _kwpt()
    at = f(np.array([width]), range_, width, SILL)[0]
    just_after = f(np.array([np.nextafter(width, np.inf)]), range_, width, SILL)[0]
    assert just_after == pytest.approx(at, rel=1e-12)
    # ... while the legacy RBM_oct2023 jumps there by a (1 - a)^2 * sill
    rbm = certainty_function("RBM_oct2023", allow_legacy=True)
    a = np.exp(-1.5 * width**2 / range_**2)
    jump = (
        rbm(np.array([width]), range_, width, SILL)[0]
        - rbm(np.array([np.nextafter(width, np.inf)]), range_, width, SILL)[0]
    )
    assert jump == pytest.approx(a * (1 - a) ** 2 * SILL, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("range_", RANGES)
def test_kernel_with_point_tail_decreases_everywhere(width, range_):
    d = np.linspace(0.0, 3.0 * max(range_, width), 4001)
    y = _kwpt()(d, range_, width, SILL)
    assert np.all(np.diff(y) <= 0)
    assert np.all(y > 0) or np.all(y[y <= 0] == 0)  # underflow only


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("range_", RANGES)
def test_kernel_with_point_tail_equals_rbm_oct2023_inside_the_kernel(width, range_):
    d = np.linspace(0.0, width, 501)
    rbm = certainty_function("RBM_oct2023", allow_legacy=True)
    np.testing.assert_array_equal(
        _kwpt()(d, range_, width, SILL), rbm(d, range_, width, SILL)
    )


def test_kernel_with_point_tail_slope_doubles_at_the_edge():
    """The value is continuous at the edge; the slope is not."""
    w, r, h = 150.0, 250.0, 1e-4
    f = _kwpt()
    left = (f(np.array([w]), r, w, 1.0) - f(np.array([w - h]), r, w, 1.0))[0] / h
    right = (f(np.array([w + h]), r, w, 1.0) - f(np.array([w]), r, w, 1.0))[0] / h
    assert right == pytest.approx(2.0 * left, rel=1e-3)


def test_kernel_with_point_tail_propagates_nan_like_the_engine_expects():
    y = _kwpt()(np.array([np.nan, 10.0]), np.array([100.0, np.nan]), 75.0, 1.0)
    assert np.isnan(y).all()


@pytest.mark.parametrize("name", sorted(LEGACY_CERTAINTY_FUNCTIONS))
def test_legacy_kernels_need_the_flag(name):
    with pytest.raises(ValueError, match="legacy MATLAB kernel") as exc:
        certainty_function(name)
    assert "kernel_with_point_tail" in str(exc.value)
    assert "ILM_oct2023" in str(exc.value)
    assert callable(certainty_function(name, allow_legacy=True))


def test_legacy_set_is_exactly_the_two_matlab_kernels():
    assert set(LEGACY_CERTAINTY_FUNCTIONS) == {"ILM_sep2023", "RBM_oct2023"}
    for name in ("kernel_with_point_tail", "FRAFA_apr2023", "ILM_oct2023"):
        certainty_function(name)  # no flag needed


DANISH_SPECS = {
    "pacep": lambda: theme_specs.pacep_spec(n_prereq=2),
    "mep": lambda: theme_specs.mep_spec(n_prereq=2, doi_fill=50.0),
    "gammalog": lambda: theme_specs.gammalog_spec(n_prereq=2),
    "reslog": lambda: theme_specs.reslog_spec(n_prereq=2),
    "skytem": lambda: theme_specs.skytem_spec(n_prereq=2),
    "fewtem": lambda: theme_specs.fewtem_spec(n_prereq=2),
    "manytem": lambda: theme_specs.manytem_spec(n_prereq=2),
    "ttem": lambda: theme_specs.ttem_spec(n_prereq=2),
}


@pytest.mark.parametrize("key", sorted(DANISH_SPECS))
def test_danish_specs_use_the_recommended_kernel(key):
    spec = DANISH_SPECS[key]()
    assert spec.cert_fun_name == "kernel_with_point_tail"
    assert spec.allow_legacy_cert_fun is False


def test_paces_and_refseis_keep_frafa():
    assert theme_specs.paces_spec(n_prereq=2).cert_fun_name == "FRAFA_apr2023"
    spec = theme_specs.refseis_spec(n_prereq=2, active_layers=[0])
    assert spec.cert_fun_name == "FRAFA_apr2023"


def _tiny_case():
    grid_x = np.arange(5) * 100.0
    grid_y = np.arange(4) * 100.0
    data = ThemeData(
        xs=np.array([150.0]),
        ys=np.array([150.0]),
        attributes={"depth": np.array([[20.0, 40.0]])},
        complexity=np.full((4, 5), 2.0),
    )
    return data, grid_x, grid_y


def test_build_theme_refuses_a_legacy_kernel_without_the_flag():
    data, gx, gy = _tiny_case()
    spec = dataclasses.replace(
        theme_specs.paces_spec(n_prereq=1), cert_fun_name="ILM_sep2023"
    )
    with pytest.raises(ValueError, match="allow_legacy_cert_fun"):
        build_theme(data, spec, gx, gy, 2)
    ok = dataclasses.replace(spec, allow_legacy_cert_fun=True)
    assert build_theme(data, ok, gx, gy, 2).shape == (4, 5, 2)
