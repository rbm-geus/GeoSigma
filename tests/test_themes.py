# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for the Phase 3 variance-map tool (``geosigma.themes``).

Three groups:

* **certainty kernels** — hand-computed small cases for each named kernel;
* **combine** — parallel-precision identity, the minimum-map floor, and the
  zero-thickness correlation map;
* **windowing equivalence (the trust anchor)** — a direct triple-loop port of
  the MATLAB window logic, asserted *identical* to the vectorised
  :func:`windowed_nearest`, both standalone and driven through the full
  :func:`build_theme` pipeline. This is what justifies the vectorisation.

Deterministic stages also get a golden ``.npz`` regression guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.themes import (
    NODATA_VARIANCE,
    ThemeData,
    apply_floor,
    available_certainty_functions,
    build_theme,
    certainty_function,
    combine_variances,
    corr_map,
    minimum_map,
    model_theme,
    register_certainty_function,
    windowed_nearest,
)
from geosigma.themes import certainty_functions as cf
from geosigma.themes import themes as theme_specs
from tests.golden import assert_matches_golden


@pytest.fixture
def restore_cert_registry():
    """Snapshot and restore the certainty-function registry around a test, so
    test registrations never leak into other tests."""
    saved = dict(cf._CERT_FUNCTIONS)
    try:
        yield
    finally:
        cf._CERT_FUNCTIONS.clear()
        cf._CERT_FUNCTIONS.update(saved)

# ---------------------------------------------------------------------------
# Certainty kernels
# ---------------------------------------------------------------------------


def test_frafa_plateau_then_gaussian():
    f = certainty_function("FRAFA_apr2023")
    dist = np.array([0.0, 5.0, 15.0])
    rng, width, sill = 10.0, 5.0, 2.0
    out = f(dist, rng, width, sill)
    # Inside/at the plateau width -> flat sill.
    assert out[0] == pytest.approx(sill)
    assert out[1] == pytest.approx(sill)
    # Outside -> sill * exp(-3 (d-w)^2 / range^2).
    expected = sill * np.exp(-3.0 * (15.0 - width) ** 2 / rng**2)
    assert out[2] == pytest.approx(expected)


def test_ilm_sep_zero_outside_width():
    f = certainty_function("ILM_sep2023")
    out = f(np.array([3.0, 10.0]), 8.0, 5.0, 1.5)
    assert out[0] == pytest.approx(1.5 * np.exp(-3.0 * (3.0 - 5.0) ** 2 / 8.0**2))
    assert out[1] == 0.0  # beyond width


def test_unknown_certainty_function_raises():
    with pytest.raises(ValueError):
        certainty_function("does_not_exist")


# ---------------------------------------------------------------------------
# register_certainty_function — the custom-kernel extension point
# ---------------------------------------------------------------------------


def _linear_kernel(dist, range_, width, sill):
    """A trivial well-formed custom kernel: linear falloff to zero at ``range_``."""
    return sill * np.clip(1.0 - dist / range_, 0.0, None)


def test_register_and_use_custom_kernel(restore_cert_registry):
    register_certainty_function("LINEAR_test", _linear_kernel)
    assert "LINEAR_test" in available_certainty_functions()
    fn = certainty_function("LINEAR_test")
    assert fn is _linear_kernel
    # And it actually evaluates as registered.
    out = fn(np.array([0.0, 5.0, 10.0]), 10.0, 0.0, 2.0)
    np.testing.assert_allclose(out, [2.0, 1.0, 0.0])


def test_register_rejects_non_callable(restore_cert_registry):
    with pytest.raises(TypeError, match="callable"):
        register_certainty_function("NOT_CALLABLE", 42)


def test_register_rejects_wrong_signature(restore_cert_registry):
    def bad(dist, range_, width):  # missing 'sill'
        return dist

    with pytest.raises(TypeError, match="signature"):
        register_certainty_function("BAD_SIG", bad)


def test_register_duplicate_name_raises(restore_cert_registry):
    # A built-in preset name is already taken.
    with pytest.raises(ValueError, match="already registered"):
        register_certainty_function("FRAFA_apr2023", _linear_kernel)


def test_register_duplicate_with_overwrite_succeeds(restore_cert_registry):
    register_certainty_function("FRAFA_apr2023", _linear_kernel, overwrite=True)
    assert certainty_function("FRAFA_apr2023") is _linear_kernel


# ---------------------------------------------------------------------------
# Combine / floor / minimum map / corr map
# ---------------------------------------------------------------------------


def test_combine_parallel_precision():
    a = np.full((2, 2, 1), 4.0)
    b = np.full((2, 2, 1), 4.0)
    # 1 / (1/4 + 1/4) = 2
    assert np.allclose(combine_variances([a, b]), 2.0)


def test_combine_ignores_infinite_theme():
    a = np.full((3, 3, 2), 5.0)
    b = np.full((3, 3, 2), np.inf)  # no information -> contributes nothing
    assert np.allclose(combine_variances([a, b]), 5.0)


def test_minimum_map_and_floor():
    terrain = np.array([[100.0]])
    bottom = np.array([[0.0]])  # depth 100 -> floor (0.01*100)^2 = 1.0
    mm = minimum_map(terrain, [bottom])
    assert mm.shape == (1, 1, 1)
    assert mm[0, 0, 0] == pytest.approx(1.0)
    # A variance below the floor is raised; one above is kept.
    assert apply_floor(np.array([[[0.2]]]), mm)[0, 0, 0] == pytest.approx(1.0)
    assert apply_floor(np.array([[[5.0]]]), mm)[0, 0, 0] == pytest.approx(5.0)


def test_corr_map_stamps_zero_thickness_wells():
    grid_x = np.array([0.0, 100.0, 200.0])
    grid_y = np.array([0.0, 100.0])
    # Two wells; only the first is zero-thickness.
    wx = np.array([150.0, 50.0])
    wy = np.array([150.0, 50.0])
    wq = np.array([3.0, 9.0])
    zt = np.array([True, False])
    grid = corr_map(grid_x, grid_y, wx, wy, wq, zt, dx=100.0, dy=100.0)
    assert grid.shape == (2, 3)
    # ceil((150-0)/100)-1 = 1 for both indices -> cell [1, 1] gets quality 3.
    assert grid[1, 1] == pytest.approx(3.0)
    # Everything else is NaN (no tie); the non-zero-thickness well is ignored.
    assert np.isnan(grid).sum() == 5


def test_model_theme_depth_ramp():
    # Single class, single layer; depth 100 m, default gradients/fmin.
    class_grid = np.zeros((1, 1), dtype=int)  # class 0 -> fmin 20
    terrain = np.array([[100.0]])
    bottom = np.array([[0.0]])
    theme = model_theme(class_grid, terrain, [bottom])
    # depth 100 >= 50: ramp = 0.15*100 + (0.3-0.15)*50 + 20 = 42.5; var = (0.5*42.5)^2
    assert theme[0, 0, 0] == pytest.approx((0.5 * 42.5) ** 2)


# ---------------------------------------------------------------------------
# Windowing equivalence — the trust anchor
# ---------------------------------------------------------------------------


def naive_windowed_nearest(
    xs, ys, grid_x, grid_y, search_radius, attributes, reducers, n_layers
):
    """Direct triple-loop port of the MATLAB theme window logic.

    Mirrors ``get_*_theme.m`` / the root ``get_*_theme.py``: for each grid cell,
    filter data points to the open ``±search_radius``-cell coordinate window,
    take the minimum distance, and reduce attributes over the minimum-distance
    ties. Deliberately un-vectorised — this is the reference the fast path must
    match.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    nx, ny = grid_x.size, grid_y.size
    R = int(search_radius)

    attrs = {}
    for name in attributes:
        arr = np.asarray(attributes[name], dtype=float)
        if arr.ndim == 1:
            arr = np.repeat(arr[:, None], n_layers, axis=1)
        attrs[name] = arr

    local_dist = np.full((ny, nx), np.nan)
    reduced = {name: np.full((ny, nx, n_layers), np.nan) for name in attributes}

    for i in range(ny):
        ylo = grid_y[max(0, i - R)]
        yhi = grid_y[min(ny - 1, i + R)]
        filt1 = (ys < yhi) & (ys > ylo)
        for j in range(nx):
            xlo = grid_x[max(0, j - R)]
            xhi = grid_x[min(nx - 1, j + R)]
            filt2 = (xs < xhi) & (xs > xlo)
            filt = filt1 & filt2
            if not filt.any():
                continue
            idx = np.nonzero(filt)[0]
            d = np.sqrt((grid_x[j] - xs[idx]) ** 2 + (grid_y[i] - ys[idx]) ** 2)
            dmin = d.min()
            local_dist[i, j] = dmin
            tie = idx[d == dmin]
            for name in attributes:
                red = reducers[name]
                if red == "first":
                    # First point in the WINDOW (lowest index), not the tie set —
                    # MATLAB ``local_X(i,j) = X(filt)(1)``.
                    reduced[name][i, j] = attrs[name][idx[0]]
                    continue
                vals = attrs[name][tie]
                if red == "mean":
                    reduced[name][i, j] = vals.mean(axis=0)
                elif red == "min":
                    reduced[name][i, j] = vals.min(axis=0)
                else:
                    reduced[name][i, j] = vals.max(axis=0)
    return local_dist, reduced


def _synthetic_case(seed=0):
    """A small synthetic theme problem with a deliberate distance tie.

    Returns coordinate axes, point coords, a per-point/per-layer attribute set,
    and a complexity grid — enough to exercise every reducer, edge cells, and a
    tie.
    """
    rng = np.random.default_rng(seed)
    nx, ny, n_layers = 8, 6, 3
    dx = 100.0
    grid_x = np.arange(nx) * dx + 500_000.0
    grid_y = np.arange(ny) * dx + 6_100_000.0

    # Random scattered points across (and slightly beyond) the grid extent.
    npts = 20
    xs = rng.uniform(grid_x[0] - dx, grid_x[-1] + dx, npts)
    ys = rng.uniform(grid_y[0] - dx, grid_y[-1] + dx, npts)
    # Add a symmetric pair straddling cell (iy=3, ix=4) -> identical distance,
    # forcing the tie-reduction path.
    cx, cy = grid_x[4], grid_y[3]
    xs = np.append(xs, [cx - 10.0, cx + 10.0])
    ys = np.append(ys, [cy, cy])
    npts = xs.size

    attributes = {
        "depth": rng.uniform(1.0, 120.0, (npts, n_layers)),
        "thick": rng.uniform(0.5, 60.0, (npts, n_layers)),
        "doi": rng.uniform(5.0, 100.0, npts),  # 1-D, broadcast across layers
    }
    reducers = {"depth": "mean", "thick": "min", "doi": "max"}

    complexity = rng.integers(0, 5, size=(ny, nx)).astype(float)
    return grid_x, grid_y, xs, ys, attributes, reducers, complexity, n_layers


@pytest.mark.parametrize("search_radius", [1, 3, 8])
def test_windowed_nearest_matches_naive_loop(search_radius):
    """Vectorised windowing == direct MATLAB-style loop (dist + every reducer)."""
    grid_x, grid_y, xs, ys, attributes, reducers, _, n_layers = _synthetic_case()

    fast_dist, fast_attrs = windowed_nearest(
        xs, ys, grid_x, grid_y, search_radius, attributes, reducers, n_layers
    )
    ref_dist, ref_attrs = naive_windowed_nearest(
        xs, ys, grid_x, grid_y, search_radius, attributes, reducers, n_layers
    )

    np.testing.assert_array_equal(np.isnan(fast_dist), np.isnan(ref_dist))
    np.testing.assert_allclose(fast_dist, ref_dist, rtol=0, atol=0, equal_nan=True)
    for name in attributes:
        np.testing.assert_allclose(
            fast_attrs[name], ref_attrs[name], rtol=0, atol=0, equal_nan=True
        )


def test_tie_reduction_is_exercised():
    """Guard: the synthetic case really does produce a multi-point tie."""
    grid_x, grid_y, xs, ys, attributes, reducers, _, n_layers = _synthetic_case()
    # With the straddling pair, cell (iy=3, ix=4) sees two points at distance 10.
    d = np.sqrt((grid_x[4] - xs) ** 2 + (grid_y[3] - ys) ** 2)
    assert np.sum(d == d.min()) >= 2


@pytest.mark.parametrize(
    "spec_factory",
    [
        lambda: theme_specs.paces_spec(n_prereq=2),
        lambda: theme_specs.gammalog_spec(n_prereq=2),
        lambda: theme_specs.reslog_spec(n_prereq=2),
        lambda: theme_specs.refseis_spec(n_prereq=2, active_layers=[0, 2]),
        lambda: theme_specs.pacep_spec(n_prereq=2),
        lambda: theme_specs.fewtem_spec(n_prereq=2),
        lambda: theme_specs.manytem_spec(n_prereq=2),
        lambda: theme_specs.ttem_spec(n_prereq=2),
    ],
)
def test_build_theme_identical_under_both_windowings(spec_factory):
    """Full theme pipeline gives the same grid with vectorised vs naive windowing."""
    grid_x, grid_y, xs, ys, attributes, _, complexity, n_layers = _synthetic_case()
    spec = spec_factory()
    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=complexity)

    fast = build_theme(data, spec, grid_x, grid_y, n_layers)
    ref = build_theme(
        data, spec, grid_x, grid_y, n_layers, window_fn=naive_windowed_nearest
    )

    assert fast.shape == (grid_y.size, grid_x.size, n_layers)
    np.testing.assert_allclose(fast, ref, rtol=0, atol=0, equal_nan=True)


def test_build_theme_basic_invariants():
    grid_x, grid_y, xs, ys, attributes, _, complexity, n_layers = _synthetic_case()
    spec = theme_specs.gammalog_spec(n_prereq=2)
    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=complexity)
    grid = build_theme(data, spec, grid_x, grid_y, n_layers)
    # Variance is strictly positive everywhere and never NaN (NaN -> sentinel).
    assert np.all(grid > 0)
    assert not np.isnan(grid).any()
    # +inf is a *valid* outcome (a location with KNOWN complexity exists but
    # beyond the certainty kernel's reach => zero certainty => infinite variance).
    # The MATLAB leaves it as Inf, and combine() turns it into zero precision. So
    # we assert "no NaN", not "all finite".
    assert np.isinf(grid).any()
    # Cells of unknown complexity (class 0) carry no usable information. With the
    # multiplicative MATLAB kernel the NaN range poisons BOTH branches: a location
    # within width gives NaN*... = NaN, and one beyond width gives 0*NaN = NaN
    # (np.where would have leaked a literal 0 -> +inf here). Either way -> sentinel.
    unknown = complexity == 0
    g = grid[unknown]
    assert np.all(g == NODATA_VARIANCE)


# ---------------------------------------------------------------------------
# MEP: the "first point in window" reducer and acquisition-type var0
#
# The Jylland reference data contains *no* wenner-2D points, so the real-data
# validation never exercises either of MEP's two wrinkles. These synthetic cases
# cover them directly.
# ---------------------------------------------------------------------------


def _first_reducer_case():
    """A case where the first point in a window is NOT the nearest point.

    Cell (iy=2, ix=2) = (200, 200) has two in-window points: point 0 sits in the
    far corner (lowest index -> wins ``first``) and point 1 sits exactly on the
    cell (the nearest -> wins any tie-set reducer). Their ``mep_type`` differ, so
    ``first`` and "nearest" give provably different answers there.
    """
    nx = ny = 5
    n_layers = 2
    dx = 100.0
    grid_x = np.arange(nx) * dx
    grid_y = np.arange(ny) * dx
    xs = np.array([40.0, 200.0])
    ys = np.array([40.0, 200.0])
    attributes = {
        "depth": np.array([[50.0, 50.0], [10.0, 10.0]]),
        "doi": np.array([200.0, 200.0]),     # never masks (doi > depth)
        "mep_type": np.array([1.0, 0.0]),    # first=wenner, nearest=not-wenner
    }
    return grid_x, grid_y, xs, ys, attributes, n_layers


def test_first_reducer_matches_naive_and_picks_window_first():
    grid_x, grid_y, xs, ys, attributes, n_layers = _first_reducer_case()
    reducers = {"depth": "mean", "doi": "mean", "mep_type": "first"}
    fast_d, fast = windowed_nearest(
        xs, ys, grid_x, grid_y, 2, attributes, reducers, n_layers
    )
    ref_d, ref = naive_windowed_nearest(
        xs, ys, grid_x, grid_y, 2, attributes, reducers, n_layers
    )
    for name in attributes:
        np.testing.assert_allclose(
            fast[name], ref[name], rtol=0, atol=0, equal_nan=True
        )
    # 'first' picks the lowest-index in-window point (type 1)...
    assert fast["mep_type"][2, 2, 0] == 1.0
    # ...whereas a tie-set reducer would pick the nearest point (type 0).
    _, near = windowed_nearest(
        xs, ys, grid_x, grid_y, 2,
        {"mep_type": attributes["mep_type"]}, {"mep_type": "mean"}, n_layers,
    )
    assert near["mep_type"][2, 2, 0] == 0.0


def test_mep_var0_depends_on_acquisition_type():
    """The wenner-2D vs not-wenner var0 branches produce different variance."""
    grid_x, grid_y, xs, ys, attributes, n_layers = _first_reducer_case()
    complexity = np.full((grid_y.size, grid_x.size), 2.0)
    spec = theme_specs.mep_spec(n_prereq=1, doi_fill=50.0)

    g_w = build_theme(
        ThemeData(xs=xs, ys=ys,
                  attributes={**attributes, "mep_type": np.array([1.0, 1.0])},
                  complexity=complexity),
        spec, grid_x, grid_y, n_layers,
    )
    g_n = build_theme(
        ThemeData(xs=xs, ys=ys,
                  attributes={**attributes, "mep_type": np.array([0.0, 0.0])},
                  complexity=complexity),
        spec, grid_x, grid_y, n_layers,
    )
    informative = (
        (g_w < NODATA_VARIANCE) & np.isfinite(g_w)
        & (g_n < NODATA_VARIANCE) & np.isfinite(g_n)
    )
    assert informative.any()
    assert not np.allclose(g_w[informative], g_n[informative])


def test_mep_build_theme_identical_under_both_windowings():
    grid_x, grid_y, xs, ys, attributes, n_layers = _first_reducer_case()
    complexity = np.full((grid_y.size, grid_x.size), 2.0)
    spec = theme_specs.mep_spec(n_prereq=1, doi_fill=50.0)
    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=complexity)
    fast = build_theme(data, spec, grid_x, grid_y, n_layers)
    ref = build_theme(
        data, spec, grid_x, grid_y, n_layers, window_fn=naive_windowed_nearest
    )
    np.testing.assert_allclose(fast, ref, rtol=0, atol=0, equal_nan=True)


def test_pacep_doi_fill_after_reduction():
    """A NaN DOI at the nearest point becomes 15 per-cell (not masked away)."""
    grid_x = np.arange(3) * 100.0
    grid_y = np.arange(3) * 100.0
    xs = np.array([100.0])
    ys = np.array([100.0])
    attributes = {
        "depth": np.array([[10.0]]),
        "doi": np.array([np.nan]),   # missing -> filled with 15 after reduction
    }
    complexity = np.full((3, 3), 2.0)
    spec = theme_specs.pacep_spec(n_prereq=1)
    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=complexity)
    grid = build_theme(data, spec, grid_x, grid_y, 1)
    # doi filled to 15 > depth 10 -> the central cell is NOT masked (informative).
    assert grid[1, 1, 0] < NODATA_VARIANCE


def test_missing_complexity_defaults_to_uniform_class_one():
    grid_x, grid_y, xs, ys, attributes, _, complexity, n_layers = _synthetic_case()
    spec = theme_specs.paces_spec(n_prereq=2)
    g_none = build_theme(
        ThemeData(xs=xs, ys=ys, attributes=attributes), spec, grid_x, grid_y, n_layers
    )
    g_one = build_theme(
        ThemeData(xs=xs, ys=ys, attributes=attributes,
                  complexity=np.ones_like(complexity)),
        spec, grid_x, grid_y, n_layers,
    )
    np.testing.assert_array_equal(g_none, g_one)


def test_refseis_inactive_layers_are_nodata():
    grid_x, grid_y, xs, ys, attributes, _, complexity, n_layers = _synthetic_case()
    spec = theme_specs.refseis_spec(n_prereq=2, active_layers=[1])
    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=complexity)
    grid = build_theme(data, spec, grid_x, grid_y, n_layers)
    assert np.all(grid[:, :, 0] == NODATA_VARIANCE)
    assert np.all(grid[:, :, 2] == NODATA_VARIANCE)


# ---------------------------------------------------------------------------
# Golden regression (deterministic stages)
# ---------------------------------------------------------------------------


def test_build_theme_golden():
    grid_x, grid_y, xs, ys, attributes, _, complexity, n_layers = _synthetic_case()
    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=complexity)
    grids = {
        spec.name: build_theme(data, spec, grid_x, grid_y, n_layers)
        for spec in [
            theme_specs.paces_spec(n_prereq=2),
            theme_specs.gammalog_spec(n_prereq=2),
            theme_specs.ttem_spec(n_prereq=2),
        ]
    }
    final = combine_variances(list(grids.values()))
    assert_matches_golden("themes_synthetic", **grids, combined=final)
