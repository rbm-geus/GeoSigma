# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for the borehole/well variance theme (``geosigma.themes.well``).

The trust anchor mirrors ``test_themes.py``: a direct triple-loop port of the
MATLAB ``get_well_theme.m`` window+sort+sum logic, asserted identical to the
binned :func:`build_well_theme`. Because the well theme sums the 3 nearest wells
and breaks distance ties by original well order, the reference also covers those
two behaviours (which the Jylland data alone may not stress).
"""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.themes import NODATA_VARIANCE, WellSet, build_well_theme


def naive_well_precision(grid_x, grid_y, w, search_radius, comp2range, n_nearest):
    """Direct triple-loop port of one well set's precision grid.

    Mirrors ``get_well_theme.m``: per cell, filter wells to the open
    ``±search_radius``-cell window, sort by distance (stable, original order),
    take the nearest ``n_nearest``, and sum ``sill·exp(-3 d²/range²)``.
    """
    grid_x = np.asarray(grid_x, float)
    grid_y = np.asarray(grid_y, float)
    nx, ny = grid_x.size, grid_y.size
    R = int(search_radius)
    q = np.asarray(w.quality, int) - 1
    K = np.asarray(w.Ks, float)[q]
    grad = np.asarray(w.gradients, float)[q]
    var0 = K**2 + (grad * np.asarray(w.depth, float)) ** 2
    sill = 1.0 / var0
    comp = np.where(np.isnan(w.comp_class), 0, w.comp_class).astype(int)
    rng = np.asarray(comp2range, float)[comp]
    xs, ys = np.asarray(w.xs, float), np.asarray(w.ys, float)

    prec = np.zeros((ny, nx))
    for i in range(ny):
        ymin = grid_y[max(0, i - R)]
        ymax = grid_y[min(ny - 1, i + R)]
        yfilt = (ys < ymax) & (ys > ymin)
        for j in range(nx):
            xmin = grid_x[max(0, j - R)]
            xmax = grid_x[min(nx - 1, j + R)]
            filt = yfilt & (xs < xmax) & (xs > xmin)
            idx = np.nonzero(filt)[0]  # ascending original order
            if idx.size == 0:
                continue
            d = np.sqrt((grid_x[j] - xs[idx]) ** 2 + (grid_y[i] - ys[idx]) ** 2)
            order = np.argsort(d, kind="stable")[:n_nearest]
            sel = idx[order]
            dsel = d[order]
            prec[i, j] = np.sum(sill[sel] * np.exp(-3.0 * dsel**2 / rng[sel] ** 2))
    return prec


def naive_well_theme(grid_x, grid_y, snap_layers, jup_layers, **kw):
    R = kw.get("search_radius", 6)
    comp2range = kw.get("comp2range", (100.0, 500.0, 400.0, 250.0, 100.0))
    n_nearest = kw.get("n_nearest", 3)
    ny, nx = np.asarray(grid_y).size, np.asarray(grid_x).size
    n_layers = len(snap_layers)
    out = np.full((ny, nx, n_layers), NODATA_VARIANCE)
    for s in range(n_layers):
        prec = np.zeros((ny, nx))
        for wset in (snap_layers[s], jup_layers[s]):
            if wset is not None and np.asarray(wset.xs).size:
                prec += naive_well_precision(
                    grid_x, grid_y, wset, R, comp2range, n_nearest
                )
        with np.errstate(divide="ignore"):
            out[:, :, s] = np.where(prec > 0, 1.0 / prec, NODATA_VARIANCE)
    return out


def _wellset(rng, npts, grid_x, grid_y):
    dx = grid_x[-1] - grid_x[0]
    dy = grid_y[-1] - grid_y[0]
    return WellSet(
        xs=rng.uniform(grid_x[0] - 0.1 * dx, grid_x[-1] + 0.1 * dx, npts),
        ys=rng.uniform(grid_y[0] - 0.1 * dy, grid_y[-1] + 0.1 * dy, npts),
        depth=rng.uniform(0.0, 120.0, npts),
        quality=rng.integers(1, 5, npts),
        comp_class=rng.integers(0, 5, npts).astype(float),
        gradients=0.5 * np.array([0.02, 0.03, 0.07, 0.1]),
        Ks=np.array([0.15, 1.0, 2.0, 2.5]),
    )


def _case(seed=0, n_layers=3):
    rng = np.random.default_rng(seed)
    nx, ny = 9, 7
    grid_x = np.arange(nx) * 100.0 + 500_000.0
    grid_y = np.arange(ny) * 100.0 + 6_100_000.0
    snap = [_wellset(rng, 12, grid_x, grid_y) for _ in range(n_layers)]
    jup = [_wellset(rng, 8, grid_x, grid_y) for _ in range(n_layers)]
    # One layer with a missing source, and one fully empty layer.
    jup[1] = None
    snap[2] = None
    jup[2] = None
    return grid_x, grid_y, snap, jup


@pytest.mark.parametrize("search_radius", [3, 6])
def test_build_well_theme_matches_naive(search_radius):
    grid_x, grid_y, snap, jup = _case()
    fast = build_well_theme(grid_x, grid_y, snap, jup, search_radius=search_radius)
    ref = naive_well_theme(grid_x, grid_y, snap, jup, search_radius=search_radius)
    np.testing.assert_allclose(fast, ref, rtol=0, atol=0, equal_nan=True)


def test_empty_layer_is_nodata():
    grid_x, grid_y, snap, jup = _case()
    grid = build_well_theme(grid_x, grid_y, snap, jup)
    # Layer 2 has no wells from either source -> all sentinel.
    assert np.all(grid[:, :, 2] == NODATA_VARIANCE)
    # Layer 0 has wells -> at least some informative cells.
    assert np.any(grid[:, :, 0] < NODATA_VARIANCE)


def test_three_nearest_cap_and_tie_order():
    """A cell with >3 equidistant-ish wells sums exactly the 3 nearest, and
    distance ties are resolved by original well order (matches the naive loop)."""
    grid_x = np.arange(5) * 100.0
    grid_y = np.arange(5) * 100.0
    # Five wells clustered around cell (2,2)=(200,200) at increasing distance,
    # plus a deliberate tie pair at equal distance.
    xs = np.array([200.0, 200.0, 140.0, 260.0, 200.0])
    ys = np.array([180.0, 220.0, 200.0, 200.0, 260.0])  # two at d=20 (tie)
    w = WellSet(
        xs=xs, ys=ys,
        depth=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        quality=np.array([1, 2, 3, 4, 1]),
        comp_class=np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        gradients=0.5 * np.array([0.02, 0.03, 0.07, 0.1]),
        Ks=np.array([0.15, 1.0, 2.0, 2.5]),
    )
    snap = [w]
    jup = [None]
    fast = build_well_theme(grid_x, grid_y, snap, jup, search_radius=6)
    ref = naive_well_theme(grid_x, grid_y, snap, jup, search_radius=6)
    np.testing.assert_allclose(fast, ref, rtol=0, atol=0, equal_nan=True)
    # Sanity: the central cell is informative and finite.
    assert fast[2, 2, 0] < NODATA_VARIANCE


# --------------------------------------------------------------------------- #
# Quality convention: 1 = best ... n = worst (input checks)
# --------------------------------------------------------------------------- #
_GRAD = 0.5 * np.array([0.02, 0.03, 0.07, 0.1])
_KS = np.array([0.15, 1.0, 2.0, 2.5])


def _one_well(quality, Ks=_KS, gradients=_GRAD):
    return WellSet(
        xs=np.array([200.0]), ys=np.array([200.0]), depth=np.array([30.0]),
        quality=np.array([quality]), comp_class=np.array([2.0]),
        gradients=gradients, Ks=Ks,
    )


def _grid():
    return np.arange(5) * 100.0, np.arange(5) * 100.0


@pytest.mark.parametrize("bad", [0, 5, -1, np.nan, 1.5])
@pytest.mark.parametrize("source", ["snapped", "database"])
def test_quality_outside_table_range_is_rejected(bad, source):
    # 0 used to index the LAST (worst) entry through negative indexing, and NaN
    # was cast to an arbitrary integer; both must now be refused.
    gx, gy = _grid()
    w = _one_well(bad)
    snap, jup = ([w], [None]) if source == "snapped" else ([None], [w])
    with pytest.raises(ValueError, match=rf"layer 0, {source} wells.*outside 1\.\.4"):
        build_well_theme(gx, gy, snap, jup)


@pytest.mark.parametrize("q", [1, 2, 3, 4])
def test_every_quality_in_range_is_accepted(q):
    gx, gy = _grid()
    grid = build_well_theme(gx, gy, [_one_well(q)], [None])
    assert grid[2, 2, 0] < NODATA_VARIANCE


def test_float_integer_quality_is_accepted():
    # Qualities read from CSV often arrive as floats (2.0); integral values are fine.
    gx, gy = _grid()
    as_int = build_well_theme(gx, gy, [_one_well(2)], [None])
    as_float = build_well_theme(gx, gy, [_one_well(2.0)], [None])
    np.testing.assert_array_equal(as_int, as_float)


def test_quality_range_follows_table_length():
    # The upper bound is len(Ks), not a hardcoded 4.
    gx, gy = _grid()
    Ks6, grad6 = np.linspace(0.1, 3.0, 6), np.linspace(0.01, 0.06, 6)
    build_well_theme(gx, gy, [_one_well(6, Ks=Ks6, gradients=grad6)], [None])
    with pytest.raises(ValueError, match=r"outside 1\.\.6"):
        build_well_theme(gx, gy, [_one_well(7, Ks=Ks6, gradients=grad6)], [None])


def test_mismatched_table_lengths_are_rejected():
    gx, gy = _grid()
    w = _one_well(1, Ks=_KS, gradients=_GRAD[:3])
    with pytest.raises(ValueError, match="gradients has 3 entries but Ks has 4"):
        build_well_theme(gx, gy, [w], [None])


@pytest.mark.parametrize(
    "Ks, gradients, bad_name",
    [
        (_KS[::-1], _GRAD, "Ks"),          # reversed K table
        (_KS, _GRAD[::-1], "gradients"),   # reversed gradient table
    ],
)
def test_decreasing_table_warns(Ks, gradients, bad_name):
    from geosigma.themes.combine import GeoSigmaWarning

    gx, gy = _grid()
    msg = rf"{bad_name} = .* decreases with quality"
    with pytest.warns(GeoSigmaWarning, match=msg):
        build_well_theme(gx, gy, [_one_well(1, Ks=Ks, gradients=gradients)], [None])


def test_decreasing_table_warns_once_per_distinct_table():
    import warnings as _w

    from geosigma.themes.combine import GeoSigmaWarning

    gx, gy = _grid()
    rev = _one_well(1, Ks=_KS[::-1])
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        build_well_theme(gx, gy, [rev, rev, rev], [rev, None, rev])
    hits = [r for r in rec if issubclass(r.category, GeoSigmaWarning)]
    assert len(hits) == 1


def test_increasing_or_flat_tables_do_not_warn():
    import warnings as _w

    from geosigma.themes.combine import GeoSigmaWarning

    gx, gy = _grid()
    flat = _one_well(2, Ks=np.array([1.0, 1.0, 2.0, 2.0]))
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        build_well_theme(gx, gy, [_one_well(1), flat], [None, None])
    assert not [r for r in rec if issubclass(r.category, GeoSigmaWarning)]


# --------------------------------------------------------------------------- #
# Per-layer range tables (range_model) and the Danish pre-Quaternary rule
# --------------------------------------------------------------------------- #
def _case_full(seed=3, n_layers=4):
    rng = np.random.default_rng(seed)
    grid_x = np.arange(9) * 100.0 + 500_000.0
    grid_y = np.arange(7) * 100.0 + 6_100_000.0
    snap = [_wellset(rng, 12, grid_x, grid_y) for _ in range(n_layers)]
    jup = [_wellset(rng, 8, grid_x, grid_y) for _ in range(n_layers)]
    return grid_x, grid_y, snap, jup


def test_range_model_uses_each_layers_table():
    from geosigma.themes import RangeGroup

    grid_x, grid_y, snap, jup = _case_full()
    graded = (100.0, 500.0, 400.0, 250.0, 100.0)
    flat = (500.0,) * 5
    model = [
        RangeGroup(layers=(0, 1), range_by_complexity=graded),
        RangeGroup(layers=(2, None), range_by_complexity=flat),
    ]
    got = build_well_theme(grid_x, grid_y, snap, jup, range_model=model)
    top = naive_well_theme(grid_x, grid_y, snap[:2], jup[:2], comp2range=graded)
    deep = naive_well_theme(grid_x, grid_y, snap[2:], jup[2:], comp2range=flat)
    np.testing.assert_array_equal(got, np.concatenate([top, deep], axis=2))


def test_danish_rule_matches_matlab_all_ones_map_for_pre_quaternary():
    # MATLAB: pre-Q wells get complexity class 1 (all-ones map) and the graded
    # table; the library: real classes and a flat 500 m table. Same result.
    from geosigma.themes import danish_well_range_model

    grid_x, grid_y, snap, jup = _case_full()
    n_prereq = 2

    def ones(w):
        return WellSet(
            xs=w.xs,
            ys=w.ys,
            depth=w.depth,
            quality=w.quality,
            comp_class=np.ones_like(w.comp_class),
            gradients=w.gradients,
            Ks=w.Ks,
        )

    snap_m = snap[:n_prereq] + [ones(w) for w in snap[n_prereq:]]
    jup_m = jup[:n_prereq] + [ones(w) for w in jup[n_prereq:]]
    matlab = build_well_theme(grid_x, grid_y, snap_m, jup_m)
    lib = build_well_theme(
        grid_x, grid_y, snap, jup, range_model=danish_well_range_model(n_prereq)
    )
    np.testing.assert_array_equal(lib, matlab)


def test_range_model_and_comp2range_are_exclusive():
    from geosigma.themes import danish_well_range_model

    grid_x, grid_y, snap, jup = _case_full()
    with pytest.raises(ValueError, match="not both"):
        build_well_theme(
            grid_x,
            grid_y,
            snap,
            jup,
            comp2range=(100.0,) * 5,
            range_model=danish_well_range_model(2),
        )


def test_range_model_must_cover_every_layer_once():
    from geosigma.themes import RangeGroup

    grid_x, grid_y, snap, jup = _case_full()
    t = (100.0,) * 5
    with pytest.raises(ValueError, match="uncovered"):
        build_well_theme(
            grid_x,
            grid_y,
            snap,
            jup,
            range_model=[RangeGroup(layers=(0, 1), range_by_complexity=t)],
        )
    with pytest.raises(ValueError, match="more than one"):
        build_well_theme(
            grid_x,
            grid_y,
            snap,
            jup,
            range_model=[
                RangeGroup(layers="all", range_by_complexity=t),
                RangeGroup(layers=(0, 0), range_by_complexity=t),
            ],
        )
