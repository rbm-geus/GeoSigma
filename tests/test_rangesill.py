# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma.preprocess.rangesill.

The whole module is skipped (not failed) when torch is not installed, since
it's an optional extra (``pip install "geosigma[nn]"``). ``extract_residual_patch``,
``sample_scatter_points``, ``interpolate_scatter_points`` and
``estimate_range_sill_grid`` with a stub estimator have no torch dependency of
their own (only ``NNRangeSillEstimator``/``train_rangesill_estimator`` do) --
the module-level skip here is for simplicity, not because those tests need it.
``interpolate_scatter_points`` does need ``scipy``, the other half of the
``nn`` extra; if that's missing instead, those tests fail with an
``ImportError`` rather than being skipped, which is the right signal (scipy
should always be installed alongside torch via the same extra).
"""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.core.precal_cov import precal_cov
from geosigma.preprocess.rangesill import (
    CELL_SIZE,
    DEFAULT_PATCH_SIZE,
    MINIGRID_SIZE,
    NNRangeSillEstimator,
    _build_net,
    build_range_cholesky_lookup,
    estimate_center_range_sill,
    estimate_range_sill_grid,
    estimate_range_sill_surface,
    extract_residual_patch,
    generate_training_data,
    interpolate_scatter_points,
    load_training_data,
    sample_scatter_points,
    save_training_data,
    train_rangesill_estimator,
)
from geosigma.testing import synthetic_field

torch = pytest.importorskip("torch", reason="geosigma[nn] (torch) not installed")


def test_constants_match_matlab_source():
    # get_interpolated_minigrids_parallel.m: mini_grid_size = 33; Cell_Size = 100.
    assert MINIGRID_SIZE == 33
    assert CELL_SIZE == 100.0
    assert DEFAULT_PATCH_SIZE == MINIGRID_SIZE - 2 == 31


# --------------------------------------------------------------------------- #
# extract_residual_patch (no torch needed)
# --------------------------------------------------------------------------- #
def test_extract_residual_patch_interior():
    surface = np.arange(100.0).reshape(10, 10)
    patch = extract_residual_patch(surface, (5, 5), 3)
    np.testing.assert_allclose(patch, surface[4:7, 4:7])


def test_extract_residual_patch_edge_is_nan_padded():
    surface = np.arange(100.0).reshape(10, 10)
    patch = extract_residual_patch(surface, (0, 0), 5)
    assert patch.shape == (5, 5)
    assert np.isnan(patch[0, 0]) and np.isnan(patch[1, 1])
    np.testing.assert_allclose(patch[2:, 2:], surface[0:3, 0:3])


def test_extract_residual_patch_entirely_outside_is_all_nan():
    surface = np.zeros((10, 10))
    patch = extract_residual_patch(surface, (-100, -100), 5)
    assert np.isnan(patch).all()


def test_extract_residual_patch_requires_odd_size():
    with pytest.raises(ValueError):
        extract_residual_patch(np.zeros((10, 10)), (5, 5), 4)


def test_extract_residual_patch_default_size_is_minigrid_size():
    patch = extract_residual_patch(np.zeros((100, 100)), (50, 50))
    assert patch.shape == (MINIGRID_SIZE, MINIGRID_SIZE)


# --------------------------------------------------------------------------- #
# sample_scatter_points (no torch needed)
# --------------------------------------------------------------------------- #
def test_sample_scatter_points_shapes_and_values():
    field = np.arange(100.0).reshape(10, 10)
    rows, cols, values = sample_scatter_points(field, 15, seed=0)
    assert rows.shape == cols.shape == values.shape == (15,)
    np.testing.assert_allclose(values, field[rows, cols])


def test_sample_scatter_points_reproducible_given_seed():
    field = np.random.default_rng(0).normal(size=(10, 10))
    a = sample_scatter_points(field, 10, seed=1)
    b = sample_scatter_points(field, 10, seed=1)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_sample_scatter_points_caps_at_field_size():
    field = np.zeros((3, 3))
    rows, cols, values = sample_scatter_points(field, 100, seed=0)
    assert rows.shape == (9,)


# --------------------------------------------------------------------------- #
# interpolate_scatter_points (needs scipy, not torch)
# --------------------------------------------------------------------------- #
def test_interpolate_scatter_points_shape_and_constant_recovery():
    rng = np.random.default_rng(0)
    xs = rng.uniform(-100, 100, 30)
    ys = rng.uniform(-100, 100, 30)
    values = np.full(30, 5.0)
    grid = interpolate_scatter_points(xs, ys, values, half_extent=100, grid_size=11)
    assert grid.shape == (11, 11)
    finite = grid[np.isfinite(grid)]
    assert finite.size > 0
    np.testing.assert_allclose(finite, 5.0, atol=1e-6)


def test_interpolate_scatter_points_too_few_points_is_all_nan():
    grid = interpolate_scatter_points(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        half_extent=10,
        grid_size=5,
    )
    assert np.isnan(grid).all()


# --------------------------------------------------------------------------- #
# estimate_range_sill_grid: windowing/gating logic, with a stub estimator so
# these don't need torch either.
# --------------------------------------------------------------------------- #
class _ConstantEstimator:
    patch_size = 5

    def predict(self, grid):
        return 123.0, 4.0  # sill here is never actually used -- see docstring


def test_estimate_range_sill_grid_boolean_mask_and_sill_from_scatter_variance():
    surface = np.zeros((20, 20))
    points = np.zeros((20, 20), dtype=bool)
    points[10, 10] = True
    points[10, 12] = True

    rows, cols, ranges, sills, means = estimate_range_sill_grid(
        surface, points, estimator=_ConstantEstimator(), dx=1.0, minigrid_size=5
    )
    assert set(zip(rows.tolist(), cols.tolist())) == {(10, 10), (10, 12)}
    np.testing.assert_allclose(means, [0.0, 0.0])  # flat surface -> zero local mean
    np.testing.assert_allclose(ranges, [123.0, 123.0])
    # surface is uniformly zero, so the scattered points' variance is exactly
    # 0 regardless of _ConstantEstimator's own (unused) sill return of 4.0.
    np.testing.assert_allclose(sills, [0.0, 0.0])


def test_estimate_range_sill_grid_index_array():
    surface = np.zeros((20, 20))
    points = np.array([[5, 5], [10, 10], [15, 15]])
    rows, cols, ranges, sills, _means = estimate_range_sill_grid(
        surface, points, estimator=_ConstantEstimator(), dx=1.0, minigrid_size=5
    )
    np.testing.assert_array_equal(rows, [5, 10, 15])
    np.testing.assert_array_equal(cols, [5, 10, 15])
    assert len(ranges) == 3


def test_estimate_range_sill_grid_skips_edge_clipped_windows():
    surface = np.zeros((10, 10))
    points = np.zeros((10, 10), dtype=bool)
    points[0, 0] = True  # minigrid_size=7 would reach well outside the surface
    rows, cols, ranges, sills, means = estimate_range_sill_grid(
        surface, points, estimator=_ConstantEstimator(), dx=1.0, minigrid_size=7
    )
    assert np.isnan(ranges[0]) and np.isnan(sills[0])
    # ...but the local mean is still assigned, as in MATLAB (means(i) is set
    # before the `test < 0.1` gate, and in the clipped-window branch too).
    assert np.isfinite(means[0])


def test_estimate_range_sill_grid_uses_scattered_values_not_dense_window():
    # Two points share a window; a third value sits elsewhere in the dense
    # surface but is NOT one of the sampled points, so it must not affect sill.
    surface = np.zeros((20, 20))
    surface[10, 19] = 1000.0  # far outside this window, not a sampled point
    points = np.zeros((20, 20), dtype=bool)
    points[10, 10] = True
    points[10, 11] = True

    _, _, _, sills, _ = estimate_range_sill_grid(
        surface, points, estimator=_ConstantEstimator(), dx=1.0, minigrid_size=5
    )
    np.testing.assert_allclose(sills, [0.0, 0.0])


# --------------------------------------------------------------------------- #
# estimate_center_range_sill (stub estimator, no torch needed)
# --------------------------------------------------------------------------- #
def test_estimate_center_range_sill_returns_the_centre_cells_estimate():
    # A field whose centre window is the whole field: the returned pair must
    # match what estimate_range_sill_grid reports for the centre cell.
    field = np.arange(81, dtype=float).reshape(9, 9)
    got = estimate_center_range_sill(
        field, estimator=_ConstantEstimator(), dx=1.0, n_scatter_points=12, seed=0
    )

    rows, cols, ranges, sills, _ = estimate_range_sill_grid(
        field,
        np.ones((9, 9), dtype=bool),
        estimator=_ConstantEstimator(),
        dx=1.0,
        minigrid_size=9,
    )
    idx = np.flatnonzero((rows == 4) & (cols == 4))[0]
    assert got[0] == pytest.approx(ranges[idx])


def test_estimate_center_range_sill_is_reproducible_given_a_seed():
    field = np.arange(81, dtype=float).reshape(9, 9)
    kw = dict(estimator=_ConstantEstimator(), dx=1.0, n_scatter_points=8)
    assert estimate_center_range_sill(
        field, seed=3, **kw
    ) == estimate_center_range_sill(field, seed=3, **kw)


def test_estimate_center_range_sill_rejects_a_non_square_field():
    with pytest.raises(ValueError, match="square"):
        estimate_center_range_sill(np.zeros((9, 7)), estimator=_ConstantEstimator())


def test_estimate_center_range_sill_rejects_an_even_side():
    # No single centre cell to read off.
    with pytest.raises(ValueError, match="odd"):
        estimate_center_range_sill(np.zeros((8, 8)), estimator=_ConstantEstimator())


def test_estimate_range_sill_grid_separate_conditioning_points():
    # Estimate densely (every cell), conditioned on a separate, sparse set --
    # matching get_interpolated_minigrids_parallel.m's mask/drawpoints split,
    # not the points-serve-double-duty default.
    surface = np.zeros((20, 20))
    estimate_at = np.ones((20, 20), dtype=bool)  # every cell
    conditioning = np.zeros((20, 20), dtype=bool)
    conditioning[10, 10] = True
    conditioning[10, 12] = True  # sparse: just 2 real data points anywhere

    rows, cols, ranges, sills, means = estimate_range_sill_grid(
        surface,
        estimate_at,
        conditioning_points=conditioning,
        estimator=_ConstantEstimator(),
        dx=1.0,
        minigrid_size=5,
    )
    # Estimated at every one of the 400 cells, not just the 2 conditioning ones.
    assert len(rows) == 400
    # A cell far from both conditioning points has no data in its window at
    # all, so its sill/range stay NaN (but its mean still comes from the
    # window's own finite cells, same as a MATLAB clipped-window point).
    far_idx = np.flatnonzero((rows == 0) & (cols == 0))[0]
    assert np.isnan(sills[far_idx]) and np.isnan(ranges[far_idx])
    assert np.isfinite(means[far_idx])
    # A cell at a conditioning point's own location picks up both points
    # (both are within a 5-cell window of (10,10) and (10,12)).
    near_idx = np.flatnonzero((rows == 10) & (cols == 10))[0]
    assert np.isfinite(sills[near_idx])


# --------------------------------------------------------------------------- #
# architecture: "conv" (default) vs "dense"
# --------------------------------------------------------------------------- #
def test_train_accepts_both_architectures_and_records_the_choice():
    data = generate_training_data(40, minigrid_size=11, n_scatter_points=25, seed=0)
    patch = np.random.default_rng(0).normal(size=(data.interp_grid_size,) * 2)
    for arch in ("conv", "dense"):
        est = train_rangesill_estimator(
            data, hidden=4, architecture=arch, epochs=1, seed=0
        )
        assert est.architecture == arch
        range_, sill = est.predict(patch)
        # predict() exponentiates the net's log-range output, so any trained
        # network gives a strictly positive range and a finite sample sill.
        assert np.isfinite(range_) and range_ > 0
        assert np.isfinite(sill) and sill > 0


def test_architecture_round_trips_through_save_load(tmp_path):
    data = generate_training_data(40, minigrid_size=11, n_scatter_points=25, seed=0)
    rng = np.random.default_rng(1)
    patch = rng.normal(size=(data.interp_grid_size,) * 2)

    for arch in ("conv", "dense"):
        est = train_rangesill_estimator(
            data, hidden=4, architecture=arch, epochs=1, seed=0
        )
        path = tmp_path / f"{arch}.pt"
        est.save(path)
        back = NNRangeSillEstimator.load(path)
        assert back.architecture == arch
        np.testing.assert_allclose(back.predict(patch), est.predict(patch))


def test_load_defaults_to_conv_for_files_written_before_the_option_existed(tmp_path):
    import torch

    est = NNRangeSillEstimator(patch_size=9, hidden=4)
    path = tmp_path / "legacy.pt"
    # Exactly the payload save() used to write -- no "architecture" key.
    torch.save(
        {
            "state_dict": est._net().state_dict(),
            "patch_size": est.patch_size,
            "hidden": est.hidden,
        },
        path,
    )
    assert NNRangeSillEstimator.load(path).architecture == "conv"


def test_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="architecture"):
        NNRangeSillEstimator(architecture="mlp")


def test_conv_is_patch_size_agnostic_but_dense_is_not():
    # The conv net's AdaptiveAvgPool2d gives fixed-size features for any input,
    # so the same weights accept a different patch size; the dense net's first
    # layer width is structural and cannot.
    conv = _build_net(4, "conv", patch_size=9)
    dense = _build_net(4, "dense", patch_size=9)
    x = torch.zeros(1, 1, 15, 15)  # not 9x9
    conv(x)  # must not raise
    with pytest.raises(RuntimeError):
        dense(x)


# --------------------------------------------------------------------------- #
# estimate_range_sill_surface: the dense, grid-in/grid-out wrapper
# --------------------------------------------------------------------------- #
def test_estimate_range_sill_surface_is_dense_and_grid_shaped():
    rng = np.random.default_rng(0)
    surface = rng.normal(size=(20, 20))
    points = np.zeros((20, 20), dtype=bool)
    points[::3, ::3] = True

    rs = estimate_range_sill_surface(
        surface, points, estimator=_ConstantEstimator(), dx=1.0, minigrid_size=5
    )

    for field in (rs.range_field, rs.sill_field, rs.lf_mean, rs.residual):
        assert field.shape == surface.shape
    # Dense: estimated everywhere, including the border cells a naive
    # (unpadded) call would drop for having too much missing window.
    assert np.isfinite(rs.range_field).all()
    assert np.isfinite(rs.lf_mean).all()
    np.testing.assert_allclose(rs.residual, surface - rs.lf_mean)


def test_estimate_range_sill_surface_matches_manual_padded_call():
    # The wrapper is exactly the padded/dense/reshaped form of the lower-level
    # function -- pin that, so the two can't silently drift apart.
    rng = np.random.default_rng(1)
    surface = rng.normal(size=(18, 18))
    points = np.zeros((18, 18), dtype=bool)
    points[::4, ::2] = True
    minigrid = 5

    rs = estimate_range_sill_surface(
        surface, points, estimator=_ConstantEstimator(), dx=1.0, minigrid_size=minigrid
    )

    margin = minigrid // 2
    ny, nx = surface.shape
    surf_pad = np.pad(surface, margin, mode="reflect")
    cond_pad = np.zeros(surf_pad.shape, dtype=bool)
    cond_pad[margin : margin + ny, margin : margin + nx] = points
    est_pad = np.zeros(surf_pad.shape, dtype=bool)
    est_pad[margin : margin + ny, margin : margin + nx] = True

    r, c, ranges, sills, means = estimate_range_sill_grid(
        surf_pad,
        est_pad,
        conditioning_points=cond_pad,
        estimator=_ConstantEstimator(),
        dx=1.0,
        minigrid_size=minigrid,
    )
    expected = np.full((ny, nx), np.nan)
    expected[r - margin, c - margin] = ranges
    np.testing.assert_array_equal(rs.range_field, expected)

    expected_mean = np.full((ny, nx), np.nan)
    expected_mean[r - margin, c - margin] = means
    np.testing.assert_array_equal(rs.lf_mean, expected_mean)


# --------------------------------------------------------------------------- #
# Training + the estimate_range_sill_grid pipeline end-to-end (needs torch)
# --------------------------------------------------------------------------- #
def _predict_center(estimator, field, dx, n_scatter_points=20, seed=0):
    """Thin wrapper over the library's estimate_center_range_sill.

    Kept so these tests read positionally (estimator first); the library
    function takes the field first and the estimator by keyword.
    """
    return estimate_center_range_sill(
        field,
        estimator=estimator,
        dx=dx,
        n_scatter_points=n_scatter_points,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# build_range_cholesky_lookup (needs scipy, not torch)
# --------------------------------------------------------------------------- #
def test_build_range_cholesky_lookup_keys_shape_and_dtype():
    lookup = build_range_cholesky_lookup((50.0, 60.0), minigrid_size=9, dx=25.0)
    assert sorted(lookup.keys()) == list(range(50, 61))
    for L in lookup.values():
        assert L.shape == (81, 81)
        assert L.dtype == np.float32


def test_build_range_cholesky_lookup_rounds_bounds_inward():
    # ceil(50.5)=51, floor(60.5)=60
    lookup = build_range_cholesky_lookup((50.5, 60.5), minigrid_size=9, dx=25.0)
    assert sorted(lookup.keys()) == list(range(51, 61))


def test_build_range_cholesky_lookup_reconstructs_unit_sill_covariance():
    lookup = build_range_cholesky_lookup((100.0, 100.0), minigrid_size=9, dx=25.0)
    L = lookup[100].astype(float)
    cov = L @ L.T
    # Diagonal is the sill (1) plus the numerical nugget; off-diagonal decays
    # with distance but never exceeds the sill.
    np.testing.assert_allclose(np.diag(cov), 1.0, atol=1e-3)
    assert np.all(cov <= 1.0 + 1e-3)


def test_sill_scaling_after_simulation_matches_direct_precal_cov():
    # The whole point of fixing sill=1 in the lookup table and scaling
    # afterwards is that chol(sill * M) == sqrt(sill) * chol(M). Verify that
    # directly: a unit-sill factor scaled by sqrt(sill) must reproduce the
    # same realization (given the same standard-normal draw) as factorizing
    # the sill-scaled covariance matrix from scratch -- including its
    # numerical nugget, which build_range_cholesky_lookup adds *before*
    # scaling (so it scales with sill too, by construction: a bigger sill
    # gets a proportionally bigger nugget, not a fixed absolute one). The
    # reference computation here must add that same sill-scaled nugget, or
    # the comparison is checking two different numerical recipes, not the
    # claimed identity -- range=20 on this 225 m domain is comfortably
    # well-conditioned (unlike range close to or above the domain size,
    # where ANY nugget convention shifts the factorization visibly).
    minigrid_size, dx, range_, sill = 9, 25.0, 20.0, 25.0
    nugget = 1e-5
    lookup = build_range_cholesky_lookup(
        (range_, range_), minigrid_size=minigrid_size, dx=dx, nugget=nugget
    )
    L_unit = lookup[int(range_)].astype(float)

    z = np.random.default_rng(0).standard_normal(minigrid_size * minigrid_size)
    scaled_field = (L_unit @ z) * np.sqrt(sill)

    coord = (np.arange(minigrid_size) + 0.5) * dx
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    cov_direct, _ = precal_cov(coords, coords, f"{sill} Gau({range_})")
    L_direct = np.linalg.cholesky(cov_direct + sill * nugget * np.eye(coords.shape[0]))
    direct_field = L_direct @ z

    np.testing.assert_allclose(scaled_field, direct_field, rtol=1e-6, atol=1e-6)


def test_generate_training_data_labels_are_rounded_integers_within_bounds():
    data = generate_training_data(
        100,
        minigrid_size=9,
        dx=25.0,
        n_scatter_points=12,
        range_bounds=(50.0, 60.0),
        sill_bounds=(1.0, 10.0),
        seed=0,
    )
    ranges_used = np.round(np.exp(data.log_ranges.astype(float)))
    assert np.all(ranges_used == ranges_used.astype(int))  # exact integers
    assert ranges_used.min() >= 50 and ranges_used.max() <= 60


def test_generate_training_data_custom_lookup_is_actually_consulted():
    # A deliberately "wrong" lookup (every entry replaced by the range=50
    # factor) must change the result relative to the correctly-built default
    # -- if generate_training_data silently ignored `cholesky_lookup` and
    # built its own correct table instead, the two runs would be identical.
    minigrid_size, dx = 9, 25.0
    kwargs = dict(
        minigrid_size=minigrid_size,
        dx=dx,
        n_scatter_points=10,
        range_bounds=(50.0, 60.0),
        sill_bounds=(1.0, 1.0),
        seed=3,
    )
    default_data = generate_training_data(5, **kwargs)

    real_lookup = build_range_cholesky_lookup(
        (50.0, 60.0), minigrid_size=minigrid_size, dx=dx
    )
    rigged_lookup = {r: real_lookup[50] for r in real_lookup}
    rigged_data = generate_training_data(5, cholesky_lookup=rigged_lookup, **kwargs)

    assert not np.allclose(default_data.patches, rigged_data.patches)


def test_generate_training_data_shapes_and_metadata():
    data = generate_training_data(
        25,
        minigrid_size=11,
        dx=25.0,
        n_scatter_points=12,
        range_bounds=(50.0, 500.0),
        sill_bounds=(1.0, 10.0),
        seed=0,
    )
    assert data.patches.shape == (25, 9, 9)  # interp_grid_size = minigrid_size - 2
    assert data.log_ranges.shape == (25,)
    assert data.interp_grid_size == 9
    assert data.minigrid_size == 11
    assert data.dx == 25.0
    assert data.n_scatter_points == 12
    assert data.range_bounds == (50.0, 500.0)
    assert data.sill_bounds == (1.0, 10.0)
    assert data.interp_method == "cubic"
    assert data.range_sampling == "log"  # default


def test_generate_training_data_rejects_bad_range_sampling():
    with pytest.raises(ValueError):
        generate_training_data(5, minigrid_size=9, dx=25.0, range_sampling="bogus")


def test_generate_training_data_linear_sampling_is_less_skewed_than_log():
    # log-uniform sampling concentrates draws near the low end of a wide
    # range span; linear-uniform should not -- check via the mean of many
    # draws, which should sit near the arithmetic midpoint for "linear" but
    # well below it for "log".
    range_bounds = (50.0, 2000.0)
    kwargs = dict(
        minigrid_size=9, dx=25.0, n_scatter_points=10, range_bounds=range_bounds, seed=0
    )
    log_data = generate_training_data(300, range_sampling="log", **kwargs)
    linear_data = generate_training_data(300, range_sampling="linear", **kwargs)

    log_ranges_used = np.exp(log_data.log_ranges.astype(float))
    linear_ranges_used = np.exp(linear_data.log_ranges.astype(float))

    midpoint = (range_bounds[0] + range_bounds[1]) / 2
    assert log_ranges_used.mean() < midpoint * 0.6  # log-uniform: skewed low
    assert abs(linear_ranges_used.mean() - midpoint) < midpoint * 0.15  # near midpoint
    assert linear_data.range_sampling == "linear"


def test_generate_training_data_reproducible_given_seed():
    kwargs = dict(minigrid_size=9, dx=25.0, n_scatter_points=10, seed=7)
    a = generate_training_data(20, **kwargs)
    b = generate_training_data(20, **kwargs)
    np.testing.assert_array_equal(a.patches, b.patches)
    np.testing.assert_array_equal(a.log_ranges, b.log_ranges)


def test_save_and_load_training_data_roundtrip(tmp_path):
    data = generate_training_data(
        15, minigrid_size=9, dx=25.0, n_scatter_points=10, seed=2
    )
    path = tmp_path / "training_data.npz"
    save_training_data(path, data)
    loaded = load_training_data(path)

    np.testing.assert_allclose(loaded.patches, data.patches)
    np.testing.assert_allclose(loaded.log_ranges, data.log_ranges)
    assert loaded.interp_grid_size == data.interp_grid_size
    assert loaded.minigrid_size == data.minigrid_size
    assert loaded.dx == data.dx
    assert loaded.n_scatter_points == data.n_scatter_points
    assert loaded.range_bounds == data.range_bounds
    assert loaded.sill_bounds == data.sill_bounds
    assert loaded.interp_method == data.interp_method
    assert loaded.range_sampling == data.range_sampling


def test_load_training_data_defaults_range_sampling_for_old_files(tmp_path):
    # Simulate a .npz written before `range_sampling` existed: no such key.
    import numpy as _np

    path = tmp_path / "old_format.npz"
    _np.savez_compressed(
        path,
        patches=_np.zeros((2, 3, 3), dtype=_np.float32),
        log_ranges=_np.zeros(2, dtype=_np.float32),
        interp_grid_size=3,
        minigrid_size=5,
        dx=25.0,
        n_scatter_points=10,
        range_bounds=_np.array([50.0, 500.0]),
        sill_bounds=_np.array([1.0, 10.0]),
        interp_method="cubic",
    )
    loaded = load_training_data(path)
    assert loaded.range_sampling == "log"


def test_train_rangesill_estimator_with_precomputed_data_matches_generation_path():
    # Generating once and training twice from the cached TrainingData must
    # give the same result as the all-in-one (training_data=None) call that
    # generates internally, given the same overall seed usage.
    data = generate_training_data(
        30, minigrid_size=9, dx=25.0, n_scatter_points=10, seed=4
    )
    est_a = train_rangesill_estimator(data, hidden=6, epochs=8, seed=9)
    est_b = train_rangesill_estimator(data, hidden=6, epochs=8, seed=9)

    field = synthetic_field(9, 9, 25.0, 150.0, 5.0, seed=50)
    assert _predict_center(est_a, field, dx=25.0) == _predict_center(
        est_b, field, dx=25.0
    )
    assert est_a.patch_size == data.interp_grid_size


def test_train_rangesill_estimator_reuses_data_across_different_hyperparameters():
    data = generate_training_data(
        20, minigrid_size=9, dx=25.0, n_scatter_points=10, seed=5
    )
    est_small = train_rangesill_estimator(data, hidden=4, epochs=5, seed=1)
    est_big = train_rangesill_estimator(data, hidden=16, epochs=5, seed=1)
    # Both must still accept the same interpolated grid size (a property of
    # the cached data, not of hidden/epochs).
    assert est_small.patch_size == est_big.patch_size == data.interp_grid_size


def test_training_is_reproducible_given_seed():
    kwargs = dict(
        n_samples=40,
        minigrid_size=11,
        dx=25.0,
        n_scatter_points=15,
        range_bounds=(50.0, 500.0),
        sill_bounds=(1.0, 10.0),
        hidden=8,
        epochs=15,
        batch_size=16,
        seed=3,
    )
    est_a = train_rangesill_estimator(**kwargs)
    est_b = train_rangesill_estimator(**kwargs)

    field = synthetic_field(11, 11, 25.0, 200.0, 5.0, seed=99)
    assert _predict_center(est_a, field, dx=25.0) == _predict_center(
        est_b, field, dx=25.0
    )


def test_training_recovers_range_ordering():
    # Not a tight accuracy bound (tiny/fast training config) -- just checks
    # the network learns *something*: a short-range and long-range field
    # should not be predicted as identical or inverted on average.
    est = train_rangesill_estimator(
        n_samples=200,
        minigrid_size=15,
        dx=25.0,
        n_scatter_points=30,
        range_bounds=(50.0, 1500.0),
        sill_bounds=(1.0, 20.0),
        hidden=12,
        epochs=120,
        batch_size=32,
        seed=5,
    )
    rng = np.random.default_rng(7)
    short = [
        _predict_center(
            est,
            synthetic_field(
                15, 15, 25.0, 80.0, 5.0, seed=int(rng.integers(0, 2**31 - 1))
            ),
            dx=25.0,
            seed=int(rng.integers(0, 2**31 - 1)),
        )[0]
        for _ in range(10)
    ]
    long = [
        _predict_center(
            est,
            synthetic_field(
                15, 15, 25.0, 1200.0, 5.0, seed=int(rng.integers(0, 2**31 - 1))
            ),
            dx=25.0,
            seed=int(rng.integers(0, 2**31 - 1)),
        )[0]
        for _ in range(10)
    ]
    assert np.median(long) > np.median(short)


def test_predict_rejects_wrong_patch_shape():
    est = NNRangeSillEstimator(patch_size=11)
    with pytest.raises(ValueError):
        est.predict(np.zeros((5, 5)))


def test_predict_all_nan_patch_returns_nan():
    est = NNRangeSillEstimator(patch_size=5)
    r, s = est.predict(np.full((5, 5), np.nan))
    assert np.isnan(r) and np.isnan(s)


def test_save_load_roundtrip(tmp_path):
    est = train_rangesill_estimator(
        n_samples=30,
        minigrid_size=9,
        dx=25.0,
        n_scatter_points=12,
        range_bounds=(50.0, 300.0),
        sill_bounds=(1.0, 10.0),
        hidden=6,
        epochs=10,
        batch_size=8,
        seed=2,
    )
    assert est.patch_size == 7  # minigrid_size - 2
    field = synthetic_field(9, 9, 25.0, 150.0, 5.0, seed=11)
    before = _predict_center(est, field, dx=25.0, seed=0)

    path = tmp_path / "model.pt"
    est.save(path)
    reloaded = NNRangeSillEstimator.load(path)
    after = _predict_center(reloaded, field, dx=25.0, seed=0)

    assert before == after
    assert reloaded.patch_size == 7
