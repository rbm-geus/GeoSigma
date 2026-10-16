# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma.kriging (state, buffers, windows, simulation)."""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.core.least_squares_inversion import least_squares_inversion
from geosigma.core.local_kriging_setup_img import local_kriging_setup_img
from geosigma.core.precal_cov import precal_cov
from geosigma.kriging import (
    build_windows,
    calculate_buffer_zones,
    connected_components_4,
    krige_layer,
    krige_window,
    pack_kriging_grid,
    simulate_layer,
    simulate_model,
    simulate_window,
    to_absolute,
)
from geosigma.kriging.simulation import blend_window_realizations
from geosigma.kriging.windows import KrigingResult, KrigingWindow


# --------------------------------------------------------------------------- #
# state: pack_kriging_grid
# --------------------------------------------------------------------------- #
def test_pack_kriging_grid_shapes_and_std_conversion():
    ny, nx = 4, 5
    surface = np.arange(ny * nx, dtype=float).reshape(ny, nx)
    variance = np.full((ny, nx), 4.0)

    grid = pack_kriging_grid(surface, variance, dx=100.0)

    assert grid.shape == (ny, nx)
    assert grid.dx == 100.0 and grid.dy == 100.0
    assert grid.xx_norm.shape == (ny * nx,)
    np.testing.assert_allclose(grid.img_dobs, surface.ravel())
    np.testing.assert_allclose(
        grid.img_unc, np.sqrt(variance).ravel()
    )  # std, not variance
    # row-major flattening: xx_norm varies fastest across columns
    np.testing.assert_allclose(grid.xx_norm[:nx], np.arange(nx))
    np.testing.assert_allclose(grid.yy_norm[:nx], np.zeros(nx))


def test_pack_kriging_grid_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        pack_kriging_grid(np.zeros((4, 5)), np.zeros((4, 4)), dx=1.0)


# --------------------------------------------------------------------------- #
# buffers: calculate_buffer_zones (literal calculate_buffer_zones.m /
# find_100_m_buffer.m translation)
# --------------------------------------------------------------------------- #
def test_calculate_buffer_zones_seven_level_ramp():
    # An 11x11 solid square has cells beyond 3 rings from its own edge, so
    # every one of the 7 non-zero/non-one weight levels is exercised.
    ny, nx = 25, 25
    mask = np.zeros((ny, nx), dtype=bool)
    mask[7:18, 7:18] = True

    w = calculate_buffer_zones(mask, ny, nx)

    assert w[12, 12] == 1.0  # deep interior
    np.testing.assert_allclose(w[7, 12], 4 / 7)  # inward ring 1 (element's own edge)
    np.testing.assert_allclose(w[8, 12], 5 / 7)  # inward ring 2
    np.testing.assert_allclose(w[9, 12], 6 / 7)  # inward ring 3
    np.testing.assert_allclose(w[6, 12], 3 / 7)  # outward ring 1
    np.testing.assert_allclose(w[5, 12], 2 / 7)  # outward ring 2
    np.testing.assert_allclose(w[4, 12], 1 / 7)  # outward ring 3
    assert w[3, 12] == 0.0  # beyond every ring
    assert set(np.unique(w).round(3)) == {
        0.0,
        0.143,
        0.286,
        0.429,
        0.571,
        0.714,
        0.857,
        1.0,
    }


def test_calculate_buffer_zones_zero_outside_bounding_box():
    ny, nx = 30, 30
    mask = np.zeros((ny, nx), dtype=bool)
    mask[5:8, 5:8] = True
    w = calculate_buffer_zones(mask, ny, nx)
    assert w[20:, :].sum() == 0.0  # far from the element: untouched, still 0


def test_calculate_buffer_zones_rejects_empty_mask():
    with pytest.raises(ValueError):
        calculate_buffer_zones(np.zeros((5, 5), dtype=bool), 5, 5)


# --------------------------------------------------------------------------- #
# windows: connected_components_4
# --------------------------------------------------------------------------- #
def test_connected_components_4_separates_disjoint_blobs():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True  # blob A
    mask[4:6, 4:6] = True  # blob B
    labels = connected_components_4(mask)

    a_labels = set(labels[0:2, 0:2].ravel())
    b_labels = set(labels[4:6, 4:6].ravel())
    assert len(a_labels) == 1 and len(b_labels) == 1
    assert a_labels != b_labels
    assert labels.max() == 2
    assert (labels[~mask] == 0).all()


def test_connected_components_4_diagonal_is_not_connected():
    mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )
    labels = connected_components_4(mask)
    assert labels[0, 0] != labels[1, 1]  # 4-connectivity: a diagonal touch doesn't join


# --------------------------------------------------------------------------- #
# windows: build_windows, krige_window / krige_layer
# --------------------------------------------------------------------------- #
def _small_grid(seed=0):
    from geosigma.testing import synthetic_field

    ny, nx = 20, 24
    surface = synthetic_field(nx, ny, 100.0, 400.0, 5.0, seed=seed)
    variance = np.full((ny, nx), 2.0)
    return pack_kriging_grid(surface, variance, dx=100.0), (ny, nx)


def test_build_windows_splits_disconnected_cluster_into_components():
    ny, nx = 12, 12
    cluster_labels = np.full((ny, nx), -1, dtype=int)  # -1: not in any cluster
    cluster_labels[1:4, 1:4] = 0
    cluster_labels[8:11, 8:11] = (
        0  # same label, spatially disjoint from the patch above
    )
    active_mask = np.ones((ny, nx), dtype=bool)
    conditioning_mask = np.zeros((ny, nx), dtype=bool)
    conditioning_mask[2, 2] = True
    conditioning_mask[9, 9] = True
    cluster_stats = {0: (300.0, 999.0)}  # sill deliberately bogus: must be ignored
    modeltheme = np.full((ny, nx), 7.0)

    grid = pack_kriging_grid(np.zeros((ny, nx)), np.ones((ny, nx)), dx=100.0)
    windows = build_windows(
        grid, cluster_labels, active_mask, conditioning_mask, cluster_stats, modeltheme
    )

    # The two disjoint 3x3 pieces of cluster 0 must become two separate windows.
    assert len(windows) == 2
    for w in windows:
        assert w.label == 0
        assert w.range_ == 300.0
        # Sill comes from modeltheme (uniform 7.0), not cluster_stats's 999.0.
        assert w.sill == pytest.approx(7.0)


def test_build_windows_rejects_conditioning_outside_active():
    cluster_labels = np.zeros((2, 2), dtype=int)
    active_mask = np.array([[True, False], [True, True]])
    conditioning_mask = np.array([[False, True], [False, False]])
    grid = pack_kriging_grid(np.zeros((2, 2)), np.ones((2, 2)), dx=100.0)
    with pytest.raises(ValueError):
        build_windows(
            grid,
            cluster_labels,
            active_mask,
            conditioning_mask,
            {0: (100.0, 1.0)},
            np.ones((2, 2)),
        )


def test_krige_window_matches_calling_core_functions_directly():
    grid, (ny, nx) = _small_grid()
    i_buf = np.ones(ny * nx, dtype=bool)
    ip_buf = np.zeros(ny * nx, dtype=bool)
    ip_buf[::7] = True  # a deterministic sparse subset

    window = KrigingWindow(
        label=0,
        component=1,
        i_buf=i_buf,
        ip_buf=ip_buf,
        range_=250.0,
        sill=4.0,
        buffer_weight=i_buf.astype(float),
    )
    result = krige_window(grid, window, cor_noise_rat=0.6)

    G, Cm, Cd, d_obs, m0 = local_kriging_setup_img(
        grid.as_global_dict(), i_buf, ip_buf, 4.0, 250.0, 0.6, dx=grid.dx, dy=grid.dy
    )
    m_est_ref, Cm_est_ref = least_squares_inversion(
        G, Cm, Cd, m0, d_obs, type=2, use_tqdm=False
    )

    np.testing.assert_allclose(result.m_est, m_est_ref)
    np.testing.assert_allclose(result.Cm_est, Cm_est_ref)


def test_krige_window_empty_conditioning_falls_back_to_prior():
    grid, (ny, nx) = _small_grid()
    i_buf = np.zeros(ny * nx, dtype=bool)
    i_buf[:10] = True
    ip_buf = np.zeros(ny * nx, dtype=bool)  # no conditioning points at all

    window = KrigingWindow(
        label=0,
        component=1,
        i_buf=i_buf,
        ip_buf=ip_buf,
        range_=250.0,
        sill=4.0,
        buffer_weight=i_buf.astype(float),
    )
    result = krige_window(grid, window, cor_noise_rat=0.6)

    np.testing.assert_allclose(result.m_est, 0.0)

    coords = np.column_stack(
        (grid.xx_norm[i_buf] * grid.dx, grid.yy_norm[i_buf] * grid.dy)
    )
    Cm_ref, _ = precal_cov(coords, coords, "4.0 Gau(250.0)")
    np.testing.assert_allclose(result.Cm_est, Cm_ref)


def test_krige_layer_returns_one_result_per_window():
    grid, (ny, nx) = _small_grid()
    cluster_labels = np.zeros((ny, nx), dtype=int)
    cluster_labels[:, nx // 2 :] = 1
    active_mask = np.ones((ny, nx), dtype=bool)
    conditioning_mask = np.zeros((ny, nx), dtype=bool)
    conditioning_mask.ravel()[::5] = True
    cluster_stats = {0: (300.0, 0.0), 1: (300.0, 0.0)}
    modeltheme = np.full((ny, nx), 4.0)

    windows = build_windows(
        grid, cluster_labels, active_mask, conditioning_mask, cluster_stats, modeltheme
    )
    results = krige_layer(grid, windows, cor_noise_rat=0.5)

    assert len(results) == len(windows)
    for r in results:
        assert np.isfinite(r.m_est).all()
        assert r.Cm_est.shape == (r.m_est.size, r.m_est.size)


# --------------------------------------------------------------------------- #
# simulation: simulate_window, blend_window_realizations, simulate_layer/model
# --------------------------------------------------------------------------- #
def test_simulate_window_is_zero_mean_and_uses_ambient_rng_state():
    Cm_est = np.eye(3) * 2.0
    dummy_window = KrigingWindow(
        label=0,
        component=1,
        i_buf=np.ones(3, bool),
        ip_buf=np.zeros(3, bool),
        range_=1.0,
        sill=1.0,
        buffer_weight=np.ones(3),
    )
    # m_est is deliberately non-zero; simulate_window must ignore it.
    result = KrigingResult(
        window=dummy_window, m_est=np.array([100.0, 200.0, 300.0]), Cm_est=Cm_est
    )

    np.random.seed(0)
    reals_a = simulate_window(result, n_reals=5)
    np.random.seed(0)
    reals_b = simulate_window(result, n_reals=5)

    assert reals_a.shape == (3, 5)
    np.testing.assert_array_equal(reals_a, reals_b)
    assert (
        np.abs(reals_a).max() < 50.0
    )  # nowhere near m_est's scale: confirms zero-mean


def test_blend_window_realizations_weighted_average_in_overlap():
    shape = (1, 3)  # 3 cells in a row
    n_cells = 3

    # Window A covers cells 0,1 (weight [1, .5]); window B covers 1,2 (weight [.5, 1]).
    i_buf_a = np.array([True, True, False])
    i_buf_b = np.array([False, True, True])
    win_a = KrigingWindow(0, 1, i_buf_a, i_buf_a, 1.0, 1.0, np.array([1.0, 0.5, 0.0]))
    win_b = KrigingWindow(1, 1, i_buf_b, i_buf_b, 1.0, 1.0, np.array([0.0, 0.5, 1.0]))

    result_a = KrigingResult(win_a, m_est=np.array([10.0, 20.0]), Cm_est=np.eye(2))
    result_b = KrigingResult(win_b, m_est=np.array([100.0, 200.0]), Cm_est=np.eye(2))

    reals_a = np.array([[10.0], [20.0]])  # (n_active=2, n_reals=1)
    reals_b = np.array([[100.0], [200.0]])

    blended = blend_window_realizations(
        shape,
        [result_a, result_b],
        [reals_a, reals_b],
        variance_correction=False,  # the raw local_kriger.m blend
    )

    # cell 0: only window A -> 10; cell 2: only window B -> 200
    # cell 1: weighted avg of A's 20 (w=0.5) and B's 100 (w=0.5) -> 60
    np.testing.assert_allclose(blended.mean.ravel(), [10.0, 60.0, 200.0])
    np.testing.assert_allclose(blended.reals.reshape(n_cells), [10.0, 60.0, 200.0])
    # post_std: both windows have Cm_est = I, so diag std = 1 everywhere covered
    np.testing.assert_allclose(blended.post_std.ravel(), [1.0, 1.0, 1.0])
    # Two equally weighted windows of equal variance overlap at cell 1, so
    # that is where the blend would shrink the draws by 1/sqrt(2).
    np.testing.assert_allclose(
        blended.deflation.ravel(), [1.0, np.sqrt(0.5), 1.0]
    )


def test_variance_correction_rescales_only_the_overlap():
    # Same geometry as above. The correction divides the draws by the
    # deflation factor, so single-window cells are untouched and the
    # overlap is scaled back up by sqrt(2).
    i_buf_a = np.array([True, True, False])
    i_buf_b = np.array([False, True, True])
    win_a = KrigingWindow(0, 1, i_buf_a, i_buf_a, 1.0, 1.0, np.array([1.0, 0.5, 0.0]))
    win_b = KrigingWindow(1, 1, i_buf_b, i_buf_b, 1.0, 1.0, np.array([0.0, 0.5, 1.0]))
    result_a = KrigingResult(win_a, m_est=np.array([10.0, 20.0]), Cm_est=np.eye(2))
    result_b = KrigingResult(win_b, m_est=np.array([100.0, 200.0]), Cm_est=np.eye(2))
    reals_a = np.array([[10.0], [20.0]])
    reals_b = np.array([[100.0], [200.0]])

    args = ((1, 3), [result_a, result_b], [reals_a, reals_b])
    raw = blend_window_realizations(*args, variance_correction=False)
    fixed = blend_window_realizations(*args, variance_correction=True)

    np.testing.assert_allclose(
        fixed.reals.reshape(3), [10.0, 60.0 / np.sqrt(0.5), 200.0]
    )
    # The posterior mean and std are blends of per-window estimates, not
    # draws, so the correction must leave them alone.
    np.testing.assert_allclose(fixed.mean, raw.mean)
    np.testing.assert_allclose(fixed.post_std, raw.post_std)


def test_variance_correction_is_a_no_op_without_overlap():
    # One window covering everything: nothing to correct, so the corrected
    # and raw blends must agree exactly.
    i_buf = np.array([True, True, True])
    win = KrigingWindow(0, 1, i_buf, i_buf, 1.0, 1.0, np.array([1.0, 1.0, 1.0]))
    result = KrigingResult(
        win, m_est=np.array([1.0, 2.0, 3.0]), Cm_est=np.diag([1.0, 4.0, 9.0])
    )
    reals = np.array([[1.0], [2.0], [3.0]])

    args = ((1, 3), [result], [reals])
    raw = blend_window_realizations(*args, variance_correction=False)
    fixed = blend_window_realizations(*args, variance_correction=True)

    np.testing.assert_allclose(fixed.deflation.ravel(), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(fixed.reals, raw.reals)


def test_variance_correction_restores_the_reported_posterior_std():
    # The point of the correction: draws whose spread matches the post_std
    # reported beside them. Two overlapping windows, unequal weights and
    # unequal variances, many realizations -> measure the realised std.
    rng = np.random.default_rng(0)
    n_reals = 20000
    sigma_a, sigma_b = 2.0, 5.0
    w_a, w_b = 0.75, 0.25

    i_buf_a = np.array([True, True])
    i_buf_b = np.array([False, True])
    win_a = KrigingWindow(0, 1, i_buf_a, i_buf_a, 1.0, 1.0, np.array([1.0, w_a]))
    win_b = KrigingWindow(1, 1, i_buf_b, i_buf_b, 1.0, 1.0, np.array([0.0, w_b]))
    result_a = KrigingResult(
        win_a, m_est=np.zeros(2), Cm_est=np.diag([sigma_a**2, sigma_a**2])
    )
    result_b = KrigingResult(win_b, m_est=np.zeros(1), Cm_est=np.diag([sigma_b**2]))

    reals_a = rng.normal(0, sigma_a, (2, n_reals))
    reals_b = rng.normal(0, sigma_b, (1, n_reals))

    args = ((1, 2), [result_a, result_b], [reals_a, reals_b])
    raw = blend_window_realizations(*args, variance_correction=False)
    fixed = blend_window_realizations(*args, variance_correction=True)

    target = fixed.post_std.ravel()[1]  # the overlap cell's reported std
    raw_std = raw.reals[0, 1, :].std()
    fixed_std = fixed.reals[0, 1, :].std()

    # Uncorrected, the draws understate the uncertainty they sit beside.
    assert raw_std < 0.95 * target
    # Corrected, they match it (3% allows for sampling noise at n=20000).
    assert abs(fixed_std - target) < 0.03 * target


def test_blend_window_realizations_uncovered_cell_is_nan():
    shape = (1, 2)
    i_buf = np.array([True, False])
    win = KrigingWindow(0, 1, i_buf, i_buf, 1.0, 1.0, np.array([1.0, 0.0]))
    result = KrigingResult(win, m_est=np.array([5.0]), Cm_est=np.eye(1))
    reals = np.array([[5.0]])

    blended = blend_window_realizations(shape, [result], [reals])
    assert blended.mean.ravel()[0] == 5.0
    assert np.isnan(blended.mean.ravel()[1])
    assert np.isnan(blended.reals.reshape(2)[1])
    assert np.isnan(blended.post_std.ravel()[1])


def _one_cluster_windows(grid, ny, nx, seed_points=4):
    cluster_labels = np.zeros((ny, nx), dtype=int)
    active_mask = np.ones((ny, nx), dtype=bool)
    conditioning_mask = np.zeros((ny, nx), dtype=bool)
    conditioning_mask.ravel()[::seed_points] = True
    cluster_stats = {0: (300.0, 0.0)}
    modeltheme = np.full((ny, nx), 4.0)
    return build_windows(
        grid, cluster_labels, active_mask, conditioning_mask, cluster_stats, modeltheme
    )


def test_simulate_layer_end_to_end_shapes_and_finite_inside_domain():
    grid, (ny, nx) = _small_grid(seed=1)
    windows = _one_cluster_windows(grid, ny, nx)

    result = simulate_layer(grid, windows, n_reals=6, cor_noise_rat=0.5, seed=42)

    assert result.mean.shape == (ny, nx)
    assert result.reals.shape == (ny, nx, 6)
    assert result.post_std.shape == (ny, nx)
    assert np.isfinite(result.mean).all()
    assert np.isfinite(result.reals).all()
    assert np.isfinite(result.post_std).all()


def test_simulate_layer_is_reproducible_given_seed():
    grid, (ny, nx) = _small_grid(seed=1)
    windows = _one_cluster_windows(grid, ny, nx)

    a = simulate_layer(grid, windows, n_reals=4, seed=7)
    b = simulate_layer(grid, windows, n_reals=4, seed=7)
    np.testing.assert_array_equal(a.reals, b.reals)


def test_simulate_model_uses_one_continuous_stream_not_independent_per_layer():
    grid, (ny, nx) = _small_grid(seed=2)
    windows = _one_cluster_windows(grid, ny, nx)
    layers = [(grid, windows), (grid, windows), (grid, windows)]

    model_result = simulate_model(layers, n_reals=3, seed=0)

    assert model_result.mean.shape == (ny, nx, 3)
    assert model_result.reals.shape == (ny, nx, 3, 3)
    assert model_result.post_std.shape == (ny, nx, 3)
    # different layers draw from the one advancing stream, so they differ
    assert not np.allclose(
        model_result.reals[:, :, 0, :], model_result.reals[:, :, 1, :], equal_nan=True
    )


def test_simulate_model_reproducible_and_matches_manual_single_stream():
    grid, (ny, nx) = _small_grid(seed=3)
    windows = _one_cluster_windows(grid, ny, nx)
    layers = [(grid, windows), (grid, windows)]

    a = simulate_model(layers, n_reals=2, seed=99)
    b = simulate_model(layers, n_reals=2, seed=99)
    np.testing.assert_array_equal(a.reals, b.reals)

    # Manually reproduce Main.m's "seed once, then just keep drawing" pattern
    # and check simulate_model's internal seeding matches it exactly.
    np.random.seed(99)
    manual_layer0 = simulate_layer(grid, windows, n_reals=2, seed=None)
    manual_layer1 = simulate_layer(grid, windows, n_reals=2, seed=None)

    np.testing.assert_array_equal(a.reals[:, :, 0, :], manual_layer0.reals)
    np.testing.assert_array_equal(a.reals[:, :, 1, :], manual_layer1.reals)


# --------------------------------------------------------------------------- #
# to_absolute: real = img_means + m_est + real  (Main.m:1038)
# --------------------------------------------------------------------------- #
def test_to_absolute_adds_lf_mean_back_to_mean_and_every_realization():
    grid, (ny, nx) = _small_grid(seed=1)
    windows = _one_cluster_windows(grid, ny, nx)
    layer = simulate_layer(grid, windows, n_reals=3, seed=11)
    lf_mean = np.arange(ny * nx, dtype=float).reshape(ny, nx)

    kriged, reals = to_absolute(layer, lf_mean)

    np.testing.assert_allclose(kriged, lf_mean + layer.mean)
    assert reals.shape == layer.reals.shape
    for k in range(reals.shape[2]):
        np.testing.assert_allclose(
            reals[:, :, k], lf_mean + layer.mean + layer.reals[:, :, k]
        )
