# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma.preprocess.som_clusters (get_clusters.m translation)."""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.preprocess.som_clusters import build_kriging_zones, som_cluster_grid


def _synthetic_points(ny, nx, n_points, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((ny, nx), dtype=bool)
    idx = rng.choice(ny * nx, size=n_points, replace=False)
    mask.ravel()[idx] = True
    range_points = np.abs(rng.normal(300.0, 100.0, n_points))
    sill_points = np.abs(rng.normal(10.0, 3.0, n_points))
    return mask, range_points, sill_points


def test_output_shapes_and_sparsity():
    ny, nx = 30, 30
    mask, range_points, sill_points = _synthetic_points(ny, nx, 200, seed=0)

    classes, range_out, sill_out = som_cluster_grid(
        range_points,
        sill_points,
        (ny, nx),
        mask,
        n_points=200,
        maxclustsize=1000,
        n_training_iter=80,
        seed=1,
    )

    assert classes.shape == (ny, nx)
    assert range_out.shape == (ny, nx)
    assert sill_out.shape == (ny, nx)
    # classes is sparse: nonzero only where mask points existed originally
    # (post_process_clusters can renumber but never adds new nonzero cells)
    assert (classes != 0).sum() == mask.sum()
    assert ((classes != 0) == mask).all()


def test_rejects_mask_shape_mismatch():
    mask = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        som_cluster_grid(np.ones(1), np.ones(1), (10, 10), mask, n_points=1)


def test_rejects_point_array_length_mismatch():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError):
        som_cluster_grid(np.ones(2), np.ones(2), (5, 5), mask, n_points=2)


def test_nan_points_are_gap_filled_not_left_nan():
    ny, nx = 25, 25
    mask, range_points, sill_points = _synthetic_points(ny, nx, 150, seed=2)
    range_points[:10] = np.nan
    sill_points[:10] = np.nan

    classes, range_out, sill_out = som_cluster_grid(
        range_points,
        sill_points,
        (ny, nx),
        mask,
        n_points=150,
        maxclustsize=1000,
        n_training_iter=80,
        seed=3,
    )

    # the reference gap-fill pass assigns every originally-invalid point a
    # class and a backfilled range/sill via its neighbourhood's cluster mean
    assert (classes[mask] != 0).all()
    assert np.isfinite(range_out[mask]).all()


def test_topology_switches_with_n_points():
    # n_points < 20000 -> 2x2 = 4 classes; >= 20000 -> 3x3 = 9 classes.
    ny, nx = 20, 20
    mask, range_points, sill_points = _synthetic_points(ny, nx, 100, seed=4)

    classes_small, _, _ = som_cluster_grid(
        range_points,
        sill_points,
        (ny, nx),
        mask,
        n_points=100,
        maxclustsize=1000,
        n_training_iter=50,
        seed=5,
    )
    assert set(np.unique(classes_small[mask]).tolist()) <= {1.0, 2.0, 3.0, 4.0}

    classes_large, _, _ = som_cluster_grid(
        range_points,
        sill_points,
        (ny, nx),
        mask,
        n_points=25000,
        maxclustsize=1000,
        n_training_iter=50,
        seed=5,
    )
    assert max(np.unique(classes_large[mask]).tolist()) <= 9.0


# --------------------------------------------------------------------------- #
# build_kriging_zones: som_cluster_grid + cluster_stats + gap fill
# --------------------------------------------------------------------------- #
def test_build_kriging_zones_covers_every_cell_and_keys_match_labels():
    rng = np.random.default_rng(3)
    ny, nx = 24, 24
    range_field = np.abs(rng.normal(400.0, 120.0, (ny, nx)))
    sill_field = np.abs(rng.normal(12.0, 4.0, (ny, nx)))

    cluster_grid, stats = build_kriging_zones(
        range_field, sill_field, seed=1, n_training_iter=25
    )

    assert cluster_grid.shape == (ny, nx)
    # Gap-filled: build_windows needs every cell to belong to a zone, unlike
    # som_cluster_grid's own (deliberately sparse) output.
    assert (cluster_grid > 0).all()
    # Every label present has stats, and each is a (range, sill) pair.
    assert set(np.unique(cluster_grid)).issubset(set(stats))
    assert all(len(v) == 2 for v in stats.values())


def test_build_kriging_zones_ignores_nan_cells():
    rng = np.random.default_rng(4)
    ny, nx = 20, 20
    range_field = np.abs(rng.normal(400.0, 120.0, (ny, nx)))
    sill_field = np.abs(rng.normal(12.0, 4.0, (ny, nx)))
    range_field[:3, :] = np.nan  # a band with no usable estimate

    cluster_grid, stats = build_kriging_zones(
        range_field, sill_field, seed=1, n_training_iter=25
    )
    # NaN cells can't be clustered directly, but still get a zone from the
    # fill -- otherwise build_windows would leave that band unkriged.
    assert (cluster_grid > 0).all()
    assert len(stats) >= 1
