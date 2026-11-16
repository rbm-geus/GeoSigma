# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma.preprocess.clusters."""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.preprocess.clusters import (
    cluster_range_sill,
    cluster_stats,
    post_process_clusters,
)


def test_cluster_range_sill_separates_two_well_separated_groups():
    rng = np.random.default_rng(0)
    n = 40
    ranges = np.concatenate([rng.normal(100.0, 5.0, n), rng.normal(1500.0, 50.0, n)])
    sills = np.concatenate([rng.normal(5.0, 0.3, n), rng.normal(40.0, 2.0, n)])

    labels = cluster_range_sill(ranges, sills, n_clusters=2, seed=1)

    assert set(np.unique(labels)) == {0, 1}
    # Every point in the first half must share one label, every point in the
    # second half the other -- the groups are separated by orders of
    # magnitude, so k-means cannot mix them regardless of which label it
    # assigns to which group.
    first_half_labels = set(labels[:n])
    second_half_labels = set(labels[n:])
    assert len(first_half_labels) == 1
    assert len(second_half_labels) == 1
    assert first_half_labels != second_half_labels


def test_cluster_range_sill_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cluster_range_sill(np.ones(5), np.ones(4), n_clusters=2)


def test_cluster_range_sill_rejects_too_many_clusters():
    with pytest.raises(ValueError):
        cluster_range_sill(np.ones(3), np.ones(3), n_clusters=5)


def test_cluster_range_sill_som_separates_two_well_separated_groups():
    rng = np.random.default_rng(0)
    n = 40
    ranges = np.concatenate([rng.normal(100.0, 5.0, n), rng.normal(1500.0, 50.0, n)])
    sills = np.concatenate([rng.normal(5.0, 0.3, n), rng.normal(40.0, 2.0, n)])

    labels = cluster_range_sill(ranges, sills, n_clusters=4, method="som", seed=1)

    assert labels.min() >= 0 and labels.max() < 4
    first_half_labels = set(labels[:n].tolist())
    second_half_labels = set(labels[n:].tolist())
    assert len(first_half_labels) == 1
    assert len(second_half_labels) == 1
    assert first_half_labels != second_half_labels


def test_cluster_range_sill_som_requires_perfect_square():
    with pytest.raises(ValueError):
        cluster_range_sill(np.ones(9), np.ones(9), n_clusters=5, method="som")


def test_cluster_range_sill_rejects_unknown_method():
    with pytest.raises(ValueError):
        cluster_range_sill(np.ones(4), np.ones(4), n_clusters=2, method="bogus")


def test_cluster_stats_nearest_rank_odd_n():
    # n=3 per cluster: nearest-rank and np.median/np.percentile happen to
    # agree here, so this only checks the basic wiring.
    labels = np.array([0, 0, 0, 1, 1, 1])
    ranges = np.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])
    sills = np.array([1.0, 2.0, 100.0, 5.0, 6.0, 7.0])

    stats = cluster_stats(ranges, sills, labels, sill_percentile=0.5)

    assert set(stats) == {0, 1}
    assert stats[0] == (200.0, 2.0)
    assert stats[1] == (20.0, 6.0)


def test_cluster_stats_nearest_rank_not_interpolated_for_even_n():
    # n=4: MATLAB's ceil(perc*n) nearest-rank index differs from
    # np.median's linear interpolation between the two middle values.
    labels = np.zeros(4, dtype=int)
    ranges = np.array([10.0, 20.0, 30.0, 40.0])
    sills = np.array([10.0, 20.0, 30.0, 40.0])

    range_, sill = cluster_stats(ranges, sills, labels, sill_percentile=0.5)[0]

    assert range_ == 20.0  # sorted[ceil(0.5*4)-1] = sorted[1], NOT np.median's 25.0
    assert sill == 20.0
    assert range_ != float(np.median(ranges))


def test_cluster_stats_high_percentile_is_conservative():
    labels = np.zeros(5, dtype=int)
    sills = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    ranges = np.ones(5) * 50.0

    low = cluster_stats(ranges, sills, labels, sill_percentile=0.5)[0][1]
    high = cluster_stats(ranges, sills, labels, sill_percentile=0.95)[0][1]
    assert high > low


def test_cluster_stats_all_nan_cluster_falls_back_to_pooled_values():
    # Cluster 0 has no finite range at all; it must borrow the pooled
    # (finite-range AND finite-sill) points from every cluster instead,
    # matching prepare_clusters.m's `index2 = ~isnan(rm) & ~isnan(sm)`.
    labels = np.array([0, 0, 1, 1])
    ranges = np.array([np.nan, np.nan, 100.0, 300.0])
    sills = np.array([np.nan, 5.0, 10.0, 30.0])

    stats = cluster_stats(ranges, sills, labels, sill_percentile=0.5)

    # Pooled finite-range-and-sill points are only (100.0, 10.0) and (300.0, 30.0).
    assert stats[0] == (100.0, 10.0)


# --------------------------------------------------------------------------- #
# post_process_clusters
# --------------------------------------------------------------------------- #
def test_post_process_clusters_leaves_small_components_untouched():
    clust = np.zeros((10, 10), dtype=int)
    clust[1:4, 1:4] = 1  # 9 cells, well under maxsize
    out = post_process_clusters(clust, maxsize=100, seed=0)
    np.testing.assert_array_equal(out, clust)


def test_post_process_clusters_splits_oversized_component():
    ny, nx = 40, 40
    clust = np.zeros((ny, nx), dtype=int)
    clust[5:35, 5:35] = 1  # 900 cells

    out = post_process_clusters(clust, maxsize=100, seed=0)

    # The whole 900-cell blob exceeds maxsize, so no cell keeps label 1;
    # it is redistributed into ceil(900/100)=9 new labels appended after it.
    assert (out == 1).sum() == 0
    new_labels = set(np.unique(out).tolist()) - {0}
    assert new_labels == {2, 3, 4, 5, 6, 7, 8, 9, 10}
    # every originally-labelled cell must still be labelled (just possibly
    # renumbered), and nothing outside the original footprint gets labelled
    assert (out != 0).sum() == 900
    assert ((out != 0) == (clust != 0)).all()
    # no split segment may be larger than roughly maxsize
    for label in new_labels:
        assert (
            out == label
        ).sum() <= 100 + 50  # k-means split sizes aren't exactly equal


def test_post_process_clusters_numbers_multiple_splits_without_collision():
    ny, nx = 60, 30
    clust = np.zeros((ny, nx), dtype=int)
    clust[2:32, 2:12] = 1  # 300 cells: one oversized component of cluster 1
    clust[2:32, 18:28] = 2  # 300 cells: one oversized component of cluster 2

    out = post_process_clusters(clust, maxsize=100, seed=0)

    labels = set(np.unique(out).tolist()) - {0}
    # cluster 1 -> ceil(300/100)=3 new labels, cluster 2 -> 3 more, none reused
    assert len(labels) == 6
    assert (out == 1).sum() == 0
    assert (out == 2).sum() == 0


def test_post_process_clusters_skips_absent_labels_without_error():
    # Only label 2 is present even though unique(clust)-1 implies Nclust=2
    # (0 and 2 -> 2 uniques -> Nclust=1, so the loop only ever checks label 1,
    # which is absent -- matches the original's own assumption that labels
    # are contiguous 1..Nclust, not a validation this function adds).
    clust = np.zeros((10, 10), dtype=int)
    clust[0:3, 0:3] = 2
    out = post_process_clusters(clust, maxsize=1000, seed=0)
    np.testing.assert_array_equal(out, clust)
