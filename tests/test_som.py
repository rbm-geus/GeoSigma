# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma.preprocess.som (Kohonen SOM)."""

from __future__ import annotations

import numpy as np

from geosigma.preprocess.som import som_predict, train_som


def test_train_som_separates_two_well_separated_blobs():
    rng = np.random.default_rng(0)
    n = 40
    blob_a = rng.normal([0.0, 0.0], 0.05, (n, 2))
    blob_b = rng.normal([5.0, 5.0], 0.05, (n, 2))
    data = np.vstack([blob_a, blob_b]).T  # (2, 2n)

    weights, node_coords = train_som(data, (2, 2), n_iterations=300, seed=1)
    labels = som_predict(weights, data)

    assert weights.shape == (4, 2)
    assert node_coords.shape == (4, 2)
    assert labels.shape == (2 * n,)
    assert labels.min() >= 1 and labels.max() <= 4  # 1-indexed, matches vec2ind

    a_labels = set(labels[:n].tolist())
    b_labels = set(labels[n:].tolist())
    assert len(a_labels) == 1
    assert len(b_labels) == 1
    assert a_labels != b_labels


def test_train_som_reproducible_given_seed():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(3, 50))

    w1, _ = train_som(data, (2, 2), n_iterations=100, seed=42)
    w2, _ = train_som(data, (2, 2), n_iterations=100, seed=42)
    np.testing.assert_array_equal(w1, w2)


def test_train_som_handles_1x1_topology_without_crashing():
    # init_radius = max(1,1)/2 = 0.5 <= 1.0: exercises the log(radius)
    # division guard rather than the normal decay branch.
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 20))
    weights, node_coords = train_som(data, (1, 1), n_iterations=20, seed=0)
    assert weights.shape == (1, 2)
    labels = som_predict(weights, data)
    assert (labels == 1).all()


def test_som_predict_picks_nearest_weight():
    weights = np.array([[0.0, 0.0], [10.0, 10.0]])
    data = np.array([[0.1, 9.9], [0.1, 9.9]])  # (2 features, 2 samples)
    labels = som_predict(weights, data)
    np.testing.assert_array_equal(labels, [1, 2])
