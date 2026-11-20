# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""A Kohonen self-organizing map, for clustering (range, sill) points.

``get_clusters.m`` uses MATLAB's ``selforgmap`` + ``train`` (Neural Network
Toolbox) on a square ``NN x NN`` topology. An earlier version of this library
claimed, without having read that source, that a SOM "buys no statistical
advantage over k-means for grouping two scalars" and skipped implementing one.
That claim doesn't survive contact with the actual algorithm:
``get_clusters.m``'s SOM is not just a clustering step but the seed for a
whole-grid neighbourhood-smoothing and gap-filling pipeline (see
:mod:`geosigma.preprocess.som_clusters`) that specifically exploits a SOM's
defining property k-means doesn't have — a topology, where nearby map nodes
represent similar (range, sill) combinations. That topology is what makes the
neighbourhood-mode cleanup passes behave sensibly (a cell's class getting
overridden by its spatial neighbours' mode tends to move it to an
*adjacent*, similar node, not an arbitrary other cluster). So it was worth
implementing after all.

This is a standard Kohonen SOM (competitive learning with a shrinking
Gaussian neighbourhood and learning rate), not a port of ``selforgmap``'s
internals, which are undocumented proprietary specifics of the MATLAB Neural
Network Toolbox (its two-phase ordering/tuning schedule, exact neighbourhood
function, exact weight initialization) that cannot be recovered from reading
the calling code. It reproduces the same *topology* (a square grid of nodes)
and *role* (competitive clustering with topological neighbourhood structure),
not MATLAB's exact trained weights.
"""

from __future__ import annotations

import numpy as np

__all__ = ["train_som", "som_predict"]


def train_som(
    data: np.ndarray,
    grid_shape: tuple[int, int],
    n_iterations: int,
    *,
    seed: int | None = None,
    init_learning_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a rectangular Kohonen SOM.

    Parameters
    ----------
    data : ndarray, shape (n_features, n_samples)
        Training data, features as rows — matching MATLAB's convention for
        ``selforgmap``'s input (and so what ``get_clusters.m`` passes it).
    grid_shape : (int, int)
        Number of SOM nodes along each topology axis, e.g. ``(2, 2)`` for
        MATLAB's ``NN*[1 1]`` with ``NN=2`` (4 nodes total).
    n_iterations : int
        Training iterations. Each presents one randomly drawn sample (online
        / stochastic training, not MATLAB's batch ordering+tuning schedule).
    seed : int, optional
        Seeds both weight initialization and sample order, for
        reproducibility.
    init_learning_rate : float, default 0.5
        Starting learning rate; decays to ~0 by the final iteration.

    Returns
    -------
    weights : ndarray, shape (n_nodes, n_features)
        Trained node weight vectors, node order matching ``node_coords``.
    node_coords : ndarray, shape (n_nodes, 2)
        Each node's ``(row, col)`` position in the topology grid — the basis
        for the neighbourhood function during training (and available to a
        caller for visualizing the map).
    """
    data = np.asarray(data, dtype=float)
    n_features, n_samples = data.shape
    gy, gx = grid_shape
    n_nodes = gy * gx
    if n_samples < 1:
        raise ValueError("data has no samples to train on")

    rng = np.random.default_rng(seed)

    node_coords = np.array([[i, j] for i in range(gy) for j in range(gx)], dtype=float)

    lo, hi = data.min(axis=1), data.max(axis=1)
    weights = rng.uniform(lo, hi, size=(n_nodes, n_features))

    init_radius = max(gy, gx) / 2.0
    # Guard log(1)=0: a 1x1 or effectively-unit radius topology has no decay
    # curve to fit, so just hold the radius roughly constant over training.
    time_const = (
        n_iterations / np.log(init_radius) if init_radius > 1.0 else n_iterations
    )

    samples = data.T  # (n_samples, n_features)
    for t in range(n_iterations):
        x = samples[rng.integers(0, n_samples)]

        dists = np.linalg.norm(weights - x, axis=1)
        bmu = int(np.argmin(dists))

        radius = init_radius * np.exp(-t / time_const)
        lr = init_learning_rate * np.exp(-t / n_iterations)

        node_dist = np.linalg.norm(node_coords - node_coords[bmu], axis=1)
        within = node_dist <= radius
        influence = np.exp(-(node_dist[within] ** 2) / (2.0 * radius**2))
        weights[within] += lr * influence[:, None] * (x - weights[within])

    return weights, node_coords


def som_predict(weights: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Assign each sample to its best-matching (nearest-weight) SOM node.

    Parameters
    ----------
    weights : ndarray, shape (n_nodes, n_features)
        From :func:`train_som`.
    data : ndarray, shape (n_features, n_samples)
        Same feature convention as :func:`train_som`'s ``data``.

    Returns
    -------
    ndarray of int, shape (n_samples,)
        **1-indexed** winning node per sample (``1..n_nodes``), matching
        MATLAB's ``vec2ind`` convention that ``get_clusters.m`` relies on.
    """
    data = np.asarray(data, dtype=float)
    samples = data.T  # (n_samples, n_features)
    dists = np.linalg.norm(samples[:, None, :] - weights[None, :, :], axis=2)
    return np.argmin(dists, axis=1) + 1
