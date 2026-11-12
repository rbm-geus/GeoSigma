# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Grouping per-point range/sill estimates into kriging zones.

Replaces ``get_clusters.m`` + ``prepare_clusters.m`` + ``post_process_clusters.m``.
The per-point range/sill estimates from :mod:`geosigma.preprocess.rangesill`
are noisy and too fine-grained to krige with directly (one local kriging
system per point would be both wasteful and unstable); clustering groups
nearby points with similar (range, sill) into a handful of zones, each with
one representative range that :mod:`geosigma.kriging` uses for its local
kriging windows.

Two clustering paths, at two different levels of fidelity:

* :func:`cluster_range_sill` — a simple, point-array-only clusterer with a
  choice of backend (``method="kmeans"`` or ``method="som"``, see
  :mod:`geosigma.preprocess.som`). No grid awareness, no smoothing passes.
  This was the only path in an earlier version of this module, when it
  claimed (wrongly, without having read the source) that SOM "buys no
  statistical advantage" and wasn't worth implementing — see
  :mod:`geosigma.preprocess.som` for why that reasoning didn't hold up once
  the actual algorithm was read.
* :func:`~geosigma.preprocess.som_clusters.som_cluster_grid` (a separate
  module) — the actual, literal translation of ``get_clusters.m``: SOM
  clustering plus its whole-grid neighbourhood-smoothing and NaN-filling
  passes, finishing with :func:`post_process_clusters`. Use this path if you
  want output that corresponds to the reference pipeline's; use
  :func:`cluster_range_sill` if you just want *a* reasonable clustering and
  don't need that correspondence.

:func:`cluster_stats` (``prepare_clusters.m``) and :func:`post_process_clusters`
(``post_process_clusters.m``) are shared by both paths.
"""

from __future__ import annotations

import numpy as np

from .._connected_components import connected_components
from .som import som_predict, train_som

__all__ = [
    "cluster_range_sill",
    "cluster_stats",
    "post_process_clusters",
]


def _kmeans(
    X: np.ndarray, n_clusters: int, *, seed: int | None, n_init: int, max_iter: int
) -> np.ndarray:
    """Minimal Lloyd's-algorithm k-means (no external dependency).

    Returns integer labels, shape ``(n_samples,)``. Ties/empty clusters are
    handled by reseeding the empty centroid at a random sample.
    """
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) exceeds n_samples ({n_samples})")

    best_labels, best_inertia = None, np.inf
    for _ in range(n_init):
        centroids = X[rng.choice(n_samples, size=n_clusters, replace=False)]
        labels = np.zeros(n_samples, dtype=int)
        for _ in range(max_iter):
            dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dist, axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                labels = new_labels
                break
            labels = new_labels
            for k in range(n_clusters):
                members = X[labels == k]
                if len(members) == 0:
                    centroids[k] = X[rng.integers(0, n_samples)]
                else:
                    centroids[k] = members.mean(axis=0)
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        inertia = float(np.sum(dist[np.arange(n_samples), labels] ** 2))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels

    return best_labels


def cluster_range_sill(
    ranges: np.ndarray,
    sills: np.ndarray,
    n_clusters: int,
    *,
    method: str = "kmeans",
    seed: int | None = None,
    n_init: int = 10,
    max_iter: int = 100,
    som_iterations: int = 250,
) -> np.ndarray:
    """Cluster per-point (range, sill) pairs into ``n_clusters`` zones.

    Parameters
    ----------
    ranges, sills : ndarray, shape (n,)
        Per-point estimates, e.g. from
        :func:`~geosigma.preprocess.rangesill.estimate_range_sill_grid`.
    n_clusters : int
        Number of zones to form. With ``method="som"`` this must be a
        perfect square (the SOM is a square topology, as in ``get_clusters.m``
        — ``NN*[1 1]`` nodes, ``Nclust = NN^2``); pick e.g. 4 or 9.
    method : {"kmeans", "som"}, default "kmeans"
        ``"kmeans"``: :func:`~geosigma.preprocess.clusters._kmeans`, a
        Lloyd's-algorithm implementation with no external dependency.
        ``"som"``: :func:`~geosigma.preprocess.som.train_som` +
        :func:`~geosigma.preprocess.som.som_predict`, a Kohonen self-organizing
        map — see that module's docstring for how it relates to MATLAB's
        ``selforgmap``.
    seed : int, optional
        Seeds initialization for reproducibility.
    n_init, max_iter : int
        ``method="kmeans"`` only: number of random restarts (best-inertia
        kept) and iterations per restart.
    som_iterations : int, default 250
        ``method="som"`` only: single-sample training updates. The default
        borrows the number from ``get_clusters.m``'s
        ``selforgmap(NN*[1 1], 250)`` but is not equivalent to it -- see
        ``n_training_iter`` in
        :func:`~geosigma.preprocess.som_clusters.som_cluster_grid`.

    Returns
    -------
    ndarray, shape (n,)
        Integer cluster label per point, in ``[0, n_clusters)``.

    Notes
    -----
    Range and sill live on very different, non-comparable scales (metres vs.
    metres-squared, and both are typically log-distributed), so both are
    log-transformed and z-scored before clustering — without this, whichever
    of the two has the larger raw numeric spread would dominate the Euclidean
    distance both backends use.
    """
    ranges = np.asarray(ranges, dtype=float)
    sills = np.asarray(sills, dtype=float)
    if ranges.shape != sills.shape:
        raise ValueError("ranges and sills must have the same shape")

    log_r = np.log(np.clip(ranges, 1e-9, None))
    log_s = np.log(np.clip(sills, 1e-9, None))
    X = np.column_stack([log_r, log_s])
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - mean) / std

    if method == "kmeans":
        return _kmeans(Xz, n_clusters, seed=seed, n_init=n_init, max_iter=max_iter)
    elif method == "som":
        side = round(n_clusters**0.5)
        if side * side != n_clusters:
            raise ValueError(
                f"method='som' requires a perfect-square n_clusters, got {n_clusters}"
            )
        weights, _ = train_som(Xz.T, (side, side), som_iterations, seed=seed)
        return som_predict(weights, Xz.T) - 1  # 1-indexed -> [0, n_clusters)
    else:
        raise ValueError(f"method must be 'kmeans' or 'som', got {method!r}")


def cluster_stats(
    ranges: np.ndarray,
    sills: np.ndarray,
    labels: np.ndarray,
    *,
    sill_percentile: float = 0.95,
) -> dict[int, tuple[float, float]]:
    """Reduce each cluster's per-point estimates to one (range, sill) pair.

    Literal translation of ``prepare_clusters.m``'s per-cluster reduction
    (the ``out`` matrix it builds and saves). Two details matter for matching
    its output exactly, not just approximating it:

    * The representative **range** is not ``np.median`` but the MATLAB
      convention ``sorted[ceil(0.5 * n) - 1]`` (nearest-rank, not
      interpolated) — the two agree for odd ``n`` and can differ by one
      element for even ``n``.
    * The representative **sill** is likewise a nearest-rank percentile
      (``sorted[ceil(sill_percentile * n) - 1]``), not ``np.percentile``'s
      linear interpolation.
    * If every point in a cluster has a NaN range (data gap), both its range
      and sill are instead drawn from the pooled set of points, across every
      cluster, that have *both* a finite range and finite sill — matching
      ``prepare_clusters.m``'s ``index2 = ~isnan(rm) & ~isnan(sm)`` fallback.

    **Only the range half of this function's output is used for kriging.**
    ``geosigma.kriging.windows.build_windows`` takes the range from here but
    sources sill from ``modeltheme`` instead (see its docstring) — matching
    ``local_kriger.m``, which reads ``clustmods(:,2)`` (range) but never reads
    ``clustmods(:,3)`` (the sill computed here). The sill this function
    returns is still computed, for parity with the reference output and for
    callers who want it (e.g. diagnostics), but it is not plumbed into
    kriging.

    Parameters
    ----------
    ranges, sills, labels : ndarray, shape (n,)
        Per-point estimates and the cluster label of each (from
        :func:`cluster_range_sill`).
    sill_percentile : float, default 0.95
        Fraction in ``[0, 1]`` (matching MATLAB's ``perc`` argument, *not* a
        0-100 percentile) of each cluster's sills used as its representative
        sill.

    Returns
    -------
    dict[int, (float, float)]
        Maps each cluster label to its ``(range, sill)``.
    """
    ranges = np.asarray(ranges, dtype=float)
    sills = np.asarray(sills, dtype=float)
    labels = np.asarray(labels)

    pooled_valid = np.isfinite(ranges) & np.isfinite(sills)

    def _nearest_rank(sorted_values: np.ndarray, frac: float) -> float:
        idx = int(np.ceil(frac * sorted_values.size)) - 1
        idx = min(max(idx, 0), sorted_values.size - 1)
        return float(sorted_values[idx])

    stats: dict[int, tuple[float, float]] = {}
    for label in np.unique(labels):
        mask = labels == label
        r = ranges[mask]
        s = sills[mask]
        if not np.isfinite(r).any():
            # Whole cluster is a data gap: fall back to the pooled points
            # (across every cluster) that have both values defined.
            r = ranges[pooled_valid]
            s = sills[pooled_valid]

        r_sorted = np.sort(r[np.isfinite(r)])
        s_sorted = np.sort(s[np.isfinite(s)])
        stats[int(label)] = (
            _nearest_rank(r_sorted, 0.5),
            _nearest_rank(s_sorted, sill_percentile),
        )
    return stats


def _mode(values: np.ndarray) -> tuple[float, int]:
    """Mode and its count, ties broken towards the smallest value.

    Matches MATLAB's ``mode()``: among equally-frequent values it returns the
    smallest, which falls out for free from ``np.unique``'s ascending sort
    plus ``argmax``'s first-occurrence tie-break.
    """
    uniq, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    return float(uniq[best]), int(counts[best])


def post_process_clusters(
    clust: np.ndarray, maxsize: int, *, seed: int | None = None
) -> np.ndarray:
    """Split oversized connected cluster components into smaller pieces.

    Literal translation of ``post_process_clusters.m``. Local kriging's cost
    grows with a window's cell count, so a cluster component larger than
    ``maxsize`` is subdivided by k-means on its cells' own ``(row, col)``
    pixel coordinates (a purely spatial split, unrelated to range/sill) into
    ``ceil(size / maxsize)`` roughly-equal-sized new clusters, appended after
    the existing labels.

    Parameters
    ----------
    clust : ndarray of int, shape (ny, nx)
        A cluster-label grid using the convention ``0`` = background /
        unassigned, ``1..Nclust`` = classes, contiguous — the convention
        :func:`~geosigma.preprocess.som_clusters.som_cluster_grid` (or any
        ``get_clusters.m``-style clusterer) produces. This function assumes
        that convention exactly as ``post_process_clusters.m`` does (it reads
        the cluster count as ``len(unique(clust)) - 1`` and loops
        ``1..Nclust``), so a label set with gaps or not starting at 1 will
        silently skip whatever it doesn't visit — matching the original's
        own fragility here rather than adding a validation the reference
        pipeline doesn't have.
    maxsize : int
        Components at or below this size are left untouched.
    seed : int, optional
        Seeds the k-means split (MATLAB's ``kmeans`` call here is
        unseeded — the original run's split is whatever its ambient RNG state
        gave it). ``None`` (default) is itself a valid, non-reproducible
        choice, matching that.

    Returns
    -------
    ndarray of int, shape (ny, nx)
        ``clust`` with oversized components relabeled; new labels continue
        from ``Nclust + 1`` upward, so the output may have more than
        ``Nclust`` distinct non-zero labels.

    Notes
    -----
    Connected components here use ``bwlabel(mask)``'s MATLAB *default*
    connectivity, which is 8 — different from ``local_kriger.m``'s own
    ``bwlabel(mask, 4)`` (see :mod:`geosigma.kriging.windows`). Preserved as
    written; it is a real difference between how the two functions define
    "one contiguous piece", not a translation choice.
    """
    clust = np.asarray(clust, dtype=int).copy()
    n_clust = len(np.unique(clust)) - 1  # excludes background 0
    next_label = n_clust

    for i in range(1, n_clust + 1):
        cur_mask = clust == i
        if not cur_mask.any():
            continue
        sub_labels = connected_components(cur_mask, connectivity=8)
        for j in range(1, int(sub_labels.max()) + 1):
            local = sub_labels == j
            size_ = int(local.sum())
            if size_ > maxsize:
                rows, cols = np.nonzero(local)
                n_segments = int(np.ceil(size_ / maxsize))
                coords = np.column_stack([rows, cols]).astype(float)
                seg_labels = _kmeans(
                    coords, n_segments, seed=seed, n_init=10, max_iter=100
                )
                clust[local] = seg_labels + 1 + next_label
                next_label += n_segments

    return clust
