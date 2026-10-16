# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Literal translation of ``get_clusters.m``.

This is the full reference pipeline, not just the SOM clustering step: it
normalizes range/sill, trains a SOM on them, then runs two whole-grid
neighbourhood-smoothing passes (a "how does the sparse point-level SOM
classification look once spread and cleaned up across the grid" step) before
finishing with :func:`~geosigma.preprocess.clusters.post_process_clusters`.
:func:`~geosigma.preprocess.clusters.cluster_range_sill` (in the sibling
module) is a much simpler point-array-only clusterer with no grid awareness
and no smoothing passes — use *this* function if you specifically want output
that corresponds to what the reference pipeline produces.

Coordinate convention
----------------------
MATLAB's ``mask`` is a set of *linear* (column-major) indices, and
``range``/``sill`` are ordered to match. This translation instead takes
``mask`` as a plain 2-D boolean grid, and expects ``range_points``/
``sill_points`` to be ordered to match ``np.flatnonzero(mask.ravel())`` —
i.e. row-major (C order), consistent with every other 2-D array in this
library. This is purely a bookkeeping convention (which order a temporary
index list is walked in); it doesn't change which grid cells anything refers
to, so it isn't a fidelity concession, just an adaptation to numpy's native
order instead of MATLAB's.
"""

from __future__ import annotations

import numpy as np

from .clusters import _mode, cluster_stats, post_process_clusters
from .som import som_predict, train_som

__all__ = ["som_cluster_grid", "build_kriging_zones"]


def som_cluster_grid(
    range_points: np.ndarray,
    sill_points: np.ndarray,
    mapsize: tuple[int, int],
    mask: np.ndarray,
    n_points: int,
    *,
    maxclustsize: int = 5000,
    n_training_iter: int = 250,
    cleanup_radius: int = 4,
    cleanup_radius_0: int = 32,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster range/sill points with a SOM, matching ``get_clusters.m``.

    Parameters
    ----------
    range_points, sill_points : ndarray, shape (n_mask,)
        Per-point range and **variance** (sill), ordered to match
        ``np.flatnonzero(mask.ravel())`` — see the module docstring.
    mapsize : (ny, nx)
        Full grid shape.
    mask : ndarray of bool, shape (ny, nx)
        Which grid cells carry a point.
    n_points : int
        ``ct`` in the original: chooses the SOM topology (``2x2`` below
        20000, else ``3x3``).
    maxclustsize : int, default 5000
        Passed to :func:`~geosigma.preprocess.clusters.post_process_clusters`
        at the end.
    n_training_iter : int, default 250
        SOM training iterations (``selforgmap(NN*[1 1], 250)``).
    cleanup_radius : int, default 4
        Neighbourhood half-width (cells) for the first smoothing pass.
    cleanup_radius_0 : int, default 32
        Starting neighbourhood half-width for the second pass's growing
        search (it starts at ``cleanup_radius_0 + 1`` and grows outward
        until it finds any classified neighbour at all).
    seed : int, optional
        Seeds the SOM's initialization and training sample order.

    Returns
    -------
    classes : ndarray of int, shape (ny, nx)
        ``0`` everywhere except at the original ``mask`` cells, which hold
        their final (smoothed, gap-filled, then
        :func:`~geosigma.preprocess.clusters.post_process_clusters`-split)
        cluster label. This sparsity is deliberate and matches the
        reference: the whole-grid smoothing passes are used to *reclassify
        the points themselves* via their neighbourhood, not to extend
        classification to the rest of the grid.
    range_out, sill_out : ndarray, shape (ny, nx)
        Denormalized range / (`sqrt`-transformed) sill, ``0`` outside
        ``mask`` except where the second smoothing pass filled a gap (see
        Notes).

    Notes
    -----
    ``sill_out`` is in **standard-deviation units**, not variance: the
    original converts sill to std (``sqrt(abs(sill))``) before normalizing
    and never converts back before returning ``sillout``. Preserved exactly;
    it does not matter for kriging fidelity because
    :mod:`geosigma.kriging.windows` sources its sill from ``modeltheme``, not
    from clustering output (see that module's docstring).

    The per-cluster mean used to backfill gaps (``mean_list`` in the
    original) is computed over **every** grid cell assigned that class after
    the first smoothing pass — not only the original point cells. Since that
    pass can propagate a class to non-point cells (whose ``range``/``sill``
    are 0, not missing-data), those zeros are included in the average,
    pulling it towards zero to a degree that depends on how much a class
    propagated beyond its original points. This looks like an odd design
    choice, but it is what the source does, and is preserved rather than
    "corrected" — the goal here is matching output, not second-guessing it.
    """
    ny, nx = mapsize
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (ny, nx):
        raise ValueError(f"mask shape {mask.shape} must equal mapsize {mapsize}")

    flat_idx = np.flatnonzero(mask.ravel())
    n_mask = flat_idx.size
    range_points = np.asarray(range_points, dtype=float)
    sill_points = np.asarray(sill_points, dtype=float)
    if range_points.shape != (n_mask,) or sill_points.shape != (n_mask,):
        raise ValueError(
            f"range_points/sill_points must have shape ({n_mask},) to match mask"
        )

    # --- normalize (range to [0,1]; sill: variance -> std, then to [0,1]) ---
    frange = np.nanmax(range_points)
    range_norm = range_points / frange
    sill_std = np.sqrt(np.abs(sill_points))
    fsill = np.nanmax(sill_std)
    sill_norm = sill_std / fsill

    valid = np.isfinite(range_norm)

    # --- SOM ---
    side = 2 if n_points < 20000 else 3
    n_clust = side * side

    features = np.column_stack([range_norm[valid], sill_norm[valid]]).T  # (2, k)
    weights, _ = train_som(features, (side, side), n_training_iter, seed=seed)
    winners = som_predict(weights, features)  # 1-indexed, length k

    preclasses = np.full(n_mask, np.nan)
    preclasses[valid] = winners

    classes_flat = np.zeros(ny * nx)
    classes_flat[flat_idx] = preclasses
    nn3 = classes_flat.reshape(ny, nx)

    rr4 = np.zeros(ny * nx)
    ss4 = np.zeros(ny * nx)
    rr4[flat_idx] = range_norm
    ss4[flat_idx] = sill_norm
    rr4 = rr4.reshape(ny, nx)
    ss4 = ss4.reshape(ny, nx)

    # --- pass 1: local-mode cleanup over the whole grid ---
    newclasses = np.zeros((ny, nx))
    for i in range(ny):
        i0, i1 = max(0, i - cleanup_radius), min(ny - 1, i + cleanup_radius)
        for j in range(nx):
            current_class = nn3[i, j]
            j0, j1 = max(0, j - cleanup_radius), min(nx - 1, j + cleanup_radius)

            neigh = nn3[i0 : i1 + 1, j0 : j1 + 1].ravel()
            neigh = neigh[neigh != 0]
            neigh = neigh[~np.isnan(neigh)]

            if neigh.size > 10:
                mode_class, n_mode = _mode(neigh)
                if n_mode > 0.2 * neigh.size and mode_class != current_class:
                    current_class = mode_class
            newclasses[i, j] = current_class

    nn4 = newclasses.copy()  # frozen snapshot: pass 2 reads nn4, writes newclasses

    # --- per-class mean range/sill over cells assigned that class (see Notes) ---
    rr4_flat, ss4_flat, nn4_flat = rr4.ravel(), ss4.ravel(), nn4.ravel()
    mean_range = np.full(n_clust + 1, np.nan)  # index 0 unused (1-indexed classes)
    mean_sill = np.full(n_clust + 1, np.nan)
    for k in range(1, n_clust + 1):
        in_k = nn4_flat == k
        r_sel = rr4_flat[in_k & ~np.isnan(rr4_flat)]
        s_sel = ss4_flat[in_k & ~np.isnan(ss4_flat)]
        if r_sel.size:
            mean_range[k] = r_sel.mean()
        if s_sel.size:
            mean_sill[k] = s_sel.mean()

    # --- pass 2: fill any still-NaN cells via a growing-radius neighbour search ---
    for i in range(ny):
        for j in range(nx):
            current_class = nn4[i, j]
            if np.isnan(current_class):
                radius = cleanup_radius_0
                neigh = np.array([])
                while neigh.size == 0:
                    radius += 1
                    i0, i1 = max(0, i - radius), min(ny - 1, i + radius)
                    j0, j1 = max(0, j - radius), min(nx - 1, j + radius)
                    neigh = nn4[i0 : i1 + 1, j0 : j1 + 1].ravel()
                    neigh = neigh[neigh != 0]
                    neigh = neigh[~np.isnan(neigh)]

                mode_class, _ = _mode(neigh)
                current_class = mode_class
                rr4[i, j] = mean_range[int(mode_class)]
                ss4[i, j] = mean_sill[int(mode_class)]
            newclasses[i, j] = current_class

    # --- assemble outputs: classes stays sparse (only at original mask cells) ---
    range_out = frange * rr4
    sill_out = fsill * ss4

    classes_flat_out = np.zeros(ny * nx)
    classes_flat_out[flat_idx] = newclasses.ravel()[flat_idx]
    classes = classes_flat_out.reshape(ny, nx)

    classes = post_process_clusters(classes, maxclustsize, seed=seed)

    return classes, range_out, sill_out


def _fill_nearest(grid: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill ``missing`` cells from their nearest non-missing neighbour."""
    from scipy.ndimage import distance_transform_edt

    _, (fill_rows, fill_cols) = distance_transform_edt(
        missing, return_distances=True, return_indices=True
    )
    return grid[fill_rows, fill_cols]


def build_kriging_zones(
    range_field: np.ndarray,
    sill_field: np.ndarray,
    *,
    seed: int | None = None,
    **som_kwargs,
) -> tuple[np.ndarray, dict]:
    """Turn dense range/sill fields into kriging zones plus their statistics.

    Wraps the three steps a caller otherwise has to sequence by hand, in an
    order that matters:

    1. :func:`som_cluster_grid` on the cells that have a finite estimate.
    2. :func:`~geosigma.preprocess.clusters.cluster_stats` on that result —
       **before** gap-filling, so each zone's representative range/sill comes
       only from cells the SOM actually classified.
    3. Fill any still-unlabelled cell from its nearest labelled neighbour.
       :func:`som_cluster_grid` deliberately labels only the cells it was
       given (matching ``get_clusters.m``), but
       :func:`~geosigma.kriging.windows.build_windows` needs every cell to
       belong to a zone or it leaves most of the grid unkriged.

    Parameters
    ----------
    range_field, sill_field : ndarray, shape (ny, nx)
        Dense per-cell estimates, e.g. from
        :func:`~geosigma.preprocess.rangesill.estimate_range_sill_surface`.
    seed : int, optional
        Seeds the SOM.
    **som_kwargs
        Passed through to :func:`som_cluster_grid` (``maxclustsize``,
        ``n_training_iter``, ``cleanup_radius``, ...).

    Returns
    -------
    cluster_grid : ndarray of int, shape (ny, nx)
        Zone label for every cell, gap-filled to full coverage.
    stats : dict[int, (float, float)]
        Per-zone ``(range, sill)``, as
        :func:`~geosigma.kriging.windows.build_windows` expects.

    Requires ``scipy`` (the gap fill), the same optional extra the rest of
    the preprocessing path needs: ``pip install "geosigma[nn]"``.
    """
    range_field = np.asarray(range_field, dtype=float)
    sill_field = np.asarray(sill_field, dtype=float)

    valid = np.isfinite(range_field) & np.isfinite(sill_field)
    ranges, sills = range_field[valid], sill_field[valid]

    cluster_grid, _, _ = som_cluster_grid(
        ranges,
        sills,
        mapsize=range_field.shape,
        mask=valid,
        n_points=int(valid.sum()),
        seed=seed,
        **som_kwargs,
    )
    stats = cluster_stats(ranges, sills, cluster_grid[valid])
    cluster_grid = _fill_nearest(cluster_grid, cluster_grid == 0)
    return cluster_grid, stats
