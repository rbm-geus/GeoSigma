# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""The borehole / well variance theme.

Why this theme has its own builder
----------------------------------
Every geophysical theme in :mod:`geosigma.themes.themes` shares one engine
(:func:`geosigma.themes.base.build_theme`): one nearest informed location per
cell, a four-argument certainty kernel with a plateau, and per-layer attributes
carried on a fixed point set. The well theme (MATLAB ``get_well_theme.m``) breaks
all three assumptions, so it gets a dedicated builder:

* **It sums the up-to-3 nearest wells**, not the single nearest. A cell's
  certainty is the sum of independent contributions from its three closest
  in-window boreholes (precision adds, like parallel variances).
* **Plain Gaussian decay, no plateau.** ``cert = sill * exp(-3 d² / range²)``
  — a three-argument kernel (distance, range, sill) with no plateau half-width,
  unlike the themed ``cert_fun(dist, range, width, sill)``.
* **Two well sets per layer.** Interpreted/snapped wells and Jupiter wells each
  contribute a precision grid; the variance is ``1 / (snap + jup)``.
* **Each layer is a different well list.** A boundary is informed by the wells
  that were interpreted (or that reach) *that* layer, so the inputs are per-layer
  well sets, not one point set with per-layer columns.
* **Range comes from the complexity at the well**, not at the grid cell.

Like the rest of :mod:`geosigma.themes`, this builder is model-agnostic: it sees
only prepared :class:`WellSet` arrays. Reading the Danish ``.mat`` wells, the
``terrain − z`` depth, the quality remap, the per-layer reach mask, and the
pre-Quaternary complexity override all live in an adapter (see ``examples/``).

``var0`` and the quality tables
-------------------------------
For a well of integer quality ``q`` (1-based) at depth-below-terrain ``z``::

    var0 = K[q]² + (gradient[q] · z)²
    sill = 1 / var0

``gradient`` and ``K`` are per-quality tables that differ between the snapped and
Jupiter sets (the snapped set is trusted more). They are carried on the
:class:`WellSet` so the builder stays free of any particular calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .base import NODATA_VARIANCE

#: Default complexity-class -> range table (class 0..4), shared with the
#: geophysical themes. Index by the complexity class *at the well*.
_COMP2RANGE = (100.0, 500.0, 400.0, 250.0, 100.0)


@dataclass
class WellSet:
    """One layer's worth of wells from a single source (snapped *or* Jupiter).

    All arrays are 1-D, length ``npoints`` (the wells informing *this* layer).

    Parameters
    ----------
    xs, ys : ndarray
        Well coordinates (east, north).
    depth : ndarray
        Depth of the boundary below terrain at the well, ``max(terrain − z, 0)``
        — the ``var0`` depth term. Computed by the adapter.
    quality : ndarray of int
        1-based borehole-quality class indexing ``gradients`` / ``Ks``.
    comp_class : ndarray of int
        Complexity class (0..4) sampled at the well; indexes the range table.
        NaN is treated as class 0 (matching MATLAB ``isnan -> 0``).
    gradients, Ks : sequence of float, length 4
        Per-quality ``var0`` tables (quality 1..4 -> index 0..3).
    """

    xs: np.ndarray
    ys: np.ndarray
    depth: np.ndarray
    quality: np.ndarray
    comp_class: np.ndarray
    gradients: Sequence[float]
    Ks: Sequence[float]


def _well_precision(grid_x, grid_y, w: WellSet, search_radius, comp2range, n_nearest):
    """Summed certainty (precision) grid for one well set, one layer.

    For each cell, gather the wells inside its ``±search_radius``-cell coordinate
    window, keep the ``n_nearest`` closest (MATLAB ``min(nwells, 3)``), and sum
    their Gaussian contributions ``sill · exp(-3 d² / range²)``.

    Implementation note: wells are binned by their nearest grid cell so each cell
    only examines wells in the surrounding ``(2R+1)²`` bin neighbourhood. Within a
    cell the candidate wells are visited in ascending original index order before
    the stable distance sort, so the up-to-3 selection breaks distance ties
    exactly as MATLAB's ``sort`` over ``wells(filt, :)`` does.
    """
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    nx, ny = grid_x.size, grid_y.size
    R = int(search_radius)
    prec = np.zeros((ny, nx))

    npts = w.xs.size
    if npts == 0:
        return prec

    q = np.asarray(w.quality, dtype=int) - 1
    K = np.asarray(w.Ks, dtype=float)[q]
    grad = np.asarray(w.gradients, dtype=float)[q]
    var0 = K**2 + (grad * np.asarray(w.depth, dtype=float)) ** 2
    sill = 1.0 / var0
    comp = np.where(np.isnan(w.comp_class), 0, w.comp_class).astype(int)
    rng = np.asarray(comp2range, dtype=float)[comp]

    xs = np.asarray(w.xs, dtype=float)
    ys = np.asarray(w.ys, dtype=float)

    # Per-cell open-window bounds (edge-clamped), identical to windowed_nearest.
    iy = np.arange(ny)
    ix = np.arange(nx)
    ylo = grid_y[np.maximum(0, iy - R)]
    yhi = grid_y[np.minimum(ny - 1, iy + R)]
    xlo = grid_x[np.maximum(0, ix - R)]
    xhi = grid_x[np.minimum(nx - 1, ix + R)]

    # Bin each well by its nearest cell (row, col).
    dx = float(grid_x[1] - grid_x[0]) if nx > 1 else 1.0
    dy = float(grid_y[1] - grid_y[0]) if ny > 1 else 1.0
    wcol = np.round((xs - grid_x[0]) / dx).astype(int)
    wrow = np.round((ys - grid_y[0]) / dy).astype(int)
    bins: dict[tuple[int, int], list[int]] = {}
    for p in range(npts):
        bins.setdefault((wrow[p], wcol[p]), []).append(p)

    for i in range(ny):
        for j in range(nx):
            cand: list[int] = []
            for bi in range(i - R - 1, i + R + 2):
                for bj in range(j - R - 1, j + R + 2):
                    lst = bins.get((bi, bj))
                    if lst:
                        cand.extend(lst)
            if not cand:
                continue
            c = np.array(sorted(cand))  # ascending original index -> stable ties
            cx, cy = xs[c], ys[c]
            inwin = (cx < xhi[j]) & (cx > xlo[j]) & (cy < yhi[i]) & (cy > ylo[i])
            if not inwin.any():
                continue
            c = c[inwin]
            d = np.sqrt((grid_x[j] - xs[c]) ** 2 + (grid_y[i] - ys[c]) ** 2)
            order = np.argsort(d, kind="stable")[:n_nearest]
            sel = c[order]
            dsel = d[order]
            prec[i, j] += np.sum(sill[sel] * np.exp(-3.0 * dsel**2 / rng[sel] ** 2))

    return prec


def build_well_theme(
    grid_x,
    grid_y,
    snap_layers: Sequence[Optional[WellSet]],
    jup_layers: Sequence[Optional[WellSet]],
    *,
    search_radius: int = 6,
    comp2range: Sequence[float] = _COMP2RANGE,
    n_nearest: int = 3,
):
    """Build the per-layer well variance grid (MATLAB ``get_well_theme`` stacked).

    Parameters
    ----------
    grid_x, grid_y : ndarray
        Ascending model coordinate axes (east, north).
    snap_layers, jup_layers : sequence of (:class:`WellSet` or None), len n_layers
        The interpreted/snapped and Jupiter well sets for each modelled layer.
        ``None`` (or an empty set) means that source informs no wells in that
        layer. A layer with *no* wells from either source is left at the
        no-information sentinel.
    search_radius : int, default 6
        Window half-width in cells.
    comp2range : sequence of float, default (100, 500, 400, 250, 100)
        Complexity-class -> range table, indexed by the well's complexity class.
    n_nearest : int, default 3
        Maximum number of nearest wells summed per cell.

    Returns
    -------
    ndarray, shape (ny, nx, n_layers)
        Variance ``1 / (snap_precision + jup_precision)``; the
        :data:`~geosigma.themes.base.NODATA_VARIANCE` sentinel where neither
        source contributes (MATLAB leaves these as ``Inf``; we follow the other
        themes and use the sentinel so the grid carries no NaN/Inf).
    """
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    nx, ny = grid_x.size, grid_y.size
    n_layers = len(snap_layers)
    if len(jup_layers) != n_layers:
        raise ValueError("snap_layers and jup_layers must have equal length")

    out = np.full((ny, nx, n_layers), NODATA_VARIANCE)
    for s in range(n_layers):
        prec = np.zeros((ny, nx))
        for wset in (snap_layers[s], jup_layers[s]):
            if wset is not None and np.asarray(wset.xs).size:
                prec += _well_precision(
                    grid_x, grid_y, wset, search_radius, comp2range, n_nearest
                )
        with np.errstate(divide="ignore"):
            var = np.where(prec > 0, 1.0 / prec, NODATA_VARIANCE)
        out[:, :, s] = var

    return out
