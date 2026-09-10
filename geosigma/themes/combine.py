# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Combine per-theme variance grids into the final per-layer variance map.

Mirrors the assembly in the MATLAB ``Main.m`` (the ``final_variance_themes``
block), kept **verbatim**:

* themes combine as independent Gaussian precisions —
  ``final = 1 / sum(1 / theme)`` over the theme axis. A theme with ``+inf``
  variance contributes exactly ``1/inf = 0`` precision. Note the builder's
  no-information sentinel is the finite ``NODATA_VARIANCE`` (``100000``), which
  contributes a small but **non-zero** precision (``1e-5``), not 0 — this is
  faithful to the MATLAB, which uses the same ``100000`` sentinel (so a cell
  where all ``N`` themes are no-information combines to ``100000 / N``);
* the result is floored from below by a depth-dependent minimum,
  ``min_map = (0.01 * (terrain - layer_bottom))^2`` (``get_minimum_map.m``);
* :func:`corr_map` builds the zero-thickness tie grid from wells
  (``get_corr_maps.m``), used downstream to enforce pinch-outs.

Note: the MATLAB ``Main.m`` adds ``0.0625`` to the peat layers before flooring;
that branch is part of the excluded peatland subsystem and is **not** ported.
"""

from __future__ import annotations

import numpy as np


def combine_variances(themes):
    """Combine theme variance grids as parallel precisions.

    Parameters
    ----------
    themes : sequence of ndarray, each shape (ny, nx, n_layers)
        Per-theme variance grids on the same grid. Use ``+inf`` where a theme
        should contribute exactly zero precision. The builder instead emits the
        finite :data:`geosigma.themes.base.NODATA_VARIANCE` (``100000``)
        sentinel for no-information cells, which contributes a small non-zero
        precision (``1e-5``); both are accepted and combined verbatim, matching
        the MATLAB (which uses the ``100000`` sentinel throughout).

    Returns
    -------
    ndarray, shape (ny, nx, n_layers)
        ``1 / sum(1 / theme)`` over the themes — the combined variance.
    """
    if len(themes) == 0:
        raise ValueError("combine_variances requires at least one theme grid")
    stack = np.stack([np.asarray(t, dtype=float) for t in themes], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.sum(1.0 / stack, axis=-1)
        combined = 1.0 / precision
    return combined


def minimum_map(terrain, layer_bottoms):
    """Depth-dependent variance floor ``(0.01 * (terrain - bottom))^2`` per layer.

    Parameters
    ----------
    terrain : ndarray, shape (ny, nx)
        Terrain / top-surface elevation.
    layer_bottoms : sequence of ndarray, each (ny, nx), or ndarray (ny, nx, n_layers)
        Bottom-surface elevation of each layer.

    Returns
    -------
    ndarray, shape (ny, nx, n_layers)
    """
    terrain = np.asarray(terrain, dtype=float)
    if isinstance(layer_bottoms, np.ndarray) and layer_bottoms.ndim == 3:
        bottoms = layer_bottoms
    else:
        bottoms = np.stack([np.asarray(b, dtype=float) for b in layer_bottoms], axis=-1)
    depth = terrain[:, :, None] - bottoms
    return (0.01 * depth) ** 2


def apply_floor(variance, min_map):
    """Raise ``variance`` to at least ``min_map`` element-wise (the MATLAB floor).

    Equivalent to ``np.maximum`` but written to match the MATLAB
    ``floor_filter = variance < min_map; variance(floor_filter) = min_map(...)``.
    """
    variance = np.array(variance, dtype=float, copy=True)
    below = variance < min_map
    variance[below] = np.asarray(min_map, dtype=float)[below]
    return variance


def corr_map(
    grid_x, grid_y, well_x, well_y, well_quality, zero_thickness, dx=None, dy=None
):
    """Zero-thickness tie grid from wells (``get_corr_maps.m``).

    Each well flagged with a zero layer thickness stamps its quality rating into
    the grid cell it falls in; all other cells are ``NaN`` ("no tie").

    Parameters
    ----------
    grid_x, grid_y : ndarray
        1-D ascending coordinate axes (east, north).
    well_x, well_y : ndarray, shape (nwells,)
        Well coordinates.
    well_quality : ndarray, shape (nwells,)
        Per-well quality rating — the **full, unfiltered** per-well array (the
        MATLAB ``wellsp.bor_qual``), not pre-filtered to the zero-thickness
        subset. See the note on quality indexing below.
    zero_thickness : ndarray of bool, shape (nwells,)
        True for wells with a defined zero layer thickness (the only ones used).
    dx, dy : float, optional
        Cell sizes; default to the spacing of ``grid_x`` / ``grid_y``.

    Returns
    -------
    ndarray, shape (ny, nx)
        Tie grid, ``NaN`` away from zero-thickness wells.

    Notes
    -----
    The MATLAB used a hardcoded 100 m cell size and ``ceil`` indexing from the
    lower-left corner; here the spacing is taken from the axes. The cell index is
    ``ceil((coord - min) / d) - 1`` (0-based), clamped to the grid, reproducing
    the MATLAB ``ceil`` binning relative to the lower-left corner.

    **Quality indexing is bit-faithful to MATLAB.** ``get_corr_maps.m`` takes the
    zero-thickness wells' *coordinates* (``xutm``/``yutm`` filtered by the
    zero-thickness flag) but their *quality* from the unfiltered ``bor_qual`` by
    position — i.e. the first ``n_zwells`` entries, ``zwells_q = wellsp.bor_qual``
    then ``curq = zwells_q(a)``. This means well ``a``'s coordinate is paired with
    the ``a``-th well's quality overall, not the ``a``-th zero-thickness well's.
    That is almost certainly a MATLAB bug, but it is reproduced here so the
    validation comparison against the MATLAB maps is clean (see the open question
    in ``TRANSLATION_PLAN.md``). When the zero-thickness wells are the leading
    wells, the two interpretations coincide.
    """
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    nx, ny = grid_x.size, grid_y.size
    if dx is None:
        dx = float(grid_x[1] - grid_x[0])
    if dy is None:
        dy = float(grid_y[1] - grid_y[0])

    grid = np.full((ny, nx), np.nan)
    zt = np.asarray(zero_thickness, dtype=bool)
    if not zt.any():
        return grid

    n_zwells = int(zt.sum())
    wx = np.asarray(well_x, dtype=float)[zt]
    wy = np.asarray(well_y, dtype=float)[zt]
    # Bit-faithful to get_corr_maps.m: quality comes from the unfiltered array
    # by position (first n_zwells entries), NOT from the zero-thickness subset.
    wq = np.asarray(well_quality, dtype=float)[:n_zwells]

    llx = grid_x.min()
    lly = grid_y.min()
    ix = np.ceil((wx - llx) / dx).astype(int) - 1
    iy = np.ceil((wy - lly) / dy).astype(int) - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    grid[iy[inside], ix[inside]] = wq[inside]
    return grid
