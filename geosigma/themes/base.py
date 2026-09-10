# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Shared engine for the variance-map themes.

What a "theme" is
-----------------
A *theme* is **a type of information source and the uncertainty model attached
to it** — not a data source and not a particular dataset. An information source
is anything that constrains where a layer boundary sits: a geophysical survey, a
borehole, a geological interpretation, a digitised cross-section, and so on. The
engine never touches the raw data. It is fed only:

* the **x, y locations** where a source of one type informs the boundary, and
* the per-location **attributes** that govern the uncertainty there — the
  interpreted boundary *depth* at that location, the layer *thickness*, the
  source's *depth-of-investigation*, etc.

From those, plus the *distance* from each grid cell to the nearest informed
location, it computes the **variance of interpreting that layer boundary's depth
given this type of information**. The deeper the boundary and the farther a cell
is from an informed location, the larger the variance — exactly how that grows
is the theme's parameterisation. The built-in themes happen to be geophysical
(PACES, GAMMALOG, RESLOG, REFSEIS, fewTEM, manyTEM, tTEM), but that is
incidental: a borehole or geological-interpretation theme is just another spec.

The algorithm (the same for every theme, only the parameters differ)
--------------------------------------------------------------------
1. **Windowed nearest-source pass.** Slide a ``±search_radius``-cell window over
   the grid; within it pick the single *nearest* informed location and record
   its distance plus that location's attributes (interpreted depth, thickness,
   depth-of-investigation), per layer. Where several locations sit at exactly the
   minimum distance, attributes are reduced over them (mean / min / max,
   theme-dependent).
2. **Complexity -> range.** A geological-complexity class grid is mapped to an
   effective correlation range, with a "prerequisite layer" override.
3. **Certainty -> variance.** ``cert = cert_fun(dist, range, kernel, 1/var0)``
   then ``variance = 1 / cert``; cells with no informing location get the
   :data:`NODATA_VARIANCE` sentinel, as do cells failing the theme's depth/DOI
   mask (e.g. the boundary lies below the source's depth-of-investigation).

Two knobs, two source archetypes
--------------------------------
The whole per-source behaviour is captured by two callables on the spec, and
between them they cover the common cases:

* ``mask_fn`` — *can this source see the boundary at all?* When the boundary
  depth exceeds what the source reached, the source carries no information and
  the cell gets :data:`NODATA_VARIANCE`. For a **borehole / well** this is the
  borehole's total depth: a layer deeper than the borehole is maximally
  uncertain from that well (``doi < depth``). For geophysics it is the
  method's depth-of-investigation.
* ``var0_fn`` — *how certain is the interpretation where the source does see
  the boundary?* Some methods have sensitivity that is roughly **constant with
  depth** (a flat ``var0``, e.g. reflection seismic); others **degrade with
  depth** (``var0`` grows with the boundary depth, e.g. PACES and the TEM
  methods). Either is just the formula chosen in the spec.

This module holds the generic, **source-agnostic, model-agnostic** machinery.
The per-theme numbers (search radius, range tables, the ``var0`` / ``kernel`` /
mask formulae, the certainty kernel) live in :mod:`geosigma.themes.themes` as
:class:`ThemeSpec` objects. Reading a particular source's data into locations +
attributes (e.g. the Danish ``.mat`` files, their column names, region
selection) is the job of *adapters* outside the library (see ``examples/``); the
engine only ever sees plain arrays.

Vectorisation
-------------
The original MATLAB nests two loops over grid cells (``ny * nx`` iterations).
Because the nearest-point assignment depends only on the *data-point* positions
— not on the layer — we instead loop over the (far fewer) **data points** and
scatter each point's contribution onto the block of grid cells whose search
window contains it. This removes the nested grid-cell loops entirely while
reproducing the window logic exactly; ``tests/test_themes.py`` asserts the
vectorised result is identical to a direct triple-loop port of the MATLAB code.

Coordinate convention
----------------------
``grid_x`` / ``grid_y`` are 1-D, **ascending** coordinate axes (east / north),
matching the MATLAB ``UTM_X`` / ``UTM_Y`` vectors. Returned grids are indexed
``[iy, ix]`` so that element ``[iy, ix]`` sits at ``(grid_x[ix], grid_y[iy])``.

Omitted: peatland / NPL
-----------------------
The MATLAB themes carry a peat-layer count ``NPL`` that (a) offsets the layer
origin and (b) prepends ``NPL`` constant-variance layers when
``include_peatlands`` is set. Both are intentionally dropped here (out of scope,
see ``CLAUDE.md``); the engine works purely in modelled-layer space
``0 .. n_layers-1``. Re-adding peat is an isolated wrapper that prepends constant
layers around this output — no engine change needed. The peat *content*
(``get_PL_themes``) is excluded entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

#: Variance sentinel for "no information" cells (the MATLAB ``100000``).
NODATA_VARIANCE = 100000.0


@dataclass
class ThemeData:
    """Model-agnostic inputs to a theme builder.

    Parameters
    ----------
    xs, ys : ndarray, shape (npoints,)
        Data-point coordinates (east, north), already deduplicated / date-filtered
        by the adapter.
    attributes : mapping of str -> ndarray, shape (npoints, n_layers)
        Per-point, per-layer model attributes the theme needs (e.g. ``"depth"``,
        ``"thick"``, ``"doi"``). A 1-D ``(npoints,)`` array is broadcast across
        layers. Which attributes are required is declared by the
        :class:`ThemeSpec`.
    complexity : ndarray of shape (ny, nx), optional
        Geological-complexity class grid (integer classes; ``0`` = unknown ->
        masked, ``1..4`` = complexity classes low..high). Indexed ``[iy, ix]``.
        The complexity map is a **Danish-specific** input (four classes, applied
        to the Quaternary). It is **optional**: when ``None`` (the default) the
        engine assumes a uniform complexity class ``1`` everywhere — i.e. a flat
        correlation range with no complexity dependence (the first non-unknown
        entry of each group's ``range_by_complexity``).
    """

    xs: np.ndarray
    ys: np.ndarray
    attributes: Mapping[str, np.ndarray]
    complexity: Optional[np.ndarray] = None


#: A layer selector for a :class:`RangeGroup`. One of:
#:
#: * the string ``"all"`` — every modelled layer;
#: * a 2-tuple ``(start, stop)`` of **0-based, inclusive** layer indices; either
#:   end may be ``None`` (``start=None`` -> first layer, ``stop=None`` -> last
#:   layer). This is the contiguous-range form;
#: * an explicit iterable of **0-based** layer indices (e.g. ``[0, 2, 5]``).
#:
#: Note the deliberate type split: a *tuple* is an inclusive range, a *list* is
#: an explicit index set. The YAML front-end (:mod:`geosigma.themes.spec_io`)
#: turns 1-based labels like ``"L01-L13"`` into the tuple form before
#: construction, so the engine itself never parses labels.
LayerSelector = Union[str, Tuple[Optional[int], Optional[int]], Sequence[int]]


@dataclass
class RangeGroup:
    """One group of layers sharing a complexity -> correlation-range table.

    This is the **model-agnostic** replacement for the Danish-specific
    ``n_prereq`` switch. A theme's range behaviour is a *list* of these groups
    (``ThemeSpec.range_model``), which must together cover every modelled layer
    exactly once:

    * **No regime change** — a single group with ``layers="all"``.
    * **Danish two-regime** (Quaternary / pre-Quaternary) — two groups split at
      the boundary layer.
    * **Per-layer variation** — one group per layer.

    Parameters
    ----------
    layers : LayerSelector
        Which modelled layers this group covers (see :data:`LayerSelector`).
    range_by_complexity : sequence of float
        Correlation range per complexity class. Index ``0`` is the *unknown*
        class (never used — class-0 cells are masked); indices ``1..4`` are the
        complexity classes. A length-5 list ``[unused, c1, c2, c3, c4]`` matches
        the Danish convention, but any length >= 2 is accepted.
    """

    layers: LayerSelector
    range_by_complexity: Sequence[float]


@dataclass
class ThemeSpec:
    """Per-theme parameters and formulae (the only thing that differs per theme).

    The three callables receive a mapping ``attrs`` of reduced attribute grids
    (each ``(ny, nx, n_layers)``) and must return arrays broadcastable to
    ``(ny, nx, n_layers)``. They encode the theme's ``var0``/``kernel``/mask
    formulae *verbatim* from the MATLAB source — the statistical model is never
    altered here.

    ``range_model`` is the model-agnostic per-layer correlation-range
    configuration (a list of :class:`RangeGroup`). It replaces the former
    Danish-specific ``n_prereq`` + two-table fields; the Danish two-regime case
    is now just a two-group ``range_model`` (see
    :mod:`geosigma.themes.themes`).
    """

    name: str
    search_radius: int
    range_model: Sequence[RangeGroup]
    cert_fun_name: str
    attributes: Sequence[str]
    reducers: Mapping[str, str]
    var0_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    kernel_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    mask_fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    #: Optional subset of layer indices that receive data; others -> NODATA.
    active_layers: Optional[Sequence[int]] = None
    #: Optional per-cell fix-up applied to the reduced attribute grids *after* the
    #: windowed nearest-point pass and *before* var0/kernel/mask. Receives and
    #: returns the ``attrs`` mapping (each ``(ny, nx, n_layers)``). Used by themes
    #: that fill missing values per-cell after reduction — e.g. PACEP/MEP fill a
    #: NaN depth-of-investigation with a constant / the mean DOI, which is *not*
    #: equivalent to filling per data point before reduction (a mixed NaN/valid
    #: tie set reduces to NaN, then gets filled).
    post_reduce_fn: Optional[
        Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]
    ] = None


_REDUCERS = ("mean", "min", "max", "first")


def _broadcast_attr(arr: np.ndarray, n_layers: int) -> np.ndarray:
    """Return ``arr`` shaped ``(npoints, n_layers)``, broadcasting a 1-D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = np.repeat(arr[:, None], n_layers, axis=1)
    if arr.shape[1] != n_layers:
        raise ValueError(
            f"attribute has {arr.shape[1]} layer columns, expected {n_layers}"
        )
    return arr


def windowed_nearest(
    xs: np.ndarray,
    ys: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    search_radius: int,
    attributes: Mapping[str, np.ndarray],
    reducers: Mapping[str, str],
    n_layers: int,
):
    """Vectorised windowed nearest-point pass.

    For every grid cell, find the nearest data point lying strictly inside the
    ``±search_radius``-cell coordinate window, and reduce the requested
    attributes over all points at exactly that minimum distance.

    Returns
    -------
    local_dist : ndarray, shape (ny, nx)
        Distance to the nearest in-window data point; ``NaN`` where the window
        held no points. (Layer-independent — the same for every layer.)
    reduced : dict of str -> ndarray, shape (ny, nx, n_layers)
        Each requested attribute, reduced over the minimum-distance ties for the
        ``mean``/``min``/``max`` reducers. The ``first`` reducer is the exception:
        it takes the value from the **first data point (lowest index) whose window
        contains the cell** — the whole window, not the tie set — matching the
        MATLAB ``local_X(i,j) = X(filt)(1)`` pattern (used by MEP's acquisition
        type). ``NaN`` where no point was found.

    Notes
    -----
    Loops over data points (not grid cells); each point scatters onto the block
    of cells whose window contains it. Distances use the same ``sqrt`` form and
    exact tie comparison as the MATLAB original, so the minimum-distance tie set
    matches bit-for-bit.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)
    nx = grid_x.size
    ny = grid_y.size
    R = int(search_radius)

    for name in attributes:
        if reducers.get(name) not in _REDUCERS:
            raise ValueError(
                f"attribute {name!r} needs a reducer in {_REDUCERS}, "
                f"got {reducers.get(name)!r}"
            )
    attrs = {n: _broadcast_attr(attributes[n], n_layers) for n in attributes}

    # Per-cell window bounds (open interval), with edge clamping, as in MATLAB:
    #   ymin = UTM_Y[max(0, i-R)] ; ymax = UTM_Y[min(ny-1, i+R)]
    idx_y = np.arange(ny)
    idx_x = np.arange(nx)
    ylo = grid_y[np.maximum(0, idx_y - R)]
    yhi = grid_y[np.minimum(ny - 1, idx_y + R)]
    xlo = grid_x[np.maximum(0, idx_x - R)]
    xhi = grid_x[np.minimum(nx - 1, idx_x + R)]

    # ---- Pass 1: minimum in-window distance per cell -------------------------
    best = np.full((ny, nx), np.inf)
    npoints = xs.size
    for p in range(npoints):
        px, py = xs[p], ys[p]
        rows = np.nonzero((ylo < py) & (py < yhi))[0]
        if rows.size == 0:
            continue
        cols = np.nonzero((xlo < px) & (px < xhi))[0]
        if cols.size == 0:
            continue
        d = np.sqrt(
            (grid_x[cols][None, :] - px) ** 2 + (grid_y[rows][:, None] - py) ** 2
        )
        block = np.ix_(rows, cols)
        best[block] = np.minimum(best[block], d)

    local_dist = np.where(np.isfinite(best), best, np.nan)

    # ---- Pass 2: reduce attributes over minimum-distance ties ----------------
    tie_names = [n for n in attrs if reducers[n] != "first"]
    count = np.zeros((ny, nx), dtype=np.int64)
    accum = {}
    for name in tie_names:
        red = reducers[name]
        if red == "mean":
            accum[name] = np.zeros((ny, nx, n_layers))
        elif red == "min":
            accum[name] = np.full((ny, nx, n_layers), np.inf)
        else:  # max
            accum[name] = np.full((ny, nx, n_layers), -np.inf)

    for p in range(npoints):
        px, py = xs[p], ys[p]
        rows = np.nonzero((ylo < py) & (py < yhi))[0]
        if rows.size == 0:
            continue
        cols = np.nonzero((xlo < px) & (px < xhi))[0]
        if cols.size == 0:
            continue
        d = np.sqrt(
            (grid_x[cols][None, :] - px) ** 2 + (grid_y[rows][:, None] - py) ** 2
        )
        tie = d == best[np.ix_(rows, cols)]
        if not tie.any():
            continue
        ti, tj = np.nonzero(tie)
        gi = rows[ti]
        gj = cols[tj]
        np.add.at(count, (gi, gj), 1)
        for name in tie_names:
            vals = attrs[name][p]  # (n_layers,)
            red = reducers[name]
            if red == "mean":
                np.add.at(accum[name], (gi, gj), vals)
            elif red == "min":
                np.minimum.at(accum[name], (gi, gj), vals)
            else:
                np.maximum.at(accum[name], (gi, gj), vals)

    has = count > 0
    reduced = {}
    for name in tie_names:
        red = reducers[name]
        out = np.full((ny, nx, n_layers), np.nan)
        if red == "mean":
            np.divide(accum[name], count[:, :, None], out=out, where=has[:, :, None])
        else:
            out[has] = accum[name][has]
        reduced[name] = out

    # ---- Pass 3: "first point in window" reducer -----------------------------
    # Window membership (not the tie set) decides; the lowest-index point covering
    # a cell wins. Iterating points in order and writing only where not yet seen
    # makes "first" win. All "first" attributes share the same winning point, so a
    # single per-cell ``seen`` mask drives them together.
    first_names = [n for n in attrs if reducers[n] == "first"]
    if first_names:
        seen = np.zeros((ny, nx), dtype=bool)
        for name in first_names:
            reduced[name] = np.full((ny, nx, n_layers), np.nan)
        for p in range(npoints):
            px, py = xs[p], ys[p]
            rows = np.nonzero((ylo < py) & (py < yhi))[0]
            if rows.size == 0:
                continue
            cols = np.nonzero((xlo < px) & (px < xhi))[0]
            if cols.size == 0:
                continue
            block = np.ix_(rows, cols)
            new = ~seen[block]
            if not new.any():
                continue
            ri, ci = np.nonzero(new)
            gi = rows[ri]
            gj = cols[ci]
            for name in first_names:
                reduced[name][gi, gj, :] = attrs[name][p]
            seen[block] = True

    return local_dist, reduced


def _resolve_group_layers(layers: LayerSelector, n_layers: int) -> List[int]:
    """Resolve a :data:`LayerSelector` to a list of 0-based layer indices.

    Out-of-range endpoints are clamped to ``0 .. n_layers-1`` (so an open-ended
    or oversized range simply contributes the layers that exist); whether the
    groups then tile the layer axis correctly is checked by :func:`_range_map`.
    """
    if isinstance(layers, str):
        if layers.lower() == "all":
            return list(range(n_layers))
        raise ValueError(
            f"unknown layer selector string {layers!r}; expected 'all', a "
            f"(start, stop) tuple, or an explicit list of indices"
        )
    if isinstance(layers, tuple) and len(layers) == 2:
        start, stop = layers
        start = 0 if start is None else int(start)
        stop = n_layers - 1 if stop is None else int(stop)
        start = max(0, start)
        stop = min(n_layers - 1, stop)
        return list(range(start, stop + 1))
    return [int(i) for i in layers]


def _range_map(complexity, range_model, n_layers):
    """Build the per-layer effective-range grid from a complexity class grid.

    Class ``0`` (unknown) -> ``NaN`` (cell ends up masked). For each
    :class:`RangeGroup`, classes ``1..`` map through that group's
    ``range_by_complexity`` table and the result is written to the group's
    layers. The groups must cover every modelled layer **exactly once** — a gap
    or an overlap is a configuration error and raises ``ValueError``.

    This is the model-agnostic generalisation of the MATLAB ``rangemap`` logic:
    the Danish two-regime behaviour (base table for the Quaternary layers, a
    pre-Quaternary override from ``n_prereq`` onward) is reproduced exactly by a
    two-group ``range_model``.
    """
    complexity = np.asarray(complexity, dtype=float)
    ny, nx = complexity.shape
    rangemap = np.full((ny, nx, n_layers), np.nan)
    covered = np.zeros(n_layers, dtype=int)

    for group in range_model:
        layers = _resolve_group_layers(group.layers, n_layers)
        table = np.asarray(group.range_by_complexity, dtype=float)
        base = np.full((ny, nx), np.nan)
        for c in range(1, table.size):
            base[complexity == c] = table[c]
        for layer in layers:
            rangemap[:, :, layer] = base
            covered[layer] += 1

    missing = np.nonzero(covered == 0)[0]
    if missing.size:
        raise ValueError(
            f"range_model leaves layer(s) {missing.tolist()} uncovered; every "
            f"modelled layer (0..{n_layers - 1}) must belong to exactly one group"
        )
    overlap = np.nonzero(covered > 1)[0]
    if overlap.size:
        raise ValueError(
            f"range_model covers layer(s) {overlap.tolist()} in more than one "
            f"group; groups must be disjoint"
        )
    return rangemap


def build_theme(
    data: ThemeData,
    spec: ThemeSpec,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_layers: int,
    window_fn=windowed_nearest,
):
    """Build a per-layer variance grid for one information-source theme.

    Parameters
    ----------
    window_fn : callable, optional
        The windowed nearest-source pass, defaulting to the vectorised
        :func:`windowed_nearest`. Exposed as a seam so tests can substitute a
        direct triple-loop reference and assert the full pipeline is identical
        under either windowing implementation.

    Returns
    -------
    ndarray, shape (ny, nx, n_layers)
        The theme's variance map; :data:`NODATA_VARIANCE` where the theme carries
        no usable information (no informing location in window, unknown
        complexity, or the theme's depth/DOI mask).
    """
    grid_x = np.asarray(grid_x, dtype=float)
    grid_y = np.asarray(grid_y, dtype=float)

    local_dist, attrs = window_fn(
        data.xs,
        data.ys,
        grid_x,
        grid_y,
        spec.search_radius,
        {n: data.attributes[n] for n in spec.attributes},
        spec.reducers,
        n_layers,
    )

    if spec.post_reduce_fn is not None:
        attrs = spec.post_reduce_fn(attrs)

    complexity = data.complexity
    if complexity is None:
        # No complexity map supplied -> uniform class 1 (flat range, no
        # complexity dependence). The complexity map is a Danish-specific input.
        complexity = np.ones((grid_y.size, grid_x.size))

    rangemap = _range_map(complexity, spec.range_model, n_layers)

    with np.errstate(invalid="ignore", divide="ignore"):
        var0map = spec.var0_fn(attrs)
        kernelmap = spec.kernel_fn(attrs)
        from .certainty_functions import certainty_function

        cert_fun = certainty_function(spec.cert_fun_name)
        dist3 = np.repeat(local_dist[:, :, None], n_layers, axis=2)
        certmap = cert_fun(dist3, rangemap, kernelmap, 1.0 / var0map)
        grid = 1.0 / certmap

    grid[np.isnan(grid)] = NODATA_VARIANCE
    grid[spec.mask_fn(attrs)] = NODATA_VARIANCE

    if spec.active_layers is not None:
        active = np.zeros(n_layers, dtype=bool)
        active[np.asarray(spec.active_layers, dtype=int)] = True
        grid[:, :, ~active] = NODATA_VARIANCE

    return grid
