# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Stacking kriged layer boundaries into a physically valid model.

Layers are kriged independently -- each gets its own conditioning points,
range/sill estimates, clusters and realizations -- which is what makes the
uncertainty on each boundary honest, but also means nothing stops two
boundaries from crossing. A lower boundary drawn above the one above it
describes a layer of negative thickness.

``get_layered_models.m`` resolves this per realization, top-down: wherever a
boundary rises above the one above it, it is pushed down to meet it. The
layer pinches out to zero thickness there instead of inverting. Doing it per
realization rather than on the ensemble mean matters -- correcting the mean
would leave the individual draws invalid, and the mean of the corrected
draws is not the correction of the mean, because the clamp is nonlinear.

What is deliberately not translated from ``get_layered_models.m``: its
pre-Quaternary binding branch (``ilay < NQlay && ilay >= 2``), which is
Danish region-specific logic driven by ``landsdel_switch``, and the
``Onshoremask``/``img_m_est + img_reals == 0`` test for "not simulated here",
which is a convention of that pipeline's masking rather than part of
stacking. Unsimulated cells are expressed as NaN on input instead, and NaN
is treated exactly as MATLAB treats it: the boundary inherits the one above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["LayeredModel", "clamp_to_upper", "stack_realizations"]


@dataclass
class LayeredModel:
    """A stack of layer boundaries that never cross.

    Attributes
    ----------
    surfaces : ndarray, shape (ny, nx, n_boundaries, n_reals)
        Every boundary of every realization, top-down along axis 2, in
        absolute elevation. When a ``terrain`` was supplied it is boundary 0
        and is identical across realizations.
    mean_model : ndarray, shape (ny, nx, n_boundaries)
        The ensemble mean of ``surfaces``. Note this is the mean of the
        *corrected* draws, which is not the same as correcting the mean --
        and is the one that corresponds to the realizations.
    n_clamped : ndarray of int, shape (n_boundaries,)
        How many (cell, realization) pairs the clamp moved for each
        boundary. Zero for boundary 0. A diagnostic: a boundary with a large
        count is one whose independent kriging disagreed with the layer
        above it often, which is worth knowing before trusting its
        thicknesses.
    """

    surfaces: np.ndarray
    mean_model: np.ndarray
    n_clamped: np.ndarray


def clamp_to_upper(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Push ``lower`` down to ``upper`` wherever it rises above it.

    The single comparison behind the whole stack, translated from
    ``get_layered_models.m``::

        indexchange = Grid > prevGrid | isnan(Grid);
        tempGrid(indexchange) = prevGrid(indexchange);

    Parameters
    ----------
    lower, upper : ndarray
        Elevations of the lower and upper boundary, same shape. ``upper`` is
        taken as given and never modified.

    Returns
    -------
    ndarray
        ``lower``, with every cell at or below ``upper``. NaN in ``lower``
        means "not simulated here" and inherits ``upper``, as in the
        reference. NaN in ``upper`` leaves ``lower`` untouched: there is no
        constraint to apply, and MATLAB arrives at the same place, since
        both ``Grid > NaN`` and ``isnan(Grid)`` are false there.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    # `lower > upper` is already False wherever lower is NaN, so the isnan
    # term is what carries MATLAB's explicit `| isnan(Grid)`.
    return np.where((lower > upper) | np.isnan(lower), upper, lower)


def stack_realizations(
    layer_reals: Sequence[np.ndarray],
    terrain: np.ndarray | None = None,
) -> LayeredModel:
    """Stack independently kriged layers, clamping each to the one above.

    Parameters
    ----------
    layer_reals : sequence of ndarray, each (ny, nx, n_reals)
        One entry per modelled boundary, ordered **top-down**, in absolute
        elevation -- i.e. what
        :func:`~geosigma.kriging.simulation.to_absolute` returns for each
        layer. All entries must share a shape; realization ``r`` of every
        layer must come from the same draw index, since they are stacked
        together into one model.
    terrain : ndarray, shape (ny, nx), optional
        An upper bound for the first layer, such as a terrain model or a
        known top surface. It becomes boundary 0 of the result and is
        repeated across realizations (it carries no kriging uncertainty of
        its own). When omitted, the first layer is unconstrained from above
        and is itself the top of the stack.

    Returns
    -------
    LayeredModel

    Raises
    ------
    ValueError
        If ``layer_reals`` is empty, the entries disagree on shape, or
        ``terrain`` does not match their grid.
    """
    if not layer_reals:
        raise ValueError("layer_reals is empty: nothing to stack")

    reals = [np.asarray(a, dtype=float) for a in layer_reals]
    shape = reals[0].shape
    if any(a.shape != shape for a in reals):
        raise ValueError(
            "every layer must have the same (ny, nx, n_reals) shape; got "
            + ", ".join(str(a.shape) for a in reals)
        )
    ny, nx, n_reals = shape

    boundaries = list(reals)
    if terrain is not None:
        terrain = np.asarray(terrain, dtype=float)
        if terrain.shape != (ny, nx):
            raise ValueError(
                f"terrain shape {terrain.shape} does not match the layer grid "
                f"{(ny, nx)}"
            )
        boundaries.insert(0, np.repeat(terrain[:, :, None], n_reals, axis=2))

    surfaces = np.empty((ny, nx, len(boundaries), n_reals))
    n_clamped = np.zeros(len(boundaries), dtype=int)

    surfaces[:, :, 0, :] = boundaries[0]
    for i, layer in enumerate(boundaries[1:], start=1):
        above = surfaces[:, :, i - 1, :]
        clamped = clamp_to_upper(layer, above)
        # A cell counts as moved only if its value actually changed, so a
        # boundary already resting exactly on the one above does not inflate
        # the diagnostic. NaN != NaN, so a cell that was unsimulated and
        # stayed unsimulated has to be excluded explicitly.
        moved = (clamped != layer) & ~(np.isnan(clamped) & np.isnan(layer))
        n_clamped[i] = int(np.count_nonzero(moved))
        surfaces[:, :, i, :] = clamped

    return LayeredModel(
        surfaces=surfaces,
        mean_model=surfaces.mean(axis=3),
        n_clamped=n_clamped,
    )
