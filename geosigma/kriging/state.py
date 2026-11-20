# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Grid packing for local kriging.

Replaces ``setup_GLOBAL_structure.m``. The MATLAB ``GLOBAL`` struct (a dict
with magic string keys, threaded by name through every kriging function)
becomes :class:`KrigingGrid`, a small typed container built once per modelled
layer by :func:`pack_kriging_grid`.

Buffer-zone weights (``calculate_buffer_zones.m`` / ``find_100_m_buffer.m``)
live in :mod:`geosigma.kriging.buffers` instead — an earlier version of this
module carried a from-scratch, parametrized buffer-ramp function here under
this name, which was a different algorithm from the original, not a
translation of it; see that module for the real one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["KrigingGrid", "pack_kriging_grid"]


@dataclass
class KrigingGrid:
    """Flat, kriging-ready view of one layer's residual surface and variance.

    Attributes
    ----------
    shape : (int, int)
        The original ``(ny, nx)`` grid shape, for reshaping results back.
    dx, dy : float
        Cell size, matching the ``dx``/``dy`` expected by
        :func:`~geosigma.core.local_kriging_setup_img.local_kriging_setup_img`.
    xx_norm, yy_norm : ndarray, shape (n_cells,)
        Grid-index coordinates (already divided by ``dx``/``dy``), flattened
        row-major to match ``surface``/``variance``.
    img_dobs : ndarray, shape (n_cells,)
        The flattened residual surface (what MATLAB called ``img_dobs``) —
        the deviation from a deterministic/background estimate that kriging
        conditions on.
    img_unc : ndarray, shape (n_cells,)
        The flattened per-cell **standard deviation** (``sqrt`` of the
        variance-map stack), matching the convention
        ``local_kriging_setup_img`` expects.
    """

    shape: tuple[int, int]
    dx: float
    dy: float
    xx_norm: np.ndarray
    yy_norm: np.ndarray
    img_dobs: np.ndarray
    img_unc: np.ndarray

    def as_global_dict(self) -> dict:
        """Return the plain dict ``local_kriging_setup_img`` consumes."""
        return {
            "xx_norm": self.xx_norm,
            "yy_norm": self.yy_norm,
            "img_dobs": self.img_dobs,
            "img_unc": self.img_unc,
        }


def pack_kriging_grid(
    surface: np.ndarray, variance: np.ndarray, *, dx: float, dy: float | None = None
) -> KrigingGrid:
    """Build a :class:`KrigingGrid` from a residual surface and its variance map.

    Parameters
    ----------
    surface : ndarray, shape (ny, nx)
        The residual surface to krige (a boundary's deviation from a
        deterministic/background estimate — what the reference pipeline calls
        ``img_dobs``).
    variance : ndarray, shape (ny, nx)
        The combined variance map for the same layer (e.g. from
        :func:`geosigma.themes.combine_variances`), in the same units as
        ``surface`` squared.
    dx : float
        Cell size in x.
    dy : float, optional
        Cell size in y; defaults to ``dx``.

    Returns
    -------
    KrigingGrid
    """
    surface = np.asarray(surface, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if surface.shape != variance.shape:
        raise ValueError(
            f"surface {surface.shape} and variance {variance.shape} must match"
        )
    dy = dx if dy is None else dy
    ny, nx = surface.shape

    row, col = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

    return KrigingGrid(
        shape=(ny, nx),
        dx=dx,
        dy=dy,
        xx_norm=col.ravel().astype(float),
        yy_norm=row.ravel().astype(float),
        img_dobs=surface.ravel(),
        img_unc=np.sqrt(variance).ravel(),
    )
