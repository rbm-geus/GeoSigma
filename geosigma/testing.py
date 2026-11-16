# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Synthetic test fixtures shared by tests, examples and training scripts.

A single place to generate "a field with a *known* range and sill" so golden
tests, the range/sill estimator's training data, and the demo scripts all draw
from the same generator instead of each hand-rolling their own. Built entirely
on the already-translated ``geosigma.core`` primitives (``precal_cov`` +
``get_reals_cholesky``), so it carries no dependency of its own.
"""

from __future__ import annotations

import numpy as np

from .core.precal_cov import precal_cov
from .core.get_reals_cholesky import get_reals_cholesky


def synthetic_field(
    nx: int,
    ny: int,
    dx: float,
    range_: float,
    sill: float,
    *,
    dy: float | None = None,
    model: str = "Gau",
    seed: int | None = None,
) -> np.ndarray:
    """Draw one realization of a stationary Gaussian random field.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions (columns, rows).
    dx : float
        Cell size in x. Used for y too unless ``dy`` is given.
    range_, sill : float
        Variogram range and sill (see :func:`~geosigma.core.precal_cov.precal_cov`).
    dy : float, optional
        Cell size in y; defaults to ``dx`` (square cells).
    model : str, default "Gau"
        Variogram model name understood by
        :func:`~geosigma.core.deformat_variogram.deformat_variogram` ("Gau",
        "Sph", ...).
    seed : int, optional
        Seeds ``numpy.random`` before drawing (via ``np.random.seed``, matching
        :func:`~geosigma.core.get_reals_cholesky.get_reals_cholesky`'s use of
        the legacy global RNG).

    Returns
    -------
    ndarray, shape (ny, nx)
        One realization, row 0 at y=0 (row-major, matching a raster array).
    """
    dy = dx if dy is None else dy
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="xy")  # both shape (ny, nx)
    coords = np.column_stack((xx.ravel(), yy.ravel()))

    statmod = f"{sill} {model}({range_})"
    cov, _ = precal_cov(coords, coords, statmod)

    if seed is not None:
        np.random.seed(seed)
    field = get_reals_cholesky(cov, Nreals=1).ravel()
    return field.reshape(ny, nx)
