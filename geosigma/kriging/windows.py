# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Local kriging windows: building them, and solving each one.

Literal translation of ``local_kriger.m``'s per-cluster-element loop
(everything up to, but not including, drawing realizations — see
:mod:`geosigma.kriging.simulation` for that). A **window** is one connected
component of one cluster — matching MATLAB's per-cluster ``bwlabel(...,4)``,
run separately for each cluster so a spatially scattered cluster becomes
several small, local kriging problems rather than one large, disjoint one —
buffered by :func:`~geosigma.kriging.buffers.calculate_buffer_zones`.

An earlier version of this module built one window per cluster label
(ignoring spatial contiguity) with an invented, parametrized buffer ramp, and
took *both* range and sill for kriging from
:func:`geosigma.preprocess.clusters.cluster_stats`. Both were wrong: see this
module's docstrings for what actually matches ``local_kriger.m`` now.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._connected_components import connected_components
from ..core.local_kriging_setup_img import local_kriging_setup_img
from ..core.least_squares_inversion import least_squares_inversion
from ..core.precal_cov import precal_cov
from .buffers import calculate_buffer_zones
from .state import KrigingGrid

__all__ = [
    "KrigingWindow",
    "KrigingResult",
    "connected_components_4",
    "build_windows",
    "krige_window",
    "krige_layer",
]


def connected_components_4(mask: np.ndarray) -> np.ndarray:
    """4-connected connected-component labeling (``bwlabel(mask, 4)``).

    Thin wrapper over :func:`geosigma._connected_components.connected_components`
    fixed at 4-connectivity, matching ``local_kriger.m``'s
    ``bwlabel(statmodcur, 4)``.
    """
    return connected_components(mask, connectivity=4)


@dataclass
class KrigingWindow:
    """One local kriging problem: one connected component of one cluster.

    Attributes
    ----------
    label : int
        The source cluster's label (bookkeeping only).
    component : int
        The connected-component id within that cluster (bookkeeping only).
    i_buf, ip_buf : ndarray of bool, shape (n_cells,)
        Flat masks (matching a
        :class:`~geosigma.kriging.state.KrigingGrid`'s flattening):
        ``i_buf`` = cells in this window's buffer zone (weight > 0);
        ``ip_buf`` = conditioning points within it. ``ip_buf`` is a subset of
        ``i_buf``.
    range_ : float
        This window's cluster's representative range, from
        :func:`geosigma.preprocess.clusters.cluster_stats`.
    sill : float
        This window's sill, sourced from ``modeltheme`` — the mean of the
        model-theme variance floor over this window's active cells — **not**
        from ``cluster_stats``. This matches ``local_kriger.m`` exactly: it
        reads ``clustmods(:,2)`` for range but recomputes ``curvariance``
        fresh as ``mean(modeltheme(i_buf))``, and never reads
        ``clustmods(:,3)`` (the cluster sill).
    buffer_weight : ndarray, shape (n_cells,)
        This window's blend weight per cell, from
        :func:`~geosigma.kriging.buffers.calculate_buffer_zones`: 1 deep in
        the element's interior, ramping to 0 over 3 cells in either direction
        across its boundary. Zero outside ``i_buf``.
    """

    label: int
    component: int
    i_buf: np.ndarray
    ip_buf: np.ndarray
    range_: float
    sill: float
    buffer_weight: np.ndarray


@dataclass
class KrigingResult:
    """The solved posterior for one :class:`KrigingWindow`."""

    window: KrigingWindow
    m_est: np.ndarray  # posterior mean, one entry per active cell (i_buf order)
    Cm_est: np.ndarray  # posterior covariance, (n_active, n_active)


def build_windows(
    grid: KrigingGrid,
    cluster_labels: np.ndarray,
    active_mask: np.ndarray,
    conditioning_mask: np.ndarray,
    cluster_stats: dict,
    modeltheme: np.ndarray,
) -> list[KrigingWindow]:
    """Build one :class:`KrigingWindow` per cluster connected component.

    Parameters
    ----------
    grid : KrigingGrid
        Supplies the grid shape and the residual surface used to identify
        numerically valid cells (matching MATLAB's
        ``~isnan(GLOBAL.img_FOHM_mask)`` conditioning-point check).
    cluster_labels : ndarray of int, shape (ny, nx)
        Per-cell cluster label. Cells outside ``active_mask`` are ignored.
    active_mask : ndarray of bool, shape (ny, nx)
        The full domain to estimate.
    conditioning_mask : ndarray of bool, shape (ny, nx)
        Cells that carry an actual observation. Must be a subset of
        ``active_mask``.
    cluster_stats : dict[int, (float, float)]
        Per-cluster ``(range, sill)`` from
        :func:`geosigma.preprocess.clusters.cluster_stats`. Only the range
        half is used here — see :attr:`KrigingWindow.sill`.
    modeltheme : ndarray, shape (ny, nx)
        This layer's model-theme variance floor
        (:func:`geosigma.themes.model_theme`), the source of each window's
        sill.

    Returns
    -------
    list[KrigingWindow]
        One window per (cluster, connected component) pair, matching
        ``local_kriger.m``'s ``for ic = 1:Nclust: ... bwlabel(...,4) ...
        for i = 1:Nlab``.
    """
    active_mask = np.asarray(active_mask, dtype=bool)
    conditioning_mask = np.asarray(conditioning_mask, dtype=bool)
    if not np.all(conditioning_mask <= active_mask):
        raise ValueError("conditioning_mask must be a subset of active_mask")

    ny, nx = grid.shape
    valid = np.isfinite(grid.img_dobs).reshape(ny, nx)
    modeltheme = np.asarray(modeltheme, dtype=float)

    windows = []
    for label, (range_, _unused_sill) in cluster_stats.items():
        footprint = (cluster_labels == label) & active_mask
        if not footprint.any():
            continue
        components = connected_components_4(footprint)
        for component_id in range(1, int(components.max()) + 1):
            element = components == component_id
            weight = calculate_buffer_zones(element, ny, nx)
            i_buf_2d = weight > 0
            ip_buf_2d = i_buf_2d & conditioning_mask & valid

            sill_cells = modeltheme[i_buf_2d]
            sill_cells = sill_cells[np.isfinite(sill_cells)]
            if not sill_cells.size:
                # No prior variance is defined anywhere under this window, so
                # there is nothing to krige with. Raised here, where the cause
                # is known, rather than left as a NaN sill to fail later as a
                # variogram-parsing error far from its origin.
                raise ValueError(
                    f"modeltheme has no finite values inside the window for "
                    f"cluster {int(label)}, component {component_id}: no prior "
                    "variance is defined there. Mask those cells out of "
                    "active_mask, or fill modeltheme over them."
                )
            sill = float(sill_cells.mean())

            windows.append(
                KrigingWindow(
                    label=int(label),
                    component=int(component_id),
                    i_buf=i_buf_2d.ravel(),
                    ip_buf=ip_buf_2d.ravel(),
                    range_=float(range_),
                    sill=sill,
                    buffer_weight=weight.ravel(),
                )
            )
    return windows


def krige_window(
    grid: KrigingGrid, window: KrigingWindow, *, cor_noise_rat: float = 0.5
) -> KrigingResult:
    """Solve one local kriging window.

    Wraps :func:`~geosigma.core.local_kriging_setup_img.local_kriging_setup_img`
    + :func:`~geosigma.core.least_squares_inversion.least_squares_inversion`,
    with the same "no conditioning points" fallback as ``local_kriger.m``:
    when a window has no data at all, build the prior covariance directly via
    :func:`~geosigma.core.precal_cov.precal_cov` and return it unconditioned
    (``m_est = 0``) rather than attempting an inversion with no data.

    ``cor_noise_rat`` defaults to ``0.5``, matching the hardcoded value
    ``local_kriger.m`` uses in production (it is not exposed as a parameter
    there at all).
    """
    if window.ip_buf.any():
        G, Cm, Cd, d_obs, m0 = local_kriging_setup_img(
            grid.as_global_dict(),
            window.i_buf,
            window.ip_buf,
            window.sill,
            window.range_,
            cor_noise_rat,
            dx=grid.dx,
            dy=grid.dy,
        )
        m_est, Cm_est = least_squares_inversion(
            G, Cm, Cd, m0, d_obs, type=2, use_tqdm=False
        )
    else:
        # No data to condition on: MATLAB builds Cm straight from the prior
        # variogram over the active cells and stops there (m_est = 0,
        # Cm_est = Cm). It hardcodes the *100 grid-spacing scaling here; we
        # use grid.dx/dy instead (the general form; equal to the original
        # whenever dx = dy = 100, the production grid spacing, and correct
        # rather than silently wrong for any other spacing).
        coords = np.column_stack(
            (
                grid.xx_norm[window.i_buf] * grid.dx,
                grid.yy_norm[window.i_buf] * grid.dy,
            )
        )
        statmod_var = f"{window.sill} Gau({window.range_})"
        Cm, _ = precal_cov(coords, coords, statmod_var)
        m_est = np.zeros(coords.shape[0])
        Cm_est = Cm

    return KrigingResult(window=window, m_est=m_est, Cm_est=Cm_est)


def krige_layer(
    grid: KrigingGrid, windows: list[KrigingWindow], *, cor_noise_rat: float = 0.5
) -> list[KrigingResult]:
    """Solve every window for one modelled layer."""
    return [krige_window(grid, w, cor_noise_rat=cor_noise_rat) for w in windows]
