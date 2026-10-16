# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Drawing realizations and blending overlapping kriging windows.

Replaces the realization-drawing and buffer-blend parts of ``local_kriger.m``
and the layer loop in Main.m's "SECTION: Kriging and Simulation" — minus the
``.mat`` checkpointing, ``redo_*`` flags, email notifications and
``Nreals_pre`` batch-seed bookkeeping, none of which are part of what
simulation *is*. A caller who wants resumability or progress notifications
wraps these functions with their own logic; that is not this module's job.

Two things an earlier version of this module got wrong, now corrected to
match ``local_kriger.m`` / Main.m:

* **Realizations are zero-mean.** ``local_kriger.m`` calls
  ``get_reals_cholesky(Cm_est, nreals)`` with no mean argument, and blends
  ``img_m_est`` (the posterior mean) and ``img_reals`` (the zero-mean draws)
  into *separate* accumulator images — the posterior mean is added back onto
  a realization only later, in Main.m's "Applying Ties" section. The earlier
  version of :func:`simulate_window` baked ``result.m_est`` into the draw
  immediately, which is numerically recoverable to the same total but is not
  what the reference pipeline's intermediate outputs look like.
* **One continuous RNG stream, not one per window/layer.** Main.m calls
  ``rng(seed)`` exactly once, before the entire multi-layer kriging loop; every
  subsequent ``get_reals_cholesky`` call draws from that same advancing
  stream. The earlier version derived an independent seed per window (and per
  layer), which changes the draw sequence relative to the reference even
  though both are valid random streams. :func:`simulate_layer` and
  :func:`simulate_model` now only seed (via ``numpy.random.seed``) once, at
  the entry point you actually call with a ``seed``.

One thing ``local_kriger.m`` itself gets wrong, departed from on purpose:
the buffer blend averages *independently drawn* window realizations, which
suppresses their variance wherever windows overlap — down to ``1/sqrt(2)``
of the intended amplitude at equal weight, while the posterior std blended
alongside them reports no such thing. ``variance_correction=True`` (the
default, see :func:`blend_window_realizations`) rescales the draws to the
variance they are supposed to have; pass ``False`` for bit-for-bit MATLAB
agreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.get_reals_cholesky import get_reals_cholesky
from .state import KrigingGrid
from .windows import KrigingResult, KrigingWindow, krige_layer

__all__ = [
    "LayerRealizations",
    "ModelRealizations",
    "simulate_window",
    "blend_window_realizations",
    "simulate_layer",
    "simulate_model",
    "to_absolute",
]


@dataclass
class LayerRealizations:
    """The finished result for one modelled boundary.

    Attributes
    ----------
    mean : ndarray, shape (ny, nx)
        Blended posterior mean (MATLAB ``img_m_est``). NaN where no window
        covered a cell.
    reals : ndarray, shape (ny, nx, n_reals)
        Blended **zero-mean** realizations (MATLAB ``img_reals``) — add
        ``mean`` back in yourself for absolute values; the reference pipeline
        keeps them separate at this stage too (see the module docstring).
        NaN where no window covered a cell.
    post_std : ndarray, shape (ny, nx)
        Blended posterior standard deviation (MATLAB ``img_post_std``),
        ``sqrt(diag(Cm_est))`` per window, blended the same way as ``mean``.
        NaN where no window covered a cell.
    deflation : ndarray, shape (ny, nx)
        How much the buffer blend would shrink realization amplitude at each
        cell — 1.0 under a single window, towards ``1/sqrt(n)`` under ``n``
        equally weighted ones. Diagnostic, and the factor
        ``blend_window_realizations(..., variance_correction=True)`` divides
        the draws by; it is reported either way, so it is also a map of the
        error you are carrying when the correction is off. NaN where no
        window covered a cell.
    """

    mean: np.ndarray
    reals: np.ndarray
    post_std: np.ndarray
    deflation: np.ndarray


@dataclass
class ModelRealizations:
    """The finished result across all modelled boundaries of a model.

    Attributes
    ----------
    mean : ndarray, shape (ny, nx, n_layers)
    reals : ndarray, shape (ny, nx, n_layers, n_reals)
    post_std : ndarray, shape (ny, nx, n_layers)
    deflation : ndarray, shape (ny, nx, n_layers)
    """

    mean: np.ndarray
    reals: np.ndarray
    post_std: np.ndarray
    deflation: np.ndarray


def simulate_window(result: KrigingResult, n_reals: int) -> np.ndarray:
    """Draw zero-mean realizations from one window's posterior.

    Thin wrapper over
    :func:`~geosigma.core.get_reals_cholesky.get_reals_cholesky` with no mean
    argument (matching ``get_reals_cholesky(Cm_est, nreals)`` in
    ``local_kriger.m`` — the posterior mean is tracked separately, see the
    module docstring), kept as its own function so window-level realizations
    can be inspected or unit-tested before blending.

    Draws from whatever the current global ``numpy.random`` state is; this
    function never seeds it itself — see :func:`simulate_layer` /
    :func:`simulate_model`.

    Returns
    -------
    ndarray, shape (n_active, n_reals)
        One column per realization, in the order of ``result.window.i_buf``'s
        active cells.
    """
    return get_reals_cholesky(result.Cm_est, Nreals=n_reals)


def to_absolute(
    layer: LayerRealizations, lf_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Add the low-frequency mean back onto a kriged layer and its draws.

    Kriging runs on the residual
    (:attr:`~geosigma.preprocess.rangesill.RangeSillSurface.residual`), and
    :class:`LayerRealizations` keeps the posterior mean and the zero-mean
    draws separate, so getting back to absolute surfaces means summing all
    three — ``real = img_means + m_est + real`` (Main.m:1038). The
    counterpart of the ``surface - lf_mean`` that produced the residual in
    the first place.

    Parameters
    ----------
    layer : LayerRealizations
        From :func:`simulate_layer`.
    lf_mean : ndarray, shape (ny, nx)
        The low-frequency background model the residual was defined against.

    Returns
    -------
    kriged_surface : ndarray, shape (ny, nx)
        ``lf_mean + layer.mean`` — the kriged estimate in absolute units.
    realizations : ndarray, shape (ny, nx, n_reals)
        Each realization in absolute units.
    """
    lf_mean = np.asarray(lf_mean, dtype=float)
    kriged_surface = lf_mean + layer.mean
    realizations = lf_mean[:, :, None] + layer.mean[:, :, None] + layer.reals
    return kriged_surface, realizations


def _deflation(var_sum: np.ndarray, std_sum: np.ndarray) -> np.ndarray:
    """How much the buffer blend shrinks realization amplitude, per cell.

    The blend ``sum(w_i r_i) / sum(w_i)`` of *independent* zero-mean window
    draws has standard deviation ``sqrt(sum(w_i^2 sigma_i^2)) / sum(w_i)``,
    while the posterior std it is supposed to represent blends as
    ``sum(w_i sigma_i) / sum(w_i)``. The ratio of the two is

        sqrt(sum(w_i^2 sigma_i^2)) / sum(w_i sigma_i)

    which is 1 where a single window covers a cell and falls towards
    ``1/sqrt(n)`` where ``n`` windows of equal weight and variance overlap —
    the familiar ``Var((A+B)/2) = V/2``. Note the common ``sum(w_i)``
    cancels, so this depends only on how the weight is *split* between
    windows, not on its magnitude.
    """
    deflation = np.ones_like(std_sum)
    # std_sum == 0 means every contributing window has zero posterior
    # variance there (a conditioning point): the draw is 0, nothing to scale.
    nonzero = std_sum > 0
    deflation[nonzero] = np.sqrt(var_sum[nonzero]) / std_sum[nonzero]
    return deflation


def blend_window_realizations(
    shape: tuple[int, int],
    results: Sequence[KrigingResult],
    window_reals: Sequence[np.ndarray],
    *,
    variance_correction: bool = True,
) -> LayerRealizations:
    """Weighted-sum blend of overlapping windows into one full-grid layer.

    Literal translation of the accumulation in ``local_kriger.m``'s main
    loop: ``new_surface = m_est .* bufz_img(i_buf) ./ img_buf_sum(i_buf)``
    (and the analogous line for realizations and for
    ``sqrt(diag(Cm_est))``), accumulated into ``img_m_est`` / ``img_reals`` /
    ``img_post_std``. ``img_buf_sum`` there is exactly the sum, over every
    window, of that window's buffer weight at a cell — which is what
    ``weight_sum`` below computes directly, rather than requiring a separate
    precomputed pass.

    Parameters
    ----------
    shape : (ny, nx)
        The full grid shape to blend onto.
    results : sequence of KrigingResult
        One entry per window (from :func:`~geosigma.kriging.windows.krige_layer`).
    window_reals : sequence of ndarray
        One ``(n_active, n_reals)`` array per window, matching ``results`` in
        order (from :func:`simulate_window`).
    variance_correction : bool, default True
        Whether to undo the variance suppression the blend introduces in
        window overlaps.

        **This is a deliberate deviation from the reference.** Each window
        draws its realizations independently, so a weighted average of them
        has variance ``sum(w^2 sigma^2) / (sum w)^2`` — not the
        ``(sum(w sigma) / sum w)^2`` that the blended ``post_std`` reports.
        Where two windows overlap at equal weight the draws come out at
        ``1/sqrt(2)`` of their intended amplitude: the realizations
        understate the uncertainty, by up to 30% in the overlaps, while the
        posterior std sitting next to them in the same object says
        otherwise. ``local_kriger.m`` has the same artefact; nothing about
        it is specific to this translation.

        ``True`` divides the draws by :func:`_deflation`, restoring the
        marginal variance to exactly what ``post_std`` claims, cell by cell.
        It is exact for marginals, costs one extra accumulator, and leaves
        the posterior mean untouched. It does **not** fully reconstruct the
        spatial covariance across a seam: averaging independent fields
        smooths them, and rescaling restores amplitude without restoring
        that structure. Fixing it properly would mean drawing the windows
        from a shared random field rather than independent ones.

        ``False`` reproduces ``local_kriger.m`` bit for bit. Use it when
        comparing against MATLAB output.

    Returns
    -------
    LayerRealizations
    """
    ny, nx = shape
    n_cells = ny * nx
    n_reals = window_reals[0].shape[1] if window_reals else 0

    weight_sum = np.zeros(n_cells)
    mean_sum = np.zeros(n_cells)
    reals_sum = np.zeros((n_cells, n_reals))
    std_sum = np.zeros(n_cells)
    # Sum of w^2 * sigma^2, the variance the weighted sum of *independent*
    # window draws actually has. Only needed for variance_correction.
    var_sum = np.zeros(n_cells)

    for result, reals in zip(results, window_reals):
        window = result.window
        idx = np.where(window.i_buf)[0]
        w = window.buffer_weight[idx]
        sigma = np.sqrt(np.diag(result.Cm_est))
        weight_sum[idx] += w
        mean_sum[idx] += w * result.m_est
        reals_sum[idx] += w[:, None] * reals
        std_sum[idx] += w * sigma
        var_sum[idx] += (w * sigma) ** 2

    covered = weight_sum > 0
    mean_flat = np.full(n_cells, np.nan)
    reals_flat = np.full((n_cells, n_reals), np.nan)
    std_flat = np.full(n_cells, np.nan)
    mean_flat[covered] = mean_sum[covered] / weight_sum[covered]
    reals_flat[covered] = reals_sum[covered] / weight_sum[covered, None]
    std_flat[covered] = std_sum[covered] / weight_sum[covered]

    deflation_flat = np.full(n_cells, np.nan)
    deflation_flat[covered] = _deflation(var_sum[covered], std_sum[covered])
    if variance_correction:
        # The draws are zero-mean, so they *are* the residual: rescaling them
        # restores amplitude without touching the posterior mean.
        reals_flat[covered] /= deflation_flat[covered, None]

    return LayerRealizations(
        mean=mean_flat.reshape(ny, nx),
        reals=reals_flat.reshape(ny, nx, n_reals),
        post_std=std_flat.reshape(ny, nx),
        deflation=deflation_flat.reshape(ny, nx),
    )


def simulate_layer(
    grid: KrigingGrid,
    windows: list[KrigingWindow],
    n_reals: int,
    *,
    cor_noise_rat: float = 0.5,
    seed: int | None = None,
    variance_correction: bool = True,
) -> LayerRealizations:
    """Krige, simulate and blend every window for one modelled layer.

    ``krige_layer`` -> ``simulate_window`` for each window in turn, all
    drawing from one continuing RNG stream -> ``blend_window_realizations``.
    This is the per-layer equivalent of the loop body in Main.m's "SECTION:
    Kriging and Simulation".

    ``cor_noise_rat`` defaults to ``0.5``, matching ``local_kriger.m``'s
    hardcoded value.

    Parameters
    ----------
    seed : int, optional
        If given, calls ``numpy.random.seed(seed)`` once before drawing any
        realizations for this layer — appropriate when calling this function
        on its own. When called from :func:`simulate_model`, leave this
        ``None`` (the default): seeding happens exactly once, in
        :func:`simulate_model`, before the loop over layers, matching
        Main.m's single ``rng(seed)`` call before its entire multi-layer
        kriging loop.
    variance_correction : bool, default True
        See :func:`blend_window_realizations`.
    """
    if seed is not None:
        np.random.seed(seed)
    results = krige_layer(grid, windows, cor_noise_rat=cor_noise_rat)
    window_reals = [simulate_window(r, n_reals) for r in results]
    return blend_window_realizations(
        grid.shape,
        results,
        window_reals,
        variance_correction=variance_correction,
    )


def simulate_model(
    layers: Sequence[tuple[KrigingGrid, list[KrigingWindow]]],
    n_reals: int,
    *,
    cor_noise_rat: float = 0.5,
    seed: int | None = None,
    variance_correction: bool = True,
) -> ModelRealizations:
    """Simulate every modelled layer of a model.

    Parameters
    ----------
    layers : sequence of (KrigingGrid, list[KrigingWindow])
        One entry per modelled boundary, in the order they should be stacked
        — typically ``RegionConfig.modelled_layer_indices`` order. Each
        layer's grid/windows are prepared independently upstream (from that
        layer's own residual surface, variance map, and clusters); this
        function only loops :func:`simulate_layer` over them and stacks the
        results.
    n_reals, cor_noise_rat : see :func:`simulate_layer`.
    seed : int, optional
        If given, calls ``numpy.random.seed(seed)`` once, before the loop
        over layers — matching Main.m's single ``rng(seed)`` call before its
        entire multi-layer kriging loop. Layers are **not** re-seeded between
        each other; all of them, and every window within them, draw from the
        one continuing stream this starts.
    variance_correction : bool, default True
        See :func:`blend_window_realizations`.

    Returns
    -------
    ModelRealizations
    """
    if seed is not None:
        np.random.seed(seed)

    means, reals, stds, deflations = [], [], [], []
    for grid, windows in layers:
        result = simulate_layer(
            grid,
            windows,
            n_reals,
            cor_noise_rat=cor_noise_rat,
            variance_correction=variance_correction,
        )
        means.append(result.mean)
        reals.append(result.reals)
        stds.append(result.post_std)
        deflations.append(result.deflation)

    return ModelRealizations(
        mean=np.stack(means, axis=-1),
        reals=np.stack(reals, axis=2),
        post_std=np.stack(stds, axis=-1),
        deflation=np.stack(deflations, axis=-1),
    )
