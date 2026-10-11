# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Windowed range/sill extraction via a pluggable :class:`RangeSillEstimator`.

Translation of ``get_interpolated_minigrids_parallel.m``, corrected against
the actual source rather than a prior paraphrase of it. At each sampled point
(e.g. from :func:`~geosigma.inpox.draw_points_inpox`):

1. A ``minigrid_size`` x ``minigrid_size`` window is extracted from the
   residual surface (``mini_grid_size = 33`` in MATLAB, 33 cells at
   ``Cell_Size = 100`` m — a 3300 m window; both are this module's defaults
   now, not the 21-cell/25 m values an earlier version of this module
   invented without checking).
2. Within that window, the **scattered points** that are themselves sample
   locations (not the whole dense window!) are picked out, mean-centred, and
   interpolated onto a smaller regular grid — ``minigrid_size - 2`` per side
   (31 by default) — via :func:`interpolate_scatter_points`. MATLAB uses
   ``scatteredInterpolant(...,'natural','linear')``; this uses **cubic**
   instead (natural-neighbour interpolation has no equivalent in this
   project's dependencies, and cubic was chosen in preference to it rather
   than as a forced substitute — see :func:`interpolate_scatter_points`).
3. That interpolated grid is what :class:`NNRangeSillEstimator` actually
   sees. ``sill`` is the sample variance of the **scattered points
   themselves, before interpolation** (``var(depths)`` in MATLAB) — not of
   the dense window, and not of the interpolated grid either (interpolation
   smooths, which would understate it). ``range`` — which depends on the
   *shape* of the spatial correlation, not just its magnitude — is what the
   network predicts.

Generating *training* data (as opposed to running inference) draws synthetic
fields rather than reading a real surface, and that's dominated by a dense
covariance factorization per field. :func:`build_range_cholesky_lookup`
precomputes that factorization once per **integer** range (fixing sill at 1,
then scaling realizations by ``sqrt(sill)`` afterwards — exact, not an
approximation, since a covariance matrix scales uniformly with sill), so
:func:`generate_training_data` looks a factor up instead of recomputing it
for every sample. See that function's docstring for the memory this costs.

The estimator is deliberately behind a :class:`RangeSillEstimator` protocol
rather than hardwired: :func:`estimate_range_sill_grid` only ever calls
``.predict(grid)``, so a different estimator can be swapped in without
touching the windowing/interpolation code. The library ships one concrete
implementation — :class:`NNRangeSillEstimator`, loaded via
:func:`load_default_rangesill_estimator` as the default.

``torch`` and ``scipy`` are required to actually train/run the network and do
the cubic interpolation, respectively, but only inside the functions/methods
that need them (imported lazily, never at module level), so
``import geosigma.preprocess.rangesill`` itself never requires either.
Install both with ``pip install "geosigma[nn]"``.
"""

from __future__ import annotations

import importlib.resources as resources
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..core.precal_cov import precal_cov

__all__ = [
    "RangeSillEstimator",
    "NNRangeSillEstimator",
    "load_default_rangesill_estimator",
    "extract_residual_patch",
    "sample_scatter_points",
    "interpolate_scatter_points",
    "estimate_range_sill_grid",
    "build_range_cholesky_lookup",
    "TrainingData",
    "generate_training_data",
    "save_training_data",
    "load_training_data",
    "train_rangesill_estimator",
]

#: Raw extraction window side length (cells), matching MATLAB's ``mini_grid_size``.
MINIGRID_SIZE = 33

#: Cell size in metres, matching MATLAB's ``Cell_Size``.
CELL_SIZE = 100.0

#: The network's actual input grid side length: the interpolated grid is
#: ``minigrid_size - 2`` per side, matching MATLAB's ``xg``/``yg`` (length
#: ``mini_grid_size - 2``). Any :class:`NNRangeSillEstimator` records its own
#: ``patch_size`` alongside its weights, so this only matters as a default
#: for functions that need to pick one before an estimator is loaded.
DEFAULT_PATCH_SIZE = MINIGRID_SIZE - 2

_DEFAULT_MODEL_FILE = "rangesill_default.pt"


class RangeSillEstimator(Protocol):
    """Interface every range/sill estimator implements.

    Anything with a ``predict(grid) -> (range, sill)`` method satisfies this
    — :class:`NNRangeSillEstimator` is the library's implementation, but a
    user is free to write their own (e.g. wrapping a different model) and
    pass it to :func:`estimate_range_sill_grid`. Note that
    :func:`estimate_range_sill_grid` overrides whatever ``sill`` an estimator
    returns with the exact scattered-point variance, for fidelity with the
    reference pipeline — see that function's docstring.
    """

    def predict(self, grid: np.ndarray) -> tuple[float, float]:
        """Return ``(range, sill)`` estimated from one interpolated grid."""
        ...


def _build_net(hidden: int):
    """Build the (untrained) range-estimation CNN. Imports torch lazily."""
    import torch.nn as nn

    class _RangeNet(nn.Module):
        def __init__(self, hidden: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),  # fixed-size features regardless of patch_size
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.head(self.features(x)).squeeze(-1)

    return _RangeNet(hidden)


def _normalize_patch(patch: np.ndarray) -> tuple[np.ndarray, float]:
    """Mean-centre and unit-scale a grid; return (normalized, sample variance).

    NaN cells (outside the surface, or outside the convex hull of the
    scattered interpolation points) are excluded from the mean/std and
    zero-filled afterwards, so the network always sees a full
    ``patch_size`` x ``patch_size`` array.

    The variance this returns is **not** generally what callers should use as
    "sill" once scattered-point interpolation is involved (it describes the
    smoothed interpolated grid, not the original scattered points) — see
    :func:`estimate_range_sill_grid` and :func:`train_rangesill_estimator`,
    both of which compute sill from the scattered points directly and ignore
    the value :meth:`NNRangeSillEstimator.predict` returns here. It remains
    useful as a self-contained fallback for a caller invoking ``.predict()``
    directly on an arbitrary grid with no scattered points of its own.
    """
    finite = np.isfinite(patch)
    if not finite.any():
        return np.zeros_like(patch), float("nan")
    values = patch[finite]
    mean = values.mean()
    std = values.std()
    sill = float(std**2)
    normed = np.where(finite, (patch - mean) / (std + 1e-8), 0.0)
    return normed, sill


class NNRangeSillEstimator:
    """Pretrained-network range estimator (+ a closed-form fallback sill).

    Parameters
    ----------
    patch_size : int
        Side length of the square (interpolated) grid this estimator expects.
    hidden : int
        Width of the network's hidden linear layer. Only matters when
        building a fresh (untrained) network; :meth:`load` reads it from the
        saved file instead.

    Notes
    -----
    The network predicts ``log(range)`` from the *normalized* grid (mean
    removed, divided by its own standard deviation) — normalizing removes the
    sill's amplitude before the network ever sees the data, so range and sill
    are architecturally decoupled rather than merely separated by training
    data variety.
    """

    def __init__(self, patch_size: int = DEFAULT_PATCH_SIZE, hidden: int = 16):
        self.patch_size = patch_size
        self.hidden = hidden
        self._model = None  # built lazily by _net(), or set directly by training/load

    def _net(self):
        if self._model is None:
            self._model = _build_net(self.hidden)
        return self._model

    def predict(self, patch: np.ndarray) -> tuple[float, float]:
        import torch

        patch = np.asarray(patch, dtype=float)
        if patch.shape != (self.patch_size, self.patch_size):
            raise ValueError(
                f"patch must be {self.patch_size}x{self.patch_size} for this "
                f"estimator, got {patch.shape}"
            )
        normed, sill = _normalize_patch(patch)
        if np.isnan(sill):
            return float("nan"), float("nan")

        net = self._net()
        net.eval()
        with torch.no_grad():
            x = torch.as_tensor(normed, dtype=torch.float32).view(
                1, 1, self.patch_size, self.patch_size
            )
            log_range = net(x).item()
        return float(np.exp(log_range)), sill

    def save(self, path) -> None:
        """Save weights + architecture config to ``path`` (``torch.save``)."""
        import torch

        torch.save(
            {
                "state_dict": self._net().state_dict(),
                "patch_size": self.patch_size,
                "hidden": self.hidden,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "NNRangeSillEstimator":
        """Load an estimator previously written by :meth:`save`."""
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        estimator = cls(patch_size=payload["patch_size"], hidden=payload["hidden"])
        net = estimator._net()
        net.load_state_dict(payload["state_dict"])
        net.eval()
        return estimator


def load_default_rangesill_estimator() -> NNRangeSillEstimator:
    """Load the pretrained range/sill estimator shipped with GeoSigma.

    The weights live at ``geosigma/preprocess/data/rangesill_default.pt`` and
    were produced by :func:`train_rangesill_estimator` — see
    ``examples/example_train_rangesill_nn.py`` for the reproducible training
    recipe that made them (the original MATLAB pipeline's pretrained net,
    ``FinalNetworkSillRangeBothNoNormalization.mat``, cannot be loaded from
    Python and its training data was never archived, so this default is a
    fresh, documented retraining on synthetic data rather than a conversion).

    Requires the optional ``torch`` dependency: ``pip install "geosigma[nn]"``.
    """
    data_dir = resources.files(__package__) / "data"
    model_path = data_dir / _DEFAULT_MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No default range/sill model found at {model_path}. Train one "
            "with geosigma.preprocess.rangesill.train_rangesill_estimator "
            "(see examples/example_train_rangesill_nn.py) and save it there, "
            "or pass an explicit `estimator=` to estimate_range_sill_grid()."
        )
    with resources.as_file(model_path) as p:
        return NNRangeSillEstimator.load(p)


def extract_residual_patch(
    surface: np.ndarray, center: tuple[int, int], patch_size: int = MINIGRID_SIZE
) -> np.ndarray:
    """Extract a ``patch_size`` x ``patch_size`` window centred on ``center``.

    This is the raw dense-window extraction MATLAB does before picking out
    the scattered points within it (``tm(y_min:y_max, x_min:x_max)``) — it is
    not itself what the network sees; see the module docstring.

    Parameters
    ----------
    surface : ndarray, shape (ny, nx)
        The residual/detrended surface to sample (e.g. an interpreted
        boundary minus a smooth trend).
    center : (row, col)
        Index into ``surface`` at the patch centre.
    patch_size : int, default MINIGRID_SIZE
        Must be odd, so the patch has a well-defined centre cell.

    Returns
    -------
    ndarray, shape (patch_size, patch_size)
        The window. Cells that fall outside ``surface`` (near an edge) are
        NaN rather than wrapped or reflected.
    """
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {patch_size}")
    half = patch_size // 2
    row, col = center
    ny, nx = surface.shape
    patch = np.full((patch_size, patch_size), np.nan, dtype=float)

    r0, r1 = row - half, row + half + 1
    c0, c1 = col - half, col + half + 1
    src_r0, src_r1 = max(r0, 0), min(r1, ny)
    src_c0, src_c1 = max(c0, 0), min(c1, nx)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return patch  # center entirely outside surface

    dst_r0, dst_c0 = src_r0 - r0, src_c0 - c0
    dst_r1 = dst_r0 + (src_r1 - src_r0)
    dst_c1 = dst_c0 + (src_c1 - src_c0)
    patch[dst_r0:dst_r1, dst_c0:dst_c1] = surface[src_r0:src_r1, src_c0:src_c1]
    return patch


def sample_scatter_points(
    field: np.ndarray, n_points: int, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick ``n_points`` random cells of ``field`` as a synthetic point cloud.

    Stands in, for synthetic training data, for MATLAB's real scattered
    borehole/interpretation points within a window — there is no "true"
    density for synthetic fields, so ``n_points`` is a free parameter (see
    :func:`train_rangesill_estimator`'s ``n_scatter_points``).

    Returns
    -------
    rows, cols : ndarray of int, shape (n_points,)
        Pixel coordinates of the sampled cells, useful for overlaying the
        points on a plot of ``field`` directly (e.g. with ``ax.scatter``).
    values : ndarray, shape (n_points,)
        ``field`` evaluated at those cells.
    """
    rng = np.random.default_rng(seed)
    ny, nx = field.shape
    n_points = min(n_points, ny * nx)
    idx = rng.choice(ny * nx, size=n_points, replace=False)
    rows, cols = np.unravel_index(idx, (ny, nx))
    return rows, cols, field[rows, cols]


def _pixel_offsets_to_xy(
    rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int], dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel coordinates to metres, relative to ``shape``'s centre cell."""
    ny, nx = shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    xs = (np.asarray(cols, dtype=float) - cx) * dx
    ys = (np.asarray(rows, dtype=float) - cy) * dx
    return xs, ys


def interpolate_scatter_points(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    half_extent: float,
    grid_size: int,
    *,
    method: str = "cubic",
) -> np.ndarray:
    """Interpolate a scattered point cloud onto a regular square grid.

    Translation of MATLAB's ``F1 = scatteredInterpolant(xs,ys,depths_new,
    'natural','linear'); TG = F1(XG,YG)`` step — using **cubic** interpolation
    (``scipy.interpolate.griddata(..., method="cubic")``) rather than natural-
    neighbour, which has no equivalent among this project's dependencies.
    Cells outside the convex hull of the input points come back NaN (cubic
    `griddata` does not extrapolate); callers normalizing the result (e.g.
    :meth:`NNRangeSillEstimator.predict` via :func:`_normalize_patch`) already
    treat NaN as "no information" and zero-fill it after normalizing, so this
    is handled gracefully downstream without needing its own extrapolation.

    Parameters
    ----------
    xs, ys : ndarray, shape (n,)
        Point coordinates in metres, relative to the target grid's centre
        (e.g. from :func:`_pixel_offsets_to_xy`).
    values : ndarray, shape (n,)
        Value at each point.
    half_extent : float
        The target grid spans ``[-half_extent, half_extent]`` along each axis.
    grid_size : int
        Number of cells per side of the target grid.
    method : {"cubic", "linear", "nearest"}, default "cubic"
        Passed straight through to ``scipy.interpolate.griddata``.

    Returns
    -------
    ndarray, shape (grid_size, grid_size)
        The interpolated grid (row 0 at ``-half_extent``, matching this
        module's row-major convention). All-NaN if there are fewer than 4
        points (too few to triangulate for cubic/linear interpolation).
    """
    from scipy.interpolate import griddata

    xg = np.linspace(-half_extent, half_extent, grid_size)
    yg = np.linspace(-half_extent, half_extent, grid_size)
    XG, YG = np.meshgrid(xg, yg)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    values = np.asarray(values, dtype=float)
    if xs.size < 4:
        return np.full((grid_size, grid_size), np.nan)

    return griddata(np.column_stack([xs, ys]), values, (XG, YG), method=method)


def _window_scatter_points(
    surface: np.ndarray,
    points_rc: np.ndarray,
    center: tuple[int, int],
    margin: int,
    dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scattered points (from the full ``points_rc`` set) within one window.

    Matches MATLAB's ``psi(PS), psj(PS)``: the conditioning points used to
    interpolate a given centre's window are whichever of the *globally*
    sampled points happen to fall inside it, not a separately-drawn set.
    Missing (NaN) values at a conditioning point are filled with the mean of
    the others in the window, matching MATLAB's
    ``pointvalues(isnan(pointvalues)) = mean(pointvalues(~isnan(pointvalues)))``.
    """
    row0, col0 = center
    rows, cols = points_rc[:, 0], points_rc[:, 1]
    within = (
        (rows >= row0 - margin)
        & (rows <= row0 + margin)
        & (cols >= col0 - margin)
        & (cols <= col0 + margin)
    )
    rows_w, cols_w = rows[within], cols[within]
    xs = (cols_w - col0).astype(float) * dx
    ys = (rows_w - row0).astype(float) * dx
    values = surface[rows_w, cols_w].astype(float)
    finite = np.isfinite(values)
    if finite.any() and not finite.all():
        values = values.copy()
        values[~finite] = values[finite].mean()
    return xs, ys, values


def estimate_range_sill_grid(
    surface: np.ndarray,
    points: np.ndarray,
    *,
    estimator: RangeSillEstimator | None = None,
    dx: float = CELL_SIZE,
    minigrid_size: int = MINIGRID_SIZE,
    interp_grid_size: int | None = None,
    interp_method: str = "cubic",
    max_nan_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate range and sill at a set of sampled grid locations.

    For each point: extract its ``minigrid_size`` x ``minigrid_size`` raw
    window (skipping it — leaving NaN — if too much of that window is missing
    data, ``max_nan_fraction``, matching MATLAB's combined "window was
    cropped at the surface edge" / "``>=10%`` NaN content" gates); find which
    of every sampled ``points`` location falls inside that window; mean-centre
    their values and interpolate them onto the regular grid the estimator
    expects (:func:`interpolate_scatter_points`); run the estimator on that
    grid for ``range``; and set ``sill`` to the exact sample variance of the
    scattered points themselves (overriding whatever the estimator itself
    returns for sill — see :class:`RangeSillEstimator`'s docstring).

    Parameters
    ----------
    surface : ndarray, shape (ny, nx)
        The residual/detrended surface to sample from.
    points : ndarray
        Either a boolean mask of shape ``(ny, nx)`` (as returned by
        :func:`~geosigma.inpox.draw_points_inpox`) or an ``(n, 2)`` array of
        explicit ``(row, col)`` indices. Serves double duty, as in MATLAB:
        both the locations to estimate *at*, and the pool of conditioning
        points available to interpolate *from* within any window.
    estimator : RangeSillEstimator, optional
        Defaults to :func:`load_default_rangesill_estimator`.
    dx : float, default CELL_SIZE
        Cell size in metres.
    minigrid_size : int, default MINIGRID_SIZE
        Raw window side length (cells).
    interp_grid_size : int, optional
        Side length of the grid handed to the estimator. Defaults to the
        estimator's own ``patch_size`` if it has one, else ``minigrid_size - 2``.
    interp_method : str, default "cubic"
        Passed to :func:`interpolate_scatter_points`.
    max_nan_fraction : float, default 0.1
        A window with at least this fraction of NaN cells is skipped.

    Returns
    -------
    rows, cols : ndarray, shape (n,)
        Grid indices of each sampled point.
    ranges, sills : ndarray, shape (n,)
        Estimated range and sill at each point (NaN where the point's window
        was skipped or had too few conditioning points to interpolate).
    """
    if estimator is None:
        estimator = load_default_rangesill_estimator()
    if interp_grid_size is None:
        interp_grid_size = getattr(estimator, "patch_size", minigrid_size - 2)

    points = np.asarray(points)
    if points.ndim == 2 and points.shape[-1] == 2 and points.dtype != bool:
        rows, cols = points[:, 0].astype(int), points[:, 1].astype(int)
    else:
        rows, cols = np.nonzero(points)
    points_rc = np.column_stack([rows, cols])

    margin = minigrid_size // 2
    half_extent = (margin - 1) * dx

    ranges = np.full(len(rows), np.nan)
    sills = np.full(len(rows), np.nan)

    for k in range(len(rows)):
        center = (int(rows[k]), int(cols[k]))
        window = extract_residual_patch(surface, center, minigrid_size)
        if np.isnan(window).mean() >= max_nan_fraction:
            continue

        xs, ys, values = _window_scatter_points(surface, points_rc, center, margin, dx)
        if values.size == 0:
            continue

        depths = values - values.mean()
        sills[k] = float(np.var(depths))

        grid = interpolate_scatter_points(
            xs, ys, depths, half_extent, interp_grid_size, method=interp_method
        )
        range_pred, _ = estimator.predict(grid)
        ranges[k] = range_pred

    return rows, cols, ranges, sills


def build_range_cholesky_lookup(
    range_bounds: tuple[float, float],
    *,
    minigrid_size: int = MINIGRID_SIZE,
    dx: float = CELL_SIZE,
    model: str = "Gau",
    dtype=np.float32,
    nugget: float = 1e-5,
) -> dict[int, np.ndarray]:
    """Precompute a unit-sill Cholesky factor for every integer range in bounds.

    This is what makes :func:`generate_training_data` fast: the dense
    covariance factorization (``~130`` ms each) is the dominant cost of
    generating a sample, and it depends on ``(range, sill)`` only through the
    *shape* of the covariance matrix, which the sill scales uniformly
    (``Cov(c·X) = c² · Cov(X)``, so ``chol(c·M) = sqrt(c) · chol(M)``). Fixing
    sill at 1 here and scaling realizations by ``sqrt(sill)`` afterwards (see
    :func:`generate_training_data`) means the factor depends on **range
    alone** — so every sample whose (randomly drawn, then rounded) range
    shares an integer gets to reuse one factorization instead of repeating it.

    Parameters
    ----------
    range_bounds : (float, float)
        Inclusive-ish range span to cover; every integer from
        ``ceil(range_bounds[0])`` to ``floor(range_bounds[1])`` gets an entry.
    minigrid_size, dx : see :func:`generate_training_data`.
    model : str, default "Gau"
        Variogram model name, as in :func:`~geosigma.testing.synthetic_field`.
    dtype : numpy dtype, default ``np.float32``
        Storage dtype for the factors. See the memory note below.
    nugget : float, default 1e-5
        Added to the diagonal *before* factorizing (i.e. to the unit-sill
        covariance), matching
        :func:`~geosigma.core.get_reals_cholesky.get_reals_cholesky`'s own
        numerical-stability nugget in magnitude. Because it's added here
        rather than after scaling, it scales with sill too when a caller
        later multiplies a realization by ``sqrt(sill)`` — a bigger sill
        gets a proportionally bigger nugget, not a fixed absolute one. This
        is deliberate (relative regularization strength stays constant
        regardless of sill) and is *exactly* equivalent to factorizing the
        sill-scaled covariance with that same sill-scaled nugget — not an
        approximation that happens to be close.

    Returns
    -------
    dict[int, ndarray]
        Maps each integer range to its ``(minigrid_size**2, minigrid_size**2)``
        lower-triangular Cholesky factor.

    Notes
    -----
    **This builds the whole table up front — there is no lazy/partial mode.**
    Each factor is ``minigrid_size**2`` square, so for the default 33-cell
    minigrid (1089x1089) that's ~4.5 MiB per integer at ``float32`` (~9 MiB at
    ``float64``). A ``range_bounds`` span of ``(50, 2000)`` is 1951 integers,
    so the whole table is roughly **8.6 GiB at float32** (**17 GiB at
    float64**). Check available RAM before widening ``range_bounds`` or
    switching to ``float64`` — there's no warning or guard here beyond this
    docstring.
    """
    lo = int(np.ceil(range_bounds[0]))
    hi = int(np.floor(range_bounds[1]))

    coord = (np.arange(minigrid_size) + 0.5) * dx
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    n_cells = coords.shape[0]
    eye = np.eye(n_cells)

    lookup: dict[int, np.ndarray] = {}
    for r in range(lo, hi + 1):
        cov, _ = precal_cov(coords, coords, f"1 {model}({r})")
        L = np.linalg.cholesky(cov + nugget * eye)
        lookup[r] = L.astype(dtype)
    return lookup


@dataclass
class TrainingData:
    """A generated (or loaded) training set for :func:`train_rangesill_estimator`.

    ``patches`` are already normalized (mean-centred, unit-scaled) and
    ``log_ranges`` already log-transformed — i.e. exactly what the network
    trains on, not raw fields. The remaining fields are the parameters that
    produced them, kept alongside the arrays so a saved dataset is
    self-describing (what estimator it's compatible with, what it was drawn
    from) rather than a bare pair of arrays.
    """

    patches: np.ndarray  # (n_samples, interp_grid_size, interp_grid_size), float32
    log_ranges: np.ndarray  # (n_samples,), float32
    interp_grid_size: int
    minigrid_size: int
    dx: float
    n_scatter_points: int
    range_bounds: tuple[float, float]
    sill_bounds: tuple[float, float]
    interp_method: str
    range_sampling: str = "log"


def generate_training_data(
    n_samples: int,
    *,
    minigrid_size: int = MINIGRID_SIZE,
    interp_grid_size: int | None = None,
    dx: float = CELL_SIZE,
    n_scatter_points: int = 40,
    interp_method: str = "cubic",
    range_bounds: tuple[float, float] = (50.0, 2000.0),
    sill_bounds: tuple[float, float] = (1.0, 50.0),
    range_sampling: str = "log",
    cholesky_lookup: dict[int, np.ndarray] | None = None,
    seed: int | None = None,
) -> TrainingData:
    """Generate a :class:`TrainingData` set of synthetic range/sill examples.

    This is the expensive half of training — and deliberately separated from
    :func:`train_rangesill_estimator` (which only needs ``torch``) so it can
    be run once, cached to disk with :func:`save_training_data`, and reused
    across many training runs with :func:`load_training_data` while you
    iterate on network size, epochs, or learning rate.

    For each of ``n_samples``: draw a (range, sill) pair from the given
    bounds (sill is always log-uniform; range is log-uniform or linear-uniform
    per ``range_sampling``); round the range to the nearest integer and look up its
    precomputed unit-sill Cholesky factor in ``cholesky_lookup``
    (:func:`build_range_cholesky_lookup` — built once automatically if you
    don't pass one yourself) rather than factorizing a fresh covariance
    matrix per sample; draw a unit-sill field from it and scale by
    ``sqrt(sill)`` (exact, not an approximation — see that function's
    docstring for why this works); sample ``n_scatter_points`` cells from the
    scaled field as a synthetic point cloud (:func:`sample_scatter_points`);
    cubic-interpolate them (:func:`interpolate_scatter_points`) onto the
    ``interp_grid_size`` grid the network will train on; normalize. Matches
    what :func:`estimate_range_sill_grid` feeds the network at inference
    time, so training and inference see the same input distribution.

    The training *label* for each sample is the **rounded** integer range
    actually used to build its field, not the pre-rounding continuous draw —
    the field's true spatial structure corresponds to the rounded value, so
    that's the honest target. ``range_bounds`` spans roughly two thousand
    integers by default, so this rounding is a sub-percent-level
    discretization, not a meaningful loss of resolution.

    Parameters
    ----------
    range_sampling : {"log", "linear"}, default "log"
        How ``range_bounds`` is sampled. ``"log"`` (the original behaviour)
        draws log-uniformly, giving equal probability per octave — most of
        the sill-estimate's known downward bias (see
        :class:`NNRangeSillEstimator`'s docstring: sample variance
        under-recovers the nominal sill when range is large relative to the
        window) comes from the long-range tail, which log-uniform sampling
        under-represents relative to linear space. ``"linear"`` draws
        uniformly in range itself, giving proportionally more long-range
        examples — useful for inspecting or training against that bias
        specifically, at the cost of the short end being relatively
        under-sampled compared to ``"log"``.
    cholesky_lookup : dict[int, ndarray], optional
        A precomputed table from :func:`build_range_cholesky_lookup`. Pass
        your own to reuse one table across several calls to this function
        (e.g. different ``n_samples`` or ``seed``) without rebuilding it each
        time; left as ``None`` (default), one is built internally covering
        ``range_bounds`` — see that function's docstring for the memory this
        takes before widening ``range_bounds``.

    Requires the optional ``scipy`` dependency (for building the lookup table
    and for the interpolation step), but not ``torch``:
    ``pip install "geosigma[nn]"`` installs both, but this function only
    needs the former.
    """
    if range_sampling not in ("log", "linear"):
        raise ValueError(
            f"range_sampling must be 'log' or 'linear', got {range_sampling!r}"
        )
    if interp_grid_size is None:
        interp_grid_size = minigrid_size - 2
    if cholesky_lookup is None:
        cholesky_lookup = build_range_cholesky_lookup(
            range_bounds, minigrid_size=minigrid_size, dx=dx
        )
    lookup_lo, lookup_hi = min(cholesky_lookup), max(cholesky_lookup)

    rng = np.random.default_rng(seed)
    log_range_lo, log_range_hi = np.log(range_bounds[0]), np.log(range_bounds[1])
    log_sill_lo, log_sill_hi = np.log(sill_bounds[0]), np.log(sill_bounds[1])

    n_cells = minigrid_size * minigrid_size
    half_extent = (minigrid_size // 2 - 1) * dx
    patches = np.empty(
        (n_samples, interp_grid_size, interp_grid_size), dtype=np.float32
    )
    log_ranges = np.empty(n_samples, dtype=np.float32)

    for i in range(n_samples):
        if range_sampling == "linear":
            range_true = float(rng.uniform(range_bounds[0], range_bounds[1]))
        else:
            range_true = float(np.exp(rng.uniform(log_range_lo, log_range_hi)))
        sill_true = float(np.exp(rng.uniform(log_sill_lo, log_sill_hi)))
        range_int = min(max(round(range_true), lookup_lo), lookup_hi)

        field_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(field_seed)
        L = cholesky_lookup[range_int]
        unit_field = (L @ np.random.randn(n_cells)).reshape(
            minigrid_size, minigrid_size
        )
        field = unit_field * np.sqrt(sill_true)

        point_seed = int(rng.integers(0, 2**31 - 1))
        rows, cols, values = sample_scatter_points(
            field, n_scatter_points, seed=point_seed
        )
        xs, ys = _pixel_offsets_to_xy(rows, cols, field.shape, dx)
        depths = values - values.mean()
        grid = interpolate_scatter_points(
            xs, ys, depths, half_extent, interp_grid_size, method=interp_method
        )
        grid = np.nan_to_num(grid, nan=0.0)  # cubic leaves gaps outside the hull
        mean, std = grid.mean(), grid.std()
        patches[i] = (grid - mean) / (std + 1e-8)
        log_ranges[i] = np.log(range_int)

    return TrainingData(
        patches=patches,
        log_ranges=log_ranges,
        interp_grid_size=interp_grid_size,
        minigrid_size=minigrid_size,
        dx=dx,
        n_scatter_points=n_scatter_points,
        range_bounds=range_bounds,
        sill_bounds=sill_bounds,
        interp_method=interp_method,
        range_sampling=range_sampling,
    )


def save_training_data(path, data: TrainingData) -> None:
    """Save a :class:`TrainingData` set to ``path`` (``.npz``, compressed)."""
    np.savez_compressed(
        path,
        patches=data.patches,
        log_ranges=data.log_ranges,
        interp_grid_size=data.interp_grid_size,
        minigrid_size=data.minigrid_size,
        dx=data.dx,
        n_scatter_points=data.n_scatter_points,
        range_bounds=np.asarray(data.range_bounds, dtype=float),
        sill_bounds=np.asarray(data.sill_bounds, dtype=float),
        interp_method=data.interp_method,
        range_sampling=data.range_sampling,
    )


def load_training_data(path) -> TrainingData:
    """Load a :class:`TrainingData` set previously written by
    :func:`save_training_data`."""
    with np.load(path, allow_pickle=False) as npz:
        return TrainingData(
            patches=npz["patches"],
            log_ranges=npz["log_ranges"],
            interp_grid_size=int(npz["interp_grid_size"]),
            minigrid_size=int(npz["minigrid_size"]),
            dx=float(npz["dx"]),
            n_scatter_points=int(npz["n_scatter_points"]),
            range_bounds=tuple(npz["range_bounds"].tolist()),
            sill_bounds=tuple(npz["sill_bounds"].tolist()),
            interp_method=str(npz["interp_method"].item()),
            # Added after some .npz files were already written; default to
            # "log" (the only mode that existed before this field did).
            range_sampling=(
                str(npz["range_sampling"].item()) if "range_sampling" in npz else "log"
            ),
        )


def train_rangesill_estimator(
    training_data: TrainingData | None = None,
    *,
    n_samples: int = 400,
    minigrid_size: int = MINIGRID_SIZE,
    interp_grid_size: int | None = None,
    dx: float = CELL_SIZE,
    n_scatter_points: int = 40,
    interp_method: str = "cubic",
    range_bounds: tuple[float, float] = (50.0, 2000.0),
    sill_bounds: tuple[float, float] = (1.0, 50.0),
    hidden: int = 16,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    loss_space: str = "log",
    seed: int | None = None,
    verbose: bool = False,
) -> NNRangeSillEstimator:
    """Train a :class:`NNRangeSillEstimator` on synthetic range/sill examples.

    Parameters
    ----------
    training_data : TrainingData, optional
        A pre-generated set (from :func:`generate_training_data` or
        :func:`load_training_data`). When given, every generation-related
        keyword below (``n_samples``, ``minigrid_size``, ``dx``, ...) is
        ignored — training runs directly on it, which is the whole point:
        generating data is the expensive step (even with the
        :func:`build_range_cholesky_lookup` speedup, building that table is
        itself a real cost, amortized across every sample that reuses it),
        so caching it once and reusing it across many training runs
        (different ``hidden``/``epochs``/``lr``) avoids paying that cost
        again. When ``None`` (default), a fresh set is generated via
        :func:`generate_training_data` using the keywords below — this keeps
        the common case a single call, as before.
    loss_space : {"log", "linear"}, default "log"
        Whether the training loss is MSE on the network's raw output (the
        log range — ``"log"``, the original behaviour) or MSE on
        ``exp()`` of that output, i.e. the range in metres (``"linear"``).
        The network always outputs log range internally regardless of this
        setting (:class:`NNRangeSillEstimator` relies on that to turn
        predictions back into a range with ``exp``) — ``"linear"`` only
        changes what the optimizer is penalized for: errors at long range
        dominate the loss much more heavily in metres than in log-space,
        so this trades accuracy at short range for accuracy at long range.
    seed : int, optional
        Seeds network initialization and minibatch shuffling (``torch``'s own
        RNG). When ``training_data`` is ``None``, also seeds the generated
        data via :func:`generate_training_data`.

    This is also how the estimator bundled as the library default
    (``geosigma/preprocess/data/rangesill_default.pt``) was produced. See
    ``examples/example_train_rangesill_nn.py`` for a runnable, evaluated
    version of this recipe, including the cache-to-disk pattern.

    Requires the optional ``torch`` dependency (and ``scipy``, transitively,
    whenever ``training_data`` must be generated): ``pip install "geosigma[nn]"``.
    """
    import torch
    import torch.nn as nn

    if loss_space not in ("log", "linear"):
        raise ValueError(f"loss_space must be 'log' or 'linear', got {loss_space!r}")

    if training_data is None:
        training_data = generate_training_data(
            n_samples,
            minigrid_size=minigrid_size,
            interp_grid_size=interp_grid_size,
            dx=dx,
            n_scatter_points=n_scatter_points,
            interp_method=interp_method,
            range_bounds=range_bounds,
            sill_bounds=sill_bounds,
            seed=seed,
        )

    if seed is not None:
        torch.manual_seed(seed)

    patches = training_data.patches
    log_ranges = training_data.log_ranges
    n = patches.shape[0]

    X = torch.as_tensor(patches, dtype=torch.float32).unsqueeze(1)
    y = torch.as_tensor(log_ranges, dtype=torch.float32)

    net = _build_net(hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss_label = "mse(log range)" if loss_space == "log" else "mse(range)"

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred = net(X[idx])
            if loss_space == "log":
                loss = loss_fn(pred, y[idx])
            else:
                loss = loss_fn(torch.exp(pred), torch.exp(y[idx]))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  {loss_label} = {epoch_loss / n:.4f}")

    net.eval()
    estimator = NNRangeSillEstimator(
        patch_size=training_data.interp_grid_size, hidden=hidden
    )
    estimator._model = net
    return estimator
