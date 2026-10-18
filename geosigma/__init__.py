# -*- coding: utf-8 -*-
"""GeoSigma — geostatistical hydrostratigraphic modelling.

A Python translation of the MATLAB "N-ret-mod" codebase: it conditions
layer boundaries on sparse observations by local kriging and draws
geostatistical realizations from the posterior covariance, so that each
boundary carries an honest uncertainty rather than a single best guess.

The workflow, and where each step lives
---------------------------------------
``examples/demo_kriging_pipeline_end_to_end.py`` runs all of this on a real
surface; the steps below are that script's steps.

1. **Draw conditioning points** — :func:`geosigma.inpox.draw_points_inpox`
   spends points where the surface is rough rather than uniformly
   (``drawpoints.m``; Madsen et al. 2024).
2. **Estimate the local variogram** —
   :func:`geosigma.preprocess.estimate_range_sill_surface` predicts range
   and sill at every cell with a pretrained network, and returns the
   low-frequency mean and the residual that kriging actually conditions on.
3. **Group cells into kriging zones** —
   :func:`geosigma.preprocess.build_kriging_zones` (SOM, ``get_clusters.m``)
   with each zone's representative range/sill.
4. **Build the variance maps** — :mod:`geosigma.themes` turns point
   observations (TEM, PACES, gamma logs, seismic, wells) into the per-layer
   variance grids kriging consumes. Standalone: supply your own instead if
   you have them.
5. **Krige and simulate** — :mod:`geosigma.kriging`:
   :func:`~geosigma.kriging.pack_kriging_grid` ->
   :func:`~geosigma.kriging.build_windows` ->
   :func:`~geosigma.kriging.simulate_layer` ->
   :func:`~geosigma.kriging.to_absolute`.
6. **Stack the layers** —
   :func:`geosigma.assembly.stack_realizations` clamps any boundary drawn
   above the one above it, per realization.

Steps 1-5 run **independently per boundary**. That independence is what
makes each boundary's uncertainty honest, and it is also what lets two
boundaries cross, which is why step 6 exists.

Composing those steps is deliberately left to the caller rather than
wrapped in a single ``run()``: the choice of point-drawing scheme,
clusterer and variance maps is the modelling, and hard-coding it would make
the library less useful, not more. See the end-to-end demo for one
composition.

Importing
---------
``import geosigma`` gives you the core primitives and INPOX only, so the
import stays light. The rest are imported explicitly::

    from geosigma.preprocess import estimate_range_sill_surface
    from geosigma.kriging import simulate_layer
    from geosigma.assembly import stack_realizations

The network estimator needs the optional extras: ``pip install
"geosigma[nn]"`` (torch, scipy). They are imported lazily, so every
subpackage imports fine without them.
"""

from .core import (
    precal_cov,
    edist,
    semivar_synth,
    deformat_variogram,
    local_kriging_setup_img,
    least_squares_inversion,
    get_reals_cholesky,
)
from .inpox import (
    draw_points_inpox,
    ftot_laplace,
    laplacian_2d,
    visualize_transfer_function,
)

__all__ = [
    "precal_cov",
    "edist",
    "semivar_synth",
    "local_kriging_setup_img",
    "deformat_variogram",
    "get_reals_cholesky",
    "least_squares_inversion",
    "draw_points_inpox",
    "ftot_laplace",
    "laplacian_2d",
    "visualize_transfer_function",
]
