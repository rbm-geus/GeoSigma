# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.assembly — final model assembly and export.

What kriging and simulation leave behind is one independently drawn ensemble
per boundary. This package turns those into a single layered model.

Implemented
-----------
* :mod:`~geosigma.assembly.layers` — stacking independently kriged
  boundaries into a model whose layers never cross
  (:func:`~geosigma.assembly.layers.stack_realizations`, the monotonicity
  part of ``get_layered_models.m``), applied per realization. This is the
  step the end-to-end demo finishes with.

Planned, not yet written (Phase 7)
----------------------------------
Named here so the intended scope is visible, but **none of these exist
yet** — do not import them:

* ``merge_layers`` — blend kriged surfaces into a background model.
* ``tie_realizations`` — zero-thickness ties at wells, plus snapped-well
  handling; isolated in a ``ties`` module so it can be dropped later. Its
  input, the tie grid, already exists as
  :func:`geosigma.themes.corr_map`.
* ``add_bottom`` — append a fixed bottom boundary.
* raster export of the mean model and realizations via rasterio.
"""

from .layers import LayeredModel, clamp_to_upper, stack_realizations

__all__ = ["LayeredModel", "clamp_to_upper", "stack_realizations"]
