# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.assembly — final model assembly and export.

Blends kriged surfaces into the background model (``merge_layers``), enforces
zero-thickness ties (``tie_realizations`` + snapped-well handling, isolated in
``ties`` so it can be dropped later), stacks layers top-down enforcing
monotonicity (``get_layered_models``), appends a fixed bottom (``add_bottom``),
and writes mean model + realizations via rasterio (``export``).

Implemented so far:

* :mod:`~geosigma.assembly.layers` — stacking independently kriged boundaries
  into a model whose layers never cross (``stack_realizations``, the
  monotonicity part of ``get_layered_models.m``), applied per realization.

Still scaffold, deferred to **Phase 7**: ``merge_layers``,
``tie_realizations``, ``add_bottom`` and raster export.
"""

from .layers import LayeredModel, clamp_to_upper, stack_realizations

__all__ = ["LayeredModel", "clamp_to_upper", "stack_realizations"]
