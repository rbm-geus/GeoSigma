# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.kriging — local kriging driver and simulation.

Translation of ``local_kriger.m``'s per-cluster-element loop, split across
four modules:

* :mod:`~geosigma.kriging.state` — working-state packing (``KrigingGrid`` /
  ``pack_kriging_grid``, replacing ``setup_GLOBAL_structure.m``).
* :mod:`~geosigma.kriging.buffers` — the buffer-zone blend weight around one
  cluster element (``calculate_buffer_zones``, translated from
  ``calculate_buffer_zones.m`` + ``find_100_m_buffer.m``: 3 outward + 3
  inward 1-cell rings via gradient-based edge detection, 8-level weight
  scheme).
* :mod:`~geosigma.kriging.windows` — 4-connected component labeling per
  cluster (``connected_components_4``, replacing ``bwlabel(...,4)``),
  building one window per component (``build_windows``) and solving it
  (``krige_window`` / ``krige_layer``), wrapping the already-translated
  ``geosigma.core.local_kriging_setup_img`` /
  ``geosigma.core.least_squares_inversion``.
* :mod:`~geosigma.kriging.simulation` — drawing realizations per window and
  blending overlapping windows into one coherent layer or model
  (``simulate_window``, ``blend_window_realizations``, ``simulate_layer``,
  ``simulate_model``), wrapping ``geosigma.core.get_reals_cholesky``.

Consumes a variance-map stack (from ``geosigma.themes`` or user-supplied) plus
clusters/range from :mod:`geosigma.preprocess`, and a model-theme grid (from
``geosigma.themes.model_theme``) as the source of each window's sill — see
:class:`~geosigma.kriging.windows.KrigingWindow`'s docstring for why sill
comes from there and not from the cluster stats.

What is deliberately **not** here: the ``redo_*``/checkpoint/email scaffolding
and region ``switch`` branches of Main.m, and zero-thickness tie enforcement
(``tie_realizations``, which stays in :mod:`geosigma.assembly` as an optional
post-pass). See ``examples/demo_geosigma_pipeline.py`` for how a full,
multi-stage run is composed instead of being a library function.

An earlier version of this package built its own buffer-ramp and windowing
logic from a paraphrased description rather than the actual MATLAB source,
and wired the cluster sill straight into kriging. Both are corrected here;
the affected modules' docstrings say what changed and why.
"""

from .state import KrigingGrid, pack_kriging_grid
from .buffers import calculate_buffer_zones
from .windows import (
    KrigingWindow,
    KrigingResult,
    connected_components_4,
    build_windows,
    krige_window,
    krige_layer,
)
from .simulation import (
    LayerRealizations,
    ModelRealizations,
    simulate_window,
    blend_window_realizations,
    simulate_layer,
    simulate_model,
    to_absolute,
)

__all__ = [
    "KrigingGrid",
    "pack_kriging_grid",
    "calculate_buffer_zones",
    "KrigingWindow",
    "KrigingResult",
    "connected_components_4",
    "build_windows",
    "krige_window",
    "krige_layer",
    "LayerRealizations",
    "ModelRealizations",
    "simulate_window",
    "blend_window_realizations",
    "simulate_layer",
    "simulate_model",
    "to_absolute",
]
