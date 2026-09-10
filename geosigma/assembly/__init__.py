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

Scaffold only (Phase 0); populated in **Phase 7**.
"""
