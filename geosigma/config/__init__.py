# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.config — region/model configuration as data, not code.

Replaces the MATLAB ``landsdel_switch`` / ``get_layer_names`` switches with a
``RegionConfig`` registry (layer names, well files, paths). No region-specific
constants live in library code.

Phase 2: :class:`RegionConfig` provides the registry.
"""

from .region import RegionConfig

__all__ = ["RegionConfig"]
