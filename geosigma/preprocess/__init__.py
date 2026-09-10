# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.preprocess — per-layer preprocessing.

masks (``get_layermasks``), point drawing (``drawpoints``), windowed range/sill
extraction via a pluggable ``RangeSillEstimator``, and clustering (SOM +
post-processing + cluster stats).

Scaffold only (Phase 0); populated in **Phase 5** (depends on Phases 2 & 4).

Exception: :func:`~geosigma.preprocess.doi.resolve_doi` — the model-agnostic
per-sounding depth-of-investigation resolver (conductive-horizon DOI cap) — is
already implemented, since the variance-map themes need it ahead of Phase 5.
"""

from .doi import resolve_doi

__all__ = ["resolve_doi"]
