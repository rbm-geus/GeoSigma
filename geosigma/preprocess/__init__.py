# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.preprocess — per-layer preprocessing.

Point drawing lives in :mod:`geosigma.inpox` (``draw_points_inpox``, already
implemented). This package covers what happens to those points next:

* :mod:`~geosigma.preprocess.rangesill` — windowed range/sill extraction via a
  pluggable ``RangeSillEstimator``, defaulting to a pretrained network
  (``NNRangeSillEstimator`` / ``load_default_rangesill_estimator``). Start at
  ``estimate_range_sill_surface``: it is the whole-grid entry point, and
  returns the low-frequency mean and the residual to krige alongside the
  range/sill fields. ``estimate_range_sill_grid`` is the per-point layer
  underneath it, and ``estimate_center_range_sill`` scores an estimator on a
  single field of known range/sill.
* :mod:`~geosigma.preprocess.clusters` — grouping per-point range/sill
  estimates into kriging zones: a simple point-array clusterer
  (``cluster_range_sill``, k-means or SOM), the per-cluster range/sill
  reduction (``cluster_stats``), and oversized-component splitting
  (``post_process_clusters``).
* :mod:`~geosigma.preprocess.som` — a Kohonen self-organizing map
  (``train_som`` / ``som_predict``), the clustering engine behind both
  ``cluster_range_sill(method="som")`` and ``som_cluster_grid`` below.
* :mod:`~geosigma.preprocess.som_clusters` — ``som_cluster_grid``, the
  literal, whole-grid translation of ``get_clusters.m`` (SOM clustering plus
  its neighbourhood-smoothing and gap-filling passes), for when you want
  output that corresponds to the reference pipeline's rather than just *a*
  reasonable clustering. ``build_kriging_zones`` is the entry point most
  callers want: it sequences ``som_cluster_grid`` -> ``cluster_stats`` ->
  gap-fill in the order that matters, and returns exactly what
  :func:`geosigma.kriging.build_windows` expects.

Masks (``get_layermasks``) are still scaffold-only, deferred to **Phase 5**.

Exception (pre-existing): :func:`~geosigma.preprocess.doi.resolve_doi` — the
model-agnostic per-sounding depth-of-investigation resolver (conductive-
horizon DOI cap) — was implemented ahead of the rest of this package, since
the variance-map themes needed it first.
"""

from .doi import resolve_doi
from .rangesill import (
    RangeSillEstimator,
    NNRangeSillEstimator,
    load_default_rangesill_estimator,
    extract_residual_patch,
    sample_scatter_points,
    interpolate_scatter_points,
    estimate_range_sill_grid,
    estimate_center_range_sill,
    RangeSillSurface,
    estimate_range_sill_surface,
    build_range_cholesky_lookup,
    TrainingData,
    generate_training_data,
    save_training_data,
    load_training_data,
    train_rangesill_estimator,
)
from .clusters import cluster_range_sill, cluster_stats, post_process_clusters
from .som import train_som, som_predict
from .som_clusters import som_cluster_grid, build_kriging_zones

__all__ = [
    "resolve_doi",
    "RangeSillEstimator",
    "NNRangeSillEstimator",
    "load_default_rangesill_estimator",
    "extract_residual_patch",
    "sample_scatter_points",
    "interpolate_scatter_points",
    "estimate_range_sill_grid",
    "estimate_center_range_sill",
    "RangeSillSurface",
    "estimate_range_sill_surface",
    "build_range_cholesky_lookup",
    "TrainingData",
    "generate_training_data",
    "save_training_data",
    "load_training_data",
    "train_rangesill_estimator",
    "cluster_range_sill",
    "cluster_stats",
    "post_process_clusters",
    "train_som",
    "som_predict",
    "som_cluster_grid",
    "build_kriging_zones",
]
