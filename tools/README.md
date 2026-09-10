# tools/

Standalone utilities that live **outside** the GeoSigma pipeline and are run on
their own to produce artifacts the pipeline (optionally) consumes.

Planned (see `TRANSLATION_PLAN.md`):

- `train_rangesill.py` — trains/exports a `RangeSillEstimator` artifact
  (Phase 4). The pipeline accepts the fitted file, or the user supplies
  precomputed range/sill grids directly.

Empty for now — this directory is scaffolded in Phase 0.
