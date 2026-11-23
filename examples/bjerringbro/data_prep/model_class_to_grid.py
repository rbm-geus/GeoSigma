"""Put a model-quality class grid on the Bjerringbro model grid.

The model theme (the background uncertainty where no data exist) takes a
model-quality ("modelling-area") class per model cell, classes 0-5 in the
Danish calibration (``geosigma.themes.DANISH_CLASS_FMIN``: surface std roots
20/20/12.5/12.5/11.25/10 m, so a HIGHER class is a better-constrained model).

GeoSigma does not adapt this grid: the workflow reads it and stops if it does
not lie on exactly the model grid. Putting a grid there is a preparation step,
done once with this script (the shifting and voting live in
``class_grids.py``, shared with ``crop_complexity.py``):

* **Same cell size.** The grid is shifted by its sub-cell offset so its cells
  coincide one to one with the model cells; classes are copied, never mixed.
* **Different cell size.** Each model cell takes the most frequent class of
  4 x 4 sample points (a majority vote). A tie goes to the **lower** class,
  which has the larger background uncertainty and so is the more conservative
  choice.

Model cells without a class (outside the grid, or nodata) get ``--fill-class``,
by default 2, the workflow's ``model_class_if_no_grid``. The output has a class
in every cell (uint8, no nodata).

The Bjerringbro example ships no model-class grid and uses class 2 everywhere.

Usage (from examples/bjerringbro/)::

    python data_prep/model_class_to_grid.py --source my_model_classes.tif \\
        --region-config region_config.json --out data/model_class.tif

then set ``CONFIG["model_class_grid"] = "data/model_class.tif"``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from class_grids import (  # noqa: E402
    load_source,
    model_grid_meta,
    to_model_grid as _to_model_grid,
    write_class_grid,
)


def to_model_grid(z, m, model_meta, fill_class=2):
    """Return ``(classes, how)``: model classes on the model grid, north-up.

    ``z`` (north-up, NaN = no class) and ``m`` are the source grid and its
    corner-referenced metadata. Ties in a majority vote go to the lower class.
    """
    out, how = _to_model_grid(
        np.asarray(z, dtype=float), m, model_meta, tie="lower", fill_class=fill_class
    )
    return out.astype("uint8"), how


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="model-class grid to adapt")
    ap.add_argument(
        "--region-config", required=True, help="the example's region_config.json"
    )
    ap.add_argument("--out", required=True, help="output GeoTIFF")
    ap.add_argument(
        "--fill-class",
        type=int,
        default=2,
        help="class for model cells without a class (default 2)",
    )
    a = ap.parse_args()
    z, m = load_source(a.source)
    mm = model_grid_meta(a.region_config)
    classes, how = to_model_grid(z, m, mm, a.fill_class)
    counts = write_class_grid(a.out, classes, mm, crs=m.get("crs"))
    print(f"Wrote {a.out}: {classes.shape[1]} x {classes.shape[0]} cells ({how})")
    print(f"  model classes: {counts}")


if __name__ == "__main__":
    main()
