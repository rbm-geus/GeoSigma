"""Put the geological complexity map on the Bjerringbro model grid.

GeoSigma does not adapt a complexity map to the model: the map you give it must
lie on exactly the model grid (same origin, cell size and number of cells), and
the workflow stops with an error if it does not. Preparing that map is a step
you do once, before running the workflow. This script is that step for the
Bjerringbro example: it reads the national complexity map (Sandersen 2021,
GEUS Rapport 2021/37), puts it on the model grid and writes
``data/complexity_bjerringbro.tif``.

How a map cell becomes a model cell
-----------------------------------
* **Same cell size** (the national map and the model are both 100 m). The two
  grids can still be offset by part of a cell: the national map's cell edges
  lie 50 m off the model's. The map is then shifted by that offset so its cells
  coincide one to one with the model cells. A half-cell offset is shifted
  +50 m east and north, exactly as MATLAB's ``import_complexitymap.m`` does
  (``Xg = info.UTM_X + 50``), so each model cell gets the class of the map cell
  50 m to its south-west. Any other offset is shifted by the smaller amount.
  Classes are copied, never mixed.
* **Different cell size.** Each model cell is sampled at 4 x 4 points spread
  over the cell and takes the most frequent class among them (a majority vote).
  Unknown cells (0, blanks) do not vote. A tie goes to the **most complex**
  class: it gives the shorter correlation range and so the more conservative
  uncertainty.

Model cells without a class (outside the map, or unknown in it) get
``--fill-class`` (default 0 = unknown, which masks the geophysics themes there).

Class order
-----------
``--order`` says which way the source map's classes run, as in the workflow's
CONFIG; there is no default. The output is always written in GeoSigma's
internal order (higher = more complex), so the workflow reads it with
``"complexity_order": "higher_is_more_complex"``. The national file stores
5 minus the report's C-index (C-index 1 = high complexity), so its values are
already in that order: 1 = low, 4 = high.

The output is a uint8 GeoTIFF with classes 0-4 and nodata 0. It is
redistributed with the permission of the map's author, Peter B.E. Sandersen
(GEUS); see the example README, "Data sources and credits".

The shifting and voting live in ``class_grids.py``, shared with
``model_class_to_grid.py``.

Usage (from examples/bjerringbro/)::

    python data_prep/crop_complexity.py \\
        --source ../../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd \\
        --order higher_is_more_complex \\
        --region-config region_config.json \\
        --out data/complexity_bjerringbro.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from geosigma.themes import COMPLEXITY_ORDERS, orient_complexity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from class_grids import (  # noqa: E402
    load_source,
    model_grid_meta,
    to_model_grid as _to_model_grid,
    write_class_grid,
)


def to_model_grid(z, m, model_meta, order, n_classes=None, fill_class=0):
    """Return ``(classes, how)``: the map on the model grid, north-up, uint8.

    ``z``/``m`` are the source map (north-up) and its corner-referenced grid
    metadata. ``classes`` follows the internal order (higher = more complex);
    ties in a majority vote go to the most complex class. ``how`` describes the
    method used, for the log.
    """
    z = np.asarray(z, dtype=float)
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0)  # 0 / blanks -> unknown
    z = orient_complexity(z, order, n_classes)
    z[z == 0] = np.nan
    out, how = _to_model_grid(z, m, model_meta, tie="higher", fill_class=fill_class)
    return out.astype("uint8"), how


def crop(source, region_config, out, order, n_classes=None, fill_class=0):
    z, m = load_source(source)  # north-up, nodata as NaN
    mm = model_grid_meta(region_config)
    classes, how = to_model_grid(z, m, mm, order, n_classes, fill_class)
    counts = write_class_grid(out, classes, mm, crs=m.get("crs"), nodata=0)
    print(f"Wrote {out}: {classes.shape[1]} x {classes.shape[0]} model cells ({how})")
    print(f"  classes (higher = more complex): {counts}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="national complexity map")
    ap.add_argument(
        "--order",
        required=True,
        choices=COMPLEXITY_ORDERS,
        help="which way the source map's classes run",
    )
    ap.add_argument(
        "--n-classes", type=int, help="number of classes (lower_is_more_complex)"
    )
    ap.add_argument(
        "--region-config", required=True, help="the example's region_config.json"
    )
    ap.add_argument("--out", required=True, help="output GeoTIFF")
    ap.add_argument(
        "--fill-class",
        type=int,
        default=0,
        help="class for model cells without a class (0 = unknown)",
    )
    a = ap.parse_args()
    crop(a.source, a.region_config, a.out, a.order, a.n_classes, a.fill_class)


if __name__ == "__main__":
    main()
