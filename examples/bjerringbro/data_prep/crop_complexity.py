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

Usage (from examples/bjerringbro/)::

    python data_prep/crop_complexity.py \\
        --source ../../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd \\
        --order higher_is_more_complex \\
        --region-config region_config.json \\
        --out data/complexity_bjerringbro.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from geosigma.config import RegionConfig
from geosigma.themes import COMPLEXITY_ORDERS, orient_complexity
from utils import load_esri_ascii_grid, load_geotiff_grid, load_surfer7_grid

LOADERS = {
    ".grd": load_surfer7_grid,
    ".tif": load_geotiff_grid,
    ".tiff": load_geotiff_grid,
    ".asc": load_esri_ascii_grid,
}


def model_grid_meta(region_config):
    """Grid metadata of the model, from its top surface."""
    cfg = RegionConfig.from_file(region_config)
    if cfg.surface_dir is not None and not Path(cfg.surface_dir).is_absolute():
        cfg.surface_dir = str(Path(region_config).parent / cfg.surface_dir)
    _, meta = load_geotiff_grid(cfg.surface_path(0))
    if not meta.get("crs"):
        meta["crs"] = cfg.crs
    return meta


def _shift(model_origin, map_origin, cell):
    """Shift (m) that makes the map's cell edges coincide with the model's.

    The offset is reduced to (-cell/2, cell/2]; a half-cell offset gives
    +cell/2, the MATLAB ``+50`` convention.
    """
    s = (model_origin - map_origin) % cell
    if np.isclose(s, cell) or np.isclose(s, 0.0):
        return 0.0
    return s if s <= cell / 2 + 1e-6 else s - cell


def _same_cell_size(z, m, mm):
    """Shift-and-copy: one map cell per model cell."""
    nx, ny = int(mm["ncols"]), int(mm["nrows"])
    sx = _shift(mm["xllcorner"], m["xllcorner"], mm["dx"])
    sy = _shift(mm["yllcorner"], m["yllcorner"], mm["dy"])
    m_top = mm["yllcorner"] + ny * mm["dy"]
    s_top = m["yllcorner"] + m["nrows"] * m["dy"]
    # Model column j takes the map cell whose west edge is (model west edge -
    # sx); model row i (from the top) the one whose top edge is (model top - sy).
    west = mm["xllcorner"] + np.arange(nx) * mm["dx"] - sx
    top = m_top - np.arange(ny) * mm["dy"] - sy
    col = np.rint((west - m["xllcorner"]) / m["dx"]).astype(int)
    row = np.rint((s_top - top) / m["dy"]).astype(int)
    okc = (col >= 0) & (col < z.shape[1])
    okr = (row >= 0) & (row < z.shape[0])
    out = np.zeros((ny, nx))
    out[np.ix_(okr, okc)] = z[np.ix_(row[okr], col[okc])]
    return out, f"same cell size; map shifted {sx:+g} m east, {sy:+g} m north"


def _majority_vote(z, m, mm, nsub=4):
    """Majority vote of nsub x nsub samples per model cell; ties -> most complex."""
    nx, ny = int(mm["ncols"]), int(mm["nrows"])
    m_top = mm["yllcorner"] + ny * mm["dy"]
    s_top = m["yllcorner"] + m["nrows"] * m["dy"]
    off = (np.arange(nsub) + 0.5) / nsub
    xs = (mm["xllcorner"] + (np.arange(nx)[:, None] + off) * mm["dx"]).ravel()
    ys = (m_top - (np.arange(ny)[:, None] + off) * mm["dy"]).ravel()
    col = np.floor((xs - m["xllcorner"]) / m["dx"]).astype(int)
    row = np.floor((s_top - ys) / m["dy"]).astype(int)
    okc = (col >= 0) & (col < z.shape[1])
    okr = (row >= 0) & (row < z.shape[0])
    samples = np.zeros((row.size, col.size))
    samples[np.ix_(okr, okc)] = z[np.ix_(row[okr], col[okc])]
    samples = (
        samples.reshape(ny, nsub, nx, nsub).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    )
    out = np.zeros((ny, nx))
    best = np.zeros((ny, nx), dtype=int)
    for c in np.unique(samples[samples > 0]):  # ascending
        cnt = np.sum(samples == c, axis=-1)
        win = (cnt > 0) & (cnt >= best)  # ties -> the higher = more complex class
        out[win], best[win] = c, cnt[win]
    how = (
        f"cell size {m['dx']:g} m vs model {mm['dx']:g} m; majority vote of "
        f"{nsub}x{nsub} samples per model cell, ties to the most complex class"
    )
    return out, how


def to_model_grid(z, m, model_meta, order, n_classes=None, fill_class=0):
    """Return ``(classes, how)``: the map on the model grid, north-up, uint8.

    ``z``/``m`` are the source map (north-up) and its corner-referenced grid
    metadata. ``classes`` follows the internal order (higher = more complex);
    ``how`` describes the method used, for the log.
    """
    z = np.asarray(z, dtype=float)
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0)  # 0 / blanks -> unknown
    z = orient_complexity(z, order, n_classes)
    same = np.isclose(m["dx"], model_meta["dx"]) and np.isclose(
        m["dy"], model_meta["dy"]
    )
    out, how = (_same_cell_size if same else _majority_vote)(z, m, model_meta)
    out = np.where(out > 0, out, fill_class)
    return out.astype("uint8"), how


def crop(source, region_config, out, order, n_classes=None, fill_class=0):
    z, m = LOADERS[Path(source).suffix.lower()](str(source))  # north-up
    mm = model_grid_meta(region_config)
    classes, how = to_model_grid(z, m, mm, order, n_classes, fill_class)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=classes.shape[0],
        width=classes.shape[1],
        count=1,
        dtype="uint8",
        crs=mm.get("crs") or m.get("crs"),
        transform=mm["transform"],
        nodata=0,
        compress="deflate",
    ) as ds:
        ds.write(classes, 1)
    found, counts = np.unique(classes, return_counts=True)
    print(f"Wrote {out}: {classes.shape[1]} x {classes.shape[0]} model cells ({how})")
    print(
        "  classes (higher = more complex): "
        + ", ".join(f"{int(c)}: {n}" for c, n in zip(found, counts))
    )


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
