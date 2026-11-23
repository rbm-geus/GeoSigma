"""Put a class grid (complexity, model quality) on the model grid.

Shared by ``crop_complexity.py`` and ``model_class_to_grid.py``. GeoSigma's
workflow does not adapt class grids: it reads them and stops if they do not lie
on exactly the model grid (same origin, cell size and number of cells). These
helpers do that preparation once, before the run.

* **Same cell size.** The source may still be offset by part of a cell. It is
  shifted by that offset so its cells coincide one to one with the model cells
  and the classes are copied, never mixed. A half-cell offset is shifted by
  +half a cell east and north (the MATLAB ``import_complexitymap.m`` ``+50`` m
  convention for 100 m cells), so each model cell takes the source cell half a
  cell to its south-west. Any other offset is shifted by the smaller amount.
* **Different cell size.** Each model cell is sampled at 4 x 4 points spread
  over the cell and takes the most frequent class among them (a majority vote).
  The caller decides who wins a tie: the most conservative class for that grid.

Unknown source cells (NaN) never give a class; model cells left without one get
``fill_class``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from geosigma.config import RegionConfig
from utils import load_esri_ascii_grid, load_geotiff_grid, load_surfer7_grid

LOADERS = {
    ".grd": load_surfer7_grid,
    ".tif": load_geotiff_grid,
    ".tiff": load_geotiff_grid,
    ".asc": load_esri_ascii_grid,
}


def load_source(path):
    """Read a source grid: ``(z, meta)``, north-up, nodata as NaN."""
    z, m = LOADERS[Path(path).suffix.lower()](str(path))
    z = np.asarray(z, dtype=float)
    nodata = m.get("nodata")
    if nodata is not None:
        z[z == nodata] = np.nan
    return z, m


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
    """Shift (m) that makes the source's cell edges coincide with the model's.

    The offset is reduced to (-cell/2, cell/2]; a half-cell offset gives
    +cell/2, the MATLAB ``+50`` convention.
    """
    s = (model_origin - map_origin) % cell
    if np.isclose(s, cell) or np.isclose(s, 0.0):
        return 0.0
    return s if s <= cell / 2 + 1e-6 else s - cell


def _same_cell_size(z, m, mm):
    nx, ny = int(mm["ncols"]), int(mm["nrows"])
    sx = _shift(mm["xllcorner"], m["xllcorner"], mm["dx"])
    sy = _shift(mm["yllcorner"], m["yllcorner"], mm["dy"])
    m_top = mm["yllcorner"] + ny * mm["dy"]
    s_top = m["yllcorner"] + m["nrows"] * m["dy"]
    # Model column j takes the source cell whose west edge is (model west edge -
    # sx); model row i (from the top) the one whose top edge is (model top - sy).
    west = mm["xllcorner"] + np.arange(nx) * mm["dx"] - sx
    top = m_top - np.arange(ny) * mm["dy"] - sy
    col = np.rint((west - m["xllcorner"]) / m["dx"]).astype(int)
    row = np.rint((s_top - top) / m["dy"]).astype(int)
    okc = (col >= 0) & (col < z.shape[1])
    okr = (row >= 0) & (row < z.shape[0])
    out = np.full((ny, nx), np.nan)
    out[np.ix_(okr, okc)] = z[np.ix_(row[okr], col[okc])]
    return out, f"same cell size; shifted {sx:+g} m east, {sy:+g} m north"


def _majority_vote(z, m, mm, tie, nsub=4):
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
    samples = np.full((row.size, col.size), np.nan)
    samples[np.ix_(okr, okc)] = z[np.ix_(row[okr], col[okc])]
    samples = (
        samples.reshape(ny, nsub, nx, nsub).transpose(0, 2, 1, 3).reshape(ny, nx, -1)
    )
    out = np.full((ny, nx), np.nan)
    best = np.zeros((ny, nx), dtype=int)
    classes = np.unique(samples[np.isfinite(samples)])
    # Visit the classes so that the one that should win a tie comes last.
    for c in classes if tie == "higher" else classes[::-1]:
        cnt = np.sum(samples == c, axis=-1)
        win = (cnt > 0) & (cnt >= best)
        out[win], best[win] = c, cnt[win]
    how = (
        f"cell size {m['dx']:g} m vs model {mm['dx']:g} m; majority vote of "
        f"{nsub}x{nsub} samples per model cell, ties to the {tie} class"
    )
    return out, how


def to_model_grid(z, m, model_meta, *, tie, fill_class):
    """Return ``(classes, how)``: ``z`` on the model grid, north-up, float.

    ``z`` (north-up, NaN = unknown) and ``m`` are the source grid and its
    corner-referenced metadata. ``tie`` (``"higher"`` or ``"lower"``) is the
    class that wins a tied vote; cells without a class get ``fill_class``.
    """
    if tie not in ("higher", "lower"):
        raise ValueError(f"tie must be 'higher' or 'lower', got {tie!r}")
    z = np.asarray(z, dtype=float)
    same = np.isclose(m["dx"], model_meta["dx"]) and np.isclose(
        m["dy"], model_meta["dy"]
    )
    if same:
        out, how = _same_cell_size(z, m, model_meta)
    else:
        out, how = _majority_vote(z, m, model_meta, tie)
    return np.where(np.isfinite(out), out, fill_class), how


def write_class_grid(out, classes, model_meta, crs=None, nodata=None):
    """Write ``classes`` (north-up) as a uint8 GeoTIFF on the model grid."""
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
        crs=model_meta.get("crs") or crs,
        transform=model_meta["transform"],
        nodata=nodata,
        compress="deflate",
    ) as ds:
        ds.write(np.asarray(classes).astype("uint8"), 1)
    found, counts = np.unique(classes, return_counts=True)
    return ", ".join(f"{int(c)}: {n}" for c, n in zip(found, counts))
