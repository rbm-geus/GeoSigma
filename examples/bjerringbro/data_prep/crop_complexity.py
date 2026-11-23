"""Cut an excerpt of the geological complexity map for the Bjerringbro example.

The national complexity map (Sandersen 2021, GEUS Rapport 2021/37) covers
onshore Denmark. The example only needs the part under its model grid, so this
script crops the map to the model extent plus a small buffer and writes it as
a GeoTIFF. The cells are copied unchanged (no resampling): the workflow puts
the classes on the model grid itself (``class_grid_to_model``), exactly as it
did with the national file, so the results are the same.

Classes 1-4 are kept; 0 (unknown) and blanks become 0, which is also the
GeoTIFF's nodata value. Both mean "no vote" in ``class_grid_to_model``.

The class order is kept as well. The national file stores 5 minus the report's
C-index (C-index 1 = high complexity), so in the file and in the excerpt
1 = low and 4 = high complexity: ``"complexity_order":
"higher_is_more_complex"`` in the workflow's CONFIG.

The excerpt is redistributed with the permission of the author, Peter B.E.
Sandersen (GEUS); see the example README, "Data sources and credits".

Usage (from examples/bjerringbro/)::

    python data_prep/crop_complexity.py \\
        --source ../../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd \\
        --region-config region_config.json \\
        --out data/complexity_bjerringbro.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geosigma.config import RegionConfig
from utils import load_esri_ascii_grid, load_geotiff_grid, load_surfer7_grid

LOADERS = {
    ".grd": load_surfer7_grid,
    ".tif": load_geotiff_grid,
    ".tiff": load_geotiff_grid,
    ".asc": load_esri_ascii_grid,
}


def model_extent(region_config):
    """(xmin, xmax, ymin, ymax) of the model grid, from its top surface."""
    cfg = RegionConfig.from_file(region_config)
    if cfg.surface_dir is not None and not Path(cfg.surface_dir).is_absolute():
        cfg.surface_dir = str(Path(region_config).parent / cfg.surface_dir)
    _, m = load_geotiff_grid(cfg.surface_path(0))
    x0, y0 = m["xllcorner"], m["yllcorner"]
    return x0, x0 + m["ncols"] * m["dx"], y0, y0 + m["nrows"] * m["dy"], cfg.crs


def crop(source, region_config, out, buffer_m=1000.0):
    z, m = LOADERS[Path(source).suffix.lower()](str(source))  # north-up
    z = np.asarray(z, dtype=float)
    xmin, xmax, ymin, ymax, crs = model_extent(region_config)
    xmin, xmax = xmin - buffer_m, xmax + buffer_m
    ymin, ymax = ymin - buffer_m, ymax + buffer_m

    # Source cells that overlap the buffered extent (columns west->east,
    # rows north->south because z is north-up).
    top = m["yllcorner"] + m["nrows"] * m["dy"]
    c0 = max(0, int(np.floor((xmin - m["xllcorner"]) / m["dx"])))
    c1 = min(m["ncols"], int(np.ceil((xmax - m["xllcorner"]) / m["dx"])))
    r0 = max(0, int(np.floor((top - ymax) / m["dy"])))
    r1 = min(m["nrows"], int(np.ceil((top - ymin) / m["dy"])))
    if c0 >= c1 or r0 >= r1:
        raise ValueError("the model extent does not overlap the complexity map")

    sub = z[r0:r1, c0:c1]
    valid = np.isfinite(sub) & np.isin(sub, (1, 2, 3, 4))
    classes = np.where(valid, sub, 0).astype("uint8")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(
        m["xllcorner"] + c0 * m["dx"], top - r0 * m["dy"], m["dx"], m["dy"]
    )
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=classes.shape[0],
        width=classes.shape[1],
        count=1,
        dtype="uint8",
        crs=m.get("crs") or crs,
        transform=transform,
        nodata=0,
        compress="deflate",
    ) as ds:
        ds.write(classes, 1)
    found = np.unique(classes[classes > 0]).tolist()
    print(
        f"Wrote {out}: {classes.shape[1]} x {classes.shape[0]} cells, "
        f"classes {found}, buffer {buffer_m:g} m"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, help="national complexity map")
    ap.add_argument(
        "--region-config", required=True, help="the example's region_config.json"
    )
    ap.add_argument("--out", required=True, help="output GeoTIFF")
    ap.add_argument("--buffer-m", type=float, default=1000.0)
    a = ap.parse_args()
    crop(a.source, a.region_config, a.out, a.buffer_m)


if __name__ == "__main__":
    main()
