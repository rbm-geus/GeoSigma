# -*- coding: utf-8 -*-
"""
Created on Fri May 22 09:10:01 2026

@author: RBM, GEUS
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def ascii_to_geotiff(ascii_path, tif_path=None, crs=None, crop_extent=None):
    """
    Convert ESRI ASCII grid to GeoTIFF.

    Parameters
    ----------
    ascii_path : str
        Input ASCII grid

    tif_path : str
        Output GeoTIFF path

    crs : str
        CRS like "EPSG:25832"

    crop_extent : tuple
        Optional:
        (xmin, xmax, ymin, ymax)

        Coordinates in map units.
    """

    ascii_path = Path(ascii_path)

    if tif_path is None:
        tif_path = ascii_path.with_suffix(".tif")

    # --------------------------------------------------------------
    # Read ASCII header
    # --------------------------------------------------------------

    header = {}

    with open(ascii_path, "r") as f:

        for _ in range(6):
            line = f.readline().split()
            header[line[0].lower()] = float(line[1])

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])

    xllcorner = header["xllcorner"]
    yllcorner = header["yllcorner"]

    cellsize = header["cellsize"]

    nodata = header.get("nodata_value", -9999)

    # --------------------------------------------------------------
    # Load raster
    # --------------------------------------------------------------

    data = np.loadtxt(ascii_path, skiprows=6).astype(np.float32)

    data[data == nodata] = np.nan

    # --------------------------------------------------------------
    # Full raster extent
    # --------------------------------------------------------------

    xmin_full = xllcorner
    xmax_full = xllcorner + ncols * cellsize

    ymin_full = yllcorner
    ymax_full = yllcorner + nrows * cellsize

    # --------------------------------------------------------------
    # Crop if requested
    # --------------------------------------------------------------

    if crop_extent is not None:

        xmin, xmax, ymin, ymax = crop_extent

        # Convert map coordinates -> array indices

        col_start = int((xmin - xmin_full) / cellsize)
        col_end = int((xmax - xmin_full) / cellsize)

        row_start = int((ymax_full - ymax) / cellsize)
        row_end = int((ymax_full - ymin) / cellsize)

        # Clip to raster bounds

        col_start = max(0, col_start)
        row_start = max(0, row_start)

        col_end = min(ncols, col_end)
        row_end = min(nrows, row_end)

        # Subset

        data = data[row_start:row_end, col_start:col_end]

        # Update dimensions

        nrows, ncols = data.shape

        # New upper-left corner

        xul = xmin_full + col_start * cellsize
        yul = ymax_full - row_start * cellsize

    else:

        xul = xmin_full
        yul = ymax_full

    # --------------------------------------------------------------
    # Create transform
    # --------------------------------------------------------------

    transform = from_origin(xul, yul, cellsize, cellsize)

    # --------------------------------------------------------------
    # Save GeoTIFF
    # --------------------------------------------------------------

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:

        out = np.where(np.isnan(data), nodata, data)

        dst.write(out, 1)

    print(f"Saved: {tif_path}")
