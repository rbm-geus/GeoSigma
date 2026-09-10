# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:05:30 2026

@author: rbm
"""

import numpy as np
import rasterio

from .grid_meta import make_grid_meta


def load_geotiff_grid(fname):
    """
    Load a GeoTIFF raster.

    Returns
    -------
    z : 2D ndarray
        Raster values (row 0 = north / top), with ``nodata`` left as stored.
    meta : dict
        Grid metadata in the canonical GeoSigma contract (see
        :mod:`utils.grid_meta`): ``ncols, nrows, dx, dy, xllcorner, yllcorner,
        nodata, crs, transform`` (plus deprecated ``cellsize``/``nodata_value``
        aliases).
    """
    with rasterio.open(fname) as ds:
        z = ds.read(1)  # first band
        transform = ds.transform

        dx = transform.a  # pixel width  (east, positive)
        dy = -transform.e  # pixel height (north); transform.e is < 0
        # Lower-left corner: upper-left y (transform.f) plus nrows * e (e < 0).
        yllcorner = transform.f + ds.height * transform.e

        meta = make_grid_meta(
            ncols=ds.width,
            nrows=ds.height,
            dx=dx,
            dy=dy,
            xllcorner=transform.c,
            yllcorner=yllcorner,
            nodata=ds.nodata,
            crs=ds.crs,
            transform=transform,
        )

    return z, meta
