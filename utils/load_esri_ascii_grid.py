# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Loader for ESRI ASCII (``.asc``) grids."""

import numpy as np

from .grid_meta import make_grid_meta


def load_esri_ascii_grid(fname):
    """
    Load an ESRI ASCII grid (.asc).

    Handles both corner-referenced (``xllcorner``/``yllcorner``) and
    centre-referenced (``xllcenter``/``yllcenter``) headers, converting the
    latter to corner coordinates.

    Returns
    -------
    data : 2D ndarray (nrows, ncols)
        Grid values with ``nodata`` cells replaced by ``NaN`` (row 0 = top).
    meta : dict
        Grid metadata in the canonical GeoSigma contract (see
        :mod:`utils.grid_meta`).
    """
    header = {}
    with open(fname, "r") as f:
        # ESRI ASCII headers are 5 or 6 lines; read keyword/value pairs until a
        # line no longer looks like a header entry.
        data_lines = []
        while True:
            pos = f.tell()
            parts = f.readline().split()
            if len(parts) == 2 and not _is_number(parts[0]):
                key, value = parts
                header[key.lower()] = float(value) if _has_dot(value) else int(value)
            else:
                f.seek(pos)
                break
        data = np.loadtxt(f)

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    cellsize = float(header["cellsize"])

    # Corner vs. centre referencing.
    if "xllcorner" in header:
        xllcorner = float(header["xllcorner"])
        yllcorner = float(header["yllcorner"])
    else:
        xllcorner = float(header["xllcenter"]) - cellsize / 2.0
        yllcorner = float(header["yllcenter"]) - cellsize / 2.0

    nodata = header.get("nodata_value", None)
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    meta = make_grid_meta(
        ncols=ncols,
        nrows=nrows,
        dx=cellsize,
        dy=cellsize,
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        nodata=nodata,
        crs=None,
    )
    return data, meta


def _has_dot(value: str) -> bool:
    return "." in value or "e" in value.lower()


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False
