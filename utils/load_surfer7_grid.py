# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Loader for Surfer 7 binary (``.grd``) grids, in the canonical contract.

Thin wrapper over the low-level :func:`ReadSurfer7.ReadSurfer7` reader at the
repo root. It does two things on top of the raw read:

* **Orientation.** Surfer stores rows south-to-north (row 0 = bottom). GeoSigma's
  contract is north-up (row 0 = top, like GeoTIFF/ESRI), so the array is flipped
  vertically.
* **Registration.** Surfer 7 grids are *node*-registered: ``xLL``/``yLL`` are the
  coordinates of the lower-left grid **node** (cell centre). The contract's
  ``xllcorner``/``yllcorner`` are cell **corners**, so we subtract half a cell.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .grid_meta import make_grid_meta

# ReadSurfer7.py lives at the repo root (not yet packaged); make it importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ReadSurfer7 import ReadSurfer7  # noqa: E402


def load_surfer7_grid(fname, crs=None):
    """
    Load a Surfer 7 ``.grd`` grid.

    Parameters
    ----------
    fname : str | Path
        Path to the Surfer 7 binary grid.
    crs : optional
        CRS to attach (Surfer grids carry no CRS); passed through to ``meta``.

    Returns
    -------
    z : 2D ndarray (nrows, ncols)
        Grid values, north-up (row 0 = top), blanks as ``NaN``.
    meta : dict
        Grid metadata in the canonical GeoSigma contract (see
        :mod:`utils.grid_meta`). ``nodata`` is Surfer's blank value.
    """
    z_southup, info = ReadSurfer7(str(fname))

    # Surfer rows run south->north; flip to north-up (row 0 = top).
    z = np.flipud(np.asarray(z_southup))

    dx = float(info["xResolution"])
    dy = float(info["yResolution"])

    # Node (centre) coords -> lower-left corner.
    xllcorner = float(info["xLL"]) - dx / 2.0
    yllcorner = float(info["yLL"]) - dy / 2.0

    meta = make_grid_meta(
        ncols=int(info["Cols"]),
        nrows=int(info["Rows"]),
        dx=dx,
        dy=dy,
        xllcorner=xllcorner,
        yllcorner=yllcorner,
        nodata=float(info["BlankValue"]),
        crs=crs,
    )
    return z, meta
