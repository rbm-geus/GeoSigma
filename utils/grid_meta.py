# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""The GeoSigma grid-metadata contract.

Every raster loaded or written by GeoSigma travels with a *metadata dict*
describing its geometry. This module is the single definition of that contract,
plus helpers to build, normalize, validate, and compare metadata.

Canonical keys
--------------
=============  ====================================================  =========
key            meaning                                               type
=============  ====================================================  =========
``ncols``      number of columns (x / east extent)                   int
``nrows``      number of rows (y / north extent)                     int
``dx``         pixel width  (east spacing, positive)                 float
``dy``         pixel height (north spacing, positive)                float
``xllcorner``  x of the lower-left **corner** of the grid            float
``yllcorner``  y of the lower-left **corner** of the grid            float
``nodata``     no-data sentinel in the source file, or ``None``      float|None
``crs``        coordinate reference system, or ``None``              CRS|str|None
``transform``  affine map (col,row) -> (x,y), north-up               Affine|None
=============  ====================================================  =========

Conventions
-----------
* x is east, y is north. Arrays are indexed ``[row, col]`` = ``[y, x]`` with
  row 0 at the **top** (north), matching rasterio / GeoTIFF.
* ``dx``/``dy`` are always positive; the north-up sign lives in ``transform``
  (``transform.e == -dy``).
* ``xllcorner``/``yllcorner`` are the lower-left *corner*, not a cell centre
  (ESRI ASCII convention). Cell centres come from
  :func:`utils.grid_coordinates_from_esri_meta`.

Legacy aliases (deprecated)
---------------------------
Earlier code used ``cellsize`` (== ``dx``) and ``nodata_value`` (== ``nodata``).
:func:`normalize_grid_meta` accepts these on input, and the loaders still emit
them alongside the canonical keys so existing scripts keep working. New code
should read the canonical keys.
"""

from __future__ import annotations

from typing import Any, Mapping

from affine import Affine

#: Keys required for a metadata dict to be considered complete/valid.
REQUIRED_KEYS = ("ncols", "nrows", "dx", "dy", "xllcorner", "yllcorner")

#: All canonical keys, in documentation order.
CANONICAL_KEYS = REQUIRED_KEYS + ("nodata", "crs", "transform")

#: Deprecated alias -> canonical key.
_ALIASES = {"cellsize": "dx", "nodata_value": "nodata"}


def transform_from_corner(
    xllcorner: float, yllcorner: float, dx: float, dy: float, nrows: int
) -> Affine:
    """Build a north-up affine transform from lower-left corner + spacing.

    The transform maps (col, row) pixel indices to (x, y) map coordinates with
    row 0 at the top, i.e. ``Affine(dx, 0, x_ul, 0, -dy, y_ul)`` where the
    upper-left corner is ``(xllcorner, yllcorner + nrows * dy)``.
    """
    y_ul = yllcorner + nrows * dy
    return Affine(dx, 0.0, xllcorner, 0.0, -dy, y_ul)


def make_grid_meta(
    *,
    ncols: int,
    nrows: int,
    dx: float,
    dy: float | None = None,
    xllcorner: float,
    yllcorner: float,
    nodata: float | None = None,
    crs: Any = None,
    transform: Affine | None = None,
) -> dict:
    """Construct a canonical metadata dict (with legacy aliases attached).

    ``dy`` defaults to ``dx`` (square cells). ``transform`` is derived from the
    corner and spacing when not supplied.
    """
    if dy is None:
        dy = dx
    if transform is None:
        transform = transform_from_corner(xllcorner, yllcorner, dx, dy, nrows)
    meta = {
        "ncols": int(ncols),
        "nrows": int(nrows),
        "dx": float(dx),
        "dy": float(dy),
        "xllcorner": float(xllcorner),
        "yllcorner": float(yllcorner),
        "nodata": nodata,
        "crs": crs,
        "transform": transform,
    }
    return _with_legacy_aliases(meta)


def _with_legacy_aliases(meta: dict) -> dict:
    """Attach deprecated ``cellsize``/``nodata_value`` aliases in place."""
    meta["cellsize"] = meta["dx"]
    meta["nodata_value"] = meta["nodata"]
    return meta


def normalize_grid_meta(meta: Mapping[str, Any]) -> dict:
    """Return a canonical metadata dict from a possibly-legacy ``meta``.

    Accepts dicts using the deprecated ``cellsize``/``nodata_value`` aliases,
    fills ``dy`` from ``dx`` when missing, derives ``transform`` from the corner
    and spacing when absent, and defaults ``crs``/``nodata`` to ``None``. The
    input is not mutated.
    """
    out = dict(meta)

    # Promote legacy aliases when the canonical key is absent.
    for alias, key in _ALIASES.items():
        if key not in out and alias in out:
            out[key] = out[alias]

    if "dx" not in out:
        raise KeyError(
            "grid metadata has no 'dx' (or legacy 'cellsize'); cannot normalize"
        )
    out.setdefault("dy", out["dx"])
    out.setdefault("nodata", None)
    out.setdefault("crs", None)

    if out.get("transform") is None:
        missing = [k for k in ("nrows", "xllcorner", "yllcorner") if k not in out]
        if missing:
            raise KeyError(
                f"cannot derive 'transform'; missing {missing} in grid metadata"
            )
        out["transform"] = transform_from_corner(
            out["xllcorner"], out["yllcorner"], out["dx"], out["dy"], out["nrows"]
        )

    out["ncols"] = int(out["ncols"])
    out["nrows"] = int(out["nrows"])
    out["dx"] = float(out["dx"])
    out["dy"] = float(out["dy"])
    out["xllcorner"] = float(out["xllcorner"])
    out["yllcorner"] = float(out["yllcorner"])
    return _with_legacy_aliases(out)


def validate_grid_meta(meta: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if ``meta`` is missing any required canonical key."""
    missing = [k for k in REQUIRED_KEYS if k not in meta]
    if missing:
        raise ValueError(
            f"grid metadata is missing required keys {missing}. "
            f"Pass it through utils.normalize_grid_meta first."
        )


def grids_aligned(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> bool:
    """Return True if two grids share geometry (shape, spacing, origin).

    This is the alignment gate for externally-supplied grids (variance maps,
    range/sill maps) against the model grid: matching ``ncols``, ``nrows``,
    ``dx``, ``dy``, ``xllcorner``, ``yllcorner``. CRS and nodata are ignored.
    """
    a = normalize_grid_meta(a)
    b = normalize_grid_meta(b)
    if (a["ncols"], a["nrows"]) != (b["ncols"], b["nrows"]):
        return False
    for key in ("dx", "dy", "xllcorner", "yllcorner"):
        if abs(a[key] - b[key]) > atol + rtol * abs(b[key]):
            return False
    return True
