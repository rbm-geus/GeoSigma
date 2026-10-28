# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Read/write the variance-map handoff format: a folder of per-layer GeoTIFFs.

This is the contract between the standalone variance-map tool and the kriging
pipeline (and the format a user supplies their own variance maps in): one
north-up GeoTIFF per layer, named ``<prefix>_<NN>.tif`` (1-based, zero-padded),
written via rasterio with an explicit CRS and affine transform. GIS-inspectable
by design.

The handoff files store the variance exactly as computed, including the
``NODATA_VARIANCE`` sentinel, which swamps a GIS colour stretch. For viewing,
:func:`write_display_stack` writes a companion ``std_<NN>.tif`` per layer: the
standard deviation in metres, with cells that carry no information as no-data.

A variance *stack* is an ``(ny, nx, n_layers)`` array indexed ``[iy, ix, layer]``
with row 0 at the top (north), matching the GeoSigma grid-metadata contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from utils.grid_meta import normalize_grid_meta

#: No-data value of the display (standard-deviation) GeoTIFFs.
DISPLAY_NODATA = -9999.0


def _validated_stack(stack, meta, what):
    """Cast ``stack`` to float32 and check it is 3-D and matches ``meta``."""
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"{what} must be 3-D (ny, nx, n_layers); got {stack.shape}")
    if (stack.shape[0], stack.shape[1]) != (meta["nrows"], meta["ncols"]):
        raise ValueError(
            f"stack grid {stack.shape[:2]} does not match meta "
            f"(nrows, ncols)=({meta['nrows']}, {meta['ncols']})"
        )
    return stack


def _write_layers(stack, meta, out_dir, prefix, nodata):
    """Write a float32 ``(ny, nx, n_layers)`` stack as ``<prefix>_NN.tif`` files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_layers = stack.shape[2]
    paths = []
    for s in range(n_layers):
        path = out_dir / f"{prefix}_{s + 1:02d}.tif"
        band = stack[:, :, s]
        if nodata is not None:
            band = np.where(np.isnan(band), nodata, band)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=meta["nrows"],
            width=meta["ncols"],
            count=1,
            dtype="float32",
            crs=meta["crs"],
            transform=meta["transform"],
            nodata=nodata,
            compress="lzw",
        ) as dst:
            dst.write(band.astype(np.float32), 1)
        paths.append(path)
    return paths


def write_variance_stack(stack, meta, out_dir, prefix="variance", nodata=None):
    """Write an ``(ny, nx, n_layers)`` variance stack as per-layer GeoTIFFs.

    This is the **handoff format** read by :func:`read_variance_stack` and the
    kriging pipeline. Values are written exactly as computed, including the
    :data:`~geosigma.themes.base.NODATA_VARIANCE` sentinel, so the file is not
    meant for viewing in a GIS; use :func:`write_display_stack` for that.

    Parameters
    ----------
    stack : ndarray, shape (ny, nx, n_layers)
    meta : mapping
        Grid metadata (canonical or legacy); normalised internally.
    out_dir : str or Path
        Destination directory (created if missing).
    prefix : str
        File-name prefix; files are ``<prefix>_01.tif`` ... (1-based).
    nodata : float, optional
        No-data value written to the files; defaults to ``meta['nodata']``.

    Returns
    -------
    list of Path
        The written file paths, in layer order.
    """
    meta = normalize_grid_meta(meta)
    stack = _validated_stack(stack, meta, "variance stack")
    if nodata is None:
        nodata = meta["nodata"]
    return _write_layers(stack, meta, out_dir, prefix, nodata)


def write_display_stack(
    stack, meta, out_dir, informed=None, prefix="std", nodata=DISPLAY_NODATA
):
    """Write a GIS-ready **standard deviation** version of a variance stack.

    Companion to :func:`write_variance_stack` for viewing in QGIS/ArcGIS. Each
    layer is written as ``<prefix>_NN.tif`` holding ``sqrt(variance)``, i.e. the
    standard deviation of the boundary depth in metres. Cells without
    information are written as GeoTIFF no-data, so they do not stretch the
    colour scale. The variance stack itself is not modified.

    Parameters
    ----------
    stack : ndarray, shape (ny, nx, n_layers)
        The variance stack, in the same orientation passed to
        :func:`write_variance_stack` (row 0 = north).
    meta : mapping
        Grid metadata (canonical or legacy); normalised internally.
    out_dir : str or Path
        Destination directory (created if missing).
    informed : ndarray of bool, shape (ny, nx, n_layers), optional
        True where the cell carries information, typically
        :func:`geosigma.themes.combine.informed_mask` of the per-theme grids
        (same orientation as ``stack``). False cells are written as no-data.
        When omitted, only non-finite cells are no-data.
    prefix : str
        File-name prefix (default ``"std"``).
    nodata : float
        No-data value (default :data:`DISPLAY_NODATA`, ``-9999``).

    Returns
    -------
    list of Path
        The written file paths, in layer order.
    """
    meta = normalize_grid_meta(meta)
    variance = np.asarray(stack, dtype=float)
    _validated_stack(variance, meta, "variance stack")
    with np.errstate(invalid="ignore"):
        std = np.sqrt(variance)
    keep = np.isfinite(std)
    if informed is not None:
        informed = np.asarray(informed, dtype=bool)
        if informed.shape != variance.shape:
            raise ValueError(
                f"informed mask shape {informed.shape} does not match "
                f"stack shape {variance.shape}"
            )
        keep &= informed
    std = np.where(keep, std, nodata).astype(np.float32)
    return _write_layers(std, meta, out_dir, prefix, nodata)


def read_variance_stack(in_dir, prefix="variance"):
    """Read a folder of per-layer GeoTIFFs back into a stack + metadata.

    Returns
    -------
    stack : ndarray, shape (ny, nx, n_layers)
        No-data cells are converted to ``NaN``.
    meta : dict
        Canonical grid metadata, taken from the first layer; the loader checks
        every layer shares the same geometry.
    """
    in_dir = Path(in_dir)
    files = sorted(in_dir.glob(f"{prefix}_*.tif"))
    if not files:
        raise FileNotFoundError(f"no '{prefix}_*.tif' files in {in_dir}")

    bands = []
    meta = None
    for path in files:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(float)
            this_meta = normalize_grid_meta(
                {
                    "ncols": src.width,
                    "nrows": src.height,
                    "dx": src.transform.a,
                    "dy": -src.transform.e,
                    "xllcorner": src.transform.c,
                    "yllcorner": src.transform.f + src.height * src.transform.e,
                    "nodata": src.nodata,
                    "crs": src.crs,
                    "transform": src.transform,
                }
            )
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            if meta is None:
                meta = this_meta
            elif (this_meta["nrows"], this_meta["ncols"]) != (
                meta["nrows"],
                meta["ncols"],
            ):
                raise ValueError(f"layer {path.name} grid differs from {files[0].name}")
            bands.append(arr)
    return np.stack(bands, axis=-1), meta
