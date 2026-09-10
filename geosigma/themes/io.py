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

A variance *stack* is an ``(ny, nx, n_layers)`` array indexed ``[iy, ix, layer]``
with row 0 at the top (north), matching the GeoSigma grid-metadata contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from utils.grid_meta import normalize_grid_meta


def write_variance_stack(stack, meta, out_dir, prefix="variance", nodata=None):
    """Write an ``(ny, nx, n_layers)`` variance stack as per-layer GeoTIFFs.

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
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(
            f"variance stack must be 3-D (ny, nx, n_layers); got {stack.shape}"
        )
    meta = normalize_grid_meta(meta)
    if (stack.shape[0], stack.shape[1]) != (meta["nrows"], meta["ncols"]):
        raise ValueError(
            f"stack grid {stack.shape[:2]} does not match meta "
            f"(nrows, ncols)=({meta['nrows']}, {meta['ncols']})"
        )
    if nodata is None:
        nodata = meta["nodata"]

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
