# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:18:10 2026

@author: RBM, GEUS
"""

#%%

import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from utils import load_geotiff_grid, grid_coordinates_from_esri_meta


# ============================================================
# Load all GeoTIFF surfaces
# ============================================================

folder = "examples/data/Jutland_hydrostrat_model"

# Find all tif files
tif_files = glob.glob(os.path.join(folder, "*.tif"))

print(f"Found {len(tif_files)} GeoTIFF files")

# ============================================================
# Load surfaces
# ============================================================

surfaces = []

for fname in tif_files:

    z, meta = load_geotiff_grid(fname)

    # Handle nodata
    nodata = meta.get("nodata_value", None)
    if nodata is not None:
        z = np.where(z == nodata, np.nan, z)

    # Mean elevation (used for sorting)
    z_mean = np.nanmean(z)

    surfaces.append({
        "name": os.path.basename(fname),
        "z": z,
        "meta": meta,
        "mean": z_mean,
    })

    print(f"Loaded: {os.path.basename(fname)} "
          f"(mean elevation = {z_mean:.2f} m)")


# ============================================================
# Sort surfaces from shallowest to deepest
# ============================================================

surfaces = sorted(surfaces, key=lambda s: s["mean"], reverse=True)

print("\nSorted surfaces:")
for s in surfaces:
    print(f"{s['name']:25s} mean = {s['mean']:.2f}")


# ============================================================
# Coordinates from first surface
# ============================================================

x, y, xx, yy = grid_coordinates_from_esri_meta(
    surfaces[0]["meta"]
)

extent = [x.min(), x.max(), y.min(), y.max()]


# ============================================================
# Common color limits
# ============================================================

vmin = min(np.nanmin(s["z"]) for s in surfaces)
vmax = max(np.nanmax(s["z"]) for s in surfaces)

print(f"\nGlobal color limits: {vmin:.1f} to {vmax:.1f}")


# ============================================================
# Plot all surfaces
# ============================================================

n_surfaces = len(surfaces)

ncols = 4
nrows = int(np.ceil(n_surfaces / ncols))

fig, axs = plt.subplots(
    nrows,
    ncols,
    figsize=(4 * ncols, 4 * nrows),
)

axs = np.atleast_1d(axs).ravel()

for i, surface in enumerate(surfaces):

    ax = axs[i]

    im = ax.imshow(
        surface["z"],
        origin="lower",
        extent=extent,
        cmap="terrain",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(surface["name"])
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")

# Hide unused axes
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

# Shared colorbar
cbar = fig.colorbar(
    im,
    ax=axs.tolist(),
    shrink=0.8,
)

cbar.set_label("Elevation [m]")

plt.tight_layout()
plt.show()