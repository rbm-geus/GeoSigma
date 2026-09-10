# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:48:28 2026

@author: rbm, GEUS
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from utils import load_geotiff_grid, grid_coordinates_from_esri_meta
from pest_integration import perturb_layer_boundaries

# %% Run pertubation shift
k = 50

perturb_layer_boundaries("examples/data/Jutland_hydrostrat_model/", k, method="shift")


# %% Run pertubation realization
k = 2

layers = [
    "topo.tif",
    "0010_Post_Glacial_Ler_Toerv_Gytje_bund.tif",
    "1100_Kvartaer_ler_Bund.tif",
    "1200_Kvartaer_sand_Bund.tif",
    "1300_Kvartaer_ler_Bund.tif",
    "1400_Kvartaer_sand_Bund.tif",
    "1500_Kvartaer_ler_Bund.tif",
    "2100_Kvartaer_sand_Bund.tif",
    "2200_Kvartaer_ler_Bund.tif",
    "2300_Kvartaer_sand_Bund.tif",
    "2400_Preq_Kvartaer_ler_Bund.tif",
    "7400_Billund_BDS2_Bund.tif",
    "7800_Billund_BDS0_Bund.tif",
    "8000_Palaeogen_ler_Bund.tif",
    "8500_Danien_Kalk_Bund.tif",
    "9000_Skrivekridt_Bund.tif",
]
perturb_layer_boundaries("examples/data/Reals/", k, method="realization", layers=layers)


# %% Load and plot results and original

# ============================================================
# Load all GeoTIFF surfaces
# ============================================================

# folder = "examples/data/Jutland_hydrostrat_model/perturbed"
folder = "examples/data/Reals/perturbed"

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

    surfaces.append(
        {
            "name": os.path.basename(fname),
            "z": z,
            "meta": meta,
            "mean": z_mean,
        }
    )

    print(f"Loaded: {os.path.basename(fname)} " f"(mean elevation = {z_mean:.2f} m)")


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

x, y, xx, yy = grid_coordinates_from_esri_meta(surfaces[0]["meta"])

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


# ============================================================
# Load all GeoTIFF surfaces - original
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

    surfaces.append(
        {
            "name": os.path.basename(fname),
            "z": z,
            "meta": meta,
            "mean": z_mean,
        }
    )

    print(f"Loaded: {os.path.basename(fname)} " f"(mean elevation = {z_mean:.2f} m)")


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

x, y, xx, yy = grid_coordinates_from_esri_meta(surfaces[0]["meta"])

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
