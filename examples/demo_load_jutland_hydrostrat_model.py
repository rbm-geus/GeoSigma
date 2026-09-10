# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:03:24 2026

@author: rbm
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Project imports
from utils import load_geotiff_grid, grid_coordinates_from_esri_meta
from plotting import add_in_between_gridlines


# ------------------------------------------------------------------
# --- Settings
# ------------------------------------------------------------------

data_folder = "tests/data/Jutland_hydrostrat_model"
n_layers = 15

plot_maps = True
plot_profile = True

profile_direction = "x"   # "x" or "y"
profile_index = None      # None → middle profile


# ------------------------------------------------------------------
# --- Load surfaces
# ------------------------------------------------------------------

surfaces = {}

# --- Top surface ---
fname_top = os.path.join(data_folder, "top_surface.tif")
z_top, meta = load_geotiff_grid(fname_top)
surfaces["top"] = z_top

# --- Layer bottoms ---
for i in range(1, n_layers + 1):
    fname = os.path.join(data_folder, f"layer_{i}_bottom.tif")
    z, _ = load_geotiff_grid(fname)
    surfaces[f"layer_{i}_bottom"] = z

# --- Build coordinates ---
x, y, xx, yy = grid_coordinates_from_esri_meta(meta)

# --- Handle nodata ---
nodata = meta.get("nodata_value", None)
if nodata is not None:
    for key in surfaces:
        surfaces[key] = np.where(surfaces[key] == nodata, np.nan, surfaces[key])


# ------------------------------------------------------------------
# --- Plot maps
# ------------------------------------------------------------------

if plot_maps:

    for name, z in surfaces.items():

        fig, ax = plt.subplots(figsize=(7, 6))

        im = ax.imshow(
            z,
            origin="lower",
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap="terrain",
            vmin=-250,
            vmax=50
        )

        ax.set_title(name)
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")

        add_in_between_gridlines(ax, x, y)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Elevation [m]")

        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------
# --- Plot vertical profile
# ------------------------------------------------------------------

if plot_profile:

    if profile_direction == "x":
        if profile_index is None:
            profile_index = len(y) // 2
        coord = x
        xlabel = "Easting [m]"

    elif profile_direction == "y":
        if profile_index is None:
            profile_index = len(x) // 2
        coord = y
        xlabel = "Northing [m]"

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, z in surfaces.items():

        if profile_direction == "x":
            profile = z[profile_index, :]
        else:
            profile = z[:, profile_index]

        ax.plot(coord, profile, label=name)

    ax.set_title("Vertical Profile Through Hydrostrat Model")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Elevation [m]")
    ax.legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()
