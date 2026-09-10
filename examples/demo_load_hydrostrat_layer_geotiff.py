# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:07:52 2026

@author: rbm
"""

import matplotlib.pyplot as plt
import numpy as np

# Project imports
from utils import load_geotiff_grid,grid_coordinates_from_esri_meta
from plotting import add_in_between_gridlines


# --- Load topo surface ---
z_topo, meta = load_geotiff_grid(
    "examples/data/Jutland_hydrostrat_model/top_surface.tif"
)

# --- Build coordinates ---
x, y, xx, yy = grid_coordinates_from_esri_meta(meta)

# --- Handle nodata ---
nodata = meta.get("nodata_value", None)
if nodata is not None:
    z_topo = np.where(z_topo == nodata, np.nan, z_topo)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(
    z_topo,
    origin="lower",
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap="terrain"
)

ax.set_title("Topography (Jutland, GeoTIFF)")
ax.set_xlabel("Easting [m]")
ax.set_ylabel("Northing [m]")

add_in_between_gridlines(ax, x, y)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Elevation [m]")

plt.tight_layout()
plt.show()


