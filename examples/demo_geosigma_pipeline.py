# -*- coding: utf-8 -*-
"""
Demo: INPOX-based conditional simulation

Steps:
1. Load surface
2. Extract points using INPOX
3. Build covariance model

Conditioning and realization drawing are NOT implemented here -- the script
stops once the prior covariance is assembled. For the full workflow through
kriging, simulation and layer stacking, see
demo_kriging_pipeline_end_to_end.py.

/RBM, 20-03-2026
"""

import numpy as np

# Utils
from utils import load_geotiff_grid, grid_coordinates_from_esri_meta

# GeoSigma
from geosigma import draw_points_inpox, visualize_transfer_function
from geosigma.core.precal_cov import precal_cov

np.random.seed(42)

# %% -------------------------------------------------
# 1. Load surface
# -------------------------------------------------
z, meta = load_geotiff_grid("data/Jutland_hydrostrat_model/top_surface.tif")

# Handle nodata
nodata = meta.get("nodata_value", None)
if nodata is not None:
    z = np.where(z == nodata, np.nan, z)

x, y, xx, yy = grid_coordinates_from_esri_meta(meta)
dx = meta["cellsize"]

n_cells = z.size


# -------------------------------------------------
# 2. INPOX point extraction
# -------------------------------------------------
# --- Parameters ---
# Madsen et al. (2024) Table 1, real-world case (T4).
ext_vals = {
    "P0": 0.1,
    "L0": 1.0,
    "A": 1.5,
    "B": 3.0,
    "C": 5.0,
    "D": 1.0,
    "E": 0.8,
    "g": 2.0,
    "x0": meta["xllcorner"],
    "y0": meta["yllcorner"],
}


# Vizualize the chosen transfer function
visualize_transfer_function(z, ext_vals, lap_max=10)

points = draw_points_inpox(z, ext_vals, dx=dx)

# Observed points
ip_buf = points.ravel()
i_buf = ~np.isnan(z).ravel()

# %%
# -------------------------------------------------
# 3. Build covariance model
# -------------------------------------------------
coords = np.column_stack((xx.ravel(), yy.ravel()))

var = 50  # rough variance guess
ran = 2000  # rough range guess

statmod = f"{var} Gau({ran})"

Cm, _ = precal_cov(coords, coords, statmod)
