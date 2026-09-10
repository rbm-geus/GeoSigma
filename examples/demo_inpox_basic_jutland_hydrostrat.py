import matplotlib.pyplot as plt
import numpy as np
from utils import load_geotiff_grid, grid_coordinates_from_esri_meta
from geosigma import draw_points_inpox, ftot_laplace,visualize_transfer_function

# ============================================================
# --- Load surface ---
z, meta = load_geotiff_grid("examples/data/Jutland_hydrostrat_model/top_surface.tif")
#z, meta = load_geotiff_grid("examples/data/Jutland_hydrostrat_model/layer_4_bottom.tif")
x, y, xx, yy = grid_coordinates_from_esri_meta(meta)

# --- INPOX Parameters ---
ext_vals = {
    "p0": 0.1,
    "a": 0.5,
    "b": 7,
    "c": 0.5,
    "d": -0.1,
    "e": 1,
    "g": 6,
    "x0": meta["xllcorner"],
    "y0": meta["yllcorner"],
}


# ============================================================
# Figure 1: Vizualize the chosen transfer function
# ============================================================
visualize_transfer_function(z, ext_vals, dx=1.0, dy=1.0, lap_max=10)



# ============================================================
# --- Run INPOX sampling ---
# ============================================================


points, plap, lapl, plap_extra = draw_points_inpox(z, ext_vals, dx=meta["cellsize"])

lap_surface = lapl.ravel()
actual_draw_percent = 100 * np.sum(points) / points.size
print(f"Actual draw percentage: {actual_draw_percent:.2f} %")




# ============================================================
# Figure 2: Spatial results (2x2)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# --- Original surface ---
im0 = axes[0, 0].imshow(z)
axes[0, 0].set_title("Original surface")
fig.colorbar(im0, ax=axes[0, 0])

# --- Laplacian ---
im1 = axes[0, 1].imshow(lapl,vmin=0,vmax=10)
axes[0, 1].set_title("Laplacian")
fig.colorbar(im1, ax=axes[0, 1])

# --- Probability added map ---
im2 = axes[0, 2].imshow(plap_extra,vmin=0,vmax=1)
axes[0, 2].set_title("Probability change to base draw")
fig.colorbar(im2, ax=axes[0, 2])

# --- Probability map final ---
im3 = axes[1, 0].imshow(plap,vmin=0,vmax=1)
axes[1, 0].set_title("Final probability normalized")
fig.colorbar(im3, ax=axes[1, 0])

# --- Sampled points ---
im4 = axes[1, 1].imshow(points)
axes[1, 1].set_title(f"Sampled points ({actual_draw_percent:.2f}%)")
fig.colorbar(im4, ax=axes[1, 1])

# Repeated point draw
Nrep = 50;
points_rep = np.zeros(np.shape(points))
for i in range(Nrep):
    points_i, _, _, _ = draw_points_inpox(z, ext_vals, dx=meta["cellsize"])
    points_rep += points_i


# --- Sampled points ---
im4 = axes[1, 2].imshow(points_rep,vmin=0,vmax=Nrep)
axes[1, 2].set_title(f"Number of draws in {Nrep:.0f} extractions")
fig.colorbar(im4, ax=axes[1, 2])

plt.tight_layout()
plt.show()