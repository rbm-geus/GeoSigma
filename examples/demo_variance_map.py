# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Demo: build per-layer variance maps end to end (the v0.1.0 tool).

Runs the standalone variance-map tool (``geosigma.themes``) from start to
finish on the bundled sample model, without any kriging:

1. Load the sample hydrostratigraphic model (top surface + 15 layer bottoms)
   through its ``RegionConfig``.
2. Generate **SYNTHETIC** information sources on that grid:

   * ``droneTEM`` — dense, shallow, airborne-style flight lines over the
     northern part of the area, using ``theme_specs/dronetem.spec.yaml``;
   * ``boreholes`` — a few sparse, deep point sources over the whole area,
     using a spec defined inline below.

3. Build one variance theme per source, add the **model theme**, combine all
   of them as parallel precisions, and floor the result with the
   depth-dependent minimum map.
4. Write the combined variance stack as per-layer GeoTIFFs: the handoff files
   the kriging pipeline will consume (``variance_NN.tif``) and a GIS-ready
   standard-deviation copy (``std_NN.tif``, in metres). Then plot it.

The model theme is the **background variance**: the prior uncertainty of the
hydrostratigraphic model itself, at every cell, before any data are considered.
It is a fixed part of the method, not an optional extra. Because it enters the
combine like any other theme, the combined variance can never exceed it, so
cells that no data source reaches get the model's own uncertainty instead of
the no-information sentinel. Without it, those cells keep the sentinel
(``100000``, or ``100000 / N`` once ``N`` sentinel themes are combined). The
demo uses one uniform model-quality class and the default depth ramp of
``geosigma.themes.model_theme``.

.. important::
   **The data are synthetic and the specs are illustrative.** The data points
   are randomly generated; their depths and thicknesses are read off the sample
   model surfaces. Neither ``dronetem.spec.yaml`` nor the inline borehole spec
   is a calibrated (Danish or otherwise) uncertainty model, and neither is the
   single-class model theme; they exist to show the workflow. For the reasoning
   behind real specs, see ``docs/bring_your_own_data.md``.

Output goes to ``examples/output/demo_variance_map/`` (git-ignored).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geosigma.config import RegionConfig
from geosigma.themes import (
    NODATA_VARIANCE,
    ThemeData,
    apply_floor,
    build_theme,
    combine_variances,
    informed_mask,
    load_spec,
    load_spec_yaml,
    minimum_map,
    model_theme,
    write_display_stack,
    write_variance_stack,
)
from utils import grid_coordinates_from_esri_meta, load_geotiff_grid

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "data" / "Jutland_hydrostrat_model" / "region_config.json"
DRONETEM_SPEC = HERE / "theme_specs" / "dronetem.spec.yaml"
OUT_DIR = HERE / "output" / "demo_variance_map"

rng = np.random.default_rng(42)


# ============================================================
# 1. Load the sample model
# ============================================================
cfg = RegionConfig.from_file(CONFIG)
# surface_dir in the sample config is relative to the repository root.
cfg.surface_dir = str(HERE.parent / cfg.surface_dir)

top, meta = load_geotiff_grid(str(cfg.surface_path(0)))
bottoms = [
    load_geotiff_grid(str(cfg.surface_path(i)))[0] for i in cfg.modelled_layer_indices
]
n_layers = cfg.n_modelled_layers

# The sample GeoTIFFs carry no CRS; take it from the region config.
meta = dict(meta, crs=cfg.crs)

# GeoTIFFs are north-up (row 0 = north); the theme engine works on ascending
# coordinate axes (row 0 = south). Flip once here, flip back when writing.
grid_x, grid_y, _, _ = grid_coordinates_from_esri_meta(meta)
terrain = np.flipud(top).astype(float)
bottom_stack = np.stack([np.flipud(b) for b in bottoms], axis=-1).astype(float)

# Per-cell boundary depth below terrain, and layer thickness, per layer.
depth_grid = terrain[:, :, None] - bottom_stack
upper = np.concatenate([terrain[:, :, None], bottom_stack[:, :, :-1]], axis=-1)
thick_grid = upper - bottom_stack

print(f"Region: {cfg.name}")
print(
    f"Grid: {terrain.shape[0]} x {terrain.shape[1]} cells @ {meta['dx']:.0f} m, "
    f"{n_layers} modelled layer boundaries"
)


def sample_at(xs, ys, grid):
    """Read a (ny, nx, n_layers) grid at the cells containing points (xs, ys)."""
    ix = np.clip(((xs - grid_x[0]) / meta["dx"] + 0.5).astype(int), 0, grid_x.size - 1)
    iy = np.clip(((ys - grid_y[0]) / meta["dy"] + 0.5).astype(int), 0, grid_y.size - 1)
    return grid[iy, ix]


# ============================================================
# 2. SYNTHETIC information sources
# ============================================================
x_min, x_max = grid_x[0], grid_x[-1]
y_min, y_max = grid_y[0], grid_y[-1]

# --- droneTEM: east-west flight lines every 100 m over the northern 60 % ---
# The spec's certainty kernel is zero beyond ~60 m, so lines must be dense.
line_ys = np.arange(y_min + 0.4 * (y_max - y_min), y_max, 100.0)
along = np.arange(x_min, x_max, 25.0)
drone_x = np.tile(along, line_ys.size) + rng.normal(0, 5, along.size * line_ys.size)
drone_y = np.repeat(line_ys, along.size) + rng.normal(0, 15, along.size * line_ys.size)
drone = ThemeData(
    xs=drone_x,
    ys=drone_y,
    attributes={
        "depth": sample_at(drone_x, drone_y, depth_grid),
        "thick": sample_at(drone_x, drone_y, thick_grid),
        # Depth of investigation per sounding (m below terrain).
        "doi": rng.uniform(60.0, 120.0, drone_x.size),
    },
)

# --- boreholes: 40 sparse points over the whole area ---
n_bh = 40
bh_x = rng.uniform(x_min, x_max, n_bh)
bh_y = rng.uniform(y_min, y_max, n_bh)
boreholes = ThemeData(
    xs=bh_x,
    ys=bh_y,
    attributes={
        "depth": sample_at(bh_x, bh_y, depth_grid),
        # For a borehole the "depth of investigation" is its total depth.
        "doi": rng.uniform(30.0, 250.0, n_bh),
    },
)

# Illustrative borehole spec, written inline in the same schema as the YAML
# files in theme_specs/. Not a calibrated uncertainty model.
borehole_spec = load_spec(
    {
        "name": "boreholes",
        "search_radius": 15,
        # Flat certainty plateau within `kernel` metres of a borehole, then a
        # Gaussian decay over `range` — so a borehole informs its whole window.
        "cert_fun": "FRAFA_apr2023",
        "attributes": {"depth": "mean", "doi": "max"},
        "range_model": [{"layers": "all", "range_by_complexity": [400] * 5}],
        "var0": "(0.5 * maximum(1, 0.02 * depth))**2",
        "kernel": "50",
        "mask": "doi < depth",
    }
)
drone_spec = load_spec_yaml(DRONETEM_SPEC)

print(f"Synthetic sources: {drone_x.size} droneTEM soundings, {n_bh} boreholes")


# ============================================================
# 3. Build themes, add the model theme, combine, floor
# ============================================================
themes = {
    "droneTEM": build_theme(drone, drone_spec, grid_x, grid_y, n_layers),
    "boreholes": build_theme(boreholes, borehole_spec, grid_x, grid_y, n_layers),
}

# Model theme = background variance (the prior for the model's quality). One
# uniform quality class here; a real model would map its own quality classes.
model_class = np.zeros(terrain.shape, dtype=int)
themes["model"] = model_theme(model_class, terrain, bottom_stack)

all_themes = list(themes.values())
combined = combine_variances(all_themes)
final = apply_floor(combined, minimum_map(terrain, bottom_stack))

for name, grid in themes.items():
    informed = np.mean(grid < NODATA_VARIANCE, axis=(0, 1))
    print(
        f"  {name:10s} informed fraction per layer: "
        + " ".join(f"{f:.2f}" for f in informed)
    )


# ============================================================
# 4. Write the handoff format and plot
# ============================================================
# Flip back to north-up for the GeoTIFFs.
paths = write_variance_stack(np.flip(final, axis=0), meta, OUT_DIR)
write_display_stack(
    np.flip(final, axis=0),
    meta,
    OUT_DIR,
    informed=np.flip(informed_mask(all_themes), axis=0),
)
print(f"Wrote {len(paths)} layers to {OUT_DIR}:")
print("  variance_NN.tif  handoff format for kriging (variance, sentinel kept)")
print("  std_NN.tif       for GIS viewing (standard deviation in m, -9999 = no data)")
print(
    "Combined variance never exceeds the model theme: "
    f"{bool(np.all(combined <= themes['model'] * (1 + 1e-12)))}"
)

extent = [x_min, x_max, y_min, y_max]
TAG = " (synthetic)"

# log10(variance) on one shared scale; cells without information are grey.
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("lightgrey")
log_kw = dict(origin="lower", extent=extent, cmap=cmap, vmin=-1, vmax=3.5)


def no_info_masked(grid):
    """log10 of a theme grid, masking the sentinel and +inf (no information)."""
    return np.ma.masked_where(
        ~np.isfinite(grid) | (grid >= NODATA_VARIANCE), np.log10(grid)
    )


fig, axs = plt.subplots(2, 5, figsize=(21, 8.5))

ax = axs[0, 0]
im = ax.imshow(terrain, origin="lower", extent=extent, cmap="terrain")
ax.hlines(line_ys, x_min, x_max, color="k", lw=0.3, label="droneTEM lines")
ax.plot(bh_x, bh_y, "r^", ms=5, label="boreholes")
ax.set_title("Terrain + data locations" + TAG, fontsize=10)
ax.legend(loc="lower right", fontsize=8)
fig.colorbar(im, ax=ax, label="Elevation [m]")

panels = [
    (axs[0, 1], themes["droneTEM"][:, :, 0], "droneTEM theme, layer 1"),
    (axs[0, 2], themes["boreholes"][:, :, 0], "Borehole theme, layer 1"),
    (axs[0, 3], themes["model"][:, :, 0], "Model theme (prior), layer 1"),
    (axs[1, 0], final[:, :, 0], "Combined + floor, layer 1"),
    (axs[1, 1], final[:, :, 3], "Combined + floor, layer 4"),
    (axs[1, 2], final[:, :, 7], "Combined + floor, layer 8"),
    (axs[1, 3], final[:, :, n_layers - 1], f"Combined + floor, layer {n_layers}"),
]
for ax, grid, title in panels:
    im = ax.imshow(no_info_masked(grid), **log_kw)
    ax.set_title(title + TAG, fontsize=10)
    fig.colorbar(im, ax=ax, label="log10 variance [m²]")

ax = axs[0, 4]
layers = np.arange(1, n_layers + 1)
ax.plot(layers, np.median(themes["model"], axis=(0, 1)), "s--", label="model theme")
ax.plot(layers, np.median(final, axis=(0, 1)), "o-", label="combined + floor")
ax.set_yscale("log")
ax.set_xlabel("Layer")
ax.set_ylabel("Median variance [m²]")
ax.set_title("Median variance per layer" + TAG, fontsize=10)
ax.legend(fontsize=8)

ax = axs[1, 4]
ax.axis("off")
ax.text(
    0,
    0.5,
    "Grey = no information from that source.\n\n"
    "The model theme bounds the combined\n"
    "variance, so every cell of the final\n"
    "map carries a finite uncertainty.\n\n"
    "Exported per layer:\n"
    "  variance_NN.tif  (handoff to kriging)\n"
    "  std_NN.tif       (std in m, for GIS)",
    fontsize=10,
    va="center",
)

for ax in [a for a in axs.ravel() if a not in (axs[0, 4], axs[1, 4])]:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    "GeoSigma v0.1.0 variance maps — SYNTHETIC data, illustrative specs "
    "(not a calibrated Danish uncertainty model)"
)
plt.tight_layout()
plt.show()
