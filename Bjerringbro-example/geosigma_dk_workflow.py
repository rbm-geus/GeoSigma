# -*- coding: utf-8 -*-
"""
GeoSigma workflow for a Danish site: variance maps + (experimental) realizations
================================================================================

Built on GeoSigma v0.1.0 (https://github.com/rbm-geus/GeoSigma).

    STEP 1  Load your deterministic hydrostratigraphic model (terrain + layer bottoms)
    STEP 2  Load the optional support grids (geological complexity, model-quality class)
    STEP 3  Build one variance "theme" per geophysical data source (SkyTEM, tTEM, PACES ...)
    STEP 4  Build the borehole (well) theme from your layer picks
    STEP 5  Add the model theme, combine all themes, apply the depth floor, export GeoTIFFs
    STEP 6  [EXPERIMENTAL] Draw realizations of every layer boundary from the variance maps

Every place you must supply your own data is marked with:   >>> YOUR DATA

How to run
----------
    python geosigma_dk_workflow.py --demo              # full run on GeoSigma's Jutland sample + synthetic data
    python geosigma_dk_workflow.py --write-templates   # writes empty input templates to ./templates/
    python geosigma_dk_workflow.py                     # runs on YOUR data (edit CONFIG first)

Install GeoSigma first (inside its repo):  pip install -e .

IMPORTANT about STEP 6
----------------------
GeoSigma v0.1.0 only ships the variance-map half. Its realization method (local
kriging conditioned on INPOX-selected points, then Cholesky simulation) is planned
for v1.0.0 and is NOT finished. STEP 6 is a simplified stand-in built from the
primitives GeoSigma already ships (``precal_cov`` + ``get_reals_cholesky``):

    realization = deterministic surface + spatially correlated Gaussian noise
                  scaled cell-by-cell by sqrt(variance map)

followed by stratigraphic ordering and preservation of zero-thickness layers.
It is useful for exploring the uncertainty, but it is not the GEUS method and has
not been validated. Replace it with the official driver once v1.0.0 is released.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from geosigma.config import RegionConfig
from geosigma.core.get_reals_cholesky import get_reals_cholesky
from geosigma.core.precal_cov import precal_cov
from geosigma.themes import (
    NODATA_VARIANCE,
    WellSet,
    apply_floor,
    build_theme,
    build_well_theme,
    combine_variances,
    informed_mask,
    load_spec_yaml,
    load_theme_data,
    minimum_map,
    model_theme,
    resolve_theme_spec,
    write_display_stack,
    write_variance_stack,
)
from utils import (
    grid_coordinates_from_esri_meta,
    load_esri_ascii_grid,
    load_geotiff_grid,
    load_surfer7_grid,
)

# =============================================================================
# CONFIG — edit this block for your site
# =============================================================================
CONFIG = {
    # --- STEP 1: the deterministic hydrostratigraphic model --------------------
    # >>> YOUR DATA: a RegionConfig JSON listing your surfaces top-to-bottom.
    # One grid file per surface, named <layer_name><file_ext> inside surface_dir.
    # Typical Danish sources:
    #   * The national hydrostratigraphic model (DK-model / HIP, hipdata.dk) cropped
    #     to your site, or the GEUS 3D model for your area
    #   * Your own GeoScene3D / Leapfrog / Petrel model exported as grids
    # Supported grid formats: GeoTIFF (.tif), ESRI ASCII (.asc), Surfer 7 (.grd).
    # All grids must share the same extent/cell size; elevations in m (DVR90);
    # CRS normally EPSG:25832 (ETRS89 / UTM 32N).
    "region_config": "region_config.json",

    # --- STEP 2: support grids (optional) -------------------------------------
    # >>> YOUR DATA: geological-complexity class grid (integers 1-4, 0 = unknown),
    # aligned with the model grid. Sets the correlation range (500/400/250/100 m in
    # the Danish specs). GEUS may share their national complexity proxy; otherwise
    # set to None => uniform class 1 (a single, long range everywhere).
    "complexity_grid": "../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd",           # e.g. "my_site/complexity.tif"
    # The grid does NOT need to match the model grid: origin shifts (e.g. half a cell),
    # larger extent or other cell size are handled by class_grid_to_model().
    # Class for model cells the complexity map does not cover. 0 = unknown, which
    # MASKS every geophysics theme there; set e.g. 2 if you prefer a moderate default.
    "complexity_fill_class": 0,

    # >>> YOUR DATA: model-quality ("modelling-area") class grid used by the model
    # theme = the background uncertainty where no data exist. Classes 0-5 with
    # surface std roots 20/20/12.5/12.5/11.25/10 m (MATLAB Jylland defaults).
    # None => one uniform class (MODEL_CLASS_IF_NO_GRID).
    "model_class_grid": None,          # e.g. "my_site/model_class.tif"
    "model_class_if_no_grid": 2,

    # Danish two-regime split: 0-based index of the first PRE-QUATERNARY modelled
    # layer. Built-in Danish specs use a different correlation range below it.
    # >>> YOUR DATA: look it up in your layer list. If your model is all Quaternary,
    # set it to the number of modelled layers (= no deep regime).
    "n_prereq": None,                  # None => n_modelled_layers

    # --- STEP 3: geophysical themes -------------------------------------------
    # One entry per data source. Each "manifest" is a ThemeData bundle
    # (manifest YAML + points CSV, see GeoSigma docs/theme_format.md;
    # run --write-templates for blank examples).
    # >>> YOUR DATA: from GERDA (GEUS geophysics database) or the survey contractor:
    #   per sounding: x, y, the DEPTH BELOW TERRAIN at which each of YOUR model's
    #   layer boundaries sits at that sounding (depth__L01..), layer thickness
    #   (thick__L01..) and depth of investigation (doi).
    #   Picking boundaries onto your layers is the "adapter" step — you do it once,
    #   e.g. by sampling your model surfaces at the sounding positions (as the demo
    #   does) or from your interpreted geophysical layer models.
    # "spec" is either a built-in Danish spec name
    #   (skytem, ttem, fewtem, manytem, paces, pacep, mep, gammalog, reslog, refseis)
    # or a path to your own YAML spec (see GeoSigma examples/theme_specs/).
    "themes": [
        {"name": "SkyTEM", "manifest": "geofysik/skytem/skytem.manifest.yaml"},
        {"name": "TEM",   "manifest": "geofysik/fewtem/fewtem.manifest.yaml"},
        {"name": "PACES",  "manifest": "geofysik/paces/paces.manifest.yaml"},
    ],

    # --- STEP 4: boreholes ------------------------------------------------------
    # >>> YOUR DATA: one CSV row per (borehole, layer boundary) pick.
    # Columns: well_id, x, y, layer (1-based modelled layer), z_boundary
    # (elevation of the boundary, m DVR90), terrain (elevation of the well top),
    # quality (1 = best ... 4 = worst), source ("snap" = interpreted/snapped wells,
    # "jupiter" = raw Jupiter wells).
    # Source: Jupiter (GEUS borehole database) + your geological interpretation.
    # Only add a row for a layer if the well actually reaches/informs it.
    # None => no well theme.
    "wells_csv": "boringer/wells.csv",                 # e.g. "my_site/wells.csv"

    # --- STEP 5: output -----------------------------------------------------------
    "out_dir": "output/variance_maps",

    # --- STEP 6: experimental realizations -------------------------------------
    "realizations": {
        "enabled": True,
        "n_reals": 10,
        "corr_range_m": 500.0,   # lateral correlation range of the error (m)
        "max_sim_cells": 2500,   # Cholesky is O(n^3): simulate on a coarse grid then upsample
        "seed": 42,
        "out_dir": "output/realizations",
    },
}


# =============================================================================
# Helpers
# =============================================================================
def load_grid(path):
    """Load a grid by extension. Returns (array north-up, meta)."""
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext in (".tif", ".tiff"):
        z, meta = load_geotiff_grid(path)
    elif ext in (".asc", ".txt"):
        z, meta = load_esri_ascii_grid(path)
    elif ext == ".grd":
        z, meta = load_surfer7_grid(path)
    else:
        raise ValueError(f"Unsupported grid format: {path}")
    z = np.asarray(z, dtype=float)
    nodata = meta.get("nodata")
    if nodata is not None:
        z[z == nodata] = np.nan
    return z, meta


def check_aligned(grid, ref_shape, name):
    if grid.shape != ref_shape:
        raise ValueError(f"{name} has shape {grid.shape}, model grid is {ref_shape}")


# =============================================================================
# STEP 1 — the deterministic model
# =============================================================================
def load_model(region_config_path):
    cfg = RegionConfig.from_file(region_config_path)
    if cfg.surface_dir is not None and not Path(cfg.surface_dir).is_absolute():
        cfg.surface_dir = str(Path(region_config_path).parent / cfg.surface_dir)

    top, meta = load_grid(cfg.surface_path(0))
    bottoms = [load_grid(cfg.surface_path(i))[0] for i in cfg.modelled_layer_indices]
    if meta.get("crs") is None and cfg.crs:
        meta = dict(meta, crs=cfg.crs)

    # GeoTIFFs are north-up; the theme engine works south-up. Flip once here.
    grid_x, grid_y, _, _ = grid_coordinates_from_esri_meta(meta)
    terrain = np.flipud(top)
    bottom_stack = np.stack([np.flipud(b) for b in bottoms], axis=-1)

    print(f"[1] Model '{cfg.name}': {terrain.shape[0]} x {terrain.shape[1]} cells "
          f"@ {meta['dx']:.0f} m, {cfg.n_modelled_layers} modelled boundaries")
    return cfg, meta, grid_x, grid_y, terrain, bottom_stack


# =============================================================================
# STEP 2 — support grids
# =============================================================================
def class_grid_to_model(path, model_meta, name, fill_class=0, nsub=4):
    """Load a CLASS grid (complexity, model quality) and put it on the model grid.

    Works by coordinates, so it handles any combination of
      * different origin (e.g. a half-cell shift: 541950 vs 542000 on a 100 m grid),
      * different extent (larger national map vs. your site),
      * different cell size.
    Origins are compared as cell CORNERS: GeoSigma's readers already convert Surfer
    .grd node coordinates and ESRI 'xllcenter' headers to corners, so a grid whose
    header *looks* 50 m off may in fact be aligned.

    Each model cell is sampled at nsub x nsub points spread over the cell; the cell
    gets the most frequent class among them (a vote). Class 0 / no-data does not
    vote. Ties go to the HIGHER class (= more complex = shorter correlation range =
    more conservative uncertainty). Cells with no valid sample get ``fill_class``.
    """
    z, m = load_grid(path)                      # north-up, corner-referenced meta
    z = np.flipud(z)                            # south-up like the engine
    mx, my = model_meta, m

    dx_off = my["xllcorner"] - mx["xllcorner"]
    dy_off = my["yllcorner"] - mx["yllcorner"]
    same_cells = (abs(my["dx"] - mx["dx"]) < 1e-6 and abs(my["dy"] - mx["dy"]) < 1e-6)
    snapped = (same_cells and abs(dx_off / mx["dx"] - round(dx_off / mx["dx"])) < 1e-6
               and abs(dy_off / mx["dy"] - round(dy_off / mx["dy"])) < 1e-6)
    print(f"[2] {name}: {m['ncols']} x {m['nrows']} cells @ {my['dx']:g} m, "
          f"corner offset vs model: dx={dx_off:+g} m, dy={dy_off:+g} m "
          f"-> {'cell edges coincide (crop/pad only)' if snapped else 'RESAMPLED by majority vote'}")

    gx, gy, _, _ = grid_coordinates_from_esri_meta(mx)           # model cell centres
    off = ((np.arange(nsub) + 0.5) / nsub - 0.5)
    sx = (gx[:, None] + off[None, :] * mx["dx"]).ravel()          # (nx*nsub,)
    sy = (gy[:, None] + off[None, :] * mx["dy"]).ravel()          # (ny*nsub,)
    ix = np.floor((sx - my["xllcorner"]) / my["dx"]).astype(int)
    iy = np.floor((sy - my["yllcorner"]) / my["dy"]).astype(int)
    okx = (ix >= 0) & (ix < z.shape[1])
    oky = (iy >= 0) & (iy < z.shape[0])

    samples = np.full((iy.size, ix.size), np.nan)
    samples[np.ix_(oky, okx)] = z[np.ix_(iy[oky], ix[okx])]
    samples[~np.isfinite(samples) | (samples <= 0)] = np.nan       # 0 = unknown: no vote
    ny, nx = gy.size, gx.size
    samples = samples.reshape(ny, nsub, nx, nsub).transpose(0, 2, 1, 3).reshape(ny, nx, -1)

    out = np.full((ny, nx), int(fill_class), dtype=int)
    best = np.zeros((ny, nx), dtype=int)
    classes = np.unique(samples[np.isfinite(samples)]).astype(int)
    for c in classes:                                  # ascending: ties -> higher class
        cnt = np.sum(samples == c, axis=-1)
        win = (cnt > 0) & (cnt >= best)
        out[win], best[win] = c, cnt[win]

    n_cells = ny * nx
    ambiguous = np.sum((best > 0) & (best < samples.shape[-1]))
    missing = np.sum(best == 0)
    print(f"    classes {classes.tolist()}; {ambiguous / n_cells:.0%} of model cells straddle "
          f"several {name} cells; {missing / n_cells:.0%} outside/no-data -> class {fill_class}")
    return out


def load_support_grids(conf, terrain, meta):
    complexity = None
    if conf["complexity_grid"]:
        complexity = class_grid_to_model(conf["complexity_grid"], meta, "complexity grid",
                                         fill_class=conf.get("complexity_fill_class", 0))

    if conf["model_class_grid"]:
        model_class = class_grid_to_model(conf["model_class_grid"], meta, "model-class grid",
                                          fill_class=conf["model_class_if_no_grid"])
    else:
        model_class = np.full(terrain.shape, conf["model_class_if_no_grid"], dtype=int)

    print(f"[2] Complexity grid: {'yes' if complexity is not None else 'none (uniform)'}; "
          f"model classes present: {sorted(np.unique(model_class).tolist())}")
    return complexity, model_class


# =============================================================================
# STEP 3 — geophysical themes
# =============================================================================
def build_geophysics_themes(conf, grid_x, grid_y, n_layers, complexity, meta):
    themes = {}
    n_prereq = conf["n_prereq"] if conf["n_prereq"] is not None else n_layers
    for entry in conf["themes"]:
        manifest = Path(entry["manifest"])
        # A manifest declares complexity as a path, "none", or "external".
        # We pass the in-memory grid only when the manifest asks for it.
        man_text = manifest.read_text()
        wants_external = "complexity: external" in man_text
        loaded = load_theme_data(
            manifest,
            complexity=complexity if wants_external else None,
            grid_meta=None if wants_external else meta,
        )
        if loaded.n_layers != n_layers:
            raise ValueError(f"{manifest}: n_layers={loaded.n_layers}, model has {n_layers}")

        spec_ref = loaded.spec_name
        if spec_ref.endswith((".yaml", ".yml")):
            spec_path = Path(spec_ref)
            if not spec_path.is_absolute():
                spec_path = manifest.parent / spec_path
            spec = load_spec_yaml(spec_path)          # your own uncertainty model
        else:
            spec = resolve_theme_spec(spec_ref, n_prereq, loaded.spec_params)  # Danish built-in

        grid = build_theme(loaded.data, spec, grid_x, grid_y, n_layers)
        themes[entry["name"]] = grid
        frac = np.mean(np.isfinite(grid) & (grid < NODATA_VARIANCE), axis=(0, 1))
        print(f"[3] Theme {entry['name']:10s} ({spec.name}): {loaded.data.xs.size} points; "
              "informed fraction per layer: " + " ".join(f"{f:.2f}" for f in frac))
    if not themes:
        print("[3] No geophysical themes configured")
    return themes


# =============================================================================
# STEP 4 — borehole theme
# =============================================================================
# Danish quality tables from MATLAB get_well_theme.m (27/6 2023):
#   var0 = K[q]^2 + (gradient[q] * depth)^2
SNAP_GRADIENTS = [0.5 * g for g in (0.02, 0.03, 0.07, 0.1)]
SNAP_KS = [0.15, 1.0, 2.0, 2.5]
JUP_GRADIENTS = [0.5 * g for g in (0.025, 0.035, 0.075, 0.105)]
JUP_KS = [0.4, 1.5, 2.5, 3.0]


def build_borehole_theme(wells_csv, grid_x, grid_y, n_layers, complexity, meta):
    df = pd.read_csv(wells_csv)
    required = {"x", "y", "layer", "z_boundary", "terrain", "quality", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{wells_csv}: missing columns {sorted(missing)}")

    # Complexity class at each well (range comes from the class AT THE WELL).
    if complexity is not None:
        ix = np.clip(((df.x.values - grid_x[0]) / meta["dx"] + 0.5).astype(int), 0, grid_x.size - 1)
        iy = np.clip(((df.y.values - grid_y[0]) / meta["dy"] + 0.5).astype(int), 0, grid_y.size - 1)
        df["comp_class"] = complexity[iy, ix]
    else:
        df["comp_class"] = 1

    snap_layers, jup_layers = [], []
    for layer in range(1, n_layers + 1):
        for src, out, grads, ks in (("snap", snap_layers, SNAP_GRADIENTS, SNAP_KS),
                                    ("jupiter", jup_layers, JUP_GRADIENTS, JUP_KS)):
            sel = df[(df.layer == layer) & (df.source.str.lower() == src)]
            if sel.empty:
                out.append(None)
                continue
            out.append(WellSet(
                xs=sel.x.to_numpy(float),
                ys=sel.y.to_numpy(float),
                depth=np.maximum(sel.terrain.to_numpy(float) - sel.z_boundary.to_numpy(float), 0.0),
                quality=sel.quality.to_numpy(int),
                comp_class=sel.comp_class.to_numpy(int),
                gradients=grads,
                Ks=ks,
            ))

    grid = build_well_theme(grid_x, grid_y, snap_layers, jup_layers, search_radius=6)
    print(f"[4] Well theme: {df.well_id.nunique() if 'well_id' in df else len(df)} wells, "
          f"{len(df)} boundary picks")
    return grid


# =============================================================================
# STEP 5 — combine, floor, export
# =============================================================================
def combine_and_export(themes, model_class, terrain, bottom_stack, meta, out_dir):
    themes = dict(themes)
    themes["model"] = model_theme(model_class, terrain, bottom_stack)   # background prior
    all_themes = list(themes.values())
    combined = combine_variances(all_themes)
    final = apply_floor(combined, minimum_map(terrain, bottom_stack))

    # Outside the model (NaN surfaces) there is no variance.
    outside = ~np.isfinite(terrain[:, :, None] + bottom_stack)
    final = np.where(outside, np.nan, final)

    out_dir = Path(out_dir)
    write_variance_stack(np.flip(final, axis=0), meta, out_dir)        # variance_NN.tif (kriging handoff)
    write_display_stack(np.flip(final, axis=0), meta, out_dir,         # std_NN.tif (m, for QGIS)
                        informed=np.flip(informed_mask(all_themes), axis=0))
    print(f"[5] Combined {len(all_themes)} themes -> {out_dir}/variance_NN.tif and std_NN.tif")
    med = np.sqrt(np.nanmedian(final, axis=(0, 1)))
    print("    median std per layer [m]: " + " ".join(f"{s:.1f}" for s in med))
    return final, themes


# =============================================================================
# STEP 6 — EXPERIMENTAL realizations (not the GeoSigma v1.0.0 method)
# =============================================================================
def _bilinear_upsample(coarse, cx, cy, fx, fy):
    """Bilinear interpolation of coarse (ny, nx, n) onto fine axes fx, fy."""
    def weights(c, f):
        i = np.clip(np.searchsorted(c, f) - 1, 0, c.size - 2)
        w = np.clip((f - c[i]) / (c[i + 1] - c[i]), 0.0, 1.0)
        return i, w
    ix, wx = weights(cx, fx)
    iy, wy = weights(cy, fy)
    a = coarse[iy][:, ix]
    b = coarse[iy][:, ix + 1]
    c = coarse[iy + 1][:, ix]
    d = coarse[iy + 1][:, ix + 1]
    wx = wx[None, :, None]
    wy = wy[:, None, None]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def simulate_realizations(final_var, terrain, bottom_stack, grid_x, grid_y, rconf):
    rng_seed = rconf["seed"]
    np.random.seed(rng_seed)          # get_reals_cholesky uses np.random
    ny, nx, n_layers = final_var.shape

    # Coarse simulation grid so the Cholesky stays tractable.
    step = max(1, int(np.ceil(np.sqrt(nx * ny / rconf["max_sim_cells"]))))
    cx = np.append(grid_x[::step], grid_x[-1]) if (nx - 1) % step else grid_x[::step]
    cy = np.append(grid_y[::step], grid_y[-1]) if (ny - 1) % step else grid_y[::step]
    cxx, cyy = np.meshgrid(cx, cy)
    pos = np.column_stack([cxx.ravel(), cyy.ravel()])
    print(f"[6] Simulating on {cy.size} x {cx.size} coarse grid (step {step} cells), "
          f"range {rconf['corr_range_m']:.0f} m, {rconf['n_reals']} realizations")

    # Unit-sill Gaussian correlation, factorised once and reused for every layer.
    corr, _ = precal_cov(pos, pos, f"1 Gau({rconf['corr_range_m']})")
    # Standard-normal correlated fields: (n_coarse, n_layers * n_reals)
    z = get_reals_cholesky(corr, Nreals=n_layers * rconf["n_reals"])
    z = z.reshape(cy.size, cx.size, n_layers, rconf["n_reals"])

    std = np.sqrt(np.where(np.isfinite(final_var), final_var, 0.0))
    upper0 = np.concatenate([terrain[:, :, None], bottom_stack[:, :, :-1]], axis=-1)
    zero_thick = (upper0 - bottom_stack) <= 1e-6

    reals = np.empty((rconf["n_reals"], ny, nx, n_layers))
    for r in range(rconf["n_reals"]):
        noise = _bilinear_upsample(z[:, :, :, r], cx, cy, grid_x, grid_y)  # (ny, nx, n_layers)
        surf = bottom_stack + noise * std
        # Stratigraphic order: no boundary above terrain or above the one above it.
        # Layers absent in the model (zero thickness) stay absent: tie to the boundary above.
        prev = terrain.copy()
        for s in range(n_layers):
            cur = np.minimum(surf[:, :, s], prev)
            cur = np.where(zero_thick[:, :, s], prev, cur)
            surf[:, :, s] = cur
            prev = cur
        reals[r] = surf
    return reals


def export_realizations(reals, meta, cfg, out_dir):
    import rasterio
    from rasterio.transform import from_origin

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_reals, ny, nx, n_layers = reals.shape
    transform = from_origin(meta["xllcorner"], meta["yllcorner"] + ny * meta["dy"],
                            meta["dx"], meta["dy"])
    names = cfg.modelled_layer_names
    for r in range(n_reals):
        rdir = out_dir / f"real_{r + 1:03d}"
        rdir.mkdir(exist_ok=True)
        for s in range(n_layers):
            data = np.flipud(reals[r, :, :, s])        # back to north-up
            with rasterio.open(rdir / f"{names[s]}.tif", "w", driver="GTiff", height=ny,
                               width=nx, count=1, dtype="float32", crs=meta.get("crs"),
                               transform=transform, nodata=-9999.0) as ds:
                ds.write(np.where(np.isfinite(data), data, -9999.0).astype("float32"), 1)
    # Ensemble statistics: easy sanity check against the variance maps.
    ens_std = np.nanstd(reals, axis=0)
    np.save(out_dir / "ensemble_std.npy", ens_std)
    print(f"[6] Wrote {n_reals} realizations x {n_layers} surfaces to {out_dir}/real_NNN/")


def qa_figure(themes, final, reals, grid_x, grid_y, out_png, layers=None):
    """Save a QA figure: per-theme coverage, final std, and realization spread."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = final.shape[2]
    layers = layers or sorted({0, n_layers // 3, (2 * n_layers) // 3, n_layers - 1})
    ext = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]
    names = [k for k in themes if k != "model"]
    ncols = len(layers)
    nrows = len(names) + 1 + (2 if reals is not None else 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows), squeeze=False)

    def show(ax, img, title, cmap="viridis", vmin=None, vmax=None, label="std [m]"):
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)

    for r, name in enumerate(names):
        for c, s in enumerate(layers):
            g = themes[name][:, :, s]
            g = np.where(np.isfinite(g) & (g < NODATA_VARIANCE), np.sqrt(g), np.nan)
            show(axs[r, c], g, f"{name} theme, layer {s + 1}", vmin=0, vmax=20)
    r = len(names)
    for c, s in enumerate(layers):
        show(axs[r, c], np.sqrt(final[:, :, s]), f"FINAL std, layer {s + 1}", vmin=0)
    if reals is not None:
        for c, s in enumerate(layers):
            show(axs[r + 1, c], reals[0, :, :, s], f"Realization 1, layer {s + 1}",
                 cmap="terrain", label="elev [m]")
            show(axs[r + 2, c], np.nanstd(reals[:, :, :, s], axis=0),
                 f"Ensemble std, layer {s + 1}", vmin=0)
    fig.suptitle("GeoSigma QA — white = no information from that source", fontsize=11)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"    QA figure: {out_png}")


# =============================================================================
# DEMO — build a fake "Danish site" from GeoSigma's bundled Jutland sample
# =============================================================================
def build_demo_inputs(repo_root: Path, demo_dir: Path):
    """Write region config, a SkyTEM ThemeData bundle and a wells CSV — synthetic data."""
    rng = np.random.default_rng(1)
    demo_dir.mkdir(parents=True, exist_ok=True)
    sample = repo_root / "examples" / "data" / "Jutland_hydrostrat_model"
    rc = json.loads((sample / "region_config.json").read_text())
    rc["surface_dir"] = str(sample)
    (demo_dir / "region_config.json").write_text(json.dumps(rc, indent=2))

    cfg, meta, gx, gy, terrain, bottoms = load_model(demo_dir / "region_config.json")
    n = cfg.n_modelled_layers
    depth = terrain[:, :, None] - bottoms
    upper = np.concatenate([terrain[:, :, None], bottoms[:, :, :-1]], axis=-1)
    thick = upper - bottoms

    def sample(xs, ys, grid):
        ix = np.clip(((xs - gx[0]) / meta["dx"] + 0.5).astype(int), 0, gx.size - 1)
        iy = np.clip(((ys - gy[0]) / meta["dy"] + 0.5).astype(int), 0, gy.size - 1)
        return grid[iy, ix]

    # SkyTEM: flight lines 200 m apart, sounding every 30 m, over the western half.
    lines = np.arange(gy[0], gy[-1], 200.0)
    along = np.arange(gx[0], gx[0] + 0.5 * (gx[-1] - gx[0]), 30.0)
    xs = np.tile(along, lines.size)
    ys = np.repeat(lines, along.size) + rng.normal(0, 10, xs.size)
    cols = {"x": xs, "y": ys}
    # DEMO SHORTCUT: real "thick" = thickness of the GERDA resistivity layer containing
    # the boundary, not the hydrostratigraphic layer thickness. Use gerda_to_themedata.py.
    d, t = sample(xs, ys, depth), sample(xs, ys, thick)
    for s in range(n):
        cols[f"depth__L{s + 1:02d}"] = d[:, s]
    for s in range(n):
        cols[f"thick__L{s + 1:02d}"] = t[:, s]
    cols["doi"] = rng.uniform(120, 250, xs.size)
    sky = demo_dir / "skytem"
    sky.mkdir(exist_ok=True)
    pd.DataFrame(cols).to_csv(sky / "skytem_points.csv", index=False)
    (sky / "skytem.manifest.yaml").write_text(_manifest_text("SkyTEM (synthetic demo)",
                                                             "skytem", "skytem_points.csv", n))

    # Wells: 60 boreholes, each informs the boundaries it reaches.
    rows = []
    wx = rng.uniform(gx[0], gx[-1], 60)
    wy = rng.uniform(gy[0], gy[-1], 60)
    wterr = sample(wx, wy, terrain[:, :, None])[:, 0]
    wbot = sample(wx, wy, bottoms)
    wdepth = rng.uniform(20, 200, 60)
    for i in range(60):
        src = "snap" if i < 20 else "jupiter"
        q = int(rng.integers(1, 5))
        for s in range(n):
            if wterr[i] - wbot[i, s] <= wdepth[i] and np.isfinite(wbot[i, s]):
                rows.append(dict(well_id=f"DGU{i:04d}", x=wx[i], y=wy[i], layer=s + 1,
                                 z_boundary=wbot[i, s], terrain=wterr[i], quality=q, source=src))
    pd.DataFrame(rows).to_csv(demo_dir / "wells.csv", index=False)

    return {
        **CONFIG,
        "region_config": str(demo_dir / "region_config.json"),
        "themes": [{"name": "SkyTEM", "manifest": str(sky / "skytem.manifest.yaml")}],
        "wells_csv": str(demo_dir / "wells.csv"),
        "out_dir": str(demo_dir / "output" / "variance_maps"),
        "realizations": {**CONFIG["realizations"], "n_reals": 5,
                         "out_dir": str(demo_dir / "output" / "realizations")},
    }


def _manifest_text(name, spec, csv_name, n_layers, complexity="none"):
    return f"""format_version: 1
name: {name}
spec: {spec}                 # built-in Danish spec name, or path to your own .spec.yaml
points_csv: {csv_name}
n_layers: {n_layers}
crs: EPSG:25832
x_column: x
y_column: y
attributes:
  depth:                    # depth BELOW TERRAIN of each modelled boundary at the sounding (m)
    per_layer: true
    prefix: depth__L
  thick:                    # thickness of each layer at the sounding (m)
    per_layer: true
    prefix: thick__L
  doi:                      # depth of investigation of the sounding (m below terrain)
    per_layer: false
    column: doi
complexity: {complexity}    # path to grid | none | external (= grid from the workflow script)
_notes:
  source: "YOUR DATA - describe the dataset (e.g. GERDA project id, contractor, year)"
  filtering: "YOUR DATA - e.g. removed coupled soundings; de-duplicated on (x, y)"
"""


def write_templates(out: Path, n_layers: int = 10):
    out.mkdir(parents=True, exist_ok=True)
    (out / "skytem").mkdir(exist_ok=True)
    cols = ["x", "y"] + [f"depth__L{i:02d}" for i in range(1, n_layers + 1)] \
        + [f"thick__L{i:02d}" for i in range(1, n_layers + 1)] + ["doi"]
    pd.DataFrame(columns=cols).to_csv(out / "skytem" / "skytem_points.csv", index=False)
    (out / "skytem" / "skytem.manifest.yaml").write_text(
        _manifest_text("SkyTEM <site>", "skytem", "skytem_points.csv", n_layers))
    pd.DataFrame(columns=["well_id", "x", "y", "layer", "z_boundary", "terrain",
                          "quality", "source"]).to_csv(out / "wells.csv", index=False)
    layer_names = ["terrain"] + [f"layer_{i:02d}_bottom" for i in range(1, n_layers + 1)]
    (out / "region_config.json").write_text(json.dumps({
        "name": "<my Danish site>",
        "layer_names": layer_names,
        "has_bottom": False,
        "surface_dir": "surfaces",
        "file_ext": ".tif",
        "crs": "EPSG:25832",
    }, indent=2))
    print(f"Templates written to {out}/ (set n_layers to match your model: --n-layers)")


# =============================================================================
# Main
# =============================================================================
def run(conf):
    cfg, meta, grid_x, grid_y, terrain, bottom_stack = load_model(conf["region_config"])
    n_layers = cfg.n_modelled_layers
    complexity, model_class = load_support_grids(conf, terrain, meta)

    themes = build_geophysics_themes(conf, grid_x, grid_y, n_layers, complexity, meta)
    if conf["wells_csv"]:
        themes["wells"] = build_borehole_theme(conf["wells_csv"], grid_x, grid_y,
                                               n_layers, complexity, meta)

    final, themes = combine_and_export(themes, model_class, terrain, bottom_stack,
                                       meta, conf["out_dir"])

    reals = None
    rconf = conf["realizations"]
    if rconf["enabled"]:
        reals = simulate_realizations(final, terrain, bottom_stack, grid_x, grid_y, rconf)
        export_realizations(reals, meta, cfg, rconf["out_dir"])
        ens = np.nanmedian(np.nanstd(reals, axis=0), axis=(0, 1))
        print("    ensemble std per layer [m]: " + " ".join(f"{s:.1f}" for s in ens))

    qa_figure(themes, final, reals, grid_x, grid_y, Path(conf["out_dir"]) / "qa_overview.png")
    return final, reals


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--demo", action="store_true", help="run on the bundled Jutland sample")
    ap.add_argument("--geosigma-repo", default=".", help="path to the GeoSigma repo (for --demo)")
    ap.add_argument("--write-templates", action="store_true")
    ap.add_argument("--n-layers", type=int, default=10)
    args = ap.parse_args()

    if args.write_templates:
        write_templates(Path("templates"), args.n_layers)
    elif args.demo:
        run(build_demo_inputs(Path(args.geosigma_repo).resolve(), Path("demo_site")))
    else:
        run(CONFIG)
