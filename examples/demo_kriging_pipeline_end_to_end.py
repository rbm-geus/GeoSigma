# -*- coding: utf-8 -*-
"""
Demo: the new preprocess/kriging pipeline, end to end

Steps:
1. Load a real surface (the whole bundled Jutland top-surface sample). It is
   used *raw* -- it is not pre-detrended, because the detrending is itself
   part of the pipeline (step 3).
2. Draw conditioning points with INPOX (geosigma.inpox), on that raw surface
   (matching drawpoints.m, which is handed the raw layer grid).
3. `estimate_range_sill_surface` -- the pretrained NN (inference, not
   training) run at **every cell**, conditioned on the sparse INPOX points,
   matching get_interpolated_minigrids_parallel.m: its `mask` argument (where
   it estimates) is the whole valid-domain mask, while the drawn points only
   pick which values fall inside any given window. The same pass returns the
   **local mean** of each window -- the low-frequency (LF) background model:
   the reference forms the data it kriges as `GLOBAL.img_dobs =
   GLOBAL.img_FOHM - GLOBAL.img_means` (setup_GLOBAL_structure.m:75) and
   reconstructs an absolute realization as `real = img_means + m_est + real`
   (Main.m:1038). So the residual kriged here is surface minus that LF mean
   -- not a plane fit or a global mean.
4. `build_kriging_zones` -- SOM-cluster those estimates into kriging zones
   (matching get_clusters.m), with each zone's representative range/sill.
5. Krige each cluster's connected components and draw realizations with the
   local-kriging driver (geosigma.kriging: buffers, windows, state,
   simulation) -- not the raw geosigma.core primitives directly, unlike
   demo_synthetic_kriging_inversion.py -- then add the LF mean back to get
   absolute surfaces again.
6. Stack the layers (geosigma.assembly), clamping any boundary that was
   drawn above the one above it, realization by realization.

Steps 2-5 run once per modelled boundary and know nothing about each other:
that independence is what makes each boundary's uncertainty honest, and
also what lets two boundaries cross. Step 6 is where that is resolved. The
demo models two boundaries -- the real cutout, and the same cutout
``LAYER_THICK`` metres lower, treated as a fully independent layer.

No real geophysical uncertainty data is bundled with this repo (see
matlab_reference/README.md's note on data files not being included), so the
two variance maps in step 5 are synthetic stand-ins; everything else is real.

This is the composed, multi-stage run geosigma.kriging's own docstring
points to; demo_geosigma_pipeline.py and demo_synthetic_kriging_inversion.py
predate this package and only exercise geosigma.core directly.

Requires the optional ``torch``/``scipy`` extras for the NN estimator:
``pip install "geosigma[nn]"``.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from utils import load_geotiff_grid
from geosigma.assembly import stack_realizations
from geosigma.inpox import draw_points_inpox
from geosigma.preprocess.rangesill import (
    load_default_rangesill_estimator,
    estimate_range_sill_surface,
)
from geosigma.preprocess.som_clusters import build_kriging_zones
from geosigma.kriging import (
    pack_kriging_grid,
    build_windows,
    simulate_layer,
    to_absolute,
)

SEED = 1
np.random.seed(SEED)

SLICE_ROW = 25  # row index for the 1D reference-vs-realizations slice (figure 3)

# Conditioning points this far (metres) either side of SLICE_ROW are drawn on
# the profile too, shrinking with distance -- a profile is a 1D cut through a
# 2D point cloud, so the points that actually constrain it mostly do not lie
# exactly on it.
PROFILE_BAND = 200.0


# %% -------------------------------------------------
# 1. The real surface, used raw
# -------------------------------------------------
z_full_1, meta = load_geotiff_grid(
    "data/Jutland_hydrostrat_model/top_surface.tif"
)

z_full_2, meta = load_geotiff_grid(
    "data/Jutland_hydrostrat_model/layer_1_bottom.tif"
)

dx = float(meta["cellsize"])

# The whole bundled sample GeoTIFF (~100x100 cells) -- it happens to be
# entirely nodata-free, so no cropping is needed here, unlike a larger real
# raster where you'd want to window out a nodata-free subset.
nodata = meta.get("nodata_value", None)
surface1 = np.where(z_full_1 == nodata, np.nan, z_full_1) if nodata is not None else z_full_1
assert np.isfinite(surface1).all(), "bundled raster unexpectedly has nodata cells"
assert surface1.shape[0] == surface1.shape[1], "pipeline below assumes a square grid"
CROP = surface1.shape[0]
assert (
    0 <= SLICE_ROW < CROP
), f"SLICE_ROW={SLICE_ROW} out of range for a {CROP}-row grid"
print(
    f"Surface: {CROP}x{CROP} cells at {dx:.0f} m, "
    f"elevation range [{surface1.min():.1f}, {surface1.max():.1f}] m"
)
surface2 = np.where(z_full_2 == nodata, np.nan, z_full_2) if nodata is not None else z_full_2

# The ensemble is far larger than what gets plotted, because the ensemble
# *mean* in figure 2 is only meaningful with enough draws behind it: the
# standard error of a mean over N draws is sigma/sqrt(N), so 7 would leave
# ~38% of the posterior std as visible noise where 200 leaves ~7%. It is
# nearly free to ask for: each window's Cholesky factor is computed once and
# then reused for every draw, so the cost of going from 8 to 200 is a wider
# matrix multiply, not 25x the work.
N_REALS = 200
N_PLOT = 7  # how many individual realizations the figures show

# Madsen et al. (2024) Table 1, real-world case (T4), tuned on a FOHM surface
# -- the same model this cutout comes from. P0 is a *floor*, not the draw
# fraction: roughness only ever adds to it, so the realised fraction lands
# above P0 (~22% here, at P0 = 0.15).
ext_vals = {
    "P0": 0.15,
    "L0": 1.0,
    "A": 1.5,
    "B": 3.0,
    "C": 5.0,
    "D": 1.0,
    "E": 0.8,
    "g": 2.0,
}


@dataclass
class LayerRun:
    """Everything steps 2-5 produce for one modelled boundary."""

    label: str
    surface: np.ndarray
    points_mask: np.ndarray
    rs: object  # RangeSillSurface
    cluster_grid: np.ndarray
    stats: dict
    result: object  # LayerRealizations
    kriged_surface: np.ndarray
    abs_reals: np.ndarray


def run_layer(layer_surface, label, seed):
    """Steps 2-5 for one boundary, start to finish and independent of others.

    Every layer gets its own INPOX draw, its own range/sill estimates, its
    own kriging zones and its own realizations -- which is what makes each
    boundary's uncertainty honest, and also what lets them cross. Step 6
    deals with that.
    """
    # Seeded here rather than per call site, so the INPOX draw and the SOM
    # differ between layers too. Sharing a seed would hand both layers the
    # same conditioning points -- the two surfaces differ by a constant, so
    # their Laplacians, and hence their draw probabilities, are identical.
    np.random.seed(seed)

    # --- 2. Conditioning points (INPOX) ---
    points_mask = draw_points_inpox(layer_surface, ext_vals, dx=dx, dy=dx)
    print(
        f"[{label}] INPOX drew {int(points_mask.sum())} points "
        f"out of {points_mask.size} cells"
    )

    # --- 3. Range/sill + the LF mean model (NN inference, not training) ---
    # One call covers what the reference does in
    # get_interpolated_minigrids_parallel.m: estimate at every cell (its
    # `mask` argument is the whole valid-domain mask) using the sparse INPOX
    # points only to decide which values fall inside each window, and return
    # the local window means as the LF background model plus the residual to
    # krige. It also reflection-pads internally so edge cells still get a
    # full window.
    rs = estimate_range_sill_surface(
        layer_surface, points_mask, estimator=estimator, dx=dx
    )
    n_valid = int((np.isfinite(rs.range_field) & np.isfinite(rs.sill_field)).sum())
    print(
        f"[{label}] range/sill at {n_valid}/{rs.range_field.size} cells, "
        f"median range={np.nanmedian(rs.range_field):.0f} m, "
        f"median sill={np.nanmedian(rs.sill_field):.1f}"
    )
    print(
        f"[{label}] LF mean [{rs.lf_mean.min():.1f}, {rs.lf_mean.max():.1f}] m  |  "
        f"residual std = {rs.residual.std():.2f} m "
        f"(surface std was {layer_surface.std():.2f} m)"
    )

    # --- 4. Cluster into kriging zones (SOM, matching get_clusters.m) ---
    # SOM-clusters the estimates, computes each zone's representative
    # range/sill from the cells the SOM actually classified, then fills the
    # rest so every cell belongs to a zone (build_windows would otherwise
    # leave most of the grid unkriged -- som_cluster_grid deliberately labels
    # only what it's given).
    cluster_grid, stats = build_kriging_zones(rs.range_field, rs.sill_field, seed=seed)
    print(
        f"[{label}] SOM formed {int(cluster_grid.max())} kriging zones, "
        f"filled to full coverage"
    )

    # --- 5. Local kriging driver: windows, solve, simulate ---
    # Two *different* variance maps, as in the reference pipeline (Main.m
    # passes `sqrt(final_variance_themes)` as GLOBAL.img_unc and `modeltheme`
    # separately; local_kriger.m reads the latter for each window's sill).
    # Both are synthetic stand-ins here -- no real theme data is bundled --
    # but modeltheme is scaled to the residual's own variance so the prior is
    # physically consistent with the data being kriged; too small a sill caps
    # how far a realization can deviate from the mean.
    _, xx = np.mgrid[0:CROP, 0:CROP]
    data_variance = np.full_like(rs.residual, 4.0)  # +/- 2 m per point
    modeltheme = rs.residual.var() * (0.8 + 0.4 * (xx / CROP))

    grid = pack_kriging_grid(rs.residual, data_variance, dx=dx)
    windows = build_windows(
        grid,
        cluster_labels=cluster_grid,
        active_mask=np.ones_like(rs.residual, dtype=bool),
        conditioning_mask=points_mask,
        cluster_stats=stats,
        modeltheme=modeltheme,
    )
    result = simulate_layer(grid, windows, N_REALS, seed=seed)

    # real = img_means + m_est + real  (Main.m:1038) -- absolute surfaces
    kriged_surface, abs_reals = to_absolute(result, rs.lf_mean)
    print(
        f"[{label}] {len(windows)} windows, posterior std "
        f"[{np.nanmin(result.post_std):.2f}, {np.nanmax(result.post_std):.2f}] m, "
        f"realization std = {np.nanstd(abs_reals):.2f} m"
    )
    return LayerRun(
        label=label,
        surface=layer_surface,
        points_mask=points_mask,
        rs=rs,
        cluster_grid=cluster_grid,
        stats=stats,
        result=result,
        kriged_surface=kriged_surface,
        abs_reals=abs_reals,
    )


estimator = load_default_rangesill_estimator()

# %% -------------------------------------------------
# 2-5. Run both layers, independently
# -------------------------------------------------

layer1 = run_layer(surface1, "layer 1", SEED)
layer2 = run_layer(surface2, "layer 2", SEED + 100)

# Figures 1-3 below look at layer 1; these aliases keep them readable.
points_mask = layer1.points_mask
rs = layer1.rs
cluster_grid = layer1.cluster_grid
n_clusters = int(cluster_grid.max())
result = layer1.result
kriged_surface = layer1.kriged_surface
abs_reals = layer1.abs_reals

# %% -------------------------------------------------
# Figure 1: the whole chain
# -------------------------------------------------
vmin, vmax = float(surface1.min()), float(surface1.max())
fig, axes = plt.subplots(2, 4, figsize=(15, 5))

im = axes[0, 0].imshow(surface1, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0, 0].set_title("1. Real surface (raw)")
fig.colorbar(im, ax=axes[0, 0], shrink=0.8)

# Estimated range, now dense (every cell), with the actual sparse INPOX
# points overlaid for context -- that scatter is no longer what's coloured
# by range (plotting all 10000 estimated cells as markers would be both
# misleading -- it'd look like "every cell is an INPOX point" -- and slow).
im = axes[0, 1].imshow(rs.range_field, cmap="plasma")
inpox_rows, inpox_cols = np.nonzero(points_mask)
axes[0, 1].scatter(inpox_cols, inpox_rows, s=3, color="black", alpha=0.5)
axes[0, 1].set_title("2+3. Estimated range\n(dots: actual INPOX points)")
fig.colorbar(im, ax=axes[0, 1], shrink=0.8, label="range (m)")

im = axes[0, 2].imshow(cluster_grid, cmap="tab20")
axes[0, 2].set_title(f"4. SOM kriging zones\n({n_clusters} clusters)")
fig.colorbar(im, ax=axes[0, 2], shrink=0.8)

im = axes[0, 3].imshow(rs.residual, cmap="RdBu_r")
axes[0, 3].set_title("Residual kriged\n(surface - LF mean)")
fig.colorbar(im, ax=axes[0, 3], shrink=0.8, label="m")

im = axes[1, 0].imshow(rs.lf_mean, cmap="viridis", vmin=vmin, vmax=vmax)
axes[1, 0].set_title("LF mean model\n(sliding-window means)")
fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

im = axes[1, 1].imshow(kriged_surface, cmap="viridis", vmin=vmin, vmax=vmax)
axes[1, 1].set_title("LF mean + kriging mean")
fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

im = axes[1, 2].imshow(result.post_std**2, cmap="magma")
axes[1, 2].set_title("Kriging variance")
fig.colorbar(im, ax=axes[1, 2], shrink=0.8, label="m^2")

im = axes[1, 3].imshow(abs_reals[:, :, 0], cmap="viridis", vmin=vmin, vmax=vmax)
axes[1, 3].set_title("A realization\n(LF mean + kriging mean + draw)")
fig.colorbar(im, ax=axes[1, 3], shrink=0.8)

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("GeoSigma: rangesill -> clustering -> local kriging -> simulation")
plt.tight_layout()


# %% -------------------------------------------------
# Figure 2: reference, ensemble mean, and individual realizations
# -------------------------------------------------
# All nine panels share the reference's vmin/vmax, which is the point: a
# realization that looks like the top-left panel is reproducing the right
# elevations, not merely the right texture.
#
# Panel 2 is the mean *of the draws*, not the kriging mean. The two estimate
# the same thing and converge as the ensemble grows, so comparing them is a
# check on the simulation rather than a second view of the same array -- any
# systematic gap would mean the draws are not centred on the posterior they
# claim to come from.
ensemble_mean = abs_reals.mean(axis=2)
# RMS, not max: the expected gap is the standard error of a mean, which is a
# per-cell RMS quantity. A max over 10000 cells is the tail of that same
# distribution and would look alarming while meaning nothing.
mean_gap_rms = float(np.sqrt(np.nanmean((ensemble_mean - kriged_surface) ** 2)))
expected_gap = float(np.sqrt(np.nanmean(result.post_std**2)) / np.sqrt(N_REALS))
print(
    f"ensemble mean vs kriging mean over {N_REALS} realizations: "
    f"RMS difference = {mean_gap_rms:.3f} m "
    f"(expected standard error {expected_gap:.3f} m)"
)

fig2, axes2 = plt.subplots(3, 3, figsize=(12, 12))
flat2 = axes2.ravel()

im = flat2[0].imshow(surface1, cmap="viridis", vmin=vmin, vmax=vmax)
flat2[0].set_title("Reference cutout", fontsize=11, fontweight="bold")
# Heavy black frame, so the one panel that is *data* rather than model output
# is obvious at a glance.
for spine in flat2[0].spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(4)

im = flat2[1].imshow(ensemble_mean, cmap="viridis", vmin=vmin, vmax=vmax)
flat2[1].set_title(f"Mean of {N_REALS} realizations", fontsize=11)

for k, ax in enumerate(flat2[2:]):
    im = ax.imshow(abs_reals[:, :, k], cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(f"Realization {k + 1}", fontsize=10)

for ax in flat2[1:]:
    for spine in ax.spines.values():
        spine.set_color("0.8")
for ax in flat2:
    ax.set_xticks([])
    ax.set_yticks([])
fig2.colorbar(im, ax=flat2.tolist(), shrink=0.6, label="elevation (m)")
fig2.suptitle(
    f"Reference cutout (boxed), the mean of {N_REALS} realizations, "
    f"and {N_PLOT} of them individually\n(shared colour scale throughout)",
    fontsize=14,
)


# %% -------------------------------------------------
# Figure 3: one row, reference vs. realizations, with the kriging clusters
# that row passes through shaded behind them, and the range/sill estimated
# at each INPOX point on that row in a smaller panel underneath
# -------------------------------------------------
# Change SLICE_ROW at the top of the script to look at a different row.
# Only N_PLOT of the ensemble are drawn: the full 200 would overplot into a
# solid red band that hides both the spread and everything behind it.
x_slice = np.arange(CROP) * dx
fig3, (ax3, ax3b) = plt.subplots(
    2, 1, figsize=(12, 7), height_ratios=[3, 1.3], sharex=True
)

for k in range(N_PLOT):
    ax3.plot(
        x_slice,
        abs_reals[SLICE_ROW, :, k],
        color="red",
        linewidth=0.7,
        alpha=0.6,
        label="Realizations" if k == 0 else None,
        zorder=3,
    )
ax3.plot(
    x_slice,
    surface1[SLICE_ROW, :],
    color="black",
    linewidth=2,
    label="Reference surface",
    zorder=3,
)

# The kriging estimate and its own uncertainty: posterior mean +/- 2 posterior
# standard deviations. The band belongs to the *estimate*, not to the
# reference -- the reference is the truth being estimated, and carries no
# kriging uncertainty of its own.
kriged_row = kriged_surface[SLICE_ROW, :]
post_std_row = result.post_std[SLICE_ROW, :]
ax3.plot(
    x_slice,
    kriged_row,
    color="black",
    linestyle="--",
    linewidth=1.8,
    label="Kriging mean",
    zorder=3,
)
ax3.plot(
    x_slice,
    kriged_row + 2 * post_std_row,
    color="black",
    linestyle="--",
    linewidth=0.9,
    label="Kriging mean +/- 2 std",
    zorder=3,
)
ax3.plot(
    x_slice,
    kriged_row - 2 * post_std_row,
    color="black",
    linestyle="--",
    linewidth=0.9,
    zorder=3,
)

# The conditioning points that actually constrain this profile. Points lying
# exactly on the row are filled black; those off it are hollow and shrink
# with distance, out to PROFILE_BAND. Each is drawn at its own elevation --
# an off-row point sits at surface[its row, its col], not at the profile's.
pt_rows, pt_cols = np.nonzero(points_mask)
pt_dist = np.abs(pt_rows - SLICE_ROW) * dx
in_band = pt_dist <= PROFILE_BAND
pt_rows, pt_cols, pt_dist = pt_rows[in_band], pt_cols[in_band], pt_dist[in_band]
on_row = pt_dist == 0

MARKER_MAX, MARKER_MIN = 46.0, 6.0
pt_sizes = MARKER_MAX - (MARKER_MAX - MARKER_MIN) * (pt_dist / PROFILE_BAND)

ax3.scatter(
    pt_cols[on_row] * dx,
    surface1[pt_rows[on_row], pt_cols[on_row]],
    s=MARKER_MAX,
    facecolors="black",
    edgecolors="black",
    linewidths=0.8,
    zorder=5,
    label="Conditioning points on the row",
)
ax3.scatter(
    pt_cols[~on_row] * dx,
    surface1[pt_rows[~on_row], pt_cols[~on_row]],
    s=pt_sizes[~on_row],
    facecolors="white",
    edgecolors="black",
    linewidths=0.8,
    zorder=4,
    label=f"... within {PROFILE_BAND:.0f} m (smaller = further)",
)
print(
    f"profile points: {int(on_row.sum())} on row {SLICE_ROW}, "
    f"{int((~on_row).sum())} more within {PROFILE_BAND:.0f} m"
)

# Cluster membership along this row, as a background fill -- same cmap and
# same colour-to-cluster mapping (vmin/vmax) as figure 1's cluster panel, so
# a given colour means the same cluster in both figures.
ylim = ax3.get_ylim()
ax3.imshow(
    cluster_grid[SLICE_ROW : SLICE_ROW + 1, :],
    extent=[x_slice[0], x_slice[-1], ylim[0], ylim[1]],
    aspect="auto",
    cmap="tab20",
    vmin=cluster_grid.min(),
    vmax=cluster_grid.max(),
    alpha=0.3,
    zorder=0,
)
ax3.set_ylim(ylim)

ax3.set_ylabel("elevation (m)")
ax3.set_title(
    f"Row {SLICE_ROW} (y = {SLICE_ROW * dx:.0f} m): reference vs. {N_PLOT} of "
    f"{N_REALS} realizations, kriging zones shaded behind"
)
ax3.legend(loc="upper right")
plt.setp(ax3.get_xticklabels(), visible=False)

# Range/sill along this row. Estimation is dense (every cell), so these read
# straight off the grid-shaped fields -- same x axis as the panel above.
row_ranges = rs.range_field[SLICE_ROW, :]
row_sills = rs.sill_field[SLICE_ROW, :]
if np.isfinite(row_ranges).any():
    (rline,) = ax3b.plot(
        x_slice,
        row_ranges,
        "o-",
        color="tab:purple",
        ms=5,
        label="range (m)",
    )
    ax3b.set_ylabel("range (m)", color="tab:purple")
    ax3b.tick_params(axis="y", labelcolor="tab:purple")

    ax3b_sill = ax3b.twinx()
    (sline,) = ax3b_sill.plot(
        x_slice,
        row_sills,
        "s-",
        color="tab:green",
        ms=5,
        label="sill [m^2]",
    )
    ax3b_sill.set_ylabel("sill (m^2)", color="tab:green")
    ax3b_sill.tick_params(axis="y", labelcolor="tab:green")

    ax3b.legend(handles=[rline, sline], loc="upper right")
else:
    ax3b.text(
        0.5,
        0.5,
        "No valid range/sill estimate on this row",
        ha="center",
        va="center",
        transform=ax3b.transAxes,
    )

ax3b.set_xlabel("x (m)")
ax3b.set_title("Estimated range/sill along this row", fontsize=10)

plt.tight_layout()


# %% -------------------------------------------------
# 6. Stack the layers, correcting crossings realization by realization
# -------------------------------------------------
# The two boundaries were kriged with no knowledge of each other, so nothing
# so far prevents layer 2 being drawn *above* layer 1 -- a layer of negative
# thickness. get_layered_models.m fixes this at the end, per realization,
# top-down: wherever the lower boundary rises above the upper one it is
# pushed down to meet it, so the layer pinches out to zero thickness instead
# of inverting.
#
# Per realization, not on the mean. The clamp is nonlinear, so the mean of
# the corrected draws is not the correction of the mean -- and correcting
# only the mean would leave every individual realization invalid, which is
# the opposite of what an ensemble is for.
raw_thickness = layer1.abs_reals - layer2.abs_reals
model = stack_realizations([layer1.abs_reals, layer2.abs_reals])
fixed_thickness = -np.diff(model.surfaces, axis=2)[:, :, 0, :]

crossing = raw_thickness < 0
# A clamped cell has lower == upper exactly, so its thickness is -0.0 once
# differenced. Adding 0.0 normalises that back to +0.0 (IEEE 754), which is
# only cosmetic but stops the summary reading as if it still went negative.
min_thickness = float(fixed_thickness.min()) + 0.0
assert min_thickness >= 0.0, "stacking left a layer of negative thickness"
print(
    f"layer crossings before correction: {int(crossing.sum())} of "
    f"{crossing.size} (cell, realization) pairs "
    f"({crossing.mean():.1%}), worst overlap {-raw_thickness.min():.2f} m"
)
print(
    f"clamped by stack_realizations: {int(model.n_clamped[1])} "
    f"-- minimum thickness now {min_thickness:.2f} m "
)

# %% -------------------------------------------------
# Figure 4: what the correction did
# -------------------------------------------------
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 4.8))

# How often each cell needed correcting, across the whole ensemble. The
# pattern is not noise: crossings concentrate where the two layers' kriging
# disagreed most, which is away from their respective conditioning points.
im = axes4[0].imshow(crossing.mean(axis=2), cmap="magma", vmin=0, vmax=1)
axes4[0].set_title("Fraction of realizations\nwhere the layers crossed")
fig4.colorbar(im, ax=axes4[0], shrink=0.85)

# One realization's thickness, before and after, on a scale centred at zero
# so negative (invalid) thickness is unmistakable: RdBu puts negatives in
# red. Scaled to the 99th percentile rather than the max, or a couple of
# deep outliers would wash the rest of the map out to near-white.
tmax = float(np.percentile(np.abs(raw_thickness[:, :, 0]), 99))
im = axes4[1].imshow(raw_thickness[:, :, 0], cmap="RdBu", vmin=-tmax, vmax=tmax)
axes4[1].set_title(
    f"Realization 1 thickness, raw\n(red = inverted, min "
    f"{raw_thickness[:, :, 0].min():.1f} m)"
)
fig4.colorbar(im, ax=axes4[1], shrink=0.85, label="m")

im = axes4[2].imshow(fixed_thickness[:, :, 0], cmap="RdBu", vmin=-tmax, vmax=tmax)
axes4[2].set_title(
    f"Same, after stacking\n(no red left, min {fixed_thickness[:, :, 0].min():.1f} m)"
)
fig4.colorbar(im, ax=axes4[2], shrink=0.85, label="m")

for ax in axes4:
    ax.set_xticks([])
    ax.set_yticks([])

fig4.tight_layout(rect=[0, 0, 1, 0.93])

# %% -------------------------------------------------
# Figure 5: both boundaries along the profile row, one realization
# -------------------------------------------------
# One realization, not N_PLOT: overlaying several puts three lines per draw
# on the axes and the before/after pair becomes impossible to follow. The
# one shown is whichever realization crosses most often on this row, so the
# figure has something to show -- picked rather than assumed, since on any
# given row most draws cross nowhere at all.
row_crossing = crossing[SLICE_ROW, :, :]  # (nx, n_reals)
k_show = int(row_crossing.sum(axis=0).argmax())
n_cross_row = int(row_crossing[:, k_show].sum())

fig5, ax5 = plt.subplots(figsize=(12, 5))
ax5.plot(
    x_slice,
    layer1.abs_reals[SLICE_ROW, :, k_show],
    color="tab:blue",
    linewidth=1.8,
    label="Layer 1 (upper, unchanged)",
    zorder=3,
)
ax5.plot(
    x_slice,
    layer2.abs_reals[SLICE_ROW, :, k_show],
    color="tab:red",
    linewidth=1.2,
    linestyle="--",
    label="Layer 2, as drawn",
    zorder=3,
)
ax5.plot(
    x_slice,
    model.surfaces[SLICE_ROW, :, 1, k_show],
    color="tab:red",
    linewidth=2.2,
    label="Layer 2, after stacking",
    zorder=4,
)
# Shade the columns the clamp actually touched, so the eye goes straight to
# them rather than hunting for where the two red lines part company.
ax5.fill_between(
    x_slice,
    *ax5.get_ylim(),
    where=row_crossing[:, k_show],
    color="0.85",
    zorder=0,
    label="Clamped here",
)
ax5.set_xlabel("x (m)")
ax5.set_ylabel("elevation (m)")
ax5.set_title(
    f"Row {SLICE_ROW}, realization {k_show + 1}: the lower boundary was drawn "
    f"above the upper one at {n_cross_row} of {CROP} cells,\nand now runs "
    f"along it there instead"
)
ax5.legend(loc="upper right")
plt.tight_layout()

# %% -------------------------------------------------
# Figure 6: four realizations as filled cross-sections
# -------------------------------------------------
# The same profile row, but drawn the way a cross-section normally is: the
# two boundaries bound three units, and the units are filled rather than the
# boundaries merely drawn. Four realizations stacked vertically, sharing
# both axes, so the eye compares like with like -- the question this figure
# answers is "how much does the geometry change between draws?", and only a
# common scale lets you answer it.
#
# These are the *corrected* surfaces from step 6, so no unit ever has
# negative thickness. Where the middle unit vanishes, the two boundaries are
# touching: a pinch-out, which is a legitimate geological outcome, and is
# exactly what the clamp produces instead of an inversion.
#
# Three fills is three *identities*, not a magnitude, so this is a
# categorical assignment: hues taken in fixed slot order from the validated
# default palette rather than picked by eye. Fills carry alpha so the
# boundaries stay readable on top of them, and each boundary is drawn as a
# white line under a dark one, which gives the pair of touching fills a
# visible gap instead of letting two saturated areas abut directly.
N_SECTIONS = 4
UNIT_ABOVE, UNIT_MIDDLE, UNIT_BELOW = "#2a78d6", "#eb6834", "#1baf7a"
FILL_ALPHA = 0.75

sec_top = model.surfaces[SLICE_ROW, :, 0, :N_SECTIONS]
sec_bot = model.surfaces[SLICE_ROW, :, 1, :N_SECTIONS]
pad = 0.08 * (sec_top.max() - sec_bot.min())
y_hi, y_lo = sec_top.max() + pad, sec_bot.min() - pad

fig6, axes6 = plt.subplots(N_SECTIONS, 1, figsize=(12, 11), sharex=True, sharey=True)
for k, ax in enumerate(axes6):
    top, bot = sec_top[:, k], sec_bot[:, k]

    ax.fill_between(
        x_slice, top, y_hi, color=UNIT_ABOVE, alpha=FILL_ALPHA, label="Above layer 1"
    )
    ax.fill_between(
        x_slice,
        bot,
        top,
        color=UNIT_MIDDLE,
        alpha=FILL_ALPHA,
        label="Between the layers",
    )
    ax.fill_between(
        x_slice, y_lo, bot, color=UNIT_BELOW, alpha=FILL_ALPHA, label="Below layer 2"
    )

    for boundary in (top, bot):
        ax.plot(x_slice, boundary, color="white", linewidth=3.2, zorder=3)
        ax.plot(x_slice, boundary, color="0.15", linewidth=1.6, zorder=4)

    # How often this draw pinched out on this row -- the visible consequence
    # of the correction, stated so it is not left to be eyeballed.
    pinched = int(row_crossing[:, k].sum())
    ax.set_ylabel("elevation (m)")
    ax.text(
        0.012,
        0.9,
        f"Realization {k + 1}"
        + (f" -- {pinched} cells pinched out" if pinched else ""),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(facecolor="white", edgecolor="0.7", boxstyle="round,pad=0.3"),
    )

axes6[0].set_ylim(y_lo, y_hi)
# No margin: matplotlib's default leaves a white gutter at each end, which on
# a filled section reads as the units stopping short of the frame.
axes6[0].set_xlim(x_slice[0], x_slice[-1])
axes6[0].legend(loc="lower left", ncol=3, fontsize=8, framealpha=0.95)
axes6[-1].set_xlabel("x (m)")
fig6.suptitle(
    f"Row {SLICE_ROW}: four realizations of the layered model, after the "
    "monotonicity correction"
)
fig6.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()
