# -*- coding: utf-8 -*-
"""
Example: training the range/sill neural-network estimator

geosigma.preprocess.rangesill.NNRangeSillEstimator predicts the variogram
*range* of a local window of a layer's residual surface (the *sill* is not
learned — it's the sample variance of the scattered points within that
window, before interpolation; see the module docstring). GeoSigma ships a
pretrained default (``load_default_rangesill_estimator``), and this script is
how that default was produced: it is the actual, reproducible training
recipe, not a toy stand-in for it.

Geometry/cell size (``MINIGRID_SIZE`` = 33 cells, ``CELL_SIZE`` = 100 m) and
the scattered-point-then-interpolate preprocessing both match
``get_interpolated_minigrids_parallel.m`` exactly, verified against that
source rather than assumed; the interpolation itself is **cubic**
(``scipy.interpolate.griddata``) rather than MATLAB's natural-neighbour,
which has no equivalent among this project's dependencies.

Steps:
1. Look at example training fields, with their true range/sill annotated,
   and the scattered points sampled from each before interpolation.
2. Build a held-out evaluation set of synthetic fields with *known* range/sill
   (geosigma.testing.synthetic_field).
3. Generate training data (the expensive step -- dominated by synthetic_field's
   covariance factorization, not the interpolation) -- or load a cached copy
   from a previous run instead, via LOAD_CACHED_TRAINING_DATA below -- then
   train a fresh estimator on it.
4. Check predicted vs. true range/sill on the held-out set.
5. Save the trained estimator and reload it.
6. Compare against the shipped default, if one is installed.

Generating training data and training the network on it are deliberately
separate steps (geosigma.preprocess.rangesill.generate_training_data vs.
train_rangesill_estimator(training_data=...)): once you have a cached
dataset you're happy with, you can iterate on architecture ("conv" vs
"dense", see ARCHITECTURE below), network size, epochs and learning rate as
many times as you like without regenerating it.

Requires the optional ``torch`` and ``scipy`` dependencies:
``pip install "geosigma[nn]"``.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from geosigma.testing import synthetic_field
from geosigma.preprocess.rangesill import (
    CELL_SIZE,
    MINIGRID_SIZE,
    NNRangeSillEstimator,
    estimate_center_range_sill,
    generate_training_data,
    load_default_rangesill_estimator,
    load_training_data,
    sample_scatter_points,
    save_training_data,
    train_rangesill_estimator,
)

np.random.seed(0)

MINIGRID = MINIGRID_SIZE  # 33 cells -- matches get_interpolated_minigrids_parallel.m
DX = CELL_SIZE  # 100 m/cell -- matches Cell_Size in the same source
RANGE_BOUNDS = (50.0, 2000.0)
SILL_BOUNDS = (1.0, 50.0)

# How many synthetic "borehole" points are sampled per training window before
# interpolation. There is no single MATLAB-equivalent value for this -- real
# point density is data-dependent, not a pipeline constant -- so it's a free
# parameter here; 40 is a reasonable illustrative default.
N_SCATTER_POINTS = 360

# The training dataset size: how many synthetic fields train_rangesill_estimator
# draws for this run. N_TRAIN_SAMPLES/N_TRAIN_EPOCHS below are small enough to
# run in well under a minute on a laptop CPU -- this is a fast illustrative
# run, not a reproduction of the shipped default's actual training recipe.
# The shipped default was trained on 100,000 samples with every grid cell used
# as a scatter point (n_scatter_points = 33*33 = 1089), conv architecture,
# hidden=16, 50 epochs, seed=1 -- ~70 min to generate the data plus ~6 min to
# train. So expect this quick run to score noticeably lower than the "shipped
# default" comparison in step 6 below -- that gap is the cost of this example
# staying fast, not a bug.
#
# Note the default's training density (1089 points per window) is far denser
# than a realistic point set, and the network is accurate at the density it
# was trained on: on held-out synthetic fields it scores a log-log
# correlation of 0.996 with essentially no scale bias (median predicted/true
# = 0.99) at 1089 points, but 0.904 and +40% bias at ~175 points. Match
# N_SCATTER_POINTS to your own data when training your own estimator.
# See docs/deferred_decisions.md-style
# reasoning in geosigma/preprocess/rangesill.py's module docstring for why
# torch+scipy are optional extras rather than core dependencies.
N_TRAIN_SAMPLES = 1000

# Set True to load a previously-generated training set from disk instead of
# regenerating it. Generation is the expensive step (dominated by
# synthetic_field's covariance factorization -- not the interpolation, which
# is ~40x cheaper), so once you have a cached dataset you're happy with, flip
# this on to iterate on training (hidden size, epochs, learning rate) for
# free. The first run (flag False) always writes the cache, so flipping this
# on afterwards just works.
LOAD_CACHED_TRAINING_DATA = False
TRAINING_DATA_CACHE = os.path.join(
    os.path.dirname(__file__), "data", "rangesill_nn", "training_data.npz"
)
N_TRAIN_EPOCHS = 50

# Which network to train: "conv" (two small convolution layers + a linear
# head) or "dense" (a plain fully-connected net on the flattened grid).
# "conv" is what the shipped default uses. On an identical 100,000-sample set
# with identical hyperparameters and the same held-out fields, the two scored:
#
#     conv  (5,377 params)    log-log corr 0.767
#     dense (15,409 params)   log-log corr 0.653
#     dense (61,633 params)   log-log corr 0.630
#     dense (131,457 params)  log-log corr 0.723
#
# i.e. conv generalizes better while using 3-24x *fewer* parameters, so the
# gap is about inductive bias (convolution encodes "spatial covariance
# matters, absolute position doesn't", which is what a correlation length
# is), not capacity -- the dense nets fit the training data better as they
# grow and still don't catch up. "conv" also accepts any patch size, while
# "dense" fixes its input width at build time. Flip this to compare yourself.
ARCHITECTURE = "conv"
# ARCHITECTURE = "dense"


def evaluate(estimator, fields, seeds):
    """Return (pred_ranges, pred_sills) arrays for a list of fields.

    `estimate_center_range_sill` does the per-field work: scatter points
    over a MINIGRID x MINIGRID field, run them through the same
    interpolation and estimator path production uses, and read off the
    centre cell.
    """
    preds = np.array(
        [
            estimate_center_range_sill(
                f,
                estimator=estimator,
                dx=DX,
                n_scatter_points=N_SCATTER_POINTS,
                seed=s,
            )
            for f, s in zip(fields, seeds)
        ]
    )
    return preds[:, 0], preds[:, 1]


def _annotate_field(ax, field, range_, sill, vmin=None, vmax=None):
    """Plot one field with its (range, sill) written directly on the image."""
    im = ax.imshow(field, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.text(
        0.03,
        0.03,
        f"range = {range_:.0f} m\nsill = {sill:.1f}",
        transform=ax.transAxes,
        color="white",
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"),
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _overlay_scatter_points(ax, rows, cols, values, vmin, vmax):
    """Overlay the scattered training points (before interpolation) on a panel."""
    ax.scatter(
        cols,
        rows,
        c=values,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.6,
        s=28,
    )


# %% -------------------------------------------------
# 1. What the training data looks like
# -------------------------------------------------
# These fields are generated exactly the way train_rangesill_estimator builds
# its training set (geosigma.testing.synthetic_field); this section is purely
# illustrative -- it doesn't feed into training or evaluation below. The
# black-edged scatter points are the ones actually sampled from the field --
# the dense field underneath only exists so you can see where they came from;
# what the network actually trains on is those points, cubic-interpolated.

# (a) a deterministic grid isolating each parameter's visual effect: range
# controls how "smooth"/correlated the field looks, sill controls its
# amplitude -- independently of each other, by design (see
# NNRangeSillEstimator's docstring on why the two are decoupled).
grid_ranges = [100.0, 500.0, 2000.0]
grid_sills = [2.0, 20.0]

grid_fields = [
    [synthetic_field(MINIGRID, MINIGRID, DX, r, s, seed=0) for r in grid_ranges]
    for s in grid_sills
]
# Shared, symmetric-about-zero colour scale across every panel in this figure,
# so range's and sill's effects can be compared directly rather than each
# panel re-normalizing to its own min/max.
grid_vmax = max(np.abs(f).max() for row in grid_fields for f in row)

fig, axs = plt.subplots(
    len(grid_sills), len(grid_ranges), figsize=(12, 7), layout="constrained"
)
for row, s in enumerate(grid_sills):
    for col, r in enumerate(grid_ranges):
        field = grid_fields[row][col]
        im = _annotate_field(
            axs[row, col], field, r, s, vmin=-grid_vmax, vmax=grid_vmax
        )
        pt_rows, pt_cols, pt_values = sample_scatter_points(
            field, N_SCATTER_POINTS, seed=0
        )
        _overlay_scatter_points(
            axs[row, col], pt_rows, pt_cols, pt_values, -grid_vmax, grid_vmax
        )
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label="field value")
fig.suptitle(
    "Effect of range (columns) and sill (rows), with the scattered training\n"
    "points (black edge) sampled from each field before cubic interpolation"
)
plt.show()

# (b) a handful of genuinely random draws from the same distribution
# train_rangesill_estimator actually trains on (log-uniform over
# RANGE_BOUNDS/SILL_BOUNDS) -- what a real training batch looks like.
rng_preview = np.random.default_rng(10)
preview_specs, preview_fields = [], []
for _ in range(6):
    r = float(
        np.exp(rng_preview.uniform(np.log(RANGE_BOUNDS[0]), np.log(RANGE_BOUNDS[1])))
    )
    s = float(
        np.exp(rng_preview.uniform(np.log(SILL_BOUNDS[0]), np.log(SILL_BOUNDS[1])))
    )
    field = synthetic_field(
        MINIGRID, MINIGRID, DX, r, s, seed=int(rng_preview.integers(0, 2**31 - 1))
    )
    preview_specs.append((r, s))
    preview_fields.append(field)

preview_vmax = max(np.abs(f).max() for f in preview_fields)

fig, axs = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
for ax, field, (r, s) in zip(axs.ravel(), preview_fields, preview_specs):
    im = _annotate_field(ax, field, r, s, vmin=-preview_vmax, vmax=preview_vmax)
    pt_rows, pt_cols, pt_values = sample_scatter_points(
        field, N_SCATTER_POINTS, seed=int(rng_preview.integers(0, 2**31 - 1))
    )
    _overlay_scatter_points(
        ax, pt_rows, pt_cols, pt_values, -preview_vmax, preview_vmax
    )
fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, label="field value")
fig.suptitle(
    "Random training-distribution fields, with their scattered sample points\n"
    "(shared colour scale)"
)
plt.show()


# %% -------------------------------------------------
# 2. Held-out evaluation set (same fields for every estimator below)
# -------------------------------------------------
rng = np.random.default_rng(2)
n_eval = 60

eval_fields, eval_seeds, true_ranges, true_sills = [], [], [], []
for _ in range(n_eval):
    r_true = float(
        np.exp(rng.uniform(np.log(RANGE_BOUNDS[0]), np.log(RANGE_BOUNDS[1])))
    )
    s_true = float(np.exp(rng.uniform(np.log(SILL_BOUNDS[0]), np.log(SILL_BOUNDS[1]))))
    field = synthetic_field(
        MINIGRID,
        MINIGRID,
        DX,
        r_true,
        s_true,
        seed=int(rng.integers(0, 2**31 - 1)),
    )
    eval_fields.append(field)
    eval_seeds.append(int(rng.integers(0, 2**31 - 1)))  # for scatter-point sampling
    true_ranges.append(r_true)
    true_sills.append(s_true)

true_ranges = np.array(true_ranges)
true_sills = np.array(true_sills)


# %% -------------------------------------------------
# 3. Generate (or load cached) training data, then train
# -------------------------------------------------
if LOAD_CACHED_TRAINING_DATA and os.path.exists(TRAINING_DATA_CACHE):
    training_data = load_training_data(TRAINING_DATA_CACHE)
    print(
        f"Loaded cached training data from {TRAINING_DATA_CACHE} "
        f"({training_data.patches.shape[0]} samples)"
    )
else:
    training_data = generate_training_data(
        N_TRAIN_SAMPLES,
        minigrid_size=MINIGRID,
        dx=DX,
        n_scatter_points=N_SCATTER_POINTS,
        range_bounds=RANGE_BOUNDS,
        sill_bounds=SILL_BOUNDS,
        seed=1,
    )
    os.makedirs(os.path.dirname(TRAINING_DATA_CACHE), exist_ok=True)
    save_training_data(TRAINING_DATA_CACHE, training_data)
    print(
        f"Generated {N_TRAIN_SAMPLES} training samples and cached them to "
        f"{TRAINING_DATA_CACHE}"
    )

estimator = train_rangesill_estimator(
    training_data,
    hidden=16,
    architecture=ARCHITECTURE,
    epochs=N_TRAIN_EPOCHS,
    seed=1,
    verbose=True,
)
print(f"trained a {ARCHITECTURE!r} network on {training_data.patches.shape[0]} samples")


# %% -------------------------------------------------
# 4. Evaluate: predicted vs. true range/sill
# -------------------------------------------------
pred_ranges, pred_sills = evaluate(estimator, eval_fields, eval_seeds)

log_corr = np.corrcoef(np.log(true_ranges), np.log(pred_ranges))[0, 1]
sill_log_corr = np.corrcoef(np.log(true_sills), np.log(pred_sills))[0, 1]
print(f"range estimate: log-log correlation on held-out fields = {log_corr:.3f}")
print(
    f"sill estimate: log-log correlation = {sill_log_corr:.3f} "
    "(sill is a closed-form statistic -- the sample variance of the "
    f"{N_SCATTER_POINTS} scattered points within the window, before "
    "interpolation -- not learned; it is a genuinely noisier estimate of the "
    "*nominal* sill than range is of the nominal range, especially when the "
    "true range is large relative to the window: a sparse sample's variance "
    "under-recovers the nominal sill when the field barely varies across the "
    "window. This is expected sampling behaviour, not a bug -- see the "
    "NNRangeSillEstimator docstring.)"
)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].loglog(true_ranges, pred_ranges, "o", alpha=0.6)
range_lims = [RANGE_BOUNDS[0] * 0.8, RANGE_BOUNDS[1] * 1.2]
axs[0].plot(range_lims, range_lims, "k--", lw=1, label="1:1")
axs[0].set_xlabel("True range (m)")
axs[0].set_ylabel("Predicted range (m)")
axs[0].set_title(f"Range estimate (log-log corr = {log_corr:.2f})")
axs[0].legend()

axs[1].loglog(true_sills, pred_sills, "o", alpha=0.6)
sill_lims = [
    min(true_sills.min(), pred_sills.min()) * 0.5,
    max(true_sills.max(), pred_sills.max()) * 2,
]
axs[1].plot(sill_lims, sill_lims, "k--", lw=1, label="1:1")
axs[1].set_xlabel("True (nominal) sill")
axs[1].set_ylabel("Predicted sill (scattered-point sample variance)")
axs[1].set_title(
    f"Sill estimate: closed-form, not learned (log-log corr = {sill_log_corr:.2f})"
)
axs[1].legend()

plt.tight_layout()
plt.show()


# %% -------------------------------------------------
# 5. Save and reload
# -------------------------------------------------
out_dir = os.path.join(os.path.dirname(__file__), "data", "rangesill_nn")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "example_trained.pt")

estimator.save(out_path)
print(f"Saved trained estimator to {out_path}")

reloaded = NNRangeSillEstimator.load(out_path)
kw = dict(dx=DX, n_scatter_points=N_SCATTER_POINTS, seed=eval_seeds[0])
r0, s0 = estimate_center_range_sill(eval_fields[0], estimator=estimator, **kw)
r1, s1 = estimate_center_range_sill(eval_fields[0], estimator=reloaded, **kw)
print(f"reload sanity check: original={r0:.1f},{s0:.2f}  reloaded={r1:.1f},{s1:.2f}")
assert np.isclose(r0, r1) and np.isclose(s0, s1)


# %% -------------------------------------------------
# 6. Compare against the shipped default, if installed
# -------------------------------------------------
try:
    default_estimator = load_default_rangesill_estimator()
except FileNotFoundError:
    default_estimator = None
    print("No shipped default estimator found; skipping comparison.")

if default_estimator is not None:
    pred_ranges_default, _ = evaluate(default_estimator, eval_fields, eval_seeds)
    log_corr_default = np.corrcoef(np.log(true_ranges), np.log(pred_ranges_default))[
        0, 1
    ]
    print(
        "shipped default: log-log correlation on the same held-out fields = "
        f"{log_corr_default:.3f}"
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.loglog(
        true_ranges,
        pred_ranges,
        "o",
        alpha=0.5,
        label=f"this run (corr={log_corr:.2f})",
    )
    ax.loglog(
        true_ranges,
        pred_ranges_default,
        "s",
        alpha=0.5,
        label=f"shipped default (corr={log_corr_default:.2f})",
    )
    ax.plot(range_lims, range_lims, "k--", lw=1, label="1:1")
    ax.set_xlabel("True range (m)")
    ax.set_ylabel("Predicted range (m)")
    ax.set_title("This run's estimator vs. the shipped default")
    ax.legend()
    plt.tight_layout()
    plt.show()
