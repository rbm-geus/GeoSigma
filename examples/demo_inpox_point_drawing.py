# -*- coding: utf-8 -*-
"""
Example: how INPOX decides where to put conditioning points

INPOX (``geosigma.inpox.draw_points_inpox``, translated from ``drawpoints.m``)
does not sample uniformly. It spends points where the surface is *rough*, on
the reasoning that a smooth area is well described by few observations while a
structurally complex one is not. Each cell's draw probability is

    p(cell) = p0  +  roughness_term(|del2(surface)|)  [ + a floor, + wells ]

and a point is drawn wherever a uniform random number falls below it.

This script takes the surface the kriging demo uses and shows the four things
worth seeing, for a grid of ``p0`` and threshold choices:

1. the surface itself,
2. its Laplacian -- the roughness measure INPOX actually reacts to,
3. the resulting per-cell draw probability, and
4. the points that come out.

Pass ``return_details=True`` to get (2) and (3) alongside (1); without it the
function just hands back the point mask.

The transfer function is Madsen et al. (2024), *Introducing INPOX*, GEUS
Bulletin 57, 8364, eqs 2-4. It rises over rough ground to ``(1+C)*P0``, then
past ``B*L0`` tapers back down towards ``(1-D)*P0`` -- zero, with the paper's
``D = 1``. That descent matters: Laplacians large enough to be interpolation
artefacts rather than geology get *suppressed*, and gradually, so a cell just
above ``B`` is not treated wildly differently from one just below.

Two parameters are varied:

``P0``
    The baseline probability -- a *floor*, not the draw fraction. A flat
    surface draws ``P0`` of its cells; a structured one draws more, because
    roughness only ever adds. The realised fraction is ``mean(f(L))``.
``C``
    How strongly roughness is favoured: the curve peaks at ``(1+C)*P0``, so
    a rough cell is ``(1+C)`` times likelier to be drawn than a flat one.
    Raising ``C`` therefore draws *more* points as well as concentrating
    them -- both effects at once, which is why the printout below reports
    the realised count rather than assuming it.

Pass ``normalize=True`` to rescale the map to mean ``P0`` instead, pinning
the expected count to a uniform draw at ``P0``. That is the setting for
benchmarking INPOX against random sampling, where the comparison only
means anything at equal point count; it keeps the gain in *placement* and
cancels the gain in count. It is not what ``drawpoints.m`` does, so it is
off by default.

Everything else is held at the paper's Table 1 values for its real-world
case (T4), which was tuned on a FOHM surface -- the same model this cutout
comes from.

Note the Laplacian is computed at **unit spacing**, in surface units per
cell^2 -- not per metre^2 -- because ``drawpoints.m`` calls ``del2(surface)``
with no spacing argument and the ``a``/``b`` constants are tuned to that
scale. Panel 2's colour bar is on that scale, and ``a`` is directly
comparable to it.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import load_geotiff_grid
from geosigma.inpox import draw_points_inpox, ftot_laplace

SEED = 1

# The same cutout the kriging demo uses.
z, meta = load_geotiff_grid("examples/data/Jutland_hydrostrat_model/top_surface.tif")
dx = float(meta["cellsize"])

# P0 is the floor: a flat surface draws P0 of its cells, a structured one
# draws more. Roughness only ever adds.
P0_VALUES = [0.02, 0.05, 0.20]

# C*P0 is the most probability roughness can add, so the curve peaks at
# (1+C)*P0: a rough cell is (1+C) times likelier to be drawn than a flat one.
# Raising C both concentrates the points and increases how many there are.
C_VALUES = [1.0, 5.0, 15.0]

# The rest of Table 1's real-world case (T4) from Madsen et al. (2024), held
# fixed. Those values were tuned on a FOHM surface -- the same model this
# cutout comes from -- so they are a sensible starting point here.
BASE = {"L0": 1.0, "A": 1.5, "B": 3.0, "D": 1.0, "E": 0.8, "g": 2.0}


def inpox(P0, C, seed=SEED):
    """Draw with this (P0, C), returning the point mask and the intermediates."""
    np.random.seed(seed)  # same random numbers each time, so only P0/C differ
    return draw_points_inpox(
        z,
        {"P0": P0, "C": C, **BASE},
        dx=dx,
        dy=dx,
        return_details=True,
    )


# %% -------------------------------------------------
# Figure 1: the inputs -- these do NOT depend on p0 or the threshold
# -------------------------------------------------
_, details = inpox(P0_VALUES[0], C_VALUES[0])
laplacian = details["laplacian"]

fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.6))

im = axes1[0].imshow(z, cmap="viridis")
axes1[0].set_title("1. The surface")
fig1.colorbar(im, ax=axes1[0], shrink=0.85, label="elevation (m)")

im = axes1[1].imshow(laplacian, cmap="magma")
axes1[1].set_title("2. |Laplacian| -- the roughness INPOX reacts to")
fig1.colorbar(im, ax=axes1[1], shrink=0.85, label="m / cell$^2$")

# The transfer function f(L) itself (Madsen et al. 2024, eqs 2-4), drawn over
# the real distribution of roughness so you can see how many cells each part
# of the curve applies to. It rises to (1+C)*P0 over rough ground, then -- past
# B*L0 -- tapers back down towards (1-D)*P0, which is zero for D = 1. The
# descent is gradual, not a threshold: that is the whole point, since a hard
# cut would treat a cell just above B very differently from one just below.
# Shown for the middle P0; a different P0 scales the curve.
P0_SHOWN = P0_VALUES[len(P0_VALUES) // 2]
lap_axis = np.linspace(0, float(laplacian.max()), 600)
axes1[2].hist(
    laplacian.ravel(),
    bins=60,
    density=True,
    color="0.8",
    label="cells' |Laplacian|",
)
ax_tf = axes1[2].twinx()
for C in C_VALUES:
    ax_tf.plot(
        lap_axis,
        ftot_laplace(lap_axis, P0_SHOWN, C=C, **BASE),
        linewidth=1.8,
        label=f"C = {C:g}  (peak {(1 + C) * P0_SHOWN:g})",
    )
ax_tf.axhline(P0_SHOWN, color="0.4", linestyle="--", linewidth=0.9)
for level, text in [
    (P0_SHOWN, f"$P_0$ = {P0_SHOWN}"),
    ((1 - BASE["D"]) * P0_SHOWN, f"$(1-D)P_0$ = {(1 - BASE['D']) * P0_SHOWN:g}"),
]:
    ax_tf.annotate(
        text,
        xy=(lap_axis[-1], level),
        xytext=(-4, 4),
        textcoords="offset points",
        fontsize=8,
        ha="right",
        color="0.3",
    )
ax_tf.set_ylim(bottom=0)
axes1[2].set_xlabel("|Laplacian| (m / cell$^2$)")
axes1[2].set_ylabel("density of cells")
ax_tf.set_ylabel("draw probability f(L)")
axes1[2].set_title(f"3. Transfer function f(L), $P_0$ = {P0_SHOWN}, B = {BASE['B']:g}")
ax_tf.legend(loc="upper left", fontsize=9)

for ax in axes1[:2]:
    ax.set_xticks([])
    ax.set_yticks([])
fig1.suptitle(
    "INPOX inputs: the surface, its roughness, and how roughness becomes probability"
)
plt.tight_layout()


# %% -------------------------------------------------
# Figures 2 and 3: probability and the resulting points, over (P0, C)
# -------------------------------------------------
probabilities, point_masks = {}, {}
for P0 in P0_VALUES:
    for C in C_VALUES:
        pts, det = inpox(P0, C)
        probabilities[(P0, C)] = det["probability"]
        point_masks[(P0, C)] = pts
        p = det["probability"]
        print(
            f"P0={P0:.2f}  C={C:<4g} ->  "
            f"probability {p.min():.3f}-{p.max():.3f}, "
            f"drew {int(pts.sum()):5d} points ({pts.mean():.3f} of cells)"
        )

# A shared colour scale so panels are comparable to each other.
prob_max = max(p.max() for p in probabilities.values())


def grid_figure(values, title, cmap, vmin, vmax, label, as_points=False):
    fig, axes = plt.subplots(
        len(P0_VALUES),
        len(C_VALUES),
        figsize=(4.2 * len(C_VALUES), 4.0 * len(P0_VALUES)),
    )
    for i, P0 in enumerate(P0_VALUES):
        for j, C in enumerate(C_VALUES):
            ax = axes[i, j]
            if as_points:
                rows, cols = np.nonzero(values[(P0, C)])
                ax.imshow(z, cmap="Greys_r", alpha=0.35)
                ax.scatter(cols, rows, s=1.5, color="crimson")
                ax.set_title(
                    f"$P_0$={P0:.2f}, C={C:g}\n{len(rows)} points "
                    f"({len(rows) / z.size:.3f} of cells)",
                    fontsize=10,
                )
            else:
                im = ax.imshow(values[(P0, C)], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"$P_0$={P0:.2f}, C={C:g}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    if not as_points:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=label)
    fig.suptitle(title, fontsize=13)
    return fig


grid_figure(
    probabilities,
    "3. Draw probability per cell (rows: $P_0$, columns: C)",
    "viridis",
    0.0,
    prob_max,
    "probability",
)
grid_figure(
    point_masks,
    "4. The points actually drawn (rows: $P_0$, columns: C)",
    None,
    None,
    None,
    None,
    as_points=True,
)

plt.show()
