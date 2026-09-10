# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 16:53:40 2026

@author: rbm
"""

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Laplace transfer function (ftot_laplace)
# -------------------------------------------------
def ftot_laplace(l, a, b, c, d, e, g):
    """
    Transfer function for Laplacian-based probability enhancement.

    Parameters
    ----------
    l : ndarray
        Absolute Laplacian values
    a, b, c, d, e, g : floats
        Transfer function parameters (same meaning as MATLAB version)

    Returns
    -------
    ftot : ndarray
        Extra probability contribution
    """

    # Ensure gradient is sufficiently steep
    mindist = min(a, b - a)

    if g < 3 / mindist:
        g = 3 / mindist
        print(f"Gradient changed to {g:.3f} to ensure curve continuity")

    # First branch (tanh transition)
    f1 = c / 2 * (np.tanh((l - a) * g) + 1)

    # Second branch (Lorentz-type decay)
    f2 = (c - d) * (1 / (((l - b) / e) ** 2 + 1)) + d

    ftot = np.where(l < b, f1, f2)

    return ftot


# -------------------------------------------------
# 2D Laplacian (MATLAB del2 equivalent)
# -------------------------------------------------
def laplacian_2d(surface, dx=1.0, dy=1.0):
    """
    Compute discrete Laplacian (similar to MATLAB del2 but explicit).

    Parameters
    ----------
    surface : 2D ndarray
    dx, dy : float
        Grid spacing

    Returns
    -------
    lapl : 2D ndarray
    """

    d2x = (
        np.roll(surface, -1, axis=1) - 2 * surface + np.roll(surface, 1, axis=1)
    ) / dx**2
    d2y = (
        np.roll(surface, -1, axis=0) - 2 * surface + np.roll(surface, 1, axis=0)
    ) / dy**2

    lapl = d2x + d2y

    return lapl


# -------------------------------------------------
# Main INPOX function
# -------------------------------------------------
def draw_points_inpox(surface, ext_vals, dx=1.0, dy=1.0, wells=None):
    """
    INPOX informed point extraction.

    Parameters
    ----------
    surface : 2D ndarray
    ext_vals : dict
        Must contain:
        p0, a, b, c, d, e, g
    dx, dy : float
        Grid spacing
    wells : dict or None
        Optional well enforcement.
        Must contain:
        wells["xutm"], wells["yutm"]
        and grid origin in ext_vals:
        ext_vals["x0"], ext_vals["y0"]

    Returns
    -------
    points : boolean 2D ndarray
    plap : probability map
    lapl : absolute Laplacian
    """

    n_cells = surface.size
    maxp = n_cells * ext_vals["p0"]

    # --- Laplacian ---
    lapl = np.abs(laplacian_2d(surface, dx, dy))

    # --- Transfer function ---
    plap_extra = ftot_laplace(
        lapl,
        ext_vals["a"],
        ext_vals["b"],
        ext_vals["c"],
        ext_vals["d"],
        ext_vals["e"],
        ext_vals["g"],
    )

    # --- Base probability ---
    p1 = ext_vals["p0"] * np.ones_like(surface)

    plap = p1 + plap_extra

    # --- Normalize to maintain expected total count ---
    expp = plap.sum()
    factor = expp / maxp
    plap = plap / factor

    # --- Optional well enforcement ---
    if wells is not None:

        x0 = ext_vals["x0"]
        y0 = ext_vals["y0"]

        for xw, yw in zip(wells["xutm"], wells["yutm"]):
            indx = int(np.ceil((xw - x0) / dx))
            indy = int(np.ceil((yw - y0) / dy))

            if 0 <= indy < plap.shape[0] and 0 <= indx < plap.shape[1]:
                plap[indy, indx] += 1

    # --- Draw random points ---
    points = np.random.rand(*surface.shape) < plap

    return points, plap, lapl, plap_extra


def visualize_transfer_function(surface, ext_vals, dx=1.0, dy=1.0, lap_max=10):
    """
    Visualize INPOX transfer function against Laplacian distribution.

    Parameters
    ----------
    surface : 2D ndarray
        Input surface
    ext_vals : dict
        Transfer function parameters (must contain a, b, c, d, e, g, p0)
    dx, dy : float
        Grid spacing
    lap_max : float
        Maximum Laplacian value for plotting

    Returns
    -------
    lap_surface : ndarray
        Absolute Laplacian values (flattened)
    ftot_test : ndarray
        Transfer function values for test range
    """

    # -------------------------------------------------
    # 1. Compute Laplacian
    # -------------------------------------------------
    lap_surface = np.abs(laplacian_2d(surface, dx=dx, dy=dy))
    lap_surface = lap_surface.ravel()

    # Remove NaNs (important for real surfaces)
    lap_surface = lap_surface[~np.isnan(lap_surface)]

    # -------------------------------------------------
    # 2. Compute transfer function curve
    # -------------------------------------------------
    lap_test = np.linspace(0, lap_max, 200)

    ftot_test = ftot_laplace(
        lap_test,
        ext_vals["a"],
        ext_vals["b"],
        ext_vals["c"],
        ext_vals["d"],
        ext_vals["e"],
        ext_vals["g"],
    )

    # -------------------------------------------------
    # 3. Plot
    # -------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Histogram of Laplacian
    ax1.hist(
        lap_surface,
        bins=np.linspace(0, lap_max, 100),
        density=True,
        alpha=0.5,
        label="Surface Laplacians",
    )
    ax1.set_xlabel("Laplacian")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper left")

    # Transfer function
    ax2 = ax1.twinx()
    ax2.plot(
        lap_test,
        ftot_test + ext_vals["p0"],
        linewidth=2,
        label="ftot + p0",
    )
    ax2.set_ylabel("Probability")
    ax2.legend(loc="upper right")

    plt.title("Transfer function vs Laplacian distribution")
    plt.tight_layout()
    plt.show()
