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
def ftot_laplace(abs_laplacian, P0, L0, A, B, C, D, E, g):
    """
    INPOX transfer function f(L): Laplacian -> draw probability.

    Implements Madsen et al. (2024), *Introducing INPOX*, GEUS Bulletin 57,
    8364, equations 2-4, using the paper's own symbols so the parameter
    tables there can be used directly::

        f1(L) = (C*P0 / 2) * (tanh((L - A*L0) * g) + 1)                (eq 2)

        f2(L) = (C + D)*P0 / ((L - B*L0)**2 / (E*L0) + 1)  -  D*P0     (eq 3)

        f(L)  = f1(L) + P0   for L <  B*L0                             (eq 4)
                f2(L) + P0   for L >= B*L0

    ``f1`` raises the probability over rough ground; ``f2`` brings it back
    down again, so Laplacians large enough to be interpolation artefacts
    rather than geology are *suppressed* rather than favoured. The handover
    at ``B*L0`` is continuous: both branches meet at ``(1+C)*P0``.

    The descent is a **taper, not a cliff**. ``f2`` decays smoothly as a
    Lorentzian of width set by ``E*L0``, approaching ``(1-D)*P0`` -- so with
    the paper's ``D = 1`` the probability tends to exactly zero, reached
    gradually rather than at a hard threshold.

    Parameters
    ----------
    abs_laplacian : ndarray
        Absolute Laplacian values, ``L``.
    P0 : float
        Baseline probability, and the intended draw fraction: the map is
        normalized so the mean probability is ``P0`` (see
        :func:`draw_points_inpox`), so ``P0 = 0.05`` draws about 5%.
    L0 : float
        Reference Laplacian. ``A``, ``B`` and ``E`` are all expressed as
        multiples of it. The paper uses ``L0 = 1`` throughout.
    A : float
        ``A*L0`` is where ``f1`` rises most steeply.
    B : float
        ``B*L0`` is the Laplacian at which ``f1`` hands over to ``f2``, i.e.
        where enhancement turns into suppression.
    C : float
        ``C*P0`` is the maximum probability added, peaking at ``(1+C)*P0``.
    D : float
        ``D*P0`` is the maximum probability subtracted; the curve tends to
        ``(1-D)*P0``. ``D = 1`` (the paper's choice everywhere) tapers to 0.
    E : float
        ``E*L0`` sets how gradually ``f2`` falls away -- larger is slower.
    g : float
        Maximum derivative of ``f1``, i.e. how abruptly it switches on.

    Returns
    -------
    ftot : ndarray
        Draw probability f(L), including the ``P0`` baseline.
    """
    L = np.asarray(abs_laplacian, dtype=float)

    # eq 2 -- enhancement over rough ground
    f1 = (C * P0 / 2.0) * (np.tanh((L - A * L0) * g) + 1.0)

    # eq 3 -- suppression of the very roughest, tapering towards (1-D)*P0
    f2 = (C + D) * P0 / ((L - B * L0) ** 2 / (E * L0) + 1.0) - D * P0

    # eq 4
    return np.where(L < B * L0, f1, f2) + P0


# -------------------------------------------------
# 2D Laplacian (MATLAB del2 equivalent)
# -------------------------------------------------
def laplacian_2d(surface, dx=1.0, dy=1.0):
    """
    Compute discrete Laplacian, matching MATLAB's ``del2``.

    ``del2`` is the *quarter*-Laplacian ``(Uxx + Uyy) / 4``, not the raw
    5-point stencil sum, and at the grid boundary it does not wrap around
    (unlike a naive ``np.roll`` implementation): it uses a one-sided
    second difference, which for a uniform grid equals the second
    difference one cell in (the second derivative of the quadratic
    interpolant through the first/last three points is constant, so the
    edge value equals its nearest interior neighbour's).

    Parameters
    ----------
    surface : 2D ndarray
    dx, dy : float
        Grid spacing

    Returns
    -------
    lapl : 2D ndarray
    """
    d2x = np.empty_like(surface, dtype=float)
    d2x[:, 1:-1] = surface[:, :-2] - 2 * surface[:, 1:-1] + surface[:, 2:]
    d2x[:, 0] = d2x[:, 1]
    d2x[:, -1] = d2x[:, -2]
    d2x /= dx**2

    d2y = np.empty_like(surface, dtype=float)
    d2y[1:-1, :] = surface[:-2, :] - 2 * surface[1:-1, :] + surface[2:, :]
    d2y[0, :] = d2y[1, :]
    d2y[-1, :] = d2y[-2, :]
    d2y /= dy**2

    return (d2x + d2y) / 4


# -------------------------------------------------
# Main INPOX function
# -------------------------------------------------
def draw_points_inpox(
    surface,
    ext_vals,
    dx=1.0,
    dy=1.0,
    wells=None,
    return_details=False,
    normalize=False,
    origin="upper",
):
    """
    INPOX informed point extraction.

    Parameters
    ----------
    surface : 2D ndarray
        The boundary surface to draw points from. Its row order matters only
        when ``wells`` are given; see ``origin``.
    ext_vals : dict
        Transfer-function parameters, named as in Madsen et al. (2024) --
        must contain ``P0``, ``L0``, ``A``, ``B``, ``C``, ``D``, ``E``,
        ``g``; see :func:`ftot_laplace`. With ``wells``, also ``x0``/``y0``.

        This project's examples use the paper's Table 1 real-world case,
        **T4**: ``L0=1, A=1.5, B=3, C=5, D=1, E=0.8, g=2``.

        ``drawpoints.m`` hardcodes a different set inside its lambdas,
        which in these symbols reads ``L0=1, A=2.4, B=12, C=1.25, D=0.25,
        E=0.09, g=0.8``. T4 is used here in preference, deliberately: it is
        the published, documented parameterisation, whereas the MATLAB
        constants are undocumented literals. The difference is not
        cosmetic. MATLAB's ``B*L0 = 12`` puts the suppression threshold far
        above the roughness of a typical FOHM cutout (median
        ``|del2| ~ 0.33``, max ~5), so its ``f2`` branch never activates at
        all, and its weaker ``C`` lifts the draw fraction only ~9% above
        ``P0`` where T4 lifts it ~45%. Pass the MATLAB values explicitly if
        you need to reproduce reference output.
    dx, dy : float
        Cell size, used **only** to convert well coordinates to grid indices
        (see ``wells``). The Laplacian is deliberately computed at unit
        spacing regardless -- see the note at its call site.
    wells : dict or None
        Optional well enforcement: each well's cell gets ``+1`` probability,
        i.e. is always drawn. Must contain ``wells["xutm"]`` and
        ``wells["yutm"]`` (map coordinates), and ``ext_vals`` must then also
        hold ``"x0"``/``"y0"``: the grid's **lower-left corner**, exactly as
        raster metadata gives it (``xllcorner``/``yllcorner``) -- not the
        centre of the lower-left cell.

        A well is assigned to the cell that *contains* it,
        ``floor((x - x0) / dx)`` from the corner. That is the cell
        ``drawpoints.m`` picks too: it rounds from the lower-left cell
        *centre* (``LLx = min(Grid_X)``), which selects the same containing
        cell. An earlier version rounded from the corner instead, which put
        a well at a cell centre exactly on ``k + 0.5`` and let numpy's
        round-half-to-even split it between ``k`` and ``k + 1``. Wells
        outside the grid are ignored.

        Translated from ``drawpoints.m``'s regular (non-peat) well boost
        only. That source also boosts peatland wells (``PL_wells``) with
        an additional ``+0.15*increase_prob`` term -- omitted here, like
        peat-specific logic elsewhere in this project (see
        ``matlab_reference/PEAT_AUDIT.md``), not an oversight.

    return_details : bool, default False
        When ``True``, also return the intermediate grids the draw was built
        from, for inspection or plotting.
    normalize : bool, default False
        Whether to rescale the probability map so its mean becomes ``P0``.

        ``False`` (the default, and what ``drawpoints.m`` does) makes ``P0``
        a **floor** that roughness adds to, so the realised draw fraction
        comes out *above* ``P0`` -- by ``mean(f(L))/P0``, which is the whole
        point of informed extraction: a structurally complex surface earns
        more points than a flat one, not merely better-placed ones.

        ``True`` rescales to mean ``P0``, pinning the expected count to what
        uniform sampling at ``P0`` would give. That is the right setting for
        *benchmarking* INPOX against a random draw, where the comparison is
        only meaningful at equal point count, but it cancels the gain in
        count and keeps only the gain in placement.
    origin : {"upper", "lower"}, default "upper"
        Which edge of the map row 0 of ``surface`` lies on, for placing
        ``wells``.

        ``"upper"``: row 0 is the **north** edge -- the layout of every
        raster loader in this repository (``load_geotiff_grid``,
        ``load_esri_ascii_grid``) and of ``imshow``'s default. ``"lower"``:
        row 0 is the south edge, the layout ``drawpoints.m`` assumes (row
        index grows from ``LLy = min(Grid_Y)``) and the one
        :mod:`geosigma.themes` produces from ascending axes.

        Getting this wrong does not fail: it mirrors every well vertically.
        An earlier version had no such parameter and always counted rows
        from the south, so on the north-up rasters the examples load, a well
        five rows from the top was boosted five rows from the bottom.

    Returns
    -------
    points : ndarray of bool, shape (ny, nx)
        The drawn point mask. Returned on its own unless ``return_details``.
    details : dict, only when ``return_details=True``
        ``"laplacian"`` -- ``|del2(surface)|``, the roughness measure, in
        index units (see the note at its call site).
        ``"transfer"`` -- ``f(L)`` straight out of :func:`ftot_laplace`,
        before any normalization or floors.
        ``"probability"`` -- the per-cell probability actually compared
        against a uniform random number: ``"transfer"``, optionally
        normalized, plus the every-10th-cell floor and any well boosts,
        clipped to 1.
    """

    if origin not in ("upper", "lower"):
        raise ValueError(f"origin must be 'upper' or 'lower', got {origin!r}")

    # --- Laplacian ---
    # Unit spacing, NOT dx/dy: drawpoints.m calls `del2(surface)` with no h
    # argument, so its Laplacian is in index units (surface units per cell^2),
    # and the A*L0 / B*L0 thresholds of the transfer function below are tuned
    # against that scale. Passing a real cell size here instead divides the
    # Laplacian by dx^2 -- at dx=100 m that is a factor of 10,000, which
    # pushes every cell below A*L0 and silently reduces INPOX to uniform
    # sampling at P0. dx/dy still matter below, for converting well
    # coordinates to indices, which is a genuinely different use (MATLAB
    # hardcodes 100 there).
    lapl = np.abs(laplacian_2d(surface))

    # --- Transfer function (Madsen et al. 2024, eqs 2-4) ---
    # f(L) already includes the P0 baseline.
    P0 = ext_vals["P0"]
    plap = ftot_laplace(
        lapl,
        P0,
        ext_vals["L0"],
        ext_vals["A"],
        ext_vals["B"],
        ext_vals["C"],
        ext_vals["D"],
        ext_vals["E"],
        ext_vals["g"],
    )
    transfer = plap.copy()  # before normalization / floors, for inspection

    # --- Optional normalization to mean P0 ---
    # Off by default: drawpoints.m does NOT do this, and rescaling here
    # would cancel exactly the effect INPOX exists to produce. Without it,
    # P0 is a floor and roughness adds to it, so the realised draw fraction
    # lands at mean(f(L)) -- above P0 by however much structure the surface
    # has. Normalizing pins the count to a uniform draw at P0 instead, which
    # is what you want when benchmarking against one, and nothing else.
    # Applied before the grid floor, which is a guarantee rather than a
    # probability and should not be diluted.
    if normalize:
        mean_p = plap.mean()
        if mean_p > 0:
            plap = plap * (P0 / mean_p)

    # Regular sparse grid of cells (every 10th row/col) always gets a
    # guaranteed baseline point, regardless of roughness. From drawpoints.m;
    # the paper has no such floor.
    grid_extra = np.zeros_like(surface, dtype=float)
    grid_extra[::10, ::10] = 1.0
    plap = plap + grid_extra

    # --- drawpoints.m's cap: at most one expected point per cell ---
    # `maxp = numel(surface)*1`, then rescale if the expected count exceeds
    # it. This binds only when the mean probability tops 1, which needs a
    # large P0 -- it is a safety valve, not the normalization above. Kept
    # here, at MATLAB's position in the sequence: after the grid floor,
    # before the well boosts.
    mean_p = plap.mean()
    if mean_p > 1.0:
        plap = plap / mean_p

    # --- Optional well enforcement ---
    if wells is not None:
        x0 = ext_vals["x0"]
        y0 = ext_vals["y0"]
        ny, nx = plap.shape

        for xw, yw in zip(wells["xutm"], wells["yutm"]):
            # The containing cell, counted from the lower-left corner. floor,
            # not round: x0/y0 are corners, so a cell-centre coordinate sits
            # exactly on k + 0.5 and rounding would split it between k and
            # k + 1 depending on k's parity.
            col = int(np.floor((xw - x0) / dx))
            row_from_south = int(np.floor((yw - y0) / dy))
            row = ny - 1 - row_from_south if origin == "upper" else row_from_south

            if 0 <= row < ny and 0 <= col < nx:
                plap[row, col] += 1

    # A probability cannot exceed 1. The grid floor and the well boosts both
    # add a flat +1, so plap routinely runs above it. Clipping here, after
    # the rescale and the boosts, leaves the draw itself unchanged -- the
    # comparison below is against rand() in [0, 1), so anything >= 1 was
    # already a certainty -- while making the returned map an honest
    # probability rather than something that reads as "165%".
    plap = np.minimum(plap, 1.0)

    # --- Draw random points ---
    points = np.random.rand(*surface.shape) < plap

    if return_details:
        return points, {
            "laplacian": lapl,
            "transfer": transfer,
            "probability": plap,
        }
    return points


def visualize_transfer_function(surface, ext_vals, lap_max=10):
    """
    Visualize INPOX transfer function against Laplacian distribution.

    Parameters
    ----------
    surface : 2D ndarray
        Input surface
    ext_vals : dict
        Transfer function parameters (must contain P0, L0, A, B, C, D, E, g)
    lap_max : float
        Maximum Laplacian value for plotting

    Returns
    -------
    lap_surface : ndarray
        Absolute Laplacian values (flattened)
    ftot_test : ndarray
        Draw probability for the test range of Laplacian values
    """

    # -------------------------------------------------
    # 1. Compute Laplacian
    # -------------------------------------------------
    # Unit spacing, matching draw_points_inpox -- the histogram has to be on
    # the same scale as the transfer-function curve it is plotted against.
    lap_surface = np.abs(laplacian_2d(surface))
    lap_surface = lap_surface.ravel()

    # Remove NaNs (important for real surfaces)
    lap_surface = lap_surface[~np.isnan(lap_surface)]

    # -------------------------------------------------
    # 2. Compute transfer function curve
    # -------------------------------------------------
    lap_test = np.linspace(0, lap_max, 200)

    ftot_test = ftot_laplace(
        lap_test,
        ext_vals["P0"],
        ext_vals["L0"],
        ext_vals["A"],
        ext_vals["B"],
        ext_vals["C"],
        ext_vals["D"],
        ext_vals["E"],
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
    # Transfer function. ftot_laplace already includes the P0 baseline, so it
    # is the draw probability as it stands -- nothing to add here.
    ax2 = ax1.twinx()
    ax2.plot(
        lap_test,
        ftot_test,
        linewidth=2,
        label="f(L)",
    )
    ax2.axhline(ext_vals["P0"], color="0.4", linestyle="--", linewidth=0.9)
    ax2.set_ylabel("Probability")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")

    plt.title("Transfer function vs Laplacian distribution")
    plt.tight_layout()
    plt.show()

    return lap_surface, ftot_test
