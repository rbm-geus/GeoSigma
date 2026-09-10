# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 10:26:15 2025

@author: rbm
"""


def add_in_between_gridlines(
    ax, x, y, color="white", lw=0.5, ls="--", alpha=1.0, axisbelow=False
):
    """
    Add gridlines in between existing x and y tick positions on an Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The target axis to add lines to.
    x, y : array-like
        The x and y tick positions (edges of the grid cells).
    color : str, optional
        Line color (default 'white').
    lw : float, optional
        Line width (default 0.5).
    ls : str, optional
        Line style (default '--').
    alpha : float, optional
        Line transparency (default 1.0).
    axisbelow : bool, optional
        Whether to draw the grid below or above the image (default False).
    """
    ax.set_axisbelow(axisbelow)

    # Compute midpoints
    x_mids = (x[:-1] + x[1:]) / 2
    y_mids = (y[:-1] + y[1:]) / 2

    # Draw vertical lines
    for xm in x_mids:
        ax.axvline(x=xm, color=color, lw=lw, ls=ls, alpha=alpha)

    # Draw horizontal lines
    for ym in y_mids:
        ax.axhline(y=ym, color=color, lw=lw, ls=ls, alpha=alpha)
