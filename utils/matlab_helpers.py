# geosigma/matlab_helpers.py
import numpy as np
import matplotlib.pyplot as plt


def matlab_meshgrid(x, y):
    """
    Return xx, yy grids in MATLAB-style column-major order.

    Equivalent to MATLAB's meshgrid(x, y) with indexing='ij'.

    Parameters
    ----------
    x : array-like
        Vector of x coordinates
    y : array-like
        Vector of y coordinates

    Returns
    -------
    xx, yy : 2D arrays
        Grid arrays in MATLAB column-major style
    """
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return xx, yy


def coords_from_meshgrid(xx, yy):
    """
    Flatten meshgrid arrays into Nx2 coordinates array, like MATLAB [xx(:), yy(:)]

    Parameters
    ----------
    xx, yy : 2D arrays
        Grid arrays from matlab_meshgrid

    Returns
    -------
    coords : Nx2 array
        Coordinates suitable for precal_cov and other functions
    """
    return np.column_stack((xx.ravel(), yy.ravel()))


def plot_covariance(cov, title="Covariance matrix"):
    """
    Plot a covariance matrix similar to MATLAB's imagesc

    Parameters
    ----------
    cov : 2D array
        Covariance matrix
    title : str
        Plot title
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cov, origin="lower", cmap="viridis")
    plt.title(title)
    plt.colorbar(label="Covariance")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.show()
