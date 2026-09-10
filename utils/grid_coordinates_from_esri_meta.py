import numpy as np
from .matlab_helpers import matlab_meshgrid
from .grid_meta import normalize_grid_meta


def grid_coordinates_from_esri_meta(meta):
    """Derive cell-centre coordinate vectors/grids from grid metadata.

    Accepts metadata using either the canonical contract (``dx``/``dy``) or the
    deprecated ``cellsize`` alias. ``x``/``y`` are cell centres; ``xx``/``yy``
    are MATLAB-style (column-major) meshgrids.
    """
    meta = normalize_grid_meta(meta)
    nx = meta["ncols"]
    ny = meta["nrows"]
    dx = meta["dx"]
    dy = meta["dy"]

    x0 = meta["xllcorner"]
    y0 = meta["yllcorner"]

    x = x0 + dx * (np.arange(nx) + 0.5)
    y = y0 + dy * (np.arange(ny) + 0.5)

    xx, yy = matlab_meshgrid(x, y)

    return x, y, xx, yy
