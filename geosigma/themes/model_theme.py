# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Model theme: a depth-dependent variance floor per modelling-area class.

Port of ``get_modeltheme.m``. Independently of the geophysical data, the
modelling area is classified into quality classes; each class sets a minimum
variance at the surface that grows with depth via a two-segment linear ramp
(steeper above 50 m, gentler below), then squared into a variance:

    f_class(x) = ( 0.5 * ramp(x; f_min[class]) )^2

    ramp(x; fmin) = (grad_less50 * x + fmin)                          if x < 50
                    (grad_abv50 * x + (grad_less50 - grad_abv50)*50 + fmin)  if x >= 50

where ``x = terrain - layer_bottom`` (layer depth below terrain). The formula is
preserved verbatim; only the model-agnostic parts are exposed as arguments. The
class grid (the MATLAB ``grid_ma_class``) is loaded by an adapter and passed in.

The peat branch (``i <= NPL`` forcing the deepest class) is omitted with the
rest of the peatland subsystem.
"""

from __future__ import annotations

import numpy as np

#: Default surface minimum-variance roots per class (MATLAB Jylland f0..f5).
DEFAULT_CLASS_FMIN = {0: 20.0, 1: 20.0, 2: 12.5, 3: 12.5, 4: 11.25, 5: 10.0}

#: Default depth ramp gradients (MATLAB ``grad_less50`` / ``grad_abv50``).
DEFAULT_GRAD_LESS50 = 0.3
DEFAULT_GRAD_ABV50 = 0.15


def _ramp_variance(depth, fmin, grad_less50, grad_abv50):
    """The two-segment ramp squared into a variance (verbatim from MATLAB)."""
    below = grad_less50 * depth + fmin
    above = grad_abv50 * depth + (grad_less50 - grad_abv50) * 50.0 + fmin
    ramp = np.where(depth < 50.0, below, above)
    return (0.5 * ramp) ** 2


def model_theme(
    class_grid,
    terrain,
    layer_bottoms,
    class_fmin=None,
    grad_less50=DEFAULT_GRAD_LESS50,
    grad_abv50=DEFAULT_GRAD_ABV50,
):
    """Build the per-layer model-theme variance floor.

    Parameters
    ----------
    class_grid : ndarray, shape (ny, nx)
        Integer modelling-area class per cell (the MATLAB ``grid_ma_class``).
    terrain : ndarray, shape (ny, nx)
        Terrain / top-surface elevation.
    layer_bottoms : sequence of ndarray (ny, nx), or ndarray (ny, nx, n_layers)
        Layer bottom-surface elevations.
    class_fmin : mapping of int -> float, optional
        Surface minimum-variance root per class; defaults to
        :data:`DEFAULT_CLASS_FMIN`. Classes absent from the mapping keep variance
        ``0`` (matching the MATLAB ``modeltheme = modeltheme2*0`` initialisation).
    grad_less50, grad_abv50 : float
        Depth-ramp gradients above / below 50 m.

    Returns
    -------
    ndarray, shape (ny, nx, n_layers)
    """
    class_grid = np.asarray(class_grid)
    terrain = np.asarray(terrain, dtype=float)
    if isinstance(layer_bottoms, np.ndarray) and layer_bottoms.ndim == 3:
        bottoms = layer_bottoms.astype(float)
    else:
        bottoms = np.stack([np.asarray(b, dtype=float) for b in layer_bottoms], axis=-1)
    n_layers = bottoms.shape[2]
    if class_fmin is None:
        class_fmin = DEFAULT_CLASS_FMIN

    theme = np.zeros((*class_grid.shape, n_layers))
    for s in range(n_layers):
        depth = terrain - bottoms[:, :, s]
        layer = np.zeros(class_grid.shape)
        for cls, fmin in class_fmin.items():
            sel = class_grid == cls
            if sel.any():
                layer[sel] = _ramp_variance(depth[sel], fmin, grad_less50, grad_abv50)
        theme[:, :, s] = layer
    return theme
