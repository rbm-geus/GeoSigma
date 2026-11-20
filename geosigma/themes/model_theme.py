# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Model theme: a depth-dependent variance floor per modelling-area class.

Port of ``get_modeltheme.m``. Independently of the geophysical data, the
modelling area is classified into quality classes; each class sets a minimum
variance at the surface that grows with depth via a two-segment linear ramp
with a break at depth ``b``, then squared into a variance:

    f_class(x) = ( 0.5 * ramp(x; f_min[class]) )^2

    ramp(x; fmin) = g_s * x + fmin                        if x <  b
                    g_d * x + (g_s - g_d) * b + fmin      if x >= b

where ``x = terrain - layer_bottom`` (layer depth below terrain), ``g_s`` is the
gradient shallower than the break and ``g_d`` the gradient deeper than it. The
ramp is continuous at ``b``. The formula is preserved verbatim from MATLAB, where
the break is fixed at 50 m and the gradients are named ``grad_less50`` (``g_s``)
and ``grad_abv50`` (``g_d``); here all three are arguments. The class grid (the
MATLAB ``grid_ma_class``) is loaded by an adapter and passed in.

The peat branch (``i <= NPL`` forcing the deepest class) is omitted with the
rest of the peatland subsystem.
"""

from __future__ import annotations

import numpy as np

#: Surface minimum-variance roots per class, **calibrated to the Danish national
#: model** (MATLAB Jylland f0..f5). Not a default: pass it explicitly as
#: ``class_fmin=DANISH_CLASS_FMIN`` only if you mean to use the Danish values.
DANISH_CLASS_FMIN = {0: 20.0, 1: 20.0, 2: 12.5, 3: 12.5, 4: 11.25, 5: 10.0}

#: Depth ramp **calibrated to the Danish national model** (MATLAB break at 50 m,
#: ``grad_less50 = 0.3``, ``grad_abv50 = 0.15``). Not a default: pass it
#: explicitly as ``depth_ramp=DANISH_DEPTH_RAMP`` only if you mean to use it.
DANISH_DEPTH_RAMP = {
    "break_depth": 50.0,
    "shallow_gradient": 0.3,
    "deep_gradient": 0.15,
}

_DEPTH_RAMP_KEYS = ("break_depth", "shallow_gradient", "deep_gradient")


def _check_depth_ramp(depth_ramp):
    """Return ``(break_depth, shallow_gradient, deep_gradient)`` as floats."""
    hint = "pass depth_ramp=DANISH_DEPTH_RAMP to use the Danish calibration"
    if depth_ramp is None or set(depth_ramp) != set(_DEPTH_RAMP_KEYS):
        raise ValueError(
            f"depth_ramp must be a mapping with exactly the keys "
            f"{list(_DEPTH_RAMP_KEYS)}; {hint}"
        )
    values = tuple(float(depth_ramp[k]) for k in _DEPTH_RAMP_KEYS)
    if not all(np.isfinite(values)) or values[0] < 0:
        raise ValueError(
            f"depth_ramp values must be finite and break_depth >= 0; got {values}"
        )
    return values


def _ramp_variance(depth, fmin, break_depth, shallow_gradient, deep_gradient):
    """The two-segment ramp squared into a variance (verbatim from MATLAB)."""
    shallow = shallow_gradient * depth + fmin
    deep = (
        deep_gradient * depth + (shallow_gradient - deep_gradient) * break_depth + fmin
    )
    ramp = np.where(depth < break_depth, shallow, deep)
    return (0.5 * ramp) ** 2


def model_theme(
    class_grid,
    terrain,
    layer_bottoms,
    *,
    class_fmin,
    depth_ramp,
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
    class_fmin : mapping of int -> float
        **Required, keyword-only.** Surface minimum-variance root per class: your
        model's background uncertainty at the surface, per quality class. There
        is no default. :data:`DANISH_CLASS_FMIN` holds the values calibrated to
        the Danish national model; pass it explicitly only if you mean to use
        them. Classes absent from the mapping keep variance ``0`` (matching the
        MATLAB ``modeltheme = modeltheme2*0`` initialisation).
    depth_ramp : mapping
        **Required, keyword-only.** How the surface value grows with depth, as a
        mapping with three keys:

        * ``break_depth`` — depth below terrain [m] where the gradient changes;
        * ``shallow_gradient`` — growth per metre of depth shallower than the
          break;
        * ``deep_gradient`` — growth per metre of depth at and below the break.

        There is no default. :data:`DANISH_DEPTH_RAMP` holds the values
        calibrated to the Danish national model (50 m, 0.3, 0.15); pass it
        explicitly only if you mean to use them.

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
    if class_fmin is None or len(class_fmin) == 0:
        raise ValueError(
            "class_fmin must map each model-quality class to its surface value; "
            "pass class_fmin=DANISH_CLASS_FMIN to use the Danish calibration"
        )
    break_depth, shallow_gradient, deep_gradient = _check_depth_ramp(depth_ramp)

    theme = np.zeros((*class_grid.shape, n_layers))
    for s in range(n_layers):
        depth = terrain - bottoms[:, :, s]
        layer = np.zeros(class_grid.shape)
        for cls, fmin in class_fmin.items():
            sel = class_grid == cls
            if sel.any():
                layer[sel] = _ramp_variance(
                    depth[sel], fmin, break_depth, shallow_gradient, deep_gradient
                )
        theme[:, :, s] = layer
    return theme
