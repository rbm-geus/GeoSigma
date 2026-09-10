# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Named certainty-decay kernels for the variance-map themes.

A *certainty function* maps the distance from a grid cell to the nearest
contributing data point into a certainty value (the reciprocal of variance).
All theme builders share these kernels; new named variants belong here and
nowhere else (see ``CLAUDE.md`` — *What to Avoid*).

Each kernel has the signature ``cert_fun(dist, range_, width, sill)``:

* ``dist``   distance from cell to nearest data point.
* ``range_`` effective correlation range (controls the Gaussian decay).
* ``width``  plateau half-width: inside it the kernel does not decay.
* ``sill``   certainty at the plateau (``1 / var0``).

All arguments broadcast as numpy arrays, so a kernel evaluates a whole grid
stack at once. The formulae are ported verbatim from the MATLAB
``certainty_function_definer.m`` / the root ``certainty_function_definer.py``;
the statistical model is preserved exactly.
"""

from __future__ import annotations

import inspect

import numpy as np

#: The exact positional parameter names every kernel must declare, in order.
_CERT_FUN_SIGNATURE = ("dist", "range_", "width", "sill")


def _frafa_apr2023(dist, range_, width, sill):
    """Flat plateau of height ``sill`` inside ``width``, Gaussian decay outside."""
    return np.where(
        dist <= width,
        sill,
        sill * np.exp(-3.0 * (dist - width) ** 2 / range_**2),
    )


def _ilm_sep2023(dist, range_, width, sill):
    """Gaussian decay from the plateau edge inside ``width``; zero outside.

    Transcribed as the MATLAB *multiplicative* form
    ``(dist<=width).*sill.*exp(...)`` (``certainty_function_definer.m``), **not**
    ``np.where(dist<=width, ..., 0)``. The two agree at every finite distance, but
    differ where ``dist`` (no contributing point) or ``range`` (unknown
    complexity) is NaN: the multiplicative form yields ``0 * NaN = NaN``, which
    :func:`~geosigma.themes.base.build_theme` maps to the ``NODATA_VARIANCE``
    sentinel — exactly as MATLAB's ``Grid(isnan(Grid)) = 100000`` does. The
    ``np.where`` form would instead substitute a literal ``0`` and leak
    ``1/0 = +inf`` at no-data cells. (A cell with a real point *beyond* ``width``
    still gives ``0 * exp(finite) = 0 -> +inf``, matching MATLAB.)
    """
    return (dist <= width) * sill * np.exp(-3.0 * (dist - width) ** 2 / range_**2)


def _ilm_oct2023(dist, range_, width, sill):
    """Gaussian decay measured from the cell (not the plateau edge); zero outside.

    Multiplicative MATLAB form, for the same NaN-propagation reason as
    :func:`_ilm_sep2023`.
    """
    return (dist <= width) * sill * np.exp(-3.0 * dist**2 / range_**2)


def _rbm_oct2023(dist, range_, width, sill):
    """Continuous two-piece Gaussian: damped decay inside, matched decay outside."""

    def _inside(d, r, w, s, damp):
        return s * np.exp(-damp * d**2 / r**2)

    def _outside(d, r, w, s):
        return s * np.exp(-3.0 * d**2 / r**2)

    sill_adjusted = sill + (sill - _inside(width, range_, width, sill, 1.5))
    return np.where(
        dist <= width,
        _inside(dist, range_, width, sill, 1.5),
        _outside(dist, range_, width, sill_adjusted),
    )


#: Registry of named certainty kernels.
_CERT_FUNCTIONS = {
    "FRAFA_apr2023": _frafa_apr2023,
    "ILM_sep2023": _ilm_sep2023,
    "ILM_oct2023": _ilm_oct2023,
    "RBM_oct2023": _rbm_oct2023,
}


def certainty_function(name: str = "FRAFA_apr2023"):
    """Return the named certainty kernel ``cert_fun(dist, range_, width, sill)``.

    Parameters
    ----------
    name : str
        One of :data:`available_certainty_functions`. Identified by a
        ``name_MONYYYY`` key, matching the MATLAB convention.

    Raises
    ------
    ValueError
        If ``name`` is not a registered kernel.
    """
    try:
        return _CERT_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown certainty function choice: {name!r}. "
            f"Available: {sorted(_CERT_FUNCTIONS)}"
        )


def available_certainty_functions():
    """Return the sorted names of all registered certainty kernels."""
    return sorted(_CERT_FUNCTIONS)


def register_certainty_function(
    name: str, fn: callable, overwrite: bool = False
) -> None:
    """Register a custom certainty kernel under ``name`` so it can be referenced
    by string from a :class:`~geosigma.themes.base.ThemeSpec`.

    This is the supported extension point for users whose decay assumptions differ
    from the four built-in Danish presets (a different data type or geological
    setting). Register the kernel once at start-up, then set
    ``cert_fun_name=name`` on the spec exactly as for a built-in.

    Why a name and not a callable on ``ThemeSpec``?
    -----------------------------------------------
    ``ThemeSpec`` references its kernel by **name string** (``cert_fun_name``),
    not by holding the function object. That indirection is deliberate and must
    stay: a spec is serialisable to / from YAML (see
    :mod:`geosigma.themes.spec_io`), and a bare Python callable cannot round-trip
    through YAML. Keeping kernels in this name->callable registry means a spec
    stores only a portable name; both built-in and user-registered kernels are
    resolved the same way at :func:`~geosigma.themes.base.build_theme` time. So
    custom kernels are added *here* (to the registry), never as a callable field
    on the spec.

    Parameters
    ----------
    name : str
        Registry key to expose the kernel under (e.g. ``"MYORG_jan2026"``).
    fn : callable
        The kernel, with the exact signature ``fn(dist, range_, width, sill)``
        (see this module's docstring for what each argument means). All arguments
        must broadcast as numpy arrays.
    overwrite : bool, optional
        By default (``False``) re-registering an existing name — including any of
        the built-in presets — raises ``ValueError`` to prevent silent shadowing.
        Pass ``True`` to replace an existing entry on purpose.

    Raises
    ------
    TypeError
        If ``fn`` is not callable, or its parameters are not exactly
        ``(dist, range_, width, sill)``.
    ValueError
        If ``name`` is already registered and ``overwrite`` is ``False``.
    """
    if not callable(fn):
        raise TypeError(
            f"certainty function {name!r} must be callable, got {type(fn).__name__}"
        )

    params = tuple(inspect.signature(fn).parameters)
    if params != _CERT_FUN_SIGNATURE:
        raise TypeError(
            f"certainty function {name!r} must have signature "
            f"{_CERT_FUN_SIGNATURE}, got {params}"
        )

    if not overwrite and name in _CERT_FUNCTIONS:
        raise ValueError(
            f"certainty function {name!r} is already registered; pass "
            f"overwrite=True to replace it deliberately"
        )

    _CERT_FUNCTIONS[name] = fn
