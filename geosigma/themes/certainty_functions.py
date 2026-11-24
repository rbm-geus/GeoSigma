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
stack at once.

``kernel_with_point_tail`` is GeoSigma's recommended kernel and the one the
Danish theme specs use. ``FRAFA_apr2023`` and ``ILM_oct2023`` are ported
verbatim from the MATLAB ``certainty_function_definer.m``. Two further MATLAB
kernels are **legacy**: ``ILM_sep2023`` (a sign error) and ``RBM_oct2023`` (a
jump at the kernel edge). They are kept only so the MATLAB grids can be
reproduced, and :func:`certainty_function` refuses them unless
``allow_legacy=True`` (on a spec: ``allow_legacy_cert_fun=True``). See
``docs/deviations_from_matlab.md``.
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
    """**Legacy.** Gaussian decay from the plateau edge inside ``width``; zero
    outside.

    The MATLAB formula has a sign error: inside the plateau ``dist - width`` is
    zero or negative, so the decay runs from the plateau edge inwards. The
    certainty is lowest at the data point (``sill * exp(-3 width^2 / range^2)``)
    and rises to ``sill`` at the edge, where it drops to zero. ``ILM_oct2023``
    is the MATLAB correction. Kept only to reproduce MATLAB runs; requires
    ``allow_legacy=True``.

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
    """**Legacy.** Two-piece Gaussian with a jump at the plateau edge.

    Inside ``width``: ``sill * exp(-1.5 d^2 / range^2)``. Outside:
    ``sill_adj * exp(-3 d^2 / range^2)`` with
    ``sill_adj = sill + (sill - inside(width))``. The adjustment adds the
    missing certainty as if the tail started at full height at the edge, but
    the tail is centred on the data point and is already damped by
    ``exp(-3 width^2 / range^2)`` there. With
    ``a = exp(-1.5 width^2 / range^2)`` the value drops from ``a * sill`` to
    ``(2 - a) a^2 * sill`` across the edge, a jump of ``a (1 - a)^2 * sill``
    (at most 4/27 of the sill, at ``width / range`` = 0.86). Ported verbatim
    from MATLAB and kept only to reproduce MATLAB runs; requires
    ``allow_legacy=True``. Use ``kernel_with_point_tail``, which is identical
    inside ``width`` and continuous at the edge.
    """

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


def _kernel_with_point_tail(dist, range_, width, sill):
    """Sensitivity kernel with a Gaussian tail centred on the data point.

    GeoSigma's recommended certainty function. A sounding is 1D information at
    its x, y location, like a borehole. Inside the sensitivity kernel
    (``dist <= width``) the certainty decays gently from the data point.
    Outside it, the certainty follows a Gaussian decay that is also centred on
    the data point and starts from the kernel's value at the edge::

        dist <= width:  sill * exp(-1.5 dist^2 / range^2)
        dist >  width:  sill * exp(-1.5 width^2 / range^2)
                             * exp(-3 (dist^2 - width^2) / range^2)

    The value is ``sill`` at the data point, decreases everywhere and is
    continuous at the edge. The slope changes there: at ``dist = width`` the
    tail decays twice as fast as the kernel. Inside ``width`` the function is
    identical to the legacy ``RBM_oct2023``; outside, it removes that kernel's
    jump.
    """
    inside = sill * np.exp(-1.5 * dist**2 / range_**2)
    edge = sill * np.exp(-1.5 * width**2 / range_**2)
    outside = edge * np.exp(-3.0 * (dist**2 - width**2) / range_**2)
    return np.where(dist <= width, inside, outside)


#: Registry of named certainty kernels.
_CERT_FUNCTIONS = {
    "kernel_with_point_tail": _kernel_with_point_tail,
    "FRAFA_apr2023": _frafa_apr2023,
    "ILM_sep2023": _ilm_sep2023,
    "ILM_oct2023": _ilm_oct2023,
    "RBM_oct2023": _rbm_oct2023,
}

#: Legacy MATLAB kernels (name -> why it is legacy). Kept only to reproduce the
#: MATLAB grids; :func:`certainty_function` refuses them unless asked.
LEGACY_CERTAINTY_FUNCTIONS = {
    "ILM_sep2023": (
        "it has a sign error, so the certainty rises with distance inside the "
        "plateau"
    ),
    "RBM_oct2023": "its value jumps at the plateau edge",
}

_LEGACY_HINT = (
    "Use 'kernel_with_point_tail' (recommended) or 'ILM_oct2023'. To reproduce "
    "a MATLAB run, pass allow_legacy=True (on a ThemeSpec: "
    "allow_legacy_cert_fun=True; in YAML: allow_legacy_cert_fun: true). See "
    "docs/deviations_from_matlab.md."
)


def check_legacy_certainty_function(name: str, allow_legacy: bool = False) -> None:
    """Raise ``ValueError`` if ``name`` is a legacy kernel and ``allow_legacy``
    is False."""
    if name in LEGACY_CERTAINTY_FUNCTIONS and not allow_legacy:
        raise ValueError(
            f"certainty function {name!r} is a legacy MATLAB kernel: "
            f"{LEGACY_CERTAINTY_FUNCTIONS[name]}. {_LEGACY_HINT}"
        )


def certainty_function(name: str = "FRAFA_apr2023", *, allow_legacy: bool = False):
    """Return the named certainty kernel ``cert_fun(dist, range_, width, sill)``.

    Parameters
    ----------
    name : str
        One of :data:`available_certainty_functions`.
    allow_legacy : bool, default False
        Allow a kernel listed in :data:`LEGACY_CERTAINTY_FUNCTIONS`. Only for
        reproducing MATLAB runs.

    Raises
    ------
    ValueError
        If ``name`` is not a registered kernel, or is a legacy kernel and
        ``allow_legacy`` is False.
    """
    try:
        fn = _CERT_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown certainty function choice: {name!r}. "
            f"Available: {sorted(_CERT_FUNCTIONS)}"
        )
    check_legacy_certainty_function(name, allow_legacy)
    return fn


def available_certainty_functions():
    """Return the sorted names of all registered certainty kernels, the legacy
    ones included (see :data:`LEGACY_CERTAINTY_FUNCTIONS`)."""
    return sorted(_CERT_FUNCTIONS)


def register_certainty_function(
    name: str, fn: callable, overwrite: bool = False
) -> None:
    """Register a custom certainty kernel under ``name`` so it can be referenced
    by string from a :class:`~geosigma.themes.base.ThemeSpec`.

    This is the supported extension point for users whose decay assumptions differ
    from the built-in presets (a different data type or geological
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
