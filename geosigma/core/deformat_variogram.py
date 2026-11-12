# Translated from mGstat's deformat_variogram.m
# (https://github.com/AUProbGeo/mGstat), MIT licensed.
# Copyright (c) 2024 Thomas Mejer Hansen. See THIRD_PARTY_NOTICES.md.

import math
import re

_COMPONENT = re.compile(r"([\d\.eE\+\-]+)\s+([A-Za-z]+)\(([^\)]+)\)")


def deformat_variogram(txt):
    """
    Parse MATLAB-style variogram string into Python dict.

    Parameters
    ----------
    txt : str
        Variogram string: one or more ``<sill> <Model>(<range>)`` components
        joined by ``+``, e.g. ``'1 Sph(5)'`` or ``'0.1 Nug(0) + 2 Gau(300)'``.

    Returns
    -------
    V_list : list of dict
        List of variogram dictionaries with keys:
        - 'type': model type
        - 'par1': sill
        - 'par2': range

    Raises
    ------
    ValueError
        If the string does not parse completely, or any sill or range is not
        finite.

        This is deliberately strict. The callers build these strings with
        f-strings from computed values, so a NaN sill arrives as
        ``'nan Gau(300)'`` -- which the component pattern cannot match. An
        earlier version returned an empty list for it, and
        :func:`~geosigma.core.precal_cov.precal_cov` then summed zero
        components into an **all-zero covariance matrix**: zero prior
        variance, silently, rather than an error. A NaN range was no better,
        parsing fine and producing an all-NaN covariance.
    """
    parts = _COMPONENT.findall(txt)
    # Everything outside the matched components must be separators only, so
    # one unparseable component in a sum cannot be dropped while its
    # neighbours survive.
    leftover = _COMPONENT.sub("", txt).replace("+", "").strip()
    if not parts or leftover:
        raise ValueError(
            f"could not parse variogram {txt!r}: expected one or more "
            "'<sill> <Model>(<range>)' components joined by '+'"
            + (f"; unparsed remainder {leftover!r}" if leftover else "")
        )

    V_list = []
    for par1, vtype, args in parts:
        args = [float(a) for a in args.split(",")]
        par2 = args[0]
        if len(args) > 1:
            print(
                "Warning: additional variogram parameters (anisotropy/angle) "
                "are ignored. Only isotropic is supported."
            )
        sill = float(par1)
        if not (math.isfinite(sill) and math.isfinite(par2)):
            raise ValueError(
                f"variogram {txt!r} has a non-finite sill or range "
                f"({sill}, {par2}) in its {vtype} component"
            )
        V_list.append({"type": vtype, "par1": sill, "par2": par2})
    return V_list
