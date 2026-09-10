# Translated from mGstat's precal_cov.m
# (https://github.com/AUProbGeo/mGstat), MIT licensed.
# Copyright (c) 2024 Thomas Mejer Hansen. See THIRD_PARTY_NOTICES.md.

import numpy as np
from .edist import edist
from .semivar_synth import semivar_synth
from .deformat_variogram import deformat_variogram


def precal_cov(pos1, pos2, V, options=None):
    """
    Compute covariance matrix based on variogram(s).

    Parameters
    ----------
    pos1, pos2 : ndarray
        Coordinate arrays of shape (n, ndim)
    V : str or list of dict
        Variogram definition string (MATLAB style) or list of variogram dicts.
    options : dict, optional
        Currently supports 'verbose': int

    Returns
    -------
    cov : ndarray
        Covariance matrix.
    d : ndarray
        Semivariance matrix.
    """
    if options is None:
        options = {}
    verbose = options.get("verbose", 0)

    if isinstance(V, str):
        V = deformat_variogram(V)
    if not isinstance(V, list):
        V = [V]

    n1 = pos1.shape[0]
    n2 = pos2.shape[0]
    semiv = np.zeros((n1, n2))

    for v in V:
        if verbose:
            print(f"Processing variogram: {v}")
        dd, _ = edist(pos1, pos2)
        semiv += semivar_synth(v, dd)

    gvar = sum(v["par1"] for v in V)
    cov = gvar - semiv
    return cov, semiv
