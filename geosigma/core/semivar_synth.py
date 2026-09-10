# Translated from mGstat's semivar_synth.m
# (https://github.com/AUProbGeo/mGstat), MIT licensed.
# Copyright (c) 2024 Thomas Mejer Hansen. See THIRD_PARTY_NOTICES.md.

import numpy as np


def semivar_synth(V, d):
    """
    Compute semivariance for a given variogram model.

    Parameters
    ----------
    V : dict
        Variogram structure with keys:
        - 'type': model type ('Sph', 'Gau', 'Exp')
        - 'par1': sill
        - 'par2': range
    d : ndarray
        Distance matrix.

    Returns
    -------
    gamma : ndarray
        Semivariance matrix.


    """
    model = V["type"].lower()
    var = V["par1"]
    range_ = V["par2"]
    h = np.asarray(d)
    gamma = np.zeros_like(h)

    if model.startswith("sph"):
        mask = h < range_
        gamma[mask] = var * (1.5 * (h[mask] / range_) - 0.5 * (h[mask] / range_) ** 3)
        gamma[~mask] = var
    elif model.startswith("exp"):
        gamma = var * (1 - np.exp(-3 * h / range_))
    elif model.startswith("gau"):
        gamma = var * (1 - np.exp(-((h / range_) ** 2)))
    else:
        raise ValueError(f"Unknown variogram model '{V['type']}'")

    return gamma
