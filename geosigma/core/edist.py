# Translated from mGstat's edist.m
# (https://github.com/AUProbGeo/mGstat), MIT licensed.
# Copyright (c) 2024 Thomas Mejer Hansen. See THIRD_PARTY_NOTICES.md.

import numpy as np


def edist(p1, p2=None, transform=None, isorange=False):
    """
    Compute Euclidean distance between two sets of points.
    Currently only isotropic variograms are supported.

    Parameters
    ----------
    p1 : array_like, shape (n1, ndim)
    p2 : array_like, shape (n2, ndim), optional
    transform : None
        Not implemented, anisotropy not supported.
    isorange : bool, optional
        Ignored for now.

    Returns
    -------
    D : ndarray, shape (n1, n2)
        Euclidean distance matrix.
    dp : ndarray, shape (n1, n2, ndim)
        Coordinate differences.

    """
    if transform is not None:
        raise NotImplementedError(
            "Anisotropy is not implemented. Only isotropic variograms are supported."
        )

    p1 = np.atleast_2d(p1)
    if p2 is None:
        p2 = np.zeros_like(p1)
    else:
        p2 = np.atleast_2d(p2)

    dp = p2[None, :, :] - p1[:, None, :]
    ndim = p1.shape[1]

    if ndim == 1:
        D = np.abs(dp[:, :, 0])
    else:
        D = np.sqrt(np.sum(dp**2, axis=2))

    return D, dp
