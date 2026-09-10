# Translated from mGstat's least_squares_inversion.m
# (https://github.com/AUProbGeo/mGstat), MIT licensed.
# Copyright (c) 2024 Thomas Mejer Hansen. See THIRD_PARTY_NOTICES.md.

import numpy as np
import time


def least_squares_inversion(G, Cm, Cd, m0, d0, type=2, use_tqdm=True):
    """
    Linear least-squares inversion following Tarantola (2005), Eq. (16-17)

    Parameters
    ----------
    G : ndarray (n_data x n_model)
        Forward (sensitivity) matrix
    Cm : ndarray (n_model x n_model)
        Model covariance matrix
    Cd : ndarray (n_data x n_data)
        Data covariance matrix
    m0 : ndarray or float
        Prior mean model (or scalar)
    d0 : ndarray
        Observed data vector
    type : int, optional
        Algorithm variant (1 or 2). Default is 2.
    use_tqdm : bool, optional
        If True, shows a progress bar (like waitbar in MATLAB)

    Returns
    -------
    m_est : ndarray
        Posterior (estimated) model vector
    Cm_est : ndarray
        Posterior model covariance matrix
    """

    # t1 = time.time()

    # If m0 is scalar, make it a vector
    if np.isscalar(m0):
        m0 = np.ones(G.shape[1]) * m0

    # Progress bar for visual feedback (optional)
    # iterator = tqdm(total=1, desc="Least-squares inversion", disable=not use_tqdm)

    if type == 2:
        # Compute data covariance in data space
        S = Cd + G @ Cm @ G.T

        # "Fast IMM style"
        K = Cm @ G.T @ np.linalg.inv(S)
        m_est = m0 + K @ (d0 - G @ m0)
        Cm_est = Cm - K @ (G @ Cm)

    else:
        # Type 1 variant
        if G.ndim == 1 or G.shape[0] == 1:
            goodG = np.nonzero(G != 0)[0]
        else:
            goodG = np.nonzero(np.sum(G, axis=0) != 0)[0]

        S = Cd + G[:, goodG] @ Cm[np.ix_(goodG, goodG)] @ G[:, goodG].T
        T = np.linalg.inv(S)

        m_est = m0 + Cm[:, goodG] @ G[:, goodG].T @ T @ (d0 - G @ m0)
        PP = Cm @ G.T @ T @ G
        Cm_est = Cm - PP[:, goodG] @ Cm[goodG, :]

    #  iterator.update(1)
    #  iterator.close()

    # t2 = time.time()
    # print(f"Elapsed time: {t2 - t1:.2f} s")

    return m_est, Cm_est
