import numpy as np


def get_reals_cholesky(cov, Nreals=1, m0=None, verbose=False):
    """
    Generate realizations from a covariance matrix using Cholesky decomposition.
    Translated from ``get_reals_cholesky.m`` in the N-ret hydrostratigraphic
    MATLAB codebase (see ``matlab_reference/``); the ``m0`` and ``verbose``
    arguments are additions in this implementation.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix (n x n).
    Nreals : int
        Number of realizations to generate.
    m0 : None, scalar, or np.ndarray
        Mean vector. If None, defaults to zero vector.
        If scalar, broadcast to vector of length n.
        If array, must have length n.

    Returns
    -------
    reals : np.ndarray
        Array of shape (n, Nreals) with realizations.
    """
    nvar = cov.shape[0]

    # Handle mean
    if m0 is None:
        m0 = np.zeros(nvar)
    elif np.isscalar(m0):
        m0 = np.ones(nvar) * m0
    else:
        m0 = np.asarray(m0)
        if m0.shape[0] != nvar:
            raise ValueError(
                f"Length of m0 ({m0.shape[0]}) does not match covariance size ({nvar})."
            )

    # Cholesky decomposition
    UT = np.linalg.cholesky(cov + 1e-5 * np.eye(nvar))

    # Generate realizations
    reals = UT @ np.random.randn(nvar, Nreals)

    # Add mean
    reals += m0[:, None]

    if verbose:
        print(
            f"get_reals_cholesky: generated {Nreals} realization(s) with {nvar} variables."
        )

    return reals
