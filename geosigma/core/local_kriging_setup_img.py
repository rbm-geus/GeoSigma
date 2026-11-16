import numpy as np
from .precal_cov import precal_cov


def local_kriging_setup_img(
    GLOBAL,
    i_buf,
    ip_buf,
    curvariance,
    currange,
    cor_noise_rat,
    dx=100,
    dy=100,
    verbose=False,
):
    """
    Set up local kriging matrices for image-based inversion or estimation.
    Translated from ``local_kriging_setup_img.m`` in the N-ret hydrostratigraphic
    MATLAB codebase (see ``matlab_reference/``).

    Parameters
    ----------
    GLOBAL : mapping
        Flat per-cell arrays, keyed ``"xx_norm"``, ``"yy_norm"`` (grid-index
        coordinates), ``"img_unc"`` (per-cell **standard deviation**) and
        ``"img_dobs"`` (the residual to condition on). In the pipeline this is
        :meth:`geosigma.kriging.state.KrigingGrid.as_global_dict`.
    i_buf : ndarray of bool, shape (n_cells,)
        Boolean **mask**, not indices: the cells of this local window, i.e. the
        model parameters being estimated.
    ip_buf : ndarray of bool, shape (n_cells,)
        Boolean **mask** of the conditioning cells -- the cells carrying an
        observation. Must be a subset of ``i_buf``.
    curvariance : float
        Prior sill of the Gaussian model covariance ``Cm``.
    currange : float
        Practical range (metres) of the Gaussian model -- used for both
        ``Cm`` and the correlated part of ``Cd``.
    cor_noise_rat : float
        Fraction, in ``[0, 1]``, of the observation-noise variance that is
        spatially correlated (with the same range as ``Cm``); the remaining
        ``1 - cor_noise_rat`` is white. ``0`` gives uncorrelated noise.
    dx : float, optional
        Physical grid spacing in x-direction (in meters). Default is 100,
        the value MATLAB hardcodes as ``xx_norm * 100``.
    dy : float, optional
        Physical grid spacing in y-direction (in meters). Default is 100.
    verbose : bool, optional
        Print a one-line message once the matrices are built.

    Returns
    -------
    G : ndarray
        Design matrix mapping model parameters to data points.
    Cm : ndarray
        Model covariance matrix.
    Cd : ndarray
        Data covariance matrix (includes correlated and uncorrelated components).
    d_obs : ndarray
        Observation vector corresponding to ip_buf.
    m0 : ndarray
        Prior mean model (currently zero vector).
    """

    # Active points only
    i_buf_local = np.where(i_buf)[0]  # indices of active points

    # --- Forward matrix G ---
    G = np.eye(len(i_buf_local))[ip_buf[i_buf_local], :]  # use only active cells

    # --- Model covariance Cm ---
    # Coordinates of model cells (scaled by dx, dy)
    coords_cm = np.column_stack(
        (GLOBAL["xx_norm"][i_buf] * dx, GLOBAL["yy_norm"][i_buf] * dy)
    )
    statmod_var = f"{curvariance} Gau({currange})"
    Cm, _ = precal_cov(coords_cm, coords_cm, statmod_var)

    # --- Data covariance Cd ---
    # Coordinates of conditioning data points
    coords_cd = np.column_stack(
        (GLOBAL["xx_norm"][ip_buf] * dx, GLOBAL["yy_norm"][ip_buf] * dy)
    )
    statmod_noi = f"1 Gau({currange})"
    Cd_shape, _ = precal_cov(coords_cd, coords_cd, statmod_noi)

    # Observation uncertainties as diagonal matrix
    img_unc_vec = np.asarray(GLOBAL["img_unc"][ip_buf], dtype=float)
    Cd_diag = np.diag(img_unc_vec)

    # Combine correlated and uncorrelated components
    Cd = (
        Cd_diag
        @ (cor_noise_rat * Cd_shape + (1 - cor_noise_rat) * np.eye(len(Cd_shape)))
        @ Cd_diag
    )

    # --- Observed data ---
    d_obs = np.asarray(GLOBAL["img_dobs"][ip_buf], dtype=float)

    # --- Prior mean model ---
    m0 = np.zeros(len(i_buf_local))

    if verbose:
        print("local_kriging_setup_img: Setup all kriging matrices")

    return G, Cm, Cd, d_obs, m0
