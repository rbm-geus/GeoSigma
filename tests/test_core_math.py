# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Phase 1 unit tests for the ``geosigma.core`` math primitives.

These check known small cases with analytically verifiable answers (and a few
structural invariants), guarding the move into ``core/`` and any later refactor.
Stochastic ``get_reals_cholesky`` is checked by reproducibility + statistical
convergence rather than golden values, per the Phase 0 verification decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.core.edist import edist
from geosigma.core.deformat_variogram import deformat_variogram
from geosigma.core.semivar_synth import semivar_synth
from geosigma.core.precal_cov import precal_cov
from geosigma.core.get_reals_cholesky import get_reals_cholesky
from geosigma.core.least_squares_inversion import least_squares_inversion
from geosigma.core.local_kriging_setup_img import local_kriging_setup_img


# --------------------------------------------------------------------------- #
# edist
# --------------------------------------------------------------------------- #
def test_edist_2d_known_distances():
    p1 = np.array([[0.0, 0.0], [3.0, 4.0]])
    p2 = np.array([[0.0, 0.0]])
    D, dp = edist(p1, p2)
    assert D.shape == (2, 1)
    np.testing.assert_allclose(D[:, 0], [0.0, 5.0])  # 3-4-5 triangle
    assert dp.shape == (2, 1, 2)


def test_edist_1d_is_absolute_difference():
    D, _ = edist(np.array([[0.0], [3.0]]), np.array([[1.0]]))
    np.testing.assert_allclose(D[:, 0], [1.0, 2.0])


def test_edist_default_p2_is_origin():
    # p2 defaults to zeros_like(p1); diagonal is each point's distance to origin
    D, _ = edist(np.array([[3.0, 4.0]]))
    np.testing.assert_allclose(D[0, 0], 5.0)


def test_edist_anisotropy_not_implemented():
    with pytest.raises(NotImplementedError):
        edist(np.array([[0.0, 0.0]]), transform=np.eye(2))


# --------------------------------------------------------------------------- #
# deformat_variogram
# --------------------------------------------------------------------------- #
def test_deformat_single_structure():
    assert deformat_variogram("1 Sph(5)") == [
        {"type": "Sph", "par1": 1.0, "par2": 5.0}
    ]


def test_deformat_nested_sum():
    V = deformat_variogram("0.5 Gau(10) + 0.5 Sph(20)")
    assert [v["type"] for v in V] == ["Gau", "Sph"]
    assert [v["par1"] for v in V] == [0.5, 0.5]
    assert [v["par2"] for v in V] == [10.0, 20.0]


# --------------------------------------------------------------------------- #
# semivar_synth
# --------------------------------------------------------------------------- #
def test_semivar_spherical_known_values():
    V = {"type": "Sph", "par1": 1.0, "par2": 10.0}
    h = np.array([0.0, 5.0, 10.0, 15.0])
    g = semivar_synth(V, h)
    # h=5: 1.5*0.5 - 0.5*0.125 = 0.6875 ; h>=range -> sill
    np.testing.assert_allclose(g, [0.0, 0.6875, 1.0, 1.0])


def test_semivar_gaussian_endpoints():
    V = {"type": "Gau", "par1": 2.0, "par2": 10.0}
    g = semivar_synth(V, np.array([0.0, 1e6]))
    np.testing.assert_allclose(g[0], 0.0)           # zero at origin
    np.testing.assert_allclose(g[1], 2.0, atol=1e-9)  # approaches sill


def test_semivar_unknown_model_raises():
    with pytest.raises(ValueError):
        semivar_synth({"type": "Bogus", "par1": 1.0, "par2": 1.0}, np.array([1.0]))


# --------------------------------------------------------------------------- #
# precal_cov
# --------------------------------------------------------------------------- #
def test_precal_cov_diagonal_is_sill_and_symmetric():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    cov, semiv = precal_cov(pos, pos, "2 Gau(15)")
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(np.diag(cov), 2.0)        # zero lag -> full sill
    np.testing.assert_allclose(cov, cov.T)               # symmetric
    np.testing.assert_allclose(np.diag(semiv), 0.0)      # zero semivariance at lag 0
    assert np.all(cov <= 2.0 + 1e-12)                    # covariance bounded by sill


def test_precal_cov_string_matches_parsed_list():
    pos = np.array([[0.0, 0.0], [5.0, 0.0]])
    cov_str, _ = precal_cov(pos, pos, "1 Sph(20)")
    cov_list, _ = precal_cov(pos, pos, deformat_variogram("1 Sph(20)"))
    np.testing.assert_allclose(cov_str, cov_list)


# --------------------------------------------------------------------------- #
# get_reals_cholesky
# --------------------------------------------------------------------------- #
def test_get_reals_cholesky_shape_and_reproducible():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    cov, _ = precal_cov(pos, pos, "2 Gau(15)")

    np.random.seed(0)
    a = get_reals_cholesky(cov, Nreals=5)
    np.random.seed(0)
    b = get_reals_cholesky(cov, Nreals=5)

    assert a.shape == (3, 5)                 # (n_points, n_reals)
    np.testing.assert_array_equal(a, b)      # seeded -> reproducible


def test_get_reals_cholesky_statistics_converge():
    pos = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    cov, _ = precal_cov(pos, pos, "2 Gau(15)")

    np.random.seed(123)
    m0 = np.array([1.0, 2.0, 3.0])
    reals = get_reals_cholesky(cov, Nreals=200_000, m0=m0)

    np.testing.assert_allclose(reals.mean(axis=1), m0, atol=0.02)
    sample_cov = np.cov(reals)
    np.testing.assert_allclose(sample_cov, cov, atol=0.05)


def test_get_reals_cholesky_rejects_mismatched_mean():
    cov = np.eye(3)
    with pytest.raises(ValueError):
        get_reals_cholesky(cov, Nreals=1, m0=np.zeros(2))


# --------------------------------------------------------------------------- #
# least_squares_inversion
# --------------------------------------------------------------------------- #
def test_lsq_type2_identity_case():
    # G=I, Cm=Cd=I, m0=0 -> K=0.5 I, m_est=0.5 d0, Cm_est=0.5 I
    G = np.eye(2)
    Cm = np.eye(2)
    Cd = np.eye(2)
    d0 = np.array([4.0, 8.0])
    m_est, Cm_est = least_squares_inversion(G, Cm, Cd, m0=0.0, d0=d0, type=2)
    np.testing.assert_allclose(m_est, [2.0, 4.0])
    np.testing.assert_allclose(Cm_est, 0.5 * np.eye(2))


def test_lsq_type1_matches_type2_full_rank():
    rng = np.random.default_rng(7)
    G = np.eye(3)
    Cm = np.diag([1.0, 2.0, 3.0])
    Cd = np.diag([0.5, 0.5, 0.5])
    d0 = rng.normal(size=3)
    m1, C1 = least_squares_inversion(G, Cm, Cd, m0=0.0, d0=d0, type=1)
    m2, C2 = least_squares_inversion(G, Cm, Cd, m0=0.0, d0=d0, type=2)
    np.testing.assert_allclose(m1, m2)
    np.testing.assert_allclose(C1, C2)


# --------------------------------------------------------------------------- #
# local_kriging_setup_img  (+ end-to-end core chain)
# --------------------------------------------------------------------------- #
def _toy_global():
    return {
        "xx_norm": np.array([0.0, 1.0, 2.0]),
        "yy_norm": np.array([0.0, 0.0, 0.0]),
        "img_unc": np.array([1.0, 1.0, 1.0]),
        "img_dobs": np.array([10.0, 20.0, 30.0]),
    }


def test_local_kriging_setup_shapes_and_content():
    GLOBAL = _toy_global()
    i_buf = np.array([True, True, True])
    ip_buf = np.array([0, 1, 2])
    G, Cm, Cd, d_obs, m0 = local_kriging_setup_img(
        GLOBAL, i_buf, ip_buf, curvariance=2.0, currange=150.0,
        cor_noise_rat=0.5, dx=100, dy=100,
    )
    assert G.shape == (3, 3) and Cm.shape == (3, 3) and Cd.shape == (3, 3)
    np.testing.assert_allclose(G, np.eye(3))            # all cells observed
    np.testing.assert_allclose(d_obs, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(m0, np.zeros(3))
    np.testing.assert_allclose(np.diag(Cm), 2.0)        # zero-lag sill
    np.testing.assert_allclose(Cd, Cd.T)                # symmetric data cov


def test_core_chain_runs_end_to_end():
    # setup -> inversion: the whole core path on a tiny problem
    GLOBAL = _toy_global()
    i_buf = np.array([True, True, True])
    ip_buf = np.array([0, 1, 2])
    G, Cm, Cd, d_obs, m0 = local_kriging_setup_img(
        GLOBAL, i_buf, ip_buf, curvariance=2.0, currange=150.0,
        cor_noise_rat=0.5,
    )
    m_est, Cm_est = least_squares_inversion(G, Cm, Cd, m0=m0, d0=d_obs, type=2)
    assert m_est.shape == (3,)
    assert Cm_est.shape == (3, 3)
    # posterior variance must not exceed the prior variance (data reduces it)
    assert np.all(np.diag(Cm_est) <= np.diag(Cm) + 1e-9)
