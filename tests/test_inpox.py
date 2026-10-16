# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma.inpox (informed point extraction).

The transfer function is checked against the closed forms in Madsen et al.
(2024), *Introducing INPOX*, GEUS Bulletin 57, 8364, eqs 2-4, and the
Laplacian against MATLAB ``del2`` semantics (quarter-Laplacian, unit
spacing, edge replication rather than periodic wrap).
"""

from __future__ import annotations

import numpy as np

from geosigma.inpox import draw_points_inpox, ftot_laplace, laplacian_2d

# Madsen et al. (2024) Table 1, real-world case (T4).
T4 = {"L0": 1.0, "A": 1.5, "B": 3.0, "C": 5.0, "D": 1.0, "E": 0.8, "g": 2.0}
P0 = 0.05


def ftot(L, **overrides):
    params = {**T4, **overrides}
    return ftot_laplace(L, params.pop("P0", P0), **params)


# -------------------------------------------------
# ftot_laplace: the paper's eqs 2-4
# -------------------------------------------------
def test_flat_ground_sits_at_the_baseline():
    # At L = 0, f1 has not switched on yet: tanh(-A*L0*g) is nearly -1, so
    # only a sliver of C*P0 leaks through and f(0) is a shade above P0.
    leak = T4["C"] * P0 * (1 - np.tanh(T4["A"] * T4["L0"] * T4["g"])) / 2
    assert abs(ftot(0.0) - (P0 + leak)) < 1e-12
    assert leak < 0.02 * P0  # and that sliver is small


def test_peak_is_one_plus_C_times_P0():
    # f1 saturates at C*P0 above A*L0, so the curve tops out just short of
    # (1+C)*P0 at the handover, short by f1's unsaturated tanh residual.
    peak = ftot(T4["B"] * T4["L0"] - 1e-9)
    deficit = T4["C"] * P0 * (1 - np.tanh((T4["B"] - T4["A"]) * T4["L0"] * T4["g"])) / 2
    assert abs(peak - ((1 + T4["C"]) * P0 - deficit)) < 1e-9
    assert deficit < 0.01 * (1 + T4["C"]) * P0


def test_handover_at_B_is_continuous():
    # The branches meet to within C*P0*(1 - tanh((B-A)*L0*g))/2, the residual
    # of f1's tanh not being fully saturated at B.
    B = T4["B"] * T4["L0"]
    jump = abs(ftot(B + 1e-9) - ftot(B - 1e-9))
    expected = T4["C"] * P0 * (1 - np.tanh((T4["B"] - T4["A"]) * T4["L0"] * T4["g"])) / 2
    assert jump < 2 * expected


def test_tapers_to_zero_rather_than_dropping():
    # D = 1 makes f2 tend to (1-D)*P0 = 0, and it gets there smoothly: the
    # whole point is that a cell just past B is not treated wildly
    # differently from one just before it.
    L = np.array([3.0, 4.0, 6.0, 10.0, 50.0, 500.0])
    f = ftot(L)
    assert np.all(np.diff(f) < 0)  # monotone decreasing past B
    assert np.all(f > 0)  # never negative
    assert f[-1] < 1e-4 * P0  # but arbitrarily close to 0
    # Not a cliff: one cell past B still carries most of the peak.
    assert ftot(3.2) > 0.5 * (1 + T4["C"]) * P0


def test_D_sets_the_floor_it_tapers_to():
    # With D < 1 the tail settles on (1-D)*P0 instead of 0.
    far = ftot(1e4, D=0.4)
    assert abs(far - 0.6 * P0) < 1e-6


def test_C_scales_the_peak_but_not_the_baseline():
    for C in (1.0, 5.0, 15.0):
        peak = ftot(T4["B"] - 1e-9, C=C)
        assert abs(peak - (1 + C) * P0) < 0.02 * (1 + C) * P0
    # The far tail lands on (1-D)*P0 = 0 for any C, so varying C changes how
    # tightly points concentrate, not the level either end settles at.
    for C in (1.0, 15.0):
        assert ftot(1e4, C=C) < 1e-6 * P0


def test_P0_scales_the_whole_curve():
    L = np.linspace(0, 10, 50)
    assert np.allclose(ftot(L, P0=0.1), 2 * ftot(L, P0=0.05))


# -------------------------------------------------
# laplacian_2d: MATLAB del2 semantics
# -------------------------------------------------
def test_quarter_laplacian_of_a_quadratic():
    # del2 is (Uxx + Uyy)/4, not the raw 5-point stencil sum. For
    # z = x^2 + y^2 that is (2 + 2)/4 = 1 everywhere, edges included
    # (the one-sided difference of a quadratic equals the interior one).
    n = 7
    i = np.arange(n, dtype=float)
    xx, yy = np.meshgrid(i, i)
    z = xx**2 + yy**2
    assert np.allclose(laplacian_2d(z), 1.0)


def test_edges_replicate_and_do_not_wrap():
    # A monotone ramp has zero curvature. A periodic implementation would
    # see a huge jump where the last column meets the first and report
    # spurious curvature at both edges.
    z = np.tile(np.arange(8, dtype=float), (5, 1))
    assert np.allclose(laplacian_2d(z), 0.0)


def test_spacing_divides_the_second_difference():
    n = 7
    i = np.arange(n, dtype=float)
    xx, yy = np.meshgrid(i, i)
    z = xx**2 + yy**2
    # Halving dx quadruples the x contribution: (2/0.25 + 2)/4 = 2.5.
    assert np.allclose(laplacian_2d(z, dx=0.5), 2.5)


# -------------------------------------------------
# draw_points_inpox
# -------------------------------------------------
def rough_and_smooth_surface(n=120, seed=0):
    """Left half noisy, right half a gentle plane."""
    rng = np.random.default_rng(seed)
    z = np.tile(np.linspace(0, 5, n), (n, 1))
    z[:, : n // 2] += rng.normal(0, 3, (n, n // 2))
    return z


def test_P0_is_a_floor_that_roughness_adds_to():
    # The default (drawpoints.m) behaviour: no rescaling, so the realised
    # draw fraction lands ABOVE P0, by however much structure the surface
    # has. A rough surface earning more points -- not merely better-placed
    # ones -- is the point of informed extraction.
    z = rough_and_smooth_surface()
    np.random.seed(0)
    pts, det = draw_points_inpox(z, {"P0": 0.05, **T4}, return_details=True)
    assert det["transfer"].mean() > 1.2 * 0.05  # roughness adds >20%
    assert pts.mean() > 0.05
    # And the draw tracks the probability map it came from.
    assert abs(pts.mean() - det["probability"].mean()) < 0.01


def test_normalize_pins_the_expected_count_to_P0():
    # normalize=True rescales the map to mean P0, giving the same expected
    # count as a uniform draw at P0 -- for benchmarking against one.
    z = rough_and_smooth_surface()
    for p0 in (0.02, 0.05, 0.20):
        np.random.seed(0)
        pts = draw_points_inpox(z, {"P0": p0, **T4}, normalize=True)
        assert abs(pts.mean() - p0) < 0.25 * p0 + 0.01
    # It is strictly fewer points than the default floor behaviour.
    np.random.seed(0)
    plain = draw_points_inpox(z, {"P0": 0.05, **T4}).mean()
    np.random.seed(0)
    scaled = draw_points_inpox(z, {"P0": 0.05, **T4}, normalize=True).mean()
    assert scaled < plain


def test_probability_never_exceeds_one():
    # The GridExtra floor and well boosts each add a flat +1, so the raw map
    # runs above 1 and has to be clipped to read as a probability.
    z = rough_and_smooth_surface()
    np.random.seed(0)
    _, det = draw_points_inpox(z, {"P0": 0.2, **T4}, return_details=True)
    assert det["probability"].max() <= 1.0
    assert det["probability"].min() > 0.0


def test_expected_count_is_capped_at_one_point_per_cell():
    # drawpoints.m's `maxp = numel(surface)*1` safety valve. It needs an
    # absurd P0 to bind, which is why it is not the normalization above.
    z = rough_and_smooth_surface()
    np.random.seed(0)
    _, det = draw_points_inpox(z, {"P0": 1.0, **T4}, return_details=True)
    assert det["transfer"].mean() > 1.0  # raw map would over-commit
    assert det["probability"].mean() <= 1.0


def test_rough_ground_gets_more_points_than_smooth():
    # This is the entire premise of INPOX; if it fails, point drawing has
    # silently degenerated to uniform sampling.
    z = rough_and_smooth_surface()
    np.random.seed(0)
    pts = draw_points_inpox(z, {"P0": 0.05, **T4})
    half = z.shape[1] // 2
    assert pts[:, :half].mean() > 2 * pts[:, half:].mean()


def test_grid_floor_guarantees_a_baseline_point():
    # drawpoints.m seeds every 10th row/col with probability 1, so those
    # cells are drawn whatever the roughness says.
    z = np.zeros((40, 40))  # perfectly flat: roughness contributes nothing
    np.random.seed(0)
    pts = draw_points_inpox(z, {"P0": 1e-6, **T4})
    assert pts[::10, ::10].all()


def test_wells_are_forced_and_snapped_by_rounding():
    z = np.zeros((40, 40))
    ext = {"P0": 1e-6, "x0": 0.0, "y0": 0.0, **T4}
    # 253 m at dx = 100 m rounds to column 3, not 2 (MATLAB round, not ceil).
    wells = {"xutm": [253.0], "yutm": [1480.0]}
    np.random.seed(0)
    pts = draw_points_inpox(z, ext, dx=100.0, dy=100.0, wells=wells)
    assert pts[15, 3]


def test_details_are_consistent_with_each_other():
    z = rough_and_smooth_surface()
    np.random.seed(0)
    pts, det = draw_points_inpox(z, {"P0": 0.05, **T4}, return_details=True)
    assert set(det) == {"laplacian", "transfer", "probability"}
    for key in det:
        assert det[key].shape == z.shape
    assert np.allclose(det["laplacian"], np.abs(laplacian_2d(z)))
    assert np.allclose(det["transfer"], ftot(det["laplacian"]))
    assert pts.dtype == bool
