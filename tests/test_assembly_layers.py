# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma.assembly.layers (monotonic layer stacking)."""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.assembly import LayeredModel, clamp_to_upper, stack_realizations


# -------------------------------------------------
# clamp_to_upper
# -------------------------------------------------
def test_clamp_leaves_a_valid_boundary_alone():
    upper = np.array([10.0, 10.0, 10.0])
    lower = np.array([9.0, 5.0, -3.0])
    np.testing.assert_array_equal(clamp_to_upper(lower, upper), lower)


def test_clamp_pulls_a_crossing_boundary_down_to_meet_the_one_above():
    upper = np.array([10.0, 10.0, 10.0])
    lower = np.array([9.0, 12.0, 10.0])
    # Only the crossing cell moves, and it lands exactly on `upper` -- the
    # layer pinches out to zero thickness rather than inverting.
    np.testing.assert_array_equal(clamp_to_upper(lower, upper), [9.0, 10.0, 10.0])


def test_clamp_never_modifies_the_upper_boundary():
    upper = np.array([10.0, 10.0])
    before = upper.copy()
    clamp_to_upper(np.array([99.0, 99.0]), upper)
    np.testing.assert_array_equal(upper, before)


def test_unsimulated_cells_inherit_the_boundary_above():
    # MATLAB's `| isnan(Grid)`: NaN means "not simulated here", and the
    # boundary takes the one above rather than leaving a hole.
    upper = np.array([10.0, 10.0])
    lower = np.array([np.nan, 4.0])
    np.testing.assert_array_equal(clamp_to_upper(lower, upper), [10.0, 4.0])


def test_nan_in_the_upper_boundary_leaves_the_lower_one_alone():
    # An unknown boundary above is not a constraint, so the lower one passes
    # through as drawn. MATLAB lands here too: with prevGrid NaN, both
    # `Grid > prevGrid` and `isnan(Grid)` are false.
    out = clamp_to_upper(np.array([4.0, 4.0]), np.array([np.nan, 10.0]))
    np.testing.assert_array_equal(out, [4.0, 4.0])


# -------------------------------------------------
# stack_realizations
# -------------------------------------------------
def _layer(values, n_reals=1):
    """An (ny, nx, n_reals) layer from a 2D array of values."""
    return np.repeat(np.asarray(values, dtype=float)[:, :, None], n_reals, axis=2)


def test_stack_enforces_monotonicity_down_the_whole_column():
    # Three boundaries where each is drawn above the one before it. The
    # clamp has to cascade: layer 3 is pulled to layer 2's *corrected*
    # position, not its original one.
    top = _layer([[0.0]])
    mid = _layer([[5.0]])
    bot = _layer([[9.0]])
    model = stack_realizations([top, mid, bot])
    np.testing.assert_allclose(model.surfaces[0, 0, :, 0], [0.0, 0.0, 0.0])


def test_stack_preserves_a_already_valid_model():
    top = _layer([[10.0, 10.0]])
    bot = _layer([[2.0, 3.0]])
    model = stack_realizations([top, bot])
    np.testing.assert_allclose(model.surfaces[:, :, 0, 0], [[10.0, 10.0]])
    np.testing.assert_allclose(model.surfaces[:, :, 1, 0], [[2.0, 3.0]])
    np.testing.assert_array_equal(model.n_clamped, [0, 0])


def test_terrain_becomes_boundary_zero_and_bounds_the_first_layer():
    terrain = np.array([[10.0, 10.0]])
    layer = _layer([[12.0, 4.0]], n_reals=3)
    model = stack_realizations([layer], terrain=terrain)

    assert model.surfaces.shape == (1, 2, 2, 3)
    # Terrain carries no kriging uncertainty, so it repeats across draws.
    np.testing.assert_allclose(model.surfaces[:, :, 0, :], 10.0)
    # The first layer is now clamped where it rose above the terrain.
    np.testing.assert_allclose(model.surfaces[0, :, 1, 0], [10.0, 4.0])
    assert model.n_clamped[0] == 0
    assert model.n_clamped[1] == 3  # one crossing cell in each of 3 draws


def test_realizations_are_corrected_independently():
    # Draw 0 crosses, draw 1 does not. Only the first may be touched.
    top = np.zeros((1, 1, 2))
    bot = np.array([[[5.0, -5.0]]])  # (1, 1, 2)
    model = stack_realizations([top, bot])
    np.testing.assert_allclose(model.surfaces[0, 0, 1, :], [0.0, -5.0])
    assert model.n_clamped[1] == 1


def test_mean_model_is_the_mean_of_corrected_draws_not_the_corrected_mean():
    # This is why the correction runs per realization. The raw draws
    # average to -1.0, which is below the layer above and so needs no
    # correction at all -- correcting the mean would be a no-op and would
    # leave draw 0 invalid. Correcting first gives a different, valid answer.
    top = np.zeros((1, 1, 2))
    bot = np.array([[[4.0, -6.0]]])
    assert bot.mean() == -1.0  # the trap: the mean alone looks fine

    model = stack_realizations([top, bot])
    np.testing.assert_allclose(model.mean_model[0, 0, 1], -3.0)
    assert model.mean_model[0, 0, 1] != bot.mean()


def test_stacked_model_has_no_negative_thickness_anywhere():
    rng = np.random.default_rng(0)
    # Four wildly overlapping boundaries -- exactly the situation that
    # independent per-layer kriging produces.
    layers = [rng.normal(-5 * i, 8.0, (12, 9, 20)) for i in range(4)]
    model = stack_realizations(layers)
    thickness = -np.diff(model.surfaces, axis=2)
    assert np.all(thickness >= 0)
    # And the clamp was genuinely exercised, so the assertion above means
    # something.
    assert model.n_clamped[1:].sum() > 0


def test_upper_boundary_is_never_moved_by_stacking():
    rng = np.random.default_rng(1)
    top = rng.normal(0, 5, (6, 6, 4))
    bot = rng.normal(0, 5, (6, 6, 4))
    model = stack_realizations([top, bot])
    np.testing.assert_array_equal(model.surfaces[:, :, 0, :], top)


def test_shapes_and_mean():
    layers = [np.zeros((3, 4, 5)), np.full((3, 4, 5), -2.0)]
    model = stack_realizations(layers)
    assert isinstance(model, LayeredModel)
    assert model.surfaces.shape == (3, 4, 2, 5)
    assert model.mean_model.shape == (3, 4, 2)
    np.testing.assert_allclose(model.mean_model[:, :, 1], -2.0)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="nothing to stack"):
        stack_realizations([])


def test_rejects_mismatched_layer_shapes():
    with pytest.raises(ValueError, match="same"):
        stack_realizations([np.zeros((2, 2, 3)), np.zeros((2, 3, 3))])


def test_rejects_mismatched_terrain():
    with pytest.raises(ValueError, match="terrain shape"):
        stack_realizations([np.zeros((2, 2, 3))], terrain=np.zeros((3, 3)))
