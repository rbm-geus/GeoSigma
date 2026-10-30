# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""The background variance (``prior``) of :func:`combine_variances`.

``prior`` is a required keyword argument: either a background-variance grid
(model theme or :func:`constant_prior`), or an explicit ``None``. The numbers
must not change: ``prior=X`` equals passing ``X`` last in the theme list, and
``prior=None`` equals the plain combine.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from geosigma.themes import (
    NODATA_VARIANCE,
    GeoSigmaWarning,
    combine_variances,
    constant_prior,
    write_variance_stack,
)
from geosigma.themes.__main__ import main as themes_cli
from utils import make_grid_meta

NY, NX, NL = 4, 3, 2


def _themes():
    rng = np.random.default_rng(1)
    a = rng.lognormal(0, 2, (NY, NX, NL))
    b = rng.lognormal(1, 2, (NY, NX, NL))
    a[0] = NODATA_VARIANCE  # row 0: no theme informs it
    b[0] = NODATA_VARIANCE
    b[1] = np.inf
    return [a, b]


def _reference_combine(grids):
    """The pre-``prior`` implementation, verbatim."""
    stack = np.stack([np.asarray(t, dtype=float) for t in grids], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.sum(1.0 / stack, axis=-1)


# ---------------------------------------------------------------------------
# The choice is required
# ---------------------------------------------------------------------------


def test_prior_is_a_required_keyword():
    with pytest.raises(TypeError):
        combine_variances(_themes())


def test_prior_cannot_be_passed_positionally():
    themes = _themes()
    with pytest.raises(TypeError):
        combine_variances(themes, themes[0])


# ---------------------------------------------------------------------------
# Numbers are unchanged
# ---------------------------------------------------------------------------


def test_prior_none_is_bit_identical_to_plain_combine():
    themes = _themes()
    with pytest.warns(GeoSigmaWarning):
        out = combine_variances(themes, prior=None)
    np.testing.assert_array_equal(out, _reference_combine(themes))


def test_prior_equals_appending_it_last():
    themes = _themes()
    prior = constant_prior([25.0, 400.0], (NY, NX))
    out = combine_variances(themes, prior=prior)
    np.testing.assert_array_equal(out, _reference_combine([*themes, prior]))


def test_prior_bounds_the_combined_variance():
    prior = constant_prior([25.0, 400.0], (NY, NX))
    out = combine_variances(_themes(), prior=prior)
    assert np.all(out <= prior)
    # Where no theme informs, the result is the prior refined only by the
    # sentinel's tiny precision: just below the prior.
    np.testing.assert_allclose(out[0, :, 0], 1 / (1 / 25 + 2 / NODATA_VARIANCE))


# ---------------------------------------------------------------------------
# The warning
# ---------------------------------------------------------------------------


def test_prior_none_warns_when_cells_are_uninformed():
    with pytest.warns(GeoSigmaWarning, match=r"6 of 24 cell-layers"):
        combine_variances(_themes(), prior=None)


def test_prior_none_warns_when_variance_reaches_sentinel():
    # Informed (finite, below the sentinel) yet the combine exceeds it:
    # one theme at +inf beside a nearly zero-certainty one.
    a = np.full((1, 1, 1), np.inf)
    b = np.full((1, 1, 1), 3e15)
    with pytest.warns(GeoSigmaWarning, match=r"1 of 1"):
        combine_variances([a, b], prior=None)


def test_prior_none_is_silent_when_every_cell_is_informed():
    themes = [np.full((NY, NX, NL), 4.0), np.full((NY, NX, NL), np.inf)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoSigmaWarning)
        out = combine_variances(themes, prior=None)
    np.testing.assert_array_equal(out, 4.0)


def test_prior_given_never_warns():
    prior = constant_prior([25.0, 400.0], (NY, NX))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoSigmaWarning)
        combine_variances(_themes(), prior=prior)


# ---------------------------------------------------------------------------
# constant_prior
# ---------------------------------------------------------------------------


def test_constant_prior_shape_and_values():
    prior = constant_prior([1.0, 2.0, 3.0], (2, 5))
    assert prior.shape == (2, 5, 3)
    np.testing.assert_array_equal(prior[1, 4], [1.0, 2.0, 3.0])
    prior[0, 0, 0] = 99.0  # a real, writeable array, not a broadcast view
    assert prior[1, 1, 0] == 1.0


@pytest.mark.parametrize("bad", [[], [1.0, 0.0], [1.0, -2.0], [np.inf], [[1.0]]])
def test_constant_prior_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        constant_prior(bad, (2, 2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_inputs(tmp_path):
    meta = make_grid_meta(
        ncols=NX, nrows=NY, dx=100.0, xllcorner=5e5, yllcorner=6.2e6, crs="EPSG:25832"
    )
    argv = []
    for i, t in enumerate(_themes()):
        d = tmp_path / f"theme{i}"
        write_variance_stack(t, meta, d)
        argv += ["--theme", str(d)]
    surf = tmp_path / "surfaces"
    (terrain,) = write_variance_stack(np.zeros((NY, NX, 1)), meta, surf, "terrain")
    bottoms = np.stack([np.full((NY, NX), -10.0), np.full((NY, NX), -20.0)], -1)
    for b in write_variance_stack(bottoms, meta, surf, "bottom"):
        argv += ["--bottom", str(b)]
    prior_dir = tmp_path / "prior"
    write_variance_stack(constant_prior([25.0, 400.0], (NY, NX)), meta, prior_dir)
    return argv + ["--terrain", str(terrain)], prior_dir


def test_cli_requires_a_background_choice(tmp_path):
    argv, _ = _write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        themes_cli(["combine", *argv, "--out", str(tmp_path / "out")])


def test_cli_prior_and_no_prior(tmp_path):
    argv, prior_dir = _write_inputs(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GeoSigmaWarning)
        assert (
            themes_cli(
                [
                    "combine",
                    *argv,
                    "--prior",
                    str(prior_dir),
                    "--out",
                    str(tmp_path / "a"),
                ]
            )
            == 0
        )
    with pytest.warns(GeoSigmaWarning):
        assert (
            themes_cli(["combine", *argv, "--no-prior", "--out", str(tmp_path / "b")])
            == 0
        )
