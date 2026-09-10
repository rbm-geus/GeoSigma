# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Unit tests for ``geosigma.preprocess.doi.resolve_doi``.

The conductive-horizon DOI estimate is model-agnostic; these check each branch
on small hand-computed cases (no Danish/Palaeogene specifics involved).
"""

from __future__ import annotations

import numpy as np
import pytest

from geosigma.preprocess.doi import resolve_doi


def test_all_finite_is_noop():
    doi = np.array([10.0, 20.0, 30.0])
    out = resolve_doi(doi)
    np.testing.assert_array_equal(out, doi)
    # A copy is returned — the caller's array is never mutated.
    assert out is not doi


def test_missing_with_thick_horizon_is_capped():
    # Point 1 is missing; horizon thick (>= 10) and shallow (top=40) -> capped at
    # min(fallback, top). fallback = mean of the finite values [60, 80] = 70, so
    # the cap (40) wins.
    doi = np.array([60.0, np.nan, 80.0])
    top = np.array([40.0, 40.0, 40.0])
    thick = np.array([25.0, 25.0, 25.0])
    out = resolve_doi(doi, horizon_top=top, horizon_thickness=thick)
    np.testing.assert_array_equal(out, [60.0, 40.0, 80.0])


def test_thick_horizon_deeper_than_fallback_uses_fallback():
    # Horizon thick but its top (500) is deeper than the regional mean -> the
    # fallback caps instead: min(fallback=70, 500) = 70.
    doi = np.array([60.0, np.nan, 80.0])
    top = np.full(3, 500.0)
    thick = np.full(3, 25.0)
    out = resolve_doi(doi, horizon_top=top, horizon_thickness=thick)
    np.testing.assert_array_equal(out, [60.0, 70.0, 80.0])


def test_missing_with_thin_horizon_falls_through_to_fallback():
    # Thin horizon (< min_horizon_thickness) imposes no cap, even though the top
    # is shallow (5) -> missing resolves to fallback (mean[60,80] = 70), NOT 5.
    doi = np.array([60.0, np.nan, 80.0])
    top = np.array([5.0, 5.0, 5.0])
    thick = np.array([3.0, 3.0, 3.0])  # < 10
    out = resolve_doi(doi, horizon_top=top, horizon_thickness=thick)
    np.testing.assert_array_equal(out, [60.0, 70.0, 80.0])


def test_horizon_none_falls_straight_to_fallback():
    doi = np.array([60.0, np.nan, 80.0, np.nan])
    out = resolve_doi(doi)  # no horizon at all
    np.testing.assert_array_equal(out, [60.0, 70.0, 80.0, 70.0])  # fallback = 70


def test_fallback_none_uses_nanmean_of_finite():
    # Explicitly exercise the nanmean default with a horizon present but thin
    # (so the fill is the fallback, i.e. the nanmean).
    doi = np.array([10.0, 20.0, 30.0, np.nan])  # mean of finite = 20
    top = np.full(4, 1.0)
    thick = np.full(4, 1.0)  # thin -> fallback
    out = resolve_doi(doi, horizon_top=top, horizon_thickness=thick)
    np.testing.assert_array_equal(out, [10.0, 20.0, 30.0, 20.0])


def test_empty_dataset_uses_fallback_when_empty():
    # All-NaN reported DOIs: nanmean is undefined -> fallback_when_empty (150).
    doi = np.array([np.nan, np.nan, np.nan])
    out = resolve_doi(doi)
    np.testing.assert_array_equal(out, [150.0, 150.0, 150.0])


def test_empty_dataset_respects_custom_fallback_when_empty():
    doi = np.array([np.nan, np.nan])
    out = resolve_doi(doi, fallback_when_empty=42.0)
    np.testing.assert_array_equal(out, [42.0, 42.0])


def test_explicit_fallback_doi_overrides_mean():
    doi = np.array([10.0, np.nan, 30.0])
    out = resolve_doi(doi, fallback_doi=99.0)
    np.testing.assert_array_equal(out, [10.0, 99.0, 30.0])


def test_nan_horizon_with_finite_thin_thickness_is_unresolved():
    # NaN top but thickness finite & >= min -> capped branch -> min(fallback, NaN)
    # = NaN. This matches the MATLAB convention (an undefined horizon top does
    # not get silently filled with the fallback in the capped branch).
    doi = np.array([np.nan])
    out = resolve_doi(doi, horizon_top=np.array([np.nan]),
                      horizon_thickness=np.array([25.0]), fallback_doi=70.0)
    assert np.isnan(out[0])


def test_reported_values_are_never_overwritten():
    # Even with a shallow thick horizon, a *recorded* DOI deeper than the cap is
    # kept — resolution only fills missing entries.
    doi = np.array([200.0, np.nan])
    top = np.array([30.0, 30.0])
    thick = np.array([50.0, 50.0])
    out = resolve_doi(doi, horizon_top=top, horizon_thickness=thick,
                      fallback_doi=70.0)
    assert out[0] == 200.0   # kept
    assert out[1] == 30.0    # capped


def test_requires_thickness_when_top_given():
    with pytest.raises(ValueError, match="horizon_thickness is required"):
        resolve_doi(np.array([np.nan]), horizon_top=np.array([10.0]))


def test_rejects_non_1d():
    with pytest.raises(ValueError, match="1-D"):
        resolve_doi(np.zeros((2, 2)))


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="match doi_reported shape"):
        resolve_doi(np.array([np.nan, 1.0]), horizon_top=np.array([1.0]),
                    horizon_thickness=np.array([1.0]))
