# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Tests for geosigma._connected_components (shared bwlabel equivalent)."""

from __future__ import annotations

import numpy as np
import pytest

from geosigma._connected_components import connected_components


def test_rejects_bad_connectivity():
    with pytest.raises(ValueError):
        connected_components(np.zeros((3, 3), dtype=bool), connectivity=6)


def test_4_connectivity_diagonal_is_not_connected():
    mask = np.array([[True, False], [False, True]])
    labels = connected_components(mask, connectivity=4)
    assert labels[0, 0] != labels[1, 1]


def test_8_connectivity_diagonal_is_connected():
    mask = np.array([[True, False], [False, True]])
    labels = connected_components(mask, connectivity=8)
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 0] != 0


def test_disjoint_blobs_get_distinct_labels():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4:6, 4:6] = True
    labels = connected_components(mask, connectivity=4)
    assert len(set(labels[0:2, 0:2].ravel())) == 1
    assert len(set(labels[4:6, 4:6].ravel())) == 1
    assert labels[0, 0] != labels[4, 4]
    assert labels.max() == 2
    assert (labels[~mask] == 0).all()


def test_all_false_mask_has_no_components():
    labels = connected_components(np.zeros((5, 5), dtype=bool), connectivity=4)
    assert labels.max() == 0
