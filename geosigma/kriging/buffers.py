# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Buffer-zone weights around one kriging cluster element.

Literal translation of ``find_100_m_buffer.m`` + ``calculate_buffer_zones.m``.
An earlier version of this module was a from-scratch redesign (a
parametrized, arbitrary-width erosion ramp) that only matched the original's
*role*, not its actual algorithm or output — this version instead reproduces
the specific fixed-geometry, 7-level scheme ``local_kriger.m`` relies on, so
its output corresponds to what the reference pipeline produces.

The scheme, exactly as in MATLAB: crop to the element's bounding box padded
by 4 cells; grow 3 rings **outward** from the element (into previously-empty
territory) and 3 rings **inward** (eroding the element itself), one cell per
ring, via repeated gradient-based border detection; assign the 8 evenly
spaced weights of ``linspace(0, 1, 8)`` — 0 far outside, climbing through the
3 outward rings, jumping to the 3 inward rings, reaching 1 in the element's
untouched interior (4+ cells from its own edge).
"""

from __future__ import annotations

import numpy as np

__all__ = ["calculate_buffer_zones"]

#: The 8 evenly spaced weight levels used by the original (``linspace(0,1,8)``).
_WEIGHTS = np.linspace(0.0, 1.0, 8)


def _find_100m_buffer(elem_img: np.ndarray, direction: str) -> np.ndarray:
    """One ring of border cells, by gradient-based edge detection.

    ``direction="positive"``: border cells just **outside** ``elem_img``
    (candidates for outward growth). ``direction="negative"``: border cells
    on the **inside** edge of ``elem_img`` (candidates for inward erosion).

    ``np.gradient`` matches MATLAB's ``gradient`` exactly for this use (unit
    spacing, central differences in the interior, one-sided at the array
    edges), so this is a direct translation, not an approximation.
    """
    grad_y, grad_x = np.gradient(elem_img)
    border = (np.abs(grad_x) + np.abs(grad_y)) > 0
    if direction == "positive":
        return border & (elem_img == 0)
    if direction == "negative":
        return border & (elem_img == 1)
    raise ValueError(f"direction must be 'positive' or 'negative', got {direction!r}")


def calculate_buffer_zones(element_mask: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Build the blend-weight map for one kriging cluster element.

    Parameters
    ----------
    element_mask : ndarray of bool, shape (ny, nx)
        One connected component of one cluster (e.g. from
        :func:`geosigma.kriging.windows.connected_components_4`). Must have
        at least one ``True`` cell.
    ny, nx : int
        Full grid shape (the output is embedded back into this size, matching
        the cropped-then-reinserted bounding-box strategy of the original).

    Returns
    -------
    ndarray, shape (ny, nx)
        Weights in ``[0, 1]``: 1 in the element's interior, ramping down
        through 3 inward and 3 outward rings (1 grid cell each) to 0 beyond.
    """
    element_mask = np.asarray(element_mask, dtype=bool)
    rows, cols = np.nonzero(element_mask)
    if rows.size == 0:
        raise ValueError("element_mask has no True cells")

    min_y = max(0, rows.min() - 4)
    max_y = min(ny - 1, rows.max() + 4)
    min_x = max(0, cols.min() - 4)
    max_x = min(nx - 1, cols.max() + 4)

    elem = element_mask[min_y : max_y + 1, min_x : max_x + 1].astype(float)

    weighted = elem.copy()

    # Outward rings (into empty space): weights[1], [2], [3] = 1/7, 2/7, 3/7,
    # assigned innermost-ring-last so it wins in the (normally empty)
    # intersection, exactly as the MATLAB assignment order does.
    grown = elem
    outward_rings = []
    for _ in range(3):
        ring = _find_100m_buffer(grown, "positive")
        outward_rings.append(ring)
        grown = grown + ring.astype(float)
    weighted[outward_rings[0]] = _WEIGHTS[3]
    weighted[outward_rings[1]] = _WEIGHTS[2]
    weighted[outward_rings[2]] = _WEIGHTS[1]

    # Inward rings (eroding the element): weights[4], [5], [6] = 4/7, 5/7, 6/7.
    eroded = elem
    inward_rings = []
    for _ in range(3):
        ring = _find_100m_buffer(eroded, "negative")
        inward_rings.append(ring)
        eroded = eroded - ring.astype(float)
    weighted[inward_rings[0]] = _WEIGHTS[4]
    weighted[inward_rings[1]] = _WEIGHTS[5]
    weighted[inward_rings[2]] = _WEIGHTS[6]

    out = np.zeros((ny, nx))
    out[min_y : max_y + 1, min_x : max_x + 1] = weighted
    return out
