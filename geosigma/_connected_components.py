# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Connected-component labeling (``bwlabel`` equivalent).

Shared by :mod:`geosigma.kriging.windows` (``bwlabel(mask, 4)``, per-cluster
element splitting in ``local_kriger.m``) and
:mod:`geosigma.preprocess.clusters` (``bwlabel(mask)`` — MATLAB's *default*
connectivity, which is 8, not 4 — in ``post_process_clusters.m``). Kept as a
private, leading-underscore module rather than duplicated in each, and rather
than placed under either ``kriging`` or ``preprocess`` specifically, since
both depend on it and neither should depend on the other.

Pure numpy/Python flood fill — no ``scipy`` dependency (consistent with the
rest of the library). Label *values* (1, 2, 3, ...) are assigned in scan
order and don't need to match MATLAB's numbering, which is itself
scan-order-dependent and arbitrary; only the partition into components needs
to match, and flood fill reproduces that exactly for a given connectivity.
"""

from __future__ import annotations

import numpy as np

__all__ = ["connected_components"]

_NEIGHBOR_OFFSETS = {
    4: ((1, 0), (-1, 0), (0, 1), (0, -1)),
    8: ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)),
}


def connected_components(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Label connected components of ``mask``.

    Parameters
    ----------
    mask : ndarray of bool, shape (ny, nx)
    connectivity : {4, 8}, default 4
        4: edge-adjacent neighbours only (MATLAB ``bwlabel(mask, 4)``).
        8: edge- and corner-adjacent neighbours (MATLAB ``bwlabel(mask)``'s
        default when no connectivity argument is given).

    Returns
    -------
    ndarray of int, shape mask.shape
        ``0`` = background; components numbered ``1..n`` in the order found.

    Notes
    -----
    O(n_true_cells) Python-loop flood fill: correct and simple, not
    vectorized. Fine for the grid sizes exercised by the tests and examples;
    a production-scale run on a full national grid may want a faster
    labeling implementation.
    """
    if connectivity not in _NEIGHBOR_OFFSETS:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    offsets = _NEIGHBOR_OFFSETS[connectivity]

    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=int)
    ny, nx = mask.shape
    current = 0
    for r0 in range(ny):
        for c0 in range(nx):
            if mask[r0, c0] and labels[r0, c0] == 0:
                current += 1
                stack = [(r0, c0)]
                labels[r0, c0] = current
                while stack:
                    r, c = stack.pop()
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        if (
                            0 <= nr < ny
                            and 0 <= nc < nx
                            and mask[nr, nc]
                            and labels[nr, nc] == 0
                        ):
                            labels[nr, nc] = current
                            stack.append((nr, nc))
    return labels
