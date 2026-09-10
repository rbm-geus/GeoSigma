# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Per-sounding depth-of-investigation (DOI) resolution.

Geophysical surveys that infer subsurface structure from electrical
conductivity (TEM, airborne EM, EM soundings) can only "see" down to a finite
**depth of investigation**: below it the signal is too attenuated to constrain
the model. Each sounding's inversion usually reports its own DOI, but some
soundings have none recorded. :func:`resolve_doi` fills those gaps.

The general mechanism (model-agnostic, lives here)
--------------------------------------------------
A **conductive horizon** — a laterally extensive, highly conductive layer (e.g.
a thick marine clay) — caps how deep a conductivity-sensitive method can see:
once the signal reaches it, it is absorbed and nothing below is resolved. So
where a sounding's DOI is missing, it can be *estimated* from such a horizon:

* if a conductive horizon is present **and thick enough** to matter, the DOI is
  capped at the depth of the horizon's top (or the regional fallback, whichever
  is shallower);
* if the horizon is **thin or absent**, it imposes no cap and the missing DOI
  falls back to a regional value (typically the mean of the recorded DOIs).

This function knows nothing about *which* layer the conductive horizon is, where
it sits in any particular stratigraphic model, or any region's geology. The
caller supplies the horizon as **data**: per-sounding top-depth and thickness
arrays (in Denmark these are sampled from the Palaeogene clay surfaces of the
national model; elsewhere they would be a different horizon, or ``None``). The
Danish column-offset bookkeeping that *locates* the Palaeogene layer is the
caller's job and must not leak in here.

Stage matters: per-sounding, before windowing
----------------------------------------------
DOI resolution is **per-sounding** and must happen **before** any spatial
windowing / nearest-point reduction, because the estimate depends on the
*sounding's own* location (its horizon depth). It cannot be deferred to a
per-cell post-reduction step the way a constant or global-mean fill can — once
soundings are reduced to per-cell values, the per-sounding horizon depth is
gone. (Constant/mean fills, e.g. PACEP's "missing -> 15" or MEP's
"missing -> mean", are location-independent and legitimately live downstream in
the theme spec; this conductive-horizon estimate does not.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def resolve_doi(
    doi_reported: np.ndarray,
    *,
    horizon_top: Optional[np.ndarray] = None,
    horizon_thickness: Optional[np.ndarray] = None,
    min_horizon_thickness: float = 10.0,
    fallback_doi: Optional[float] = None,
    fallback_when_empty: float = 150.0,
) -> np.ndarray:
    """Resolve per-sounding DOI, filling missing (NaN) reported values.

    Finite entries of ``doi_reported`` are kept verbatim. Each missing entry is
    replaced by an estimate:

    * **with a conductive horizon** (``horizon_top`` given): where the horizon is
      *thick* (``horizon_thickness >= min_horizon_thickness``) the estimate is
      ``min(fallback_doi, horizon_top)`` — the DOI is capped at the horizon's
      top; where the horizon is *thin* (``horizon_thickness <
      min_horizon_thickness``) it imposes no cap and the estimate is
      ``fallback_doi``;
    * **without a horizon** (``horizon_top is None``): every missing entry
      becomes ``fallback_doi``.

    Parameters
    ----------
    doi_reported : ndarray, shape (npoints,)
        Reported / inversion DOI per sounding; ``NaN`` marks "not recorded".
        Not modified in place (a copy is returned).
    horizon_top : ndarray, shape (npoints,), optional
        Depth to the **top** of the conductive horizon at each sounding. ``None``
        (default) means no conductive horizon is supplied — missing DOIs fall
        straight to ``fallback_doi``.
    horizon_thickness : ndarray, shape (npoints,), optional
        Thickness of the conductive horizon at each sounding. Required when
        ``horizon_top`` is given; ignored when it is ``None``.
    min_horizon_thickness : float, default 10.0
        A horizon thinner than this imposes no DOI cap. ``NaN`` thickness is
        treated as **not thin** (so it follows the capped branch); since an
        undefined horizon also has a ``NaN`` top, such a sounding resolves to
        ``NaN`` (DOI left unresolved) rather than silently capped to a number.
    fallback_doi : float, optional
        Regional DOI used where the horizon imposes no cap (and the only value
        used when ``horizon_top is None``). ``None`` (default) means "use the
        mean of the finite ``doi_reported`` values".
    fallback_when_empty : float, default 150.0
        Used as ``fallback_doi`` only when it was ``None`` *and* there are no
        finite ``doi_reported`` values to average.

    Returns
    -------
    ndarray, shape (npoints,)
        A new array: ``doi_reported`` with its missing entries filled. Soundings
        whose estimate is itself ``NaN`` (undefined horizon) stay ``NaN``.

    Notes
    -----
    The "thin -> no cap" branch is expressed semantically rather than via the
    MATLAB original's ``min(mean, top + 1000 * (thick < min))`` arithmetic trick;
    the two agree for any realistic input (a regional DOI never exceeds a
    horizon top plus a kilometre), and using ``thickness < min_horizon_thickness``
    as the thin-test reproduces the original's ``NaN``-thickness branch exactly.
    """
    doi = np.array(doi_reported, dtype=float)  # copy; never mutate the caller's
    if doi.ndim != 1:
        raise ValueError(f"doi_reported must be 1-D, got shape {doi.shape}")

    missing = np.isnan(doi)
    if not missing.any():
        return doi  # all reported -> nothing to fill

    # Resolve the regional fallback once (mean of finite reported DOIs).
    if fallback_doi is None:
        finite = doi[np.isfinite(doi)]
        fallback_doi = (
            float(finite.mean()) if finite.size else float(fallback_when_empty)
        )
    fallback_doi = float(fallback_doi)

    if horizon_top is None:
        est = np.full(doi.shape, fallback_doi)
    else:
        if horizon_thickness is None:
            raise ValueError("horizon_thickness is required when horizon_top is given")
        top = np.asarray(horizon_top, dtype=float)
        thickness = np.asarray(horizon_thickness, dtype=float)
        if top.shape != doi.shape or thickness.shape != doi.shape:
            raise ValueError(
                "horizon_top and horizon_thickness must match doi_reported shape "
                f"{doi.shape}; got {top.shape} and {thickness.shape}"
            )
        # Thin test matches the MATLAB convention: NaN thickness is NOT thin, so
        # it follows the capped branch (and resolves to NaN via a NaN top).
        thin = thickness < min_horizon_thickness
        est = np.where(thin, fallback_doi, np.minimum(fallback_doi, top))

    doi[missing] = est[missing]
    return doi
