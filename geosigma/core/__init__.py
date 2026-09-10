# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.core — low-level geostatistical math.

Home of the already-translated primitives (``precal_cov``, ``edist``,
``deformat_variogram``, ``semivar_synth``, ``least_squares_inversion``,
``get_reals_cholesky``, ``local_kriging_setup_img``). Moved here in **Phase 1**;
these are re-exported from the top-level ``geosigma`` package for convenience.
"""

from .precal_cov import precal_cov
from .edist import edist
from .semivar_synth import semivar_synth
from .deformat_variogram import deformat_variogram
from .local_kriging_setup_img import local_kriging_setup_img
from .least_squares_inversion import least_squares_inversion
from .get_reals_cholesky import get_reals_cholesky

__all__ = [
    "precal_cov",
    "edist",
    "semivar_synth",
    "deformat_variogram",
    "local_kriging_setup_img",
    "least_squares_inversion",
    "get_reals_cholesky",
]
