# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""geosigma.themes — STANDALONE variance-map tool.

Builds per-layer variance grids from geophysical data sources and combines them
into the final per-layer variance map that the kriging pipeline consumes. It is
**independent of kriging**: the pipeline takes a variance-grid stack as an input,
whether produced here or supplied by the user.

Layout
------
* :mod:`~geosigma.themes.base` — the generic, model-agnostic engine
  (:class:`ThemeData`, :class:`ThemeSpec`, :func:`build_theme`,
  :func:`windowed_nearest`).
* :mod:`~geosigma.themes.themes` — the built-in theme specs (PACES, GAMMALOG,
  RESLOG, REFSEIS, SkyTEM, fewTEM, manyTEM, tTEM), transcribed verbatim from the
  MATLAB source.
* :mod:`~geosigma.themes.certainty_functions` — the named certainty-decay kernels.
* :mod:`~geosigma.themes.model_theme` — the depth-dependent model-area floor.
* :mod:`~geosigma.themes.combine` — parallel-precision combine, minimum-map floor,
  zero-thickness correlation map.
* :mod:`~geosigma.themes.io` — the handoff format (folder of per-layer GeoTIFFs).
* ``python -m geosigma.themes combine ...`` — CLI for the combine stage.

Model-agnostic boundary
------------------------
The engine and specs only ever see plain arrays. Loading raw geophysical data
(the Danish ``.mat`` files, their column names, region selection, and the
region-specific depth-of-investigation estimation) is the job of *adapters* that
live in ``examples/`` — never in this package.

Omitted: peatland / NPL (intentionally, not lost)
-------------------------------------------------
The MATLAB themes carried a peat-layer subsystem. Two pieces touched the theme
construction and are **deliberately dropped** here (out of scope per
``CLAUDE.md``):

1. the ``NPL`` layer-origin offset and the ``include_peatlands`` branch that
   *prepended* ``NPL`` constant-variance layers to each theme's output;
2. ``Main.m``'s ``+0.0625`` adjustment to the peat layers before flooring.

Neither is statistical: the engine works purely in modelled-layer space, and
re-adding peat would be an isolated wrapper that prepends constant-variance
layers around the engine output — no engine change required. The peat *content*
itself (``get_PL_themes`` / ``get_PL_themes_ILM``) is excluded entirely.
"""

from .base import (
    NODATA_VARIANCE,
    RangeGroup,
    ThemeData,
    ThemeSpec,
    build_theme,
    windowed_nearest,
)
from .certainty_functions import (
    available_certainty_functions,
    certainty_function,
    register_certainty_function,
)
from .spec_io import (
    compile_expr,
    dump_spec,
    dump_spec_yaml,
    load_spec,
    load_spec_yaml,
)
from .combine import apply_floor, combine_variances, corr_map, minimum_map
from .model_theme import model_theme
from .well import WellSet, build_well_theme
from .io import read_variance_stack, write_variance_stack
from .data_io import LoadedTheme, load_theme_data, resolve_theme_spec
from . import themes

__all__ = [
    "NODATA_VARIANCE",
    "ThemeData",
    "ThemeSpec",
    "RangeGroup",
    "build_theme",
    "windowed_nearest",
    "load_spec",
    "load_spec_yaml",
    "dump_spec",
    "dump_spec_yaml",
    "compile_expr",
    "WellSet",
    "build_well_theme",
    "certainty_function",
    "available_certainty_functions",
    "register_certainty_function",
    "combine_variances",
    "minimum_map",
    "apply_floor",
    "corr_map",
    "model_theme",
    "read_variance_stack",
    "write_variance_stack",
    "LoadedTheme",
    "load_theme_data",
    "resolve_theme_spec",
    "themes",
]
