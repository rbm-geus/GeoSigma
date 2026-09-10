# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""The built-in geophysical theme specifications.

Each function returns a :class:`~geosigma.themes.base.ThemeSpec` carrying one
theme's parameters and its ``var0`` / ``kernel`` / mask formulae, transcribed
**verbatim** from the corresponding root-level ``get_*_theme.py`` (themselves
direct MATLAB translations). The generic engine in
:mod:`geosigma.themes.base` consumes these specs; the Danish ``.mat`` loading
that feeds them lives in adapters outside the library (see ``examples/``).

Layer indexing here is modelled-layer space (``0 .. n_layers-1``); the peat
offset ``NPL`` is dropped (see :mod:`geosigma.themes.base`). ``n_prereq`` is the
modelled-layer index from which the "prerequisite-layer" range override applies
(the MATLAB ``Npreq - NPL``).

The DOI ("depth of investigation") values are supplied by the adapter already
resolved: PACES overwrites DOI to a constant 18 m, and the TEM themes estimate
missing DOI from region-specific Palaeogene marker layers — that estimation is
data preparation and belongs in the adapter, not in these model-agnostic specs.
"""

from __future__ import annotations

import numpy as np

from .base import RangeGroup, ThemeSpec

# Complexity-class -> range tables shared by most themes. Index 0 is the
# "unknown" slot (never used: complexity 0 is masked to NaN range); indices
# 1..4 are complexity classes low..high.
_COMP2RANGE = [100, 500, 400, 250, 100]
_COMP2RANGE_PREQ = [500, 500, 500, 500, 500]

# tTEM is the exception: its MATLAB source (``get_tTEM_theme.m``) uses a *flat*
# range table — every complexity class (and the prerequisite override) maps to a
# 100 m range, not the graded table above. Kept distinct so tTEM stays
# bit-faithful to the MATLAB output.
_COMP2RANGE_TTEM = [100, 100, 100, 100, 100]
_COMP2RANGE_PREQ_TTEM = [100, 100, 100, 100, 100]

# The borehole-LOG themes (GAMMALOG/RESLOG) effectively have NO pre-Quaternary
# range override: in their MATLAB source the override is a dormant bug. Unlike
# fewTEM/PACES (which set ``rangemap = complexities``, a 3D array), the log themes
# set ``rangemap = complexity`` — a *2D* array — so the line
# ``rangemap(:,:,Npreq:end) = ...`` indexes a singleton 3rd dimension and is a
# no-op; the 2D graded ``comp2range`` is then broadcast across every layer. We
# reproduce this bit-for-bit by giving the pre-Q table the same graded values as
# ``comp2range`` (so the engine's override, if it fires, changes nothing).
_COMP2RANGE_PREQ_LOG = _COMP2RANGE


def _two_regime(n_prereq, base_table, preq_table):
    """Danish two-regime ``range_model``: ``base_table`` for the Quaternary
    layers ``0 .. n_prereq-1``, ``preq_table`` from ``n_prereq`` onward.

    ``n_prereq`` is the (0-based) modelled-layer index of the first
    pre-Quaternary layer (the MATLAB ``Npreq - NPL``). It stays a parameter of
    these *Danish* built-in specs — it is no longer a field of the model-agnostic
    :class:`~geosigma.themes.base.ThemeSpec`, which sees only the resulting
    groups. This reproduces the old ``rangemap(:,:,n_prereq:end) = preq`` logic
    exactly (an empty group when ``n_prereq`` falls outside the layer range).
    """
    return [
        RangeGroup(layers=(0, n_prereq - 1), range_by_complexity=base_table),
        RangeGroup(layers=(n_prereq, None), range_by_complexity=preq_table),
    ]


def paces_spec(n_prereq: int) -> ThemeSpec:
    """PACES theme (search radius 1 cell; FRAFA plateau kernel)."""
    return ThemeSpec(
        name="PACES",
        search_radius=1,
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ),
        cert_fun_name="FRAFA_apr2023",
        attributes=("depth",),
        reducers={"depth": "mean"},
        # var0 = (0.5 * max(2, 0.25*depth))^2 ; kernel = 75 ; DOI fixed at 18.
        var0_fn=lambda a: (0.5 * np.maximum(2.0, 0.25 * a["depth"])) ** 2,
        kernel_fn=lambda a: np.full_like(a["depth"], 75.0),
        mask_fn=lambda a: 18.0 < a["depth"],
    )


def pacep_spec(n_prereq: int, cert_fun_name: str = "ILM_sep2023") -> ThemeSpec:
    """PACEP theme (search radius 1 cell; plateau half-width 75).

    Like PACES, ``get_PACEP_theme.m`` takes the run's ``cert_fun_choice``
    (defaulting to ``FRAFA_apr2023``); the Jylland reference run used
    ``ILM_sep2023`` (the same global choice that SkyTEM/fewTEM/manyTEM validated
    under). ``var0 = (0.5*max(5, 0.5*depth))^2`` and missing DOI is filled
    **per-cell after reduction** with a constant 15 (see ``post_reduce_fn``),
    then cells whose boundary lies below the DOI are masked.
    """
    return ThemeSpec(
        name="PACEP",
        search_radius=1,
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ),
        cert_fun_name=cert_fun_name,
        attributes=("depth", "doi"),
        reducers={"depth": "mean", "doi": "mean"},
        var0_fn=lambda a: (0.5 * np.maximum(5.0, 0.5 * a["depth"])) ** 2,
        kernel_fn=lambda a: np.full_like(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
        post_reduce_fn=_pacep_post_reduce,
    )


def _pacep_post_reduce(a):
    # MATLAB: local_doi(isnan(local_doi)) = 15  (per-cell, after the window pass).
    a["doi"] = np.where(np.isnan(a["doi"]), 15.0, a["doi"])
    return a


def mep_spec(
    n_prereq: int, doi_fill: float, cert_fun_name: str = "ILM_sep2023"
) -> ThemeSpec:
    """MEP (multi-electrode profiling) theme (search radius 2; plateau width 75).

    Two MEP-specific wrinkles, both transcribed from ``get_MEP_theme.m``:

    * **Acquisition-type-dependent ``var0``.** Each data point is classified
      ``wenner_2d`` (datasubtype starting ``"wen"``) or not. ``var0`` uses
      ``0.5*max(2.4, 0.25*depth)`` for wenner-2D points and
      ``0.5*max(1.25, 0.15*depth)`` otherwise. The type is selected by the
      ``first`` reducer — the first data point in each cell's window, **not** the
      nearest — exactly as the MATLAB ``local_MEP_type = Type(filt)(1)``.
    * **DOI fill.** Missing DOI is filled per-cell after reduction with
      ``doi_fill`` — the mean of the source's non-NaN ``doilower`` values, which
      the adapter computes from the (filtered) point set and passes in.

    Like PACES/PACEP, the MATLAB default ``cert_fun_choice`` is ``FRAFA_apr2023``
    but the Jylland reference run used ``ILM_sep2023``.
    """
    return ThemeSpec(
        name="MEP",
        search_radius=2,
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ),
        cert_fun_name=cert_fun_name,
        attributes=("depth", "doi", "mep_type"),
        reducers={"depth": "mean", "doi": "mean", "mep_type": "first"},
        var0_fn=_mep_var0,
        kernel_fn=lambda a: np.full_like(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
        post_reduce_fn=_make_mep_post_reduce(doi_fill),
    )


def _mep_var0(a):
    # var0 = (wenner_var0 * is_wenner + other_var0 * is_other)^2
    # is_wenner is 1 / 0 / NaN (no point); the NaN propagates to NODATA downstream.
    depth = a["depth"]
    is_wenner = a["mep_type"]
    is_other = is_wenner == 0
    wenner_var0 = 0.5 * np.maximum(2.4, 0.25 * depth)
    other_var0 = 0.5 * np.maximum(1.25, 0.15 * depth)
    return (wenner_var0 * is_wenner + other_var0 * is_other) ** 2


def _make_mep_post_reduce(doi_fill):
    def _post(a):
        # MATLAB: local_doi(isnan(local_doi)) = mean(DOIs(~isnan(DOIs)))
        a["doi"] = np.where(np.isnan(a["doi"]), doi_fill, a["doi"])
        return a

    return _post


def gammalog_spec(n_prereq: int) -> ThemeSpec:
    """Gamma-log theme (search radius 6; ILM Gaussian kernel; min thickness)."""
    return ThemeSpec(
        name="GAMMALOG",
        search_radius=6,
        # no pre-Q override (MATLAB no-op): both regimes share the graded table.
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ_LOG),
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick"),
        reducers={"depth": "mean", "thick": "min"},
        var0_fn=_gammalog_var0,
        kernel_fn=lambda a: np.full_like(a["depth"], 150.0),
        mask_fn=lambda a: a["thick"] < a["depth"],
    )


def _gammalog_var0(a):
    # var0 = (0.5*3)^2 = 2.25 everywhere; sentinel 100000 where depth < 3.
    var0 = (0.5 * 3.0 + a["depth"] * 0.0) ** 2
    var0[a["depth"] < 3.0] = 100000.0
    return var0


def reslog_spec(n_prereq: int) -> ThemeSpec:
    """Resistivity-log theme (search radius 6; ILM Gaussian kernel; min thickness)."""
    return ThemeSpec(
        name="RESLOG",
        search_radius=6,
        # no pre-Q override (MATLAB no-op): both regimes share the graded table.
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ_LOG),
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick"),
        reducers={"depth": "mean", "thick": "min"},
        var0_fn=_reslog_var0,
        kernel_fn=lambda a: np.full_like(a["depth"], 150.0),
        mask_fn=lambda a: a["thick"] < a["depth"],
    )


def _reslog_var0(a):
    # var0 = 0.25 + (0.5*0.09*depth)^2 ; sentinel 100000 where depth < 10.
    var0 = 0.25 + (0.5 * 0.09 * a["depth"]) ** 2
    var0[a["depth"] < 10.0] = 100000.0
    return var0


def refseis_spec(n_prereq: int, active_layers) -> ThemeSpec:
    """Reflection-seismic theme (search radius 8; FRAFA plateau; thickest layers).

    ``active_layers`` are the modelled-layer indices the seismic constrains (the
    MATLAB ``layvec``: layers whose geophysical thickness exceeds 1000 m). All
    other layers receive the NODATA variance.
    """
    return ThemeSpec(
        name="REFSEIS",
        search_radius=8,
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ),
        cert_fun_name="FRAFA_apr2023",
        attributes=("thick",),
        reducers={"thick": "max"},
        # var0 = (0.5*15)^2 = 56.25 constant ; kernel = 1 ; mask thin layers.
        var0_fn=lambda a: np.full_like(a["thick"], (0.5 * 15.0) ** 2),
        kernel_fn=lambda a: np.full_like(a["thick"], 1.0),
        mask_fn=lambda a: a["thick"] < 100.0,
        active_layers=active_layers,
    )


def skytem_spec(n_prereq: int, cert_fun_name: str = "ILM_sep2023") -> ThemeSpec:
    """Airborne TEM (SkyTEM) theme (search radius 6).

    Unlike fewTEM/manyTEM/REFSEIS (which hardcode their kernel), the MATLAB
    ``get_SkyTEM_theme.m`` takes the run's ``cert_fun_choice`` (defaulting to
    ``FRAFA_apr2023``); the Jylland reference run used ``ILM_sep2023``. ``var0`` is
    driven by the **minimum** model thickness over the nearest-point ties (not the
    mean), and the plateau half-width is ``max(2*depth, 75)``.
    """
    return ThemeSpec(
        name="SkyTEM",
        search_radius=6,
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ),
        cert_fun_name=cert_fun_name,
        attributes=("depth", "thick", "doi"),
        reducers={"depth": "mean", "thick": "min", "doi": "mean"},
        var0_fn=_skytem_var0,
        kernel_fn=lambda a: np.maximum(2.0 * a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
    )


def _skytem_var0(a):
    # var0 = (0.5*1.2*thick_min)^2 ; sentinel 100000 where depth < 5.
    var0 = (0.5 * 1.2 * a["thick"]) ** 2
    var0[a["depth"] < 5.0] = 100000.0
    return var0


def fewtem_spec(n_prereq: int) -> ThemeSpec:
    """Few-layer TEM theme (search radius 6; ILM Gaussian kernel)."""
    return ThemeSpec(
        name="fewTEM",
        search_radius=6,
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ),
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "doi"),
        reducers={"depth": "mean", "doi": "mean"},
        var0_fn=_fewtem_var0,
        kernel_fn=lambda a: np.maximum(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
    )


def _fewtem_var0(a):
    # var0 = (max(1.2, 0.5*0.15*depth) + 2)^2 ; sentinel 10000 where depth < 10.
    var0 = (np.maximum(1.2, 0.5 * 0.15 * a["depth"]) + 2.0) ** 2
    var0[a["depth"] < 10.0] = 10000.0
    return var0


def manytem_spec(n_prereq: int) -> ThemeSpec:
    """Many-layer TEM theme (search radius 6; ILM Gaussian kernel)."""
    return ThemeSpec(
        name="manyTEM",
        search_radius=6,
        range_model=_two_regime(n_prereq, _COMP2RANGE, _COMP2RANGE_PREQ),
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick", "doi"),
        reducers={"depth": "mean", "thick": "mean", "doi": "mean"},
        var0_fn=_manytem_var0,
        kernel_fn=lambda a: np.maximum(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
    )


def _manytem_var0(a):
    # var0 = (0.5*1.2*thick)^2 ; sentinel 100000 where depth < 7.
    var0 = (0.5 * 1.2 * a["thick"]) ** 2
    var0[a["depth"] < 7.0] = 100000.0
    return var0


def ttem_spec(n_prereq: int) -> ThemeSpec:
    """Towed TEM (tTEM) theme (search radius 6; ILM Gaussian kernel)."""
    return ThemeSpec(
        name="tTEM",
        search_radius=6,
        range_model=_two_regime(n_prereq, _COMP2RANGE_TTEM, _COMP2RANGE_PREQ_TTEM),
        cert_fun_name="ILM_sep2023",
        attributes=("depth", "thick", "doi"),
        reducers={"depth": "mean", "thick": "mean", "doi": "mean"},
        var0_fn=_ttem_var0,
        kernel_fn=lambda a: np.maximum(a["depth"], 75.0),
        mask_fn=lambda a: a["doi"] < a["depth"],
    )


def _ttem_var0(a):
    # var0 = (0.5*1.2*thick)^2 ; sentinel 10000 where depth < 2.
    var0 = (0.5 * 1.2 * a["thick"]) ** 2
    var0[a["depth"] < 2.0] = 10000.0
    return var0


#: The built-in theme specs, keyed by the name a manifest uses in ``spec:``.
#: Every factory has the signature ``factory(n_prereq, **spec_params)`` so a
#: caller can resolve any of them uniformly (see
#: :func:`geosigma.themes.data_io.resolve_theme_spec`): the Danish two-regime
#: ``n_prereq`` is positional; theme-specific data-derived scalars (REFSEIS
#: ``active_layers``, MEP ``doi_fill``) arrive as keyword ``spec_params``.
THEME_SPEC_FACTORIES = {
    "paces": paces_spec,
    "pacep": pacep_spec,
    "mep": mep_spec,
    "gammalog": gammalog_spec,
    "reslog": reslog_spec,
    "refseis": refseis_spec,
    "skytem": skytem_spec,
    "fewtem": fewtem_spec,
    "manytem": manytem_spec,
    "ttem": ttem_spec,
}


def available_theme_specs():
    """Return the sorted names of the built-in theme specs."""
    return sorted(THEME_SPEC_FACTORIES)
