# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Load a theme's inputs from the portable three-file ThemeData format.

This is the **input-side** counterpart to :mod:`geosigma.themes.io` (which writes
the per-layer variance *stack* as GeoTIFFs). Where ``io`` is the handoff *out* of
the variance-map tool, this module is the handoff *in*: a self-describing,
GIS/spreadsheet-inspectable bundle of the point data one information-source theme
needs, decoupled from any particular raw data format (the Danish ``.mat`` files,
their column names, region selection). An *adapter* converts a raw source into
this format once; the format then loads back into a
:class:`~geosigma.themes.base.ThemeData` with no adapter in the loop.

The three-file format
---------------------
1. a **manifest** YAML — declares the theme name, the built-in ``spec`` it uses,
   the points CSV, the modelled-layer count, the coordinate columns, the per-layer
   / per-point attribute blocks, the complexity contract, and any data-derived
   ``spec_params`` (see below);
2. a **points CSV** — one row per data point, with ``x``/``y`` columns, per-layer
   attribute blocks stored as ``<prefix>LNN`` columns (e.g. ``depth__L01`` …
   ``depth__L43``), and per-point scalar columns;
3. an **optional complexity grid** — a separate raster (Surfer 7 / GeoTIFF / ESRI
   ASCII). It is *not* embedded in the CSV; see the complexity contract below.

Manifest schema (mapping ``ThemeData`` + spec resolution)::

    format_version: 1           # spec revision (optional; absent -> 1)
    name: REFSEIS               # human label
    spec: refseis               # a built-in factory in geosigma.themes.themes
    points_csv: refseis_points.csv
    n_layers: 43                # modelled-layer count (peat already excluded)
    crs: EPSG:25832             # optional, carried through for provenance
    x_column: x
    y_column: y
    attributes:
      thick:                    # per-layer block: reassemble thick__L01..L43
        per_layer: true
        prefix: thick__L
      doi:                      # per-point scalar: a single column
        per_layer: false
        column: doi
    complexity: external        # 'external' | 'none' | {path, order} (see below)
    spec_params:                # data-derived scalars the spec factory needs
      active_layers: [12, 13, ...]

The complexity contract (explicit — no silent default)
------------------------------------------------------
A theme's correlation range depends on a geological-complexity class grid. That
grid is a *separate* input and its absence is a real modelling choice, so the
manifest must state it **explicitly** — a missing / ``null`` field is an error,
not a silent fall-back to uniform complexity. The three valid declarations are:

* ``complexity: {path: <path>, order: <order>}`` — load the grid from disk.
  ``order`` is required and has no default; see *Class order* below. When a
  ``grid_meta`` is supplied to the loader the grid is required to be aligned
  with it (via :func:`utils.grids_aligned`); a misaligned grid raises.
* ``complexity: external`` — the grid is supplied at load time by the caller
  through the ``complexity=`` argument (e.g. a crop sampled onto the working
  grid). Omitting it raises rather than silently defaulting.
* ``complexity: none`` (or ``uniform``) — the explicit opt-in to *no* complexity
  map: :class:`ThemeData.complexity` stays ``None`` and the engine assumes a flat
  range (uniform class 1, the least complex class). This is the only way to get
  the old implicit default, and it is now a deliberate, recorded choice.

Class order (explicit — no default)
-----------------------------------
Inside GeoSigma a higher complexity class means a more complex geology: class
``0`` is unknown (masked) and ``1 .. n`` run from least to most complex. Every
``range_by_complexity`` table is indexed in that order. Complexity maps are not
all coded that way, so a map read from disk must say which way its values run:

* ``order: higher_is_more_complex`` — the values are used as they are.
* ``order: lower_is_more_complex`` — the values are reversed on loading,
  ``c -> n_classes + 1 - c`` for ``c`` in ``1 .. n_classes`` (``0`` stays ``0``).
  This needs ``n_classes`` as well, e.g.
  ``{path: map.tif, order: lower_is_more_complex, n_classes: 4}``: the map's own
  maximum is not used, because a crop need not contain the top class.

A grid supplied as an array (``complexity: external``, or
:class:`~geosigma.themes.base.ThemeData` built directly) is taken to be in the
internal order already; read it with :func:`read_complexity_grid` to apply a
declared order.

Spec resolution
---------------
The manifest names a built-in spec (``spec: refseis``) rather than embedding the
full formula, because the built-ins (:mod:`geosigma.themes.themes`) are Danish
*two-regime* specs parameterised by ``n_prereq`` — a model constant that is not
part of the portable data bundle. :func:`resolve_theme_spec` maps the name to its
factory and applies the caller's ``n_prereq`` plus the manifest's
``spec_params`` (the data-derived scalars the generic attribute mechanism cannot
express — REFSEIS ``active_layers``, MEP ``doi_fill``). A user with a fully
self-contained model instead authors a declarative spec with
:mod:`geosigma.themes.spec_io`, which needs no ``n_prereq``.

Model-agnostic boundary
-----------------------
The loader only ever produces plain arrays wrapped in a :class:`ThemeData`; it
holds no region names, paths, or column conventions of its own — those live in
the manifest a caller points it at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .base import ThemeData, ThemeSpec
from .themes import THEME_SPEC_FACTORIES

#: Manifest keys that must be present. ``complexity`` is deliberately *not* here:
#: its absence is handled by :func:`_resolve_complexity` so that a missing key
#: gets the same explicit "declare it as a path / 'external' / 'none'" guidance as
#: an explicit ``null`` — rather than the generic "missing required key" message.
_REQUIRED_KEYS = (
    "name",
    "spec",
    "points_csv",
    "n_layers",
    "x_column",
    "y_column",
    "attributes",
)

#: Explicit "no complexity grid" declarations (engine falls back to uniform).
_COMPLEXITY_NONE = ("none", "uniform")

#: Valid values of a complexity map's ``order``: which way its classes run. The
#: internal convention is ``higher_is_more_complex``; a map declared
#: ``lower_is_more_complex`` is reversed when it is read.
COMPLEXITY_ORDERS = ("higher_is_more_complex", "lower_is_more_complex")

#: Format revisions this loader understands. A manifest with no ``format_version``
#: is treated as :data:`_DEFAULT_FORMAT_VERSION` (the ten Danish local manifests
#: predate the key); an explicit unsupported value raises.
_SUPPORTED_FORMAT_VERSIONS = (1,)
_DEFAULT_FORMAT_VERSION = 1


@dataclass
class LoadedTheme:
    """The result of :func:`load_theme_data`.

    Attributes
    ----------
    data : ThemeData
        The point cloud + attributes (+ resolved complexity), ready for
        :func:`geosigma.themes.base.build_theme`.
    spec_name : str
        The built-in spec the manifest names (resolve with
        :func:`resolve_theme_spec`).
    n_layers : int
        Modelled-layer count declared by the manifest (peat already excluded).
    spec_params : dict
        Data-derived scalars the spec factory needs (e.g. ``active_layers``,
        ``doi_fill``); empty for most themes.
    name : str
        Human-readable theme label from the manifest.
    crs : Any
        The manifest's CRS string (or ``None``); carried through for provenance.
    """

    data: ThemeData
    spec_name: str
    n_layers: int
    spec_params: dict
    name: str
    crs: Any = None


def _load_manifest(manifest_path) -> dict:
    import yaml

    with open(manifest_path, "r", encoding="utf-8") as fh:
        man = yaml.safe_load(fh)
    if not isinstance(man, Mapping):
        raise ValueError(
            f"{manifest_path}: top-level manifest YAML must be a mapping, "
            f"got {type(man).__name__}"
        )
    missing = [k for k in _REQUIRED_KEYS if k not in man]
    if missing:
        raise ValueError(
            f"{manifest_path}: manifest is missing required key(s): {missing}"
        )
    version = man.get("format_version", _DEFAULT_FORMAT_VERSION)
    if version not in _SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"{manifest_path}: unsupported format_version {version!r}; this loader "
            f"supports {list(_SUPPORTED_FORMAT_VERSIONS)}. Omit the key to default "
            f"to version {_DEFAULT_FORMAT_VERSION}."
        )
    return dict(man)


def _check_label_width(cols, prefix, attr_name, manifest_path):
    """Enforce constant-width per-layer labels.

    The loader assigns a per-layer block's columns to layers in **lexicographic**
    (text) sort order and never parses the label integers. Zero-padded labels of
    constant width (``L01, L02, …, L43``) sort in layer order; mixed-width labels
    (``L1`` vs ``L10``) sort as ``L1, L10, L11, …, L2`` and would map columns to
    the **wrong** layers with no other symptom. Rather than trust the depositor to
    pad, verify it: the suffix after ``prefix`` must be the same character width
    for every matched column, which for the numeric ``LNN`` convention is exactly
    the condition "alphabetical order equals layer order". A single mismatch
    raises with the two offending columns named.
    """
    by_width = {}
    for c in cols:
        by_width.setdefault(len(c) - len(prefix), c)
    if len(by_width) > 1:
        wmin, wmax = min(by_width), max(by_width)
        raise ValueError(
            f"{manifest_path}: per-layer attribute {attr_name!r} has labels of "
            f"inconsistent width after prefix {prefix!r}: {by_width[wmin]!r} "
            f"(width {wmin}) and {by_width[wmax]!r} (width {wmax}). Per-layer "
            f"labels must be zero-padded to a constant width so alphabetical order "
            f"equals layer order (e.g. L01, L02, …, L43); otherwise columns are "
            f"silently assigned to the wrong layers."
        )


def _read_attributes(df, attributes, n_layers, manifest_path):
    """Reassemble the manifest's declared attributes from the CSV columns.

    Per-layer blocks become ``(npoints, n_layers)`` arrays from their prefixed
    columns; per-point scalars become ``(npoints,)`` arrays (NaNs preserved — any
    fill is the spec's downstream job). Layer-count and missing-column mismatches
    raise with a clear message.
    """
    out = {}
    for attr_name, block in attributes.items():
        if not isinstance(block, Mapping):
            raise ValueError(
                f"{manifest_path}: attribute {attr_name!r} must be a mapping with "
                f"either 'per_layer'+'prefix' or 'column', got "
                f"{type(block).__name__}"
            )
        if block.get("per_layer"):
            prefix = block.get("prefix")
            if not prefix:
                raise ValueError(
                    f"{manifest_path}: per-layer attribute {attr_name!r} needs a "
                    f"'prefix'"
                )
            cols = sorted(c for c in df.columns if c.startswith(prefix))
            if not cols:
                raise ValueError(
                    f"{manifest_path}: per-layer attribute {attr_name!r} matched no "
                    f"CSV columns with prefix {prefix!r}"
                )
            if len(cols) != n_layers:
                raise ValueError(
                    f"{manifest_path}: per-layer attribute {attr_name!r} has "
                    f"{len(cols)} column(s) matching prefix {prefix!r} but the "
                    f"manifest declares n_layers={n_layers}"
                )
            _check_label_width(cols, prefix, attr_name, manifest_path)
            out[attr_name] = df[cols].to_numpy(dtype=float)
        else:
            col = block.get("column")
            if not col:
                raise ValueError(
                    f"{manifest_path}: per-point attribute {attr_name!r} needs a "
                    f"'column' (or set per_layer: true with a 'prefix')"
                )
            if col not in df.columns:
                raise ValueError(
                    f"{manifest_path}: attribute {attr_name!r} references column "
                    f"{col!r}, absent from the points CSV"
                )
            out[attr_name] = df[col].to_numpy(dtype=float)
    return out


def _load_grid(path):
    """Load a complexity grid, dispatching on file extension. Returns (z, meta)."""
    from utils import (
        load_esri_ascii_grid,
        load_geotiff_grid,
        load_surfer7_grid,
    )

    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".grd":
        return load_surfer7_grid(str(path))
    if ext in (".tif", ".tiff"):
        return load_geotiff_grid(str(path))
    if ext in (".asc", ".txt"):
        return load_esri_ascii_grid(str(path))
    raise ValueError(
        f"unsupported complexity grid extension {ext!r} for {path!r}; expected "
        f".grd (Surfer 7), .tif/.tiff (GeoTIFF), or .asc/.txt (ESRI ASCII)"
    )


def orient_complexity(grid, order, n_classes=None):
    """Return a complexity class grid in the internal order (higher = more complex).

    Parameters
    ----------
    grid : array_like
        Complexity classes: ``0`` = unknown, ``1 .. n`` = classes.
    order : str
        One of :data:`COMPLEXITY_ORDERS`; how the values in ``grid`` run.
    n_classes : int, optional
        Number of classes. Required for ``"lower_is_more_complex"``, where class
        ``c`` becomes ``n_classes + 1 - c``; ignored otherwise.

    With ``"higher_is_more_complex"`` the grid is returned as a float array with
    its values unchanged.
    """
    if order not in COMPLEXITY_ORDERS:
        raise ValueError(
            f"complexity order {order!r} is not one of {list(COMPLEXITY_ORDERS)}"
        )
    grid = np.asarray(grid, dtype=float)
    if order == "higher_is_more_complex":
        return grid
    if n_classes is None:
        raise ValueError(
            "complexity order 'lower_is_more_complex' needs n_classes (the "
            "number of complexity classes) to reverse the classes"
        )
    if isinstance(n_classes, bool) or int(n_classes) != n_classes or n_classes < 1:
        raise ValueError(f"n_classes must be a positive integer, got {n_classes!r}")
    n = int(n_classes)
    valid = np.isin(grid, np.arange(n + 1))
    if not valid.all():
        bad = np.unique(grid[~valid])[:5].tolist()
        raise ValueError(
            f"complexity grid has values outside 0..{n} (e.g. {bad}); cannot "
            f"reverse it with n_classes={n}"
        )
    return np.where(grid > 0, n + 1 - grid, 0.0)


def read_complexity_grid(path, *, order, n_classes=None, grid_meta=None):
    """Read a complexity class grid from disk in the internal order.

    Parameters
    ----------
    path : str or path-like
        Surfer 7 (``.grd``), GeoTIFF (``.tif``/``.tiff``) or ESRI ASCII
        (``.asc``/``.txt``) raster.
    order : str
        Required, no default: one of :data:`COMPLEXITY_ORDERS`.
    n_classes : int, optional
        Required with ``order="lower_is_more_complex"``.
    grid_meta : dict, optional
        If given, the grid must be aligned with it (:func:`utils.grids_aligned`).

    Returns
    -------
    (grid, meta)
        ``grid`` as stored (north-up, row 0 = top) as a float array in which
        higher means more complex; ``meta`` is the raster's grid metadata.
    """
    grid, gmeta = _load_grid(path)
    if grid_meta is not None:
        from utils import grids_aligned

        if not grids_aligned(gmeta, grid_meta):
            raise ValueError(
                f"complexity grid {str(path)!r} is not aligned with the supplied "
                f"grid_meta (shape/spacing/origin differ)."
            )
    return orient_complexity(grid, order, n_classes), gmeta


def _resolve_complexity(man, manifest_path, complexity, grid_meta):
    """Apply the explicit complexity contract; return the array (or ``None``)."""
    decl = man.get("complexity")

    if decl is None:
        raise ValueError(
            f"{manifest_path}: complexity is unset (missing or null). Declare it "
            f"explicitly as a grid path ({{path: <grid>, order: <order>}}), "
            f"'external' (supplied via the complexity= argument), or 'none' "
            f"(uniform range — the deliberate no-grid choice)."
        )

    # Explicit "no grid" -> uniform range (ThemeData.complexity stays None).
    if isinstance(decl, str) and decl.strip().lower() in _COMPLEXITY_NONE:
        if complexity is not None:
            raise ValueError(
                f"{manifest_path}: manifest declares complexity {decl!r} (no grid) "
                f"but a complexity= array was supplied; remove one."
            )
        return None

    # Supplied at load time by the caller (already in the internal order).
    if isinstance(decl, str) and decl.strip().lower() == "external":
        if complexity is None:
            raise ValueError(
                f"{manifest_path}: manifest declares complexity 'external'; supply "
                f"the grid via the complexity= argument."
            )
        return np.asarray(complexity, dtype=float)

    # Otherwise a concrete grid path, which must declare its class order.
    if isinstance(decl, str):
        raise ValueError(
            f"{manifest_path}: complexity grid {decl!r} needs its class order. "
            f"Write it as complexity: {{path: {decl}, order: <order>}} with order "
            f"one of {list(COMPLEXITY_ORDERS)} (lower_is_more_complex also needs "
            f"n_classes)."
        )
    if not isinstance(decl, Mapping) or not isinstance(decl.get("path"), str):
        raise ValueError(
            f"{manifest_path}: cannot interpret complexity={decl!r}; expected "
            f"{{path: ..., order: ...}}, 'external', or 'none'."
        )
    unknown = set(decl) - {"path", "order", "n_classes"}
    if unknown:
        raise ValueError(
            f"{manifest_path}: unknown complexity key(s) {sorted(unknown)}; "
            f"expected path, order and (with lower_is_more_complex) n_classes."
        )
    path, order = decl["path"], decl.get("order")
    if order is None:
        raise ValueError(
            f"{manifest_path}: complexity grid {path!r} has no order. Add order: "
            f"one of {list(COMPLEXITY_ORDERS)} (lower_is_more_complex also needs "
            f"n_classes); there is no default."
        )
    if order not in COMPLEXITY_ORDERS:
        raise ValueError(
            f"{manifest_path}: complexity order {order!r} is not one of "
            f"{list(COMPLEXITY_ORDERS)}."
        )
    if complexity is not None:
        raise ValueError(
            f"{manifest_path}: manifest points complexity at a grid path but a "
            f"complexity= array was also supplied; remove one."
        )
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), path)
    try:
        grid, _ = read_complexity_grid(
            path, order=order, n_classes=decl.get("n_classes"), grid_meta=grid_meta
        )
    except ValueError as exc:
        raise ValueError(f"{manifest_path}: {exc}") from exc
    return grid


def load_theme_data(
    manifest_path,
    *,
    complexity: Optional[np.ndarray] = None,
    grid_meta: Optional[Mapping[str, Any]] = None,
) -> LoadedTheme:
    """Load a :class:`LoadedTheme` from the three-file ThemeData format.

    Parameters
    ----------
    manifest_path : str or path-like
        Path to the theme's manifest YAML. The points CSV (and any relative
        complexity grid path) are resolved relative to the manifest's directory.
    complexity : ndarray, optional
        The complexity class grid, required iff the manifest declares
        ``complexity: external``. It must already be in the internal order
        (higher = more complex; see :func:`read_complexity_grid`). An error is
        raised if it is supplied when the manifest instead names a grid path or
        ``none``.
    grid_meta : mapping, optional
        Grid metadata of the working grid. When the manifest references a
        complexity *path*, the loaded grid is checked against this via
        :func:`utils.grids_aligned`; a misaligned grid raises.

    Returns
    -------
    LoadedTheme
        Carries the :class:`ThemeData`, the named ``spec`` and its ``spec_params``
        (resolve with :func:`resolve_theme_spec`), the modelled-layer count, and
        provenance metadata.
    """
    man = _load_manifest(manifest_path)

    import pandas as pd

    csv_path = man["points_csv"]
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(
            os.path.dirname(os.path.abspath(manifest_path)), csv_path
        )
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"{manifest_path}: points CSV not found at {csv_path!r}"
        )
    df = pd.read_csv(csv_path)

    n_layers = int(man["n_layers"])
    if n_layers < 1:
        raise ValueError(f"{manifest_path}: n_layers must be >= 1, got {n_layers}")

    for axis in ("x_column", "y_column"):
        if man[axis] not in df.columns:
            raise ValueError(
                f"{manifest_path}: {axis} {man[axis]!r} is absent from the points "
                f"CSV (columns: {list(df.columns)[:8]}...)"
            )
    xs = df[man["x_column"]].to_numpy(dtype=float)
    ys = df[man["y_column"]].to_numpy(dtype=float)

    if not isinstance(man["attributes"], Mapping) or not man["attributes"]:
        raise ValueError(f"{manifest_path}: 'attributes' must be a non-empty mapping")
    attributes = _read_attributes(df, man["attributes"], n_layers, manifest_path)

    comp = _resolve_complexity(man, manifest_path, complexity, grid_meta)

    data = ThemeData(xs=xs, ys=ys, attributes=attributes, complexity=comp)
    return LoadedTheme(
        data=data,
        spec_name=str(man["spec"]),
        n_layers=n_layers,
        spec_params=dict(man.get("spec_params", {}) or {}),
        name=str(man["name"]),
        crs=man.get("crs"),
    )


def resolve_theme_spec(
    spec_name: str,
    n_prereq: int,
    spec_params: Optional[Mapping[str, Any]] = None,
) -> ThemeSpec:
    """Resolve a named built-in spec to a :class:`ThemeSpec`.

    Maps ``spec_name`` to its factory in :data:`geosigma.themes.themes
    .THEME_SPEC_FACTORIES` and calls it as ``factory(n_prereq, **spec_params)``.
    ``n_prereq`` is the (0-based) first pre-Quaternary modelled-layer index — the
    Danish two-regime split — supplied by the caller (it is a model constant, not
    part of the portable data bundle). ``spec_params`` carries the data-derived
    scalars a factory needs beyond ``n_prereq`` (REFSEIS ``active_layers``, MEP
    ``doi_fill``).

    A fully self-contained model that does not use the Danish two-regime built-ins
    should author a declarative spec with :mod:`geosigma.themes.spec_io` instead,
    which needs no ``n_prereq``.
    """
    if spec_name not in THEME_SPEC_FACTORIES:
        raise ValueError(
            f"unknown theme spec {spec_name!r}; available built-ins: "
            f"{sorted(THEME_SPEC_FACTORIES)}. For a self-contained model, author a "
            f"declarative spec via geosigma.themes.spec_io instead."
        )
    factory = THEME_SPEC_FACTORIES[spec_name]
    params = dict(spec_params or {})
    try:
        return factory(n_prereq, **params)
    except TypeError as exc:
        raise ValueError(
            f"spec {spec_name!r} rejected spec_params {sorted(params)}: {exc}"
        ) from exc
