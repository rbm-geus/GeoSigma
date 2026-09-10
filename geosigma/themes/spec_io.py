# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Declarative (YAML/dict) configuration of a theme's uncertainty model.

A :class:`~geosigma.themes.base.ThemeSpec` is normally written in Python, but
its whole surface — search radius, certainty kernel, per-layer correlation
range, and the ``var0``/``kernel``/``mask`` formulae — is simple enough to be
expressed in a YAML file. This module is that front-end, so a practitioner
*without deep Python experience* can configure a theme for their own data type.

The mapping is one-to-one with :class:`ThemeSpec`::

    name: droneTEM
    search_radius: 4
    cert_fun: ILM_sep2023          # a name from certainty_function_definer
    combine: nearest               # only 'nearest' (the engine); see below
    attributes:                    # name -> tie reducer (mean|min|max|first)
      depth: mean
      thick: min
      doi:   mean
    range_model:                   # model-agnostic per-layer range config
      - layers: "L01-L04"          # 1-based label range (or 'all', or [0,1,2])
        range_by_complexity: [50, 300, 250, 150, 80]
    var0:   "(0.5 * 1.2 * thick)**2"     # expression over the attributes
    kernel: "maximum(depth, 60)"
    mask:   "doi < depth"                # cell -> NODATA where this is true
    active_layers: [0, 1, 2, 3]          # optional; layers this source informs

The ``range_model`` block is the model-agnostic replacement for the former
Danish ``n_prereq`` field (see :class:`~geosigma.themes.base.RangeGroup`): a
single group covering ``"all"`` layers means *no* complexity/range regime change;
the Danish case is two groups split at the Quaternary boundary; one group per
layer gives full per-layer control. Specifying groups by 1-based **label**
(``"L01-L13"``) rather than index sidesteps the 0-based/1-based ``n_prereq``
ambiguity entirely.

The expression mini-language
----------------------------
``var0``, ``kernel`` and ``mask`` are strings evaluated against the reduced
attribute grids (``depth``, ``thick``, ``doi``, …) — exactly the mapping the
Python callables receive. Only arithmetic, comparisons, boolean ``and``/``or``
and a small whitelist of numpy functions (:data:`_ALLOWED_FUNCS`) are permitted;
the strings are parsed with :mod:`ast` and any other construct (attribute access,
indexing, calls to non-whitelisted names, imports, …) is rejected, so loading a
spec never executes arbitrary code. New *certainty kernels* still belong in
``certainty_function_definer`` / :mod:`geosigma.themes.certainty_functions`, not
here (see ``CLAUDE.md``).

Not expressible in YAML
-----------------------
``post_reduce_fn`` (PACEP/MEP per-cell DOI fill) has no declarative form and is
intentionally omitted; themes needing it stay Python-authored.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping

import numpy as np

from .base import LayerSelector, RangeGroup, ThemeSpec
from .certainty_functions import available_certainty_functions

#: numpy functions callable from a spec expression. Names are deliberately the
#: bare numpy names the Python specs already use (``maximum`` not ``max``).
_ALLOWED_FUNCS = {
    "maximum": np.maximum,
    "minimum": np.minimum,
    "where": np.where,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "abs": np.abs,
    "clip": np.clip,
}

#: The combine modes the windowed engine implements. ``sum_n_nearest`` (summed
#: precision of the N nearest sources) is the *well* builder's behaviour and has
#: its own path (:func:`geosigma.themes.well.build_well_theme`), so it is not a
#: ThemeSpec/`build_theme` option.
_SUPPORTED_COMBINE = ("nearest",)


# ---------------------------------------------------------------------------
# Layer-label <-> selector
# ---------------------------------------------------------------------------

_LABEL_RANGE = re.compile(r"^L(\d+)\s*-\s*L(\d+)$", re.IGNORECASE)
_LABEL_OPEN = re.compile(r"^L(\d+)\s*-\s*end$", re.IGNORECASE)
_LABEL_ONE = re.compile(r"^L(\d+)$", re.IGNORECASE)


def parse_layers(value: Any) -> LayerSelector:
    """Parse a YAML ``layers:`` value into a :data:`LayerSelector`.

    Accepts ``"all"``; a 1-based label range ``"L01-L13"``; an open range
    ``"L14-end"``; a single label ``"L05"``; or an explicit list of **0-based**
    integer indices ``[0, 1, 2]`` (passed through unchanged). 1-based labels are
    converted to the engine's 0-based inclusive ``(start, stop)`` tuples.
    """
    if isinstance(value, str):
        s = value.strip()
        if s.lower() == "all":
            return "all"
        m = _LABEL_RANGE.match(s)
        if m:
            return (int(m.group(1)) - 1, int(m.group(2)) - 1)
        m = _LABEL_OPEN.match(s)
        if m:
            return (int(m.group(1)) - 1, None)
        m = _LABEL_ONE.match(s)
        if m:
            i = int(m.group(1)) - 1
            return (i, i)
        raise ValueError(
            f"cannot parse layer selector {value!r}; expected 'all', a label "
            f"range like 'L01-L13', an open range 'L14-end', a single 'L05', or "
            f"an explicit list of 0-based indices"
        )
    if isinstance(value, (list, tuple)):
        # Explicit 0-based index list. (A YAML list is always explicit; the
        # tuple-range form is internal and produced only by label parsing.)
        return [int(i) for i in value]
    raise TypeError(f"unsupported layers value {value!r} ({type(value).__name__})")


def format_layers(selector: LayerSelector) -> Any:
    """Inverse of :func:`parse_layers` — render a selector back to YAML form."""
    if isinstance(selector, str):
        return selector
    if isinstance(selector, tuple) and len(selector) == 2:
        start, stop = selector
        lo = 1 if start is None else int(start) + 1
        if stop is None:
            return f"L{lo:02d}-end"
        hi = int(stop) + 1
        return f"L{lo:02d}" if lo == hi else f"L{lo:02d}-L{hi:02d}"
    return [int(i) for i in selector]


# ---------------------------------------------------------------------------
# Safe expression mini-language
# ---------------------------------------------------------------------------

# AST node types permitted in a spec expression. Anything else (Attribute,
# Subscript, comprehensions, lambdas, walrus, starred, f-strings, …) is rejected.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    # operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Lt,
    ast.Gt,
    ast.LtE,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
)


class _ExprValidator(ast.NodeVisitor):
    """Reject any node outside the whitelist or any unknown name."""

    def __init__(self, allowed_names):
        self._allowed_names = set(allowed_names)

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"disallowed expression element: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError(f"only numeric constants are allowed, got {node.value!r}")

    def visit_Name(self, node):
        if node.id not in self._allowed_names:
            raise ValueError(
                f"unknown name {node.id!r} in expression; allowed names are "
                f"{sorted(self._allowed_names)}"
            )

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError(
                f"only calls to {sorted(_ALLOWED_FUNCS)} are allowed in expressions"
            )
        for arg in node.args:
            self.visit(arg)
        if node.keywords:
            raise ValueError("keyword arguments are not allowed in expressions")


def compile_expr(expr: str, attribute_names):
    """Compile a spec expression string into a callable ``fn(attrs) -> ndarray``.

    ``attrs`` is the mapping of reduced attribute grids the theme engine builds.
    The returned callable carries the original source on ``fn.expr`` so a spec
    can be serialised back to YAML losslessly.
    """
    expr = str(expr)
    allowed = set(attribute_names) | set(_ALLOWED_FUNCS)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression {expr!r}: {exc}") from exc
    _ExprValidator(allowed).visit(tree)
    code = compile(tree, "<themespec-expr>", "eval")

    def fn(attrs: Mapping[str, np.ndarray]):
        namespace = {**_ALLOWED_FUNCS, **dict(attrs)}
        return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 (sandboxed)

    fn.expr = expr
    return fn


# ---------------------------------------------------------------------------
# dict <-> ThemeSpec
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = (
    "name",
    "search_radius",
    "cert_fun",
    "attributes",
    "range_model",
    "var0",
    "kernel",
    "mask",
)


def load_spec(d: Mapping[str, Any]) -> ThemeSpec:
    """Build a :class:`ThemeSpec` from a plain mapping (already-parsed YAML)."""
    missing = [k for k in _REQUIRED_KEYS if k not in d]
    if missing:
        raise ValueError(f"theme spec is missing required key(s): {missing}")

    cert_fun = d["cert_fun"]
    if cert_fun not in available_certainty_functions():
        raise ValueError(
            f"unknown cert_fun {cert_fun!r}; available: "
            f"{available_certainty_functions()}. New kernels belong in "
            f"certainty_function_definer / themes.certainty_functions."
        )

    combine = d.get("combine", "nearest")
    if combine not in _SUPPORTED_COMBINE:
        raise ValueError(
            f"combine={combine!r} is not supported by build_theme "
            f"(supported: {list(_SUPPORTED_COMBINE)}). The summed-N-nearest mode "
            f"is the well builder (geosigma.themes.well)."
        )

    attributes = dict(d["attributes"])  # name -> reducer
    if not attributes:
        raise ValueError("theme spec needs at least one attribute")

    range_model = [
        RangeGroup(
            layers=parse_layers(g["layers"]),
            range_by_complexity=list(g["range_by_complexity"]),
        )
        for g in d["range_model"]
    ]

    attr_names = tuple(attributes)
    return ThemeSpec(
        name=str(d["name"]),
        search_radius=int(d["search_radius"]),
        range_model=range_model,
        cert_fun_name=cert_fun,
        attributes=attr_names,
        reducers=attributes,
        var0_fn=compile_expr(d["var0"], attr_names),
        kernel_fn=compile_expr(d["kernel"], attr_names),
        mask_fn=compile_expr(d["mask"], attr_names),
        active_layers=(
            [int(i) for i in d["active_layers"]]
            if d.get("active_layers") is not None
            else None
        ),
    )


def dump_spec(spec: ThemeSpec) -> dict:
    """Serialise a :class:`ThemeSpec` back to a plain (YAML-ready) dict.

    Only specs built through this module round-trip: the ``var0``/``kernel``/
    ``mask`` source is recovered from ``fn.expr``. Hand-written Python specs whose
    formulae are lambdas (the Danish built-ins) cannot be serialised — a clear
    error is raised rather than emitting an opaque ``<lambda>``.
    """

    def _expr(fn, what):
        src = getattr(fn, "expr", None)
        if src is None:
            raise ValueError(
                f"cannot serialise {spec.name!r}: its {what} formula is a Python "
                f"callable, not a spec_io expression. Only YAML/dict-authored "
                f"specs round-trip."
            )
        return src

    if spec.post_reduce_fn is not None:
        raise ValueError(
            f"cannot serialise {spec.name!r}: post_reduce_fn has no YAML form."
        )

    out: dict = {
        "name": spec.name,
        "search_radius": int(spec.search_radius),
        "cert_fun": spec.cert_fun_name,
        "combine": "nearest",
        "attributes": dict(spec.reducers),
        "range_model": [
            {
                "layers": format_layers(g.layers),
                "range_by_complexity": list(g.range_by_complexity),
            }
            for g in spec.range_model
        ],
        "var0": _expr(spec.var0_fn, "var0"),
        "kernel": _expr(spec.kernel_fn, "kernel"),
        "mask": _expr(spec.mask_fn, "mask"),
    }
    if spec.active_layers is not None:
        out["active_layers"] = [int(i) for i in spec.active_layers]
    return out


def load_spec_yaml(path) -> ThemeSpec:
    """Load a :class:`ThemeSpec` from a YAML file."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(data)}")
    return load_spec(data)


def dump_spec_yaml(spec: ThemeSpec, path=None) -> str:
    """Serialise a :class:`ThemeSpec` to YAML; write to ``path`` if given.

    Returns the YAML text either way.
    """
    import yaml

    text = yaml.safe_dump(dump_spec(spec), sort_keys=False)
    if path is not None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text
