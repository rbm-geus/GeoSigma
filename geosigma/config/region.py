# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Model/region configuration as data.

This module replaces the MATLAB ``landsdel_switch`` / ``get_layer_names``
switch statements (which hardcoded the Danish regions) with a single,
model-agnostic :class:`RegionConfig`. A configuration is *data* — built in code,
or loaded from a JSON/YAML file — and carries no region-specific constants in
library code. Danish (or any other) specifics belong in example/data files, not
here.

Layer model
-----------
``layer_names`` lists the ordered surfaces (interfaces) of the model from top to
bottom. By convention:

* index 0 is the terrain / top surface,
* the last entry is the model bottom (unless ``has_bottom=False``).

The *modelled* layer boundaries are the interfaces GeoSigma actually kriges.
Mirroring the MATLAB rule "there are ``Ninterfaces-2`` layers to be modelled
(minus terrain and minus bottom)", the default modelled set is every interface
except the terrain and the bottom. Region quirks that the MATLAB code hardcoded
(e.g. "in Jylland the chalk is not modelled", or Anholt/Læsø having no bottom)
are expressed here as *data* via ``modelled_layers`` / ``has_bottom`` rather than
as branches in code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class RegionConfig:
    """Configuration for one hydrostratigraphic model / region.

    Parameters
    ----------
    name : str
        Human-readable model/region name (free-form; no behaviour attached).
    layer_names : sequence of str
        Ordered interface/surface names, top to bottom. Index 0 is the terrain;
        the last entry is the bottom (unless ``has_bottom`` is False).
    modelled_layers : sequence of int, optional
        Explicit indices (into ``layer_names``) of the boundaries to krige. When
        ``None``, defaults to every interface except the terrain (index 0) and,
        if ``has_bottom``, the bottom (last index).
    has_bottom : bool, default True
        Whether the last entry of ``layer_names`` is a fixed model bottom that is
        not itself modelled. Set False for models with no bottom surface.
    well_files : mapping, optional
        Named references to well data files, e.g.
        ``{"interpreted": "...", "jupiter": "..."}``. Free-form; not interpreted
        here.
    surface_dir : str, optional
        Directory containing the surface grids; used by :meth:`surface_path`.
    file_ext : str, default ".tif"
        File extension for surface grids.
    crs : str, optional
        Coordinate reference system for the model (e.g. ``"EPSG:25832"``).
    metadata : mapping, optional
        Free-form extras (e.g. ``{"preq_index": 21}``). Never required by the
        library.
    """

    name: str
    layer_names: Sequence[str]
    modelled_layers: Sequence[int] | None = None
    has_bottom: bool = True
    well_files: Mapping[str, str] = field(default_factory=dict)
    surface_dir: str | None = None
    file_ext: str = ".tif"
    crs: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.layer_names = list(self.layer_names)
        if len(self.layer_names) < 2:
            raise ValueError(
                "layer_names must list at least a top and one boundary "
                f"(got {len(self.layer_names)})"
            )
        if len(set(self.layer_names)) != len(self.layer_names):
            raise ValueError("layer_names must be unique")

        if self.modelled_layers is not None:
            self.modelled_layers = list(self.modelled_layers)
            n = len(self.layer_names)
            for idx in self.modelled_layers:
                if not 0 <= idx < n:
                    raise ValueError(
                        f"modelled_layers index {idx} out of range [0, {n})"
                    )

        self.well_files = dict(self.well_files)
        self.metadata = dict(self.metadata)

    # -- derived quantities ------------------------------------------------
    @property
    def n_interfaces(self) -> int:
        """Number of surfaces/interfaces (``len(layer_names)``)."""
        return len(self.layer_names)

    @property
    def modelled_layer_indices(self) -> list[int]:
        """Resolved indices of the boundaries to krige.

        Returns ``modelled_layers`` if set; otherwise every interface except the
        terrain (0) and, when ``has_bottom``, the last index.
        """
        if self.modelled_layers is not None:
            return list(self.modelled_layers)
        last = self.n_interfaces - (1 if self.has_bottom else 0)
        return list(range(1, last))

    @property
    def n_modelled_layers(self) -> int:
        """Number of modelled boundaries."""
        return len(self.modelled_layer_indices)

    @property
    def modelled_layer_names(self) -> list[str]:
        """Names of the modelled boundaries, in order."""
        return [self.layer_names[i] for i in self.modelled_layer_indices]

    # -- paths -------------------------------------------------------------
    def surface_path(self, layer: int | str) -> Path:
        """Path to a surface grid, by interface index or name.

        Requires ``surface_dir``. The filename is ``<name><file_ext>``.
        """
        if self.surface_dir is None:
            raise ValueError("surface_dir is not set; cannot build a surface path")
        name = self.layer_names[layer] if isinstance(layer, int) else layer
        if not isinstance(layer, int) and name not in self.layer_names:
            raise KeyError(f"{name!r} is not in layer_names")
        return Path(self.surface_dir) / f"{name}{self.file_ext}"

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        """Return a plain-dict representation (JSON/YAML-serialisable)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegionConfig":
        """Build a :class:`RegionConfig` from a plain mapping.

        Unknown keys are rejected to catch typos early.
        """
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown RegionConfig keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_file(cls, path: str | Path) -> "RegionConfig":
        """Load a config from a ``.json``, ``.yaml`` or ``.yml`` file.

        YAML support requires ``pyyaml`` (present in ``environment.yml`` but not
        in the lightweight ``requirements.txt``); JSON works with the stdlib.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - dep-specific
                raise ImportError(
                    "PyYAML is required to read YAML config files; install it or "
                    "use a .json file."
                ) from exc
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ValueError(f"unsupported config extension: {suffix!r}")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Write the config to ``.json`` (stdlib) or ``.yaml``/``.yml`` (pyyaml)."""
        path = Path(path)
        suffix = path.suffix.lower()
        data = self.to_dict()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover - dep-specific
                raise ImportError(
                    "PyYAML is required to write YAML config files; install it or "
                    "use a .json file."
                ) from exc
            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        elif suffix == ".json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            raise ValueError(f"unsupported config extension: {suffix!r}")
