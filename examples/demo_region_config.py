# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Demo: drive surface loading from a RegionConfig instead of hardcoded paths.

Loads the example region config for the bundled sample GeoTIFFs, then iterates
the modelled boundaries, loading each surface via ``RegionConfig.surface_path``
and reporting its grid metadata in the canonical contract.
"""

from pathlib import Path

from geosigma.config import RegionConfig
from utils import load_geotiff_grid, grids_aligned

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "examples" / "data" / "Jutland_hydrostrat_model" / "region_config.json"


def main() -> None:
    cfg = RegionConfig.from_file(CONFIG)
    print(f"Region: {cfg.name}")
    print(f"{cfg.n_interfaces} interfaces, {cfg.n_modelled_layers} modelled boundaries\n")

    # Reference grid = the terrain/top surface.
    _, ref_meta = load_geotiff_grid(str(cfg.surface_path(0)))

    for idx in cfg.modelled_layer_indices:
        name = cfg.layer_names[idx]
        z, meta = load_geotiff_grid(str(cfg.surface_path(name)))
        aligned = grids_aligned(meta, ref_meta)
        print(
            f"  [{idx:2d}] {name:16s} shape={z.shape} "
            f"dx={meta['dx']:.0f} aligned_to_top={aligned}"
        )


if __name__ == "__main__":
    main()
