# -*- coding: utf-8 -*-
#
# GeoSigma — original work.
# Author:     Rasmus Bødker Madsen (rbm@geus.dk)
# Co-authors: Frederik Falk; Claude (Anthropic)
"""Command-line entry point for the standalone variance-map tool.

Run as ``python -m geosigma.themes <command>``.

``combine``
    Combine several per-layer theme variance folders into a single final
    variance stack: parallel-precision combine, then floor by the
    depth-dependent minimum map from terrain + layer bottoms. This stage is
    fully model-agnostic (it only reads variance GeoTIFFs + surfaces), so it
    lives in the library CLI.

Building individual theme variance maps from raw geophysical data requires a
data *adapter* that knows the source schema (e.g. the Danish ``.mat`` files);
those adapters live in ``examples/`` and call
:func:`geosigma.themes.build_theme` directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import load_geotiff_grid
from utils.grid_meta import grids_aligned

from .combine import apply_floor, combine_variances, minimum_map
from .io import read_variance_stack, write_variance_stack


def _cmd_combine(args):
    theme_stacks = []
    ref_meta = None
    for theme_dir in args.theme:
        stack, meta = read_variance_stack(theme_dir, prefix=args.prefix)
        if ref_meta is None:
            ref_meta = meta
        elif not grids_aligned(meta, ref_meta):
            raise SystemExit(
                f"theme folder {theme_dir} is not grid-aligned with {args.theme[0]}"
            )
        theme_stacks.append(stack)

    combined = combine_variances(theme_stacks)

    terrain, t_meta = load_geotiff_grid(args.terrain)
    if not grids_aligned(t_meta, ref_meta):
        raise SystemExit("terrain raster is not grid-aligned with the theme grids")
    bottoms = []
    for bpath in args.bottom:
        b, b_meta = load_geotiff_grid(bpath)
        if not grids_aligned(b_meta, ref_meta):
            raise SystemExit(
                f"bottom raster {bpath} is not grid-aligned with the theme grids"
            )
        bottoms.append(b)
    if len(bottoms) != combined.shape[2]:
        raise SystemExit(
            f"got {len(bottoms)} bottom surfaces but "
            f"{combined.shape[2]} layers in the themes"
        )

    floored = apply_floor(combined, minimum_map(terrain, bottoms))
    paths = write_variance_stack(floored, ref_meta, args.out, prefix=args.out_prefix)
    print(f"Wrote {len(paths)} layer(s) to {Path(args.out).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m geosigma.themes")
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser(
        "combine", help="combine theme variance folders into a final stack"
    )
    c.add_argument(
        "--theme",
        action="append",
        required=True,
        metavar="DIR",
        help="a per-layer theme variance folder (repeatable)",
    )
    c.add_argument("--terrain", required=True, help="terrain / top-surface GeoTIFF")
    c.add_argument(
        "--bottom",
        action="append",
        required=True,
        metavar="TIF",
        help="layer bottom-surface GeoTIFF, in layer order (repeatable)",
    )
    c.add_argument(
        "--out", required=True, help="output folder for the final variance stack"
    )
    c.add_argument(
        "--prefix",
        default="variance",
        help="input theme file prefix (default: variance)",
    )
    c.add_argument(
        "--out-prefix",
        default="variance",
        help="output file prefix (default: variance)",
    )
    c.set_defaults(func=_cmd_combine)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
