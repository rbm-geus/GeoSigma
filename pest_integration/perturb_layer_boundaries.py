# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:30:02 2026

@author: RBM, GEUS
"""

import os
import glob
import shutil
from pathlib import Path

import rasterio
import numpy as np


def perturb_layer_boundaries(
    folder,
    k=1,
    method="shift",
    output_folder="perturbed",
    layers=None,
):
    """
    Perturb layer-boundary GeoTIFFs.

    Parameters
    ----------
    folder : str or Path
        Folder containing the base GeoTIFF layer boundaries.

    k : float or int, optional
        Perturbation parameter.

        method="shift":
            shift all subsurface layers downward by k.

        method="realization":
            use realization number k.

    method : str, optional
        Perturbation method.

        "shift"        : subtract k from all subsurface layers
        "realization"  : replace layers with realization k

    output_folder : str, optional
        Name of output subfolder.

    Notes
    -----
    top_surface.tif is always copied unchanged.
    """

    folder = Path(folder)

    out_dir = folder / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {method}")

    # ------------------------------------------------------------
    # Method: shift
    # ------------------------------------------------------------

    if method == "shift":

        tif_files = sorted(folder.glob("*.tif"))

        print(f"Found {len(tif_files)} GeoTIFF files")

        for fname in tif_files:

            name = fname.stem

            print(f"Processing: {fname.name}")

            with rasterio.open(fname) as src:

                z = src.read(1)
                profile = src.profile.copy()
                nodata = src.nodata

            # --------------------------------------------------------
            # Terrain unchanged, other layers shifted
            # --------------------------------------------------------

            if name.lower() == "top_surface":
                z_new = z.copy()

            else:

                z_new = z.astype(float).copy()

                if nodata is not None:
                    mask = z_new == nodata
                    z_new[~mask] -= k
                else:
                    z_new -= k

            # --------------------------------------------------------
            # Output
            # --------------------------------------------------------

            out_name = f"{name}_perturbed.tif"
            out_path = out_dir / out_name

            with rasterio.open(out_path, "w", **profile) as dst:

                dst.write(z_new, 1)

            print(f"  -> wrote {out_name}")

    # --------------------------------------------------------
    # Method: realization
    # --------------------------------------------------------

    elif method == "realization":

        realization_dir = folder / f"realization{k}"

        if not realization_dir.exists():
            raise FileNotFoundError(f"Realization folder not found: {realization_dir}")

        # clean output folder
        for f in out_dir.glob("*.tif"):
            f.unlink()

        # select files
        if layers is None:
            tif_files = list(realization_dir.glob("*.tif"))
        else:
            # ensure list for safe handling
            layers = set(layers)

            tif_files = [f for f in realization_dir.glob("*.tif") if f.name in layers]

        if not tif_files:
            raise ValueError("No matching layers found to copy.")

        print(f"Using realization {k}")
        print(f"Found {len(tif_files)} GeoTIFF files")
        print(f"Copying {len(tif_files)} GeoTIFF files")

        for tif_file in tif_files:
            shutil.copy2(tif_file, out_dir / tif_file.name)

        return

    else:

        raise ValueError(f"Unknown method '{method}'")

    print("Done.")
