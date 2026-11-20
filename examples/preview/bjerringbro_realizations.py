# -*- coding: utf-8 -*-
"""
Experimental realizations for the Bjerringbro example (v1.0.0 preview)
======================================================================

Runs the Bjerringbro variance-map workflow
(``examples/bjerringbro/geosigma_dk_workflow.py``) and then draws a small
ensemble of layer-model realizations from the resulting variance maps:

    realization = deterministic surface + spatially correlated Gaussian noise
                  scaled cell by cell by sqrt(variance map)

followed by stratigraphic ordering (no boundary above the one above it) and
preservation of zero-thickness layers.

This is a simplified stand-in built from GeoSigma's core primitives
(``precal_cov`` + ``get_reals_cholesky``). It is NOT the GEUS method (local
kriging conditioned on INPOX points, see ``demo_kriging_pipeline_end_to_end.py``)
and has not been validated. The correlation range (500 m) is a choice, not
estimated from data.

    python examples/preview/bjerringbro_realizations.py

Writes ``examples/output/bjerringbro_realizations/real_NNN/<layer>.tif`` and
``ensemble_std.npy``, plus the variance maps in the example's own ``output/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from geosigma.core.get_reals_cholesky import get_reals_cholesky
from geosigma.core.precal_cov import precal_cov

HERE = Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent / "bjerringbro"
sys.path.insert(0, str(EXAMPLE_DIR))

import geosigma_dk_workflow as workflow  # noqa: E402

REALIZATIONS = {
    "n_reals": 10,
    "corr_range_m": 500.0,  # lateral correlation range of the error (m)
    "max_sim_cells": 2500,  # Cholesky is O(n^3): simulate coarse, then upsample
    "seed": 42,
    "out_dir": HERE.parent / "output" / "bjerringbro_realizations",
}


def _bilinear_upsample(coarse, cx, cy, fx, fy):
    """Bilinear interpolation of coarse (ny, nx, n) onto fine axes fx, fy."""

    def weights(c, f):
        i = np.clip(np.searchsorted(c, f) - 1, 0, c.size - 2)
        w = np.clip((f - c[i]) / (c[i + 1] - c[i]), 0.0, 1.0)
        return i, w

    ix, wx = weights(cx, fx)
    iy, wy = weights(cy, fy)
    a = coarse[iy][:, ix]
    b = coarse[iy][:, ix + 1]
    c = coarse[iy + 1][:, ix]
    d = coarse[iy + 1][:, ix + 1]
    wx = wx[None, :, None]
    wy = wy[:, None, None]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def simulate_realizations(final_var, terrain, bottom_stack, grid_x, grid_y, rconf):
    np.random.seed(rconf["seed"])  # get_reals_cholesky uses np.random
    ny, nx, n_layers = final_var.shape

    # Coarse simulation grid so the Cholesky stays tractable.
    step = max(1, int(np.ceil(np.sqrt(nx * ny / rconf["max_sim_cells"]))))
    cx = np.append(grid_x[::step], grid_x[-1]) if (nx - 1) % step else grid_x[::step]
    cy = np.append(grid_y[::step], grid_y[-1]) if (ny - 1) % step else grid_y[::step]
    cxx, cyy = np.meshgrid(cx, cy)
    pos = np.column_stack([cxx.ravel(), cyy.ravel()])
    print(
        f"[R] Simulating on {cy.size} x {cx.size} coarse grid (step {step} cells), "
        f"range {rconf['corr_range_m']:.0f} m, {rconf['n_reals']} realizations"
    )

    # Unit-sill Gaussian correlation, factorised once and reused for every layer.
    corr, _ = precal_cov(pos, pos, f"1 Gau({rconf['corr_range_m']})")
    # Standard-normal correlated fields: (n_coarse, n_layers * n_reals)
    z = get_reals_cholesky(corr, Nreals=n_layers * rconf["n_reals"])
    z = z.reshape(cy.size, cx.size, n_layers, rconf["n_reals"])

    std = np.sqrt(np.where(np.isfinite(final_var), final_var, 0.0))
    upper0 = np.concatenate([terrain[:, :, None], bottom_stack[:, :, :-1]], axis=-1)
    zero_thick = (upper0 - bottom_stack) <= 1e-6

    reals = np.empty((rconf["n_reals"], ny, nx, n_layers))
    for r in range(rconf["n_reals"]):
        noise = _bilinear_upsample(z[:, :, :, r], cx, cy, grid_x, grid_y)
        surf = bottom_stack + noise * std
        # Stratigraphic order: no boundary above terrain or the one above it.
        # Layers absent in the model (zero thickness) stay absent.
        prev = terrain.copy()
        for s in range(n_layers):
            cur = np.minimum(surf[:, :, s], prev)
            cur = np.where(zero_thick[:, :, s], prev, cur)
            surf[:, :, s] = cur
            prev = cur
        reals[r] = surf
    return reals


def export_realizations(reals, meta, cfg, out_dir):
    import rasterio
    from rasterio.transform import from_origin

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_reals, ny, nx, n_layers = reals.shape
    transform = from_origin(
        meta["xllcorner"], meta["yllcorner"] + ny * meta["dy"], meta["dx"], meta["dy"]
    )
    names = cfg.modelled_layer_names
    for r in range(n_reals):
        rdir = out_dir / f"real_{r + 1:03d}"
        rdir.mkdir(exist_ok=True)
        for s in range(n_layers):
            data = np.flipud(reals[r, :, :, s])  # back to north-up
            with rasterio.open(
                rdir / f"{names[s]}.tif",
                "w",
                driver="GTiff",
                height=ny,
                width=nx,
                count=1,
                dtype="float32",
                crs=meta.get("crs"),
                transform=transform,
                nodata=-9999.0,
            ) as ds:
                ds.write(np.where(np.isfinite(data), data, -9999.0).astype("float32"), 1)
    # Ensemble statistics: an easy sanity check against the variance maps.
    np.save(out_dir / "ensemble_std.npy", np.nanstd(reals, axis=0))
    print(f"[R] Wrote {n_reals} realizations x {n_layers} surfaces to {out_dir}")


def main():
    conf = workflow.resolve_paths(workflow.CONFIG)
    final = workflow.run(conf)
    cfg, meta, grid_x, grid_y, terrain, bottom_stack = workflow.load_model(
        conf["region_config"]
    )
    reals = simulate_realizations(
        final, terrain, bottom_stack, grid_x, grid_y, REALIZATIONS
    )
    export_realizations(reals, meta, cfg, REALIZATIONS["out_dir"])
    ens = np.nanmedian(np.nanstd(reals, axis=0), axis=(0, 1))
    print("    ensemble std per layer [m]: " + " ".join(f"{s:.1f}" for s in ens))


if __name__ == "__main__":
    main()
