# -*- coding: utf-8 -*-
"""
Build wells.csv (the borehole input of geosigma_dk_workflow.py) the way GEUS did
================================================================================

GEUS' well theme (MATLAB get_well_theme.m / prepare_wells.m) uses two sets:

1. SNAPPED wells  ("snap")  — the model's interpretation points ("tolkningspunkter"):
   wells where the modeller picked a boundary and snapped the surface to it.
   Per point: x, y, which boundary, the picked elevation, quality 1-4.
   Typical source: your GeoScene3D model (interpretation points / "snap points")
   exported to CSV.                                              >>> YOUR DATA

2. JUPITER wells  ("jupiter") — every other borehole in Jupiter. A borehole counts
   for a boundary only if it is deep enough to REACH that boundary, and only if
   it was drilled no later than the model was made. Its boundary elevation is the
   model's own surface at the well (the well supports the model there).
   Source: Jupiter BOREHOLE table exported to CSV (any Jupiter download:
   PCJupiterXL, GeoScene3D, QGIS ...).                           >>> YOUR DATA

Wells already in the snapped set are not counted again as Jupiter wells.

Quality 1-4 (1 = best): the classic Danish model classes good / medium / poor /
very poor. Snapped points normally carry it from the model project. For Jupiter
wells use a quality column if you have one (model database, GeoScene3D); otherwise
JUPITER_QUALITY_RULE below assigns one — it is a PLACEHOLDER, edit it.

Usage
-----
    python wells_to_csv.py --region-config region_config.json \
        --jupiter boringer/jupiter.csv \
        --out boringer/wells.csv

Then set CONFIG["wells_csv"] = "boringer/wells.csv" in geosigma_dk_workflow.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# >>> YOUR DATA: column names in your files -> names used here
# ------------------------------------------------------------------------------
JUPITER_COLUMNS = {           # Jupiter BOREHOLE table (standard Jupiter names)
    "BOREHOLENO": "well_id",
    "XUTM": "x",
    "YUTM": "y",
    "ELEVATION": "elevation",         # m DVR90 at the well top (may be empty)
    "DRILLDEPTH": "drilldepth",       # m below the well top
    "DRILLENDDATE": "drilldate",      # date or year; used for --model-year
    # "QUALITY": "quality",           # uncomment if your export has a 1-4 quality
}
SNAPPED_COLUMNS = {           # interpretation points exported from the model project
    "WELL_ID": "well_id",             # optional (DGU no.)
    "X": "x",
    "Y": "y",
    "LAYER": "layer",                 # boundary: 1-based number OR surface name from region_config
    "Z": "z",                         # picked boundary elevation, m DVR90
    "QUALITY": "quality",             # 1-4
}


def JUPITER_QUALITY_RULE(df):
    """PLACEHOLDER quality 1-4 for Jupiter wells without a quality column.

    Rationale: newer wells are on average better described. Replace with your own
    classification (e.g. from the model database) when you have one.
    """
    year = df["year"]
    q = pd.Series(4, index=df.index)          # unknown year -> very poor
    q[year >= 1950] = 3
    q[year >= 1980] = 2                       # never 1: reserve 'good' for checked/GEUS-described wells
    return q.astype(int)


# ------------------------------------------------------------------------------
def _read(path, colmap):
    df = pd.read_csv(path, sep=None, engine="python")
    up = {c.upper(): c for c in df.columns}
    ren = {up[k.upper()]: v for k, v in colmap.items() if k.upper() in up}
    df = df.rename(columns=ren)
    return df


def _surfaces(region_config):
    sys.path.insert(0, str(Path(__file__).parent))
    from gerda_to_themedata import load_model_surfaces
    cfg, meta, gx, gy, terrain, bottoms = load_model_surfaces(region_config)
    return cfg, meta, gx, gy, terrain, bottoms


def _cell(x, y, meta, gx, gy):
    ix = np.floor((x - (gx[0] - meta["dx"] / 2)) / meta["dx"]).astype(int)
    iy = np.floor((y - (gy[0] - meta["dy"] / 2)) / meta["dy"]).astype(int)
    ok = (ix >= 0) & (ix < gx.size) & (iy >= 0) & (iy < gy.size)
    return np.clip(ix, 0, gx.size - 1), np.clip(iy, 0, gy.size - 1), ok


def build_wells(region_config, jupiter=None, snapped=None, model_year=None, out="wells.csv"):
    cfg, meta, gx, gy, terrain, bottoms = _surfaces(region_config)
    names = cfg.modelled_layer_names
    n = len(names)
    rows = []
    snap_ids, snap_xy = set(), np.empty((0, 2))

    # ---------------- snapped interpretation points ----------------
    if snapped:
        s = _read(snapped, SNAPPED_COLUMNS)
        miss = {"x", "y", "layer", "z", "quality"} - set(s.columns)
        if miss:
            sys.exit(f"{snapped}: missing columns {sorted(miss)} (found {list(s.columns)}); edit SNAPPED_COLUMNS")
        if s.layer.dtype == object:                       # surface names -> numbers
            lut = {nm.lower(): i + 1 for i, nm in enumerate(names)}
            s["layer"] = s.layer.astype(str).str.lower().map(lut)
            bad = s.layer.isna().sum()
            if bad:
                print(f"  note: {bad} snapped points have a layer name not in region_config -> skipped")
        s = s.dropna(subset=["x", "y", "layer", "z"])
        s["layer"] = s.layer.astype(int)
        s = s[(s.layer >= 1) & (s.layer <= n)]
        ix, iy, ok = _cell(s.x.values, s.y.values, meta, gx, gy)
        s = s[ok]; ix, iy = ix[ok], iy[ok]
        s["terrain"] = terrain[iy, ix]
        s["quality"] = s.quality.clip(1, 4).astype(int)
        if "well_id" not in s:
            s["well_id"] = [f"snap_{i}" for i in range(len(s))]
        s["source"] = "snap"
        rows.append(s[["well_id", "x", "y", "layer", "z", "terrain", "quality", "source"]]
                    .rename(columns={"z": "z_boundary"}))
        snap_ids = set(s.well_id.astype(str).str.replace(" ", "").str.lower())
        snap_xy = s[["x", "y"]].drop_duplicates().values
        print(f"snapped: {len(s)} boundary picks from {s.well_id.nunique()} wells")

    # ---------------- Jupiter boreholes ----------------
    if jupiter:
        j = _read(jupiter, JUPITER_COLUMNS)
        miss = {"x", "y", "drilldepth"} - set(j.columns)
        if miss:
            sys.exit(f"{jupiter}: missing columns {sorted(miss)} (found {list(j.columns)[:15]}); edit JUPITER_COLUMNS")
        j = j.dropna(subset=["x", "y", "drilldepth"])
        n0 = len(j)
        ix, iy, ok = _cell(j.x.values, j.y.values, meta, gx, gy)
        j = j[ok].copy(); ix, iy = ix[ok], iy[ok]
        n_in = len(j)
        # year filter (drilled no later than the model)
        if "drilldate" in j:
            d = j.drilldate.astype(str).str.strip()
            yr = pd.to_numeric(d.str[:4], errors="coerce")
            yr2 = pd.to_datetime(d, errors="coerce", dayfirst=True, format="mixed").dt.year
            j["year"] = yr.where(yr.between(1000, 2100), yr2)
        else:
            j["year"] = np.nan
        keep = pd.Series(True, index=j.index)
        if model_year:
            keep &= ~(j.year > model_year)                 # unknown year is kept
        # not already a snapped well (by DGU no., else by position within 1 m)
        if "well_id" in j and snap_ids:
            keep &= ~j.well_id.astype(str).str.replace(" ", "").str.lower().isin(snap_ids)
        if len(snap_xy):
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(snap_xy).query(j[["x", "y"]].values)
            keep &= dist > 1.0
        j, ix, iy = j[keep.values], ix[keep.values], iy[keep.values]
        terr = terrain[iy, ix]
        elev = j.elevation.values.astype(float) if "elevation" in j else np.full(len(j), np.nan)
        elev = np.where(np.isfinite(elev), elev, terr)          # GEUS: missing elevation -> terrain
        well_bottom = elev - j.drilldepth.values.astype(float)
        if "quality" in j and j.quality.notna().any():
            q = j.quality.fillna(4).clip(1, 4).astype(int).values
            qsrc = "from file"
        else:
            q = JUPITER_QUALITY_RULE(j).values
            qsrc = "PLACEHOLDER rule (JUPITER_QUALITY_RULE)"
        wid = j.well_id.astype(str).values if "well_id" in j else np.array([f"jup_{i}" for i in range(len(j))])
        counts = []
        for b in range(n):
            zb = bottoms[iy, ix, b]
            reach = np.isfinite(zb) & (well_bottom <= zb)       # well reaches the boundary
            counts.append(int(reach.sum()))
            if reach.any():
                rows.append(pd.DataFrame(dict(
                    well_id=wid[reach], x=j.x.values[reach], y=j.y.values[reach], layer=b + 1,
                    z_boundary=zb[reach], terrain=terr[reach], quality=q[reach], source="jupiter")))
        print(f"jupiter: {n0} wells in file, {n_in} inside the model grid, {len(j)} after "
              f"year/snapped filters; quality {qsrc}")
        print("  wells reaching each boundary: " + " ".join(str(c) for c in counts))

    if not rows:
        sys.exit("Nothing to write: give --jupiter and/or --snapped")
    w = pd.concat(rows, ignore_index=True)
    w = w[w.z_boundary <= w.terrain + 1e-6]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    w.round(3).to_csv(out, index=False)
    print(f"wrote {out}: {len(w)} boundary picks "
          f"({(w.source == 'snap').sum()} snap, {(w.source == 'jupiter').sum()} jupiter)")
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jupiter + interpretation points -> wells.csv for GeoSigma")
    ap.add_argument("--region-config", required=True)
    ap.add_argument("--jupiter", help="Jupiter BOREHOLE table as CSV")
    ap.add_argument("--snapped", help="model interpretation points (snap points) as CSV")
    ap.add_argument("--model-year", type=int, help="drop Jupiter wells drilled after this year")
    ap.add_argument("--out", default="wells.csv")
    a = ap.parse_args()
    build_wells(a.region_config, a.jupiter, a.snapped, a.model_year, a.out)
