# -*- coding: utf-8 -*-
"""
Convert a GERDA geophysical model export into a GeoSigma ThemeData bundle
=========================================================================

Writes, for one data source (SkyTEM, tTEM, ground TEM, PACES, MEP ...):

    <out_dir>/<name>_points.csv        one row per sounding
    <out_dir>/<name>.manifest.yaml     describes the CSV for GeoSigma

What goes in each CSV column (same meaning as in the GEUS/DK-model workflow)
----------------------------------------------------------------------------
x, y          Sounding position (EPSG:25832, UTM32 ETRS89).
depth__LNN    Depth BELOW TERRAIN of YOUR hydrostratigraphic boundary NN at the
              sounding. It comes from your MODEL SURFACES sampled at (x, y) — not
              from the geophysics. (MATLAB: dkm_d_pos.)
thick__LNN    Thickness of the GEOPHYSICAL model layer (the GERDA resistivity
              layer) in which boundary NN falls at that sounding. Thick
              resistivity layer = poor vertical resolution = larger variance.
              (MATLAB: dkm_odvthk_pos.) NaN if the boundary is in the bottom
              half-space.
doi           Depth of investigation of the sounding, m below surface. GEUS used
              GERDA's "doilower" (the DEEPER DOI estimate, sensitivity threshold
              0.6; "doiupper" is the shallower, conservative one).

PACES needs only depth__LNN (its DOI is fixed at 18 m inside the spec), so use
--attributes depth.

Input
-----
EITHER the GERDA database itself — Access (.mdb/.accdb) or Firebird (.fdb/.gdb),
read by gerda_db.py (keep it in the same folder; see its header for Firebird
settings). Only positions inside your model grid, and only the --datatype you ask
for, are read from the database.

OR a "long" CSV, one row per sounding-layer (e.g. made by gerda_db.py, or your own
join of GERDA's model and layer tables), with these columns (rename via COLUMN_MAP
below if your headers differ):

    model_id    unique id of the sounding / 1D model
    x, y        UTM32 coordinates
    layer_no    1, 2, 3 ... from the top
    depth_top   depth to the top of the resistivity layer, m below surface
    depth_bot   depth to its bottom (empty / NaN for the bottom half-space)
    rho         resistivity (not used by GeoSigma, handy for QA)
    doi         the DOI of the model (repeated on every row is fine)
    year        acquisition year (optional, for --min-year)

Usage
-----
    python gerda_to_themedata.py --list --gerda gerda_export.fdb \
        --region-config region_config.json          # what is in my area?

    python gerda_to_themedata.py --gerda gerda_export.fdb --datatype skytem \
        --region-config region_config.json \
        --name skytem --spec skytem --out geofysik/skytem

    python gerda_to_themedata.py --gerda paces_export.csv \
        --region-config region_config.json \
        --name paces --spec paces --attributes depth --out geofysik/paces

    python gerda_to_themedata.py --demo     # fabricates a GERDA-like export and converts it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# >>> YOUR DATA: map YOUR export's header names -> the names this script expects.
# Leave a key out if your file already uses the expected name.
COLUMN_MAP = {
    # "MODELID": "model_id",
    # "XUTM": "x",
    # "YUTM": "y",
    # "LAYER": "layer_no",
    # "DEPTH_TOP": "depth_top",
    # "DEPTH_BOTTOM": "depth_bot",
    # "RHO": "rho",
    # "DOILOWER": "doi",
    # "YEAR": "year",
}


def read_gerda_long(path, column_map=COLUMN_MAP, bbox=None, datatypes=None):
    """GERDA Access/Firebird database or a long CSV -> long table.

    bbox / datatypes are pushed into the database query (Firebird) so a large
    export is not read in full; for CSV input they are applied later by convert().
    """
    if str(path).lower().endswith((".mdb", ".accdb", ".fdb", ".gdb", ".ib")):
        sys.path.insert(0, str(Path(__file__).parent))
        from gerda_db import read_gerda
        return read_gerda(path, bbox=bbox, datatypes=datatypes).sort_values(["model_id", "layer_no"])
    df = pd.read_csv(path, sep=None, engine="python")      # sniff , or ;
    df = df.rename(columns=column_map)
    need = {"model_id", "x", "y", "layer_no", "depth_top", "depth_bot"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"GERDA file is missing columns {sorted(missing)}. "
                 f"Found: {list(df.columns)}. Edit COLUMN_MAP.")
    return df.sort_values(["model_id", "layer_no"])


def load_model_surfaces(region_config):
    """Terrain + modelled bottoms, south-up, plus cell-centre axes."""
    from geosigma.config import RegionConfig
    from utils import grid_coordinates_from_esri_meta
    sys.path.insert(0, str(Path(__file__).parent))
    from geosigma_dk_workflow import load_grid

    cfg = RegionConfig.from_file(region_config)
    if cfg.surface_dir and not Path(cfg.surface_dir).is_absolute():
        cfg.surface_dir = str(Path(region_config).parent / cfg.surface_dir)
    top, meta = load_grid(cfg.surface_path(0))
    bots = [load_grid(cfg.surface_path(i))[0] for i in cfg.modelled_layer_indices]
    gx, gy, _, _ = grid_coordinates_from_esri_meta(meta)
    return cfg, meta, gx, gy, np.flipud(top), np.stack([np.flipud(b) for b in bots], -1)


# --------------------------------------------------------------------------------
# Which GeoSigma spec fits which GERDA data type
# --------------------------------------------------------------------------------
FEW_LAYER_MAX = 7     # TEM models with <= this many layers count as "few-layer" (fewTEM)


def suggest_spec(datatype, subtype, numlayers):
    """Return (spec, attributes, note) for a GERDA datatype/subtype/model size."""
    dt = str(datatype).lower() if pd.notna(datatype) else ""
    st = str(subtype).upper() if pd.notna(subtype) else ""
    nl = int(numlayers) if pd.notna(numlayers) else 0
    if dt == "skytem" or st.startswith("SKYTEM"):
        return "skytem", "depth,thick,doi", "airborne TEM"
    if dt == "tem":
        if st.startswith("TTEM"):
            return "ttem", "depth,thick,doi", "towed TEM"
        if st == "PATEM":
            return "ttem", "depth,thick,doi", "pulled-array TEM: closest built-in is tTEM (judgement)"
        if nl and nl <= FEW_LAYER_MAX:
            return "fewtem", "depth,doi", f"ground TEM, few-layer models ({nl} layers)"
        return "manytem", "depth,thick,doi", f"ground TEM, many-layer/smooth models ({nl} layers)"
    if dt == "paces":
        return "paces", "depth", "PACES; use pacep (depth,doi) for the newer, deeper PACEP system"
    if dt == "mep":
        return "mep", "depth,doi,mep_type", "MEP/ERT 2D; mep_type and doi_fill are added automatically"
    if dt == "sert":
        return "CUSTOM", "depth,doi", "streamed ERT: no built-in; start from pacep as a YAML spec"
    if dt in ("schlumberger", "wenner"):
        return "CUSTOM", "depth,doi", "old DC soundings/profiles: no built-in; start from fewtem/pacep"
    if dt == "hem":
        return "CUSTOM", "depth,thick,doi", "helicopter FEM: no built-in; start from skytem"
    return "NONE", "", "no 1D layer model GeoSigma can use (em3x/gcm, mt ...)"


def grid_bbox(region_config):
    """Model grid extent (xmin, ymin, xmax, ymax) from the region config."""
    _, meta, gx, gy, *_ = load_model_surfaces(region_config)
    return (gx[0] - meta["dx"] / 2, gy[0] - meta["dy"] / 2,
            gx[-1] + meta["dx"] / 2, gy[-1] + meta["dy"] / 2)


def list_contents(gerda, region_config=None):
    bbox = grid_bbox(region_config) if region_config else None
    if bbox:
        print(f"(only positions inside the model grid {bbox[0]:.0f},{bbox[1]:.0f} - "
              f"{bbox[2]:.0f},{bbox[3]:.0f})")
    df = read_gerda_long(gerda, bbox=bbox)
    head = df.groupby("model_id").first()
    for c in ("datatype", "subtype", "numlayers", "year", "inversion"):
        if c not in head:
            head[c] = np.nan
    grp = head.groupby([head.datatype.astype(object).fillna("?"), head.subtype.astype(object).fillna("-"),
                        head.numlayers.fillna(0).astype(int)], dropna=False)
    print(f"{Path(gerda).name}: {len(head)} model positions\n")
    print(f"{'datatype':10s} {'subtype':12s} {'layers':>6s} {'positions':>9s}  {'years':11s}  spec      attributes          note")
    for (dt, st, nl), g in grp:
        spec, attrs, note = suggest_spec(dt, None if st == "-" else st, nl)
        yrs = g.year.dropna()
        yr = f"{int(yrs.min())}-{int(yrs.max())}" if len(yrs) else "?"
        print(f"{dt:10s} {st:12s} {nl:6d} {len(g):9d}  {yr:11s}  {spec:9s} {attrs:19s} {note}")
    print("\nConvert one group at a time, e.g.\n  python gerda_to_themedata.py --gerda GERDA_FILE "
          "--region-config RC.json --datatype skytem --spec skytem --name skytem --out my_site/skytem")


def filter_rows(df, datatype=None, subtype=None, min_layers=None, max_layers=None):
    def norm(v):
        return [s.strip().lower() for s in v.split(",")] if v else None
    if datatype:
        df = df[df.datatype.astype(str).str.lower().isin(norm(datatype))]
    if subtype:
        st = df.subtype.astype(str).str.lower()
        df = df[np.logical_or.reduce([st.str.startswith(s) for s in norm(subtype)])]
    if min_layers is not None:
        df = df[df.numlayers >= min_layers]
    if max_layers is not None:
        df = df[df.numlayers <= max_layers]
    if df.empty:
        sys.exit("No models left after filtering — check --datatype/--subtype/--min/max-layers (try --list).")
    return df


def convert(gerda, region_config, name, spec, out_dir, attributes=("depth", "thick", "doi"),
            min_year=None, complexity="none", datatype=None, subtype=None,
            min_layers=None, max_layers=None):
    cfg, meta, gx, gy, terrain, bottoms = load_model_surfaces(region_config)
    n_layers = cfg.n_modelled_layers

    bbox = (gx[0] - meta["dx"] / 2, gy[0] - meta["dy"] / 2,
            gx[-1] + meta["dx"] / 2, gy[-1] + meta["dy"] / 2)
    dts = [d.strip() for d in datatype.split(",")] if datatype else None
    df = read_gerda_long(gerda, bbox=bbox, datatypes=dts)
    if any(v is not None for v in (datatype, subtype, min_layers, max_layers)):
        df = filter_rows(df, datatype, subtype, min_layers, max_layers)
    if min_year is not None and "year" in df:
        df = df[df.year >= min_year]
    if spec == "mep" and "mep_type" not in attributes:
        attributes = tuple(attributes) + ("mep_type",)

    # One row per sounding: position, DOI, and its layer stack.
    head = df.groupby("model_id", sort=False).first()
    n_raw = len(head)
    # Keep soundings inside the model grid only.
    inside = ((head.x >= gx[0] - meta["dx"] / 2) & (head.x <= gx[-1] + meta["dx"] / 2) &
              (head.y >= gy[0] - meta["dy"] / 2) & (head.y <= gy[-1] + meta["dy"] / 2))
    head = head[inside]
    # De-duplicate identical positions, keep the last (GEUS did the same).
    head = head[~head.duplicated(subset=["x", "y"], keep="last")]
    print(f"{name}: {n_raw} soundings in file -> {len(head)} inside grid after de-duplication")

    # depth__L: YOUR model boundaries at each sounding, as depth below terrain.
    ix = np.clip(((head.x.values - gx[0]) / meta["dx"] + 0.5).astype(int), 0, gx.size - 1)
    iy = np.clip(((head.y.values - gy[0]) / meta["dy"] + 0.5).astype(int), 0, gy.size - 1)
    depth = terrain[iy, ix][:, None] - bottoms[iy, ix]          # (n_soundings, n_layers)

    # thick__L: thickness of the resistivity layer containing each boundary depth.
    thick = np.full_like(depth, np.nan)
    layers = df[df.model_id.isin(head.index)].groupby("model_id", sort=False)
    row_of = {mid: k for k, mid in enumerate(head.index)}
    for mid, lay in layers:
        k = row_of[mid]
        top = lay.depth_top.to_numpy(float)
        bot = lay.depth_bot.to_numpy(float)
        bot_filled = np.where(np.isnan(bot), np.inf, bot)       # half-space
        for s in range(n_layers):
            d = depth[k, s]
            if not np.isfinite(d):
                continue
            hit = np.where((top <= d) & (d < bot_filled))[0]
            if hit.size:
                thick[k, s] = bot[hit[0]] - top[hit[0]]          # NaN in the half-space

    cols = {"x": head.x.values, "y": head.y.values}
    if "depth" in attributes:
        cols.update({f"depth__L{s + 1:02d}": depth[:, s] for s in range(n_layers)})
    if "thick" in attributes:
        cols.update({f"thick__L{s + 1:02d}": thick[:, s] for s in range(n_layers)})
    if "doi" in attributes:
        if "doi" not in head:
            sys.exit("--attributes includes doi but the GERDA file has no doi column")
        cols["doi"] = head.doi.values     # NaN kept: fill it later (see bring_your_own_data.md, choice 9)
    spec_params = {}
    if "mep_type" in attributes:
        # 1 = Wenner 2D (GERDA subtype starting 'wen'), 0 = other MEP layouts (as GEUS).
        sub = head.subtype if "subtype" in head else pd.Series("", index=head.index)
        cols["mep_type"] = sub.astype(str).str.lower().str.startswith("wen").astype(int).values
    if spec == "mep":
        # The MEP spec fills missing DOI with the mean DOI of the source (spec_params).
        spec_params["doi_fill"] = round(float(np.nanmean(head.doi.values)), 3)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"{name}_points.csv"
    pd.DataFrame(cols).round(3).to_csv(csv, index=False)

    attr_yaml = ""
    if "depth" in attributes:
        attr_yaml += "  depth:\n    per_layer: true\n    prefix: depth__L\n"
    if "thick" in attributes:
        attr_yaml += "  thick:\n    per_layer: true\n    prefix: thick__L\n"
    if "doi" in attributes:
        attr_yaml += "  doi:\n    per_layer: false\n    column: doi\n"
    if "mep_type" in attributes:
        attr_yaml += "  mep_type:\n    per_layer: false\n    column: mep_type\n"
    if spec_params:
        attr_yaml += "spec_params:\n" + "".join(f"  {k}: {v}\n" for k, v in spec_params.items())
    manifest = out_dir / f"{name}.manifest.yaml"
    manifest.write_text(f"""format_version: 1
name: {name}
spec: {spec}
points_csv: {csv.name}
n_layers: {n_layers}
crs: EPSG:25832
x_column: x
y_column: y
attributes:
{attr_yaml}complexity: {complexity}
_notes:
  source: "GERDA export {Path(gerda).name}"
  filtering: "inside model grid; de-duplicated on (x, y) keeping last;{f' year >= {min_year};' if min_year else ''} {n_raw} -> {len(head)} soundings"
  attribute_columns:
    depth: "hydrostratigraphic boundary depth below model terrain, sampled from {Path(region_config).name}"
    thick: "thickness of the GERDA resistivity layer containing the boundary; NaN = half-space"
    doi: "GERDA doi (raw, NaN preserved)"
""")
    print(f"wrote {csv}\nwrote {manifest}")
    frac = np.mean(np.isfinite(thick), axis=0) if "thick" in attributes else None
    if frac is not None:
        print("share of soundings where boundary falls inside a finite resistivity layer, per layer:\n  "
              + " ".join(f"{f:.2f}" for f in frac))
    return manifest


def make_fake_gerda(region_config, path, rng=np.random.default_rng(3)):
    """Fabricate a GERDA-like long table (SkyTEM-ish lines) for testing."""
    _, meta, gx, gy, *_ = load_model_surfaces(region_config)
    rows = []
    mid = 0
    for y in np.arange(gy[0], gy[-1], 250.0):
        for x in np.arange(gx[0], gx[-1], 40.0):
            mid += 1
            thk = rng.uniform(3, 12, 1) * 1.15 ** np.arange(19)      # 20-layer smooth model
            top = np.concatenate([[0], np.cumsum(thk)])
            for L in range(20):
                rows.append(dict(MODELID=mid, XUTM=x + rng.normal(0, 5), YUTM=y,
                                 LAYER=L + 1, DEPTH_TOP=top[L],
                                 DEPTH_BOTTOM=top[L + 1] if L < 19 else np.nan,
                                 RHO=rng.lognormal(3.5, 0.8), DOILOWER=rng.uniform(80, 220),
                                 YEAR=2019))
    # GERDA repeats position per layer; make x identical within a model
    df = pd.DataFrame(rows)
    df["XUTM"] = df.groupby("MODELID").XUTM.transform("first")
    df.to_csv(path, index=False, sep=";")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gerda")
    ap.add_argument("--region-config")
    ap.add_argument("--name", default="skytem")
    ap.add_argument("--spec", default="skytem",
                    help="built-in name (skytem, ttem, fewtem, manytem, paces, pacep, mep ...) "
                         "or a path to your own .spec.yaml (relative to the manifest)")
    ap.add_argument("--attributes", default="depth,thick,doi")
    ap.add_argument("--min-year", type=int)
    ap.add_argument("--complexity", default="none", help="none | external | path to grid")
    ap.add_argument("--out", default="theme_out")
    ap.add_argument("--list", action="store_true",
                    help="summarise what the GERDA file contains and suggest a spec per group "
                         "(add --region-config to only look inside your model area)")
    ap.add_argument("--datatype", help="keep only these GERDA datatypes, e.g. skytem or tem,paces")
    ap.add_argument("--subtype", help="keep subtypes starting with, e.g. TTEM or SKYTEM2 or wen")
    ap.add_argument("--min-layers", type=int, help="keep models with at least N layers")
    ap.add_argument("--max-layers", type=int, help="keep models with at most N layers")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--geosigma-repo", default=".")
    a = ap.parse_args()

    if a.list:
        if not a.gerda:
            ap.error("--list needs --gerda")
        list_contents(a.gerda, a.region_config)
        sys.exit(0)
    if a.demo:
        import json
        repo = Path(a.geosigma_repo).resolve()
        d = Path("gerda_demo"); d.mkdir(exist_ok=True)
        rc = json.loads((repo / "examples/data/Jutland_hydrostrat_model/region_config.json").read_text())
        rc["surface_dir"] = str(repo / "examples/data/Jutland_hydrostrat_model")
        (d / "region_config.json").write_text(json.dumps(rc))
        COLUMN_MAP.update({"MODELID": "model_id", "XUTM": "x", "YUTM": "y", "LAYER": "layer_no",
                           "DEPTH_TOP": "depth_top", "DEPTH_BOTTOM": "depth_bot", "RHO": "rho",
                           "DOILOWER": "doi", "YEAR": "year"})
        g = make_fake_gerda(d / "region_config.json", d / "fake_gerda_skytem.csv")
        convert(g, d / "region_config.json", "skytem", "skytem", d / "skytem")
        convert(g, d / "region_config.json", "paces", "paces", d / "paces", attributes=("depth",))
    else:
        if not (a.gerda and a.region_config):
            ap.error("--gerda and --region-config are required (or use --demo)")
        convert(a.gerda, a.region_config, a.name, a.spec, a.out,
                tuple(a.attributes.split(",")), a.min_year, a.complexity,
                a.datatype, a.subtype, a.min_layers, a.max_layers)
