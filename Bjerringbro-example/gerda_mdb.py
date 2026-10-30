# -*- coding: utf-8 -*-
"""
Read a GERDA Access export (.mdb) into the "long" table gerda_to_themedata.py uses
==================================================================================

GERDA tables used (standard GERDA schema, same for PACES, SkyTEM, tTEM, TEM, MEP):
    odvpos    one row per 1D model position: model, position, xutm, yutm, elevation
    odvlayer  one row per resistivity layer: model, position, layer, rho, thickness, depbottom
    odvdoi    depth of investigation per position: doiupper (shallow/conservative,
              threshold 1.2) and doilower (deeper, threshold 0.6)
    odvmodse  model -> dataset (and datatype: paces, skytem, ...)
    dataset   acquisition dates (recstartda)

Output columns: model_id, x, y, elevation, layer_no, depth_top, depth_bot, rho,
doi, doi_upper, year, datatype, gerda_model, gerda_position.

Reading the .mdb:
    * Linux/macOS: needs mdbtools (apt install mdbtools / brew install mdbtools)
    * Windows:     needs pyodbc + "Microsoft Access Driver (*.mdb, *.accdb)"
                   (installed with Office, or the free "Access Database Engine")

    python gerda_mdb.py paces_fyn_del1.mdb  -> paces_fyn_del1_long.csv
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ("odvpos", "odvlayer", "odvdoi", "odvmodse", "dataset")


def _odbc_connect(mdb_path: Path):
    """Connect with the Access ODBC driver, working around its path quirks."""
    import pyodbc

    drivers = [d for d in pyodbc.drivers() if "Access" in d]
    if not drivers:
        sys.exit("No Microsoft Access ODBC driver found. Install the 'Microsoft Access Database "
                 "Engine' with the SAME bitness (64-bit) as your Python.\n"
                 f"Drivers visible to this Python: {pyodbc.drivers()}")
    driver = next((d for d in drivers if "accdb" in d), drivers[0])

    def connect(p):
        return pyodbc.connect(f"DRIVER={{{driver}}};DBQ={p};ReadOnly=1;")

    try:
        return connect(mdb_path), None
    except pyodbc.Error as exc:
        # Typical causes: æøå or other non-ASCII characters, ';' in the path,
        # OneDrive/SharePoint or very long paths. Retry from a plain temp copy.
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="gerda_"))
        tmp = tmpdir / "gerda_export.mdb"
        shutil.copy2(mdb_path, tmp)
        try:
            con = connect(tmp)
        except pyodbc.Error:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(
                f"Access driver could not open {mdb_path} (also not as a temp copy {tmp}).\n"
                f"Original error: {exc}\n"
                "Try: copy the file to C:\\temp\\paces.mdb and pass that path; check the file "
                "opens in Access; make sure the download finished (file not 0 bytes)."
            ) from exc
        print(f"note: opened a temporary copy ({tmp}); the original path was rejected by the driver")
        return con, tmpdir


def _read_tables(mdb):
    mdb_path = Path(mdb).expanduser().resolve()      # the driver needs an absolute path
    if not mdb_path.is_file():
        sys.exit(f"File not found: {mdb_path}")
    mdb = str(mdb_path)
    if shutil.which("mdb-export"):
        out = {}
        for t in TABLES:
            txt = subprocess.run(["mdb-export", mdb, t], capture_output=True,
                                 text=True, check=True).stdout
            out[t] = pd.read_csv(io.StringIO(txt))
        return out
    try:
        import pyodbc
    except ImportError:
        sys.exit("Install mdbtools (Linux/macOS) or pyodbc + the Access ODBC driver (Windows).")
    con, tmpdir = _odbc_connect(mdb_path)
    out = {}
    try:
        cur = con.cursor()
        for t in TABLES:
            cur.execute(f"SELECT * FROM [{t}]")
            cols = [c[0] for c in cur.description]
            df = pd.DataFrame.from_records([tuple(r) for r in cur.fetchall()], columns=cols)
            for c in df.columns:                      # Decimal -> float
                if df[c].dtype == object and c not in ("recstartda", "datatype"):
                    conv = pd.to_numeric(df[c], errors="coerce")
                    if conv.notna().sum() == df[c].notna().sum():
                        df[c] = conv
            out[t] = df
    finally:
        con.close()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return out


def read_gerda_mdb(mdb, doi_column="doilower"):
    t = _read_tables(mdb)
    pos, lay, doi = t["odvpos"], t["odvlayer"], t["odvdoi"]
    if pos.empty or lay.empty:
        sys.exit(f"{mdb}: no 1D models in odvpos/odvlayer (is this a model export, not raw data?)")

    # Acquisition year per model (model -> dataset -> recstartda).
    ds = t["dataset"][["dataset", "recstartda"]].copy()
    ds["year"] = pd.to_datetime(ds.recstartda, errors="coerce", format="mixed").dt.year
    mset = t["odvmodse"][["model", "dataset", "datatype"]].merge(ds, on="dataset", how="left")
    mset = mset.groupby("model").agg(year=("year", "max"), datatype=("datatype", "first"))

    lay = lay.sort_values(["model", "position", "layer"]).copy()
    # deptop is often empty in GERDA: top of layer n = bottom of layer n-1 (0 for layer 1).
    prev_bot = lay.groupby(["model", "position"]).depbottom.shift(1)
    lay["depth_top"] = lay.deptop.where(lay.deptop.notna(), prev_bot.fillna(0.0))
    lay.loc[lay.layer == 1, "depth_top"] = lay.loc[lay.layer == 1, "depth_top"].fillna(0.0)
    lay["depth_bot"] = lay.depbottom          # NaN for the bottom half-space

    df = (lay[["model", "position", "layer", "depth_top", "depth_bot", "rho"]]
          .merge(pos[["model", "position", "xutm", "yutm", "elevation"]], on=["model", "position"])
          .merge(doi[["model", "position", "doiupper", "doilower"]], on=["model", "position"], how="left")
          .merge(mset, left_on="model", right_index=True, how="left"))

    df = df.rename(columns={"xutm": "x", "yutm": "y", "layer": "layer_no",
                            "model": "gerda_model", "position": "gerda_position",
                            "doiupper": "doi_upper"})
    df["doi"] = df[doi_column if doi_column in df else "doilower"]
    df["model_id"] = df.gerda_model.astype(str) + "_" + df.gerda_position.astype(str)
    cols = ["model_id", "x", "y", "elevation", "layer_no", "depth_top", "depth_bot", "rho",
            "doi", "doi_upper", "year", "datatype", "gerda_model", "gerda_position"]
    return df[cols].sort_values(["gerda_model", "gerda_position", "layer_no"]).reset_index(drop=True)


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_long.csv")
    df = read_gerda_mdb(src)
    df.to_csv(out, index=False)
    n = df.model_id.nunique()
    print(f"{src.name}: {n} model positions, {len(df)} layer rows, "
          f"datatype={df.datatype.dropna().unique().tolist()}, years={sorted(df.year.dropna().unique().astype(int).tolist())}")
    print(f"x {df.x.min():.0f}-{df.x.max():.0f}, y {df.y.min():.0f}-{df.y.max():.0f} (EPSG:25832)")
    print(f"DOI (doilower) median {df.groupby('model_id').doi.first().median():.1f} m")
    print(f"wrote {out}")
