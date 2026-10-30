# -*- coding: utf-8 -*-
"""
Read a GERDA export — Access (.mdb/.accdb) or Firebird (.fdb/.gdb) — into the
"long" table that gerda_to_themedata.py uses (one row per sounding-layer).
Replaces gerda_mdb.py.

    python gerda_db.py gerda_export.fdb                         # whole database -> CSV
    python gerda_db.py gerda_export.fdb out.csv --bbox 571800,6123300,580000,6129800
    python gerda_db.py gerda_export.fdb out.csv --datatype skytem,tem
    python gerda_db.py gerda_export.fdb --schema                # list tables/columns (diagnostics)

GERDA tables used (same schema in both formats):
    odvpos    one row per 1D model position: model, position, xutm, yutm, elevation, numlayers
    odvlayer  one row per resistivity layer: model, position, layer, rho, depbottom (, deptop)
    odvdoi    depth of investigation per position: doiupper (threshold 1.2), doilower (0.6)
    odvmodse  model -> dataset, datatype (paces, skytem, tem, mep ...)
    dataset   acquisition date (recstartda)
    optional: temhea/skyhea/mephea/hemhea (acquisition subtype), onedvmod (inversion type)

Output columns: model_id, x, y, elevation, layer_no, depth_top, depth_bot, rho,
doi, doi_upper, year, datatype, subtype, inversion, numlayers, gerda_model, gerda_position.

Firebird
--------
Large GERDA downloads (> 2 GB) come as Firebird databases. For these the filters
(--bbox, --datatype; gerda_to_themedata.py sets them from your model grid
automatically) are applied INSIDE the database, so only the soundings you need are
read into memory.

You need:
  1. A Python driver:  pip install fdb      (works with Firebird 2.5 and 3.0 clients)
                       (firebird-driver is used instead if fdb is missing; it needs
                        a Firebird 3+ client library)
  2. A Firebird that can open the file. GERDA/Aarhus Workbench/GeoScene3D use
     Firebird 2.5. If you already have Workbench or GeoScene3D, its Firebird 2.5
     server is running and the defaults below work as they are.
     Without it: install Firebird 2.5 (Classic or SuperServer) from firebirdsql.org.
     A Firebird 3/4/5 install can NOT open a 2.5 database file directly
     ("unsupported on-disk structure"): use 2.5, or convert the file with
     gbak (backup with 2.5, restore with the newer version).

>>> YOUR SETTINGS: edit FIREBIRD below, or set the environment variables
    GERDA_FB_HOST, GERDA_FB_USER, GERDA_FB_PASSWORD, GERDA_FB_CLIENT, GERDA_FB_CHARSET.

Access
------
Windows: pyodbc + "Microsoft Access Driver (*.mdb, *.accdb)" (Office or the free
Access Database Engine, same 32/64-bit as Python). Linux/macOS: mdbtools.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# >>> YOUR SETTINGS (Firebird)
# ------------------------------------------------------------------------------
FIREBIRD = {
    # "localhost" = use the Firebird server on this PC (normal with Workbench/GeoScene3D).
    # None = embedded mode (no server; needs an embedded engine, e.g. fbembed.dll for 2.5).
    "host": os.environ.get("GERDA_FB_HOST", "localhost") or None,
    "user": os.environ.get("GERDA_FB_USER", "SYSDBA"),
    "password": os.environ.get("GERDA_FB_PASSWORD", "masterkey"),
    # Path to fbclient.dll / fbembed.dll / libfbclient.so if Python can't find it,
    # e.g. r"C:\Program Files (x86)\Firebird\Firebird_2_5\bin\fbclient.dll"
    "client_library": os.environ.get("GERDA_FB_CLIENT") or None,
    "charset": os.environ.get("GERDA_FB_CHARSET", "WIN1252"),
}

ACCESS_EXT = (".mdb", ".accdb")
FIREBIRD_EXT = (".fdb", ".gdb", ".ib")
DB_EXT = ACCESS_EXT + FIREBIRD_EXT

TABLES = ("odvpos", "odvlayer", "odvdoi", "odvmodse", "dataset")
OPTIONAL = ("temhea", "skyhea", "mephea", "hemhea", "onedvmod")
_TEXT_COLS = ("recstartda", "datatype", "datsubtype", "temsubtype", "mepsubtype",
              "hemsubtype", "modsubtype")

# Columns read per table. Required ones must exist; the rest become NULL if absent.
COLUMNS = {
    "odvpos":   (["model", "position", "xutm", "yutm"], ["elevation", "numlayers"]),
    "odvlayer": (["model", "position", "layer", "depbottom"], ["rho", "deptop"]),
    "odvdoi":   (["model", "position"], ["doiupper", "doilower"]),
    "odvmodse": (["model", "dataset"], ["datatype"]),
    "dataset":  (["dataset"], ["recstartda"]),
    "temhea":   (["dataset"], ["temsubtype"]),
    "skyhea":   (["dataset"], ["temsubtype"]),
    "mephea":   (["dataset"], ["mepsubtype"]),
    "hemhea":   (["dataset"], ["hemsubtype"]),
    "onedvmod": (["model"], ["modsubtype"]),
}
# POSITION is a reserved word in Firebird, so an export may quote or rename it.
ALIASES = {"position": ["position", "position_", "pos", "posno", "positionno", "modposition"]}


# ==============================================================================
# Firebird
# ==============================================================================
def _fb_connect(path: Path, conf=FIREBIRD):
    host = conf["host"]
    attempts = [host] + ([None] if host else [])          # fall back to embedded once
    errors = []
    try:
        import fdb as drv
        kind = "fdb"
    except ImportError:
        try:
            from firebird import driver as drv
            kind = "firebird-driver"
        except ImportError:
            sys.exit("No Firebird driver for Python. Run:  pip install fdb")
    for h in attempts:
        dsn = f"{h}:{path}" if h else str(path)
        try:
            if kind == "fdb":
                kw = dict(dsn=dsn, user=conf["user"], password=conf["password"],
                          charset=conf["charset"])
                if conf["client_library"]:
                    kw["fb_library_name"] = conf["client_library"]
                con = drv.connect(**kw)
            else:
                if conf["client_library"]:
                    drv.driver_config.fb_client_library.value = conf["client_library"]
                con = drv.connect(dsn, user=conf["user"], password=conf["password"],
                                  charset=conf["charset"])
            mode = f"server '{h}'" if h else "embedded"
            if errors:
                print(f"note: server connection failed, using embedded mode instead:\n{errors[0]}")
            print(f"Firebird: connected via {kind}, {mode}")
            return con
        except Exception as exc:                          # noqa: BLE001 - report below
            errors.append(f"  {dsn!r}: {str(exc).strip().splitlines()[0] if str(exc).strip() else exc}"
                          + "".join(f"\n      {ln.strip()}" for ln in str(exc).splitlines()[1:4]))
    text = "\n".join(errors)
    hint = []
    low = text.lower()
    if "on-disk structure" in low or "ods" in low:
        hint.append("The file was made by another Firebird version than the one you are using. "
                    "GERDA uses Firebird 2.5: use a 2.5 server/client (Workbench/GeoScene3D "
                    "install), or gbak backup (2.5) + restore (newer).")
    if "network request" in low or "connection refused" in low or "connect" in low:
        hint.append("No Firebird server answered. Start the Firebird service, or set "
                    "FIREBIRD['host'] = None for embedded mode.")
    if "user name and password" in low or "login" in low or "authentication" in low:
        hint.append("Wrong user/password. Defaults are SYSDBA / masterkey; set GERDA_FB_USER "
                    "and GERDA_FB_PASSWORD.")
    if "client library" in low or "fbclient" in low or "fbembed" in low or "load" in low:
        hint.append("Python can't find the Firebird client library. Set FIREBIRD['client_library'] "
                    "(or GERDA_FB_CLIENT) to the full path of fbclient.dll — same 32/64-bit as Python.")
    if "i/o error" in low or "no such file" in low or "cannot find" in low:
        hint.append("The server cannot open the file. The path must be on a LOCAL disk of the "
                    "server PC (not a network drive), and the Firebird service must have read access.")
    raise RuntimeError(f"Could not open {path} with Firebird:\n{text}\n\n" +
                       ("\n".join("-> " + h for h in hint) if hint else ""))


def _fb_schema(cur):
    cur.execute(
        "SELECT TRIM(rf.RDB$RELATION_NAME), TRIM(rf.RDB$FIELD_NAME) "
        "FROM RDB$RELATION_FIELDS rf JOIN RDB$RELATIONS r "
        "ON r.RDB$RELATION_NAME = rf.RDB$RELATION_NAME "
        "WHERE COALESCE(r.RDB$SYSTEM_FLAG, 0) = 0 ORDER BY 1, rf.RDB$FIELD_POSITION")
    schema = {}
    for tab, col in cur.fetchall():
        schema.setdefault(tab.lower(), {"_name": tab})[col.lower()] = col
    return schema


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def _fb_cols(schema, table, need_all=True):
    """Exact, quoted column expressions for COLUMNS[table] in canonical order."""
    info = schema[table]
    req, opt = COLUMNS[table]
    out = []
    for canon in req + opt:
        found = next((info[a] for a in ALIASES.get(canon, [canon]) if a in info), None)
        if found is None:
            if canon in req and need_all:
                cols = [c for c in info if c != "_name"]
                raise RuntimeError(f"Table {info['_name']} has no column for '{canon}'. "
                                   f"Columns: {cols}. Add the right name to ALIASES in gerda_db.py.")
            out.append((canon, None))
        else:
            out.append((canon, found))
    return out


def _fb_fetch(cur, sql, params, names, label, chunk=100_000):
    t0 = time.time()
    cur.execute(sql, params)
    parts = []
    while True:
        rows = cur.fetchmany(chunk)
        if not rows:
            break
        parts.append(pd.DataFrame.from_records([tuple(r) for r in rows], columns=names))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=names)
    print(f"  {label:9s} {len(df):>10,d} rows  ({time.time() - t0:.1f} s)")
    return df


def _read_tables_firebird(path: Path, bbox=None, datatypes=None):
    con = _fb_connect(path)
    try:
        cur = con.cursor()
        schema = _fb_schema(cur)
        missing = [t for t in TABLES if t not in schema]
        if missing:
            raise RuntimeError(f"{path}: tables {missing} not found. Tables in the file: "
                               f"{sorted(schema)[:40]} (is this a GERDA model export?)")

        # Position filter, evaluated inside Firebird on odvpos (alias p).
        pcols = dict(_fb_cols(schema, "odvpos"))
        where, params = [], []
        if bbox is not None:
            where.append(f"p.{_q(pcols['xutm'])} BETWEEN ? AND ? AND p.{_q(pcols['yutm'])} BETWEEN ? AND ?")
            params += [float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])]
        if datatypes:
            mc = dict(_fb_cols(schema, "odvmodse"))
            if mc["datatype"]:
                marks = ",".join("?" * len(datatypes))
                where.append(f"EXISTS (SELECT 1 FROM {_q(schema['odvmodse']['_name'])} s "
                             f"WHERE s.{_q(mc['model'])} = p.{_q(pcols['model'])} "
                             f"AND LOWER(s.{_q(mc['datatype'])}) IN ({marks}))")
                params += [d.lower() for d in datatypes]
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        def select(alias, cols):
            return ", ".join(f"{alias}.{_q(c)}" if c else "NULL" for _, c in cols)

        out = {}
        pos_cols = _fb_cols(schema, "odvpos")
        out["odvpos"] = _fb_fetch(
            cur, f"SELECT {select('p', pos_cols)} FROM {_q(schema['odvpos']['_name'])} p{where_sql}",
            params, [c for c, _ in pos_cols], "odvpos")

        for t in ("odvlayer", "odvdoi"):
            cols = _fb_cols(schema, t)
            tc = dict(cols)
            join = (f"JOIN {_q(schema['odvpos']['_name'])} p ON p.{_q(pcols['model'])} = x.{_q(tc['model'])} "
                    f"AND p.{_q(pcols['position'])} = x.{_q(tc['position'])}")
            sql = f"SELECT {select('x', cols)} FROM {_q(schema[t]['_name'])} x " + \
                  (join + where_sql if where else "")
            out[t] = _fb_fetch(cur, sql, params if where else [], [c for c, _ in cols], t)

        for t in ("odvmodse", "dataset") + OPTIONAL:
            if t not in schema:
                out[t] = pd.DataFrame()
                continue
            cols = _fb_cols(schema, t, need_all=t in TABLES)
            if dict(cols)[COLUMNS[t][0][0]] is None:      # optional table without its key
                out[t] = pd.DataFrame()
                continue
            out[t] = _fb_fetch(cur, f"SELECT {select('x', cols)} FROM {_q(schema[t]['_name'])} x",
                               [], [c for c, _ in cols], t)
    finally:
        con.close()
    return out


def print_schema(path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in FIREBIRD_EXT:
        con = _fb_connect(path)
        schema = _fb_schema(con.cursor())
        con.close()
        for t in sorted(schema):
            print(f"{schema[t]['_name']}: {', '.join(v for k, v in schema[t].items() if k != '_name')}")
    else:
        for t, df in _read_tables_access(path).items():
            print(f"{t}: {', '.join(df.columns)}")


# ==============================================================================
# Access
# ==============================================================================
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


def _read_tables_access(mdb_path: Path):
    mdb = str(mdb_path)
    if shutil.which("mdb-export"):
        out = {}
        for t in TABLES + OPTIONAL:
            r = subprocess.run(["mdb-export", mdb, t], capture_output=True, text=True)
            if r.returncode != 0 or not r.stdout.strip():
                if t in TABLES:
                    sys.exit(f"{mdb}: cannot read table {t}: {r.stderr.strip()}")
                out[t] = pd.DataFrame()
                continue
            out[t] = pd.read_csv(io.StringIO(r.stdout))
        return out
    try:
        import pyodbc
    except ImportError:
        sys.exit("Install mdbtools (Linux/macOS) or pyodbc + the Access ODBC driver (Windows).")
    con, tmpdir = _odbc_connect(mdb_path)
    out = {}
    try:
        cur = con.cursor()
        for t in TABLES + OPTIONAL:
            try:
                cur.execute(f"SELECT * FROM [{t}]")
            except pyodbc.Error:
                if t in TABLES:
                    raise
                out[t] = pd.DataFrame()
                continue
            cols = [c[0] for c in cur.description]
            out[t] = pd.DataFrame.from_records([tuple(r) for r in cur.fetchall()], columns=cols)
    finally:
        con.close()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return out


def _filter_access(t, bbox, datatypes):
    """Same filters as the Firebird SQL, applied in pandas."""
    pos = t["odvpos"]
    keep = pd.Series(True, index=pos.index)
    if bbox is not None:
        keep &= pos.xutm.between(bbox[0], bbox[2]) & pos.yutm.between(bbox[1], bbox[3])
    if datatypes and "datatype" in t["odvmodse"]:
        ms = t["odvmodse"]
        ok = ms.loc[ms.datatype.astype(str).str.lower().isin([d.lower() for d in datatypes]), "model"]
        keep &= pos.model.isin(ok)
    if keep.all():
        return t
    t = dict(t)
    t["odvpos"] = pos[keep]
    key = t["odvpos"][["model", "position"]]
    for name in ("odvlayer", "odvdoi"):
        t[name] = t[name].merge(key, on=["model", "position"])
    return t


# ==============================================================================
# Common
# ==============================================================================
def _clean(t):
    """Lower-case column names, Decimal/text numbers -> float."""
    for name, df in t.items():
        if df is None or df.empty:
            continue
        df.columns = [c.lower() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object and c not in _TEXT_COLS:
                conv = pd.to_numeric(df[c], errors="coerce")
                if conv.notna().sum() == df[c].notna().sum():
                    df[c] = conv
        for canon, alts in ALIASES.items():           # renamed POSITION column etc.
            if canon not in df:
                for a in alts:
                    if a in df:
                        df.rename(columns={a: canon}, inplace=True)
                        break
    return t


def _assemble(t, doi_column="doilower"):
    pos, lay, doi = t["odvpos"], t["odvlayer"], t["odvdoi"]
    if pos.empty or lay.empty:
        sys.exit("No 1D models found (odvpos/odvlayer empty) — wrong area/datatype filter, "
                 "or an export with raw data only?")
    for c in ("elevation", "numlayers"):
        if c not in pos:
            pos[c] = np.nan
    for c in ("deptop", "rho"):
        if c not in lay:
            lay[c] = np.nan
    for c in ("doiupper", "doilower"):
        if c not in doi:
            doi[c] = np.nan

    ds = t["dataset"][["dataset", "recstartda"]].copy() if "recstartda" in t["dataset"] \
        else t["dataset"][["dataset"]].assign(recstartda=pd.NaT)
    ds["year"] = pd.to_datetime(ds.recstartda, errors="coerce", format="mixed").dt.year
    ms = t["odvmodse"]
    if "datatype" not in ms:
        ms = ms.assign(datatype=np.nan)
    mset = ms[["model", "dataset", "datatype"]].merge(ds, on="dataset", how="left")

    sub = []
    for tab, col in (("temhea", "temsubtype"), ("skyhea", "temsubtype"),
                     ("mephea", "mepsubtype"), ("hemhea", "hemsubtype")):
        h = t.get(tab, pd.DataFrame())
        if not h.empty and col in h and "dataset" in h:
            sub.append(h[["dataset", col]].rename(columns={col: "subtype"}))
    if sub:
        mset = mset.merge(pd.concat(sub).dropna().drop_duplicates("dataset"), on="dataset", how="left")
    else:
        mset["subtype"] = np.nan
    mset = mset.groupby("model").agg(year=("year", "max"), datatype=("datatype", "first"),
                                     subtype=("subtype", "first"))
    inv = t.get("onedvmod", pd.DataFrame())
    if not inv.empty and "modsubtype" in inv:
        mset = mset.join(inv.drop_duplicates("model").set_index("model")["modsubtype"].rename("inversion"))
    else:
        mset["inversion"] = np.nan

    lay = lay.sort_values(["model", "position", "layer"]).copy()
    # deptop is often empty in GERDA: top of layer n = bottom of layer n-1 (0 for layer 1).
    prev_bot = lay.groupby(["model", "position"]).depbottom.shift(1)
    lay["depth_top"] = lay.deptop.where(lay.deptop.notna(), prev_bot)
    lay.loc[lay.layer == lay.groupby(["model", "position"]).layer.transform("min"), "depth_top"] = \
        lay.depth_top.fillna(0.0)
    lay["depth_bot"] = lay.depbottom          # NaN for the bottom half-space

    df = (lay[["model", "position", "layer", "depth_top", "depth_bot", "rho"]]
          .merge(pos[["model", "position", "xutm", "yutm", "elevation", "numlayers"]],
                 on=["model", "position"])
          .merge(doi[["model", "position", "doiupper", "doilower"]].drop_duplicates(["model", "position"]),
                 on=["model", "position"], how="left")
          .merge(mset, left_on="model", right_index=True, how="left"))

    df = df.rename(columns={"xutm": "x", "yutm": "y", "layer": "layer_no",
                            "model": "gerda_model", "position": "gerda_position",
                            "doiupper": "doi_upper"})
    df["doi"] = df[doi_column if doi_column in df else "doilower"]
    df["model_id"] = df.gerda_model.astype("int64").astype(str) + "_" + \
        df.gerda_position.astype("int64").astype(str)
    cols = ["model_id", "x", "y", "elevation", "layer_no", "depth_top", "depth_bot", "rho",
            "doi", "doi_upper", "year", "datatype", "subtype", "inversion", "numlayers",
            "gerda_model", "gerda_position"]
    return df[cols].sort_values(["gerda_model", "gerda_position", "layer_no"]).reset_index(drop=True)


def read_gerda(path, bbox=None, datatypes=None, doi_column="doilower"):
    """Read a GERDA .mdb/.accdb/.fdb/.gdb into the long table.

    bbox      (xmin, ymin, xmax, ymax) in EPSG:25832 — keep positions inside.
    datatypes list like ["skytem", "tem"] — keep only these GERDA datatypes.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        sys.exit(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext in FIREBIRD_EXT:
        t = _clean(_read_tables_firebird(path, bbox, datatypes))
    elif ext in ACCESS_EXT:
        t = _filter_access(_clean(_read_tables_access(path)), bbox, datatypes)
    else:
        sys.exit(f"Unknown database type {ext}; expected one of {DB_EXT}")
    return _assemble(t, doi_column)


read_gerda_mdb = read_gerda          # backwards compatible name


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GERDA .mdb/.fdb -> long CSV")
    ap.add_argument("db")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--bbox", help="xmin,ymin,xmax,ymax (EPSG:25832)")
    ap.add_argument("--datatype", help="e.g. skytem or tem,paces")
    ap.add_argument("--schema", action="store_true", help="print tables and columns, then stop")
    a = ap.parse_args()
    if a.schema:
        print_schema(a.db)
        sys.exit(0)
    src = Path(a.db)
    out = Path(a.out) if a.out else src.with_name(src.stem + "_long.csv")
    bbox = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    dts = [d.strip() for d in a.datatype.split(",")] if a.datatype else None
    df = read_gerda(src, bbox, dts)
    df.to_csv(out, index=False)
    print(f"{src.name}: {df.model_id.nunique()} model positions, {len(df)} layer rows, "
          f"datatype={df.datatype.dropna().unique().tolist()}, "
          f"years={sorted(df.year.dropna().unique().astype(int).tolist())}")
    print(f"x {df.x.min():.0f}-{df.x.max():.0f}, y {df.y.min():.0f}-{df.y.max():.0f} (EPSG:25832)")
    print(f"wrote {out}")
