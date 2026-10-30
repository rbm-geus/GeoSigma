# Bjerringbro example

Uncertainty of a hydrostratigraphic layer model at Bjerringbro, built with
GeoSigma from real Danish data: GERDA geophysics (SkyTEM, ground TEM, PACES) and
Jupiter boreholes. The example produces per-layer **variance maps** and a small
ensemble of layer-model **realizations**.

| | |
|---|---|
| Area | 10 × 10 km, x 536 000–546 000, y 6 250 000–6 260 000 (EPSG:25832) |
| Grid | 100 m, 100 × 100 cells |
| Model | 15 layer boundaries below terrain (`region_config.json`) |
| Data (Downloaded 2026/10/29) | 495 SkyTEM, 128 ground-TEM and 2 800 PACES soundings; 17 Jupiter wells (163 boundary picks, only wells deeper than 75 meters) |

## Run it

The folder must sit in the root of the GeoSigma repository. It reads the model
surfaces from `../examples/data/Jutland_hydrostrat_model` and the complexity map
from `../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd`.

```bash
cd GeoSigma
pip install -e .              # once
cd Bjerringbro-example
python geosigma_dk_workflow.py
```

This takes about 15 seconds and writes:

| Output | Content |
|---|---|
| `output/variance_maps/variance_NN.tif` | Variance [m²] of boundary NN, the input for kriging/simulation |
| `output/variance_maps/std_NN.tif` | Standard deviation [m] for viewing in QGIS |
| `output/variance_maps/qa_overview.png` | Data coverage per source, final std and realizations |
| `output/realizations/real_NNN/` | 10 realizations, one GeoTIFF per boundary |

All settings are in the `CONFIG` block at the top of `geosigma_dk_workflow.py`.

## Folder contents

```
Bjerringbro-example/
├── region_config.json        layer list and location of the model surfaces
├── geofysik/                 ThemeData bundles (points CSV + manifest) per method
│   ├── skytem/               SkyTEM            -> spec skytem
│   ├── fewtem/               ground TEM        -> spec fewtem
│   └── paces/                PACES             -> spec paces
├── boringer/
│   ├── jupiter.csv           Jupiter BOREHOLE extract (65 wells)
│   └── wells.csv             borehole input for the well theme
├── geosigma_dk_workflow.py   main script: themes -> variance maps -> realizations
├── gerda_db.py               reads GERDA exports (Access .mdb or Firebird .fdb)
├── gerda_to_themedata.py     GERDA models -> ThemeData bundle
└── wells_to_csv.py           Jupiter (+ interpretation points) -> wells.csv
```

## How the inputs were made

The geophysics comes from a GERDA Firebird export (`bjerringbro_geofysik.fdb`,
not included because of its size). Each bundle was made with commands like:

```bash
python gerda_to_themedata.py --list --gerda bjerringbro_geofysik.fdb --region-config region_config.json
python gerda_to_themedata.py --gerda bjerringbro_geofysik.fdb --region-config region_config.json \
    --datatype tem --subtype skytem --spec skytem --name skytem --out geofysik/skytem
python gerda_to_themedata.py --gerda bjerringbro_geofysik.fdb --region-config region_config.json \
    --datatype tem --spec fewtem --name fewtem --out geofysik/fewtem
python gerda_to_themedata.py --gerda bjerringbro_geofysik.fdb --region-config region_config.json \
    --datatype paces --spec paces --name paces --out geofysik/paces
```

The well input was made from the Jupiter extract:

```bash
python wells_to_csv.py --region-config region_config.json --jupiter boringer/jupiter.csv --out boringer/wells.csv
```

Each manifest's `_notes` records the source file and how many soundings were kept.

## Limitations

- **Realizations are a simplified stand-in.** GeoSigma v0.1.0 only provides the
  variance maps. The realizations here are the deterministic surfaces plus
  correlated noise scaled by the variance map, with the layers kept in order.
  This is not GEUS's planned v1.0.0 method (local kriging and conditional
  simulation), and it has not been validated. The correlation range (500 m) is
  a choice, not estimated from data.
- **Borehole quality is a placeholder.** `jupiter.csv` has no quality class, so
  `wells_to_csv.py` assigned 2–4 from the drilling year. No interpretation points
  ("snapped" wells) are included, and no `--model-year` filter was applied, so
  wells drilled after the model was made are counted too.
- **The complexity map is a provisional proxy.** It comes from the national
  MATLAB reference data. See the notes in `docs/theme_format.md` on its origin
  and limitations.
- **The model-quality class is uniform** (class 2 everywhere), because no
  model-class grid was supplied.
- **Coverage is sparse.** Each geophysical method informs at most about 5 % of
  the cells per layer, so most of the area carries the model theme's background
  uncertainty (median std 6–28 m, increasing with depth).

## Data

Geophysical models: GERDA, GEUS. Boreholes: Jupiter, GEUS.
