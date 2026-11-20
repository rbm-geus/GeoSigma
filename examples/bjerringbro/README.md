# Bjerringbro example

Uncertainty of a hydrostratigraphic layer model at Bjerringbro, built with
GeoSigma from real Danish data: GERDA geophysics (SkyTEM, ground TEM, PACES) and
Jupiter boreholes. The example produces per-layer **variance maps**, the
GeoSigma v0.1.0 output.

Realizations drawn from these variance maps are a v1.0.0 preview: see
[`../preview/bjerringbro_realizations.py`](../preview/bjerringbro_realizations.py).

| | |
|---|---|
| Area | 10 × 10 km, x 536 000–546 000, y 6 250 000–6 260 000 (EPSG:25832) |
| Grid | 100 m, 100 × 100 cells |
| Model | 15 layer boundaries below terrain (`region_config.json`) |
| Data (Downloaded 2026/10/29) | 495 SkyTEM, 128 ground-TEM and 2 800 PACES soundings; 17 Jupiter wells (163 boundary picks, only wells deeper than 75 meters) |

## Layers

The model has 15 layers from FOHM Jylland (`data/FohmLayers.csv`). Each surface
is the **bottom** of a layer. Layers 1–10 are Quaternary and 11–15
pre-Quaternary, so the bottom of layer 10 (the base of the Quaternary) takes
the Quaternary correlation ranges and the pre-Quaternary ranges start at the
bottom of layer 11: `n_prereq = 10`, the 0-based index of that boundary in the
variance stack.

| Model layer | FOHM model layer | FOHM layer number | Name | Regime |
|---:|---:|---:|---|---|
| 1 | 1 | 10 | Soil (Jordbund) | Quaternary |
| 2 | 11 | 1100 | Quaternary clay | Quaternary |
| 3 | 12 | 1200 | Quaternary sand | Quaternary |
| 4 | 13 | 1300 | Quaternary clay | Quaternary |
| 5 | 14 | 1400 | Quaternary sand | Quaternary |
| 6 | 15 | 1500 | Quaternary clay | Quaternary |
| 7 | 16 | 2100 | Quaternary sand | Quaternary |
| 8 | 17 | 2200 | Quaternary clay | Quaternary |
| 9 | 18 | 2300 | Quaternary sand | Quaternary |
| 10 | 19 | 2400 | Quaternary clay | Quaternary |
| 11 | 64 | 7400 | Billund BDS2 | pre-Quaternary |
| 12 | 68 | 7800 | Billund BDS0 | pre-Quaternary |
| 13 | 69 | 8000 | Palaeogene clay | pre-Quaternary |
| 14 | 70 | 8500 | Danian limestone | pre-Quaternary |
| 15 | 71 | 9000 | Chalk (Skrivekridt) | pre-Quaternary |

## Run it

The example lives in `examples/bjerringbro/` of the GeoSigma repository. It
reads the model surfaces from `../data/Jutland_hydrostrat_model`; everything
else is in this folder. It runs from any working directory.

```bash
cd GeoSigma
pip install -e .              # once
python examples/bjerringbro/geosigma_dk_workflow.py
```

This takes about 15 seconds and writes:

| Output | Content |
|---|---|
| `output/variance_maps/variance_NN.tif` | Variance [m²] of boundary NN, the input for kriging/simulation |
| `output/variance_maps/std_NN.tif` | Standard deviation [m] for viewing in QGIS |
| `output/variance_maps/qa_overview.png` | Data coverage per source and final std |

All settings are in the `CONFIG` block at the top of `geosigma_dk_workflow.py`.

## Folder contents

```
examples/bjerringbro/
├── region_config.json        layer list and location of the model surfaces
├── data/
│   ├── FohmLayers.csv        model layer -> FOHM layer number and name
│   └── complexity_bjerringbro.tif   excerpt of the geological complexity map
├── geosigma_dk_workflow.py   main script: themes -> variance maps
├── geofysik/                 ThemeData bundles (points CSV + manifest) per method
│   ├── skytem/               SkyTEM            -> spec skytem
│   ├── fewtem/               ground TEM        -> spec fewtem
│   └── paces/                PACES             -> spec paces
├── boringer/
│   ├── jupiter.csv           Jupiter BOREHOLE extract (65 wells)
│   └── wells.csv             borehole input for the well theme
├── data_prep/
│   ├── gerda_db.py               reads GERDA exports (Access .mdb or Firebird .fdb)
│   ├── gerda_to_themedata.py     GERDA models -> ThemeData bundle
│   ├── wells_to_csv.py           Jupiter (+ interpretation points) -> wells.csv
│   └── crop_complexity.py        national complexity map -> data/ excerpt

```

## How the inputs were made

The geophysics comes from a GERDA Firebird export (`bjerringbro_geofysik.fdb`,
not included because of its size). Each bundle was made with commands like these, run from
`examples/bjerringbro/`:

```bash
python data_prep/gerda_to_themedata.py --list --gerda bjerringbro_geofysik.fdb --region-config region_config.json
python data_prep/gerda_to_themedata.py --gerda bjerringbro_geofysik.fdb --region-config region_config.json \
    --datatype tem --subtype skytem --spec skytem --name skytem --out geofysik/skytem
python data_prep/gerda_to_themedata.py --gerda bjerringbro_geofysik.fdb --region-config region_config.json \
    --datatype tem --spec fewtem --name fewtem --out geofysik/fewtem
python data_prep/gerda_to_themedata.py --gerda bjerringbro_geofysik.fdb --region-config region_config.json \
    --datatype paces --spec paces --name paces --out geofysik/paces
```

The well input was made from the Jupiter extract:

```bash
python data_prep/wells_to_csv.py --region-config region_config.json --jupiter boringer/jupiter.csv --out boringer/wells.csv
```

The complexity-map excerpt was cut from the national map:

```bash
python data_prep/crop_complexity.py --source ../../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd     --region-config region_config.json --out data/complexity_bjerringbro.tif
```

Each manifest's `_notes` records the source file and how many soundings were kept.

## Limitations

- **Borehole quality is a placeholder.** `jupiter.csv` has no quality class, so
  `wells_to_csv.py` assigned 2–4 from the drilling year. No interpretation points
  ("snapped" wells) are included, and no `--model-year` filter was applied, so
  wells drilled after the model was made are counted too.
- **The complexity map is a provisional proxy.** `data/complexity_bjerringbro.tif`
  is an excerpt of the national map (the model grid plus 1 km), cut with
  `data_prep/crop_complexity.py`; see "Geological complexity map" below and
  `docs/theme_format.md`.
- **The model-quality class is uniform** (class 2 everywhere), because no
  model-class grid was supplied.
- **Coverage is sparse.** Each geophysical method informs at most about 5 % of
  the cells per layer, so most of the area carries the model theme's background
  uncertainty (median std 6–28 m, increasing with depth).

## Data

Geophysical models: GERDA, GEUS. Boreholes: Jupiter, GEUS.
