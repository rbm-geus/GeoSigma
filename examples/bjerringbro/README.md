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
| Data (downloaded 2026-10-29) | 495 SkyTEM, 128 ground-TEM and 2 800 PACES soundings; 17 Jupiter wells (163 boundary picks, only wells deeper than 75 meters) |

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
│   └── complexity_bjerringbro.tif   complexity map on the model grid (excerpt)
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
│   ├── crop_complexity.py        national complexity map -> model grid
│   ├── model_class_to_grid.py    a model-quality class grid -> model grid
│   └── class_grids.py            shared shift / majority-vote helpers

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

The complexity map was put on the model grid from the national map:

```bash
python data_prep/crop_complexity.py --source ../../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd --order higher_is_more_complex --region-config region_config.json --out data/complexity_bjerringbro.tif
```

Each manifest's `_notes` records the source file and how many soundings were kept.

## Limitations

- **Borehole quality is a placeholder.** `jupiter.csv` has no quality class, so
  `wells_to_csv.py` assigned 2–4 from the drilling year. No interpretation points
  ("snapped" wells) are included, and no `--model-year` filter was applied, so
  wells drilled after the model was made are counted too.
- **The complexity map is a provisional proxy.** `data/complexity_bjerringbro.tif`
  is an excerpt of the national map on the model grid, made with
  `data_prep/crop_complexity.py`; see "Geological complexity map" below and
  `docs/theme_format.md`.
- **The model-quality class is uniform** (class 2 everywhere), because no
  model-class grid was supplied. A grid you add must lie on the model grid with
  a class in every cell, like the complexity map; the workflow checks this and
  stops otherwise. `data_prep/model_class_to_grid.py` puts a grid there (a
  shift for the same cell size, otherwise a majority vote in which the lower,
  more uncertain class wins ties).
- **Coverage is sparse.** Each geophysical method informs at most about 5 % of
  the cells per layer, so most of the area carries the model theme's background
  uncertainty (median std 6–28 m, increasing with depth).


## Data sources and credits

This example uses data from the sources below. Each source remains the
property of its provider and is used under the provider's terms; check those
terms before reusing the data.

### Geophysics: GERDA (GEUS)

The SkyTEM, ground-TEM and PACES models come from GERDA, the Danish national
geophysical database, maintained by the Geological Survey of Denmark and
Greenland (GEUS).

| | |
|---|---|
| Provider | GEUS |
| Database | GERDA |
| Export | Firebird export `bjerringbro_geofysik.fdb` (not included because of its size) |
| Download date | 2026-10-29 |
| GERDA version / export tool | https://data.geus.dk/geusmap/?mapname=gerda |
| Area | x 536 000–546 000, y 6 250 000–6 260 000 (EPSG:25832) |
| Used | 495 SkyTEM, 128 ground-TEM and 2 800 PACES soundings (after the filtering recorded in each manifest's `_notes`) |
| Availability | Publicly available data |
| Terms | Freely available for download from GEUS; see "Conditions for data re-use" in the GERDA metadata record: https://data.geus.dk/geonetwork/srv/api/records/a1b21c8e-2276-4386-abd2-3ef4ca5698b8 |
| Redistribution | Redistribution of this excerpt in GeoSigma confirmed by the GERDA database owner at GEUS (2026-11-17) |

### Boreholes: Jupiter (GEUS)

The boreholes come from Jupiter, the Danish national borehole database,
maintained by GEUS.

| | |
|---|---|
| Provider | GEUS |
| Database | Jupiter, table BOREHOLE |
| Columns | `BOREHOLENO`, `XUTM`, `YUTM`, `ELEVATION`, `DRILLDEPTH`, `DRILLENDDATE` |
| Download date | 2026-10-29 |
| Export tool | https://data.geus.dk/geusmap/?mapname=jupiter |
| Selection | Boreholes with a drill depth of ≥ 75 m inside x 532 260–549 700, y 6 247 225–6 263 500 (EPSG:25832). The extract `boringer/jupiter.csv` holds 65 boreholes; 17 of them lie inside the model grid and give 163 boundary picks. |
| Availability | Publicly available data |
| Terms | http://pubs.geus.net/betingelser/Jupiter_betingelser.pdf |

Jupiter does not provide a borehole quality rating in this extract. The
example therefore uses a **placeholder rule based on drilling year**
(`data_prep/wells_to_csv.py`): quality 2 for boreholes drilled in 1980 or
later, 3 for 1950–1979, and 4 if the year is unknown (quality 1 is not used).
GeoSigma's convention is 1 = best … n = worst. This is an illustrative
choice, not measured borehole quality; replace it with real ratings where
available, and check the direction of the scale in the source data. The
borehole uncertainty tables (`Ks`, `gradients`) are the Danish values from
the original MATLAB code (`get_well_theme.m`, 2023-06-27), given as an
example, not as defaults.

The borehole search area is larger than the model grid (x 536 000–546 000,
y 6 250 000–6 260 000). [TODO: confirm the reason, e.g. a buffer so that
boreholes just outside the grid are included.]

The extract was made with this query. [TODO: confirm the spelling of the
end-date column in the query as run (`drillenddate` here, `drilendate` in an
earlier draft; the extract's header is `DRILLENDDATE`).]
```sql
SELECT
      [boreholeno],
      [drilldepth],
      [elevation],
      [xutm],
      [yutm],
      [drillenddate],
      [GEOM]
FROM [pcjupiterxl].[dbo].[borehole]
WHERE
      [drilldepth] >= 75
      AND [GEOM].STWithin(
          geometry::STGeomFromText(
              'POLYGON((
                  532260 6247225,
                  549700 6247225,
                  549700 6263500,
                  532260 6263500,
                  532260 6247225
              ))',
              25832
          )
      ) = 1
      ORDER BY boreholeno;
```

### Hydrostratigraphic layer model

The 15 layer boundaries are read from
`../data/Jutland_hydrostrat_model` (see `region_config.json`).

| | |
|---|---|
| Model | FOHM Jylland, version [TODO: version as downloaded] [TODO: original FOHM or the locally updated version used for the nitrogen retention map 2025?] |
| Provider | Miljøstyrelsen (Danish Environmental Protection Agency, MST) |
| Distributed by | GEUS (official versions) |
| Download date | 2026-03-20 |
| Availability | Publicly available data |
| Citation | Miljøstyrelsen (2020). FOHM – Fælles Offentlig Hydrologisk Model. [TODO: full reference as given in GEUS reports] |

### Geological complexity map

The correlation ranges depend on a geological complexity map. The example
ships an excerpt of the national map, `data/complexity_bjerringbro.tif`, on
exactly the model grid (100 × 100 cells of 100 m). GeoSigma does not adapt a
complexity map to the model: the workflow reads the map and stops with an
error if it does not lie on the model grid. Putting it there is a preparation
step, done once by `data_prep/crop_complexity.py`:

- **Same cell size** (as here: both 100 m). The national map's cell edges lie
  50 m off the model's, so the map is shifted +50 m east and north, as in
  MATLAB's `import_complexitymap.m`, and each model cell takes the class of the
  map cell 50 m to its south-west. Classes are copied one to one, never mixed.
- **Different cell size.** Each model cell takes the most frequent class of
  4 × 4 sample points (a majority vote); a tie goes to the most complex class,
  which gives the shorter range and so the more conservative uncertainty.

Before this, the workflow did a majority vote itself at every run. The
shifted copy gives a different class in 216 of the 10 000 model cells (186
from 3 to 2, 30 from 2 to 3). The final variance maps are unchanged, because
the geophysics manifests use `complexity: none` and no borehole lies in a
changed cell.
It is a provisional proxy; see the limitations above and `docs/theme_format.md`.

The map divides onshore Denmark into four classes of expected geological
complexity. The report numbers them with a C-index from high (C-index 1) to
low (C-index 4). The map file is coded the other way round: its value is
5 − C-index, so in the file 1 = low and 4 = high complexity. That is also
GeoSigma's internal order (a higher class is more complex), so the workflow
declares `"complexity_order": "higher_is_more_complex"` in `CONFIG` and the
values are used as they are. The setting is required whenever a complexity
grid is given; a map coded like the report would be declared
`"lower_is_more_complex"` with `"complexity_n_classes": 4` and reversed on
loading. The map describes roughly the uppermost 30 m of the subsurface, is a
basic, low-resolution map (not intended for scales finer than about
1:300 000) and is described by its author as preliminary. This is why
GeoSigma treats a complexity map as optional and why it is used here for the
shallow layers only.

| | |
|---|---|
| Producer | GEUS (P.B.E. Sandersen) |
| Version / date | As published in GEUS Rapport 2021/37 (the map is unchanged since). The file used here is the June 2023 export from the national MATLAB reference data. |
| Permission | Used and redistributed (excerpt) with the permission of the author, Peter B.E. Sandersen (GEUS) |
| Citation | Sandersen, P.B.E. (2021). A basic geological complexity map for use in the implementation of the MapField concept. GEUS Rapport 2021/37. Geological Survey of Denmark and Greenland. https://data.geus.dk/pure-pdf/GEUS-R_2021_37_web.pdf |
| Used in | Hansen, B., Aamand, J., Blicher-Mathiesen, G. et al. (2024). Assessing groundwater denitrification spatially is the key to targeted agricultural nitrogen regulation. *Scientific Reports* 14, 5538. https://doi.org/10.1038/s41598-024-55984-9 |
