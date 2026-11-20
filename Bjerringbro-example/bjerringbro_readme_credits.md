
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
| Download date | 2026/10/29 |
| GERDA version / export tool | https://data.geus.dk/geusmap/?mapname=gerda |
| Area | x 536 000–546 000, y 6 250 000–6 260 000 (EPSG:25832) |
| Used | 495 SkyTEM, 128 ground-TEM and 2 800 PACES soundings (after the filtering recorded in each manifest's `_notes`) |
| Availability | Publicly available data |

### Boreholes: Jupiter (GEUS)

The boreholes come from Jupiter, the Danish national borehole database,
maintained by GEUS.

| | |
|---|---|
| Provider | GEUS |
| Database | Jupiter, table BOREHOLE |
| Columns | `BOREHOLENO`, `XUTM`, `YUTM`, `ELEVATION`, `DRILLDEPTH`, `DRILLENDDATE` |
| Download date | 2026/10/29 |
| Export tool | https://data.geus.dk/geusmap/?mapname=jupiter |
| Selection | Boreholes with a drill depth of ≥ 75 m inside x (532 260-549 700), y (6 247 225-6 263 500) (EPSG:25832). The extract `boringer/jupiter.csv` holds 65 boreholes; 17 of them lie inside the model grid and give 163 boundary picks. |
| Availability | Publicly available data |

The extract was made with this query:
```sql
SELECT
      [boreholeno],
      [drilldepth],
      [elevation],
      [xutm],
      [yutm],
      [drilendate],
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
`../examples/data/Jutland_hydrostrat_model` (see `region_config.json`).

| | |
|---|---|
| Model | FOHM jylland |
| Source / provider | FOHM|
| Download or delivery date | 2026/3/20 |
| Availability | Publicly available data |

### Geological complexity map

The correlation ranges depend on a geological complexity map, read from
`../matlab_reference/N-ret-mod---Source-main/Kompleksitetskort.grd` (the
national MATLAB reference data). It is a provisional proxy; see the
limitations above and `docs/theme_format.md`.

| | |
|---|---|
| Producer | Hansen, B., Aamand, J., Blicher-Mathiesen, G. et al., 2024 |
| Version / date | 2026/06/15 |
| Citation | Hansen, B., Aamand, J., Blicher-Mathiesen, G. et al. Assessing groundwater denitrification spatially is the key to targeted agricultural nitrogen regulation. Sci Rep 14, 5538 (2024). https://doi.org/10.1038/s41598-024-55984-9 |

