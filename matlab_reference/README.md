# MATLAB Reference Code

This folder contains the original MATLAB source code for the N-ret hydrostratigraphic modelling workflow (`N-ret-mod---Source-main/`). It is included strictly as reference material for developers working on GeoSigma.

**Do not modify these files.** The authoritative implementation is in the `geosigma/` Python package.

The original code cannot be run from this repository. It has hardcoded paths to GEUS network drives (`N:\PROJEKTER\...`, `Y:\Stochastic_hydrostratigraphy\...`) and depends on data files that are not included here.

---

## What the Original Code Does

The MATLAB code generates multiple equally probable geostatistical realizations of the national-scale hydrostratigraphic model of Denmark. It covers five regions (Jylland, Fyn, Sjælland, Anholt-Læsø) with 8–49 geological layers each.

The workflow in `Main.m` proceeds as follows:

1. **Load surfaces** — Read layer boundary grids (ASCII / Surfer format) and well interpretations
2. **Build uncertainty maps** — For each layer, combine up to 12 spatial uncertainty "themes" (wells, SkyTEM, PACES, gamma logs, seismic, model class, etc.) into a single variance map using a harmonic mean
3. **Estimate variogram parameters** — Use a pre-trained neural network to predict local kriging range and sill from 33×33 mini-grids of the layer surface
4. **Cluster cells** — Group grid cells by similar range/sill; compute median range and 95th-percentile sill per cluster
5. **Krige** — Perform local kriging per cluster to produce a mean estimate and kriging error
6. **Generate realizations** — Draw stochastic realizations via Cholesky decomposition, apply well tie constraints, and assemble layer-consistent 3D models
7. **Export** — Write realizations as ASCII grids

---

## Translation Status

Update this table as translation progresses.

### Core algorithms → `geosigma/`

| MATLAB file | What it does | Python equivalent |
|---|---|---|
| `get_reals_cholesky.m` | Realization generation via Cholesky | `geosigma/get_reals_cholesky.py` ✓ |
| `certainty_function_definer.m` | Named certainty-decay functions | `certainty_function_definer.py` (to be moved into package) |
| `local_kriger.m` | Local kriging per cluster | `geosigma/local_kriging_setup_img.py` (partial) |
| `get_interpolated_minigrids_parallel.m` | ML-based range/sill prediction | not yet translated |
| `prepare_clusters.m` | Cluster statistics (median range, P95 sill) | not yet translated |
| `setup_GLOBAL_structure.m` | Kriging input structure | not yet translated |
| `tie_realizations.m` | Apply well tie constraints | not yet translated |
| `get_layered_models.m` | Assemble 3D layer cube | not yet translated |
| `merge_layers.m` | Blend kriged model with background | not yet translated |
| `add_bottom.m` | Append base chalk layer | not yet translated |
| `normalize_UTM_coord.m` | Coordinate normalization | not yet translated |

### Grid I/O → `utils/`

| MATLAB file | What it does | Python equivalent |
|---|---|---|
| `dtm_read_plot.m` | Read ESRI ASCII grid | `utils/load_esri_ascii_grid.py` ✓ |
| `ReadSurfer7.m` | Read Surfer 7 binary GRD | `ReadSurfer7.py` (to be moved into utils/) |
| `write_grid_ascii.m` | Write ESRI ASCII grid | not yet translated |

### Uncertainty themes → to be organised into package

| MATLAB file | What it does | Python equivalent |
|---|---|---|
| `get_well_theme.m` | Well-based certainty map | not yet translated |
| `get_SkyTEM_theme.m` | SkyTEM geophysics certainty | not yet translated |
| `get_PACEP_theme.m` | PACES geophysics certainty | `get_PACES_theme.py` (root, to be refactored) |
| `get_MEP_theme.m` | Magnetoelastic certainty | not yet translated |
| `get_GAMMALOG_theme.m` | Gamma log certainty | `get_GAMMALOG_theme.py` (root, to be refactored) |
| `get_RESLOG_theme.m` | Resistivity log certainty | `get_RESLOG_theme.py` (root, to be refactored) |
| `get_REFSEIS_theme.m` | Seismic reflector certainty | `get_REFSEIS_theme.py` (root, to be refactored) |
| `get_fewTEM_theme.m` | Sparse TEM certainty | `get_fewTEM_theme.py` (root, to be refactored) |
| `get_manyTEM_theme.m` | Dense TEM certainty | `get_manyTEM_theme.py` (root, to be refactored) |
| `get_tTEM_theme.m` | tTEM certainty | `get_tTEM_theme.py` (root, to be refactored) |
| `get_modeltheme.m` | Model quality class certainty | not yet translated |
| `get_PL_themes_ILM.m` | Peatland layer themes | not yet translated |
| `import_complexitymap.m` | Load geological complexity map | `import_complexitymap.py` (root, to be refactored) |

### Denmark-specific orchestration — not translated

| MATLAB file | What it does |
|---|---|
| `Main.m` | Main workflow orchestrator (1557 lines) |
| `Run_Batches_Final.m` | Batch runner for all regions |
| `landsdel_switch.m` | Region routing (Jylland / Fyn / Sjælland / …) |
| `get_layer_names.m` | Danish geological layer names |
| `get_wells.m` | Load well interpretation data |
| `get_layermasks.m` | Create onshore / layer-exists masks |
| `prepare_for_kriging.m` | Single-layer preprocessing |
| `prepare_wells.m` | Filter and structure well data |
| `exportmodels.m` | Export realizations as ASCII grids |

---

## Key Concepts for Translators

**Uncertainty combination** — All theme variance grids are combined using the harmonic mean:
```
final_variance = 1 / sum(1 / theme_variance_i, over all themes i)
```
Many certain data sources drive the result toward low variance; a single uncertain source has limited effect.

**Variogram parameter estimation** — A pre-trained MATLAB neural network (`FinalNetworkSillRangeBothNoNormalization.mat`) predicts the local kriging range and sill from a 33×33 grid (100 m cells, ~3.2 km extent) of layer elevation values. Translating this requires exporting the network weights and re-implementing inference in Python (e.g. via ONNX, or retraining with scikit-learn / PyTorch).

**GLOBAL struct** — The central data structure passed through the kriging workflow. Fields include grid geometry (`UTMX`, `UTMY`, `Nx`, `Ny`), background model surface (`img_FOHM`), kriging parameters per cell (`img_ranges`, `img_variances`), and conditioning point locations (`xp`, `yp`).

**Data files not included** — The `.mat` files for geophysical uncertainty themes and the `.grd` / `.asc` layer surface grids reside on GEUS network storage and are not part of this repository.
