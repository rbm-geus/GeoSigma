# GeoSigma

GeoSigma is an open-source Python library for geostatistical uncertainty modelling of 3D subsurface layer boundaries. Each layer boundary is treated as a spatially variable Gaussian field: GeoSigma quantifies the uncertainty of every boundary at every grid location and, once the simulation half is complete, will generate multiple equally probable hydrostratigraphic realizations from a deterministic input model.

The library is designed to be model-agnostic: the same workflow applies to any hydrostratigraphic model, independent of region or data source.

## Quick start

```bash
pip install -e .                       # from the repository root; see Installation
python examples/demo_variance_map.py   # builds per-layer variance maps end to end
```

## Release scope: v0.1.0 and v1.0.0

GeoSigma is being built in two halves. **v0.1.0 contains only the first half, the variance-map tool.**

| | **v0.1.0 (this release)** | **v1.0.0 (planned)** |
|---|---|---|
| **What you get** | Per-layer **variance maps** of every layer boundary, built from your information sources (geophysics, boreholes, …) | Geostatistical **realizations** of the full layer model, conditioned using those variance maps |
| **Included** | Theme engine (`geosigma.themes`): built-in and YAML-defined themes, the ThemeData input format, parallel-precision combine, minimum-variance floor, GeoTIFF export · grid I/O (GeoTIFF, ESRI ASCII, Surfer 7) · `RegionConfig` · the translated geostatistical primitives (`precal_cov`, `least_squares_inversion`, `get_reals_cholesky`, INPOX, …) | Preprocessing (range/sill estimation, clustering) · the local kriging driver · realization generation on a whole model · layer assembly (zero-thickness ties, stratigraphic order) and export |
| **Demos** | [`examples/`](examples/): run end to end | [`examples/preview/`](examples/preview/): incomplete previews |

The variance maps are a self-contained product. The kriging stage will take a per-layer variance-grid stack as input, whether it comes from GeoSigma's theme tool or from your own uncertainty method.

### Validation status (v0.1.0)

The built-in themes were run on the same input data as the original MATLAB implementation of the national hydrostratigraphic model of Denmark and compared cell by cell with the MATLAB reference grids, on spatial windows of the Jutland region. The comparison harness uses the full national datasets and is not part of this repository. Some reference grids in that MATLAB run were cached from older runs on older input data; where that limits a comparison, the table says so.

| Theme | Information source | Status |
|---|---|---|
| fewTEM | Ground-based TEM, few-layer models | ✅ Bit-exact (max \|Δ\| 7e-12; 2 of 294,654 cells differ at the window edge from complexity-map padding) |
| manyTEM | Ground-based TEM, many-layer models | ✅ Bit-exact (max \|Δ\| 1.8e-12) |
| SkyTEM | Airborne TEM | ✅ Bit-exact (max \|Δ\| 1.5e-11) |
| PACEP | Pulled-array continuous electrical profiling | ✅ Bit-exact (max \|Δ\| 1.4e-14) |
| MEP | Multi-electrode profiling | ✅ Bit-exact (max \|Δ\| 8.5e-14, including Wenner-array data) |
| REFSEIS | Reflection seismics | ✅ Bit-exact (max \|Δ\| 2.9e-11) where the reference and the input come from the same dataset. The reference was cached before deeper seismic picks were added to the input, so layers resolved only by those picks could not be compared. |
| GAMMALOG | Gamma borehole logs | ✅ Bit-exact (max \|Δ\| 9e-16) |
| RESLOG | Resistivity borehole logs | ✅ Bit-exact (max \|Δ\| 2e-13) |
| Wells | Lithological borehole descriptions | ◐ **Partly validated.** Snapped wells: coverage matches, and values match exactly where the reference used the same complexity map. The Jupiter-borehole contribution is missing from the reference, so it is untested. |
| tTEM | Towed TEM | ◐ **Partly validated.** No-data handling matches inside the combined map. No tTEM reference grid exists and the comparison window held no tTEM data, so informative values are untested. |
| PACES | Pulled-array continuous electrical sounding | ⬜ **Not validated.** No PACES reference grid exists, and the combined reference was built without the current PACES contribution. |
| **Combine + floor** | All themes → final variance map | ✅ Matches to machine precision (median relative difference 0) wherever the reference is current. The remaining 2.3 % of cells come from the cached PACES and REFSEIS references. |

"Bit-exact" means agreement to floating-point round-off (max absolute difference ≤ 1e-10 m²). Closing the open rows needs a fresh MATLAB reference run on current data. The geostatistical primitives that v1.0.0 builds on are covered by unit tests; the simulation workflow itself is not yet validated.

## Capabilities

- **v0.1.0:** Construct per-layer variance maps for layer boundaries from multiple information sources (TEM, PACES, gamma and resistivity logs, seismic reflectors, boreholes, or your own source defined in a YAML spec)
- **v0.1.0:** Export variance maps as GeoTIFF rasters for use in downstream modelling workflows
- **v1.0.0:** Condition each layer boundary by local kriging, then draw geostatistical realizations directly from the posterior covariance by Cholesky decomposition (`get_reals_cholesky`), not by sequential simulation
- **Experimental:** Integrate with PEST for parameter perturbation and ensemble-based uncertainty propagation (to be reviewed together with the simulation for v1.0.0)

## Installation

GeoSigma is pure Python and its dependencies all ship binary wheels on PyPI, so
a plain `pip` install needs no compiler and no system GDAL. **`pip install -e .`
is required:** the demos import `geosigma`, `utils` and `plotting` as installed
packages, so without it they only work when started from the repository root.

**With pip (any virtual environment):**

```bash
git clone https://github.com/rbm-geus/GeoSigma.git
cd GeoSigma
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

A cold install from PyPI takes roughly two minutes, mostly spent downloading
`numpy`, `pandas`, `matplotlib` and `rasterio`.

**With conda** (use this if you want the GDAL/PROJ stack managed by conda, e.g.
to share it with other geospatial tools):

```bash
git clone https://github.com/rbm-geus/GeoSigma.git
cd GeoSigma
conda env create -f environment.yml
conda activate geosigma
pip install -e .
```

`environment.yml` is the authoritative conda dependency declaration;
`pyproject.toml` declares the pip dependencies.

### Check the install

```bash
python -c "import geosigma, importlib.metadata as m; print('geosigma', m.version('geosigma'))"
python -c "from geosigma.themes import build_theme, ThemeSpec; print('themes OK')"
```

Running `pytest` from the repository root runs the full test suite (148 tests,
a few seconds); it needs the `dev` extra: `pip install -e ".[dev]"`.

## Getting Started

The `examples/` directory contains standalone v0.1.0 scripts. They locate their
data relative to the script file, so they run from any working directory.

- `demo_variance_map.py`: **start here.** Runs the variance-map tool end to end: two synthetic information sources on the sample model → per-layer themes → combined, floored variance map → GeoTIFFs in `examples/output/demo_variance_map/`. The data points are synthetic and the theme specs are illustrative, not a calibrated uncertainty model.
- `demo_region_config.py`: describe a model's layers and surface files with a `RegionConfig`.
- `demo_load_hydrostrat_model_geotiff.py`, `demo_load_jutland_hydrostrat_model.py`, `demo_load_hydrostrat_layer_geotiff.py`: load and plot the sample Jutland model (GeoTIFF).
- `demo_load_funen_hydrostrat_model.py`: load a model supplied as ESRI ASCII grids (Funen).
- `theme_specs/`: example YAML theme specs. `dronetem.spec.yaml` describes a hypothetical, illustrative source.

[`examples/preview/`](examples/preview/) holds **incomplete v1.0.0 previews** of the simulation workflow (conditioning, realizations, INPOX point extraction). They do not run end to end; see the README there.

Example data (sample Jutland and Funen hydrostratigraphic models) is included in `examples/data/`.

> **The demos are figure-driven.** Each script ends in a blocking
> `plt.show()` window, so look for the plot window and close it to let the
> script exit. To run one headless (in CI, or over a connection with no
> display), force the non-interactive backend first:
>
> ```bash
> MPLBACKEND=Agg python examples/demo_variance_map.py
> ```

## Documentation

- [`docs/theme_format.md`](docs/theme_format.md) — the ThemeData three-file format: the portable, model-agnostic bundle (points CSV, manifest, optional complexity grid) in which a theme's input data are supplied and archived.
- [`docs/bring_your_own_data.md`](docs/bring_your_own_data.md) — the scientific-choices guide: the twelve modelling decisions behind a theme, how to reason about each for your own region and data, and a complete worked example.

## Background

GeoSigma is developed at the Geological Survey of Denmark and Greenland (GEUS) as a clean Python reimplementation of a MATLAB-based workflow originally developed for generating stochastic realizations of the national-scale hydrostratigraphic model of Denmark. The original MATLAB source is included in `matlab_reference/` for reference and is not intended to be run directly.

Some core functions are adapted from the open-source [mGstat](https://github.com/AUProbGeo/mGstat) MATLAB library (T.M. Hansen), which is MIT-licensed; those files carry individual credit notices.

## License

ISC License — see [`LICENSE`](LICENSE).

Third-party components, and the mGstat attribution required by its MIT license, are documented in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
