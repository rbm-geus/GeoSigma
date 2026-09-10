# GeoSigma

GeoSigma is an open-source Python library for geostatistical simulation of 3D subsurface layer boundaries. Each layer boundary is modelled as a spatially variable Gaussian field, enabling users to generate multiple equally probable hydrostratigraphic realizations from a deterministic input model and quantify spatial uncertainty at every grid location.

The library is designed to be model-agnostic: the same workflow applies to any hydrostratigraphic model, independent of region or data source.

## Capabilities

- Construct spatial variance maps and correlation structures for layer boundaries from multiple geophysical data themes (TEM, PACES, gamma logs, seismic reflectors, etc.)
- Condition each layer boundary by local kriging, then draw geostatistical realizations directly from the posterior covariance by Cholesky decomposition (`get_reals_cholesky`) — not by sequential simulation
- Export results as GeoTIFF rasters for use in downstream modelling workflows
- Integrate with PEST for parameter perturbation and ensemble-based uncertainty propagation

## Installation

GeoSigma is pure Python and its dependencies all ship binary wheels on PyPI, so
a plain `pip` install needs no compiler and no system GDAL.

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

The `examples/` directory contains standalone scripts demonstrating key workflows:

- `demo_geosigma_pipeline.py` — full workflow from surface input to geostatistical realizations
- `demo_synthetic_kriging_inversion.py` — synthetic kriging with masking
- `demo_inpox_basic_jutland_hydrostrat.py` — INPOX conditioning point selection on a real model

Example data (the Jutland hydrostratigraphic model) is included in `examples/data/`.

> **The demos are figure-driven, not log-driven.** Each script ends in a
> blocking `plt.show()` window and prints little or nothing to stdout, so a
> quiet terminal is the expected behaviour, not a hang or a failure — look for
> the plot window. Close it to let the script continue or exit. To run one
> headless (in CI, or over a connection with no display), force the
> non-interactive backend first:
>
> ```bash
> MPLBACKEND=Agg python examples/demo_synthetic_kriging_inversion.py
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
