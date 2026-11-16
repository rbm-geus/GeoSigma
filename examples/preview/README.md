# Preview demos (v1.0.0)

The scripts in this folder belong to the **simulation half** of GeoSigma,
which is planned for **v1.0.0**: estimating the local variogram, clustering
it into kriging zones, conditioning each layer boundary by local kriging,
drawing geostatistical realizations, and stacking the layers. The modules
behind them (`geosigma.preprocess`, `geosigma.kriging`, `geosigma.assembly`)
are implemented and unit-tested, and the demos run. They are previews
because the simulation is still under review: expect the scripts and their
APIs to change before v1.0.0.

**v0.1.0 ships the variance-map tool only.** For the released workflow, use
[`../demo_variance_map.py`](../demo_variance_map.py).

## The simulation and MATLAB

The simulation implements the method published in Madsen et al. (2026),
*Mathematical Geosciences* (doi
[10.1007/s11004-025-10263-9](https://doi.org/10.1007/s11004-025-10263-9)),
with deliberate improvements over the MATLAB implementation. Its output can
therefore **not be compared with MATLAB bit for bit**. For example, blending
overlapping kriging windows uses `variance_correction=True` by default;
`variance_correction=False` gives MATLAB's behaviour. All deviations are
listed in [`../../docs/deviations_from_matlab.md`](../../docs/deviations_from_matlab.md).

## Demos

| Script | What it shows | Needs |
|---|---|---|
| `demo_kriging_pipeline_end_to_end.py` | The whole simulation on two real boundaries of the sample model: INPOX points → range/sill estimation → SOM kriging zones → local kriging and realizations → layer stacking with crossing boundaries clamped. The variance maps are synthetic stand-ins. Takes about 5 minutes. | `nn` extra |
| `demo_train_rangesill_nn.py` | Trains the range/sill network from scratch on synthetic fields and compares it with the shipped default. This is the reproducible recipe behind `geosigma/preprocess/data/rangesill_default.pt`. Writes to `examples/output/rangesill_nn/`. Takes about 4 minutes. | `nn` extra |
| `demo_inpox_point_drawing.py` | The INPOX transfer function and how its parameters change which points are drawn | |
| `demo_inpox_basic_jutland_hydrostrat.py` | INPOX selection of conditioning points on the sample model | |
| `demo_geosigma_pipeline.py` | Surface → INPOX points → covariance model, using the core primitives directly. Stops once the prior covariance is assembled; see `demo_kriging_pipeline_end_to_end.py` for the full workflow. | |
| `demo_synthetic_kriging_inversion.py` | The kriging, least-squares and Cholesky primitives on a small synthetic grid | |

The `nn` extra is `pip install -e ".[nn]"` (torch and scipy). On Windows
with conda, see the known issue in the main [README](../../README.md).

Like the other demos, each one ends in a blocking `plt.show()` window. To run
one headless, set `MPLBACKEND=Agg`.
