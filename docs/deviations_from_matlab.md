# Deviations from the MATLAB reference

GeoSigma started as a translation of the MATLAB code behind the national
hydrostratigraphic model of Denmark (`matlab_reference/`). The two halves of
the library relate to that code differently:

- **The variance-map tool (v0.1.0, `geosigma.themes`)** reproduces the MATLAB
  output. The built-in themes are validated cell by cell against MATLAB
  reference grids (see *Validation status* in the [README](../README.md)).
- **The simulation (v1.0.0: `geosigma.preprocess`, `geosigma.kriging`,
  `geosigma.assembly`)** implements the method published in
  Madsen et al. (2026), *Mathematical Geosciences*, with **deliberate
  improvements**. Most of them depart from the MATLAB implementation; one,
  the merging of realizations from the cluster areas (`variance_correction`),
  departs from the published method itself. The output can therefore **not
  be compared with MATLAB bit for bit**. Where an improvement has a switch,
  the default is the improved behaviour and the original behaviour is
  available on request (see the table).

This page lists every deliberate deviation, and separately the corrections
that brought earlier Python code back in line with MATLAB.

## Summary

| Area | GeoSigma | Departs from | Get the original behaviour |
|---|---|---|---|
| Range/sill estimation | Network trained from scratch, predicts range only; sill in closed form | MATLAB: pretrained network predicts range and sill | Not possible (MATLAB network cannot be loaded) |
| Scattered-point interpolation | Cubic | MATLAB: natural neighbour | Not available |
| Edge windows | Reflection-padded, full window everywhere | MATLAB: windows near the edge cropped or skipped | Not available |
| Merging realizations from the cluster areas | `variance_correction=True` (default) | Published method (Madsen et al. 2026), as implemented in `local_kriger.m`: no correction | `variance_correction=False` (same algorithm; realizations differ, see below) |
| SOM clustering | Standard Kohonen SOM, new implementation | MATLAB: `selforgmap` (Deep Learning Toolbox) | Not possible (proprietary internals) |
| INPOX transfer function | Published parameters (Madsen et al. 2024, Table 1, T4) | MATLAB: undocumented literals in `drawpoints.m` | Pass the MATLAB values in `ext_vals` (same transfer function; drawn points differ, see below) |
| Layer stacking | No pre-Quaternary binding branch | MATLAB: region-specific branch in `get_layered_models.m` | Out of scope (region-specific) |

## Deliberate deviations (simulation, v1.0.0)

### Range/sill estimation: retrained network, sill without network bias

Module: `geosigma.preprocess.rangesill`. MATLAB:
`get_interpolated_minigrids_parallel.m` with the pretrained network
`FinalNetworkSillRangeBothNoNormalization.mat`.

Reference for the range/sill network and the sill without noise bias:
Falk & Madsen (2023), *GEUS Bulletin* 53 (see References).

> **To be confirmed by Frederik:** the description of the sill (a sill
> without noise bias, i.e. a network trained without noise) and the code
> (the sill as a closed-form sample variance, see below) point to two
> different mechanisms. The wording below is kept until this is confirmed.

- **The network is retrained, not converted.** The MATLAB network cannot be
  loaded outside MATLAB, so GeoSigma ships a network trained from scratch on
  synthetic Gaussian fields (`geosigma/preprocess/data/rangesill_default.pt`).
  The full recipe is reproducible with
  `examples/preview/demo_train_rangesill_nn.py`.
- **The network predicts range only.** It sees each window normalized (mean
  removed, divided by its own standard deviation), so amplitude is removed
  before the network sees the data and range and sill are decoupled by
  construction.
- **The sill is computed in closed form, not predicted.** It is the sample
  variance of the scattered points inside the window, taken before
  interpolation. It therefore carries no prediction error or bias from the
  network. It is also not computed from the interpolated grid, which would
  understate it because interpolation smooths. One limitation is known and
  documented: when the range is large relative to the window, a sparse
  sample's variance under-recovers the nominal sill. That is sampling
  behaviour, not a bug.
- **The estimator is pluggable.** Any object with `.predict(grid)` can replace
  the network (`RangeSillEstimator` protocol), and precomputed range/sill maps
  can be supplied directly.

### Scattered-point interpolation: cubic instead of natural neighbour

MATLAB interpolates each window's scattered points with
`scatteredInterpolant(..., 'natural', 'linear')`. GeoSigma uses cubic
interpolation (`interpolate_scatter_points`), because natural-neighbour
interpolation is not available in the project's dependencies. The shipped
network is trained on cubic-interpolated windows, so training and inference
are consistent.

### Edge windows: reflection padding

`estimate_range_sill_surface` reflection-pads the surface by half a window,
so cells near the boundary get a full window. Without this, a 33-cell window
drops a border of about 16 cells, which is more than half the cells of a
100 × 100 grid.

### Merging realizations from the cluster areas: `variance_correction`

Module: `geosigma.kriging.simulation` (`blend_window_realizations`,
`simulate_layer`, `simulate_model`). Published method: Madsen et al. (2026),
*Mathematical Geosciences*. MATLAB: `local_kriger.m`.

**This is a deviation from the published method**, not only from its MATLAB
implementation. In the method, each cluster area (one connected component of
one cluster, with a buffer zone around it) is kriged and simulated on its
own, and the realizations of neighbouring areas are merged by a weighted
average where their buffer zones overlap. Because each area draws its
realizations independently, the merged draws have less variance than the
posterior standard deviation reported for the same cells: up to about 30 %
less where two areas overlap at equal weight.

With `variance_correction=True` (the **default**), the merged draws are
rescaled so that each cell's marginal variance matches the reported
posterior standard deviation. The posterior mean is unchanged.
`variance_correction=False` gives the published method's behaviour and uses
the same algorithm as `local_kriger.m`. The realizations are still not
identical to MATLAB's, because MATLAB and Python use different random-number
generators.

*Documentation of the improvement follows.*

### SOM clustering: a new implementation

Module: `geosigma.preprocess.som`, `geosigma.preprocess.som_clusters`.
MATLAB: `get_clusters.m` with `selforgmap` + `train`.

GeoSigma implements a standard Kohonen self-organizing map (competitive
learning with a shrinking Gaussian neighbourhood and learning rate) on the
same square topology. It is not a port of `selforgmap`: the toolbox's
internals (training schedule, neighbourhood function, weight initialization)
are proprietary and cannot be recovered from the calling code. The
surrounding pipeline (neighbourhood-mode cleanup with radii 4 and 32,
`maxclustsize = 5000`, representative statistics per cluster) follows
`get_clusters.m`. Cluster labels and weights will differ from MATLAB's.

### INPOX: the published parameters

Module: `geosigma.inpox`. MATLAB: `drawpoints.m`.

The transfer function follows Madsen et al. (2024), *GEUS Bulletin* 57,
eqs. 2–4, with the paper's symbols. The examples use the paper's real-world
case **T4** (`L0=1, A=1.5, B=3, C=5, D=1, E=0.8, g=2`). `drawpoints.m`
hardcodes a different set (in the same symbols: `L0=1, A=2.4, B=12, C=1.25,
D=0.25, E=0.09, g=0.8`). With those values the suppression branch never
activates on a typical surface, and the draw fraction rises only about 9 %
above `P0`, against about 45 % with T4. To use MATLAB's transfer function,
pass the MATLAB values in `ext_vals`. The drawn points will still not be
identical to MATLAB's, because the draw is random and MATLAB and Python use
different random-number generators.

### Layer stacking: no region-specific branch

Module: `geosigma.assembly.layers`. MATLAB: `get_layered_models.m`.

Stacking follows MATLAB: boundaries are clamped top-down, per realization,
so a layer pinches out instead of inverting. Two parts are deliberately not
translated: the pre-Quaternary binding branch, which is Danish
region-specific logic driven by `landsdel_switch`, and the onshore-mask test
for "not simulated here". Unsimulated cells are given as NaN instead.

## Corrections that now match MATLAB

These were defects in the earlier Python translation, fixed in the
simulation branch. They are **not** deviations from MATLAB: after the fix,
Python and MATLAB agree.

### Gaussian variogram: practical range (factor 3)

`geosigma.core.semivar_synth` now uses
`gamma = sill * (1 - exp(-3 (h / range)^2))`, so `range` is the practical
range (95 % of the sill). Before the fix, the factor 3 was missing, and the
effective correlation length was about 1.73 times too long in every kriging
solve.

Source: mGstat `precal_cov.m:111` calls `semivar_synth(V(iV),d,0)`, so
`gstat = 0`, which selects the GSLIB2/Goovaerts branch at
`semivar_synth.m:105`: `sgamma=v1.*(1-exp(-3*h.^2/v2.^2))`
(<https://github.com/AUProbGeo/mGstat>). The N-ret-mod code calls
`precal_cov` with `Gau` models in `local_kriging_setup_img.m:15,19` and
`local_kriger.m:262`.

Tests: `test_semivar_gaussian_uses_mgstat_practical_range` and
`test_precal_cov_gaussian_matches_mgstat_at_the_range` in
`tests/test_core_math.py`.

### Noise covariance: unit sill

`geosigma.core.local_kriging_setup_img` builds the shape of the correlated
observation noise as `1 Gau(range)`, as `local_kriging_setup_img.m:18` does.
Before the fix, it used the cluster variance as the sill, which applied that
variance twice.

Test: `test_local_kriging_noise_shape_has_unit_sill` in
`tests/test_core_math.py`.

### Themes: NaN handling in `max` (`np.fmax`)

Nine places in `geosigma.themes.themes` used `np.maximum` where MATLAB uses
`max`. NumPy propagates NaN, while MATLAB ignores it, so a NaN in one data
point could put the no-information value on cells that other data informed
well. They now use `np.fmax`, which has MATLAB's semantics. The themes
remain bit-exact against the MATLAB reference grids after this change.

## References

- Madsen, R. B., Falk, F. A., Møller, I., Troldborg, L., & Høyer, A.-S.
  (2026). Probabilistic Modeling and Uncertainty Simulations of
  National-Scale Hydrostratigraphic Models: An Example from Denmark.
  *Mathematical Geosciences*. <https://doi.org/10.1007/s11004-025-10263-9>
- Falk, F. A., & Madsen, R. B. (2023). Machine learning-based estimation and
  clustering of statistics within stratigraphic models as exemplified in
  Denmark. *GEUS Bulletin*, 53.
  <https://doi.org/10.34194/geusb.v53.8353> (reference for the range/sill
  network and the sill without noise bias)
- Madsen, R. B., Falk, F. A., Møller, I., & Høyer, A.-S. (2024).
  Introducing INPOX: a method for informed point extraction from geological
  2D surfaces exemplified on the Danish national hydrostratigraphic model.
  *GEUS Bulletin*, 57. <https://doi.org/10.34194/geusb.v57.8364>
