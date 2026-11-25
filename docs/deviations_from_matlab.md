# Deviations from the MATLAB reference

GeoSigma started as a translation of the MATLAB code behind the national
hydrostratigraphic model of Denmark (`matlab_reference/`). The two halves of
the library relate to that code differently:

- **The variance-map tool (v0.1.0, `geosigma.themes`)** reproduces the MATLAB
  output. The built-in themes are validated cell by cell against MATLAB
  reference grids (see *Validation status* in the [README](../README.md)).
  Errors in the MATLAB themes are not reproduced by default; they are
  listed under *MATLAB errors not reproduced* below.
- **The simulation (v1.0.0: `geosigma.preprocess`, `geosigma.kriging`,
  `geosigma.assembly`)** implements the method published in
  Madsen et al. (2026), *Mathematical Geosciences*, with **deliberate
  improvements**. Most of them depart from the MATLAB implementation; one,
  the merging of realizations from the cluster areas (`variance_correction`),
  departs from the published method itself. The output can therefore **not
  be compared with MATLAB bit for bit**. Where an improvement has a switch,
  the default is the improved behaviour and the original behaviour is
  available on request (see the table).

This page lists every deliberate deviation, the MATLAB errors the themes do
not reproduce, and separately the corrections that brought earlier Python code
back in line with MATLAB.

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
| REFSEIS thickness mask (themes) | Thin-layer mask applied to the cells it describes | MATLAB: mask applied at shifted linear indices (`get_REFSEIS_theme.m`) | Not available (validation harness only) |
| Unknown complexity (themes) | Class 0 blanked in every layer | MATLAB: blanked in the first modelled layer only (eight themes) | Not available (validation harness only) |
| Borehole-log mask (themes) | GAMMALOG/RESLOG mask boundaries below the log's reach (`depth > doi`, `doi` = `maxdepth`) | MATLAB: `thick < depth`, which never fires on the national data (`get_GAMMALOG_theme.m`, `get_RESLOG_theme.m`) | Not available (validation harness only) |
| Certainty function (themes) | Danish specs use `kernel_with_point_tail` | MATLAB national run: `ILM_sep2023` (sign error); also `RBM_oct2023` (jump at the plateau edge) | `cert_fun_name="ILM_sep2023"` (or `"RBM_oct2023"`) with `allow_legacy_cert_fun=True` |

## Deliberate deviations (simulation, v1.0.0)

### Range/sill estimation: retrained network, sill without network bias

Module: `geosigma.preprocess.rangesill`. MATLAB:
`get_interpolated_minigrids_parallel.m` with the pretrained network
`FinalNetworkSillRangeBothNoNormalization.mat`.

Reference for the range/sill network: Falk & Madsen (2023), *GEUS Bulletin*
53 (see References).

Two separate points, confirmed by Frederik Falk:

- **The sill is the sample variance.** In its current form the estimator
  does not predict the sill; it uses the sample variance of the window's
  points (details below).
- **"Trained without noise" refers to the training data, not the sill.** Each
  realization used as a training example is no longer contaminated by
  components from another realization.

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

## MATLAB errors not reproduced (variance maps, v0.1.0)

Three errors in the MATLAB theme scripts change which cells a theme informs,
and two MATLAB certainty functions are flawed. GeoSigma does what the MATLAB
code evidently intends. The private validation harness emulates the three
errors and uses the legacy certainty functions, so the comparison against
the MATLAB grids stays cell by cell; with the emulation, REFSEIS on Zealand
is bit-exact (see the [README](../README.md)).

### REFSEIS: the thickness mask lands on the wrong cells

Module: `geosigma.themes.themes.refseis_spec`. MATLAB:
`get_REFSEIS_theme.m`, `toi_filter = local_model_thick < 100;
Grid(toi_filter) = 100000;`.

The theme blanks cells whose nearest seismic point reports a layer thinner
than 100 m. In MATLAB, `local_model_thick` is not preallocated (REFSEIS is
the only theme where this is the case). It grows by assignment inside the
`parfor`, so its size is (last row with data, last column with data,
deepest active layer), smaller than `Grid` whenever the data do not reach
the north or east edge of the grid. Cells never assigned hold 0, so they
count as thin. `Grid(toi_filter)` with a logical mask smaller than `Grid`
uses the mask's true positions as linear indices into `Grid`, so the mask
is applied to the wrong cells, and mostly to the wrong layers.

On Zealand, the mask lands only in layers 1–8, never on the active
seismic layers 10–12, so MATLAB applies no thickness mask to them at all.
On Jutland it lands in layers 1–28, so the deepest seismic layers get no
thickness mask either. GeoSigma applies the mask to the cells and layers it
describes.

### Unknown complexity is blanked in the first layer only

Module: `geosigma.themes.base` (class 0 -> no information). MATLAB:
`complexity2(complexity==0) = NaN;` in `get_MEP_theme.m`,
`get_PACEP_theme.m`, `get_PACES_theme.m`, `get_REFSEIS_theme.m`,
`get_TEM_theme.m`, `get_fewTEM_theme.m`, `get_manyTEM_theme.m` and
`get_tTEM_theme.m`.

`complexity2` is a 3D array (one complexity grid per layer), but the mask
`complexity==0` is 2D, so it selects cells in the first layer only. In the
deeper layers, cells with unknown complexity keep class 0 and get the
range at index 0 of the range table. `get_SkyTEM_theme.m` uses the 3D mask
`complexities==0` and blanks every layer; `get_GAMMALOG_theme.m` and
`get_RESLOG_theme.m` work on 2D grids, so the question does not arise
there. GeoSigma blanks class 0 in every layer, as SkyTEM does. The two
differ only at cells with unknown complexity (outside the complexity map)
in layers below the first.

### Borehole logs: the depth mask never fires

Module: `geosigma.themes.themes.gammalog_spec` and `reslog_spec`. MATLAB:
`get_GAMMALOG_theme.m` and `get_RESLOG_theme.m`,
`doi_filter = local_model_thick < local_depth; Grid(doi_filter) = 100000;`.

The mask is meant to drop boundaries deeper than the log reaches. Both scripts
read the log's reach (`DOIs = table2array(MEP.InfoTable(:,"maxdepth"))`) but
never use it; the mask compares the boundary depth with the thickness array
instead. In the national inputs that array models a log as one layer from the
surface to `maxdepth`: `thick` equals `maxdepth` for every boundary at or above
it and is 10 000 (the half-space code) for every boundary below it. A boundary
below the log therefore gets `thick = 10000`, which is never less than its
depth, and the mask never fires. Every boundary within the search radius of a
log, however deep, is informed by it with the log's full certainty (gamma
`var0 = 2.25 m²`; resistivity `var0 = 0.25 + (0.045 depth)²`).

GeoSigma gives each log its reach as the per-point attribute `doi`
(`maxdepth`) and masks `depth > doi`; where several logs tie at the nearest
distance, the shallowest reach is used. Masked cells get the no-information
value, like every other mask.

On Jutland (national run, MATLAB kernel, full grid) this removes log
information from 31 781 cell-layers, at most 1 505 cells in any one layer. In
those cells the median final combined standard deviation rises from 2.1 m to
21 m (95th percentile of the increase 31 m, maximum 48 m).

Module: `geosigma.themes.certainty_functions`. MATLAB:
`certainty_function_definer.m`.

**`ILM_sep2023` has a sign error.**

```matlab
% ILM_sep2023
(dist<=width).*sill.*(exp(-(3*power((dist-width),2))./(power(range,2))))
% ILM_oct2023 (the MATLAB correction)
(dist<=width).*sill.*(exp(-(3*power((dist),2))./(power(range,2))))
```

Inside the plateau (`dist <= width`), `dist - width` is zero or negative.
The decay is therefore measured from the plateau edge inwards: the certainty
is lowest at the data point (`sill * exp(-3 width^2 / range^2)`) and rises to
`sill` at the plateau edge, where it drops to zero. With a 150 m plateau and
a 250 m range, a cell on top of a data point gets 34 % of the sill and a
cell 150 m away gets 100 %.

**`RBM_oct2023` jumps at the plateau edge.**

```matlab
ins = @(dist,range,width,sill,damp) sill.*(exp(-(damp*power((dist),2))./(power(range,2))));
out = @(dist,range,width,sill)      sill.*(exp(-(3*power((dist),2))./(power(range,2))));
cert_fun = @(dist,range,width,sill) (dist<=width).*ins(dist,range,width,sill,1.5) + ...
           (dist>width).*out(dist,range,width,sill+(sill-ins(width,range,width,sill,1.5)));
```

The sill adjustment `sill + (sill - ins(width))` adds the missing certainty
as if the tail started at full height at the edge, but the tail is centred on
the data point and is already damped by `exp(-3 width^2 / range^2)` there.
With `a = exp(-1.5 width^2 / range^2)` the value drops from `a * sill` to
`(2 - a) a^2 * sill` across the edge, a jump of `a (1 - a)^2 * sill`. It is
largest, 4/27 of the sill, at `width / range` = 0.86 (for example 0.14 of
the sill at a 75 m plateau and a 100 m range).

**What GeoSigma does.** The Danish specs (PACEP, MEP, SkyTEM, fewTEM,
manyTEM, tTEM and the gamma and resistivity logs) use the new
`kernel_with_point_tail`. It treats a sounding as 1D information at its x, y
location, like a borehole: inside the sensitivity kernel it equals
`RBM_oct2023`, and outside it a Gaussian decay centred on the data point
starts from the kernel's value at the edge:

```
dist <= width:  sill * exp(-1.5 dist^2 / range^2)
dist >  width:  sill * exp(-1.5 width^2 / range^2) * exp(-3 (dist^2 - width^2) / range^2)
```

The value is continuous at the edge; the slope changes there. PACES and
REFSEIS keep `FRAFA_apr2023`.

The national MATLAB run (2023) used `ILM_sep2023` for the eight themes above
and for PACES on Funen; the Funen-MST run used `RBM_oct2023` or
`ILM_oct2023`. Both legacy functions are kept so that the MATLAB grids can be
reproduced, and they are validated bit-exactly that way. They raise an error
unless the spec sets `allow_legacy_cert_fun=True` (YAML:
`allow_legacy_cert_fun: true`).

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
  network)
- Madsen, R. B., Falk, F. A., Møller, I., & Høyer, A.-S. (2024).
  Introducing INPOX: a method for informed point extraction from geological
  2D surfaces exemplified on the Danish national hydrostratigraphic model.
  *GEUS Bulletin*, 57. <https://doi.org/10.34194/geusb.v57.8364>
