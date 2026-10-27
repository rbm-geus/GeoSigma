# Preview demos (v1.0.0) — incomplete

The scripts in this folder belong to the **simulation half** of GeoSigma, which
is planned for **v1.0.0**: conditioning layer boundaries by local kriging and
drawing geostatistical realizations. That part of the MATLAB workflow
(preprocessing, the local kriging driver, and layer assembly) is **not
translated yet**, so these demos are previews, not working workflows.

**v0.1.0 ships the variance-map tool only.** For a demo that runs end to end,
use [`../demo_variance_map.py`](../demo_variance_map.py).

| Script | What it shows | Status |
|---|---|---|
| `demo_geosigma_pipeline.py` | Surface → INPOX points → covariance model → conditioning → realizations | **Stops after step 3.** Conditioning and realizations need the kriging driver. |
| `demo_synthetic_kriging_inversion.py` | The translated kriging, least-squares and Cholesky primitives on a small synthetic grid | Runs, but uses the primitives directly. The real-model driver is not translated yet. |
| `demo_inpox_basic_jutland_hydrostrat.py` | INPOX selection of conditioning points on the sample model | Runs. The extracted points are used only by the (future) simulation. |

The primitives these demos call (`precal_cov`, `least_squares_inversion`,
`get_reals_cholesky`, `draw_points_inpox`, …) are translated and unit-tested.
What is missing is the orchestration that applies them to a whole
hydrostratigraphic model. Expect these scripts and their APIs to change before
v1.0.0.

Like the other demos, each one ends in a blocking `plt.show()` window. To run
one headless, set `MPLBACKEND=Agg`.
