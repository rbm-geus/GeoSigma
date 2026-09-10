# PEAT_AUDIT.md — peat/NPL content of the legacy MATLAB reference

One-time, read-only audit of every `.m` file in
`matlab_reference/N-ret-mod---Source-main/` to mark which files are peat/NPL
specific, so future translation sessions can skip or carve them with confidence
instead of re-reading them. Peat/NPL handling is **out of scope for GeoSigma**
(see `CLAUDE.md`); this file only locates it, it does not endorse porting it.

## Categories

- **(a) entirely peat-specific** — the file exists only to read, correct, or
  build the peat layers. Skip wholesale.
- **(b) mixed** — contains both peat-specific logic and logic relevant to the
  rest of the translation. Cannot be skipped wholesale; the peat parts are
  named per row.
- **(c) not peat** — no peat-specific logic. (Files with no `NPL`/`peat` token
  at all are listed compactly at the end.)

## Key distinction used throughout

Most `NPL` occurrences are **not** peat *interpretation* — they are layer-offset
bookkeeping: `Nlay-NPL` = "number of modelled (non-peat) layers" and
`Npreq-NPL` = the range-override layer index. That arithmetic is the
*translation-relevant* part and is already handled in GeoSigma's modelled-layer
space (where the data is peat-free, so NPL is effectively 0). The genuinely
peat-specific code is almost always a *separable add-on*: in the theme files it
is only the trailing `if include_peatlands == 1` block that prepends `NPL`
constant placeholder layers (those placeholders are later overwritten by
`get_PL_themes`/`PL_themes` in `Main`). That block is what to drop.

## Audit table

| filename | category | notes |
|---|---|---|
| get_PL_themes.m | a | Builds the peat-layer themes from the Jupiter peat databases (`Tørv.csv`, post-/sen-glacial layers). Entirely peat. |
| get_PL_themes_ILM.m | a | ILM-kernel variant of `get_PL_themes.m`; same peat-theme construction. Entirely peat. |
| ReadWriteTiles.m | a | Reads/stitches the interpreted peat-land tile grids (`..._Tolkede_tiles\`, `Lag1..5_Bund_..._Tørv_...`), removes artifacts, returns the snapped peat list (`SnapList`). The peat-tile subsystem. Entirely peat. |
| get_PACES_theme.m | b | Theme math is core. Peat-specific: only the `if include_peatlands == 1` prepend block, **lines 184–188**. (`Nlay-NPL`, `Npreq-NPL` are layer-offset bookkeeping, not peat.) |
| get_PACEP_theme.m | b | As above. Peat block **lines 188–192**. |
| get_MEP_theme.m | b | As above. Peat block **lines 205–209**. |
| get_SkyTEM_theme.m | b | As above. Peat block **lines 246–250**. |
| get_TEM_theme.m | b | As above. Peat block **lines 236–240**. (Legacy combined-TEM; superseded by few/many/tTEM.) |
| get_fewTEM_theme.m | b | As above. Peat block **lines 230–234**. |
| get_manyTEM_theme.m | b | As above. Peat block **lines 231–235**. |
| get_tTEM_theme.m | b | As above. Peat block **lines 226–230**. |
| get_GAMMALOG_theme.m | b | As above. Peat block **lines 172–176**. |
| get_RESLOG_theme.m | b | As above. Peat block **lines 173–177**. |
| get_REFSEIS_theme.m | b | As above. Peat block **lines 162–166**. |
| Main.m | b | Orchestrator. Peat-specific: the "Read Peat-Land Layers and Write To Files" section (**~136–190**, incl. `ReadWriteTiles` call), the `get_PL_themes_ILM` calls + `AllThemes(:,:,1:NPL,1) = PL_themes` assignments (**~581–665**), and the peat variance bump `final_variance_themes(:,:,1:NPL) += 0.0625` (**723–724**). All themes are called with `include_peatlands=1`, but those prepended slots are placeholders overwritten by `PL_themes`. Everything else (region setup, kriging driver, assembly) is in-scope. |
| landsdel_switch.m | b | Region registry (replaced by `config/region.py`). Peat-specific: the per-region `NPL = 5/3/0/0` assignments (**lines 9, 33, 47, 56, 70**). The rest (region name, Nlay, Npreq, well filenames) is in-scope. |
| get_layer_names.m | b | Builds layer-name lists (replaced by `RegionConfig`). Peat-specific: the `PLnames` table (**lines 3–9**) and the two `for i = 1:NPL` prepend loops (**lines 14–25**) that put peat names in the first NPL slots. The geological-layer naming after is in-scope. |
| get_layermasks.m | b | Builds onshore / layer-exists / combined masks (in-scope). Peat-specific: the "Combining peatland masks" branch using `get_layer_names`/`for jj = 2:(NPL+1)` (**~79–150**) and the `if itnum > NPL` gate (**~705**). NOTE: the header "DO LAYER CORRECTIONS TO ACCOMODATE PEAT LANDS" (**line 358**) is mislabeled — the code beneath it is generic negative-thickness correction, not peat. |
| get_modeltheme.m | b | Depth-dependent variance floor (in-scope). Peat-specific: the `if i <= NPL` branches (**lines 54–56 and ~109**) that assign the constant `f5map` to peat layers instead of the depth ramp. |
| get_wells.m | b | Loads well points (in-scope). Peat-specific: prepends `NPL` empty cells for peat well slots — `well_out = [cell(NPL,1); wells_m]` (**line 81**); comment at line 9 and dead line 76. |
| prepare_for_kriging.m | b | Per-layer kriging-prep struct builder (in-scope). Peat-specific: only threads the `peat_log`, `NPL`, `Snap_PL` arguments through to `get_layermasks` (**line 35**) and `get_interpolated_minigrids_parallel` (**line 70**); no peat math of its own. |
| get_interpolated_minigrids_parallel.m | b | Range/sill minigrid predictor (in-scope). Peat-specific: the `if layn(1) < (NPL+1)` branches (**lines 41 and 122**) that switch to the snapped-peat (`Snap_PL`) path for peat layers. |
| plot_profile_view.m | b | Plotting/diagnostic (out of translation scope regardless). Peat-specific: the `for z = 1:NPL` peat-layer loop (**line 319**) vs `for z = (NPL+1):Nlay` (**line 331**). |
| convert2ascii_resample.m | c | Generic grid-resampler. Only matched the audit because its comments mention `ReadWriteTiles`; the function itself is not peat (comment: "også brugt til at resample Jylland i Main.m"). Called by the peat tile pipeline but reusable. |
| run_plot_profile_view.m | c | One-line wrapper that calls `plot_profile_view` and forwards an `NPL` variable; no peat logic. |
| remove_artifacts.m | c | Generic surface artifact-removal from a bottom grid using points; no peat token. Invoked during peat-tile cleanup but not peat-specific. |

## Category (c) — no peat-specific logic (remaining files)

All of the following contain **no** peat/NPL-specific code and can be read for
translation without any peat carve-out:

`add_bottom.m`, `calculate_buffer_zones.m`, `certainty_function_definer.m`,
`check_fyn_surface.m`, `dk8_prep.m`, `drawpoints.m`, `dtm_read_plot.m`,
`exportmodels.m`, `find_100_m_buffer.m`, `FYN_MST_prep.m`, `get_clusters.m`,
`get_corr_maps.m`, `get_DK8_points.m`, `get_LOOP2points.m`,
`get_minimum_map.m`, `get_points_from_database.m`, `get_reals_cholesky.m`,
`get_vpoints.m`, `import_complexitymap.m`, `import_vendsyssel.m`,
`insert_dk9_mask.m`, `local_kriger.m`, `local_kriging_setup_img.m`,
`merge_layers.m`, `normalize_UTM_coord.m`, `plot_and_compute_statmods.m`,
`plot_dk.m`, `plot_final_uncertainties.m`, `plot_validate_uncertainty_theme.m`,
`post_process_clusters.m`, `prepare_clusters.m`, `ReadSurfer7.m`,
`region_to_landsdel.m`, `Run_Batches_Final.m`, `sendolmail.m`,
`setup_GLOBAL_structure.m`, `Sjaelland_mask_issue.m`, `surface_tester.m`,
`test_plot_final_reals.m`, `Test_skyTEM_uncertainties.m`, `tie_realizations.m`,
`write_grid_ascii.m`.

## Summary

- **(a) entirely peat — skip wholesale (3):** `get_PL_themes.m`,
  `get_PL_themes_ILM.m`, `ReadWriteTiles.m`.
- **(b) mixed — carve out the named peat sections (20):** the 11 theme files
  (peat = only the `if include_peatlands` prepend block), plus `Main.m`,
  `landsdel_switch.m`, `get_layer_names.m`, `get_layermasks.m`,
  `get_modeltheme.m`, `get_wells.m`, `prepare_for_kriging.m`,
  `get_interpolated_minigrids_parallel.m`, `plot_profile_view.m`.
- **(c) not peat (50):** everything else (table's three `c` rows + the list above).
