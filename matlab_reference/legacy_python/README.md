# Legacy Python theme translations

These files are the **original one-to-one Python translations** of the MATLAB
theme scripts in `../N-ret-mod---Source-main/`. They are kept here as
**translation references only** — a readable record of how each MATLAB routine
was first carried over, useful when auditing the packaged implementation
against its source.

**They are superseded by `geosigma/themes/`.** The packaged engine is the
validated implementation; these scripts are not maintained, not imported by the
library, and not covered by the test suite.

## Do not use these to produce results

They contain a known **pre-Quaternary indexing off-by-one**. Each script slices
the pre-Quaternary range regime as:

```python
rangemap[:, :, Npreq - NPL:] = rangemap2[:, :, Npreq - NPL:]
```

The pre-Quaternary boundary is the 0-based modelled-layer index
`Npreq - NPL - 1`, so the `- 1` shift is missing and the regime boundary lands
one layer too deep. The packaged engine is unaffected: it takes an
already-0-based `n_prereq` and splits the range regimes at that index directly
(`geosigma/themes/themes.py`, `_two_regime`).

Any grids produced by these scripts are wrong at the Quaternary /
pre-Quaternary transition layer. Use `geosigma.themes` instead.

## Contents

| File | Superseded by |
|---|---|
| `get_PACES_theme.py` | `geosigma.themes.themes.paces_spec` |
| `get_PACEP_theme.py` | `geosigma.themes.themes.pacep_spec` |
| `get_MEP_theme.py` | `geosigma.themes.themes.mep_spec` |
| `get_SkyTEM_theme.py` | `geosigma.themes.themes.skytem_spec` |
| `get_GAMMALOG_theme.py` | `geosigma.themes.themes.gammalog_spec` |
| `get_RESLOG_theme.py` | `geosigma.themes.themes.reslog_spec` |
| `get_REFSEIS_theme.py` | `geosigma.themes.themes.refseis_spec` |
| `get_fewTEM_theme.py` | `geosigma.themes.themes.fewtem_spec` |
| `get_manyTEM_theme.py` | `geosigma.themes.themes.manytem_spec` |
| `get_tTEM_theme.py` | `geosigma.themes.themes.ttem_spec` |
| `main.py` | — scratch driver that called the above |
| `landsdel_switch.py` | `geosigma.config.region.RegionConfig` |

`main.py` was the ad-hoc driver script for these translations. It is archived
alongside them because it imported them directly. It is not runnable from this
directory (it also imports `import_complexitymap`, which remains at the repo
root) and it carries a hardcoded Danish test area and a peat-layer count
(`NPL`), which is out of scope for GeoSigma — see the peat exclusion note in
`CLAUDE.md`.

`landsdel_switch.py` is the translation of the MATLAB region switch
(`landsdel_switch.m`): a hardcoded `if`/`elif` chain over the five Danish
regions returning layer count, layer names and well filenames. It is
**functionally superseded by `geosigma/config/region.py`** (`RegionConfig`),
which carries the same information as a model-agnostic, serialisable
configuration object rather than region names baked into library code. It is
kept here only as a record of the values the Danish setup used.

These scripts also retain other traits of the original translation that the
packaged engine has since dropped: hardcoded relative data paths, per-cell
Python loops over the grid, and the `include_peatlands` / `NPL` parameters.

## Not runnable as committed

Nine of the ten theme scripts (all but `get_PACES_theme.py`) begin with

```python
from matio import load_from_mat
```

No `matio` module exists anywhere in this repository, and none is declared in
`environment.yml` or `requirements.txt`. Those scripts therefore fail at import
time as committed. This is deliberate: they are **references, not usable
code**. Read them for the translation record; do not expect to execute them.
