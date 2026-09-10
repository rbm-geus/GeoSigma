# The ThemeData three-file format

This document specifies the **ThemeData three-file format**: a portable,
self-describing bundle that carries the point inputs of a single
information-source *theme* into GeoSigma's variance-map engine. It is the
input-side deposit standard — the form in which the point data behind a
published variance map should be archived so that the map can be reproduced
without the original raw files or the code that read them.

The format is deliberately model-agnostic. It records *what a theme's data are*
(where each contributing observation sits, and the per-observation attributes
that govern the uncertainty it implies) without embedding any region, grid, or
statistical-model specifics. Those specifics are supplied separately at load
time. Throughout, the DK-model of Denmark is used only for illustration; the
Danish configurations shipped with GeoSigma are one instantiation of the format,
not part of the standard.

The normative reference implementation is the loader
`geosigma.themes.data_io.load_theme_data`. Where this document states that some
input "raises", it describes that loader's behaviour; the accompanying phrases
are the guidance the loader emits.

This document defines the **files**. The companion document
[`bring_your_own_data.md`](bring_your_own_data.md) covers the **scientific
choices** — certainty kernels, range models, and the design of a theme's
statistical spec. The two are complementary: a valid bundle in this format is
not by itself a modelling decision, and vice versa.

---

## 1. Overview: the three files

A theme is deposited as three files that share a directory:

| File | Role | Required |
|------|------|----------|
| **Manifest** (YAML) | Declares the theme: its label, the spec it uses, the points file, the modelled-layer count, the coordinate and attribute column mapping, the complexity declaration, and any derived parameters. | yes |
| **Points CSV** | One row per contributing observation. Holds the coordinate columns, the per-layer attribute blocks, and the per-point scalar columns the manifest declares. | yes |
| **Complexity grid** (raster) | A geological-complexity class grid governing the correlation-range regime. Referenced by the manifest, not embedded in the CSV. | optional (but the manifest must *declare* its presence or absence — see §4) |

The manifest is the entry point. Paths inside it (the points CSV, and a
complexity grid referenced by path) are resolved relative to the manifest's own
directory unless given as absolute paths, so a bundle is relocatable as a unit.

The manifest and CSV are plain text and are meant to be inspected in a text
editor, a spreadsheet, or a GIS. Nothing in the format is binary except,
optionally, the complexity raster (which may itself be an ASCII grid).

---

## 2. The points CSV

The points CSV is a standard comma-separated file with a single header row,
read with default CSV conventions (comma delimiter, standard empty/NaN tokens).
Every column the manifest references must be numeric: coordinate and attribute
values are coerced to floating point, and an empty cell becomes `NaN`.

Each **row is one contributing observation** (a borehole, a geophysical sounding,
an interpreted point, and so on). The point's identity is its row; row order is
preserved as the point index. Row order is normally immaterial, but note that a
spec may use a reducer that depends on it (for example, a "first point in window"
rule that selects by lowest index). Where a theme's spec is order-sensitive, the
deposited row order is part of the data and should be stable and meaningful.

A CSV carries three kinds of column.

### 2.1 Coordinate columns

Two columns give each point's map coordinates: easting (x) and northing (y). The
manifest names them with `x_column` and `y_column`; the names themselves are
free (they need not be `x`/`y`). If either named column is absent from the CSV,
the load raises, reporting the missing axis and the CSV's first columns.

The coordinate reference system is not stored in the CSV. It may be recorded in
the manifest's optional `crs` field for provenance (§3).

### 2.2 Per-layer attribute blocks

Many attributes vary per modelled layer boundary — an interpreted boundary
depth, a layer thickness — so they are stored as a **block of columns, one per
layer**, sharing a common prefix and distinguished by a layer label:

```
depth__L01, depth__L02, depth__L03, …, depth__L43
```

The manifest declares the block by its prefix (`depth__L` above) and the theme's
modelled-layer count `n_layers`. The loader then applies this rule:

> Collect every CSV column whose name **begins with the prefix**, sort those
> names **lexicographically (as text)**, require the count to equal `n_layers`,
> and assign them to modelled layers 1…`n_layers` **in that sorted order**.

Two consequences are normative:

1. **Labels must be zero-padded to a constant width** so that lexicographic
   order coincides with layer order: `L01, L02, …, L43` (and `L001, L002, …`
   for models with 100 or more layers). Unpadded labels sort incorrectly
   (`L1, L10, L11, L2, …`) and would map columns to the wrong layers. The label
   number itself is not parsed — only the sort position is used — so the padding
   is what carries the ordering. **The loader enforces this**: it checks that the
   label suffix (the text after the prefix) is the same character width for every
   column in a block and **raises**, naming the two offending columns, if it is
   not — so a mis-padded block fails loudly instead of silently mis-assigning
   layers.
2. **The prefix must be unique** to its block. Matching is by string prefix, so
   a prefix that is also the start of an unrelated column name would capture that
   column. The `<name>__L` double-underscore convention is recommended precisely
   to avoid such collisions.

Layer labels are **1-based** (`L01` is the first modelled layer). A block must
contain exactly `n_layers` columns; a block whose matched-column count differs
from `n_layers` raises, and a prefix that matches no column raises. Each block is
matched independently, so a theme may carry several per-layer blocks (for
example `depth__L…` and `thick__L…`) side by side.

A per-layer value may be `NaN` where the source did not resolve that boundary at
that point; the loader preserves it verbatim. Whether and how a `NaN` is later
filled is the spec's concern, not the format's (§6).

### 2.3 Per-point scalar columns

An attribute that is a single value per observation — a depth-of-investigation,
a quality flag, an acquisition type — is stored as **one column**, named
directly by the manifest. It is read as a one-dimensional array over the points
and, where a spec needs it per layer, is broadcast across layers by the engine.
`NaN` is preserved. A per-point column named by the manifest but absent from the
CSV raises.

---

## 3. The manifest

The manifest is a YAML mapping. The loader reads the following keys; any key it
does not recognise (for example `_notes`, §6) is ignored, which is what makes the
manifest safely extensible for provenance.

| Key | Type | Required | Meaning |
|-----|------|:--------:|---------|
| `format_version` | integer | no (default `1`) | The revision of this specification the bundle follows. The loader accepts it and raises on an unsupported value; an absent key is treated as version `1`. Recommended for archival deposits so a bundle self-describes its spec revision. |
| `name` | string | yes | Human-readable label for the theme. Carried through as provenance; not otherwise interpreted. |
| `spec` | string | yes | Identifier of the statistical spec the theme uses. The runtime resolves it to a `ThemeSpec`; the format does not define the spec itself (§5). |
| `points_csv` | string (path) | yes | The points CSV, relative to the manifest directory (or absolute). A missing file raises. |
| `n_layers` | integer ≥ 1 | yes | Number of modelled layer boundaries. Every per-layer block must have exactly this many columns. A value below 1 raises. |
| `x_column` | string | yes | Name of the easting column in the CSV. |
| `y_column` | string | yes | Name of the northing column in the CSV. |
| `attributes` | mapping | yes | Declares the attribute blocks/columns to reassemble (§3.1). Must be a non-empty mapping. |
| `complexity` | string \| mapping | yes (as a declaration) | The complexity contract (§4). Must be declared explicitly; a missing or null value raises. |
| `spec_params` | mapping | no | Data-derived scalar parameters the spec needs that cannot be reconstructed from the deposited points (§3.2). Defaults to empty. |
| `crs` | string | no | Coordinate reference system of the point coordinates, e.g. `EPSG:25832`. Carried through as provenance; not used in computation. |

If the top-level YAML is not a mapping, the load raises. If any required key is
absent, the load raises listing the missing keys.

### 3.1 The `attributes` mapping

`attributes` maps each attribute name to a small block that tells the loader how
to find it in the CSV. There are exactly two block forms:

```yaml
attributes:
  depth:                 # a per-layer block
    per_layer: true
    prefix: depth__L
  doi:                   # a per-point scalar
    per_layer: false
    column: doi
```

- A **per-layer block** sets `per_layer: true` and gives a `prefix` (§2.2). A
  block missing its `prefix` raises.
- A **per-point scalar** sets `per_layer: false` (or omits `per_layer`) and gives
  a `column` (§2.3). A block missing its `column` raises.

A block that is not a mapping, or that provides neither form, raises with a
message naming the offending attribute.

### 3.2 `spec_params`: derived parameters that survive filtering

Some parameters a spec needs are neither generic per-point/per-layer arrays nor
model constants — they are **scalars derived from the source data during
preparation, in a way that cannot be recovered from the deposited (filtered,
de-duplicated) points**. Because they cannot be recomputed downstream, they must
be recorded in the manifest. `spec_params` is that record: a mapping of named
scalars passed through to the spec when it is resolved.

The canonical case is a list of the **layers a source actually constrains**. If
that list is computed from the *unfiltered* source (before date filtering and
coordinate de-duplication remove rows), it cannot be reconstructed from the
filtered CSV alone, so it is stored:

```yaml
spec_params:
  active_layers: [12, 13, 14, 15, 16, 17, 18, 19, 20]
```

Another case is a single fill value — for example a mean of an attribute over the
filtered set, used to fill missing values — reduced to one number at preparation
time and stored as `spec_params: { doi_fill: 17.4 }`.

`spec_params` is optional and defaults to empty; most themes need none. Values
the resolving spec does not accept raise at resolution time, reporting the
rejected parameters.

---

## 4. The complexity contract

A theme's correlation range is set by a **geological-complexity class grid**: a
raster of integer class codes co-registered with the working grid, where each
class selects a range from the spec's range model. Because supplying or
withholding this grid materially changes the variance map, its status is treated
as a modelling decision that must be **stated explicitly**. A manifest whose
`complexity` field is missing or null raises, with guidance to choose one of the
three declarations below; there is no silent default to "no complexity".

The three valid declarations are:

1. **A grid path** — `complexity: complexity.tif` (or the mapping form
   `complexity: { path: complexity.tif }`). The grid is loaded from disk,
   relative to the manifest directory unless absolute. This is the natural choice
   for a static deposit.

2. **`external`** — the grid is not on disk but is supplied to the loader at load
   time by the calling program (for instance a class grid produced or cropped
   in-process). This is a runtime convenience rather than a deposit form. If the
   manifest says `external` but no grid is supplied, the load raises.

3. **`none`** (equivalently `uniform`) — the deliberate opt-in to *no* complexity
   grid. The theme's complexity is left undefined and the engine assumes a single
   uniform class, i.e. a flat correlation range with no complexity dependence.
   This is the only way to obtain uniform behaviour, and it is recorded rather
   than implicit.

Supplying a grid two ways at once is rejected: declaring a path (or `none`) while
also passing a grid to the loader raises, as does declaring `external` and
supplying nothing.

### 4.1 Grid formats and alignment

A referenced complexity grid may be any of:

| Extension | Format |
|-----------|--------|
| `.tif`, `.tiff` | GeoTIFF |
| `.asc`, `.txt` | ESRI ASCII grid |
| `.grd` | Surfer 7 binary grid |

Any other extension raises.

The grid must be **co-registered with the working grid** the theme is evaluated
on. When the caller supplies the working grid's metadata, a path-loaded grid is
required to match it in shape, cell spacing, and origin; a grid that differs
raises as "not aligned". (The coordinate reference system and no-data sentinel are
not part of this check.) If no working-grid metadata is supplied, the grid is
loaded without an alignment check and alignment becomes the depositor's
responsibility.

### 4.2 Class semantics

Complexity values are non-negative integer class codes. The code `0` denotes
**unknown / undefined** — cells with class 0 carry no range and are masked out of
the theme there. Positive codes `1…K` select the correlation range through the
spec's range model. The number of classes and their meaning are a property of the
spec and the source, not of the format; the Danish configurations use four
classes (1–4), but any positive coding is admissible.

---

## 5. What the format deliberately excludes

The format carries a theme's *data*, and nothing else. Three categories are
intentionally kept out, each because it belongs to a different, separately
declared layer of the system.

- **Grid geometry and region configuration.** The working grid's extent, cell
  size, and origin, and any model-level constants (such as the index at which a
  two-regime range model switches), are *not* in the bundle. They are supplied to
  the loader/engine externally, from the region configuration and grid metadata.
  The manifest records `n_layers` — a count the data must be consistent with —
  but not the grid the data will be rasterised onto.

- **Spec definitions.** The manifest names a spec; it does not contain the spec's
  formulae (the variance model, certainty kernel, mask, and range tables). A spec
  is a `ThemeSpec` resolved by the runtime — either a built-in referenced by name,
  or a declarative spec authored separately. Keeping the spec out of the data
  bundle is what lets the same deposited points be re-evaluated under a revised
  or alternative statistical model.

- **Raw-source conversion (adapter) logic.** Turning a proprietary or
  region-specific raw source (native binary tables, source-specific column names,
  the filtering and de-duplication that select which observations to keep) into
  these three files is *adapter* work, performed once, upstream of the format.
  The loader never sees the raw source; an adapter is not part of, and not
  required by, a finished bundle.

The dividing line is: if a quantity can be recovered from the deposited files
plus an externally declared model and grid, it is not stored here; if it is a
datum of the theme itself, it is.

---

## 6. Provenance conventions: the `_notes` field

The loader ignores any manifest key it does not recognise. By convention,
provenance is recorded under a top-level **`_notes`** mapping — free-form, never
interpreted, and therefore safe to structure however a deposit requires. For an
archival deposit the following are recommended.

- **Source.** The raw dataset(s) the points were derived from, with enough
  identity (name, version, owner, DOI or accession) to trace them.

- **Filtering applied.** Every selection and reduction between the raw source and
  the deposited CSV, stated so the row set is reproducible — in particular any
  temporal or quality filter, and the **de-duplication rule** (which duplicates
  were dropped and which kept, e.g. "de-duplicated on (x, y) keeping the last
  occurrence"). Record the before/after point counts.

- **Whether attribute columns are raw or pre-resolved.** State, per attribute,
  whether a column holds the value **as observed** or a value **already
  processed**. This distinction is load-bearing: some specs expect a raw column
  and do their own filling/masking downstream (for example, exporting a
  depth-of-investigation with `NaN` where it was not measured and letting the spec
  fill it per cell), while other themes deposit an already-resolved column. A
  reader who mistakes one for the other will misinterpret the data even though the
  file loads without error.

- **Origin and conditions of referenced grids.** For a complexity grid (or other
  raster) that comes from elsewhere, record its origin and character, and any
  condition the originator attaches to its reuse or deposit — including, where the
  grid may be deposited openly, a condition that its origin and limitations be
  stated alongside it (see the worked example below).

### Worked example: a third-party complexity grid

When the complexity grid comes from another project — and is a provisional proxy
rather than a finished, validated dataset — the deposit note should state its
origin and limitations plainly, so the grid can be archived openly without being
mistaken for an authoritative complexity dataset. The originator of the grid used
in the Danish deposit has approved its open deposit on this condition. The
following pattern is recommended for near-verbatim reuse:

```yaml
_notes:
  complexity_grid: >
    complexity.tif — geological-complexity class raster (classes 1–4;
    0 = unknown/masked), co-registered with the model grid, used only to set the
    correlation-range regime.
    ORIGIN: produced by <originating project / owner> for <that project's
    purpose> and reused here with the originator's agreement. It is included in
    this deposit openly, on the understanding that this entry states its origin
    and limitations (below).
    CHARACTER: a crude, provisional proxy for Quaternary geological complexity.
    The classes are an approximate indicator derived for a different objective,
    not a directly measured or independently validated quantity.
    LIMITATIONS: coverage is incomplete — cells outside the source footprint are
    set to class 0 (unknown) and masked; class boundaries are indicative rather
    than surveyed; the grid is static and is not maintained or updated with the
    model. It should be treated as an approximate input, not an authoritative
    complexity dataset.
```

Because the format forbids a silent default (§4), a reader who chooses to omit the
grid has an explicit, recorded fallback — re-running the theme with `complexity:
none` for the uniform-range variant — rather than an accidental one, and the
change in result is a documented choice.

---

## 7. A complete minimal example

A three-layer synthetic theme with one per-layer block (`depth`), one per-point
scalar (`doi`), and no complexity grid.

**`reflector.manifest.yaml`**

```yaml
format_version: 1
name: Example reflector theme
spec: my_reflector_spec
points_csv: reflector_points.csv
n_layers: 3
crs: EPSG:25832
x_column: x
y_column: y
attributes:
  depth:
    per_layer: true
    prefix: depth__L
  doi:
    per_layer: false
    column: doi
complexity: none
_notes:
  source: Synthetic example for the format specification; not real data.
  filtering: none.
  attribute_columns:
    depth: raw interpreted boundary depth (m); NaN where a boundary is unresolved.
    doi: raw depth-of-investigation (m); NaN preserved (the spec fills it).
```

**`reflector_points.csv`** (first rows)

```csv
x,y,depth__L01,depth__L02,depth__L03,doi
150.0,110.0,4.0,9.0,14.0,50.0
250.0,60.0,5.0,11.0,20.0,
90.0,210.0,3.0,7.0,12.0,33.0
```

This bundle reassembles to a point cloud of three observations, a `depth` array
of shape (3 points × 3 layers), a per-point `doi` array of length 3 (with the
second point's `doi` preserved as `NaN`), and no complexity grid. To evaluate it,
a caller resolves `my_reflector_spec` to a `ThemeSpec` and supplies the working
grid — neither of which is part of the bundle (§5).
```
