# sherepar



## Generate a template mesh

The demo needs an OBJ mesh.  Generate a simple ellipsoid:

```bash
cd /path/to/sherepar
python examples/generate_mesh_elipsoid.py   # writes data/ellipsoid.obj  (note: existing script name)
```

Or provide your own genus-0 closed surface OBJ.

---

## Run the first attempt for deformations 
#TODO: I need to include validation fro quasi-conformal mapping and more signals

```bash
cd /path/to/sherepar
python examples/demo_stage1.py              \
    --root   /tmp/stage1_output            \
    --template data/ellipsoid.obj          \
    --n      5
```

Output layout:

```
/tmp/stage1_output/
  surfaces/   ← deformed OBJ meshes
  signals/    ← per-vertex signal arrays (.npy)
  labels/     ← metadata JSON files
```

---

## Python-only usage (without the C++ build)

The `spherepar.benchmark.signals` module (signal generators) and the
`Surface` / `SurfaceFactory` classes are pure Python.  You can import
and unit-test them without the `graphop` extension:

```python
import numpy as np
from spherepar.benchmark.signals import isotropic_gaussian, anisotropic_gaussian

V = np.random.randn(500, 3)           # dummy vertices
center = V[0]
f = isotropic_gaussian(V, center, sigma=0.2, amplitude=1.0)
print(f.shape)  # (500,)
```

`SurfaceFactory` raises `ImportError` at construction time if `graphop` is
not found; `Surface` and the signal functions work independently.

---

## Notes on the project structure

| Path                               | Purpose                                      |
|------------------------------------|----------------------------------------------|
| `graphop/deformation.h`            | C++ ARAP/SRE-ARAP backend header             |
| `graphop/deformation.cpp`          | C++ implementation (CGAL)                    |
| `graphop/bindings.cpp`             | pybind11 module definition                   |
| `graphop/main.cpp`                 | Standalone demo executable (unchanged)       |
| `CMakeLists.txt`                   | Top-level build (builds both targets)        |
| `spherepar/benchmark/__init__.py`  | Package entry-point                          |
| `spherepar/benchmark/surface.py`   | `Surface` and `SurfaceFactory` classes       |
| `spherepar/benchmark/signals.py`   | Isotropic + anisotropic Gaussian signals     |
| `examples/demo_stage1.py`          | End-to-end demo script                       |


# some installation notes

```bash
# install CGAL and pybind11 (system package manager or conda)
sudo apt install libcgal-dev libeigen3-dev
pip install pybind11 trimesh scipy numpy=1.24
```

I am using `python=3.10` and `numpy=1.24` to avoid some compatibility issues with CGAL and pybind11.  Adjust as needed for your environment.
1. install also `cmake` and `dev` tools for building the C++ extension.
```bash
(sherepar) sauron@mordor:sherepar$ cpp --version
cpp (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
(sherepar) sauron@mordor:sherepar$ cmake --version
cmake version 3.22.1

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```
2. need also 'rtree'` for spatial queries in the deformation code.  Install via pip or conda:
```bash
# need also rtree
pip install rtree
```

# Data generation

## Noise deformation cases

`examples/script_to_generate_dataset.py` also supports spatially correlated
vertex-noise cases. All noise-case outputs are validated as watertight and
retain the face connectivity from the repaired/deformed input mesh.

- `case4_noise`: noise only; no graphop deformation is applied.
- `case5_small_noise`: the `case2_small` graphop deformation followed by noise.
- `case6_large_noise`: the `case3_large` graphop deformation followed by noise.

The noise is independent Gaussian displacement in each XYZ coordinate. Its
displacement field, rather than the mesh coordinates themselves, is smoothed
with an edge-length-weighted graph heat kernel:

```bash
python examples/script_to_generate_dataset.py \
    data/meshes \
    --output-root data/generated_noise \
    --deformation-cases case4_noise,case5_small_noise,case6_large_noise \
    --noise-sigma 0.01 \
    --noise-smooth-sigma 1.0
```

- `--noise-sigma`: standard deviation in mesh coordinate units (default `0.01`).
- `--noise-smooth-sigma`: graph heat-kernel scale in edge-length units
  (default `1.0`).

## Fixed regression-signal centre

Use `--signal-center "X,Y,Z"` to select the nearest vertex to an XYZ point on
each repaired input template. The selected vertex index is reused across all
deformed samples from that template:

```bash
python examples/script_to_generate_dataset.py \
    data/meshes \
    --output-root data/generated_fixed_center \
    --signal-center "0.2,-0.5,1.0"
```

The option fixes only the one-centre isotropic regression signal
(`iso_001_reg` / `iso_001_cls`) and its paired anisotropic signal (`aniso_001`).
Multi-centre isotropic classification signals remain random. If the selected
vertex cannot form a valid anisotropic fixed-gauge frame after deformation, the
generator logs and skips that sample rather than selecting another centre.

## Importing existing mesh files

Use `examples/script_to_generate_dataset_from_files.py` when the input meshes
already exist and should be copied into the dataset layout without deformation.
For every supported input mesh (`.obj`, `.ply`, `.stl`, or `.off`), it writes a
mesh OBJ, one per-vertex signal array, one label JSON, and a spherical
parametrization (FLASH by default, or CEM).

The importer and benchmark generator also support `--param-method spheremap`,
which delegates spherical mapping to the compiled `SphereMap` backend from the
`MoebiusRegistration` repository. The wrapper preserves vertex order and face
connectivity, stores the exact command and solver settings in the spherical
sidecar, and retains an output sphere for inspection when geometric validation
reports collapsed or folded triangles. Its defaults match
`MoebiusRegistration/scripts/example.py`; use the `--spheremap-*` options to
override them.

All parametrization methods require a single closed genus-0 surface. The
importer always skips open, disconnected, and non-genus-0 meshes before any
dataset artifacts are written; the final summary reports how many were
filtered.

CEM and SphereMap support opt-in conformal post-centering with
`--mobius-center`. See [Möbius centering after CEM](MOBIUS_CENTERING.md) for
the transform convention,
metadata schema, resume behavior, and `pmconv` integration.

CEM uses the paper's experimental stereographic partition radius `1.2` by
default; override it with `--cem-radius VALUE`. The implementation reports
negative cotangent weights and validates the first candidate and final accepted
sphere for collapsed or folded triangles. These conditions are warnings, so
artifacts remain available for inspection. If an iteration increases the
Dirichlet energy, that candidate is discarded and CEM returns the preceding
accepted iterate. The radius and diagnostics are stored in the spherical JSON
sidecar, and the radius is also stored in the primary label for resume matching.

Phase 2 CEM controls are opt in: `--use-idt-remesh` flips connectivity without
moving vertices, `--adaptive-radius` searches the configured
`--cem-radius-candidates`, and `--reject-retry` applies
`--cem-max-collapsed-faces` after at most `--cem-max-attempts`. The sphere OBJ
always retains the original mesh faces. A reject/retry failure also retains its
best sphere and spherical JSON for diagnosis, but the primary label records
`parametrization.success=false`, a rejection `error`, and the retained paths.

Imports resume by default: only samples with complete mesh, signal, label, and
required sphere artifacts are skipped. Use `--no-resume` to regenerate every
selected input and overwrite its artifacts.

### Dataset-wide spherical error plots

To analyze an existing backend output root, use
`examples/plot_dataset_spherical_errors.py`. The root should contain matching
`meshes/` and `spheres/` directories; files are paired by stem:

```bash
python examples/plot_dataset_spherical_errors.py \
  --root data/faust_flash \
  --backend flash \
  --output-root data/faust_flash/error_analysis
```

The command writes `metrics.json`, `metrics.csv`, `summary_report.md`, and
colored sphere and original-mesh plots per sample (`<sample>_sphere.png` and
`<sample>_mesh.png`). Plot colors identify actual-zero, near-zero, folded, and
ordinary triangles. Run it separately with `--backend cem` or
`--backend cmcf` for the corresponding dataset roots.

To smooth a whole registration directory before parametrization, use the
batch wrapper around the quality-aware sliver smoother:

```bash
python examples/smooth_registration_dataset.py \
  --input-dir /path/to/registrations \
  --output-dir /path/to/registrations_smoothed \
  --method quality \
  --iterations 10 \
  --angle-threshold 10
```

Relative filenames and mesh connectivity are preserved. The default `quality`
method is the existing sliver-aware tangential smoother. Use
`--method laplacian` for synchronous uniform neighbor-average smoothing; this
intentionally permits normal displacement. The output directory also receives
`smoothing_metrics.json` and `smoothing_metrics.csv`.

Add `--plot` to write a side-by-side original/smoothed PNG next to each
output mesh.

For Laplacian mode, `--laplacian-iterations`/`--iterations` controls the
number of synchronous passes, `--laplacian-step`/`--step` controls the move
fraction, and `--laplacian-rings` controls the stencil size (`1` is the usual
one-ring Laplacian, `2` includes two-hop neighbors).

Generic meshes receive an all-zero `float32` signal with one value per vertex:

```bash
python examples/script_to_generate_dataset_from_files.py \
    --input-dir data/source_meshes \
    --output-root data/imported_meshes
```

For FAUST, point `--faust-dir` at the FAUST root containing `registrations/`:

```bash
python examples/script_to_generate_dataset_from_files.py \
    --faust-dir /path/to/FAUST \
    --output-root data/imported_faust
```

`--faust-dir` automatically selects FAUST mode; `--dataname FAUST` is not
required. In this mode the signal is the `float32` vertex index array
`[0, 1, ..., N-1]`, and the label metadata records `dataname: "FAUST"`. The
vertex ordering is therefore part of the signal definition. These imported
datasets contain one `main` signal per mesh, rather than the isotropic and
anisotropic Gaussian signal set produced by `script_to_generate_dataset.py`.

Cylinders uses a flat input directory and the same zero-valued signal:

```bash
python examples/script_to_generate_dataset_from_files.py \
    --cylinders-dir /path/to/cylinders \
    --output-root data/imported_cylinders
```

For ModelNet40, point to the root arranged as
`<class_name>/{train,test}/*.off`. At the default `--percentage 100`, its
provided train/test partitions are preserved in `folds/*/modelnet40_cls/` and
the validation files are empty. A smaller percentage samples each class
reproducibly and creates standard folds instead:

Preserve the supplied ModelNet40 split:

```bash
python examples/script_to_generate_dataset_from_files.py \
    --modelnet40-dir /path/to/ModelNet40 \
    --percentage 100 \
    --output-root data/imported_modelnet40
```

Create normal folds from a reproducible 25% per-class subset:

```bash
python examples/script_to_generate_dataset_from_files.py \
    --modelnet40-dir /path/to/ModelNet40 \
    --percentage 25 \
    --num-folds 5 \
    --split-seed 42 \
    --output-root data/imported_modelnet40_25pct
```

Count the generated ModelNet40 class distribution:

```bash
python examples/count_modelnet40_classes.py data/imported_modelnet40
```

### ADNI clinical classification

Use the ADNI mesh directory together with the local `participants.tsv` manifest.
The importer maps `CN`/`SMC` to class 0 (`CN`), `EMCI`/`LMCI`/`MCI` to
class 1 (`MCI`), and `AD` to class 2 (`AD`). The manifest is read locally;
its contents are not copied into the generated dataset. Folds are stratified
by participant, so no participant's sessions or hip sides can cross a split:

```bash
python examples/script_to_generate_dataset_from_files.py \
    --input-dir /path/to/ADNI/fixmodels_mni \
    --adni-participants /path/to/ADNI/participants.tsv \
    --adni-session first \
    --adni-hip both \
    --param-method cem \
    --anchor-strategy central_regular \
    --output-root data/imported_adni_cls
```

Use `--adni-session none` to retain every session, or `last` for the latest
numeric session per participant. `--adni-hip left` and `right` select one side.
The generated labels contain an `adni_cls` task with the integer label,
participant ID, diagnosis, session, and hip metadata; folds are written under
`folds/fold*/adni_cls/`.

For CEM outputs, add `--force-outward-winding` to reverse every inward
spherical face. This is a mesh-repair mode: it intentionally removes local
fold orientation from the saved mesh, and records the changed face count in
`face_winding_correction` metadata.

## MNIST

Use `examples/script_to_generate_dataset.py` with `--signal-type mnist`.

Basic command:
```bash
cd /path/to/spherepar
python examples/script_to_generate_dataset.py \
    data/meshes \
    --output-root data/generated_mnist \
    --signal-type mnist \
    --seed 41
```

MNIST-specific options:

- `--mnist-percentage <float>`: percentage of the full MNIST set to use (`0.1` to `100.0`).
  - Example: `--mnist-percentage 10` generates 10% of 70,000 images (7,000 samples).
- `--mnist-total-count <int>`: explicit number of MNIST samples to generate.
  - If provided, it **overrides** `--mnist-percentage`.
  - Must be `<= 70000`.

### How an MNIST image is projected onto a template

Each generated signal is a single `float32` value for every vertex of the
generated mesh.  The projection is performed in two resampling stages:

```text
28 x 28 MNIST image
  -> bilinear perspective projection on a 60 x 60 Driscoll--Healy S2 grid
  -> bilinear sampling of that grid at each mesh vertex direction
  -> one intensity per mesh vertex
```

More precisely, `save_sample_signal()` loads the selected OpenML
`mnist_784` image, converts its pixels to `[0, 1]`, and calls the S2CNN-derived
`project_2d_on_sphere()` helper with bandwidth `B = 30`.  Its
Driscoll-Healy grid has `2B x 2B = 60 x 60` samples, with

- polar angle `theta_j = pi * j / (2B)`, and
- azimuth `phi_k = pi * k / B`.

The helper uses a perspective projection whose origin is just beyond the
north pole, `(0, 0, 2.001)` in its shifted-sphere coordinates.  It bilinearly
samples the 28 x 28 image at the resulting planar coordinates; samples outside
the image are zero.  The helper then normalizes each spherical image to its own
`[0, 255]` range and stores it as `uint8`.  The generator divides it by 255
again before the mesh resampling, so the grid supplied to the mesh is in
`[0, 1]`.  Consequently, the final signal preserves the digit's spatial
pattern but not its original absolute grayscale scale.

For a mesh vertex `v`, the generator computes a direction from the mesh center
of mass `c`,

```text
d = (v - c) / ||v - c||
theta = acos(d_z)
phi = atan2(d_y, d_x) mapped to [0, 2pi)
```

It bilinearly samples the 60 x 60 grid at `(theta, phi)`.  Azimuth wraps at
`2pi`; polar sampling is clamped at the grid limits.  Finally, the value is
multiplied by `--signal-amplitude` and saved to
`signals/<sample_id>_mnist.npy` in the same order as `mesh.vertices`.

This mapping is radial and uses the generated mesh itself (including any
deformation), so the same spherical digit is sampled at the deformed vertex
directions.  It does **not** use `--param-method`, FLASH, or CEM to construct
the MNIST signal; those optional parametrizations are written as additional
artifacts only.  The orientation is fixed by the template coordinate axes:
`+z` is the north-pole direction and `+x`/`+y` define azimuth.  Thus templates
with different poses, centers of mass, or deformations can display the digit
at different vertex locations even when the MNIST index is the same.

The label's `signal` object records the source `mnist_index` and digit label,
plus `projection_method: "s2cnn_grid_to_mesh"`, the Driscoll-Healy grid,
bandwidth 30, and bilinear interpolation.  This is enough to identify the
source image and reproduce the current projection convention.

### Full MNIST generation examples

Full MNIST (70,000) **with deformations** (`case2_small,case3_large`):
```bash
python examples/script_to_generate_dataset.py \
    data/meshes \
    --output-root data/generated_mnist_full_deformed \
    --signal-type mnist \
    --mnist-total-count 70000 \
    --deformation-cases case2_small,case3_large \
    --seed 42
```

Full MNIST (70,000) **without deformations** (`case1_no`):
```bash
python examples/script_to_generate_dataset.py \
    data/meshes \
    --output-root data/generated_mnist_full_nodeform \
    --signal-type mnist \
    --mnist-total-count 70000 \
    --deformation-cases case1_no \
    --seed 42
```

### Output structure

```text
data/generated_mnist/
  meshes/              # generated mesh OBJ files
  signals/             # *_mnist.npy signals (MNIST projected on vertices)
  labels/              # sample label JSON files (includes mnist_index, mnist_label)
  spheres/             # spherical parametrization outputs (if enabled)
  logs/                # errors.log
  folds/               # only when --create-splits is used
    fold1/
      mnist_cls/
        train.txt
        val.txt
        test.txt
    ...
    summary.json
```

Key points:

- MNIST is downloaded automatically via `sklearn.datasets.fetch_openml("mnist_784")`.
- MNIST generation requires exactly one template mesh in `input_dir`.
- Samples use MNIST indices in dataset order, without repeats.
- When `--param-method flash` or `--param-method cem` is used, MNIST deformation cases also write spherical parametrization outputs.
- For `--split-tasks mnist_cls`: `train.txt` and `val.txt` use MNIST train partition (`mnist_index < 60000`), `test.txt` uses MNIST test partition (`mnist_index >= 60000`).

## Anisotropic Gaussian: projection and orientation target

The anisotropic signal is evaluated on the **generated (already deformed)**
mesh.  It is not transferred from the template mesh and it does not use the
optional FLASH/CEM spherical parametrization.  The output is one scalar value
per generated-mesh vertex, in exactly `mesh.vertices` order.

The generator first chooses one center vertex `c` on the generated mesh (the
anisotropic signal currently has one center).  It then computes a local
two-dimensional coordinate for **every** mesh vertex `x`, relative to that
center.  Those coordinates are needed because the Gaussian is an ellipse: it
must know both each vertex's distance from `c` and which in-plane direction it
lies in, so it can apply different widths parallel and perpendicular to the
major axis.

Concretely, the implementation treats the origin as the sphere center,
normalizes `c` and every `x` to directions on the unit sphere, and applies the
sphere logarithm map at `c_hat = c / ||c||`:

```text
y     = x / ||x||
theta = acos(clamp(dot(y, c_hat), -1, 1))
z     = theta / sin(theta) * (y - dot(y, c_hat) * c_hat)
```

Here `z` is the resulting 3-D vector in the 2-D plane tangent to the unit
sphere at `c_hat`; its norm is `theta`, the spherical (angular) distance from
the center direction.  If `x` has exactly the same direction as `c`—in
particular, for the selected center vertex itself—then `theta = 0`.  The code
uses the limiting value `z = 0` rather than evaluating the undefined-looking
ratio `theta / sin(theta)` at zero.

Normalizing to the unit sphere removes radial distance from the origin before
the local coordinates are computed.  Consequently, two vertices on the same
ray from the origin have the same local direction; the signal depends on their
spherical location, not how far they are from the origin.

The log map and the orientation frame use the **same** tangent plane,
`T_c_hat S2`.  The log map places every vertex in that plane as a vector `z`.
To orient the anisotropic Gaussian signal, a stable Hughes-Möller orthonormal
basis `(hm_e1, hm_e2)` is then constructed in that same plane.  It does not
move the vertices into a second tangent space; it only supplies perpendicular
directions with which to measure the already-computed vector `z`.

A sampled angle `delta` (modulo `pi`) selects the physical major-axis direction
within this basis:

```text
v = cos(delta) * hm_e1 + sin(delta) * hm_e2
v_perp = c_hat x v
```

For each mesh vertex `x`, the scalar signal value is obtained by taking the two
coordinates of its log-map vector `z`: one along `v` and one along
`v_perp`.  The widths are applied to those coordinates, and the result is

```text
u_parallel = dot(z, v)
u_perpendicular = dot(z, v_perp)
```

The signal value is then

```text
f(x) = A * exp(-1/2 * ((u_parallel / sigma_parallel)^2
                     + (u_perpendicular / sigma_perpendicular)^2))
```

Here `A` is `amplitude`; `sigma_parallel` (also called `sigma_u` internally)
is the width along `v`; and `sigma_perpendicular` (`sigma_v`) is the width
along `v_perp`.  In the standard generator, `sigma_perpendicular` is supplied
explicitly or is computed as `--signal-sigma-ratio * sigma_parallel`.  The
widths are measured in the log-map's unit-sphere angular-distance coordinate,
even though the current label schema records their units as
`surface_distance` for compatibility.

### How anisotropic orientation labels are computed

The Hughes-Möller frame is only used to sample the physical axis; it is not
the regression gauge.  To form a reproducible label, the generator projects a
fixed world gauge `g = (0, 0, 1)` into the tangent plane:

```text
g1 = normalize(g - dot(g, c_hat) * c_hat)
g2 = c_hat x g1
phi = atan2(dot(v, g2), dot(v, g1)) mod pi
orientation target = [cos(2 * phi), sin(2 * phi)]
```

Centers for which the projected gauge is too short (within the configured
threshold, default `0.05`) are rejected and another center is sampled.  With
a fixed center, the sample is skipped instead.  The doubled-angle target makes
the label unchanged when the undirected ellipse axis is reversed (`v` and
`-v` describe the same orientation).

In `signals[]`, the `aniso_000` entry records the signal definition and
diagnostics: `parameters.sigma_parallel`, `parameters.sigma_perpendicular`,
`parameters.orientation_angles` (`phi`), and
`parameters.orientation_targets_doubled_angle`.  Its `orientation_debug`
stores the physical `major_axis`, sampling angle `delta`, fixed-gauge frame
`gauge_e1/gauge_e2`, and Hughes--Möller frame `hm_e1/hm_e2`.

The corresponding `task_groups.anisotropic_gaussian.tasks` labels are derived
directly from that metadata:

- `center_regression.label`: XYZ coordinate of the selected center vertex on
  the deformed mesh.
- `amplitude_regression.label`: `A` used in the formula.
- `anisotropic_parameters_regression.label`: `sigma_parallel`,
  `sigma_perpendicular`, `orientation` (`phi`), its doubled-angle target, and
  the Hughes--Möller basis for diagnostics/reconstruction.
- `orientation_regression.label`: only `[cos(2phi), sin(2phi)]`; `valid` is
  true when this two-element target was produced.

## Labels

Each generated sample has a label file at:

- `labels/<sample_id>.json` (for example `labels/sphere_C000_s000010.json`)

Current schema is `schema_version: "0.2"`.

### Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `str` | Label schema version. |
| `sample_id` | `str` | Unique sample identifier (`<template>_sXXXXXX`). |
| `name` | `str` | Same as `sample_id`. |
| `metadata` | `object` | Dataset-level metadata (name/version/template/case/seed). |
| `paths` | `object` | Canonical dataset-root-relative artifact paths. |
| `mesh` | `object` | Mesh stats and geometry metadata. |
| `signal_files` | `object` | Mapping of signal keys to `.npy` files. |
| `signals` | `array` | Per-signal metadata entries (isotropic + anisotropic). |
| `task_groups` | `object` | Supervised task labels grouped by signal family. |
| `quality_checks` | `object` | Integrity flags written by generator. |
| `deformation` | `object` | Deformation parameters used to produce the mesh. |
| `parametrization` | `object` | Spherical parametrization status (`method`, `success`, `error`). |
| `random_seed` | `int` | Per-sample seed. |
| `warnings` | `array` | Generation warnings, if any. |
| `sphere_path` | `str` | Expected relative path to the sphere mesh; check `parametrization.success` before loading it. |

### `metadata`

| Field | Type | Meaning |
|---|---|---|
| `dataset_name` | `str` | Dataset logical name. |
| `dataset_version` | `str` | Dataset format version. |
| `template_id` | `str` | Source template mesh id. |
| `deformation_case` | `str` | Case name (`case1_no` through `case6_large_noise`). |
| `created_by` | `str` | Generator script identifier. |
| `random_seed` | `int` | Sample seed used for reproducibility. |

### `paths`

| Field | Type | Meaning |
|---|---|---|
| `mesh` | `str` | Relative path to sample OBJ (for example `meshes/...obj`). |
| `signal` | `str` | Canonical signal path (`iso_001_reg` for dual-Gaussian labels). |
| `label` | `str` | Relative path to this JSON label. |
| `sphere` | `str` | Expected relative path to the spherical parametrization OBJ. |
| `spherical_label` | `str` | Expected relative path to the spherical-parametrization metadata JSON. |

The flat `mesh_path`, `signal_path`, `label_path`, and `sphere_path` fields
mirror these canonical paths for compatibility. Use `parametrization.success`
to determine whether the sphere artifacts were actually created.

### `mesh`

| Field | Type | Meaning |
|---|---|---|
| `n_vertices` | `int` | Number of vertices. |
| `n_faces` | `int` | Number of triangular faces. |
| `topology_id` | `str` | Template/topology identifier. |
| `is_watertight` | `bool` | Watertightness flag. |
| `is_orientable` | `bool` | Orientability flag. |
| `coordinate_system` | `str` | Coordinate space (`xyz`). |
| `units` | `str` | Geometry units (`normalized`). |
| `distance_stats` | `object` | Deformation distance summary (mean/std). |

### `signal_files`

Typical keys:

- `iso_<N>`: isotropic signal with `N` centers (classification-style)
- `iso_001_cls`: single-center isotropic alias for classification tasks
- `iso_001_reg`: single-center isotropic alias for regression tasks
- `aniso_001`: single-center anisotropic signal

Values are relative `.npy` paths under `signals/`.

### `signals[]`

Each entry describes one stored signal tensor.

Common fields:

| Field | Type | Meaning |
|---|---|---|
| `signal_id` | `str` | Internal id (for example `iso_000`, `aniso_000`). |
| `family` | `str` | `isotropic` or `anisotropic`. |
| `model` | `str` | Signal model (`surface_gaussian`). |
| `storage` | `object` | `path_key`, `dtype`, `shape`, normalization info. |
| `num_centers` | `int` | Number of centers. |
| `centers` | `array` | Center coordinates (`[x,y,z]` per center). |
| `center_vertex_ids` | `array` | Vertex index of each center. |
| `center_sampling` | `object` | How center(s) were sampled/matched. |
| `amplitudes` | `array` | Per-center amplitudes. |
| `parameters` | `object` | Family-specific parameters. |
| `generation` | `object` | Post-processing/generation options. |

Isotropic `parameters`:

- `sigmas`: per-center width values.
- `distance_type`: distance convention.
- `sigma_units`: units for sigma.

Anisotropic `parameters`:

- `sigma_parallel`, `sigma_perpendicular`
- `orientation_angles`: gauge-relative angle(s) `phi` in radians, modulo `pi`
- `orientation_targets_doubled_angle`: `[cos(2phi), sin(2phi)]`
- `hm_basis.e1`, `hm_basis.e2`: Hughes–Möller first tangent basis used for axis sampling
- `orientation_period`: `pi`
- `orientation_target`: representation name (`cos2phi_sin2phi`)
- `frame`, `distance_type`, `sigma_units`

Anisotropic `orientation_debug` (extra diagnostics):

- `center_unit`
- `major_axis` (physical major axis `v`)
- `delta` (sampled HM perturbation angle in `[0,pi)`)
- `phi` (gauge-relative angle, modulo `pi`)
- `target_doubled_angle`
- `gauge_e1`, `gauge_e2` (projected fixed-gauge frame)
- `hm_e1`, `hm_e2` (HM tangent frame used to construct `v`)

### `task_groups`

Two groups are typically present:

- `isotropic_gaussian`
- `anisotropic_gaussian`

Each group has:

- `signal_id`, `family`
- `tasks`: task definitions with `valid`, `label`, `dtype`, and task-specific metadata.

Common task entries:

- `number_of_centers`
- `center_regression`
- `sigma_regression` (isotropic)
- `amplitude_regression`
- `anisotropic_parameters_regression` (anisotropic)
- `orientation_regression` (anisotropic)

Important: anisotropic `orientation_regression.label` uses doubled-angle encoding:

- `label = [cos(2phi), sin(2phi)]`

This makes labels invariant to axis sign (`v` and `-v` are equivalent).

### `quality_checks`

Boolean integrity checks written into the label (file existence, signal length consistency, finite values, etc.).

### `deformation`

Fields vary by case, typically including:

- `max_ratio`, `num_candidates`, `group_candidates`, `alpha`
- `smooth_iterations`, `ring_size`
- `deform_method`, `max_iter`

### `parametrization`

| Field | Type | Meaning |
|---|---|---|
| `method` | `str \| null` | `flash`, `cem`, `spheremap`, or `null` when not run. |
| `success` | `bool` | Whether spherical parametrization succeeded. |
| `error` | `str \| null` | Error text if parametrization failed. |
| `cem_radius` | `float \| null` | Requested base CEM radius. |
| `cem_selected_radius` | `float \| null` | Radius selected by the optional search. |
| `use_idt_remesh` | `bool` | Whether connectivity-only IDT preprocessing was requested. |
| `adaptive_radius` | `bool` | Whether collapsed results trigger radius search. |
| `reject_retry` | `bool` | Whether the collapse acceptance policy is enforced. |
