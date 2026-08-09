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

## Importing existing meshes (generic and FAUST)

Use `examples/script_to_generate_dataset_from_obj.py` when the input meshes
already exist and should be copied into the dataset layout without deformation.
For every supported input mesh (`.obj`, `.ply`, `.stl`, or `.off`), it writes a
mesh OBJ, one per-vertex signal array, one label JSON, and a spherical
parametrization (FLASH by default, or CEM).

Generic meshes receive an all-zero `float32` signal with one value per vertex:

```bash
python examples/script_to_generate_dataset_from_obj.py \
    --input-dir data/source_meshes \
    --output-root data/imported_meshes
```

For FAUST, point `--faust-dir` at the FAUST root containing `registrations/`:

```bash
python examples/script_to_generate_dataset_from_obj.py \
    --faust-dir /path/to/FAUST \
    --output-root data/imported_faust
```

`--faust-dir` automatically selects FAUST mode; `--dataname FAUST` is not
required. In this mode the signal is the `float32` vertex index array
`[0, 1, ..., N-1]`, and the label metadata records `dataname: "FAUST"`. The
vertex ordering is therefore part of the signal definition. These imported
datasets contain one `main` signal per mesh, rather than the isotropic and
anisotropic Gaussian signal set produced by `script_to_generate_dataset.py`.

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
| `paths` | `object` | Relative paths for mesh and label file. |
| `mesh` | `object` | Mesh stats and geometry metadata. |
| `signal_files` | `object` | Mapping of signal keys to `.npy` files. |
| `signals` | `array` | Per-signal metadata entries (isotropic + anisotropic). |
| `task_groups` | `object` | Supervised task labels grouped by signal family. |
| `quality_checks` | `object` | Integrity flags written by generator. |
| `deformation` | `object` | Deformation parameters used to produce the mesh. |
| `parametrization` | `object` | Spherical parametrization status (`method`, `success`, `error`). |
| `random_seed` | `int` | Per-sample seed. |
| `warnings` | `array` | Generation warnings, if any. |
| `sphere_path` | `str \| null` | Relative path to generated sphere mesh when available. |

### `metadata`

| Field | Type | Meaning |
|---|---|---|
| `dataset_name` | `str` | Dataset logical name. |
| `dataset_version` | `str` | Dataset format version. |
| `template_id` | `str` | Source template mesh id. |
| `deformation_case` | `str` | Case name (`case1_no`, `case2_small`, `case3_large`). |
| `created_by` | `str` | Generator script identifier. |
| `random_seed` | `int` | Sample seed used for reproducibility. |

### `paths`

| Field | Type | Meaning |
|---|---|---|
| `mesh` | `str` | Relative path to sample OBJ (for example `meshes/...obj`). |
| `label` | `str` | Relative path to this JSON label. |

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
| `method` | `str \| null` | `flash`, `cem`, or `null` when not run. |
| `success` | `bool` | Whether spherical parametrization succeeded. |
| `error` | `str \| null` | Error text if parametrization failed. |
