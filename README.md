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

## MNIST

The `generate_mnist.py` script creates a synthetic dataset of 2D shapes based on the MNIST digit dataset. It generates deformed versions of the digit '0' and saves them as OBJ files along with corresponding signal data.


How to use script_to_generate_dataset.py to create MNIST dataset

Basic Command Structure:
```bash
 cd /path/to/spherepar
 python examples/script_to_generate_dataset.py \
     --input-dir data/meshes \
     --output-root data/generated_mnist \
     --signal-type mnist \
     --n-samples-per-mesh 9 \
     --seed 41
```


MNIST-Specific Parameters:

 --signal-type mnist
     Use MNIST images instead of synthetic Gaussian signals

 --signal-amplitude 1.0
     Scale factor for MNIST pixel intensities (default 1.0)

 --deformation-cases case1_no
     Optional: skip deformations, only generate signals on original mesh
     (remove to generate both cases)

Example Usage:

Option 1: Generate MNIST signals with deformations
```bash

 python examples/script_to_generate_dataset.py \
     --input-dir data/ \
     --output-root data/generated_mnist \
     --signal-type mnist \
     --n-samples-per-mesh 5 \
     --seed 42 \
     --deformation-cases case2_small,case3_large
```

Option 2: Generate MNIST signals without deformations (original meshes only)

```bash
 python examples/script_to_generate_dataset.py \
     --input-dir data/ \
     --output-root data/generated_mnist_no_deform \
     --signal-type mnist \
     --n-samples-per-mesh 1 \
     --deformation-cases case1_no \
     --seed 42
```

Output Structure:
```
 data/generated_mnist/
     meshes/          — mesh OBJ files
     signals/         — *_mnist.npy files (MNIST projected onto vertices)
     labels/          — JSON metadata (includes mnist_index, mnist_label)
     spheres/         — spherical parametrization OBJ files
     logs/            — errors.log
```


 Key Points:

 - MNIST images downloaded automatically via scikit-learn's fetch_openml()
 - Spherical projection: S2CNN Driscoll-Healy sphere sample -> mesh vertices (bilinear interpolation)
 - MNIST requires exactly one template mesh in the input directory
 - Each sample includes: original mesh index, MNIST digit label (0-9)
 - MNIST samples are taken in dataset order (no repeats), up to --mnist-percentage or --mnist-total-count
 - MNIST projection uses full-sphere resampling (no ROI masking)
 - No signal_sigma/signal_num_centers needed for MNIST (ignored automatically)
