# BUILD.md — Stage 1 synthetic benchmark pipeline

This document explains how to build the `graphop` C++ extension and run the
Stage 1 demo inside the `sherepar` project.

---

## Prerequisites

Install system packages (Ubuntu / Debian):

```bash
sudo apt-get install -y \
    libcgal-dev      \   # CGAL 5.x (header-only + Boost)
    libeigen3-dev    \   # Eigen 3.4 (required by CGAL's solver backend)
    pybind11-dev     \   # pybind11 C++ headers
    python3-pybind11 \   # CMake find-module for pybind11
    python3-numpy    \   # NumPy (also needed at build time for include path)
    python3-scipy    \   # SciPy (Python-side only)
    cmake            \   # CMake ≥ 3.14
    build-essential      # GCC / G++
```

Or install Python packages via pip if preferred:

```bash
pip install pybind11 numpy scipy trimesh
```

---

## Build

From the repository root:

```bash
mkdir -p build && cd build
cmake ..                       \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=..
make -j$(nproc)
make install          # copies graphop.so → repo root
```

After `make install` the compiled extension sits at the repo root:

```
sherepar/
  graphop.cpython-3xx-linux-gnu.so   ← importable as "import graphop"
  ...
```

> **Tip**: you can also `cd build && cmake --install .` instead of `make install`.

---

## Verify the build

```python
import sys
sys.path.insert(0, "/path/to/sherepar")   # only needed if not on PYTHONPATH
import graphop
help(graphop.deform_surface)
```

---

## Generate a template mesh

The demo needs an OBJ mesh.  Generate a simple ellipsoid:

```bash
cd /path/to/sherepar
python examples/generate_mesh_elipsoid.py   # writes data/ellipsoid.obj  (note: existing script name)
```

Or provide your own genus-0 closed surface OBJ.

---

## Run the Stage 1 demo

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