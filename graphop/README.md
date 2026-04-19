# Build this module

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

