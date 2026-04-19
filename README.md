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

