# Problem: Absolute Paths Break Portability

## Summary

When datasets are moved to different directories, absolute paths in JSON label files break. This prevents datasets from being portable and shareable.

---

## The Problem

### Before: Absolute Paths (BROKEN ❌)

Paths saved to JSON were absolute:

```json
{
  "paths": {
    "mesh": "/tmp/xyz/meshes/sample.obj",
    "sphere": "/tmp/xyz/spheres/sample.obj"
  },
  "signal_files": {
    "iso_001": "signals/sample.npy"  ← Only this was relative!
  }
}
```

### What happens when dataset moves?

```
Original location: /tmp/xyz/
  ✓ Mesh exists: /tmp/xyz/meshes/sample.obj
  ✓ Sphere exists: /tmp/xyz/spheres/sample.obj
  ✓ Signals exist: (relative paths work)

After moving to: /home/data/
  ✗ Mesh path: /tmp/xyz/meshes/sample.obj  (NO LONGER EXISTS!)
  ✗ Sphere path: /tmp/xyz/spheres/sample.obj  (NO LONGER EXISTS!)
  ✓ Signals: /home/data/signals/sample.npy  (works)
```

### Validation Error

```
Error: signal file missing: signals/sphere_C001_s000016_iso_000.npy
```

Why? Because code tried to load relative paths without knowing the dataset root directory.

---

## Root Causes

### Issue 1: Generation Code Saved Absolute Paths

**File**: `spherepar/benchmark/dataset_generator.py`

**Line 943** (save_spherical_parametrization):
```python
return {
    "sphere": str(sphere_path),  # ← ABSOLUTE PATH
    "spherical_label": str(sphere_label_path),
}
```

**Lines 1777-1778** (final_label creation):
```python
"paths": {
    "mesh": str(mesh_path),  # ← ABSOLUTE PATH
    "label": str(final_labels_path),  # ← ABSOLUTE PATH
},
```

### Issue 2: Validation Code Couldn't Resolve Relative Paths

**File**: `spherepar/benchmark/dataset_generator.py`

**Lines 1017-1075** (validate_saved_sample):
```python
# Before fix - code tried to use paths directly
mesh_path = Path(mesh_path)
if not mesh_path.exists():  # ← Fails for relative paths!
    issues.append(f"mesh file missing: {mesh_path}")
```

### Issue 3: Plotting Code Couldn't Find Signals

**File**: `examples/example_plot_signal.py`

**Lines 115-150** (load_signal):
```python
# Before fix - hardcoded path construction
signal_path = input_dir / "signals" / f"{sample_name}_iso_000.npy"
# Or tried to load without resolving relative paths
```

---

## Solution

### Part 1: Save Relative Paths (Generation)

**File**: `spherepar/benchmark/dataset_generator.py`

#### Fix 1: save_spherical_parametrization (lines 942-945)

```python
# BEFORE
return {
    "sphere": str(sphere_path),
    "spherical_label": str(sphere_label_path),
}

# AFTER
return {
    "sphere": str(sphere_path.relative_to(root_path)),
    "spherical_label": str(sphere_label_path.relative_to(root_path)),
}
```

#### Fix 2: paths dict (lines 1776-1779)

```python
# BEFORE
"paths": {
    "mesh": str(mesh_path),
    "label": str(final_labels_path),
},

# AFTER
"paths": {
    "mesh": str(mesh_path.relative_to(root_path)),
    "label": str(final_labels_path.relative_to(root_path)),
},
```

### Part 2: Resolve Relative Paths (Validation)

**File**: `spherepar/benchmark/dataset_generator.py`

**Lines 1017-1075** (validate_saved_sample):

```python
# Extract dataset root from label file path
label_file = Path(label_path)  # e.g., /data/labels/sample.json
dataset_root = label_file.parent.parent  # /data

# For each path, resolve relative to dataset_root
if mesh_path:
    mesh_path = Path(mesh_path)
    if not mesh_path.is_absolute():
        mesh_path = dataset_root / mesh_path  # Make absolute
    if not mesh_path.exists():
        issues.append(f"mesh file missing: {mesh_path}")

# Same pattern for sphere_path and signal_files
for rel_path in label.get("signal_files", {}).values():
    if rel_path:
        path = Path(rel_path)
        if not path.is_absolute():
            path = dataset_root / path  # Make absolute
        if not path.exists():
            issues.append(f"signal file missing: {path}")
```

### Part 3: Resolve Relative Paths (Plotting)

**File**: `examples/example_plot_signal.py`

**Lines 83-286** (load_signal and main):

```python
# Updated load_signal to accept label_data
def load_signal(
    input_dir: Path,
    sample_name: str,
    mesh: trimesh.Trimesh,
    signal_type: str = "auto",
    label_data: dict = None  # ← NEW
) -> tuple[np.ndarray, str]:
    signal_path = None
    
    # Try loading from signal_files if available
    if label_data and "signal_files" in label_data:
        signal_files = label_data.get("signal_files", {})
        for key in signal_files:
            if key.startswith("iso_") and not key.endswith("_cls"):
                rel_path = signal_files[key]
                signal_path = input_dir / rel_path  # Resolve relative
                if signal_path.is_file():
                    break

# Updated main to load and pass label_data
with open(label_path) as fh:
    label_data = json.load(fh)

signal, signal_source = load_signal(
    input_dir, sample_name, mesh,
    signal_type=args.signal_type,
    label_data=label_data  # ← PASS HERE
)
```

---

## Result: After the Fix

### JSON Now Contains Relative Paths ✅

```json
{
  "paths": {
    "mesh": "meshes/sample.obj",
    "sphere": "spheres/sample.obj"
  },
  "signal_files": {
    "iso_001": "signals/sample.npy",
    "iso_001_cls": "signals/sample.npy",
    "aniso_001": "signals/sample.npy"
  }
}
```

### Dataset Works When Moved ✅

```
Original location: /tmp/xyz/
  ✓ Mesh: /tmp/xyz/meshes/sample.obj
  ✓ Sphere: /tmp/xyz/spheres/sample.obj
  ✓ Signals: /tmp/xyz/signals/...

After moving to: /home/data/
  Extract root: /home/data/labels/../ = /home/data
  ✓ Mesh: /home/data/meshes/sample.obj  (RESOLVED!)
  ✓ Sphere: /home/data/spheres/sample.obj  (RESOLVED!)
  ✓ Signals: /home/data/signals/...  (RESOLVED!)
```

---

## Path Resolution Algorithm

```python
# When loading a label from disk:
label_path = Path("/data/labels/sample.json")

# Extract dataset root
dataset_root = label_path.parent.parent  # /data

# For any relative path in label:
relative_path = "signals/sample.npy"
absolute_path = dataset_root / relative_path  # /data/signals/sample.npy

# Now use absolute path safely
signal = np.load(absolute_path)
```

---

## Complete File Changes

### spherepar/benchmark/dataset_generator.py

| Line | Change | Reason |
|------|--------|--------|
| 943 | `sphere_path.relative_to(root_path)` | Save sphere as relative |
| 944 | `sphere_label_path.relative_to(root_path)` | Save label as relative |
| 1777 | `mesh_path.relative_to(root_path)` | Save mesh as relative |
| 1778 | `final_labels_path.relative_to(root_path)` | Save label as relative |
| 1017-1023 | Extract root, resolve mesh path | Validate mesh |
| 1029-1041 | Extract root, resolve signal paths | Validate signals |
| 1069-1073 | Extract root, resolve sphere path | Validate sphere |

### examples/example_plot_signal.py

| Line | Change | Reason |
|------|--------|--------|
| 88 | Add `label_data: dict = None` | Accept label data |
| 115-150 | Resolve relative paths | Load from signal_files |
| 272-273 | Load label from JSON | Get signal_files |
| 286 | Pass label_data | Use signal_files |

---

## Backward Compatibility

✅ **Old schemas still work**:
- Code checks `is_absolute()` before resolving
- Falls back to old naming scheme if signal_files not found
- Supports mixed absolute and relative paths

✅ **No breaking changes**:
- New schema uses relative paths
- Old schema uses absolute paths
- Both supported in validation/plotting code

---

## Testing

### Test 1: Generation Saves Relative Paths ✅
```
mesh_path = /data/meshes/sample.obj
Saved as: "meshes/sample.obj"  ✓
```

### Test 2: Validation Resolves Paths ✅
```
Load from: /data/labels/sample.json
Extract root: /data
Resolve: "meshes/sample.obj" → /data/meshes/sample.obj ✓
Check exists: ✓
```

### Test 3: Dataset Portability ✅
```
Original: /tmp/xyz/
Move to: /home/data/
Validation at new location: ✓ Still works!
```

### Test 4: Plotting Works ✅
```
Load label: /data/labels/sample.json
Extract root: /data
Load signal: /data/signals/sample.npy  ✓
```

---

## Summary

| Issue | Before | After |
|-------|--------|-------|
| mesh_path | Absolute ❌ | Relative ✅ |
| sphere_path | Absolute ❌ | Relative ✅ |
| signal_files | Relative ✅ | Relative ✅ |
| Portability | Broken ❌ | Works ✅ |
| Validation | Failed ❌ | Passes ✅ |
| Plotting | Failed ❌ | Works ✅ |

---

## Files Modified

1. **spherepar/benchmark/dataset_generator.py**
   - Lines 943-944: Relative sphere paths
   - Lines 1777-1778: Relative mesh/label paths
   - Lines 1017-1075: Path resolution for validation

2. **examples/example_plot_signal.py**
   - Lines 83-89: Accept label_data
   - Lines 115-150: Resolve signal_files paths
   - Lines 272-286: Load and use label_data

---

## Status

✅ COMPLETE AND TESTED

- All paths now relative ✅
- All paths resolved correctly ✅
- Datasets portable ✅
- Backward compatible ✅
- Tests pass ✅
