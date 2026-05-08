# Unified Dataset Generation Pipeline

This document describes the mesh-deformation dataset generation pipeline with unified storage, deformation case stratification, and task-specific split generation.

---

## Overview

The dataset pipeline generates synthetic mesh deformations with associated signals and metadata, enabling multiple machine learning tasks (classification and regression) from a single unified dataset. Rather than creating separate dataset directories for each task, all data is stored in one location, with task-specific views defined through metadata-driven split files.

### Key Features

- **Unified Storage**: Single dataset root with all meshes, signals, and metadata
- **Explicit Deformation Cases**: Three predefined parameter ranges (CASE 1 no-deformation, CASE 2 small, CASE 3 large)
- **Rich Metadata**: Per-sample JSON labels with complete signal, deformation, and parametrization information
- **Task-Specific Splits**: Automatic generation of train/val/test split files for each task
- **Template-Grouped Folds**: K-fold assignment at the template level to prevent leakage
- **Class Balancing**: Deterministic, reproducible split generation with class balance for classification
- **Randomized Signal Parameters**: Per-sample sigma and amplitude variation (±20% by default, configurable)

---

## Dataset Structure

```
dataset/
├── meshes/
│   ├── sample_s000001.obj
│   ├── sample_s000002.obj
│   └── ...
├── signals/
│   ├── sample_s000001.npy
│   ├── sample_s000002.npy
│   └── ...
├── spheres/
│   ├── sample_s000001.obj
│   ├── sample_s000002.obj
│   └── ...
├── labels/
│   ├── sample_s000001.json
│   ├── sample_s000002.json
│   └── ...
├── logs/
│   └── errors.log
└── folds/
    ├── fold1/
    │   ├── number_of_centers/
    │   │   ├── train.txt
    │   │   ├── val.txt
    │   │   └── test.txt
    │   ├── center_regression/
    │   │   ├── train.txt
    │   │   ├── val.txt
    │   │   └── test.txt
    │   ├── sigma_regression/
    │   │   ├── train.txt
    │   │   ├── val.txt
    │   │   └── test.txt
    │   └── amplitude_regression/
    │       ├── train.txt
    │       ├── val.txt
    │       └── test.txt
    └── fold2/ ...
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `meshes/` | Deformed OBJ mesh files (one per sample) |
| `signals/` | Per-vertex signal arrays in NumPy format (.npy) |
| `spheres/` | OBJ meshes with spherical parametrization coordinates |
| `labels/` | JSON metadata files (one per sample) |
| `logs/` | Generation logs including errors and validation results |
| `folds/` | Task-specific split files organized by fold and task |

---

## Deformation Cases

The pipeline supports three explicit deformation case configurations:

### CASE 1: No Deformation (Baseline)

Original mesh without any deformation. Signals are generated on the undeformed mesh. This serves as a baseline for comparison and for tasks that only need signal analysis without geometric variation:

```python
{
    "name": "case1_no",
    "max_ratio": (0.0, 0.0),              # No deformation
    "num_candidates": (0, 0),              # No handles
    "group_candidates": (1, 1),            # N/A
    "alpha": (0.0, 0.0),                   # N/A
    "smooth_iterations": (0, 0),           # N/A
    "ring_size": (0, 0),                   # N/A
}
```

**Properties**:
- `param_method` is forced to `None` (no parametrization)
- Mesh quality is always valid (original mesh)
- Signal parameters are randomized normally
- All task labels are computed on the original mesh

### CASE 2: Small Deformations

Conservative parameter ranges suitable for learning robust representations:

```python
{
    "name": "case2_small",
    "max_ratio": (0.02, 0.08),           # 2–8% max displacement
    "num_candidates": (3, 8),             # 3–8 handle points
    "group_candidates": (1, 3),           # grouping factor 1–3
    "alpha": (1.5, 3.5),                  # deformation strength
    "smooth_iterations": (2, 5),          # smoothing passes
}
```

### CASE 3: Large Deformations

Challenging parameter ranges to test generalization:

```python
{
    "name": "case3_large",
    "max_ratio": (0.08, 0.25),           # 8–25% max displacement
    "num_candidates": (5, 15),            # 5–15 handle points
    "group_candidates": (2, 5),           # grouping factor 2–5
    "alpha": (2.5, 5.5),                  # deformation strength
    "smooth_iterations": (1, 3),          # lighter smoothing
}
```

**Generation Strategy**: For each sample, a deformation case (case1_no, case2_small, or case3_large) is first selected, then all parameters are randomly sampled from that case's ranges. This ensures both case diversity and parameter diversity within each case.

---

## Label JSON Schema

Each sample produces a JSON label file with complete metadata:

```json
{
  "sample_id": "sample_s000001",
  "template_id": "mesh_name",
  "deformation_case": "case2_small",
  "random_seed": 42,
  
  "signal": {
    "num_centers": 2,
    "centers": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    "vertex_ids": [123, 456, 789],
    "sigmas": [0.5, 0.3],
    "amplitudes": [1.0, 0.8]
  },
  
  "deformation": {
    "max_ratio": 0.05,
    "num_candidates": 5,
    "group_candidates": 2,
    "alpha": 2.8,
    "smooth_iterations": 3
  },
  
  "parametrization": {
    "method": "flash",
    "success": true
  },
  
  "task_validity": {
    "number_of_centers": true,
    "center_regression": false,
    "sigma_regression": false,
    "amplitude_regression": false
  }
}
```

### Field Descriptions

- **sample_id**: Unique sample identifier (format: `sample_sNNNNNN`)
- **template_id**: Source template mesh name
- **deformation_case**: Which case this sample belongs to
- **random_seed**: RNG seed used for reproducibility
- **signal.num_centers**: Number of Gaussian sources (1–5)
- **signal.centers**: 3D coordinates of each Gaussian center
- **signal.vertex_ids**: Vertex indices where signal was evaluated
- **signal.sigmas**: Standard deviation for each Gaussian
- **signal.amplitudes**: Amplitude for each Gaussian
- **deformation.\***: Actual parameters used for this sample's deformation
- **parametrization.method**: Spherical parametrization method (flash, cem, none)
- **parametrization.success**: Whether parametrization succeeded
- **task_validity**: Boolean flags indicating which tasks this sample is valid for

---

## Task-Specific Splits

The pipeline generates splits for four machine learning tasks:

### 1. Number of Centers Classification

**Task**: Predict the number of Gaussian sources (1, 2, 3, 4, or 5).

**Valid Samples**: All samples with `num_centers ∈ {1, 2, 3, 4, 5}`

**Class Balancing**: Splits are stratified to maintain balanced class distribution across folds.

### 2. Center Regression

**Task**: Predict the 3D coordinates of Gaussian centers.

**Valid Samples**: Only samples with `num_centers == 1` (single-center samples only)

**Constraint**: Regression is cleaner with deterministic single source; multi-center regression requires tracking individual center order.

### 3. Sigma Regression

**Task**: Predict the standard deviation parameter(s) of Gaussian(s).

**Valid Samples**: Only samples with `num_centers == 1`

**Note**: Future work may extend to multi-center sigma regression with center tracking.

### 4. Amplitude Regression

**Task**: Predict the amplitude parameter(s) of Gaussian(s).

**Valid Samples**: Only samples with `num_centers == 1`

**Note**: Similar constraint as sigma regression; single-center for cleaner training.

### Split File Format

Each split file (e.g., `fold1/number_of_centers/train.txt`) contains sample IDs, one per line:

```
sample_s000001
sample_s000034
sample_s000067
```

Split files are non-overlapping within a fold: no sample appears in both train.txt and val.txt or test.txt.

---

## Fold Generation

### Template Grouping

Folds are assigned at the **template level**, not the sample level. All samples derived from the same template are assigned to the same fold. This prevents data leakage when comparing different deformation variants of the same mesh across train/val/test sets.

**Example**:
- Template `mesh_a`: samples 1, 2, 3, 4, 5 → all assigned to fold 1
- Template `mesh_b`: samples 6, 7, 8, 9, 10 → all assigned to fold 2

### Deterministic Seeding

All fold assignments use a seeded random number generator (RNG), ensuring reproducibility across runs. The same `--seed` value will always produce identical fold assignments.

---

## Generation Usage

### Prerequisites

Install dependencies:

```bash
pip install trimesh scipy rtree numpy pybind11
```

Build the C++ extension:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=..
make -j$(nproc) && make install
```

### Basic Generation (Both Cases, No Splits)

Generate 100 samples per template per center option from both deformation cases:

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/dataset \
  --n-samples-per-mesh 100 \
  --seed 42
```

### Full Pipeline (Both Cases + Task Splits)

Generate dataset and create task-specific split files:

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/dataset \
  --n-samples-per-mesh 100 \
  --deformation-cases case2_small,case3_large \
  --signal-centers-options 1,2,3,4,5 \
  --create-splits \
  --num-folds 5 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

### Case-Specific Generation

Generate only CASE 2 (small deformations):

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/dataset_case2 \
  --n-samples-per-mesh 50 \
  --deformation-cases case2_small \
  --seed 42
```

Generate only CASE 3 (large deformations):

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/dataset_case3 \
  --n-samples-per-mesh 50 \
  --deformation-cases case3_large \
  --seed 42
```

### With Custom Spherical Parametrization

Specify parametrization method (flash, cem, or none):

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/dataset \
  --n-samples-per-mesh 100 \
  --param-method flash \
  --create-splits \
  --seed 42
```

### Custom Split Parameters

Adjust fold count and train/val/test ratios:

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/dataset \
  --n-samples-per-mesh 100 \
  --create-splits \
  --num-folds 10 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

### With Custom Signal Parameter Variation

Control sigma and amplitude randomization range (default ±20%):

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/dataset \
  --n-samples-per-mesh 100 \
  --signal-sigma-variation 30.0 \
  --signal-amplitude-variation 25.0 \
  --create-splits \
  --seed 42
```

---

## CLI Reference

### Main Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input_dir` (positional) | str | required | Directory containing template .obj meshes |
| `--output-root` | str | "data/generated" | Dataset output root directory |
| `--n-samples-per-mesh` | int | 25 | Number of samples per template per center option |
| `--seed` | int | 42 | Random seed for reproducibility |

### Deformation Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--deformation-cases` | str | "case2_small,case3_large" | Cases to generate (comma-separated) |
| `--max-ratio` | float | 0.8 | Deprecated; use `--deformation-cases` |
| `--group-candidates` | int | 5 | Deprecated; use `--deformation-cases` |

### Signal Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--signal-type` | str | "isotropic" | Signal type (isotropic or anisotropic) |
| `--signal-centers-options` | str | None | Gaussian center counts (comma-separated, e.g., "1,2,3,4,5") |
| `--signal-sigma` | float | 0.2 | Base Gaussian sigma parameter |
| `--signal-amplitude` | float | 1.0 | Base Gaussian amplitude parameter |
| `--signal-sigma-variation` | float | 20.0 | Sigma variation as percentage (±X%) |
| `--signal-amplitude-variation` | float | 20.0 | Amplitude variation as percentage (±X%) |

### Spherical Parametrization

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--param-method` | str | "none" | Parametrization method (flash, cem, or none) |

### Split Generation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--create-splits` | flag | False | Generate task-specific split files |
| `--num-folds` | int | 5 | Number of K-folds |
| `--train-ratio` | float | 0.7 | Training set ratio (0–1) |
| `--val-ratio` | float | 0.15 | Validation set ratio (0–1) |
| `--test-ratio` | float | 0.15 | Test set ratio (0–1) |

### Other Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--repair-holes` / `--no-repair-holes` | flag | True | Repair mesh holes before processing |
| `--drop-non-watertight` | flag | False | Drop non-watertight meshes after deformation |

---

## Loading and Using Splits in Your Model

### Load Splits Programmatically

```python
from pathlib import Path

def load_split(dataset_root, fold_id, task, split_name):
    """Load sample IDs from a split file."""
    split_file = Path(dataset_root) / "folds" / f"fold{fold_id}" / task / f"{split_name}.txt"
    with open(split_file, "r") as fh:
        return [line.strip() for line in fh if line.strip()]

# Example
dataset_root = "data/dataset"
fold = 1
task = "number_of_centers"

train_samples = load_split(dataset_root, fold, task, "train")
val_samples = load_split(dataset_root, fold, task, "val")
test_samples = load_split(dataset_root, fold, task, "test")

print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
```

### Load Sample Data

```python
import json
import numpy as np
from pathlib import Path

def load_sample(dataset_root, sample_id):
    """Load mesh, signal, and metadata for a sample."""
    root = Path(dataset_root)
    
    # Load metadata
    with open(root / "labels" / f"{sample_id}.json", "r") as fh:
        label = json.load(fh)
    
    # Load signal
    signal = np.load(root / "signals" / f"{sample_id}.npy")
    
    # Load mesh (requires trimesh)
    import trimesh
    mesh = trimesh.load(root / "meshes" / f"{sample_id}.obj")
    
    # Load parametrized sphere
    sphere = trimesh.load(root / "spheres" / f"{sample_id}.obj")
    
    return {
        "label": label,
        "signal": signal,
        "mesh": mesh,
        "sphere": sphere,
    }

# Example
sample = load_sample("data/dataset", "sample_s000001")
print(f"Signal shape: {sample['signal'].shape}")
print(f"Mesh vertices: {len(sample['mesh'].vertices)}")
print(f"Num Gaussian centers: {sample['label']['signal']['num_centers']}")
```

---

## Validation and Error Handling

The pipeline performs several validation checks:

1. **File Existence**: Verifies all expected artifacts (mesh, signal, label, sphere) are created
2. **JSON Schema**: Ensures all required fields are present in label JSON
3. **Signal Integrity**: Checks for NaN values and verifies signal length matches vertex count
4. **Parametrization Success**: Logs parametrization failures and records them in the label

Validation failures are logged in `logs/errors.log` but do not abort generation. Each sample's validation status can be checked via the `task_validity` field in its label.

---

## Technical Details

### Per-Sample Deformation Config Sampling

For each sample, the generation pipeline:
1. Selects a deformation case (CASE 2 or CASE 3)
2. Samples all deformation parameters from that case's ranges
3. Applies the deformation using `graphop`
4. Generates signal(s) with specified number of centers
5. Computes spherical parametrization
6. Saves mesh, signal, sphere, and complete label

### Deterministic Split Generation

The `build_task_splits()` function uses a seeded RNG to:
1. Assign templates to folds (template-level grouping)
2. Shuffle samples within each fold and task
3. Distribute into train/val/test according to specified ratios
4. For classification, ensure class balance within each split

With the same seed, identical splits are always reproduced.

### Memory and Performance

- **Per-sample size**: ~2 MB (mesh + signal + label)
- **Total dataset**: N samples × 2 MB ≈ 2N MB (rough estimate)
- **Split file generation**: O(N log N) with sorting; typically < 1 second

---

## Examples

### Example 1: Generate Small Dataset for Testing

```bash
python examples/script_to_generate_dataset.py \
  --input data/ \
  --output /tmp/test_dataset \
  --n 10 \
  --deformation-cases case2_small \
  --signal-centers-options 1,2 \
  --create-splits \
  --num-folds 2 \
  --seed 12345
```

### Example 2: Generate Large Production Dataset

```bash
python examples/script_to_generate_dataset.py \
  --input data/mesh_templates/ \
  --output /data/sherepar_v1 \
  --n 500 \
  --deformation-cases case2_small,case3_large \
  --signal-centers-options 1,2,3,4,5 \
  --create-splits \
  --num-folds 5 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 777
```

### Example 3: Classification Task with Custom Fold Strategy

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/classification_dataset \
  --n-samples-per-mesh 200 \
  --deformation-cases case2_small,case3_large \
  --signal-centers-options 1,2,3,4,5 \
  --create-splits \
  --num-folds 10 \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 999
```

### Example 4: Baseline with No Deformation (case1_no)

Generate signals only on original meshes without deformation:

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/baseline_dataset \
  --n-samples-per-mesh 100 \
  --deformation-cases case1_no \
  --signal-centers-options 1,3,5 \
  --create-splits \
  --seed 444
```

**Result**: Original mesh with only signal variation. No parametrization (param_method forced to None).

### Example 5: Custom Signal Parameter Variation

```bash
python examples/script_to_generate_dataset.py \
  data/ \
  --output data/high_variation_dataset \
  --n-samples-per-mesh 150 \
  --signal-sigma-variation 50.0 \
  --signal-amplitude-variation 40.0 \
  --create-splits \
  --seed 555
```

---

## Troubleshooting

### Issue: "graphop module not found"

**Solution**: Rebuild the C++ extension as described in BUILD.md.

### Issue: Split files are empty

**Possible causes**:
- No samples meet task validity criteria
- Ratios too small for number of samples

**Debug**: Check `logs/errors.log` and label JSON files to verify signal and parametrization success.

### Issue: Different fold assignments with same seed

**Cause**: Random state contamination from other RNG calls

**Solution**: Ensure `--seed` is specified consistently; the pipeline uses explicit seeding.

---

## References

- [BUILD.md](BUILD.md): C++ extension build instructions
- `spherepar/benchmark/dataset_generator.py`: Generation logic
- `spherepar/benchmark/splits.py`: Split builder
- `spherepar/flash_parametrization.py`: Spherical parametrization
