"""
spherepar.benchmark.dataset_generator
======================================

Dataset generator for mesh-deformation training data.

Pipeline
--------
1. Load every .obj from a directory with trimesh.
2. Repair holes when possible (fill boundary loops).
3. Sample deformation centers inside a cube with 110 % of the bounding-box
   volume, centered at the mesh center of mass.
4. Constrain displacement to ≤ 10 % of the distance from the sampled center
   to the center of mass.
5. Deform with the graphop CGAL backend.
6. Smooth and validate the result.
7. Compute mesh-to-mesh distance statistics (mean, std).
8. Generate a synthetic signal on the deformed surface.
9. Save only valid samples; log all failures.

Output layout (under a configurable root)::

    data/generated/
        meshes/     — deformed OBJ meshes
        signals/    — per-vertex signal arrays (.npy)
        labels/     — deformation parameters (.json)
        logs/       — per-run error log

Public API
----------
load_meshes_from_directory(directory)
repair_mesh_if_needed(mesh, name, error_log)
compute_sampling_cube_from_volume(mesh)
sample_handle_centers(cube_center, cube_half, n, rng)
compute_valid_displacement(center, com, max_ratio)
extract_roi_patch(vertices, center, radius)
deform_mesh_with_graphop(mesh_path, handle_id, target_pos, roi_ids)
smooth_and_validate_mesh(mesh)
compute_patch_to_mesh_stats(patch_points, mesh)
save_sample(root, name, mesh, stats, meta)
append_error_log(log_path, name, reason)
generate_dataset(input_dir, output_root, ...)
build_arg_parser()
main()
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.smoothing

from spherepar.benchmark.utils import extract_roi_patch
from spherepar.benchmark.surface import Surface, SurfaceFactory
from spherepar.benchmark.signals import sample_hm_major_axis
from spherepar.spherical_parametrization import compute_spherical_parametrization
from spherepar.benchmark.splits import build_task_splits, DEFAULT_TASKS
from spherepar.s2cnn.gendata import get_projection_grid, project_2d_on_sphere


# ---------------------------------------------------------------------------
# graphop import (lazy, so the module can be imported without the .so)
# ---------------------------------------------------------------------------
def _load_graphop_extension():
    """Load the compiled graphop extension from the repository root."""
    repo_root = Path(__file__).resolve().parents[2]
    for pattern in ("graphop*.so", "graphop*.pyd"):
        for ext_path in sorted(repo_root.glob(pattern)):
            sys.modules.pop("graphop", None)
            spec = importlib.util.spec_from_file_location("graphop", ext_path)
            if spec is None or spec.loader is None:
                continue
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "deform_surface"):
                    return module
            except Exception as e:
                # Skip incompatible .so files (e.g., Python version mismatch)
                continue

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import graphop as module  # type: ignore[import-not-found]

    if not hasattr(module, "deform_surface"):
        raise ImportError("Imported 'graphop' but did not find the compiled extension exports")
    return module


try:
    _graphop = _load_graphop_extension()
    _GRAPHOP_AVAILABLE = True
except ImportError:
    _graphop = None  # type: ignore[assignment]
    _GRAPHOP_AVAILABLE = False

# Cached MNIST images (loaded on demand)
_MNIST_IMAGES: Optional[np.ndarray] = None
_MNIST_LABELS: Optional[np.ndarray] = None


DEFORMATION_CASES: Dict[str, Dict[str, Any]] = {
    "case1_no": {
        # No deformation - only signals are generated on original mesh
        # Parametrization method is forced to None for this case
        "max_ratio": (0.0, 0.0),
        "num_candidates": (0, 0),
        "group_candidates": (1, 1),
        "alpha": (0.0, 0.0),
        "smooth_iterations": (0, 0),
        "ring_size": (0, 0),
    },
    "case2_small": {
        "max_ratio": (0.1, 0.3),
        "num_candidates": (5, 10),
        "group_candidates": (1, 5),
        "alpha": (0.2, 0.6),
        "smooth_iterations": (5, 15),
        "ring_size": (1, 3),
    },
    "case3_large": {
        "max_ratio": (0.2, 0.4),
        "num_candidates": (10, 15),
        "group_candidates": (1, 5),
        "alpha": (0.2, 0.6),
        "smooth_iterations": (5, 15),
        "ring_size": (2, 5),
    },
}


# ===========================================================================
# 1. Mesh loading
# ===========================================================================

def load_meshes_from_directory(
        directory: str,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Load mesh files from *directory* using trimesh.

    Returns a list of (filename_stem, trimesh.Trimesh) pairs for files
    that trimesh can load and that contain at least one face.

    Parameters
    ----------
    directory:
        Path to a directory containing mesh files.

    Returns
    -------
    list of (name, mesh) tuples
    """
    directory = Path(directory)
    results: List[Tuple[str, trimesh.Trimesh]] = []

    for mesh_path in sorted(directory.iterdir()):
        if not mesh_path.is_file():
            continue
        try:
            loaded = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                # merge all geometries into one
                meshes = list(loaded.geometry.values())
                if not meshes:
                    raise ValueError("Scene contains no geometry")
                mesh = trimesh.util.concatenate(meshes)
            else:
                mesh = loaded
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError("No triangular faces found")
            results.append((mesh_path.stem, mesh))
        except Exception as exc:  # noqa: BLE001
            # Caller can choose to log these; we just skip silently here
            _ = exc
            continue

    return results


# ===========================================================================
# 2. Mesh repair
# ===========================================================================

def repair_mesh_if_needed(
        mesh: trimesh.Trimesh,
        name: str,
        error_log: str,
) -> Optional[trimesh.Trimesh]:
    """Try to repair a mesh with holes.

    Parameters
    ----------
    mesh:
        Input mesh (modified in-place for repair attempts).
    name:
        Identifier used in error log messages.
    error_log:
        Path to the error log file.

    Returns
    -------
    trimesh.Trimesh or None
        The (possibly repaired) mesh, or None if repair failed.
    """
    if mesh.is_watertight:
        return mesh

    # Attempt 1: trimesh built-in fill_holes
    repaired = mesh.copy()
    trimesh.repair.fill_holes(repaired)
    if repaired.is_watertight:
        return repaired

    # Attempt 2: fix winding + fill again
    trimesh.repair.fix_winding(repaired)
    trimesh.repair.fill_holes(repaired)
    if repaired.is_watertight:
        return repaired

    # Could not repair
    append_error_log(
        error_log, name,
        "mesh has holes and automatic repair failed; skipping"
    )
    return None


# ===========================================================================
# 3. Sampling cube
# ===========================================================================

def compute_sampling_cube_from_volume(
        mesh: trimesh.Trimesh,
) -> Tuple[np.ndarray, float]:
    """Compute a sampling cube with 110 % of the mesh bounding-box volume.

    Steps
    -----
    1. Compute bounding-box volume V_bbox.
    2. V_cube = 1.1 * V_bbox.
    3. s = V_cube^(1/3).
    4. Center cube at mesh center of mass.

    Parameters
    ----------
    mesh:
        Input mesh.

    Returns
    -------
    center : np.ndarray, shape (3,)
        Center of mass of the mesh.
    half_side : float
        Half the side length of the sampling cube (s / 2).
    """
    bounds = mesh.bounds          # shape (2, 3): [min, max]
    bbox_extents = bounds[1] - bounds[0]
    v_bbox = float(np.prod(bbox_extents))
    v_cube = 1.1 * v_bbox
    s = v_cube ** (1.0 / 3.0)
    center = np.array(mesh.center_mass, dtype=float)
    return center, s / 2.0


# ===========================================================================
# 4. Handle-center sampling
# ===========================================================================

def sample_handle_centers(
        cube_center: np.ndarray,
        cube_half: float,
        n: int,
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample *n* candidate deformation centers inside the sampling cube.

    Parameters
    ----------
    cube_center:
        Center of mass (center of the cube), shape (3,).
    cube_half:
        Half side length of the cube.
    n:
        Number of candidates to sample.
    rng:
        NumPy random generator; created with default seed if None.

    Returns
    -------
    np.ndarray, shape (n, 3)
    """
    if rng is None:
        rng = np.random.default_rng()
    lo = cube_center - cube_half
    hi = cube_center + cube_half
    return rng.uniform(lo, hi, size=(n, 3))

# ===========================================================================
# 4a. Handle-vertices sampling
# ===========================================================================

def sample_handle_vertices(
        mesh: trimesh.Trimesh,
        n: int,
        rng: Optional[np.random.Generator] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """Sample *n* candidate vertex to act as handle in the mesh.

    Parameters
    ----------
    mesh:
        Input mesh.
    n:
        Number of candidates to sample.
    rng:
        NumPy random generator; created with default seed if None.

    Returns
    -------
    list of vertex positions, shape (n)
    """
    if rng is None:
        rng = np.random.default_rng()
    num_vertices = len(mesh.vertices)
    indices = rng.choice(num_vertices, size=n, replace=False)
    return mesh.vertices[indices], indices



# ===========================================================================
# 5. Displacement constraint
# ===========================================================================

def rotation_euler(
        vector: np.ndarray,
        euler_angles_deviation: List[float]):
    """ Rotation of vector with the euler angles deviation.

    Parameters
    ----------
    vector:
        Input vector to rotate, shape (3,).
    euler_angles_deviation:
        List of 3 Euler angle deviations (in radians) for rotation around x, y, z axes, respectively.

    Returns
    -------
    np.ndarray, shape (3,)
        Rotated vector.
    """
    euler_rotation_matrix = trimesh.transformations.euler_matrix(*euler_angles_deviation)[:3, :3]
    return euler_rotation_matrix @ vector



def compute_valid_displacement(
        center: np.ndarray,
        com: np.ndarray,
        normal: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        max_ratio: Optional[float] = 0.1,
) -> np.ndarray:
    """Sample a displacement vector with magnitude ≤ max_ratio * dist(center, com).

    Parameters
    ----------
    center:
        Deformation / handle center, shape (3,).
    com:
        Mesh center of mass, shape (3,).
    normal:
        Normal vector at the handle vertex, shape (3,).
    rng:
        NumPy random generator.
    max_ratio:
        Maximum displacement as a fraction of dist(center, com). Default 0.1.

    Returns
    -------
    np.ndarray, shape (3,)
        Displacement vector satisfying the magnitude constraint.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = float(np.linalg.norm(center - com))
    max_mag = max_ratio * d
    if max_mag < 1e-12:
        return np.zeros(3)
    # Sample direction uniformly on a solid cone pi/6 of S^2 then scale by a random fraction
    euler_angles_deviation = rng.uniform(-math.pi/6, math.pi/6, 3)
    direction = rotation_euler(normal, euler_angles_deviation)
    # the direction can be inverted
    flip_direction = rng.choice([1.0, -1.0])
    print('..... flip direction', flip_direction)
    direction = direction * flip_direction
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= norm
    magnitude = rng.uniform(0.0, max_mag)
    displacement = direction * magnitude
    return displacement

# ===========================================================================
# 7. Deformation via graphop
# ===========================================================================

def deform_mesh_with_graphop(
        mesh_path: str,
        handle_id: int | List[int],
        target_pos: np.ndarray,
        ring_size: float = 0.0,
        roi_ids: Optional[List[int]] = None,
        method: str = "sre_arap",
        alpha: float = 0.02,
        max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Deform a mesh using the graphop CGAL backend.

    Parameters
    ----------
    mesh_path:
        Path to the template OBJ file.
    handle_id:
        0-based vertex index of the deformation handle.
    target_pos:
        Target position for the handle, shape (3,).
    ring_size:
        Euclidean radius around each handle. All vertices within this radius
        are translated by the same displacement as the handle.
    roi_ids:
        Optional region-of-interest vertex indices (None = whole mesh).
    method, alpha, max_iter:
        Passed to graphop.deform_surface.

    Returns
    -------
    V_new : np.ndarray, shape (N, 3)
    F     : np.ndarray, shape (M, 3)
    meta  : dict
    """
    if not _GRAPHOP_AVAILABLE:
        raise ImportError(
            "graphop C++ extension is not available. Build it with CMake (see BUILD.md)."
        )
    handle_ids = [int(handle_id)] if isinstance(handle_id, (int, np.integer)) else [int(h) for h in handle_id]
    target = np.asarray(target_pos, dtype=np.float64).reshape(-1)
    # DEV-NOTE: graphop reads te mesh and deforms in memory. the output is vertices and faces
    V_new, F, meta = _graphop.deform_surface(
        mesh_path=mesh_path,
        handle_ids=handle_ids,
        target_positions=target,
        ring_size=ring_size,
        roi_ids=list(roi_ids) if roi_ids is not None else [],
        method=method,
        alpha=alpha,
        max_iter=max_iter,
    )
    return V_new, F, meta


# ===========================================================================
# 8. Smoothing and validation
# ===========================================================================

def smooth_and_validate_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        iterations: int = 3,
        drop_non_watertight: bool = False,
) -> Tuple[Optional[trimesh.Trimesh], Dict[str, Any]]:
    """Apply Humphrey smoothing and validate the resulting mesh.

    Parameters
    ----------
    vertices:
        Vertex positions, shape (N, 3).
    faces:
        Face connectivity, shape (M, 3).
    iterations:
        Number of smoothing passes.
    drop_non_watertight:
        Whether to reject meshes that are not watertight.

    Returns
    -------
    (trimesh.Trimesh or None, dict)
        Validated mesh, or None if invalid after smoothing, together with a
        quality report containing degeneracy and watertightness checks.
    """
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    trimesh.smoothing.filter_humphrey(mesh, iterations=iterations)
    areas = mesh.area_faces
    degenerate_face_count = int(np.count_nonzero(areas <= 0.0))
    quality = {"degenerate_face_count": degenerate_face_count}

    if degenerate_face_count > 0:
        quality["validation_error"] = "degenerate_faces"
        return None, quality

    # Remove duplicate / unreferenced geometry without hiding degeneracy.
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    quality["is_watertight"] = bool(mesh.is_watertight)

    if len(mesh.faces) == 0:
        quality["validation_error"] = "no_faces"
        return None, quality
    if drop_non_watertight and not quality["is_watertight"]:
        quality["validation_error"] = "non_watertight"
        return None, quality
    return mesh, quality


# ===========================================================================
# 9. Distance statistics
# ===========================================================================

def compute_patch_to_mesh_stats(
        patch_points: np.ndarray,
        mesh: trimesh.Trimesh,
) -> Dict[str, float]:
    """Compute point-to-surface distances from patch points to a mesh.

    Uses trimesh's efficient nearest-point query (BVH-accelerated).

    Parameters
    ----------
    patch_points:
        Query points, shape (K, 3).
    mesh:
        Target mesh.

    Returns
    -------
    dict with keys 'mean_distance' and 'std_distance'.
    """
    if len(patch_points) == 0:
        return {"mean_distance": 0.0, "std_distance": 0.0}
    _, distances, _ = trimesh.proximity.closest_point(mesh, patch_points)
    return {
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
    }


def _ensure_dataset_dirs(root: str) -> Dict[str, Path]:
    root_path = Path(root)
    dirs = {
        "meshes": root_path / "meshes",
        "signals": root_path / "signals",
        "spheres": root_path / "spheres",
        "labels": root_path / "labels",
        "folds": root_path / "folds",
        "logs": root_path / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _make_sample_id(sample_idx: int, template_name: Optional[str] = None) -> str:
    """Generate sample ID, optionally including template name."""
    if template_name:
        # Sanitize template name to use only alphanumeric + underscores
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in template_name)
        return f"{safe_name}_s{sample_idx:06d}"
    return f"sample_s{sample_idx:06d}"


_SIGNAL_SAMPLE_SUFFIX = re.compile(r"^(?P<sample_id>.+)_(?:iso|aniso)_\d+$")
_SAMPLE_INDEX_SUFFIX = re.compile(r"_s(?P<index>\d+)$")


def _signal_sample_id(signal_path: Path) -> str:
    """Return the owning sample ID for a saved signal file."""
    stem = signal_path.stem
    if stem.endswith("_mnist"):
        return stem[:-len("_mnist")]
    match = _SIGNAL_SAMPLE_SUFFIX.match(stem)
    return match.group("sample_id") if match else stem


def _list_completed_samples(root: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Find samples with all artifacts needed to safely resume generation.

    A primary label shares its stem with the mesh.  Per-signal and spherical
    metadata labels do not, so that convention cleanly excludes those helper
    labels.  A successful parametrization also requires its sphere OBJ.
    """
    root_path = Path(root)
    meshes = {path.stem for path in (root_path / "meshes").glob("*.obj")}
    spheres = {path.stem for path in (root_path / "spheres").glob("*.obj")}
    signals = {
        _signal_sample_id(path)
        for path in (root_path / "signals").glob("*.npy")
    }
    label_paths = {
        path.stem: path
        for path in (root_path / "labels").glob("*.json")
        if path.stem in meshes
    }

    completed: Dict[str, Dict[str, Any]] = {}
    for sample_id in meshes & signals & set(label_paths):
        try:
            with open(label_paths[sample_id], "r") as fh:
                label = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        parametrization = label.get("parametrization", {})
        if not isinstance(parametrization, dict):
            parametrization = {}
        param_method = parametrization.get("method")
        param_success = bool(parametrization.get("success"))
        param_error = parametrization.get("error")
        if param_success and sample_id not in spheres:
            continue
        # The primary label is written before parametrization starts.  A label
        # with a requested method but no result is therefore an interrupted
        # sample, not a completed one.  A recorded parametrization failure is
        # retained because that is how the existing generator records a saved
        # sample when parametrization itself fails.
        if param_method is not None and not param_success and param_error is None:
            continue

        completed[sample_id] = label

    counts = {
        "meshes": len(meshes),
        "spheres": len(spheres),
        "signals": len(signals),
        "labels": len(label_paths),
    }
    return completed, counts


def _label_generation_value(label: Dict[str, Any], key: str) -> Optional[str]:
    """Read a generation field from either supported label schema."""
    metadata = label.get("metadata")
    if isinstance(metadata, dict) and metadata.get(key) is not None:
        return str(metadata[key])
    value = label.get(key)
    return str(value) if value is not None else None


def _next_sample_index(sample_ids: List[str]) -> int:
    """Return an ID counter that cannot overwrite an existing completed sample."""
    indexes = []
    for sample_id in sample_ids:
        match = _SAMPLE_INDEX_SUFFIX.search(sample_id)
        if match:
            indexes.append(int(match.group("index")))
    return max(indexes) + 1 if indexes else 0


def _sample_deformation_config(case_name: str, rng: np.random.Generator) -> Dict[str, Any]:
    if case_name not in DEFORMATION_CASES:
        raise ValueError(f"Unknown deformation case {case_name!r}")
    cfg = DEFORMATION_CASES[case_name]
    return {
        "deformation_case": case_name,
        "max_ratio": float(rng.uniform(*cfg["max_ratio"])),
        "num_candidates": int(rng.integers(cfg["num_candidates"][0], cfg["num_candidates"][1] + 1)),
        "group_candidates": int(rng.choice(cfg["group_candidates"])),
        "alpha": float(rng.uniform(*cfg["alpha"])),
        "smooth_iterations": int(rng.integers(cfg["smooth_iterations"][0], cfg["smooth_iterations"][1] + 1)),
        "ring_size": int(rng.integers(cfg["ring_size"][0], cfg["ring_size"][1] + 1)),
    }


def _randomize_signal_parameters(
        sigma: float,
        amplitude: float,
        num_centers: int,
        sigma_ani: float = None,
        variation_percent: float = 20.0,
        rng: Optional[np.random.Generator] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Randomize signal parameters (sigma, amplitude, orientation) by ±variation_percent.
    
    Parameters
    ----------
    sigma : float
        Base sigma value.
    amplitude : float
        Base amplitude value.
    num_centers : int
        Number of centers (output list length).
    variation_percent : float
        Variation range as percentage (default 20%, means ±20%).
    rng : np.random.Generator, optional
        Random number generator. If None, uses default.
    
    Returns
    -------
    Tuple[List[float], List[float], List[float], List[float]]
        (sigma_list, amplitude_list, orientation_list, sigma_ani_list).
        Orientations are in radians, uniformly sampled from [0, π).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Convert percent to fraction
    variation_frac = variation_percent / 100.0
    
    sigma_list = []
    sigma_ani_list = []
    amplitude_list = []
    orientation_list = []
    # Devnote: anisotropic use only one center so just one pair.
    if sigma_ani is None:
        sigma_ani = sigma + 1 # TODO: this quick fix
    sigma_anisotropic_factor = rng.uniform(1.0 - variation_frac, 1.0 + variation_frac)

    sigma_ani_list.append(float(sigma_anisotropic_factor* sigma_ani))
    
    for _ in range(num_centers):
        # Sample variation factor: (1 - variation_frac) to (1 + variation_frac)
        sigma_factor = rng.uniform(1.0 - variation_frac, 1.0 + variation_frac)
        amplitude_factor = rng.uniform(1.0 - variation_frac, 1.0 + variation_frac)
        
        sigma_list.append(float(sigma * sigma_factor))
        amplitude_list.append(float(amplitude * amplitude_factor))
        
        # Orientation angle uniformly in [0, π) radians
        orientation = rng.uniform(0.0, np.pi)
        orientation_list.append(float(orientation))
    
    return sigma_list, amplitude_list, orientation_list, sigma_ani_list


def _effective_param_method(
    signal_type: Optional[str],
    case_name: str,
    param_method: Optional[str],
) -> Optional[str]:
    """Keep parametrization off for case1_no, otherwise use the requested method."""
    if case_name == "case1_no":
        return None
    return param_method


def _safe_unlink(path: Optional[str]) -> None:
    """Remove a temporary file path if present."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


# ===========================================================================
# 10. Save sample
# ===========================================================================

def save_sample_mesh(root: str, name: str, mesh: trimesh.Trimesh) -> str:
    """
    Save the object on the correct file structure
    Parameters
    ----------
    root
    name

    Returns
    -------

    """
    root_path = Path(root)
    meshes_dir = root_path / "meshes"
    labels_dir = root_path / "labels"
    signal_dir = root_path / "signals"

    for d in (meshes_dir, labels_dir, signal_dir):
        d.mkdir(parents=True, exist_ok=True)

    mesh_path = meshes_dir / f"{name}.obj"
    # Write OBJ
    mesh.export(str(mesh_path))
    return str(mesh_path)


def _to_relative_dataset_path(path_value: Optional[str], dataset_root: Path) -> Optional[str]:
    """Return a dataset-root-relative path when possible."""
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def _roi_patch_for_ratio(
        vertices: np.ndarray,
        center: np.ndarray,
        ratio: float,
) -> Tuple[float, np.ndarray]:
    if not (0.0 < ratio <= 1.0):
        raise ValueError("roi_vertex_ratio must be in the interval (0, 1]")
    if vertices.size == 0:
        raise ValueError("vertices must not be empty")
    n_vertices = vertices.shape[0]
    target_count = max(1, int(np.ceil(ratio * n_vertices)))
    distances = np.linalg.norm(vertices - center, axis=1)
    max_radius = float(distances.max())
    roi_radius = max_radius
    roi_indices = np.array([], dtype=np.intp)
    for r in np.linspace(0.0, max_radius, num=100):
        _, roi_indices = extract_roi_patch(vertices, center, radius=float(r))
        if len(roi_indices) >= target_count:
            roi_radius = float(r)
            break
    if len(roi_indices) < target_count:
        _, roi_indices = extract_roi_patch(vertices, center, radius=max_radius)
        roi_radius = max_radius
    return roi_radius, roi_indices


def _mesh_vertex_angles(vertices: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-vertex spherical angles (theta, phi) from mesh-centered directions."""
    dirs = vertices - center
    norms = np.linalg.norm(dirs, axis=1)
    norms[norms <= 0] = 1e-8
    dirs = dirs / norms[:, None]
    z = dirs[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))  # [0, pi]
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])  # (-pi, pi]
    phi[phi < 0] += 2.0 * np.pi  # [0, 2pi)
    return theta, phi


def _sample_dh_grid_at_angles(grid_values: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Bilinearly sample a DH grid at spherical angles (theta, phi)."""
    if grid_values.ndim != 2:
        raise ValueError(f"grid_values must be 2D, got shape {grid_values.shape}")
    n_beta, n_alpha = grid_values.shape
    beta_pos = np.clip(theta * n_beta / np.pi, 0.0, (n_beta - 1) - 1e-8)
    alpha_pos = (phi % (2.0 * np.pi)) * n_alpha / (2.0 * np.pi)

    b0 = np.floor(beta_pos).astype(np.intp)
    b1 = np.clip(b0 + 1, 0, n_beta - 1)
    a0 = np.floor(alpha_pos).astype(np.intp) % n_alpha
    a1 = (a0 + 1) % n_alpha

    wb = beta_pos - b0
    wa = alpha_pos - np.floor(alpha_pos)

    v00 = grid_values[b0, a0]
    v01 = grid_values[b0, a1]
    v10 = grid_values[b1, a0]
    v11 = grid_values[b1, a1]

    top = (1.0 - wa) * v00 + wa * v01
    bottom = (1.0 - wa) * v10 + wa * v11
    return ((1.0 - wb) * top + wb * bottom).astype(np.float32)



def save_sample_signal(
        root: str,
        name: str,
        mesh: trimesh.Trimesh,
        stats: Dict[str, float],
        meta: Dict[str, Any],
        template_id: str,
        deformation_case: str,
        random_seed: int,
        signal_factory: Optional[SurfaceFactory] = None,
        signal_type: Optional[str] = None,
        signal_sigma: float = 0.2,
        signal_sigma_ani: float = 0.8,
        signal_sigma_u: Optional[float] = None,
        signal_sigma_v: Optional[float] = None,
        signal_sigma_ratio: float = 0.5,
        signal_amplitude: float = 1.0,
        signal_num_centers: int = 1,
        signal_sigma_values: Optional[List[float]] = None,
        signal_amplitude_values: Optional[List[float]] = None,
        signal_orientation_values: Optional[List[float]] = None,
        signal_centers: Optional[List[int]] = None,
        mnist_index: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
        skip_mesh_save: bool = False,
) -> Dict[str, str]:
    """Save a valid dataset sample to disk.
    # TODO: this method is not isolated saving sample it also generating the signal. Method is overloaded! FIX

    Directory layout::

        <root>/
          meshes/<name>.obj
          signals/<name>.npy
          labels/<name>.json

    Parameters
    ----------
    root:
        Root output directory.
    name:
        Base filename (without extension).
    mesh:
        Deformed, validated trimesh.Trimesh.
    stats:
        Distance statistics dict.
    meta:
        Deformation / generation metadata dict.
    signal_factory:
        Factory used to compute synthetic signals on the saved mesh.
    signal_type:
        Synthetic signal family. ``None`` disables signal generation.
    signal_sigma:
        Width parameter used for the generated signal.
    signal_sigma_ani:
        Base anisotropic sigma (used when signal_sigma_u is not provided).
    signal_sigma_u:
        Explicit anisotropic sigma_u (overrides signal_sigma_ani).
    signal_sigma_v:
        Explicit anisotropic sigma_v (overrides ratio-based sigma_v).
    signal_sigma_ratio:
        Ratio used to compute sigma_v from sigma_u when signal_sigma_v is not provided.
    signal_amplitude:
        Amplitude used for the generated signal.
    signal_num_centers:
        Number of centers used to generate the synthetic signal.
    mnist_index:
        Optional fixed MNIST index (used when signal_type is 'mnist').
    rng:
        Random generator used to sample the signal center.

    Returns
    -------
    dict mapping the saved output kinds to their file paths.
    """
    root_path = Path(root)
    meshes_dir = root_path / "meshes"
    labels_dir = root_path / "labels"
    signal_dir = root_path / "signals"

    for d in (meshes_dir, labels_dir, signal_dir):
        d.mkdir(parents=True, exist_ok=True)

    mesh_path = meshes_dir / f"{name}.obj"
    labels_path = labels_dir / f"{name}.json"

    # Write OBJ mesh
    if not skip_mesh_save:
        mesh.export(str(mesh_path))

    # Write signal arrays
    # create a surface without signal
    # create use signal factory maybe as dummy objec just to use methos pdate signal as create signals
    signal_info: Dict[str, Any] = {}
    signal_path: Optional[str] = None
    signal_centers_coords: List[List[float]] = []
    signal_center_ids: List[int] = []
    sigmas: List[float] = []
    amplitudes: List[float] = []
    if signal_type is not None:
        if rng is None:
            rng = np.random.default_rng()

        # Special-case: MNIST projection does not depend on signal_factory
        if signal_type == "mnist":
            # Lazy-load MNIST images via scikit-learn's fetch_openml
            global _MNIST_IMAGES, _MNIST_LABELS
            try:
                if _MNIST_IMAGES is None:
                    from sklearn.datasets import fetch_openml

                    mn = fetch_openml("mnist_784", version=1, as_frame=False)
                    data = np.asarray(mn["data"], dtype=np.float32)
                    targets = np.asarray(mn.get("target"), dtype=np.int32) if mn.get("target") is not None else None
                    # reshape to (N, 28, 28)
                    data = data.reshape(-1, 28, 28)
                    # normalize to [0, 1]
                    if data.max() > 1.0:
                        data = data / 255.0
                    _MNIST_IMAGES = data
                    _MNIST_LABELS = targets
            except Exception as exc:  # pragma: no cover - fetching may fail in offline env
                raise RuntimeError(f"Failed to load MNIST dataset: {exc}")

            imgs = _MNIST_IMAGES
            labels = _MNIST_LABELS
            if imgs is None:
                raise RuntimeError("MNIST images not available")

            if mnist_index is None:
                idx = int(rng.integers(0, imgs.shape[0]))
            else:
                idx = int(mnist_index)
            if idx < 0 or idx >= imgs.shape[0]:
                raise ValueError(f"mnist_index {idx} is out of range (0..{imgs.shape[0] - 1})")
            img = imgs[idx]
            mnist_bandwidth = 30

            # Build S2CNN spherical sample (DH grid), then resample it on mesh points.
            grid = get_projection_grid(b=mnist_bandwidth)
            sample = project_2d_on_sphere(np.expand_dims(img, 0), grid)
            sample_grid = np.asarray(sample[0], dtype=np.float32) / 255.0

            verts = np.asarray(mesh.vertices, dtype=float)
            center = np.array(mesh.center_mass, dtype=float)
            theta, phi = _mesh_vertex_angles(verts, center)
            intensities = _sample_dh_grid_at_angles(sample_grid, theta, phi)
            signal_arr = (intensities * float(signal_amplitude)).astype(np.float32)

            # Save signal array
            save_path = signal_dir / f"{name}_mnist.npy"
            np.save(str(save_path), signal_arr)
            signal_path = str(save_path)

            signal_info = {
                "signal_type": "mnist",
                "mnist_index": int(idx),
                "mnist_label": int(labels[idx]) if labels is not None else None,
                "projection_method": "s2cnn_grid_to_mesh",
                "projection_grid_type": "Driscoll-Healy",
                "projection_bandwidth": int(mnist_bandwidth),
                "projection_interpolation": "bilinear",
                "signal_file": _to_relative_dataset_path(str(save_path), root_path),
                "num_centers": 0,
                "centers": [],
                "center_vertex_ids": [],
                "sigmas": [],
                "amplitudes": [],
            }
            meta["signal"] = signal_info
            # Store MNIST label for later use in tasks section
            meta["mnist_digit_label"] = int(labels[idx]) if labels is not None else None
            # Adjust local variables used later when writing labels/tasks
            signal_num_centers = 0
            signal_centers_coords = []
            signal_center_ids = []
            sigmas = []
            amplitudes = []
        else:
            if signal_factory is None:
                raise ValueError("signal_factory is required when signal_type is not None and not 'mnist'")
            if signal_num_centers <= 0:
                raise ValueError("signal_num_centers must be positive")

            # Allow caller to provide explicit vertex centers (used when matching signals)
            if signal_centers is not None:
                signal_centers = [int(idx) for idx in signal_centers]
            else:
                signal_center_idx = rng.integers(0, len(mesh.vertices), size=signal_num_centers)
                signal_centers = [int(idx) for idx in np.atleast_1d(signal_center_idx)]

            signal_centers_coords = [np.asarray(mesh.vertices[idx], dtype=float).tolist() for idx in signal_centers]
            signal_center_ids = signal_centers
            if signal_sigma_values is None:
                sigmas = [float(signal_sigma)] * signal_num_centers
            else:
                sigmas = [float(v) for v in signal_sigma_values]
            if signal_amplitude_values is None:
                amplitudes = [float(signal_amplitude)] * signal_num_centers
            else:
                amplitudes = [float(v) for v in signal_amplitude_values]
            if signal_type == "isotropic":
                signal_params: Dict[str, Any] = {
                    "centers": signal_centers,
                    "sigmas": sigmas,  # Now pass the list of sigmas per center
                    "amplitudes": amplitudes,  # Pass per-center amplitudes
                }
            else:
                # For anisotropic: use sigma_u and sigma_v with CLI-controlled precedence.
                # The physical axis is generated from HM tangent frame + sampled perturbation.
                sigma_u = sigmas[0] if sigmas else float(signal_sigma_ani)
                if signal_sigma_u is not None:
                    sigma_u = float(signal_sigma_u)
                sampled_delta = signal_orientation_values[0] if signal_orientation_values else None
                if signal_sigma_v is not None:
                    sigma_v = float(signal_sigma_v)
                else:
                    sigma_v = max(sigma_u * float(signal_sigma_ratio), 1e-6)

                gauge_eps = 1e-8
                gauge_min_projected_norm = 0.05
                center_idx = int(signal_centers[0])
                hm_info: Optional[Dict[str, Any]] = None
                last_error: Optional[Exception] = None
                max_attempts = min(max(len(mesh.vertices), 1), 256)
                for _ in range(max_attempts):
                    center_candidate = np.asarray(mesh.vertices[center_idx], dtype=float)
                    try:
                        hm_info = sample_hm_major_axis(
                            center=center_candidate,
                            rng=rng,
                            delta=float(sampled_delta) if sampled_delta is not None else None,
                            eps=gauge_eps,
                            min_gauge_projection_norm=gauge_min_projected_norm,
                        )
                        break
                    except ValueError as exc:
                        last_error = exc
                        center_idx = int(rng.integers(0, len(mesh.vertices)))
                        sampled_delta = None
                if hm_info is None:
                    raise RuntimeError(
                        f"Failed to sample anisotropic HM frame/axis after {max_attempts} attempts: {last_error}"
                    )

                # Keep center metadata aligned with the accepted gauge-valid center.
                signal_centers = [center_idx]
                signal_center_ids = [center_idx]
                signal_centers_coords = [np.asarray(mesh.vertices[center_idx], dtype=float).tolist()]

                signal_params = {
                    "center": center_idx,
                    "sigma_u": sigma_u,
                    "sigma_v": sigma_v,
                    "amplitude": amplitudes[0],
                    "major_axis": np.asarray(hm_info["major_axis"], dtype=float).tolist(),
                    "delta": float(hm_info["delta"]),
                    "orientation_angle": float(hm_info["delta"]),
                    "phi": float(hm_info["phi"]),
                    "target_doubled_angle": np.asarray(hm_info["target"], dtype=float).tolist(),
                    "hm_e1": np.asarray(hm_info["e1"], dtype=float).tolist(),
                    "hm_e2": np.asarray(hm_info["e2"], dtype=float).tolist(),
                    "gauge_e1": np.asarray(hm_info["g1"], dtype=float).tolist(),
                    "gauge_e2": np.asarray(hm_info["g2"], dtype=float).tolist(),
                    "gauge": [0.0, 0.0, 1.0],
                    "gauge_eps": gauge_eps,
                    "gauge_min_projected_norm": gauge_min_projected_norm,
                }

            surface = Surface(
                vertices=np.asarray(mesh.vertices, dtype=np.float64),
                faces=np.asarray(mesh.faces, dtype=np.int32),
                deform_meta=_json_safe(meta.get("deform_meta", {})),
                root=root,
                fname=name,
            )
            surface = signal_factory.compute_signal(surface, signal_type, signal_params)
            signal_paths = surface.save_only_signal()
            signal_path = signal_paths["signal"]
            signal_info = {
                "signal_type": signal_type,
                "signal_params": _json_safe(signal_params),
                "signal_meta": _json_safe(surface.signal_meta),
                "signal_file": _to_relative_dataset_path(signal_paths["signal"], root_path),
                "signal_label_file": _to_relative_dataset_path(signal_paths["signal_label"], root_path),
                "num_centers": int(signal_num_centers),
                "centers": signal_centers_coords,
                "center_vertex_ids": signal_center_ids,
                "sigmas": sigmas,
                "amplitudes": amplitudes,
            }
            meta["signal"] = signal_info

    # Write JSON metadata
    mesh_path_rel = str(mesh_path.relative_to(root_path))
    label_path_rel = str(labels_path.relative_to(root_path))
    signal_path_rel = _to_relative_dataset_path(signal_path, root_path)
    sphere_path_rel = str((root_path / "spheres" / f"{name}.obj").relative_to(root_path))
    signal_payload: Dict[str, Any] = {
        "num_centers": int(signal_num_centers) if signal_type is not None else 0,
        "centers": signal_centers_coords,
        "center_vertex_ids": signal_center_ids,
        "sigmas": sigmas,
        "amplitudes": amplitudes,
        "family": signal_type,
    }
    signal_payload.update(_json_safe(meta.get("signal", {})))
    signal_payload["family"] = signal_type

    label = {
        "sample_id": name,
        "name": name,
        "template_id": template_id,
        "deformation_case": deformation_case,
        "mesh_file": mesh_path_rel,
        "mesh_path": mesh_path_rel,
        "signal_file": signal_path_rel,
        "signal_path": signal_path_rel,
        "sphere_path": sphere_path_rel,
        "label_path": label_path_rel,
        "n_vertices": int(mesh.vertices.shape[0]),
        "n_faces": int(mesh.faces.shape[0]),
        "distance_stats": stats,
        "signal": signal_payload,
        "deformation": _json_safe(meta.get("deformation", meta)),
        "parametrization": {
            "method": meta.get("parametrization_method"),
            "success": False,
        },
        "random_seed": int(random_seed),
        "warnings": _json_safe(meta.get("warnings", [])),
    }
    # task-facing metadata used by split builder
    num_centers = int(label["signal"]["num_centers"])
    label["tasks"] = {
        "number_of_centers": {"valid": num_centers in [1, 2, 3, 4, 5], "label": num_centers},
        "center_regression": {"valid": num_centers == 1, "label": label["signal"]["centers"][0] if num_centers == 1 else None},
        "sigma_regression": {"valid": num_centers == 1, "label": label["signal"]["sigmas"][0] if num_centers == 1 else None},
        "amplitude_regression": {"valid": num_centers == 1, "label": label["signal"]["amplitudes"][0] if num_centers == 1 else None},
    }
    
    # Add MNIST classification task if present
    if "mnist_digit_label" in meta:
        mnist_label = meta["mnist_digit_label"]
        label["tasks"]["mnist_cls"] = {
            "valid": mnist_label is not None and 0 <= mnist_label < 10,
            "label": mnist_label
        }
    with open(labels_path, "w") as fh:
        json.dump(label, fh, indent=2)

    result = {
        "mesh": str(mesh_path),
        "labels": str(labels_path),
    }
    if signal_path is not None:
        result["signal"] = signal_path
    if signal_type and "signal_params" in meta.get("signal", {}):
        result["signal_params"] = meta["signal"]["signal_params"]
    return result


# ===========================================================================
# 11. Spherical parametrization save
# ===========================================================================

def save_spherical_parametrization(
        root: str,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        method: str = "flash",
        cem_eps: float = 1e-6,
        cem_max_iters: int = 100,
        cem_verbose: bool = False,
) -> Dict[str, str]:
    """Compute and save spherical parametrization mesh and metadata label."""
    root_path = Path(root)
    spheres_dir = root_path / "spheres"
    labels_dir = root_path / "labels"
    spheres_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sphere_vertices, sphere_meta = compute_spherical_parametrization(
        vertices=vertices,
        faces=faces,
        method=method,
        cem_eps=cem_eps,
        cem_max_iters=cem_max_iters,
        cem_verbose=cem_verbose,
        verify=True,
    )

    sphere_mesh = trimesh.Trimesh(
        vertices=sphere_vertices,
        faces=np.asarray(faces, dtype=np.int32),
        process=False,
    )
    sphere_path = spheres_dir / f"{name}.obj"
    sphere_mesh.export(str(sphere_path))

    sphere_label_path = labels_dir / f"{name}_spherical.json"
    sphere_label = {
        "name": name,
        "method": method,
        "sphere_file": str(sphere_path.relative_to(root_path)),
        "metadata": _json_safe(sphere_meta),
    }
    with open(sphere_label_path, "w") as fh:
        json.dump(sphere_label, fh, indent=2)

    return {
        "sphere": str(sphere_path.relative_to(root_path)),
        "spherical_label": str(sphere_label_path.relative_to(root_path)),
    }


def _update_sample_label(
        label_path: str,
        updates: Dict[str, Any],
) -> None:
    """ Insert data in the labels JSON files.

    Parameters
    ----------
    label_path
    updates

    Returns
    -------

    """
    with open(label_path, "r") as fh:
        label = json.load(fh)
    # shallow recursive-like merge for top-level keys and nested dicts
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(label.get(key), dict):
            label[key].update(value)
        else:
            label[key] = value
    with open(label_path, "w") as fh:
        json.dump(label, fh, indent=2)


def validate_saved_sample(
        label_path: str,
) -> Tuple[bool, List[str]]:
    """Validate artifact integrity for a generated sample."""
    issues: List[str] = []
    p = Path(label_path)
    if not p.exists():
        return False, [f"missing label file: {label_path}"]
    with open(p, "r") as fh:
        label = json.load(fh)

    # Check for new dual-signal schema or old schema
    has_new_schema = "signal_files" in label and "signals" in label
    
    # Required fields vary by schema version
    if has_new_schema:
        required = [
            "sample_id",
            "mesh",
            "signal_files",
            "signals",
        ]
    else:
        required = [
            "sample_id", "template_id", "deformation_case",
            "mesh_path", "label_path",
            "deformation", "parametrization", "random_seed",
        ]
    
    for key in required:
        if key not in label:
            issues.append(f"missing field in label: {key}")

    # Validate mesh
    mesh_path = None
    if has_new_schema:
        # New schema: mesh path is in paths.mesh
        mesh_path = label.get("paths", {}).get("mesh")
    else:
        # Old schema: directly in mesh_path
        mesh_path = label.get("mesh_path")
    
    if mesh_path:
        mesh_path = Path(mesh_path)
        # Handle relative paths (resolve relative to dataset root)
        if not mesh_path.is_absolute():
            label_file = Path(label_path)
            dataset_root = label_file.parent.parent  # labels/ -> dataset_root
            mesh_path = dataset_root / mesh_path
        if not mesh_path.exists():
            issues.append(f"mesh file missing: {mesh_path}")
    
    # Validate signals
    signal_paths = []
    label_file = Path(label_path)
    dataset_root = label_file.parent.parent  # labels/ -> dataset_root
    
    if has_new_schema:
        # Convert relative paths to absolute paths
        for rel_path in label.get("signal_files", {}).values():
            if rel_path:
                path = Path(rel_path)
                if path.is_absolute():
                    signal_paths.append(path)  # Already absolute
                else:
                    signal_paths.append(dataset_root / path)  # Make absolute relative to dataset_root
    else:
        # Old schema: signal_path is relative, resolve it
        signal_path = label.get("signal_path", "")
        if signal_path:
            path = Path(signal_path)
            if path.is_absolute():
                signal_paths.append(path)  # Already absolute
            else:
                signal_paths.append(dataset_root / path)  # Make absolute relative to dataset_root
    
    for signal_path in signal_paths:
        if not signal_path.exists():
            issues.append(f"signal file missing: {signal_path}")
        else:
            try:
                signal = np.load(signal_path)
                if np.isnan(signal).any():
                    issues.append(f"signal contains NaN: {signal_path.name}")
                if mesh_path and mesh_path.exists():
                    mesh = trimesh.load(str(mesh_path), force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        meshes = list(mesh.geometry.values())
                        mesh = trimesh.util.concatenate(meshes) if meshes else None
                    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
                        issues.append("mesh file could not be loaded for validation")
                    elif len(signal) != len(mesh.vertices):
                        issues.append(f"signal length ({len(signal)}) != n_vertices ({len(mesh.vertices)}) in {signal_path.name}")
            except Exception as e:
                issues.append(f"could not validate signal {signal_path.name}: {e}")

    # Validate parametrization if present
    sphere_path = label.get("sphere_path") or (label.get("paths", {}).get("sphere") if has_new_schema else None)
    if sphere_path:
        sphere_path = Path(sphere_path)
        # Handle relative paths (resolve relative to dataset root)
        if not sphere_path.is_absolute():
            label_file = Path(label_path)
            dataset_root = label_file.parent.parent  # labels/ -> dataset_root
            sphere_path = dataset_root / sphere_path
        if bool(label.get("parametrization", {}).get("success")) and not sphere_path.exists():
            issues.append(f"sphere file missing despite parametrization success: {sphere_path}")

    return len(issues) == 0, issues


# ===========================================================================
# 12. Error log
# ===========================================================================

def append_error_log(
        log_path: str,
        name: str,
        reason: str,
        template_id: Optional[str] = None,
        deformation_case: Optional[str] = None,
        traceback_text: Optional[str] = None,
) -> None:
    """Append a failure entry to the error log file.

    Parameters
    ----------
    log_path:
        Path to the log file (created if absent).
    name:
        Identifier of the failed mesh / sample.
    reason:
        Human-readable failure reason.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(log_path, "a") as fh:
        parts = [timestamp, name]
        if template_id is not None:
            parts.append(f"template={template_id}")
        if deformation_case is not None:
            parts.append(f"case={deformation_case}")
        parts.append(reason)
        fh.write("  ".join(parts) + "\n")
        if traceback_text:
            fh.write(traceback_text.rstrip() + "\n")


# ===========================================================================
# 13. Top-level generator
# ===========================================================================

def generate_dataset(
        input_dir: str,
        output_root: str = "data/generated",
        n_samples_per_mesh: int = 25,
        patch_radius_ratio: float = 0.15,
        smoothing_iterations: int = 3,
        group_candidates: int = 5,
        roi_vertex_ratio: float = 0.3,
        max_ratio: float = 0.8,
        ring_size: Optional[int] = None,
        deform_method: str = "sre_arap",
        alpha: float = 0.02,
        max_iter: int = 50,
        seed: int = 42,
        repair_holes: bool = True,
        drop_non_watertight: bool = False,
        signal_type: Optional[str] = "isotropic",
        signal_sigma: float = 0.2,
        signal_sigma_ani: float = 0.8,
        signal_sigma_u: Optional[float] = None,
        signal_sigma_v: Optional[float] = None,
        signal_sigma_ratio: float = 0.5,
        signal_amplitude: float = 1.0,
        signal_num_centers: int = 1,
        signal_centers_options: Optional[List[int]] = None,
        signal_sigma_variation_percent: float = 20.0,
        signal_amplitude_variation_percent: float = 20.0,
        param_method: Optional[str] = None,
        cem_eps: float = 1e-6,
        cem_max_iters: int = 100,
        cem_verbose: bool = False,
        deformation_cases: Optional[List[str]] = None,
        create_splits: bool = False,
        split_tasks: Optional[List[str]] = None,
        num_folds: int = 5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int = 0,
        group_by_template: bool = True,
        offset_sample_counter: int = 0,
        resume: bool = True,
        mnist_percentage: float = 100.0,
        mnist_total_count: Optional[int] = None,

) -> int:
    """Run the full dataset generation pipeline.

    Parameters
    ----------
    input_dir:
        Directory containing input .obj meshes.
    output_root:
        Root directory for generated output.
    n_samples_per_mesh:
        Number of deformation samples to attempt per input mesh.
    signal_type:
        Synthetic signal family attached after deformation. ``None`` disables
        signal generation. The default is ``"isotropic"``.
    mnist_percentage:
        Percentage of MNIST dataset to use (0.1-100.0). Only applies when signal_type is 'mnist'.
    mnist_total_count:
        Explicit total count of MNIST samples. Overrides mnist_percentage if provided.
    Returns
    -----
    total_saved: int
        Number of samples successfully saved.
    
    """
    if not _GRAPHOP_AVAILABLE:
        raise ImportError(
            "graphop C++ extension is required. Build it with CMake (see BUILD.md)."
        )
    if not (0.0 < roi_vertex_ratio <= 1.0):
        raise ValueError("roi_vertex_ratio must be in the interval (0, 1]")
    if ring_size is not None and ring_size < 0.0:
        raise ValueError("ring_size must be non-negative")
    if signal_sigma <= 0.0:
        raise ValueError("signal_sigma must be positive")
    if signal_num_centers <= 0:
        raise ValueError("signal_num_centers must be positive")
    if signal_type not in (None, "isotropic", "anisotropic", "mnist"):
        raise ValueError("signal_type must be one of None, 'isotropic', 'anisotropic', or 'mnist'")
    if signal_type is None:
        raise ValueError("signal_type cannot be None for this dataset pipeline; each sample must include a signal.")
    if param_method not in (None, "flash", "cem"):
        raise ValueError("param_method must be one of None, 'flash', or 'cem'")
    
    # MNIST-specific validation
    if signal_type == "mnist":
        if not (0.1 <= mnist_percentage <= 100.0):
            raise ValueError(f"mnist_percentage must be in range [0.1, 100.0], got {mnist_percentage}")
        if mnist_total_count is not None and mnist_total_count <= 0:
            raise ValueError("mnist_total_count must be positive")
    
    if deformation_cases is None:
        deformation_cases = ["case2_small", "case3_large"]
    for case_name in deformation_cases:
        if case_name not in DEFORMATION_CASES:
            raise ValueError(f"Unknown deformation case: {case_name}")

    output_root = str(output_root)
    _ensure_dataset_dirs(output_root)
    log_path = str(Path(output_root) / "logs" / "errors.log")
    rng = np.random.default_rng(seed)
    # --- Load meshes --------------------------------------------------------
    print(f"Loading meshes from: {input_dir}")
    mesh_pairs = load_meshes_from_directory(input_dir)
    if not mesh_pairs:
        print("No .obj files found or all failed to load.")
        return 0
    print(f"  Loaded {len(mesh_pairs)} mesh(es)")
    
    # MNIST requires exactly one template mesh
    if signal_type == "mnist" and len(mesh_pairs) != 1:
        raise ValueError(
            f"MNIST dataset generation requires exactly one template mesh, "
            f"but {len(mesh_pairs)} meshes were found. Please provide a single mesh."
        )

    total_saved = 0
    total_failed = 0

    if signal_centers_options is None:
        signal_centers_options = list(range(1, signal_num_centers + 1))
    signal_centers_options = [int(v) for v in signal_centers_options if int(v) > 0]
    if not signal_centers_options:
        raise ValueError("signal_centers_options must contain at least one positive integer")
    
    # MNIST sample count override and configuration
    if signal_type == "mnist":
        # Calculate total MNIST samples from percentage or explicit count
        total_mnist_count = 70000  # Full MNIST dataset size
        if mnist_total_count is not None:
            if mnist_total_count > total_mnist_count:
                raise ValueError(
                    f"mnist_total_count must be <= {total_mnist_count}, got {mnist_total_count}"
                )
            samples_per_center_and_case = mnist_total_count
        else:
            samples_per_center_and_case = int((mnist_percentage / 100.0) * total_mnist_count)
        signal_centers_options = [1]  # Force single center for MNIST
        # deformation_cases = ["case1_no"]  # Force no deformation for MNIST
    else:
        # Fix sample count logic: n_samples_per_mesh is per center option
        # Total samples = n_samples_per_mesh * len(signal_centers_options) * len(deformation_cases)
        samples_per_center_and_case = n_samples_per_mesh
    
    total_samples_plan = samples_per_center_and_case * len(signal_centers_options) * len(deformation_cases)
    total_samples_requested = total_samples_plan * len(mesh_pairs)

    if resume:
        completed_samples, artifact_counts = _list_completed_samples(output_root)
    else:
        completed_samples = {}
        artifact_counts = {"meshes": 0, "spheres": 0, "signals": 0, "labels": 0}
    current_template_ids = {mesh_name for mesh_name, _ in mesh_pairs}
    requested_cases = set(deformation_cases)
    completed_for_request = [
        sample_id
        for sample_id, label in completed_samples.items()
        if _label_generation_value(label, "template_id") in current_template_ids
        and _label_generation_value(label, "deformation_case") in requested_cases
    ]
    # Skip the already-completed slots in this invocation.  The counter itself
    # is global to the output root so separately generated deformation cases
    # never overwrite each other's sample IDs.
    resume_slots = min(len(completed_for_request), total_samples_requested)
    requested_offset = int(offset_sample_counter or 0)
    sample_idx = max(requested_offset, _next_sample_index(list(completed_samples)))

    if resume:
        print(
            "Resume scan: "
            f"meshes={artifact_counts['meshes']}, "
            f"spheres={artifact_counts['spheres']}, "
            f"signals={artifact_counts['signals']}, "
            f"labels={artifact_counts['labels']}, "
            f"complete={len(completed_samples)}"
        )
    else:
        print("Resume disabled: generating the full requested plan from the configured offset.")
    print(
        f"Generation plan: {total_samples_requested} sample(s) "
        f"({total_samples_plan} per mesh); "
        f"resuming after {resume_slots} matching sample(s) at index {sample_idx}."
    )
    mnist_index_iter: Optional[Iterator[int]] = None
    if signal_type == "mnist":
        if total_samples_plan > total_mnist_count:
            raise ValueError(
                f"Requested {total_samples_plan} MNIST samples, but dataset has {total_mnist_count} images"
            )
        mnist_index_iter = iter(range(resume_slots, total_samples_plan))

    planned_slot = 0
    for mesh_name, template_mesh in mesh_pairs:
        print(f"\nProcessing: {mesh_name}")

        # --- Repair ---------------------------------------------------------
        if repair_holes:
            mesh = repair_mesh_if_needed(template_mesh, mesh_name, log_path)
            if mesh is None:
                total_failed += 1
                continue
        else:
            mesh = template_mesh

        # --- Geometry info --------------------------------------------------
        com, half_side = compute_sampling_cube_from_volume(mesh)
        bbox_diag = float(np.linalg.norm(
            mesh.bounds[1] - mesh.bounds[0]
        ))
        patch_radius = patch_radius_ratio * bbox_diag



        number_vertices = len(mesh.vertices)
        for case_name in deformation_cases:
            for signal_num_centers_choice in signal_centers_options:

                for _ in range(samples_per_center_and_case):
                    if planned_slot < resume_slots:
                        planned_slot += 1
                        continue
                    planned_slot += 1
                    tmp_path: Optional[str] = None
                    if case_name == "case1_no":
                        # No deformation case: use original mesh and only generate signals
                        deformed = mesh.copy()
                        quality = {"is_watertight": True}  # Original mesh is always valid
                        deformation_failed = False
                        handle_ids = []
                        handle_positions = []
                        displacements = []
                        target_positions = []
                        deform_meta = {}
                        sample_seed = seed
                        sample_rng = rng
                        sampled_cfg = _sample_deformation_config(case_name, rng)
                        
                    else:
                        # Deformation cases (case2_small, case3_large)
                        sample_seed = seed
                        sample_rng = rng
                        sampled_cfg = _sample_deformation_config(case_name, sample_rng)

                        # the number of candidates means I will deform as many times the candidates says.
                        num_candidates = min(int(sampled_cfg["num_candidates"]), number_vertices)
                        candidate_ids = sample_rng.choice(number_vertices, size=num_candidates, replace=False)
                        handle_ids = [int(v) for v in np.atleast_1d(candidate_ids)]
                        
                        # Apply group_candidates logic
                        # If group_candidates is has a number>1 then we take additional candidate to make more deformations at the same times
                        # it is basically deform on more directions
                        n_group_candidates = sampled_cfg["group_candidates"]
                        pool_grouped_candidates = {}
                        if n_group_candidates > 1:
                            # add to the pool random additional candidates to group with the main one
                            for handle_id in handle_ids:
                                pool_grouped_candidates[handle_id] = [handle_id] # add itself
                                # generate new candites
                                pool_grouped_candidates[handle_id].extend(
                                    int(v) for v in sample_rng.choice(
                                        [idx for idx in range(number_vertices) if idx != handle_id],
                                        size=n_group_candidates - 1,
                                        replace=False,
                                    )
                                )
                                
                        else:
                            pool_grouped_candidates = {handle_id: [handle_id] for handle_id in handle_ids}

                        # for each candidate perform deformation with its group of candidates.

                        # We need a file on disk for the graphop backend
                        with tempfile.NamedTemporaryFile(
                                suffix=".obj", delete=False, mode="w"
                        ) as tmp:
                            tmp_path = tmp.name
                        try:
                            mesh.export(tmp_path)
                        except Exception as exc:  # noqa: BLE001
                            append_error_log(log_path, mesh_name, f"could not export temp OBJ: {exc}")
                            os.unlink(tmp_path)
                            total_failed += 1
                            continue


                        deformed = mesh.copy() # Mesh is nice and smooth here, it will be deformed by each handle
                        quality = None  # Track mesh quality across deformations
                        deformation_failed = False  # Track if any deformation failed

                        for handle_id in handle_ids:
                            # for each cadidate builds the rois and displacements
                            roi_union: set[int] = set()
                            target_positions: List[np.ndarray] = []
                            displacements: List[np.ndarray] = []
                            handle_positions: List[np.ndarray] = []

                            handle_ids_in_group = pool_grouped_candidates[handle_id]

                            for handle_id in handle_ids_in_group:
                                handle_center = deformed.vertices[handle_id]
                                normal = deformed.vertex_normals[handle_id]
                                _, nearest_ids = extract_roi_patch(deformed.vertices, handle_center, 0.0)
                                for r in np.linspace(0.0, bbox_diag, num=100):
                                    _, nearest_ids = extract_roi_patch(deformed.vertices, handle_center, radius=r)
                                    if len(nearest_ids) > roi_vertex_ratio * number_vertices:
                                        print(f"Found {len(nearest_ids)} at handle_id={handle_id} which is {100 * len(nearest_ids) / number_vertices} % of vertices ")
                                        print(f"Radius used: {r:.4f}")
                                        break

                                roi_union.update(int(idx) for idx in nearest_ids)
                                handle_pos = deformed.vertices[handle_id].copy()
                                displacement = compute_valid_displacement(
                                    handle_pos,
                                    com,
                                    normal,
                                    sample_rng,
                                    max_ratio=sampled_cfg["max_ratio"],
                                )
                                target_positions.append(handle_pos + displacement)
                                displacements.append(displacement)
                                handle_positions.append(handle_pos)

                            if not roi_union:
                                append_error_log(
                                    log_path,
                                    sample_name,
                                    "empty ROI for sampled deformation",
                                    template_id=mesh_name,
                                    deformation_case=case_name,
                                )
                                total_failed += 1
                                deformation_failed = True
                                break  # Stop deforming if ROI is empty
                            # now deforms for this candidate
                            try:
                                # Use CLI ring_size if provided, otherwise use the sampled value from the case
                                effective_ring_size = ring_size if ring_size is not None else sampled_cfg["ring_size"]
                                V_new, F_new, deform_meta = deform_mesh_with_graphop(
                                    mesh_path=tmp_path,
                                    handle_id=handle_ids_in_group,
                                    target_pos=np.asarray(target_positions, dtype=np.float64),
                                    ring_size=effective_ring_size,
                                    roi_ids=sorted(roi_union),
                                    method=deform_method,
                                    alpha=float(sampled_cfg["alpha"]),
                                    max_iter=max_iter,
                                )
                            except Exception as exc:  # noqa: BLE001
                                append_error_log(
                                    log_path,
                                    sample_name,
                                    f"deformation failed: {exc}",
                                    template_id=mesh_name,
                                    deformation_case=case_name,
                                    traceback_text=traceback.format_exc(),
                                )
                                total_failed += 1
                                deformation_failed = True
                                break  # Stop deforming if deformation fails

                            # Smooth + validate geometry
                            # Gets a new mesh,
                            deformed, quality = smooth_and_validate_mesh(
                                V_new,
                                F_new,
                                int(sampled_cfg["smooth_iterations"]),
                                drop_non_watertight=drop_non_watertight,
                            )

                            if deformed is None:
                                append_error_log(
                                    log_path,
                                    sample_name,
                                    "mesh invalid after deformation/smoothing "
                                    f"(watertight={quality.get('is_watertight')}, "
                                    f"degenerate_faces={quality.get('degenerate_face_count')}, "
                                    f"reason={quality.get('validation_error', 'unknown')})",
                                    template_id=mesh_name,
                                    deformation_case=case_name,
                                )
                                total_failed += 1
                                deformation_failed = True
                                break  # Stop deforming if mesh is invalid

                            # save this mesh to continue deforming
                            deformed.export(str(tmp_path))

                        # Skip sample if deformation failed
                        if deformation_failed or quality is None:
                            os.unlink(tmp_path)
                            continue

                    sample_name = _make_sample_id(sample_idx, template_name=mesh_name)
                    sample_idx += 1
                    mnist_index: Optional[int] = None
                    if signal_type == "mnist":
                        if mnist_index_iter is None:
                            raise RuntimeError("MNIST indices were not initialized")
                        try:
                            mnist_index = next(mnist_index_iter)
                        except StopIteration as exc:
                            raise RuntimeError("MNIST index sequence exhausted") from exc
                    
                    # For case1_no, create a temporary file with the original mesh (wasn't created during deformation)
                    if case_name == "case1_no":
                        with tempfile.NamedTemporaryFile(
                                suffix=".obj", delete=False, mode="w"
                        ) as tmp:
                            tmp_path = tmp.name
                        try:
                            mesh.export(tmp_path)
                        except Exception as exc:  # noqa: BLE001
                            append_error_log(log_path, mesh_name, f"could not export temp OBJ for case1_no: {exc}")
                            os.unlink(tmp_path)
                            total_failed += 1
                            continue

                    signal_factory = None
                    if signal_type is not None:
                        signal_factory = SurfaceFactory(root=output_root, template_mesh_path=tmp_path)
                    stats = compute_patch_to_mesh_stats(mesh.vertices, deformed)
                    sample_num_centers = signal_num_centers_choice

                    # Randomize signal parameters
                    sigma_list, amplitude_list, orientation_list, sigma_ani_list = _randomize_signal_parameters(
                        sigma=signal_sigma,
                        amplitude=signal_amplitude,
                        num_centers=sample_num_centers,
                        sigma_ani=signal_sigma_ani,
                        variation_percent=signal_sigma_variation_percent,
                        rng=sample_rng,
                    )

                    warnings: List[str] = []
                    if signal_type == "anisotropic" and sample_num_centers != 1:
                        warnings.append("anisotropic signal currently supports one center; forcing num_centers=1")
                        sample_num_centers = 1
                        sigma_list = [sigma_list[0]]
                        amplitude_list = [amplitude_list[0]]
                        orientation_list = [orientation_list[0]]
                    
                    # Use CLI ring_size if provided, otherwise use the sampled value from the case
                    effective_ring_size = ring_size if ring_size is not None else sampled_cfg["ring_size"]
                    
                    generation_meta: Dict[str, Any] = {
                        "deformation": {
                            "max_ratio": float(sampled_cfg["max_ratio"]),
                            "num_candidates": int(sampled_cfg["num_candidates"]),
                            "group_candidates": bool(sampled_cfg["group_candidates"]),
                            "alpha": float(sampled_cfg["alpha"]),
                            "smooth_iterations": int(sampled_cfg["smooth_iterations"]),
                            "ring_size": int(effective_ring_size),
                            "deform_method": deform_method,
                            "max_iter": int(max_iter),
                        },
                        "template_mesh": mesh_name,
                        "handle_id": handle_ids,
                        "handle_original_pos": [pos.tolist() for pos in handle_positions],
                        "displacement": [disp.tolist() for disp in displacements],
                        "target_pos": [pos.tolist() for pos in target_positions],
                        "center_of_mass": com.tolist(),
                        "sampling_cube_half_side": float(half_side),
                        "patch_radius": float(patch_radius),
                        "roi_vertex_ratio": float(roi_vertex_ratio),
                        "mesh_quality": _json_safe(quality),
                        "deform_meta": _json_safe(deform_meta),
                        "parametrization_method": None if case_name == "case1_no" else param_method,
                        "warnings": warnings,
                    }

                    try:
                        # If a signal is requested, generate both isotropic and anisotropic
                        # For legacy behavior we create both iso/aniso when signal_type is isotropic/anisotropic.
                        # If signal_type == 'mnist' we skip the dual-generation and call save_sample_signal once below.
                        if signal_type is not None and signal_type != "mnist":
                            # Create two factories (one per signal family)
                            signal_factory_iso = SurfaceFactory(root=output_root, template_mesh_path=tmp_path)
                            signal_factory_aniso = SurfaceFactory(root=output_root, template_mesh_path=tmp_path)

                            # First: isotropic with suffix (main, used for classification)
                            paths_iso = save_sample_signal(root=output_root,
                                                           name=f"{sample_name}_iso_000",
                                                           mesh=deformed,
                                                           stats=stats, meta=generation_meta,
                                                           template_id=mesh_name,
                                                           deformation_case=case_name,
                                                           random_seed=sample_seed,
                                                           signal_factory=signal_factory_iso,
                                                           signal_type="isotropic",
                                                           signal_sigma=signal_sigma,
                                                           signal_sigma_ani=signal_sigma_ani,
                                                           signal_sigma_u=signal_sigma_u,
                                                           signal_sigma_v=signal_sigma_v,
                                                           signal_sigma_ratio=signal_sigma_ratio,
                                                           signal_amplitude=signal_amplitude,
                                                           signal_num_centers=sample_num_centers,
                                                           signal_sigma_values=sigma_list,
                                                           signal_amplitude_values=amplitude_list,
                                                           signal_orientation_values=orientation_list,
                                                           rng=sample_rng,
                                                           skip_mesh_save=True)


                            # Extract centers chosen for isotropic signal to match them for anisotropic
                            try:
                                with open(paths_iso["labels"], "r") as fh:
                                    label_iso = json.load(fh)
                                iso_center_ids = label_iso.get("signal", {}).get("center_vertex_ids")
                            except Exception as exc:
                                print(f"WARNING: Failed to read isotropic signal centers: {exc}")
                                traceback.print_exc()
                                iso_center_ids = None

                            # Create a single-center isotropic signal for regression (iso_001).
                            # If the main isotropic already used a single center, reuse it.
                            if int(label_iso.get("signal", {}).get("num_centers", 1)) == 1:
                                paths_iso_single = paths_iso
                            else:
                                paths_iso_single = save_sample_signal(root=output_root,
                                                                     name=f"{sample_name}_iso_001",
                                                                     mesh=deformed,
                                                                     stats=stats, meta=generation_meta,
                                                                     template_id=mesh_name,
                                                                     deformation_case=case_name,
                                                                     random_seed=sample_seed,
                                                                     signal_factory=signal_factory_iso,
                                                                     signal_type="isotropic",
                                                                     signal_sigma=signal_sigma,
                                                                     signal_sigma_ani=signal_sigma_ani,
                                                                     signal_sigma_u=signal_sigma_u,
                                                                     signal_sigma_v=signal_sigma_v,
                                                                     signal_sigma_ratio=signal_sigma_ratio,
                                                                     signal_amplitude=signal_amplitude,
                                                                     signal_num_centers=1,
                                                                     signal_sigma_values=[sigma_list[0]] if sigma_list else None,
                                                                     signal_amplitude_values=[amplitude_list[0]] if amplitude_list else None,
                                                                     signal_orientation_values=[orientation_list[0]] if orientation_list else None,
                                                                     rng=sample_rng,
                                                                     skip_mesh_save=True)

                            # Second: anisotropic with suffix (force single center)
                            paths_aniso = save_sample_signal(root=output_root,
                                                             name=f"{sample_name}_aniso_000",
                                                             mesh=deformed,
                                                             stats=stats, meta=generation_meta,
                                                             template_id=mesh_name,
                                                             deformation_case=case_name,
                                                             random_seed=sample_seed,
                                                             signal_factory=signal_factory_aniso,
                                                             signal_type="anisotropic",
                                                             signal_sigma=signal_sigma,
                                                             signal_sigma_ani=signal_sigma_ani,
                                                             signal_sigma_u=signal_sigma_u,
                                                             signal_sigma_v=signal_sigma_v,
                                                             signal_sigma_ratio=signal_sigma_ratio,
                                                             signal_amplitude=signal_amplitude,
                                                             signal_num_centers=1,
                                                             signal_sigma_values=[
                                                                 sigma_ani_list[0]] if sigma_ani_list else None,
                                                             signal_amplitude_values=[
                                                                 amplitude_list[0]] if amplitude_list else None,
                                                             signal_orientation_values=[
                                                                 orientation_list[0]] if orientation_list else None,
                                                             signal_centers=iso_center_ids,
                                                             rng=sample_rng,
                                                             skip_mesh_save=True)

                            # Merge label JSONs: prefer anisotropic label as base, then insert both signals
                            try:
                                with open(paths_aniso["labels"], "r") as fh:
                                    label_aniso = json.load(fh)
                                with open(paths_iso["labels"], "r") as fh:
                                    label_iso = json.load(fh)
                                with open(paths_iso_single["labels"], "r") as fh:
                                    label_iso_single = json.load(fh)
                            except Exception as exc:
                                # If reading failed, fall back to returning anisotropic paths
                                print(f"ERROR: Failed to merge label JSONs: {exc}")
                                traceback.print_exc()
                                paths = paths_aniso
                            else:
                                n_vertices = int(label_aniso.get("n_vertices", len(deformed.vertices)))
                                iso_single_signal_path = paths_iso_single.get("signal")
                                
                                # Build signal_files with exactly 3 signals:
                                # 1. iso_{N:03d} - main isotropic with N centers (classification)
                                # 2. iso_001_cls - single-center isotropic (classification)
                                # 3. iso_001_reg - single-center isotropic (regression, same file as cls)
                                # 4. aniso_001 - anisotropic (always 1 center, regression)
                                
                                # Convert paths to be relative to output_root for portability
                                dataset_root = Path(output_root)
                                iso_path = Path(paths_iso.get("signal"))
                                iso_single_path = Path(iso_single_signal_path)
                                aniso_path = Path(paths_aniso.get("signal"))
                                
                                signal_files = {
                                    f"iso_{sample_num_centers:03d}": str(iso_path.relative_to(dataset_root)),
                                    "iso_001_cls": str(iso_single_path.relative_to(dataset_root)),
                                    "iso_001_reg": str(iso_single_path.relative_to(dataset_root)),
                                    f"aniso_{1:03d}": str(aniso_path.relative_to(dataset_root)),
                                }

                                # build signals entries (minimal set from existing labels)
                                sig_iso = label_iso.get("signal", {})
                                sig_aniso = label_aniso.get("signal", {})
                                
                                # Extract signal parameters from the return dicts (which now include signal_params)
                                sig_iso_params = paths_iso.get("signal_params", {})
                                sig_aniso_params = paths_aniso.get("signal_params", {})
                                sig_aniso_meta = sig_aniso.get("signal_meta", {}) if isinstance(sig_aniso, dict) else {}
                                sigma_u = sig_aniso_params.get("sigma_u", sig_aniso.get("sigmas", [None])[0] if sig_aniso.get("sigmas") else None)
                                sigma_v = sig_aniso_params.get("sigma_v", None)
                                orientation_angle = sig_aniso_meta.get("phi", sig_aniso_params.get("phi", None))
                                orientation_target = sig_aniso_meta.get(
                                    "target_doubled_angle",
                                    sig_aniso_params.get("target_doubled_angle"),
                                )
                                orientation_target_valid = (
                                    isinstance(orientation_target, (list, tuple))
                                    and len(orientation_target) == 2
                                )
                                orientation_debug = {
                                    "center_unit": sig_aniso_meta.get("center_unit"),
                                    "major_axis": sig_aniso_meta.get("major_axis", sig_aniso_params.get("major_axis")),
                                    "delta": sig_aniso_meta.get("delta", sig_aniso_params.get("delta")),
                                    "phi": orientation_angle,
                                    "target_doubled_angle": orientation_target,
                                    "gauge_e1": sig_aniso_meta.get("gauge_e1", sig_aniso_params.get("gauge_e1")),
                                    "gauge_e2": sig_aniso_meta.get("gauge_e2", sig_aniso_params.get("gauge_e2")),
                                    "hm_e1": sig_aniso_meta.get("hm_e1", sig_aniso_params.get("hm_e1")),
                                    "hm_e2": sig_aniso_meta.get("hm_e2", sig_aniso_params.get("hm_e2")),
                                }
                                # TODO: make this a function for two is okay.. if I need to increase this will explote.
                                signals_list = [
                                    {
                                        "signal_id": "iso_000",
                                        "family": "isotropic",
                                        "model": "surface_gaussian",
                                        "storage": {
                                            "path_key": "iso_000",
                                            "dtype": "float32",
                                            "shape": [n_vertices],
                                            "normalization": "none",
                                        },
                                        "num_centers": sig_iso.get("num_centers", 1),
                                        "centers": sig_iso.get("centers", []),
                                        "center_vertex_ids": sig_iso.get("center_vertex_ids", []),
                                        "center_sampling": {
                                            "method": "random_vertex",
                                            "seed": int(sample_seed),
                                            "avoid_boundary": True,
                                            "min_pairwise_distance": None,
                                        },
                                        "amplitudes": sig_iso.get("amplitudes", []),
                                        "parameters": {
                                            "sigmas": sig_iso.get("sigmas", []),
                                            "distance_type": "geodesic",
                                            "sigma_units": "surface_distance",
                                        },
                                        "generation": {
                                            "combine_centers": "sum",
                                            "clip_min": None,
                                            "clip_max": None,
                                            "normalize_after_generation": False,
                                        },
                                    },
                                    {
                                        "signal_id": "aniso_000",
                                        "family": "anisotropic",
                                        "model": "surface_gaussian",
                                        "storage": {
                                            "path_key": "aniso_000",
                                            "dtype": "float32",
                                            "shape": [n_vertices],
                                            "normalization": "none",
                                        },
                                        "num_centers": sig_aniso.get("num_centers", 1),
                                        "centers": sig_aniso.get("centers", []),
                                        "center_vertex_ids": sig_aniso.get("center_vertex_ids", []),
                                        "center_sampling": {
                                            "method": "matched_to_signal",
                                            "matched_signal_id": "iso_000",
                                            "seed": int(sample_seed),
                                            "avoid_boundary": True,
                                            "min_pairwise_distance": None,
                                        },
                                        "amplitudes": sig_aniso.get("amplitudes", []),
                                        "parameters": {
                                            "sigma_parallel": sigma_u,
                                            "sigma_perpendicular": sigma_v,
                                            "orientation_angles": [orientation_angle] if orientation_angle is not None else None,
                                            "orientation_targets_doubled_angle": [orientation_target] if orientation_target is not None else None,
                                            "hm_basis": {
                                                "e1": orientation_debug.get("hm_e1"),
                                                "e2": orientation_debug.get("hm_e2"),
                                            },
                                            "angle_units": "radians",
                                            "orientation_period": 3.1415926536,
                                            "orientation_target": "cos2phi_sin2phi",
                                            "frame": "sphere_tangent",
                                            "distance_type": "local_tangent_geodesic",
                                            "sigma_units": "surface_distance",
                                        },
                                        "generation": {
                                            "combine_centers": "sum",
                                            "clip_min": None,
                                            "clip_max": None,
                                            "normalize_after_generation": False,
                                        },
                                        "orientation_debug": orientation_debug,
                                    },
                                ]

                                # tasks groups (minimal)
                                task_groups = {
                                    "isotropic_gaussian": {
                                        "signal_id": "iso_000",
                                        "family": "isotropic",
                                        "tasks": {
                                            "number_of_centers": {"valid": True, "label": sig_iso.get("num_centers", 1), "dtype": "int64"},
                                            "center_regression": {"valid": True, "label": sig_iso.get("centers", [None])[0], "dtype": "float32", "target_space": "xyz"},
                                            "sigma_regression": {"valid": True, "label": sig_iso.get("sigmas", [None])[0], "dtype": "float32", "units": "surface_distance"},
                                            "amplitude_regression": {"valid": True, "label": sig_iso.get("amplitudes", [None])[0], "dtype": "float32"},
                                        },
                                    },
                                    "anisotropic_gaussian": {
                                        "signal_id": "aniso_000",
                                        "family": "anisotropic",
                                        "tasks": {
                                            "number_of_centers": {"valid": True, "label": sig_aniso.get("num_centers", 1), "dtype": "int64"},
                                            "center_regression": {"valid": True, "label": sig_aniso.get("centers", [None])[0], "dtype": "float32", "target_space": "xyz"},
                                            "amplitude_regression": {"valid": True, "label": sig_aniso.get("amplitudes", [None])[0], "dtype": "float32"},
                                            "anisotropic_parameters_regression": {"valid": True, "label": {"sigma_parallel": sigma_u, "sigma_perpendicular": sigma_v, "orientation": orientation_angle, "orientation_target_doubled_angle": orientation_target, "hm_e1": orientation_debug.get("hm_e1"), "hm_e2": orientation_debug.get("hm_e2")}, "dtype": "float32", "units": {"sigma_parallel": "surface_distance", "sigma_perpendicular": "surface_distance", "orientation": "radians"}, "orientation_period": 3.1415926536, "target_order": ["sigma_parallel", "sigma_perpendicular", "orientation"]},
                                            "orientation_regression": {"valid": bool(orientation_target_valid), "label": orientation_target, "dtype": "float32", "representation": "cos2phi_sin2phi", "period": 3.1415926536},
                                        },
                                    },
                                }

                                # Create a final label with mesh info, then merge signals
                                root_path = Path(output_root)
                                save_sample_mesh(root=output_root, name=sample_name, mesh=deformed)
                                # Ensure mesh is saved for label paths
                                mesh_path = root_path / "meshes" / f"{sample_name}.obj"
                                if not mesh_path.exists():
                                    raise FileNotFoundError("Mesh was not created properly."
                                                            " This breaks the dataset structure as signals and "
                                                            "labels expect the mesh to exist. Check previous "
                                                            "error logs for issues during mesh saving.")
                                final_labels_path = root_path / "labels" / f"{sample_name}.json"

                                final_label = {
                                    "schema_version": "0.2",
                                    "sample_id": sample_name,
                                    "name": sample_name,
                                    "metadata": {
                                        "dataset_name": "deformations_dataset",
                                        "dataset_version": "0.2",
                                        "template_id": mesh_name,
                                        "deformation_case": case_name,
                                        "created_by": "generate_deformations_dataset.py",
                                        "random_seed": int(sample_seed),
                                    },
                                    "paths": {
                                        "mesh": str(mesh_path.relative_to(root_path)),
                                        "label": str(final_labels_path.relative_to(root_path)),
                                    },
                                    "mesh": {
                                        "n_vertices": n_vertices,
                                        "n_faces": int(deformed.faces.shape[0]),
                                        "topology_id": mesh_name,
                                        "is_watertight": True,
                                        "is_orientable": True,
                                        "coordinate_system": "xyz",
                                        "units": "normalized",
                                        "distance_stats": _json_safe(stats),
                                    },
                                    "signal_files": signal_files,
                                    "signals": signals_list,
                                    "task_groups": task_groups,
                                    "quality_checks": { # TODO: Remove is not used! Hard coded!!!
                                        "mesh_loaded": True,
                                        "signal_files_exist": True,
                                        "label_consistency": True,
                                        "all_signal_lengths_match_n_vertices": True,
                                        "centers_on_surface": True,
                                        "finite_signal_values": True,
                                    },
                                    "deformation": _json_safe(generation_meta.get("deformation", generation_meta)),
                                    "parametrization": {
                                        "method": generation_meta.get("parametrization_method"),
                                        "success": False,
                                    },
                                    "random_seed": int(sample_seed),
                                    "warnings": _json_safe(generation_meta.get("warnings", [])),
                                }

                                # Write mesh if not already written
                                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                                if not mesh_path.exists():
                                    deformed.export(str(mesh_path))

                                # Write merged label
                                final_labels_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(final_labels_path, "w") as fh:
                                    json.dump(final_label, fh, indent=2)

                                paths = {
                                    "mesh": str(mesh_path),
                                    "labels": str(final_labels_path),
                                    "signals": [paths_iso.get("signal"), paths_aniso.get("signal")],
                                }
                        else:
                            paths = save_sample_signal(root=output_root, name=sample_name, mesh=deformed, stats=stats,
                                                       meta=generation_meta, template_id=mesh_name,
                                                       deformation_case=case_name, random_seed=sample_seed,
                                                       signal_factory=signal_factory, signal_type=signal_type,
                                                       signal_sigma=signal_sigma,
                                                       signal_sigma_ani=signal_sigma_ani,
                                                       signal_sigma_u=signal_sigma_u,
                                                       signal_sigma_v=signal_sigma_v,
                                                       signal_sigma_ratio=signal_sigma_ratio,
                                                       signal_amplitude=signal_amplitude,
                                                       signal_num_centers=sample_num_centers,
                                                       signal_sigma_values=sigma_list,
                                                       signal_amplitude_values=amplitude_list,
                                                       mnist_index=mnist_index,
                                                       rng=sample_rng,
                                                       skip_mesh_save=(case_name == "case1_no"))
                            # Ensure mesh file exists on disk; some callers may skip mesh write — write it here if missing
                            try:
                                mesh_file_path = Path(paths.get('mesh'))
                                if mesh_file_path and not mesh_file_path.exists():
                                    mesh_file_path.parent.mkdir(parents=True, exist_ok=True)
                                    deformed.export(str(mesh_file_path))
                            except Exception:
                                # Log but continue; label writing will surface errors if mesh still missing
                                pass
                    except Exception as exc:  # noqa: BLE001
                        append_error_log(
                            log_path,
                            sample_name,
                            f"save_sample failed: {exc}",
                            template_id=mesh_name,
                            deformation_case=case_name,
                            traceback_text=traceback.format_exc(),
                        )
                        total_failed += 1
                        _safe_unlink(tmp_path)
                        continue

                    param_success = False
                    param_error = None
                    effective_param_method = _effective_param_method(signal_type, case_name, param_method)
                    if effective_param_method is not None:
                        try:
                            sphere_paths = save_spherical_parametrization(
                                root=output_root,
                                name=sample_name,
                                vertices=np.asarray(deformed.vertices, dtype=np.float64),
                                faces=np.asarray(deformed.faces, dtype=np.int32),
                                method=effective_param_method,
                                cem_eps=cem_eps,
                                cem_max_iters=cem_max_iters,
                                cem_verbose=cem_verbose,
                            )
                            paths.update(sphere_paths)
                            param_success = True
                        except Exception as exc:  # noqa: BLE001
                            param_error = str(exc)
                            append_error_log(
                                log_path,
                                sample_name,
                                f"spherical parametrization failed: {exc}",
                                template_id=mesh_name,
                                deformation_case=case_name,
                                traceback_text=traceback.format_exc(),
                            )

                    label_updates: Dict[str, Any] = {
                        "parametrization": {
                            "method": effective_param_method,
                            "success": bool(param_success),
                            "error": param_error,
                        },
                        "sphere_path": paths.get("sphere"),
                    }
                    if paths.get("sphere"):
                        label_updates["paths"] = {"sphere": paths["sphere"]}

                    _update_sample_label(paths["labels"], label_updates)

                    ok, issues = validate_saved_sample(paths["labels"])
                    if not ok:
                        append_error_log(
                            log_path,
                            sample_name,
                            "validation failed: " + "; ".join(issues),
                            template_id=mesh_name,
                            deformation_case=case_name,
                        )
                        total_failed += 1
                        _safe_unlink(tmp_path)
                        continue

                    print(
                        f"  saved {sample_name} → {paths['labels']} "
                        f"(mean_dist={stats['mean_distance']:.4f}, std_dist={stats['std_distance']:.4f})"
                    )
                    total_saved += 1

                    # Clean up temp file
                    _safe_unlink(tmp_path)


    if create_splits:
        try:
            build_task_splits(
                dataset_root=output_root,
                tasks=split_tasks if split_tasks is not None else DEFAULT_TASKS,
                num_folds=num_folds,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=split_seed,
                group_by_template=group_by_template,
            )
        except Exception as exc:  # noqa: BLE001
            append_error_log(
                log_path,
                "split_builder",
                f"split generation failed: {exc}",
                traceback_text=traceback.format_exc(),
            )
            raise

    print(f"\nDone. Saved: {total_saved} samples, Failed: {total_failed}")
    print(f"Error log  : {log_path}")
    print(f"Output root: {output_root}")
    return total_saved


# ===========================================================================
# Helpers
# ===========================================================================

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/non-serialisable types to JSON-safe Python."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ===========================================================================
# 14. CLI
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a mesh deformation dataset with graphop and trimesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing input mesh files.")
    parser.add_argument("--output-root", default="data/generated", help="Root directory for generated output.")
    parser.add_argument("--n-samples-per-mesh", type=int, default=25, help="Number of deformation samples to attempt per input mesh.")
    parser.add_argument("--patch-radius-ratio", type=float, default=0.15, help="Patch radius as a fraction of the mesh bounding-box diagonal.")
    parser.add_argument("--smoothing-iterations", type=int, default=3, help="Base smoothing passes (overridden by deformation case configuration).")
    parser.add_argument("--group-candidates", type=int, default=5, help="Legacy parameter kept for backward compatibility.")
    parser.add_argument(
        "--roi-vertex-ratio",
        type=float,
        default=0.3,
        help="ROI-growth stop criterion as a fraction of the mesh vertex count. For MNIST, this sets the ROI coverage around the north pole.",
    )
    parser.add_argument("--max-ratio", type=float, default=0.8, help="Legacy parameter kept for backward compatibility.")
    parser.add_argument("--ring-size", type=int, default=None, help="Euclidean translation ring radius (if not provided, value is sampled from the deformation case).")
    parser.add_argument(
        "--deform-method",
        choices=("sre_arap", "original_arap", "spokes_and_rims"),
        default="sre_arap",
        help="graphop deformation algorithm.",
    )
    parser.add_argument("--alpha", type=float, default=0.02, help="SRE-ARAP smoothness weight.")
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum ARAP iterations.")
    parser.add_argument(
        "--signal-type",
        choices=("isotropic", "anisotropic", "mnist"),
        default="isotropic",
        help="Synthetic signal family attached after deformation.",
    )
    parser.add_argument("--signal-sigma", type=float, default=0.2, help="Signal width parameter for isotropic signals.")
    parser.add_argument("--signal-sigma-ani", type=float, default=0.8, help="Signal width parameter for anisotropic signals.")
    parser.add_argument("--signal-sigma-u", type=float, default=None, help="Base anisotropic sigma (sigma_u). If not set, falls back to --signal-sigma-ani.")
    parser.add_argument("--signal-sigma-v", type=float, default=None, help="Explicit anisotropic sigma_v; overrides ratio-based sigma_v.")
    parser.add_argument("--signal-sigma-ratio", type=float, default=0.5, help="sigma_v = ratio * sigma_u when --signal-sigma-v not set. Default 0.5 for backward compatibility.")
    parser.add_argument("--signal-amplitude", type=float, default=1.0, help="Signal amplitude.")
    parser.add_argument(
        "--mnist-percentage",
        type=float,
        default=100.0,
        help="Percentage of MNIST dataset to use (0.1-100.0). Only applies when --signal-type mnist. Default: 100%% (all 70,000 images).",
    )
    parser.add_argument(
        "--mnist-total-count",
        type=int,
        default=None,
        help="Explicit total count of MNIST samples to generate. Overrides --mnist-percentage if provided.",
    )
    parser.add_argument("--signal-num-centers", type=int, default=1, help="Maximum number of signal centers (used when --signal-centers-options is not provided).")
    parser.add_argument(
        "--signal-centers-options",
        type=str,
        default=None,
        help="Comma-separated center counts to sample per generated sample (e.g. '1,2,3,4,5').",
    )
    parser.add_argument(
        "--signal-sigma-variation",
        type=float,
        default=20.0,
        help="Variation range for sigma as percentage (default 20%%, means ±20%%).",
    )
    parser.add_argument(
        "--signal-amplitude-variation",
        type=float,
        default=20.0,
        help="Variation range for amplitude as percentage (default 20%%, means ±20%%).",
    )
    parser.add_argument(
        "--param-method",
        choices=("flash", "cem", "none"),
        default="none",
        help="Spherical parametrization method saved under spheres/ and labels/*_spherical.json.",
    )
    parser.add_argument("--cem-eps", type=float, default=1e-6, help="CEM convergence epsilon (if --param-method cem).")
    parser.add_argument("--cem-max-iters", type=int, default=100, help="CEM max iterations (if --param-method cem).")
    parser.add_argument("--cem-verbose", action="store_true", help="Enable CEM verbose logs (if --param-method cem).")
    parser.add_argument(
        "--deformation-cases",
        type=str,
        default="case2_small,case3_large",
        help="Comma-separated deformation cases to generate (subset of: case2_small, case3_large).",
    )
    parser.add_argument("--create-splits", action="store_true", help="Build task-specific fold split files after generation.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds to generate.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training ratio used by split builder.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio used by split builder.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Testing ratio used by split builder.")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed used by split builder.")
    parser.add_argument("--group-by-template", dest="group_by_template", action="store_true", default=True, help="Keep samples from same template in same fold (default true).")
    parser.add_argument("--no-group-by-template", dest="group_by_template", action="store_false", help="Allow fold assignment at sample level.")
    parser.add_argument(
        "--split-tasks",
        type=str,
        default="number_of_centers,center_regression,sigma_regression,amplitude_regression",
        help="Comma-separated task names to build splits for (supported: number_of_centers, center_regression, sigma_regression, amplitude_regression, mnist_cls).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable automatic resume and generate the full requested plan from sample index 0.",
    )
    parser.add_argument("--no-repair-holes", action="store_true", help="Disable hole repair on non-watertight input meshes.")
    parser.add_argument("--drop-non-watertight", action="store_true", help="Drop deformations that are not watertight after smoothing/validation.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for dataset generation."""
    args = build_arg_parser().parse_args(argv)
    signal_type = args.signal_type
    param_method = None if args.param_method == "none" else args.param_method
    signal_centers_options = None
    if args.signal_centers_options:
        signal_centers_options = [int(x.strip()) for x in args.signal_centers_options.split(",") if x.strip()]
    deformation_cases = [x.strip() for x in args.deformation_cases.split(",") if x.strip()]
    split_tasks = [x.strip() for x in args.split_tasks.split(",") if x.strip()]
    generate_dataset(
        input_dir=args.input_dir,
        output_root=args.output_root,
        n_samples_per_mesh=args.n_samples_per_mesh,
        patch_radius_ratio=args.patch_radius_ratio,
        smoothing_iterations=args.smoothing_iterations,
        group_candidates=args.group_candidates,
        roi_vertex_ratio=args.roi_vertex_ratio,
        max_ratio=args.max_ratio,
        ring_size=args.ring_size,
        deform_method=args.deform_method,
        alpha=args.alpha,
        max_iter=args.max_iter,
        signal_type=signal_type,
        signal_sigma=args.signal_sigma,
        signal_sigma_ani=args.signal_sigma_ani,
        signal_amplitude=args.signal_amplitude,
        signal_num_centers=args.signal_num_centers,
        signal_centers_options=signal_centers_options,
        signal_sigma_variation_percent=args.signal_sigma_variation,
        signal_amplitude_variation_percent=args.signal_amplitude_variation,
        param_method=param_method,
        cem_eps=args.cem_eps,
        cem_max_iters=args.cem_max_iters,
        cem_verbose=args.cem_verbose,
        deformation_cases=deformation_cases,
        create_splits=args.create_splits,
        split_tasks=split_tasks,
        num_folds=args.num_folds,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        group_by_template=args.group_by_template,
        seed=args.seed,
        repair_holes=not args.no_repair_holes,
        drop_non_watertight=args.drop_non_watertight,
        offset_sample_counter=0,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
