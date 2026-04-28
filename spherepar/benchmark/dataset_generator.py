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
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.smoothing

from spherepar.benchmark.surface import Surface, SurfaceFactory


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
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "deform_surface"):
                return module

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


# ===========================================================================
# 1. Mesh loading
# ===========================================================================

def load_meshes_from_directory(
        directory: str,
) -> List[Tuple[str, trimesh.Trimesh]]:
    """Load all .obj files from *directory*.

    Returns a list of (filename_stem, trimesh.Trimesh) pairs for every file
    that loaded successfully and has at least one face.

    Parameters
    ----------
    directory:
        Path to a directory containing .obj files.

    Returns
    -------
    list of (name, mesh) tuples
    """
    directory = Path(directory)
    results: List[Tuple[str, trimesh.Trimesh]] = []

    for obj_path in sorted(directory.glob("*.obj")):
        try:
            loaded = trimesh.load(str(obj_path), force="mesh")
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
            results.append((obj_path.stem, mesh))
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
# 6. ROI patch extraction
# ===========================================================================

def extract_roi_patch(
        vertices: np.ndarray,
        center: np.ndarray,
        radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract vertices within *radius* of *center* using a KD-tree query.

    Uses scipy.spatial.cKDTree (O(n log n) construction, O(log n) per query)
    for efficient spatial indexing.

    Parameters
    ----------
    vertices:
        All mesh vertices, shape (N, 3).
    center:
        Query point, shape (3,).
    radius:
        Search radius.

    Returns
    -------
    patch_vertices : np.ndarray, shape (K, 3)
        Positions of vertices in the patch.
    patch_indices : np.ndarray, shape (K,)
        0-based indices into *vertices*.
    """
    from scipy.spatial import cKDTree  # pylint: disable=import-outside-toplevel

    tree = cKDTree(vertices)
    indices = np.array(tree.query_ball_point(center, radius), dtype=np.intp)
    if len(indices) == 0:
        # Fallback: nearest single vertex
        print("Fallback no nearest in radius ", radius)
        _, idx = tree.query(center)
        indices = np.array([idx], dtype=np.intp)
    return vertices[indices], indices


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


# ===========================================================================
# 10. Save sample
# ===========================================================================

def save_sample(
        root: str,
        name: str,
        mesh: trimesh.Trimesh,
        stats: Dict[str, float],
        meta: Dict[str, Any],
        signal_factory: Optional[SurfaceFactory] = None,
        signal_type: Optional[str] = None,
        signal_sigma: float = 0.2,
        signal_amplitude: float = 1.0,
        signal_num_centers: int = 1,
        rng: Optional[np.random.Generator] = None,
) -> Dict[str, str]:
    """Save a valid dataset sample to disk.

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
    signal_amplitude:
        Amplitude used for the generated signal.
    signal_num_centers:
        Number of centers used to generate the synthetic signal.
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

    # Write OBJ
    mesh.export(str(mesh_path))

    # Write signal arrays
    # create a surface without signal
    # create use signal factory maybe as dummy objec just to use methos pdate signal as create signals
    signal_info: Dict[str, Any] = {}
    signal_path: Optional[str] = None
    if signal_type is not None:
        if signal_factory is None:
            raise ValueError("signal_factory is required when signal_type is not None")
        if rng is None:
            rng = np.random.default_rng()
        if signal_num_centers <= 0:
            raise ValueError("signal_num_centers must be positive")

        signal_center_idx = rng.integers(0, len(mesh.vertices), size=signal_num_centers)
        signal_centers = [int(idx) for idx in np.atleast_1d(signal_center_idx)]
        if signal_type == "isotropic":
            signal_params: Dict[str, Any] = {
                "centers": signal_centers,
                "sigma": signal_sigma,
                "amplitude": signal_amplitude,
            }
        else:
            signal_params = {
                "center": signal_centers[0],
                "sigma_u": signal_sigma,
                "sigma_v": max(signal_sigma * 0.5, 1e-6),
                "amplitude": signal_amplitude,
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
            "signal_file": signal_paths["signal"],
            "signal_label_file": signal_paths["signal_label"],
        }
        meta["signal"] = signal_info

    # Write JSON metadata
    label = {
        "name": name,
        "mesh_file": str(mesh_path),
        "signal_file": signal_path,
        "n_vertices": int(mesh.vertices.shape[0]),
        "n_faces": int(mesh.faces.shape[0]),
        "distance_stats": stats,
        "deformation": _json_safe(meta),
    }
    with open(labels_path, "w") as fh:
        json.dump(label, fh, indent=2)

    result = {
        "mesh": str(mesh_path),
        "labels": str(labels_path),
    }
    if signal_path is not None:
        result["signal"] = signal_path
    return result


# ===========================================================================
# 11. Error log
# ===========================================================================

def append_error_log(log_path: str, name: str, reason: str) -> None:
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
        fh.write(f"{timestamp}  {name}  {reason}\n")


# ===========================================================================
# 12. Top-level generator
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
        ring_size: float = 0.0,
        deform_method: str = "sre_arap",
        alpha: float = 0.02,
        max_iter: int = 50,
        seed: int = 42,
        repair_holes: bool = True,
        drop_non_watertight: bool = False,
        signal_type: Optional[str] = "isotropic",
        signal_sigma: float = 0.2,
        signal_amplitude: float = 1.0,
        signal_num_centers: int = 1,
) -> None:
    """Run the full dataset generation pipeline.

    Parameters
    ----------
    input_dir:
        Directory containing input .obj meshes.
    output_root:
        Root directory for generated output.
    n_samples_per_mesh:
        Number of deformation samples to attempt per input mesh.
    patch_radius_ratio:
        ROI patch radius as a fraction of the mesh bounding-box diagonal.
    smoothing_iterations:
        Humphrey smoothing passes applied after deformation.
    group_candidates:
        Number of sampled handle vertices grouped into each deformation call.
    roi_vertex_ratio:
        ROI-growth stop criterion expressed as a fraction of the mesh vertex
        count.
    max_ratio:
        Maximum displacement as a fraction of dist(handle, center_of_mass).
    ring_size:
        Euclidean translation ring radius passed to graphop.deform_surface.
    deform_method:
        graphop deformation algorithm ('sre_arap', 'original_arap',
        'spokes_and_rims').
    alpha:
        SRE-ARAP smoothness weight.
    max_iter:
        Maximum ARAP iterations.
    seed:
        Random seed for reproducibility.
    repair_holes:
        Whether to attempt hole repair on meshes that are not watertight.
    drop_non_watertight:
        Whether to discard deformations that are not watertight after
        smoothing/validation.
    signal_type:
        Synthetic signal family attached after deformation. ``None`` disables
        signal generation. The default is ``"isotropic"``.
    signal_sigma:
        Width parameter used for the default isotropic Gaussian signal.
    signal_amplitude:
        Amplitude used for the default isotropic Gaussian signal.
    signal_num_centers:
        Number of centers used for isotropic signal generation.
    """
    global nearest_ids
    if not _GRAPHOP_AVAILABLE:
        raise ImportError(
            "graphop C++ extension is required. Build it with CMake (see BUILD.md)."
        )
    if group_candidates <= 0:
        raise ValueError("group_candidates must be positive")
    if not (0.0 < roi_vertex_ratio <= 1.0):
        raise ValueError("roi_vertex_ratio must be in the interval (0, 1]")
    if max_ratio < 0.0:
        raise ValueError("max_ratio must be non-negative")
    if ring_size < 0.0:
        raise ValueError("ring_size must be non-negative")
    if signal_sigma <= 0.0:
        raise ValueError("signal_sigma must be positive")
    if signal_num_centers <= 0:
        raise ValueError("signal_num_centers must be positive")
    if signal_type not in (None, "isotropic", "anisotropic"):
        raise ValueError("signal_type must be one of None, 'isotropic', or 'anisotropic'")

    output_root = str(output_root)
    log_path = str(Path(output_root) / "logs" / "errors.log")
    rng = np.random.default_rng(seed)
    # --- Load meshes --------------------------------------------------------
    print(f"Loading meshes from: {input_dir}")
    mesh_pairs = load_meshes_from_directory(input_dir)
    if not mesh_pairs:
        print("No .obj files found or all failed to load.")
        return
    print(f"  Loaded {len(mesh_pairs)} mesh(es)")

    total_saved = 0
    total_failed = 0

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

        signal_factory = None
        if signal_type is not None:
            signal_factory = SurfaceFactory(root=output_root, template_mesh_path=tmp_path)

        # --- Sample and deform ----------------------------------------------
        # TODO: this will be used for deformation rotations
        # candidates = sample_handle_centers(com, half_side, n_samples_per_mesh, rng)
        handle_centers, candidates= sample_handle_vertices(mesh, n_samples_per_mesh, rng)
        normals = mesh.vertex_normals[candidates]
        # handle_centers = [mesh.vertices[i] for i in candidates]

        sample_idx = 0
        number_vertices = len(mesh.vertices)

        for group_start in range(0, len(candidates), group_candidates):
            group_stop = min(group_start + group_candidates, len(candidates))
            group_handle_ids = [int(h) for h in candidates[group_start:group_stop]]
            group_handle_centers = handle_centers[group_start:group_stop]
            group_normals = normals[group_start:group_stop]

            roi_union: set[int] = set()
            target_positions: List[np.ndarray] = []
            displacements: List[np.ndarray] = []
            handle_positions: List[np.ndarray] = []

            for handle_id, handle_center, normal in zip(group_handle_ids, group_handle_centers, group_normals):
                print('normal', normal)
                _, nearest_ids = extract_roi_patch(mesh.vertices, handle_center, 0.0)
                for r in np.linspace(0.0, bbox_diag, num=100):
                    _, nearest_ids = extract_roi_patch(mesh.vertices, handle_center, radius=r)
                    if len(nearest_ids) > roi_vertex_ratio * number_vertices:
                        print(
                            f"Found {len(nearest_ids)} nearest ids for the handle {handle_center} "
                            f"which is the {100 * len(nearest_ids) / number_vertices:.2f}% of the handle center."
                            f" Radius used: {r:.4f}")
                        break

                roi_union.update(int(idx) for idx in nearest_ids)

                handle_pos = mesh.vertices[handle_id].copy()
                displacement = compute_valid_displacement(handle_pos, com, normal, rng, max_ratio=max_ratio)
                target_pos = handle_pos + displacement
                # if mesh.contains([target_pos]):
                #     print("Warning: target position is inside the mesh; deformation may fail or produce degenerate geometry.")
                #     print("Inverting direction....")
                #     displacement = -displacement
                #     target_pos = handle_pos + displacement

                print(" diplacement: ", displacement)
                print(" target_pos: ", target_pos)
                print(" handle_id: ", handle_id)

                handle_positions.append(handle_pos)
                displacements.append(displacement)
                target_positions.append(target_pos)

            if not roi_union:
                append_error_log(log_path, mesh_name, "empty ROI for grouped deformation")
                total_failed += 1
                continue

            sample_name = f"{mesh_name}_s{sample_idx:04d}"

            try:
                V_new, F_new, deform_meta = deform_mesh_with_graphop(
                    mesh_path=tmp_path,
                    handle_id=group_handle_ids,
                    target_pos=np.asarray(target_positions, dtype=np.float64),
                    ring_size=ring_size,
                    roi_ids=sorted(roi_union),
                    method=deform_method,
                    alpha=alpha,
                    max_iter=max_iter,
                )
            except Exception as exc:  # noqa: BLE001
                # let's make it dangerously catch all exceptions.
                print("Warning: check if a real error happened see log: ", exc)
                append_error_log(log_path, sample_name, f"deformation failed: {exc}")
                total_failed += 1
                continue

            # Smooth + validate
            deformed, quality = smooth_and_validate_mesh(
                V_new,
                F_new,
                smoothing_iterations,
                drop_non_watertight=drop_non_watertight,
            )
            print(
                f"    mesh quality: watertight={quality['is_watertight']} "
                f"degenerate_faces={quality['degenerate_face_count']}"
            )
            if deformed is None:
                append_error_log(
                    log_path, sample_name,
                    "mesh invalid after deformation/smoothing "
                    f"(watertight={quality['is_watertight']}, "
                    f"degenerate_faces={quality['degenerate_face_count']}, "
                    f"reason={quality.get('validation_error', 'unknown')})"
                )
                total_failed += 1
                continue

            # patch_center = np.mean(np.asarray(target_positions, dtype=np.float64), axis=0)
            #
            # # ROI patch
            # patch_verts, patch_idxs = extract_roi_patch(
            #     deformed.vertices, patch_center, patch_radius
            # )

            # Distance statistics
            stats = compute_patch_to_mesh_stats(mesh.vertices, deformed)

            # Augment metadata
            generation_meta: Dict[str, Any] = {
                "template_mesh": mesh_name,
                "handle_id": group_handle_ids,
                "handle_original_pos": [pos.tolist() for pos in handle_positions],
                "displacement": [disp.tolist() for disp in displacements],
                "target_pos": [pos.tolist() for pos in target_positions],
                "center_of_mass": com.tolist(),
                "sampling_cube_half_side": float(half_side),
                "patch_radius": float(patch_radius),
                "roi_vertex_ratio": float(roi_vertex_ratio),
                "group_candidates": int(group_candidates),
                "max_ratio": float(max_ratio),
                "ring_size": float(ring_size),
                "mesh_quality": _json_safe(quality),
                "deform_meta": _json_safe(deform_meta),
            }
            # Save
            paths = save_sample(
                root=output_root,
                name=sample_name,
                mesh=deformed,
                stats=stats,
                meta=generation_meta,
                signal_factory=signal_factory,
                signal_type=signal_type,
                signal_sigma=signal_sigma,
                signal_amplitude=signal_amplitude,
                signal_num_centers=signal_num_centers,
                rng=rng,
            )
            print(
                f"  [{sample_idx+1}/{n_samples_per_mesh}] saved → {paths['labels']} "
                f"(mean_dist={stats['mean_distance']:.4f}, "
                f"std_dist={stats['std_distance']:.4f})"
            )
            total_saved += 1
            sample_idx += 1

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    print(f"\nDone. Saved: {total_saved} samples, Failed: {total_failed}")
    print(f"Error log  : {log_path}")
    print(f"Output root: {output_root}")


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
# 13. CLI
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate a mesh deformation dataset with graphop and trimesh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing input .obj meshes.")
    parser.add_argument("--output-root", default="data/generated", help="Root directory for generated output.")
    parser.add_argument("--n-samples-per-mesh", type=int, default=25, help="Number of deformation samples to attempt per input mesh.")
    parser.add_argument("--patch-radius-ratio", type=float, default=0.15, help="Patch radius as a fraction of the mesh bounding-box diagonal.")
    parser.add_argument("--smoothing-iterations", type=int, default=3, help="Number of Humphrey smoothing passes after deformation.")
    parser.add_argument("--group-candidates", type=int, default=5, help="Number of sampled handle vertices grouped into each deformation call.")
    parser.add_argument("--roi-vertex-ratio", type=float, default=0.3, help="ROI-growth stop criterion as a fraction of the mesh vertex count.")
    parser.add_argument("--max-ratio", type=float, default=0.8, help="Maximum displacement magnitude as a fraction of dist(handle, center_of_mass).")
    parser.add_argument("--ring-size", type=float, default=0.0, help="Euclidean translation ring radius passed to graphop.deform_surface.")
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
        choices=("isotropic", "anisotropic", "none"),
        default="isotropic",
        help="Synthetic signal family attached after deformation.",
    )
    parser.add_argument("--signal-sigma", type=float, default=0.2, help="Signal width parameter.")
    parser.add_argument("--signal-amplitude", type=float, default=1.0, help="Signal amplitude.")
    parser.add_argument("--signal-num-centers", type=int, default=1, help="Number of centers used for isotropic signal generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--no-repair-holes", action="store_true", help="Disable hole repair on non-watertight input meshes.")
    parser.add_argument("--drop-non-watertight", action="store_true", help="Drop deformations that are not watertight after smoothing/validation.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for dataset generation."""
    args = build_arg_parser().parse_args(argv)
    signal_type = None if args.signal_type == "none" else args.signal_type
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
        signal_amplitude=args.signal_amplitude,
        signal_num_centers=args.signal_num_centers,
        seed=args.seed,
        repair_holes=not args.no_repair_holes,
        drop_non_watertight=args.drop_non_watertight,
    )


if __name__ == "__main__":
    main()
