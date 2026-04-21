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
7. Extract a local ROI patch around the deformation center.
8. Compute patch-to-mesh distance statistics (mean, std).
9. Save only valid samples; log all failures.

Output layout (under a configurable root)::

    data/generated/
        meshes/     — deformed OBJ meshes
        patches/    — ROI patch vertex positions (.npy)
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
save_sample(root, name, mesh, patch, stats, meta)
append_error_log(log_path, name, reason)
generate_dataset(input_dir, output_root, ...)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.smoothing

# ---------------------------------------------------------------------------
# graphop import (lazy, so the module can be imported without the .so)
# ---------------------------------------------------------------------------
try:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    import graphop as _graphop
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
# 5. Displacement constraint
# ===========================================================================

def compute_valid_displacement(
        center: np.ndarray,
        com: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        max_ratio: float = 0.1,
) -> np.ndarray:
    """Sample a displacement vector with magnitude ≤ max_ratio * dist(center, com).

    Parameters
    ----------
    center:
        Deformation / handle center, shape (3,).
    com:
        Mesh center of mass, shape (3,).
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
    # Sample direction uniformly on S^2 then scale by a random fraction
    direction = rng.standard_normal(3)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction /= norm
    magnitude = rng.uniform(0.0, max_mag)
    return direction * magnitude


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
        _, idx = tree.query(center)
        indices = np.array([idx], dtype=np.intp)
    return vertices[indices], indices


# ===========================================================================
# 7. Deformation via graphop
# ===========================================================================

def deform_mesh_with_graphop(
        mesh_path: str,
        handle_id: int,
        target_pos: np.ndarray,
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
    target = np.asarray(target_pos, dtype=np.float64).ravel()
    V_new, F, meta = _graphop.deform_surface(
        mesh_path=mesh_path,
        handle_ids=[handle_id],
        target_positions=target,
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
) -> Optional[trimesh.Trimesh]:
    """Apply Humphrey smoothing and validate the resulting mesh.

    Parameters
    ----------
    vertices:
        Vertex positions, shape (N, 3).
    faces:
        Face connectivity, shape (M, 3).
    iterations:
        Number of smoothing passes.

    Returns
    -------
    trimesh.Trimesh or None
        Validated mesh, or None if invalid after smoothing.
    """
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    trimesh.smoothing.filter_humphrey(mesh, iterations=iterations)
    # Remove degenerate and duplicate faces via boolean mask
    mask = mesh.nondegenerate_faces()
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()

    if len(mesh.faces) == 0:
        return None
    # Check for degenerate / zero-area faces
    areas = mesh.area_faces
    if np.any(areas <= 0.0):
        return None
    return mesh


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
        patch_vertices: np.ndarray,
        patch_indices: np.ndarray,
        stats: Dict[str, float],
        meta: Dict[str, Any],
) -> Dict[str, str]:
    """Save a valid dataset sample to disk.

    Directory layout::

        <root>/
          meshes/<name>.obj
          patches/<name>_vertices.npy
          patches/<name>_indices.npy
          labels/<name>.json

    Parameters
    ----------
    root:
        Root output directory.
    name:
        Base filename (without extension).
    mesh:
        Deformed, validated trimesh.Trimesh.
    patch_vertices:
        ROI patch vertex positions, shape (K, 3).
    patch_indices:
        ROI vertex indices into the deformed mesh, shape (K,).
    stats:
        Distance statistics dict.
    meta:
        Deformation / generation metadata dict.

    Returns
    -------
    dict mapping 'mesh', 'patch_vertices', 'patch_indices', 'labels'
    to the paths of saved files.
    """
    root_path = Path(root)
    meshes_dir = root_path / "meshes"
    patches_dir = root_path / "patches"
    labels_dir = root_path / "labels"
    for d in (meshes_dir, patches_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    mesh_path = meshes_dir / f"{name}.obj"
    patch_v_path = patches_dir / f"{name}_vertices.npy"
    patch_i_path = patches_dir / f"{name}_indices.npy"
    labels_path = labels_dir / f"{name}.json"

    # Write OBJ
    mesh.export(str(mesh_path))

    # Write patch arrays
    np.save(str(patch_v_path), patch_vertices)
    np.save(str(patch_i_path), patch_indices)

    # Write JSON metadata
    label = {
        "name": name,
        "mesh_file": str(mesh_path),
        "patch_vertices_file": str(patch_v_path),
        "patch_indices_file": str(patch_i_path),
        "n_vertices": int(mesh.vertices.shape[0]),
        "n_faces": int(mesh.faces.shape[0]),
        "n_patch_points": int(len(patch_indices)),
        "distance_stats": stats,
        "deformation": _json_safe(meta),
    }
    with open(labels_path, "w") as fh:
        json.dump(label, fh, indent=2)

    return {
        "mesh": str(mesh_path),
        "patch_vertices": str(patch_v_path),
        "patch_indices": str(patch_i_path),
        "labels": str(labels_path),
    }


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
        n_samples_per_mesh: int = 5,
        patch_radius_ratio: float = 0.15,
        smoothing_iterations: int = 3,
        deform_method: str = "sre_arap",
        alpha: float = 0.02,
        max_iter: int = 50,
        seed: int = 42,
        repair_holes: bool = True,
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
    """
    if not _GRAPHOP_AVAILABLE:
        raise ImportError(
            "graphop C++ extension is required. Build it with CMake (see BUILD.md)."
        )

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

        # --- Sample and deform ----------------------------------------------
        candidates = sample_handle_centers(com, half_side, n_samples_per_mesh, rng)
        sample_idx = 0

        for i, handle_center in enumerate(candidates):
            # Find the nearest vertex to use as the graphop handle
            _, nearest_ids = extract_roi_patch(mesh.vertices, handle_center, 0.0)
            handle_id = int(nearest_ids[0])
            handle_pos = mesh.vertices[handle_id].copy()

            # Constrained displacement
            displacement = compute_valid_displacement(handle_pos, com, rng)
            target_pos = handle_pos + displacement

            sample_name = f"{mesh_name}_s{sample_idx:04d}"

            # Deform
            try:
                V_new, F_new, deform_meta = deform_mesh_with_graphop(
                    mesh_path=tmp_path,
                    handle_id=handle_id,
                    target_pos=target_pos,
                    roi_ids=None,
                    method=deform_method,
                    alpha=alpha,
                    max_iter=max_iter,
                )
            except Exception as exc:  # noqa: BLE001
                append_error_log(
                    log_path, sample_name, f"deformation failed: {exc}"
                )
                total_failed += 1
                continue

            # Smooth + validate
            deformed = smooth_and_validate_mesh(V_new, F_new, smoothing_iterations)
            if deformed is None:
                append_error_log(
                    log_path, sample_name,
                    "mesh invalid after deformation/smoothing"
                )
                total_failed += 1
                continue

            # ROI patch
            patch_verts, patch_idxs = extract_roi_patch(
                deformed.vertices, target_pos, patch_radius
            )

            # Distance statistics
            stats = compute_patch_to_mesh_stats(patch_verts, deformed)

            # Augment metadata
            generation_meta: Dict[str, Any] = {
                "template_mesh": mesh_name,
                "handle_id": handle_id,
                "handle_original_pos": handle_pos.tolist(),
                "displacement": displacement.tolist(),
                "target_pos": target_pos.tolist(),
                "center_of_mass": com.tolist(),
                "sampling_cube_half_side": float(half_side),
                "patch_radius": float(patch_radius),
                "deform_meta": _json_safe(deform_meta),
            }

            # Save
            paths = save_sample(
                root=output_root,
                name=sample_name,
                mesh=deformed,
                patch_vertices=patch_verts,
                patch_indices=patch_idxs,
                stats=stats,
                meta=generation_meta,
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