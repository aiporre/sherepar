"""Compatibility wrapper for spherical parametrization utilities."""
from __future__ import annotations

from typing import Tuple, Dict, Any, List

import numpy as np
import trimesh

from spherepar.cem_parametrization import stretch_parametrization
from spherepar.flash_parametrization import (  # noqa: F401
    flash_map,
    load_mesh_with_trimesh,
)
from spherepar.mesh import MeshFactory


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face normals using cross product of edge vectors.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions (N, 3).
    faces : np.ndarray
        Face indices (M, 3).
    
    Returns
    -------
    np.ndarray
        Face normals (M, 3), not normalized.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    normals = np.cross(edge1, edge2)
    return normals


def verify_topology_preserved(
    vertices_orig: np.ndarray,
    faces_orig: np.ndarray,
    vertices_mapped: np.ndarray,
    faces_mapped: np.ndarray,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify that parametrization preserved vertex-face correspondence.
    
    Parameters
    ----------
    vertices_orig : np.ndarray
        Original vertex positions (N, 3).
    faces_orig : np.ndarray
        Original face indices (M, 3).
    vertices_mapped : np.ndarray
        Mapped vertex positions (N, 3).
    faces_mapped : np.ndarray
        Mapped face indices (M, 3).
    
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_valid, report) where report contains validation details.
    """
    report: Dict[str, Any] = {
        "n_vertices_orig": len(vertices_orig),
        "n_faces_orig": len(faces_orig),
        "n_vertices_mapped": len(vertices_mapped),
        "n_faces_mapped": len(faces_mapped),
        "errors": [],
    }
    
    # Check vertex count preserved
    if len(vertices_mapped) != len(vertices_orig):
        report["errors"].append(
            f"Vertex count mismatch: {len(vertices_orig)} -> {len(vertices_mapped)}"
        )
    
    # Check face count preserved
    if len(faces_mapped) != len(faces_orig):
        report["errors"].append(
            f"Face count mismatch: {len(faces_orig)} -> {len(faces_mapped)}"
        )
    
    # Check face values identical
    if not np.array_equal(faces_orig, faces_mapped):
        report["errors"].append("Face array values differ (topology changed)")
    
    # Check all face indices valid
    n_vertices = len(vertices_mapped)
    invalid_indices = []
    for i, face in enumerate(faces_mapped):
        for j, idx in enumerate(face):
            if idx < 0 or idx >= n_vertices:
                invalid_indices.append((i, j, idx))
    
    if invalid_indices:
        report["errors"].append(
            f"Invalid face indices found: {len(invalid_indices)} bad references"
        )
        report["invalid_indices"] = invalid_indices
    
    is_valid = len(report["errors"]) == 0
    report["is_valid"] = is_valid
    
    return is_valid, report


def verify_normal_orientation_preserved(
    vertices_orig: np.ndarray,
    faces: np.ndarray,
    vertices_mapped: np.ndarray,
    dot_product_threshold: float = 0.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Verify that face normals still point outward (not inverted).
    
    Parameters
    ----------
    vertices_orig : np.ndarray
        Original vertex positions (N, 3).
    faces : np.ndarray
        Face indices (M, 3) - same for both original and mapped.
    vertices_mapped : np.ndarray
        Mapped vertex positions on sphere (N, 3).
    dot_product_threshold : float
        Minimum dot product between original and mapped normals to be considered
        correctly oriented. Default 0.0 means normals should point in same hemisphere.
    
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (is_valid, report) where report contains orientation details.
    """
    report: Dict[str, Any] = {
        "normals_oriented_correctly": True,
        "n_flipped_normals": 0,
        "flipped_face_ids": [],
        "errors": [],
    }
    
    # Compute normals for original and mapped meshes
    normals_orig = _compute_face_normals(vertices_orig, faces)
    normals_mapped = _compute_face_normals(vertices_mapped, faces)
    
    # Normalize normals
    norms_orig = np.linalg.norm(normals_orig, axis=1, keepdims=True)
    norms_orig = np.where(norms_orig > 1e-12, norms_orig, 1.0)
    normals_orig_normalized = normals_orig / norms_orig
    
    norms_mapped = np.linalg.norm(normals_mapped, axis=1, keepdims=True)
    norms_mapped = np.where(norms_mapped > 1e-12, norms_mapped, 1.0)
    normals_mapped_normalized = normals_mapped / norms_mapped
    
    # Compute dot products
    dot_products = np.sum(normals_orig_normalized * normals_mapped_normalized, axis=1)
    
    # Identify flipped normals
    flipped_mask = dot_products < dot_product_threshold
    flipped_face_ids = np.where(flipped_mask)[0].tolist()
    
    if len(flipped_face_ids) > 0:
        report["normals_oriented_correctly"] = False
        report["n_flipped_normals"] = len(flipped_face_ids)
        report["flipped_face_ids"] = flipped_face_ids
        report["errors"].append(
            f"Found {len(flipped_face_ids)} faces with flipped normals"
        )
    
    # Check if normals point radially outward on sphere (positive dot with position)
    face_centers = (
        vertices_mapped[faces[:, 0]]
        + vertices_mapped[faces[:, 1]]
        + vertices_mapped[faces[:, 2]]
    ) / 3.0
    
    radial_dots = np.sum(normals_mapped_normalized * face_centers, axis=1)
    inward_mask = radial_dots < 0.0
    inward_face_ids = np.where(inward_mask)[0].tolist()
    
    if len(inward_face_ids) > 0:
        report["all_normals_radial"] = False
        report["n_inward_normals"] = len(inward_face_ids)
        report["inward_face_ids"] = inward_face_ids
        report["errors"].append(
            f"Found {len(inward_face_ids)} faces with inward-pointing normals"
        )
    else:
        report["all_normals_radial"] = True
    
    is_valid = len(report["errors"]) == 0
    report["is_valid"] = is_valid
    
    return is_valid, report



def compute_spherical_parametrization(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str = "flash",
    cem_eps: float = 1e-6,
    cem_max_iters: int = 100,
    cem_verbose: bool = False,
    verify: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute spherical parametrization of a mesh.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions (N, 3).
    faces : np.ndarray
        Face indices (M, 3).
    method : str
        Parametrization method: 'flash' or 'cem'.
    cem_eps : float
        CEM convergence tolerance.
    cem_max_iters : int
        CEM maximum iterations.
    cem_verbose : bool
        CEM verbose output.
    verify : bool
        If True, validate topology and normal orientation after parametrization.
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (sphere_vertices, metadata).
    """
    vertices_orig = np.asarray(vertices, dtype=np.float64).copy()
    faces_orig = np.asarray(faces, dtype=np.int32).copy()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if method == "flash":
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        sphere_vertices = flash_map(mesh)
        meta: Dict[str, Any] = {"method": "flash"}
    elif method == "cem":
        mesh_surf = MeshFactory.make_mesh("surf", vertices, faces)
        stretch = stretch_parametrization(mesh_surf, eps=cem_eps, max_iters=cem_max_iters, verbose=cem_verbose)
        sphere_vertices = stretch.convert_mesh().get_vertices_collection()
        meta = {
            "method": "cem",
            "eps": float(cem_eps),
            "max_iters": int(cem_max_iters),
            "verbose": bool(cem_verbose),
        }
    else:
        raise ValueError("method must be one of: 'flash', 'cem'")

    norms = np.linalg.norm(sphere_vertices, axis=1)
    meta.update(
        {
            "n_vertices": int(sphere_vertices.shape[0]),
            "n_faces": int(faces.shape[0]),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
        }
    )
    
    # Verify topology and normal orientation
    if verify:
        topology_valid, topology_report = verify_topology_preserved(
            vertices_orig, faces_orig, sphere_vertices, faces
        )
        meta["topology_valid"] = topology_valid
        meta["topology_report"] = topology_report
        
        orientation_valid, orientation_report = verify_normal_orientation_preserved(
            vertices_orig, faces, sphere_vertices
        )
        meta["orientation_valid"] = orientation_valid
        meta["orientation_report"] = orientation_report
    
    return sphere_vertices, meta

