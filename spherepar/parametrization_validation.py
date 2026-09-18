"""Geometry diagnostics shared by spherical parametrization algorithms."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.spatial import cKDTree


SPHERE_UNIT_NORM_TOLERANCE = 1e-5
SPHERE_MIN_VERTEX_SEPARATION = 1e-5
SPHERE_MIN_AREA_RELATIVE_TO_MEDIAN = 1e-4
SPHERE_ORIENTATION_TOLERANCE = 1e-12


def validate_sphere_parameterization(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    sphere_vertices: np.ndarray,
    sphere_faces: np.ndarray,
) -> Dict[str, Any]:
    """Measure collapsed and folded geometry in a vertex-aligned sphere map."""
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    mesh_faces = np.asarray(mesh_faces, dtype=np.int64)
    sphere_vertices = np.asarray(sphere_vertices, dtype=np.float64)
    sphere_faces = np.asarray(sphere_faces, dtype=np.int64)
    report: Dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "thresholds": {
            "unit_norm_tolerance": SPHERE_UNIT_NORM_TOLERANCE,
            "min_vertex_separation": SPHERE_MIN_VERTEX_SEPARATION,
            "min_area_relative_to_median": SPHERE_MIN_AREA_RELATIVE_TO_MEDIAN,
            "orientation_tolerance": SPHERE_ORIENTATION_TOLERANCE,
        },
        "n_mesh_vertices": int(len(mesh_vertices)),
        "n_sphere_vertices": int(len(sphere_vertices)),
        "n_mesh_faces": int(len(mesh_faces)),
        "n_sphere_faces": int(len(sphere_faces)),
    }

    if sphere_vertices.shape != mesh_vertices.shape:
        report["errors"].append("sphere vertex array does not match mesh vertex array")
    if sphere_faces.shape != mesh_faces.shape or not np.array_equal(mesh_faces, sphere_faces):
        report["errors"].append("sphere faces do not match mesh face ordering and winding")
    if sphere_vertices.ndim != 2 or sphere_vertices.shape[1:] != (3,) or len(sphere_vertices) < 4:
        report["errors"].append("sphere vertices must have shape [V, 3] with at least four vertices")
    if not np.isfinite(sphere_vertices).all():
        report["errors"].append("sphere vertices contain non-finite coordinates")

    geometry_compatible = (
        sphere_vertices.ndim == 2
        and sphere_vertices.shape[1:] == (3,)
        and len(sphere_vertices) >= 4
        and np.isfinite(sphere_vertices).all()
        and sphere_faces.shape == mesh_faces.shape
        and np.array_equal(mesh_faces, sphere_faces)
    )
    if geometry_compatible:
        norms = np.linalg.norm(sphere_vertices, axis=1)
        max_unit_norm_error = float(np.max(np.abs(norms - 1.0)))
        report["max_unit_norm_error"] = max_unit_norm_error
        if np.any(norms == 0.0) or max_unit_norm_error > SPHERE_UNIT_NORM_TOLERANCE:
            report["errors"].append("sphere vertices are not unit length")

        nonzero_norms = np.where(norms > 0.0, norms, 1.0)
        unit_vertices = sphere_vertices / nonzero_norms[:, None]
        nearest = cKDTree(unit_vertices).query(unit_vertices, k=2)[0][:, 1]
        min_separation = float(nearest.min())
        near_duplicate_count = int(np.count_nonzero(nearest <= SPHERE_MIN_VERTEX_SEPARATION))
        report["min_vertex_separation"] = min_separation
        report["near_duplicate_vertex_count"] = near_duplicate_count
        if min_separation <= SPHERE_MIN_VERTEX_SEPARATION:
            report["errors"].append("sphere has collapsed or near-duplicate vertices")

        triangles = unit_vertices[mesh_faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        median_twice_area = float(np.median(twice_area))
        min_twice_area = float(twice_area.min())
        area_threshold = max(
            SPHERE_ORIENTATION_TOLERANCE,
            SPHERE_MIN_AREA_RELATIVE_TO_MEDIAN * median_twice_area,
        )
        degenerate_count = int(np.count_nonzero(twice_area <= area_threshold))
        report.update({
            "min_twice_area": min_twice_area,
            "median_twice_area": median_twice_area,
            "area_threshold": float(area_threshold),
            "degenerate_face_count": degenerate_count,
        })
        if median_twice_area <= 0.0 or degenerate_count:
            report["errors"].append("sphere has degenerate or near-collapsed triangles")

        signed_orientation = np.einsum("ij,ij->i", cross, triangles.mean(axis=1))
        near_zero_orientation_count = int(
            np.count_nonzero(np.abs(signed_orientation) <= SPHERE_ORIENTATION_TOLERANCE)
        )
        outward_count = int(np.count_nonzero(signed_orientation > SPHERE_ORIENTATION_TOLERANCE))
        inward_count = int(np.count_nonzero(signed_orientation < -SPHERE_ORIENTATION_TOLERANCE))
        report.update({
            "outward_face_count": outward_count,
            "inward_face_count": inward_count,
            "near_zero_orientation_face_count": near_zero_orientation_count,
            "orientation": (
                "mixed" if outward_count and inward_count
                else "outward" if outward_count
                else "inward" if inward_count
                else "undetermined"
            ),
        })
        if near_zero_orientation_count or (outward_count and inward_count):
            report["errors"].append("sphere has folded or inconsistently oriented faces")

    report["is_valid"] = not report["errors"]
    return report
