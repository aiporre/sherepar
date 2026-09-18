"""Area-weighted Möbius centering for vertex-aligned spherical meshes.

This module ports the small part of Misha Kazhdan's ``SphericalGeometry``
implementation used by the sphere-map normalizer.  Triangle masses are fixed
from the input surface; transformed spherical triangle centers are used only
as quadrature points.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


TRANSFORM_KIND = "inversion_sequence_v1"


def _points3(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [..., 3]")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def mobius_transform(points: np.ndarray, center: Sequence[float]) -> np.ndarray:
    """Apply one spherical inversion ``m_c`` to a batch of points.

    The convention is ``m_c(p) = c + (1-|c|^2)(p+c)/|p+c|^2``.
    For points on the unit sphere its inverse is ``m_-c``.
    """
    points = _points3(points, "points")
    center = _points3(np.asarray(center), "center")
    if center.shape != (3,):
        raise ValueError("center must have shape [3]")
    center_norm_sq = float(center @ center)
    if center_norm_sq >= 1.0:
        raise ValueError("Möbius center must lie strictly inside the unit ball")
    shifted = points + center
    denominator = np.einsum("...i,...i->...", shifted, shifted)
    if np.any(denominator <= np.finfo(np.float64).tiny):
        raise ValueError("Möbius transform is singular at p = -center")
    return center + (1.0 - center_norm_sq) * shifted / denominator[..., None]


def inverse_mobius_transform(points: np.ndarray, center: Sequence[float]) -> np.ndarray:
    """Apply the inverse of one ``m_c`` transformation."""
    return mobius_transform(points, -np.asarray(center, dtype=np.float64))


def transform_to_json(centers: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return the canonical, versioned, JSON-safe transform representation."""
    array = np.asarray(centers, dtype=np.float64)
    if array.size == 0:
        array = np.empty((0, 3), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("centers must have shape [K, 3] and be finite")
    if np.any(np.linalg.norm(array, axis=1) >= 1.0):
        raise ValueError("all Möbius centers must lie strictly inside the unit ball")
    return {"kind": TRANSFORM_KIND, "centers": array.tolist()}


def centers_from_transform(transform: Mapping[str, Any] | None) -> np.ndarray:
    """Validate a serialized transform and return its centers as float64."""
    if transform is None:
        return np.empty((0, 3), dtype=np.float64)
    if not isinstance(transform, Mapping) or transform.get("kind") != TRANSFORM_KIND:
        raise ValueError(f"unsupported Möbius transform; expected kind={TRANSFORM_KIND!r}")
    return np.asarray(transform_to_json(transform.get("centers", []))["centers"], dtype=np.float64).reshape(-1, 3)


def apply_mobius_sequence(
    points: np.ndarray,
    transform: Mapping[str, Any] | Sequence[Sequence[float]],
) -> np.ndarray:
    """Apply incremental centers in their stored (forward) order."""
    centers = (centers_from_transform(transform) if isinstance(transform, Mapping)
               else centers_from_transform(transform_to_json(transform)))
    result = _points3(points, "points").copy()
    for center in centers:
        result = mobius_transform(result, center)
    return result


def apply_inverse_mobius_sequence(
    points: np.ndarray,
    transform: Mapping[str, Any] | Sequence[Sequence[float]],
) -> np.ndarray:
    """Apply negated incremental centers in reverse order."""
    centers = (centers_from_transform(transform) if isinstance(transform, Mapping)
               else centers_from_transform(transform_to_json(transform)))
    result = _points3(points, "points").copy()
    for center in centers[::-1]:
        result = inverse_mobius_transform(result, center)
    return result


def original_face_weights(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute normalized Euclidean triangle areas on the original mesh."""
    vertices = _points3(vertices, "vertices")
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("original mesh has zero or non-finite total triangle area")
    return areas / total


def spherical_face_centers(sphere_vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return normalized sums of spherical triangle vertices."""
    sphere_vertices = _points3(sphere_vertices, "sphere_vertices")
    faces = np.asarray(faces, dtype=np.int64)
    sums = sphere_vertices[faces].sum(axis=1)
    norms = np.linalg.norm(sums, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("a spherical face has an undefined normalized vertex-sum center")
    return sums / norms[:, None]


def area_weighted_centroid(
    sphere_vertices: np.ndarray,
    faces: np.ndarray,
    face_weights: np.ndarray,
) -> np.ndarray:
    """Evaluate the fixed-mass spherical centroid used by the reference."""
    weights = np.asarray(face_weights, dtype=np.float64)
    if weights.shape != (len(faces),) or not np.isfinite(weights).all():
        raise ValueError("face_weights must be a finite vector with one entry per face")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("face_weights must sum to one")
    return np.einsum("f,fi->i", weights, spherical_face_centers(sphere_vertices, faces))


def _spherical_triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    unit = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    a, b, c = (unit[faces[:, index]] for index in range(3))
    numerator = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    denominator = 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2.0 * np.arctan2(numerator, denominator)


def centering_diagnostics(
    original_vertices: np.ndarray,
    faces: np.ndarray,
    sphere_vertices: np.ndarray,
    face_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure centroid, normalized spherical area distortion, and bad faces."""
    original_vertices = _points3(original_vertices, "original_vertices")
    sphere_vertices = _points3(sphere_vertices, "sphere_vertices")
    faces = np.asarray(faces, dtype=np.int64)
    weights = original_face_weights(original_vertices, faces) if face_weights is None else np.asarray(face_weights, dtype=np.float64)
    centroid = area_weighted_centroid(sphere_vertices, faces, weights)

    triangles = original_vertices[faces]
    original_areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    spherical_areas = _spherical_triangle_areas(sphere_vertices, faces)
    positive = original_areas > np.finfo(np.float64).eps
    ratios = np.full(len(faces), np.nan, dtype=np.float64)
    if np.any(positive):
        raw = spherical_areas[positive] / original_areas[positive]
        area_scale = float(spherical_areas.sum() / original_areas.sum())
        ratios[positive] = raw / area_scale if area_scale > 0.0 else raw
    finite_ratios = ratios[np.isfinite(ratios)]

    sphere_triangles = sphere_vertices[faces]
    cross = np.cross(sphere_triangles[:, 1] - sphere_triangles[:, 0], sphere_triangles[:, 2] - sphere_triangles[:, 0])
    twice_areas = np.linalg.norm(cross, axis=1)
    degenerate = twice_areas <= 1e-14
    orientation = np.einsum("ij,ij->i", cross, sphere_triangles.mean(axis=1))
    nondegenerate_orientation = orientation[~degenerate]
    if nondegenerate_orientation.size:
        expected_sign = 1.0 if np.count_nonzero(nondegenerate_orientation > 0.0) >= np.count_nonzero(nondegenerate_orientation < 0.0) else -1.0
        flipped_count = int(np.count_nonzero(nondegenerate_orientation * expected_sign < 0.0))
    else:
        flipped_count = 0
    return {
        "centroid": centroid.tolist(),
        "centroid_norm": float(np.linalg.norm(centroid)),
        "normalized_spherical_area_ratio": {
            "min": float(finite_ratios.min()) if finite_ratios.size else None,
            "max": float(finite_ratios.max()) if finite_ratios.size else None,
            "mean": float(finite_ratios.mean()) if finite_ratios.size else None,
        },
        "degenerate_face_count": int(np.count_nonzero(degenerate)),
        "flipped_face_count": flipped_count,
    }


def center_spherical_mesh(
    original_vertices: np.ndarray,
    faces: np.ndarray,
    sphere_vertices: np.ndarray,
    *,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
    max_tangent_step_norm: float = 2.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Center a spherical mesh with Poincaré-damped Gauss–Newton updates.

    Raises ``RuntimeError`` if the requested tolerance is not attained.
    """
    if max_iterations < 0 or tolerance <= 0.0 or max_tangent_step_norm <= 0.0:
        raise ValueError("invalid centering solver parameters")
    faces = np.asarray(faces, dtype=np.int64)
    result = _points3(sphere_vertices, "sphere_vertices").copy()
    norms = np.linalg.norm(result, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-7):
        raise ValueError("Möbius centering requires unit-sphere vertices")
    weights = original_face_weights(original_vertices, faces)
    before = centering_diagnostics(original_vertices, faces, result, weights)
    centers: list[np.ndarray] = []

    converged = False
    for _ in range(max_iterations + 1):
        quadrature = spherical_face_centers(result, faces)
        centroid = np.einsum("f,fi->i", weights, quadrature)
        if float(centroid @ centroid) <= tolerance * tolerance:
            converged = True
            break
        if len(centers) == max_iterations:
            break
        jacobian = 2.0 * np.einsum(
            "f,fij->ij",
            weights,
            np.eye(3)[None, :, :] - quadrature[:, :, None] * quadrature[:, None, :],
        )
        tangent_step = np.linalg.lstsq(jacobian, -centroid, rcond=None)[0]
        step_norm = float(np.linalg.norm(tangent_step))
        if not np.isfinite(step_norm) or step_norm == 0.0:
            raise RuntimeError("Möbius centering Gauss–Newton step is singular")
        damped_norm = np.tanh(min(step_norm, max_tangent_step_norm))
        center = tangent_step * (damped_norm / step_norm)
        result = mobius_transform(result, center)
        centers.append(center)

    after = centering_diagnostics(original_vertices, faces, result, weights)
    if not converged:
        raise RuntimeError(
            "Möbius centering did not converge after "
            f"{max_iterations} iterations (centroid norm={after['centroid_norm']:.3e}, "
            f"tolerance={tolerance:.3e})"
        )
    metadata = {
        "enabled": True,
        "transform": transform_to_json(centers),
        "iterations": len(centers),
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
        "max_tangent_step_norm": float(max_tangent_step_norm),
        "before": before,
        "after": after,
    }
    return result, metadata


# Short aliases useful to downstream callers.
mobius_forward = mobius_transform
mobius_inverse = inverse_mobius_transform
apply_transform = apply_mobius_sequence
apply_inverse_transform = apply_inverse_mobius_sequence

