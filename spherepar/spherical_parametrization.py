"""Compatibility wrapper for spherical parametrization utilities."""
from __future__ import annotations

from typing import Tuple, Dict, Any, Callable, List, Optional, Sequence

import numpy as np
import trimesh

from spherepar.cem_parametrization import (
    _cotangent_weight_diagnostics,
    _face_angle_diagnostics,
    stretch_parametrization,
)
from spherepar.idt_remesh import connectivity_hash, intrinsic_delaunay_remesh
from spherepar.flash_parametrization import (  # noqa: F401
    flash_map,
    flash_map_with_diagnostics,
    load_mesh_with_trimesh,
)
from spherepar.spheremap_parametrization import spheremap_map_with_diagnostics
from spherepar.mesh import MeshFactory
from spherepar.mobius_centering import center_spherical_mesh
from spherepar.parametrization_validation import (
    SPHERE_MIN_AREA_RELATIVE_TO_MEDIAN,
    SPHERE_MIN_VERTEX_SEPARATION,
    SPHERE_ORIENTATION_TOLERANCE,
    SPHERE_UNIT_NORM_TOLERANCE,
    validate_sphere_parameterization,
)


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
    cem_radius: float = 1.2,
    verify: bool = True,
    mobius_center: bool = False,
    cem_input_diagnostics_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    anchor_diagnostics: bool = False,
    anchor_strategy: str = "regular",
    anchor_regularity_percentile: float = 10.0,
    use_idt_remesh: bool = False,
    adaptive_radius: bool = False,
    cem_radius_candidates: Sequence[float] = (1.1, 1.3, 1.4, 1.5),
    reject_retry: bool = False,
    cem_max_attempts: int = 5,
    cem_max_collapsed_faces: int = 0,
    spheremap_binary: Optional[str] = None,
    spheremap_repository: Optional[str] = None,
    spheremap_auto_build: bool = False,
    spheremap_iters: int = 25,
    spheremap_step_size: float = 1.0,
    spheremap_threads: int = 4,
    spheremap_no_center: bool = False,
    spheremap_degree: Optional[int] = 4,
    spheremap_a_steps: Optional[int] = 10,
    spheremap_a_step_size: Optional[float] = 0.05,
    spheremap_poincare_max_norm: Optional[float] = 2.0,
    spheremap_c2i: Optional[int] = 0,
    spheremap_gss_tolerance: Optional[float] = 1e-6,
    spheremap_lump: bool = False,
    spheremap_verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute spherical parametrization of a mesh.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions (N, 3).
    faces : np.ndarray
        Face indices (M, 3).
    method : str
        Parametrization method: 'flash', 'cem', or 'spheremap'.
    cem_eps : float
        CEM convergence tolerance.
    cem_max_iters : int
        CEM maximum iterations.
    cem_verbose : bool
        CEM verbose output.
    cem_radius : float
        CEM stereographic partition radius.
    verify : bool
        If True, validate topology and normal orientation after parametrization.
    mobius_center : bool
        If True, apply area-weighted Möbius centering after CEM or SphereMap.
    cem_input_diagnostics_callback : callable, optional
        Receives CEM input-quality diagnostics immediately before Algorithm 4.1.
    anchor_diagnostics : bool
        If True, measure CEM collapse against Algorithm 4.1 anchor-hop distance.
    anchor_strategy : str
        CEM Algorithm 4.1 selector: ``regular`` or ``central_regular``.
    anchor_regularity_percentile : float
        Inclusive regularity percentile used by ``central_regular``.
    use_idt_remesh, adaptive_radius, reject_retry : bool
        Opt-in Phase 2 connectivity preprocessing, radius search, and
        acceptance policy. Radius candidates and collapse/attempt limits are
        recorded in the returned metadata; sphere faces remain original.
    
    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        (sphere_vertices, metadata).
    """
    vertices_orig = np.asarray(vertices, dtype=np.float64).copy()
    faces_orig = np.asarray(faces, dtype=np.int32).copy()
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if mobius_center and method not in ("cem", "spheremap"):
        raise ValueError("mobius_center is supported only with method='cem' or 'spheremap'")
    if anchor_diagnostics and method != "cem":
        raise ValueError("anchor_diagnostics is supported only with method='cem'")
    if (use_idt_remesh or adaptive_radius or reject_retry) and method != "cem":
        raise ValueError("CEM Phase 2 options are supported only with method='cem'")

    if method == "flash":
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        sphere_vertices, flash_diagnostics = flash_map_with_diagnostics(mesh)
        meta: Dict[str, Any] = {
            "method": "flash",
            "flash_diagnostics": flash_diagnostics,
            "success": bool(flash_diagnostics.get("success", False)),
        }
    elif method == "spheremap":
        sphere_vertices, spheremap_diagnostics = spheremap_map_with_diagnostics(
            vertices,
            faces,
            binary=spheremap_binary,
            repository=spheremap_repository,
            auto_build=spheremap_auto_build,
            iters=spheremap_iters,
            step_size=spheremap_step_size,
            threads=spheremap_threads,
            no_center=spheremap_no_center,
            degree=spheremap_degree,
            a_steps=spheremap_a_steps,
            a_step_size=spheremap_a_step_size,
            poincare_max_norm=spheremap_poincare_max_norm,
            c2i=spheremap_c2i,
            gss_tolerance=spheremap_gss_tolerance,
            lump=spheremap_lump,
            verbose=spheremap_verbose,
        )
        meta = {
            "method": "spheremap",
            "spheremap_diagnostics": spheremap_diagnostics,
            "spheremap_binary": spheremap_binary,
            "spheremap_repository": spheremap_repository,
            "spheremap_auto_build": bool(spheremap_auto_build),
            "spheremap_iters": int(spheremap_iters),
            "spheremap_step_size": float(spheremap_step_size),
            "spheremap_threads": int(spheremap_threads),
            "spheremap_no_center": bool(spheremap_no_center),
            "spheremap_degree": spheremap_degree,
            "spheremap_a_steps": spheremap_a_steps,
            "spheremap_a_step_size": spheremap_a_step_size,
            "spheremap_poincare_max_norm": spheremap_poincare_max_norm,
            "spheremap_c2i": spheremap_c2i,
            "spheremap_gss_tolerance": spheremap_gss_tolerance,
            "spheremap_lump": bool(spheremap_lump),
            "spheremap_verbose": bool(spheremap_verbose),
            "success": bool(spheremap_diagnostics.get("success", False)),
        }
    elif method == "cem":
        original_mesh = MeshFactory.make_mesh("surf", vertices, faces)
        if use_idt_remesh:
            before_laplacian = original_mesh.get_laplacian_matrix(weight="cotangent").toarray()
            before_cotangent = _cotangent_weight_diagnostics(original_mesh, before_laplacian)
            before_angles = _face_angle_diagnostics(
                original_mesh,
                target_face_count=before_cotangent["affected_triangle_count"],
            )
            remeshed_faces, remeshing = intrinsic_delaunay_remesh(vertices, faces)
            if not np.array_equal(vertices, vertices_orig) or len(vertices) != len(vertices_orig):
                raise AssertionError("IDT preprocessing changed CEM vertices")
            mesh_surf = MeshFactory.make_mesh("surf", vertices, remeshed_faces)
        else:
            remeshed_faces = faces
            mesh_surf = original_mesh
            remeshing = {
                "enabled": False,
                "flip_count": 0,
                "converged": True,
                "topology_hash_before": connectivity_hash(len(vertices), faces),
                "topology_hash_after": connectivity_hash(len(vertices), faces),
                "vertex_count_before": int(len(vertices)),
                "vertex_count_after": int(len(vertices)),
                "vertices_exactly_unchanged": True,
            }
        stretch = stretch_parametrization(
            mesh_surf,
            eps=cem_eps,
            max_iters=cem_max_iters,
            verbose=cem_verbose,
            radius=cem_radius,
            input_diagnostics_callback=cem_input_diagnostics_callback,
            anchor_diagnostics=anchor_diagnostics,
            anchor_strategy=anchor_strategy,
            anchor_regularity_percentile=anchor_regularity_percentile,
            adaptive_radius=adaptive_radius,
            radius_candidates=cem_radius_candidates,
            reject_retry=reject_retry,
            max_attempts=cem_max_attempts,
            max_collapsed_faces=cem_max_collapsed_faces,
            validation_vertices=vertices_orig,
            validation_faces=faces_orig,
        )
        sphere_vertices = stretch.convert_mesh().get_vertices_collection()
        after_angles = stretch.cem_diagnostics["input_mesh_quality"]
        after_cotangent = stretch.cem_diagnostics["cotangent_weights"]
        if not use_idt_remesh:
            before_angles = after_angles
            before_cotangent = after_cotangent
        remeshing.update({
            "before": {
                "input_mesh_quality": before_angles,
                "cotangent_weights": before_cotangent,
            },
            "after": {
                "input_mesh_quality": after_angles,
                "cotangent_weights": after_cotangent,
            },
        })
        meta = {
            "method": "cem",
            "eps": float(cem_eps),
            "max_iters": int(cem_max_iters),
            "verbose": bool(cem_verbose),
            "cem_radius": float(cem_radius),
            "cem_selected_radius": float(stretch.cem_diagnostics["selected_radius"]),
            "use_idt_remesh": bool(use_idt_remesh),
            "adaptive_radius": bool(adaptive_radius),
            "cem_radius_candidates": [float(value) for value in cem_radius_candidates],
            "reject_retry": bool(reject_retry),
            "cem_max_attempts": int(cem_max_attempts),
            "cem_max_collapsed_faces": int(cem_max_collapsed_faces),
            "anchor_strategy": anchor_strategy,
            "anchor_regularity_percentile": float(anchor_regularity_percentile),
            "cem_diagnostics": stretch.cem_diagnostics,
            "remeshing": remeshing,
            "radius_attempts": stretch.cem_diagnostics["radius_attempts"],
            "acceptance": stretch.cem_diagnostics["acceptance"],
            "cache_statistics": (
                (stretch.cem_diagnostics.get("anchor") or {}).get("eccentricity_cache")
            ),
            "mobius_center": bool(mobius_center),
        }
    else:
        raise ValueError("method must be one of: 'flash', 'cem', 'spheremap'")

    if mobius_center:
        sphere_vertices, centering_meta = center_spherical_mesh(
            vertices_orig, faces_orig, sphere_vertices
        )
        meta["mobius_centering"] = centering_meta
        print(
            "Möbius centering: "
            f"centroid {centering_meta['before']['centroid_norm']:.3e} -> "
            f"{centering_meta['after']['centroid_norm']:.3e} "
            f"({centering_meta['iterations']} iteration(s))"
        )

    meta["mobius_center"] = bool(mobius_center)

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
        meta["sphere_validation"] = validate_sphere_parameterization(
            vertices_orig, faces_orig, sphere_vertices, faces
        )
        if method in ("flash", "spheremap"):
            validation = meta["sphere_validation"]
            if not validation.get("is_valid", False):
                meta["success"] = False
                label = "FLASH" if method == "flash" else "SphereMap"
                meta["error"] = label + " sphere validation failed: " + "; ".join(validation.get("errors", []))
            elif not meta.get("success", False):
                diagnostics = (
                    meta.get("flash_diagnostics", {})
                    if method == "flash" else meta.get("spheremap_diagnostics", {})
                )
                meta["error"] = diagnostics.get("error") or f"{method} solver did not produce a validated map"

    if method in ("flash", "spheremap") and not meta.get("success", False) and "error" not in meta:
        diagnostics = (
            meta.get("flash_diagnostics", {})
            if method == "flash" else meta.get("spheremap_diagnostics", {})
        )
        meta["error"] = diagnostics.get("error") or f"{method} solver did not produce a validated map"

    return sphere_vertices, meta
