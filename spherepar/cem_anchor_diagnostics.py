"""Topology and collapse diagnostics for Algorithm 4.1's anchor face."""
from __future__ import annotations

from collections import deque
import time
from typing import Any, Sequence

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from spherepar.mesh import Face, MeshSurf
from spherepar.idt_remesh import connectivity_hash
from spherepar.parametrization_validation import (
    SPHERE_MIN_AREA_RELATIVE_TO_MEDIAN,
    SPHERE_ORIENTATION_TOLERANCE,
    collapsed_face_geometry,
)


_ECCENTRICITY_CACHE: dict[str, dict[int, tuple[int, float]]] = {}
_ECCENTRICITY_CACHE_ENABLED = True
_ECCENTRICITY_TOTALS = {"hits": 0, "misses": 0, "compute_time": 0.0, "lookup_time": 0.0}


def reset_eccentricity_cache(*, enabled: bool = True) -> None:
    """Clear the process-local cache and select whether lookups may reuse it."""
    global _ECCENTRICITY_CACHE_ENABLED
    _ECCENTRICITY_CACHE.clear()
    _ECCENTRICITY_CACHE_ENABLED = bool(enabled)
    _ECCENTRICITY_TOTALS.update(hits=0, misses=0, compute_time=0.0, lookup_time=0.0)


def eccentricity_cache_statistics() -> dict[str, Any]:
    requests = _ECCENTRICITY_TOTALS["hits"] + _ECCENTRICITY_TOTALS["misses"]
    return {
        "enabled": _ECCENTRICITY_CACHE_ENABLED,
        "connectivity_count": len(_ECCENTRICITY_CACHE),
        **_ECCENTRICITY_TOTALS,
        "hit_rate": float(_ECCENTRICITY_TOTALS["hits"] / requests) if requests else 0.0,
    }


# Explicit aliases make the benchmarking/test hooks discoverable without
# coupling callers to the implementation's private cache name.
reset_anchor_eccentricity_cache = reset_eccentricity_cache
get_anchor_eccentricity_cache_statistics = eccentricity_cache_statistics


def candidate_vertex_eccentricities(
    mesh: MeshSurf,
    candidate_vertex_ids: Sequence[int],
) -> tuple[dict[int, int], dict[str, Any]]:
    """Return eccentricities with lazy, connectivity-keyed source caching."""
    adjacency = build_vertex_adjacency(mesh)
    key = connectivity_hash(len(adjacency), mesh.get_faces_collection())
    cache = _ECCENTRICITY_CACHE.setdefault(key, {}) if _ECCENTRICITY_CACHE_ENABLED else {}
    hits = misses = 0
    compute_time = lookup_time = avoided = 0.0
    values: dict[int, int] = {}
    for source in sorted({int(value) for value in candidate_vertex_ids}):
        lookup_start = time.perf_counter()
        cached = cache.get(source)
        lookup_time += time.perf_counter() - lookup_start
        if cached is not None:
            hits += 1
            values[source] = int(cached[0])
            avoided += float(cached[1])
            continue
        misses += 1
        compute_start = time.perf_counter()
        eccentricity = int(np.max(bfs_hop_distances(adjacency, source)))
        elapsed = time.perf_counter() - compute_start
        compute_time += elapsed
        values[source] = eccentricity
        if _ECCENTRICITY_CACHE_ENABLED:
            cache[source] = (eccentricity, elapsed)
    _ECCENTRICITY_TOTALS["hits"] += hits
    _ECCENTRICITY_TOTALS["misses"] += misses
    _ECCENTRICITY_TOTALS["compute_time"] += compute_time
    _ECCENTRICITY_TOTALS["lookup_time"] += lookup_time
    requests = hits + misses
    return values, {
        "enabled": _ECCENTRICITY_CACHE_ENABLED,
        "connectivity_hash": key,
        "hits": int(hits),
        "misses": int(misses),
        "hit_rate": float(hits / requests) if requests else 0.0,
        "compute_time_seconds": float(compute_time),
        "lookup_time_seconds": float(lookup_time),
        "estimated_bfs_time_avoided_seconds": float(avoided),
    }


def build_vertex_adjacency(mesh: MeshSurf) -> list[list[int]]:
    """Build one unweighted adjacency list from ``Mesh.edges``."""
    adjacency: list[list[int]] = [[] for _ in range(len(mesh.vertices))]
    for u, v in mesh.edges:
        u_id, v_id = int(u), int(v)
        adjacency[u_id].append(v_id)
        adjacency[v_id].append(u_id)
    return adjacency


def bfs_hop_distances(adjacency: Sequence[Sequence[int]], source: int) -> np.ndarray:
    """Return unweighted shortest-path distances from one source vertex."""
    if source < 0 or source >= len(adjacency):
        raise ValueError(f"anchor vertex {source} is outside the mesh")
    distances = np.full(len(adjacency), -1, dtype=np.int64)
    distances[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        vertex = queue.popleft()
        next_distance = distances[vertex] + 1
        for neighbor in adjacency[vertex]:
            if distances[neighbor] < 0:
                distances[neighbor] = next_distance
                queue.append(int(neighbor))
    unreachable = np.flatnonzero(distances < 0)
    if len(unreachable):
        preview = unreachable[:10].tolist()
        suffix = " ..." if len(unreachable) > len(preview) else ""
        raise ValueError(
            f"mesh has {len(unreachable)} vertex/vertices unreachable from anchor "
            f"{source}: {preview}{suffix}"
        )
    return distances


def collect_anchor_geometry(
    mesh: MeshSurf,
    anchor_face: Face,
    *,
    anchor_strategy: str = "regular",
    anchor_regularity_percentile: float = 10.0,
) -> dict[str, Any]:
    """Collect JSON-safe geometry and minimum hop distances for an anchor."""
    anchor_ids = [int(anchor_face.u.id), int(anchor_face.v.id), int(anchor_face.w.id)]
    anchor_positions = np.asarray(
        [anchor_face.u.pos, anchor_face.v.pos, anchor_face.w.pos], dtype=np.float64
    )
    anchor_edge_lengths = [
        float(np.linalg.norm(anchor_positions[0] - anchor_positions[1])),
        float(np.linalg.norm(anchor_positions[1] - anchor_positions[2])),
        float(np.linalg.norm(anchor_positions[2] - anchor_positions[0])),
    ]
    mesh_edge_lengths = np.asarray(
        [np.linalg.norm(edge.u.pos - edge.v.pos) for edge in mesh.edges.values()],
        dtype=np.float64,
    )
    adjacency = build_vertex_adjacency(mesh)
    per_anchor = [bfs_hop_distances(adjacency, anchor_id) for anchor_id in anchor_ids]
    minimum_distances = np.min(np.stack(per_anchor, axis=0), axis=0)
    result = {
        "strategy": anchor_strategy,
        "regularity_percentile": float(anchor_regularity_percentile),
        "vertex_ids": anchor_ids,
        "positions": anchor_positions.tolist(),
        "regularity": float(anchor_face.regularity()),
        "scale_normalized_regularity": float(anchor_face.scale_normalized_regularity()),
        "vertex_eccentricities": [int(np.max(distances)) for distances in per_anchor],
        "edge_lengths": anchor_edge_lengths,
        "mesh_edge_length_mean": float(np.mean(mesh_edge_lengths)),
        "mesh_edge_length_median": float(np.median(mesh_edge_lengths)),
        "minimum_hop_distances": minimum_distances.tolist(),
    }
    cache_diagnostics = getattr(mesh, "anchor_cache_diagnostics", None)
    if cache_diagnostics is not None:
        result["eccentricity_cache"] = cache_diagnostics
    return result


def _group_summary(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "status": "empty group"}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "status": "ok",
    }


def _undefined_statistic(status: str) -> dict[str, Any]:
    return {"statistic": None, "p_value": None, "status": status}


def _spearman_statistic(x: np.ndarray, y: np.ndarray, label: str) -> dict[str, Any]:
    if len(x) < 2:
        return _undefined_statistic(f"undefined: fewer than two vertices for {label}")
    if np.unique(x).size < 2:
        return _undefined_statistic("undefined: anchor distances are constant")
    if np.unique(y).size < 2:
        return _undefined_statistic(f"undefined: {label} is constant")
    result = spearmanr(x, y)
    statistic = result.statistic if hasattr(result, "statistic") else result.correlation
    return {
        "statistic": float(statistic),
        "p_value": float(result.pvalue),
        "status": "ok",
    }


def compute_anchor_collapse_diagnostics(
    mesh_vertices: np.ndarray,
    faces: np.ndarray,
    sphere_vertices: np.ndarray,
    minimum_hop_distances: Sequence[int],
    *,
    sphere_stage: str,
    relative_threshold: float = SPHERE_MIN_AREA_RELATIVE_TO_MEDIAN,
    absolute_twice_area_threshold: float = SPHERE_ORIENTATION_TOLERANCE,
) -> dict[str, Any]:
    """Measure collapse severity and its association with anchor distance."""
    mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    sphere_vertices = np.asarray(sphere_vertices, dtype=np.float64)
    distances = np.asarray(minimum_hop_distances, dtype=np.int64)
    if distances.shape != (len(mesh_vertices),):
        raise ValueError("minimum anchor-hop distances must contain one value per vertex")
    if np.any(distances < 0):
        raise ValueError("minimum anchor-hop distances contain unreachable vertices")

    _, sphere_twice_areas, collapsed_face_ids, threshold = collapsed_face_geometry(
        sphere_vertices,
        faces,
        relative_threshold=relative_threshold,
        absolute_twice_area_threshold=absolute_twice_area_threshold,
    )
    mesh_triangles = mesh_vertices[faces]
    mesh_twice_areas = np.linalg.norm(
        np.cross(
            mesh_triangles[:, 1] - mesh_triangles[:, 0],
            mesh_triangles[:, 2] - mesh_triangles[:, 0],
        ),
        axis=1,
    )
    if np.any(mesh_twice_areas <= 0.0):
        raise ValueError("original mesh contains a zero-area face")

    collapsed_mask = np.zeros(len(mesh_vertices), dtype=bool)
    if len(collapsed_face_ids):
        collapsed_mask[np.unique(faces[collapsed_face_ids].ravel())] = True

    # Compute in log space to keep zero-area spherical faces finite.
    sphere_area_safe = np.maximum(sphere_twice_areas, np.finfo(np.float64).tiny)
    face_severity = np.log10(mesh_twice_areas) - np.log10(sphere_area_safe)
    vertex_severity = np.full(len(mesh_vertices), -np.inf, dtype=np.float64)
    for corner in range(3):
        np.maximum.at(vertex_severity, faces[:, corner], face_severity)
    if not np.isfinite(vertex_severity).all():
        raise ValueError("one or more vertices have no incident face")

    collapsed_distances = distances[collapsed_mask]
    non_collapsed_distances = distances[~collapsed_mask]
    if len(collapsed_distances) and len(non_collapsed_distances):
        mann_whitney = mannwhitneyu(
            collapsed_distances,
            non_collapsed_distances,
            alternative="two-sided",
            method="asymptotic",
        )
        mann_whitney_result = {
            "statistic": float(mann_whitney.statistic),
            "p_value": float(mann_whitney.pvalue),
            "status": "ok",
        }
    else:
        mann_whitney_result = _undefined_statistic(
            "undefined: collapsed and non-collapsed groups must both be non-empty"
        )

    return {
        "analyzed_sphere_stage": sphere_stage,
        "collapse_threshold": {
            "relative_to_median_twice_area": float(relative_threshold),
            "absolute_twice_area": float(absolute_twice_area_threshold),
            "median_twice_area": float(np.median(sphere_twice_areas)),
            "effective_twice_area": float(threshold),
        },
        "collapsed_face_count": int(len(collapsed_face_ids)),
        "collapsed_face_ids": collapsed_face_ids.tolist(),
        "collapsed_vertex_count": int(np.count_nonzero(collapsed_mask)),
        "non_collapsed_vertex_count": int(np.count_nonzero(~collapsed_mask)),
        "collapsed_vertex_ids": np.flatnonzero(collapsed_mask).tolist(),
        "distance_summary": {
            "collapsed": _group_summary(collapsed_distances),
            "non_collapsed": _group_summary(non_collapsed_distances),
        },
        "collapse_severity_log10_area_ratio": vertex_severity.tolist(),
        "statistics": {
            "mann_whitney_u": mann_whitney_result,
            "spearman_collapsed_flag": _spearman_statistic(
                distances.astype(np.float64), collapsed_mask.astype(np.int8), "collapsed flag"
            ),
            "spearman_collapse_severity": _spearman_statistic(
                distances.astype(np.float64), vertex_severity, "collapse severity"
            ),
        },
    }
