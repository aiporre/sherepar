"""Deterministic fixed-vertex intrinsic-Delaunay edge flipping."""
from __future__ import annotations

import hashlib
import heapq
import math
from collections import defaultdict
from typing import Any

import numpy as np


def connectivity_hash(vertex_count: int, faces: np.ndarray) -> str:
    """Hash vertex count and canonical undirected adjacency."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = set()
    for a, b, c in faces.tolist():
        edges.update((tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))))
    payload = np.asarray(sorted(edges), dtype="<i8").tobytes()
    digest = hashlib.sha256()
    digest.update(int(vertex_count).to_bytes(8, "little", signed=False))
    digest.update(payload)
    return digest.hexdigest()


def _edge_faces(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces.tolist()):
        a, b, c = (int(value) for value in face)
        for edge in (tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))):
            result[edge].append(face_id)
    return dict(result)


def _angle(vertex: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    u = first - vertex
    v = second - vertex
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("zero-length edge in IDT input")
    return float(np.arccos(np.clip(np.dot(u, v) / denominator, -1.0, 1.0)))


def _opposite_angle_sum(
    vertices: np.ndarray, faces: np.ndarray, edge: tuple[int, int], incident: list[int]
) -> float:
    if len(incident) != 2:
        raise ValueError(f"edge {edge} is not interior (incident faces={len(incident)})")
    u, v = edge
    opposites = [next(int(x) for x in faces[index] if int(x) not in edge) for index in incident]
    return sum(_angle(vertices[opposite], vertices[u], vertices[v]) for opposite in opposites)


def _oriented_edge(face: np.ndarray, edge: tuple[int, int]) -> tuple[int, int, int]:
    values = [int(value) for value in face]
    u0, v0 = edge
    for index in range(3):
        u, v = values[index], values[(index + 1) % 3]
        if {u, v} == {u0, v0}:
            return u, v, values[(index + 2) % 3]
    raise ValueError(f"face {values} does not contain edge {edge}")


def _twice_area(vertices: np.ndarray, face: np.ndarray) -> float:
    triangle = vertices[np.asarray(face, dtype=np.int64)]
    return float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))


def _flip_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    edge: tuple[int, int],
    incident: list[int],
    edge_map: dict[tuple[int, int], list[int]],
    area_tolerance: float,
) -> tuple[np.ndarray, str | None]:
    first_id, second_id = sorted(incident)
    u, v, a = _oriented_edge(faces[first_id], edge)
    second_u, second_v, b = _oriented_edge(faces[second_id], edge)
    if (second_u, second_v) != (v, u):
        return faces, "orientation_breaking"
    diagonal = tuple(sorted((a, b)))
    if a == b or diagonal in edge_map:
        return faces, "duplicate_diagonal"

    replacement_first = np.asarray((a, b, v), dtype=faces.dtype)
    replacement_second = np.asarray((b, a, u), dtype=faces.dtype)
    if (
        _twice_area(vertices, replacement_first) <= area_tolerance
        or _twice_area(vertices, replacement_second) <= area_tolerance
    ):
        return faces, "degenerate_face"

    faces[first_id] = replacement_first
    faces[second_id] = replacement_second
    return faces, None


def intrinsic_delaunay_remesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float = 1e-12,
    max_flips: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Flip non-Delaunay interior edges without moving or adding vertices.

    The smallest violating edge is handled first after every flip.  Refusals
    are explicit errors because silently leaving a violating edge would make
    the advertised IDT preprocessing ambiguous.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    original_vertices = vertices.copy()
    result = np.asarray(faces, dtype=np.int32).copy()
    if result.ndim != 2 or result.shape[1:] != (3,):
        raise ValueError("IDT faces must have shape [F, 3]")
    if not np.isfinite(vertices).all():
        raise ValueError("IDT vertices contain non-finite coordinates")
    if tolerance < 0.0:
        raise ValueError("IDT tolerance must be non-negative")
    if any(_twice_area(vertices, face) <= np.finfo(np.float64).eps for face in result):
        raise ValueError("IDT input contains a degenerate face")

    initial_edges = _edge_faces(result)
    nonmanifold = sorted(edge for edge, incident in initial_edges.items() if len(incident) > 2)
    boundary = sorted(edge for edge, incident in initial_edges.items() if len(incident) == 1)
    if nonmanifold:
        raise ValueError(f"IDT input contains non-manifold edge {nonmanifold[0]}")
    cap = 10 * len(initial_edges) if max_flips is None else int(max_flips)
    if cap < 0:
        raise ValueError("IDT max_flips must be non-negative")

    edge_map = _edge_faces(result)
    queue: list[tuple[int, int]] = []
    for edge in sorted(edge_map):
        incident = edge_map[edge]
        if len(incident) == 2 and _opposite_angle_sum(vertices, result, edge, incident) > math.pi + tolerance:
            heapq.heappush(queue, edge)

    flips = 0
    while queue:
        edge = heapq.heappop(queue)
        incident = edge_map.get(edge, [])
        if len(incident) != 2:
            continue
        angle_sum = _opposite_angle_sum(vertices, result, edge, incident)
        if angle_sum <= math.pi + tolerance:
            continue
        if flips >= cap:
            raise RuntimeError(
                f"IDT did not converge within safety cap ({cap} flips); next edge={edge}"
            )
        affected_face_ids = sorted(incident)
        old_edges = {
            tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
            for face in result[affected_face_ids]
            for index in range(3)
        }
        updated, refusal = _flip_faces(
            vertices,
            result,
            edge,
            incident,
            edge_map,
            np.finfo(np.float64).eps,
        )
        if refusal is not None:
            raise RuntimeError(
                f"IDT cannot flip non-Delaunay edge {edge} "
                f"(opposite-angle sum={angle_sum:.16g}): {refusal}"
            )
        result = updated
        flips += 1
        for old_edge in old_edges:
            old_incident = edge_map.get(old_edge, [])
            retained = [face_id for face_id in old_incident if face_id not in affected_face_ids]
            if retained:
                edge_map[old_edge] = retained
            else:
                edge_map.pop(old_edge, None)
        affected_edges = set()
        for face_id in affected_face_ids:
            face = result[face_id]
            for index in range(3):
                new_edge = tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                edge_map.setdefault(new_edge, []).append(face_id)
                affected_edges.add(new_edge)
        for affected_edge in sorted(affected_edges):
            if len(edge_map[affected_edge]) == 2:
                heapq.heappush(queue, affected_edge)

    if len(vertices) != len(original_vertices) or not np.array_equal(vertices, original_vertices):
        raise AssertionError("IDT changed the vertex array")
    return result, {
        "enabled": True,
        "flip_count": int(flips),
        "converged": True,
        "safety_cap": int(cap),
        "boundary_edge_count": int(len(boundary)),
        "topology_hash_before": connectivity_hash(len(vertices), faces),
        "topology_hash_after": connectivity_hash(len(vertices), result),
        "vertex_count_before": int(len(vertices)),
        "vertex_count_after": int(len(vertices)),
        "vertices_exactly_unchanged": True,
    }
