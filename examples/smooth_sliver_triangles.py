#!/usr/bin/env python3
"""Reduce sliver triangles with deterministic, quality-aware mesh smoothing.

The smoother keeps the input connectivity and vertex count unchanged.  Each
candidate move is a tangential Laplacian move (or a full Laplacian move when
``--allow-normal-motion`` is selected).  A move is accepted only when the
quality of the incident faces improves lexicographically:

1. fewer faces below ``--angle-threshold``;
2. smaller sum of squared angle deficits;
3. larger worst incident minimum angle.

Moves that create a degenerate face or reverse a face normal are rejected.
This makes the script suitable as a preprocessing step before spherical
parameterization without changing face indices.

Example::

    python examples/smooth_sliver_triangles.py \
        input.ply --output smoothed.obj --iterations 10 --angle-threshold 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import trimesh


_EPS = 1e-14


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"no geometry found in {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"{path} is not a triangular surface mesh")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"{path} has no valid 3-D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"{path} must contain at least one triangular face")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{path} contains non-finite vertex coordinates")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def face_minimum_angles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return the smallest angle in every face, in degrees."""

    triangles = vertices[faces]
    a = triangles[:, 1] - triangles[:, 0]
    b = triangles[:, 2] - triangles[:, 1]
    c = triangles[:, 0] - triangles[:, 2]
    lengths = np.column_stack((np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)))
    if np.any(lengths <= _EPS):
        raise ValueError("mesh contains a zero-length triangle edge")

    # At vertex 0 the two adjacent edges are -c and a; cyclically repeat.
    edges = (a, b, c)
    angles = []
    for first, second in ((-c, a), (-a, b), (-b, c)):
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        cosine = np.einsum("ij,ij->i", first, second) / denominator
        angles.append(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return np.rad2deg(np.column_stack(angles).min(axis=1))


def _face_normals_and_twice_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return cross, np.linalg.norm(cross, axis=1)


def _topology_data(faces: np.ndarray, vertex_count: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    neighbors = [set() for _ in range(vertex_count)]
    incident: list[list[int]] = [[] for _ in range(vertex_count)]
    for face_id, (a, b, c) in enumerate(faces):
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
        incident[a].append(face_id)
        incident[b].append(face_id)
        incident[c].append(face_id)
    return (
        [np.asarray(sorted(values), dtype=np.int64) for values in neighbors],
        [np.asarray(values, dtype=np.int64) for values in incident],
    )


def _boundary_vertices(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (min(int(u), int(v)), max(int(u), int(v)))
            counts[edge] = counts.get(edge, 0) + 1
    boundary = np.zeros(vertex_count, dtype=bool)
    for (u, v), count in counts.items():
        if count == 1:
            boundary[u] = boundary[v] = True
    return boundary


def _quality_score(angles: np.ndarray, threshold: float) -> tuple[int, float, float]:
    deficits = np.maximum(threshold - angles, 0.0)
    return (int(np.count_nonzero(deficits > 0.0)), float(np.dot(deficits, deficits)), -float(angles.min()))


def _vertex_normal(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    normals, _ = _face_normals_and_twice_areas(vertices, faces[face_ids])
    normal = normals.sum(axis=0)
    length = float(np.linalg.norm(normal))
    return normal / length if length > _EPS else np.zeros(3, dtype=np.float64)


def smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 10,
    angle_threshold: float = 10.0,
    step: float = 0.35,
    min_step: float = 1.0 / 128.0,
    preserve_boundary: bool = True,
    allow_normal_motion: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Smooth vertices while accepting only local sliver-quality improvements."""

    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not 0.0 < angle_threshold < 60.0:
        raise ValueError("angle_threshold must be in (0, 60)")
    if not 0.0 < step <= 1.0:
        raise ValueError("step must be in (0, 1]")
    if not 0.0 < min_step <= step:
        raise ValueError("min_step must be in (0, step]")

    result = np.asarray(vertices, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    neighbors, incident = _topology_data(faces, len(result))
    boundary = _boundary_vertices(faces, len(result)) if preserve_boundary else np.zeros(len(result), dtype=bool)
    accepted = 0
    rejected = 0

    for _ in range(iterations):
        angles = face_minimum_angles(result, faces)
        face_priority = np.maximum(angle_threshold - angles, 0.0)
        order = sorted(
            (vertex_id for vertex_id in range(len(result)) if len(neighbors[vertex_id]) and not boundary[vertex_id]),
            key=lambda vertex_id: (-float(face_priority[incident[vertex_id]].max()), vertex_id),
        )
        accepted_this_round = 0
        for vertex_id in order:
            face_ids = incident[vertex_id]
            if len(face_ids) == 0:
                continue
            current = result[vertex_id].copy()
            displacement = result[neighbors[vertex_id]].mean(axis=0) - current
            if not allow_normal_motion:
                normal = _vertex_normal(result, faces, face_ids)
                displacement -= normal * float(np.dot(displacement, normal))
            if np.linalg.norm(displacement) <= _EPS:
                continue

            old_angles = angles[face_ids]
            old_score = _quality_score(old_angles, angle_threshold)
            old_normals, old_areas = _face_normals_and_twice_areas(result, faces[face_ids])
            local_step = step
            improved = False
            while local_step >= min_step:
                candidate = current + local_step * displacement
                trial = result[faces[face_ids]].copy()
                for local_face, face in enumerate(faces[face_ids]):
                    local_index = int(np.flatnonzero(face == vertex_id)[0])
                    trial[local_face, local_index] = candidate
                trial_cross = np.cross(trial[:, 1] - trial[:, 0], trial[:, 2] - trial[:, 0])
                trial_areas = np.linalg.norm(trial_cross, axis=1)
                orientation_ok = np.einsum("ij,ij->i", old_normals, trial_cross) > _EPS * np.maximum(old_areas, 1.0)
                if np.all(trial_areas > _EPS) and np.all(orientation_ok):
                    trial_angles = face_minimum_angles(
                        np.vstack((trial.reshape(-1, 3),)),
                        np.arange(len(trial) * 3, dtype=np.int64).reshape(-1, 3),
                    )
                    new_score = _quality_score(trial_angles, angle_threshold)
                    if new_score < old_score:
                        result[vertex_id] = candidate
                        angles[face_ids] = trial_angles
                        accepted += 1
                        accepted_this_round += 1
                        improved = True
                        break
                local_step *= 0.5
            if not improved:
                rejected += 1
        if accepted_this_round == 0:
            break

    final_angles = face_minimum_angles(result, faces)
    before_angles = face_minimum_angles(np.asarray(vertices, dtype=np.float64), faces)
    stats: dict[str, float | int] = {
        "iterations_requested": int(iterations),
        "accepted_vertex_moves": int(accepted),
        "rejected_vertex_moves": int(rejected),
        "slivers_before": int(np.count_nonzero(before_angles < angle_threshold)),
        "slivers_after": int(np.count_nonzero(final_angles < angle_threshold)),
        "minimum_angle_before_degrees": float(before_angles.min()),
        "minimum_angle_after_degrees": float(final_angles.min()),
        "median_minimum_angle_before_degrees": float(np.median(before_angles)),
        "median_minimum_angle_after_degrees": float(np.median(final_angles)),
    }
    return result, stats


def laplacian_smooth_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 10,
    step: float = 0.35,
    preserve_boundary: bool = True,
    angle_threshold: float = 10.0,
    ring_size: int = 1,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply synchronous uniform Laplacian smoothing.

    Every interior vertex moves toward the arithmetic mean of its one-ring
    neighbors.  Unlike :func:`smooth_mesh`, the displacement is not projected
    into a tangent plane and is not quality-gated; a normal component is an
    intentional part of standard Laplacian smoothing.
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    if not 0.0 <= step <= 1.0:
        raise ValueError("step must be in [0, 1]")
    if ring_size < 1:
        raise ValueError("ring_size must be at least 1")
    if not 0.0 < angle_threshold < 60.0:
        raise ValueError("angle_threshold must be in (0, 60)")

    result = np.asarray(vertices, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError("vertices must be finite with shape [N, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be non-empty triangles")
    neighbors, _ = _topology_data(faces, len(result))
    if ring_size > 1:
        one_ring = [set(values.tolist()) for values in neighbors]
        expanded = []
        for vertex_id in range(len(result)):
            visited = {vertex_id}
            frontier = {vertex_id}
            for _ in range(ring_size):
                frontier = {neighbor for node in frontier for neighbor in one_ring[node]} - visited
                visited.update(frontier)
            expanded.append(np.asarray(sorted(visited - {vertex_id}), dtype=np.int64))
        neighbors = expanded
    boundary = _boundary_vertices(faces, len(result)) if preserve_boundary else np.zeros(len(result), dtype=bool)
    moved = 0
    displacement_norms: list[float] = []

    for _ in range(iterations):
        old = result.copy()
        for vertex_id, vertex_neighbors in enumerate(neighbors):
            if boundary[vertex_id] or len(vertex_neighbors) == 0:
                continue
            displacement = old[vertex_neighbors].mean(axis=0) - old[vertex_id]
            result[vertex_id] = old[vertex_id] + float(step) * displacement
            displacement_norms.append(float(np.linalg.norm(float(step) * displacement)))
            moved += 1
        if not np.isfinite(result).all():
            raise ValueError("Laplacian smoothing produced non-finite coordinates")

    before_angles = face_minimum_angles(np.asarray(vertices, dtype=np.float64), faces)
    after_angles = face_minimum_angles(result, faces)
    stats: dict[str, float | int] = {
        "method": "laplacian",
        "laplacian_ring_size": int(ring_size),
        "iterations_requested": int(iterations),
        "completed_iterations": int(iterations),
        "accepted_vertex_moves": int(moved),
        "rejected_vertex_moves": 0,
        "mean_vertex_displacement": float(np.mean(displacement_norms)) if displacement_norms else 0.0,
        "max_vertex_displacement": float(np.max(displacement_norms)) if displacement_norms else 0.0,
        "slivers_before": int(np.count_nonzero(before_angles < angle_threshold)),
        "slivers_after": int(np.count_nonzero(after_angles < angle_threshold)),
        "minimum_angle_before_degrees": float(before_angles.min()),
        "minimum_angle_after_degrees": float(after_angles.min()),
        "median_minimum_angle_before_degrees": float(np.median(before_angles)),
        "median_minimum_angle_after_degrees": float(np.median(after_angles)),
    }
    return result, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce sliver triangles without changing mesh topology.")
    parser.add_argument("input", type=Path, help="input OBJ/PLY/STL/OFF mesh")
    parser.add_argument("--output", type=Path, required=True, help="output mesh path")
    parser.add_argument("--method", choices=("quality", "laplacian"), default="quality", help="smoothing algorithm")
    parser.add_argument("--iterations", "--laplacian-iterations", dest="iterations", type=int, default=10, help="number of smoothing passes")
    parser.add_argument("--angle-threshold", type=float, default=10.0, help="sliver threshold in degrees")
    parser.add_argument("--step", "--laplacian-step", dest="step", type=float, default=0.35, help="Laplacian step fraction")
    parser.add_argument("--laplacian-rings", type=int, default=1, help="one-ring, two-ring, ... Laplacian neighborhood")
    parser.add_argument("--min-step", type=float, default=1.0 / 128.0, help="smallest line-search step")
    parser.add_argument("--allow-normal-motion", action="store_true", help="allow normal displacement; default is tangential smoothing")
    parser.add_argument("--move-boundary", action="store_true", help="smooth boundary vertices instead of keeping them fixed")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mesh = load_mesh(args.input)
        if args.method == "laplacian":
            smoothed, stats = laplacian_smooth_mesh(
                mesh.vertices,
                mesh.faces,
                iterations=args.iterations,
                step=args.step,
                preserve_boundary=not args.move_boundary,
                angle_threshold=args.angle_threshold,
                ring_size=args.laplacian_rings,
            )
        else:
            smoothed, stats = smooth_mesh(
                mesh.vertices,
                mesh.faces,
                iterations=args.iterations,
                angle_threshold=args.angle_threshold,
                step=args.step,
                min_step=args.min_step,
                preserve_boundary=not args.move_boundary,
                allow_normal_motion=args.allow_normal_motion,
            )
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=smoothed, faces=mesh.faces, process=False).export(output)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Input:  {args.input}")
    print(f"Output: {output}")
    print(f"Sliver faces (< {args.angle_threshold:g}°): {stats['slivers_before']} -> {stats['slivers_after']}")
    print(f"Minimum angle: {stats['minimum_angle_before_degrees']:.6f}° -> {stats['minimum_angle_after_degrees']:.6f}°")
    print(f"Method: {stats['method']}")
    print(f"Accepted vertex moves: {stats['accepted_vertex_moves']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
