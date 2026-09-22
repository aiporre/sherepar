from __future__ import annotations

import numpy as np
import pytest

from spherepar import cem_parametrization as cem
from spherepar.cem_anchor_diagnostics import (
    eccentricity_cache_statistics,
    reset_eccentricity_cache,
)
from spherepar.idt_remesh import connectivity_hash, intrinsic_delaunay_remesh
from spherepar.mesh import MeshSurf, StretchFunction


def _tetrahedron(scale: float = 1.0) -> MeshSurf:
    vertices = scale * np.asarray(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float64
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return MeshSurf(vertices, faces)


def test_idt_flips_known_non_delaunay_patch_without_moving_vertices():
    vertices = np.asarray(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 0.2, 0]], dtype=np.float64
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    before = vertices.copy()

    remeshed, diagnostics = intrinsic_delaunay_remesh(vertices, faces)

    edges = {
        tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
        for face in remeshed
        for index in range(3)
    }
    assert (1, 3) in edges
    assert (0, 2) not in edges
    assert diagnostics["flip_count"] == 1
    assert diagnostics["converged"] is True
    assert diagnostics["topology_hash_before"] != diagnostics["topology_hash_after"]
    assert np.array_equal(vertices, before)
    normals = np.cross(
        vertices[remeshed[:, 1]] - vertices[remeshed[:, 0]],
        vertices[remeshed[:, 2]] - vertices[remeshed[:, 0]],
    )
    assert np.all(normals[:, 2] > 0.0)


def test_idt_degenerate_and_safety_cap_fail_clearly():
    degenerate_vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float64
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    with pytest.raises(ValueError, match="degenerate"):
        intrinsic_delaunay_remesh(degenerate_vertices, faces)

    vertices = np.asarray(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 0.2, 0]], dtype=np.float64
    )
    with pytest.raises(RuntimeError, match="safety cap"):
        intrinsic_delaunay_remesh(vertices, faces, max_flips=0)


def test_idt_duplicate_diagonal_and_nonmanifold_fail_clearly():
    vertices = np.asarray([
        [0.1890533818, -0.5227484415, -0.4130635434],
        [-2.4414673826, 1.7997073827, 1.1441658720],
        [-0.3254228369, 0.7738065867, 0.2812106698],
        [-0.5538228364, 0.9775674511, -0.3105565467],
    ])
    tetra_faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    with pytest.raises(RuntimeError, match="duplicate_diagonal"):
        intrinsic_delaunay_remesh(vertices, tetra_faces)

    nonmanifold_faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int32)
    nonmanifold_vertices = np.vstack((vertices, [[0.0, 0.0, 2.0]]))
    with pytest.raises(ValueError, match="non-manifold"):
        intrinsic_delaunay_remesh(nonmanifold_vertices, nonmanifold_faces)


def test_connectivity_hash_ignores_coordinates_and_changes_with_faces():
    first = _tetrahedron(1.0)
    second = _tetrahedron(7.0)
    assert connectivity_hash(4, first.get_faces_collection()) == connectivity_hash(
        4, second.get_faces_collection()
    )
    changed = np.asarray([[0, 1, 2], [0, 2, 4]], dtype=np.int32)
    baseline = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    assert connectivity_hash(5, baseline) != connectivity_hash(5, changed)


def test_eccentricity_cache_reuses_same_connectivity_across_coordinates():
    reset_eccentricity_cache(enabled=True)
    first = _tetrahedron(1.0)
    second = _tetrahedron(3.0)
    first.get_central_regular_face(100.0)
    first_stats = eccentricity_cache_statistics()
    second.get_central_regular_face(100.0)
    second_stats = eccentricity_cache_statistics()
    assert first_stats["misses"] == 4
    assert first_stats["hits"] == 0
    assert second_stats["misses"] == 4
    assert second_stats["hits"] == 4
    assert second.anchor_cache_diagnostics["hit_rate"] == 1.0


def _fake_prepared(mesh: MeshSurf):
    initial = StretchFunction(mesh, np.arange(len(mesh.vertices), dtype=np.complex128))
    initial.anchor_diagnostics = None
    return initial, np.eye(len(mesh.vertices)), {}, {}


def _fake_result(mesh: MeshSurf, radius: float, collapsed: int) -> StretchFunction:
    result = StretchFunction(mesh, np.full(len(mesh.vertices), radius + 0j, np.complex128))
    result.cem_diagnostics = {
        "final_validation": {"is_valid": collapsed == 0, "degenerate_face_count": collapsed},
        "convergence": {"stop_reason": "constructed"},
    }
    return result


def test_adaptive_radius_selection_tie_break_and_shared_initialization(monkeypatch, capsys):
    mesh = _tetrahedron()
    prepare_calls = []
    initial_ids = []
    collapsed = {1.2: 3, 1.1: 1, 1.3: 1}

    def prepare(*args, **kwargs):
        prepared = _fake_prepared(mesh)
        prepare_calls.append(prepared)
        return prepared

    def attempt(*args, **kwargs):
        initial_ids.append(id(kwargs["_prepared"][0]))
        radius = kwargs["radius"]
        return _fake_result(mesh, radius, collapsed[radius])

    monkeypatch.setattr(cem, "_prepare_cem", prepare)
    monkeypatch.setattr(cem, "_stretch_parametrization_attempt", attempt)
    selected = cem.stretch_parametrization(
        mesh,
        radius=1.2,
        adaptive_radius=True,
        radius_candidates=(1.1, 1.3),
        max_attempts=3,
    )
    assert len(prepare_calls) == 1
    assert len(set(initial_ids)) == 1
    assert selected.cem_diagnostics["selected_radius"] == 1.1
    assert len(selected.cem_diagnostics["radius_attempts"]) == 3
    assert "[CEM radius]" in capsys.readouterr().out


def test_radius_numerical_failure_continues_and_reject_caps_attempts(monkeypatch):
    mesh = _tetrahedron()
    monkeypatch.setattr(cem, "_prepare_cem", lambda *args, **kwargs: _fake_prepared(mesh))
    calls = []

    def attempt(*args, **kwargs):
        radius = kwargs["radius"]
        calls.append(radius)
        if radius == 1.2:
            raise np.linalg.LinAlgError("constructed")
        return _fake_result(mesh, radius, 2)

    monkeypatch.setattr(cem, "_stretch_parametrization_attempt", attempt)
    selected = cem.stretch_parametrization(
        mesh,
        reject_retry=True,
        radius_candidates=(1.1, 1.3, 1.4, 1.5),
        max_attempts=3,
        max_collapsed_faces=0,
    )
    assert calls == [1.2, 1.1, 1.3]
    assert selected.cem_diagnostics["acceptance"]["accepted"] is False
    assert len(selected.cem_diagnostics["radius_attempts"]) == 3
