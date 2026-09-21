from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from spherepar.benchmark.dataset_generator import save_spherical_parametrization
from spherepar.cem_anchor_diagnostics import (
    bfs_hop_distances,
    build_vertex_adjacency,
    collect_anchor_geometry,
    compute_anchor_collapse_diagnostics,
)
from spherepar.mesh import Face, MeshFactory, Vertex
from spherepar.parametrization_validation import collapsed_face_geometry
from spherepar.spherical_parametrization import compute_spherical_parametrization


def _tetrahedron():
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return vertices, faces, MeshFactory.make_mesh("surf", vertices, faces)


def test_scale_normalized_regularity_is_scale_invariant_and_rejects_zero_edges():
    face = Face(
        Vertex([0.0, 0.0, 0.0], 0),
        Vertex([2.0, 0.0, 0.0], 1),
        Vertex([0.25, 1.0, 0.0], 2),
    )
    scaled = Face(
        Vertex([0.0, 0.0, 0.0], 0),
        Vertex([14.0, 0.0, 0.0], 1),
        Vertex([1.75, 7.0, 0.0], 2),
    )
    assert scaled.scale_normalized_regularity() == pytest.approx(
        face.scale_normalized_regularity()
    )

    zero_edge_face = Face(
        Vertex([1.0, 1.0, 1.0], 0),
        Vertex([1.0, 1.0, 1.0], 1),
        Vertex([1.0, 1.0, 1.0], 2),
    )
    with pytest.raises(ValueError, match="mean edge length"):
        zero_edge_face.scale_normalized_regularity()


@pytest.mark.parametrize("percentile", [-1.0, 100.1, np.nan, np.inf])
def test_central_regular_rejects_invalid_percentile(percentile):
    _, _, mesh = _tetrahedron()
    with pytest.raises(ValueError, match="regularity_percentile"):
        mesh.get_central_regular_face(percentile)


def test_central_regular_avoids_tiny_off_center_regular_face():
    synthetic = trimesh.creation.icosphere(subdivisions=2)
    vertices = np.asarray(synthetic.vertices, dtype=np.float64).copy()
    vertices *= np.asarray([1.17, 0.93, 1.31])
    tiny_face_ids = synthetic.faces[0]
    epsilon = 1e-5
    vertices[tiny_face_ids] = np.asarray([
        [epsilon, 0.0, 0.0],
        [-epsilon / 2.0, np.sqrt(3.0) * epsilon / 2.0, 0.0],
        [-epsilon / 2.0, -np.sqrt(3.0) * epsilon / 2.0, 0.0],
    ])
    mesh = MeshFactory.make_mesh("surf", vertices, synthetic.faces)

    assert mesh.get_most_regular_face().id == tuple(int(value) for value in tiny_face_ids)
    assert mesh.get_central_regular_face().id == (12, 74, 71)


def test_central_regular_tie_breaks_by_face_id_and_reuses_unreachable_failure():
    vertices = np.asarray(
        [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    mesh = MeshFactory.make_mesh("surf", vertices, faces)
    assert mesh.get_central_regular_face(100.0).id == (0, 1, 3)

    disconnected = MeshFactory.make_mesh(
        "surf",
        np.asarray([
            [0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0],
            [3, 0, 0], [4, 0, 0], [3.5, np.sqrt(3) / 2, 0],
        ], dtype=np.float64),
        np.asarray([[0, 1, 2], [3, 4, 5]]),
    )
    with pytest.raises(ValueError, match="unreachable from anchor"):
        disconnected.get_central_regular_face(100.0)


def test_tetrahedron_anchor_bfs_and_edge_statistics():
    vertices, _, mesh = _tetrahedron()
    adjacency = build_vertex_adjacency(mesh)
    for source in range(4):
        distances = bfs_hop_distances(adjacency, source)
        assert distances[source] == 0
        assert np.all(distances[np.arange(4) != source] == 1)

    face = next(iter(mesh.faces.values()))
    anchor = collect_anchor_geometry(mesh, face)
    assert anchor["vertex_ids"] == [0, 2, 1]
    assert anchor["minimum_hop_distances"] == [0, 0, 0, 1]
    assert anchor["positions"] == vertices[[0, 2, 1]].tolist()
    assert np.allclose(anchor["edge_lengths"], [1.0, np.sqrt(2.0), 1.0])
    all_edge_lengths = np.asarray([1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)])
    assert anchor["mesh_edge_length_mean"] == pytest.approx(all_edge_lengths.mean())
    assert anchor["mesh_edge_length_median"] == pytest.approx(np.median(all_edge_lengths))


def test_bfs_fails_clearly_for_disconnected_mesh():
    with pytest.raises(ValueError, match="unreachable from anchor 0"):
        bfs_hop_distances([[1], [0], [3], [2]], 0)


def test_collapse_statistics_are_json_safe_and_handle_empty_constant_groups():
    vertices, faces, mesh = _tetrahedron()
    anchor = collect_anchor_geometry(mesh, next(iter(mesh.faces.values())))

    none_collapsed = compute_anchor_collapse_diagnostics(
        vertices,
        faces,
        vertices,
        anchor["minimum_hop_distances"],
        sphere_stage="test",
    )
    assert none_collapsed["collapsed_vertex_count"] == 0
    assert none_collapsed["distance_summary"]["collapsed"]["mean"] is None
    assert none_collapsed["statistics"]["mann_whitney_u"]["statistic"] is None
    assert none_collapsed["statistics"]["spearman_collapsed_flag"]["statistic"] is None

    all_collapsed = compute_anchor_collapse_diagnostics(
        vertices,
        faces,
        vertices,
        anchor["minimum_hop_distances"],
        sphere_stage="test",
        relative_threshold=2.0,
    )
    assert all_collapsed["collapsed_vertex_count"] == 4
    assert all_collapsed["distance_summary"]["non_collapsed"]["median"] is None
    assert all_collapsed["statistics"]["mann_whitney_u"]["p_value"] is None
    assert all_collapsed["statistics"]["spearman_collapse_severity"]["statistic"] is None
    json.dumps(all_collapsed, allow_nan=False)

    zero_area_sphere = vertices.copy()
    zero_area_sphere[1] = zero_area_sphere[0]
    finite_zero_area = compute_anchor_collapse_diagnostics(
        vertices,
        faces,
        zero_area_sphere,
        anchor["minimum_hop_distances"],
        sphere_stage="test",
    )
    assert np.isfinite(finite_zero_area["collapse_severity_log10_area_ratio"]).all()


def test_plot_and_validation_share_collapse_threshold():
    from examples.plot_faust_cem_error import collapsed_geometry

    vertices, faces, _ = _tetrahedron()
    expected = collapsed_face_geometry(vertices, faces, 0.25, 1e-7)
    actual = collapsed_geometry(vertices, faces, 0.25, 1e-7)
    for expected_value, actual_value in zip(expected[:3], actual[:3]):
        assert np.array_equal(expected_value, actual_value)
    assert expected[3] == actual[3]


def test_anchor_flag_does_not_change_cem_coordinates_and_is_opt_in():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    kwargs = dict(
        vertices=mesh.vertices,
        faces=mesh.faces,
        method="cem",
        cem_max_iters=1,
        cem_verbose=False,
        verify=False,
    )
    plain_vertices, plain_meta = compute_spherical_parametrization(**kwargs)
    diagnostic_vertices, diagnostic_meta = compute_spherical_parametrization(
        **kwargs, anchor_diagnostics=True
    )
    assert np.array_equal(plain_vertices, diagnostic_vertices)
    assert "anchor" not in plain_meta["cem_diagnostics"]
    assert diagnostic_meta["cem_diagnostics"]["anchor"]["analyzed_sphere_stage"] == "pre_mobius_cem"


def test_explicit_regular_strategy_preserves_default_coordinates_and_anchor():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    kwargs = dict(
        vertices=mesh.vertices,
        faces=mesh.faces,
        method="cem",
        cem_max_iters=1,
        cem_verbose=False,
        verify=False,
        anchor_diagnostics=True,
    )
    default_vertices, default_meta = compute_spherical_parametrization(**kwargs)
    explicit_vertices, explicit_meta = compute_spherical_parametrization(
        **kwargs,
        anchor_strategy="regular",
        anchor_regularity_percentile=37.0,
    )
    assert np.array_equal(default_vertices, explicit_vertices)
    default_anchor = default_meta["cem_diagnostics"]["anchor"]
    explicit_anchor = explicit_meta["cem_diagnostics"]["anchor"]
    assert default_anchor["vertex_ids"] == explicit_anchor["vertex_ids"]
    assert explicit_anchor["strategy"] == "regular"
    assert explicit_anchor["regularity_percentile"] == 37.0
    assert len(explicit_anchor["vertex_eccentricities"]) == 3
    assert np.isfinite(explicit_anchor["scale_normalized_regularity"])


def test_dataset_save_persists_anchor_metadata_and_log(tmp_path: Path):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    log_path = tmp_path / "logs" / "errors.log"
    paths = save_spherical_parametrization(
        root=str(tmp_path),
        name="sample",
        vertices=mesh.vertices,
        faces=mesh.faces,
        method="cem",
        cem_max_iters=1,
        anchor_diagnostics=True,
        anchor_strategy="central_regular",
        anchor_regularity_percentile=25.0,
        log_path=str(log_path),
    )
    sidecar = json.loads((tmp_path / paths["spherical_label"]).read_text())
    anchor = sidecar["metadata"]["cem_diagnostics"]["anchor"]
    assert sidecar["metadata"]["anchor_strategy"] == "central_regular"
    assert sidecar["metadata"]["anchor_regularity_percentile"] == 25.0
    assert anchor["strategy"] == "central_regular"
    assert anchor["regularity_percentile"] == 25.0
    assert len(anchor["vertex_eccentricities"]) == 3
    assert len(anchor["minimum_hop_distances"]) == len(mesh.vertices)
    assert "[CEM anchor]" in log_path.read_text()


def test_plot_anchor_flag_reconstructs_legacy_sidecar(tmp_path: Path, monkeypatch):
    from examples import plot_faust_cem_error

    dataset = tmp_path / "legacy"
    for dirname in ("meshes", "spheres", "labels"):
        (dataset / dirname).mkdir(parents=True)
    mesh = trimesh.creation.icosphere(subdivisions=1)
    mesh.export(dataset / "meshes" / "sample.obj")
    mesh.export(dataset / "spheres" / "sample.obj")
    (dataset / "labels" / "sample_spherical.json").write_text(json.dumps({
        "name": "sample",
        "metadata": {"method": "cem", "cem_diagnostics": {}},
    }))
    output = dataset / "collapsed.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_faust_cem_error.py",
            "--dataset-root", str(dataset),
            "--sample-id", "sample",
            "--anchor-diagnostics",
            "--output", str(output),
            "--no-show",
        ],
    )
    plot_faust_cem_error.main()
    assert (dataset / "collapsed_anchor_distance.png").is_file()
    assert "[CEM anchor]" in (dataset / "logs" / "errors.log").read_text()
