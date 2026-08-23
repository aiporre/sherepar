from __future__ import annotations

import json
import numpy as np
from pathlib import Path

import pytest
import trimesh

from spherepar.benchmark.dataset_generator import (
    DEFORMATION_CASES,
    _GRAPHOP_AVAILABLE,
    _nearest_vertex_index,
    _validate_noise_mesh,
    apply_gaussian_smoothed_vertex_noise,
    build_arg_parser,
    generate_dataset,
    parse_signal_center,
)


def _mean_edge_displacement_jump(vertices: np.ndarray, faces: np.ndarray) -> float:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return float(np.mean(np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)))


def test_noise_is_reproducible_and_preserves_face_connectivity():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    kwargs = {
        "vertices": mesh.vertices,
        "faces": mesh.faces,
        "noise_sigma": 0.01,
        "noise_smooth_sigma": 1.0,
    }
    first, meta = apply_gaussian_smoothed_vertex_noise(rng=np.random.default_rng(4), **kwargs)
    second, _ = apply_gaussian_smoothed_vertex_noise(rng=np.random.default_rng(4), **kwargs)

    assert np.allclose(first, second)
    assert meta["filter_target"] == "vertex_displacement"
    assert meta["face_connectivity"] == "preserved"
    checked, quality = _validate_noise_mesh(first, mesh.faces)
    assert checked is not None, quality
    assert np.array_equal(checked.faces, mesh.faces)


def test_zero_noise_leaves_vertices_unchanged():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    vertices, _ = apply_gaussian_smoothed_vertex_noise(
        mesh.vertices,
        mesh.faces,
        noise_sigma=0.0,
        noise_smooth_sigma=1.0,
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(vertices, mesh.vertices)


def test_heat_filter_reduces_noise_field_edge_variation():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    raw_vertices, _ = apply_gaussian_smoothed_vertex_noise(
        mesh.vertices,
        mesh.faces,
        noise_sigma=0.02,
        noise_smooth_sigma=0.0,
        rng=np.random.default_rng(9),
    )
    smooth_vertices, _ = apply_gaussian_smoothed_vertex_noise(
        mesh.vertices,
        mesh.faces,
        noise_sigma=0.02,
        noise_smooth_sigma=1.0,
        rng=np.random.default_rng(9),
    )
    raw_noise = raw_vertices - mesh.vertices
    smooth_noise = smooth_vertices - mesh.vertices
    assert _mean_edge_displacement_jump(smooth_noise, mesh.faces) < _mean_edge_displacement_jump(raw_noise, mesh.faces)


def test_noise_validation_rejects_nonwatertight_meshes():
    mesh = trimesh.creation.box()
    checked, quality = _validate_noise_mesh(mesh.vertices, mesh.faces[:-1])
    assert checked is None
    assert quality["validation_error"] == "non_watertight"


def test_noise_cases_and_cli_defaults_are_registered():
    assert {"case4_noise", "case5_small_noise", "case6_large_noise"} <= set(DEFORMATION_CASES)
    args = build_arg_parser().parse_args(["input"])
    assert args.noise_sigma == 0.01
    assert args.noise_smooth_sigma == 1.0


def test_mesh_loader_accepts_off_files(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    trimesh.creation.icosphere(subdivisions=1).export(input_dir / "sphere.off")

    from spherepar.benchmark.dataset_generator import load_meshes_from_directory

    meshes = load_meshes_from_directory(str(input_dir))
    assert len(meshes) == 1
    assert meshes[0][0] == "sphere"
    assert len(meshes[0][1].faces) > 0


def test_signal_center_parser_and_nearest_template_vertex():
    assert parse_signal_center("1, 2.5, -3") == (1.0, 2.5, -3.0)
    with pytest.raises(Exception):
        parse_signal_center("1,2")
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    assert _nearest_vertex_index(vertices, np.array([0.9, 1.1, 1.0])) == 1


def test_case4_noise_writes_watertight_mesh_and_noise_metadata(tmp_path: Path):
    if not _GRAPHOP_AVAILABLE:
        pytest.skip("generator retains its graphop dependency")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    trimesh.creation.icosphere(subdivisions=2, radius=1.0).export(input_dir / "sphere.obj")
    output_root = tmp_path / "output"

    saved = generate_dataset(
        str(input_dir),
        output_root=str(output_root),
        n_samples_per_mesh=1,
        signal_type="isotropic",
        deformation_cases=["case4_noise"],
        param_method=None,
        noise_sigma=0.01,
        noise_smooth_sigma=1.0,
        seed=123,
    )

    assert saved == 1
    label_path = output_root / "labels" / "sphere_s000000.json"
    label = json.loads(label_path.read_text())
    deformation = label["deformation"]
    assert deformation["type"] == "gaussian_smoothed_vertex_noise"
    assert deformation["noise_sigma"] == 0.01
    assert deformation["noise_smooth_sigma"] == 1.0
    mesh = trimesh.load(output_root / label["paths"]["mesh"], force="mesh")
    assert mesh.is_watertight


def test_fixed_center_pins_only_the_one_center_regression_pair(tmp_path: Path):
    if not _GRAPHOP_AVAILABLE:
        pytest.skip("generator retains its graphop dependency")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    source_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    source_mesh.export(input_dir / "sphere.obj")
    output_root = tmp_path / "output"
    requested_center = source_mesh.vertices[0]
    fixed_vertex_id = _nearest_vertex_index(source_mesh.vertices, requested_center)

    saved = generate_dataset(
        str(input_dir),
        output_root=str(output_root),
        n_samples_per_mesh=1,
        signal_type="isotropic",
        signal_centers_options=[2],
        deformation_cases=["case4_noise"],
        param_method=None,
        signal_center=requested_center,
        seed=123,
    )

    assert saved == 1
    label = json.loads((output_root / "labels" / "sphere_s000000.json").read_text())
    fixed_meta = label["metadata"]["fixed_signal_center"]
    assert fixed_meta["method"] == "nearest_template_vertex"
    assert fixed_meta["template_vertex_id"] == fixed_vertex_id
    assert fixed_meta["requested_xyz"] == pytest.approx(requested_center.tolist())
    assert label["signals"][0]["num_centers"] == 2
    assert label["signals"][0]["center_sampling"]["method"] == "random_vertex"
    aniso = next(signal for signal in label["signals"] if signal["signal_id"] == "aniso_000")
    assert aniso["center_vertex_ids"] == [fixed_vertex_id]
    assert aniso["center_sampling"]["method"] == "nearest_template_vertex"
    assert label["task_groups"]["isotropic_gaussian"]["tasks"]["center_regression"]["label"] == pytest.approx(aniso["centers"][0])


def test_invalid_fixed_gauge_center_skips_the_sample(tmp_path: Path):
    if not _GRAPHOP_AVAILABLE:
        pytest.skip("generator retains its graphop dependency")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    mesh = trimesh.creation.uv_sphere(count=[16, 16])
    mesh.export(input_dir / "sphere.obj")
    output_root = tmp_path / "output"

    saved = generate_dataset(
        str(input_dir),
        output_root=str(output_root),
        n_samples_per_mesh=1,
        signal_type="isotropic",
        deformation_cases=["case1_no"],
        param_method=None,
        signal_center=[0.0, 0.0, -1.0],
        seed=123,
    )

    assert saved == 0
    assert "fixed anisotropic center vertex" in (output_root / "logs" / "errors.log").read_text()
