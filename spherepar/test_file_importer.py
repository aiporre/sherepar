from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


def _load_importer_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "script_to_generate_dataset_from_files.py"
    spec = importlib.util.spec_from_file_location("dataset_files_importer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_file_importer_discovers_flat_and_modelnet40_inputs(tmp_path: Path):
    importer = _load_importer_module()
    flat_dir = tmp_path / "cylinders"
    flat_dir.mkdir()
    trimesh.creation.icosphere(subdivisions=1).export(flat_dir / "cylinder.off")
    assert [item.sample_name for item in importer._flat_mesh_inputs(flat_dir, "CYLINDERS")] == ["cylinder"]

    modelnet_root = tmp_path / "ModelNet40"
    for class_name in ("chair", "table"):
        for source_split in ("train", "test"):
            mesh_dir = modelnet_root / class_name / source_split
            mesh_dir.mkdir(parents=True)
            trimesh.creation.icosphere(subdivisions=1).export(mesh_dir / f"{class_name}_{source_split}.off")

    all_inputs = importer._modelnet40_mesh_inputs(modelnet_root, 100.0, seed=4)
    assert len(all_inputs) == 4
    assert {(item.class_name, item.class_id) for item in all_inputs} == {("chair", 0), ("table", 1)}
    assert {item.source_split for item in all_inputs} == {"train", "test"}

    sampled = importer._modelnet40_mesh_inputs(modelnet_root, 50.0, seed=4)
    assert len(sampled) == 2
    assert {item.class_name for item in sampled} == {"chair", "table"}


def test_file_importer_resume_defaults_to_enabled():
    importer = _load_importer_module()
    parser = importer.build_arg_parser()
    args = parser.parse_args(["--cylinders-dir", "input", "--output-root", "output"])
    assert args.resume is True
    args = parser.parse_args(["--cylinders-dir", "input", "--output-root", "output", "--no-resume"])
    assert args.resume is False


def test_file_importer_mobius_center_flag_is_opt_in():
    importer = _load_importer_module()
    parser = importer.build_arg_parser()
    base = ["--cylinders-dir", "input", "--output-root", "output"]
    assert parser.parse_args(base).mobius_center is False
    assert parser.parse_args(base + ["--param-method", "cem", "--mobius-center"]).mobius_center is True


def test_file_importer_cem_radius_defaults_and_override():
    importer = _load_importer_module()
    parser = importer.build_arg_parser()
    base = ["--cylinders-dir", "input", "--output-root", "output"]
    assert parser.parse_args(base).cem_radius == 1.2
    assert parser.parse_args(base + ["--cem-radius", "1.6"]).cem_radius == 1.6


def test_spherical_validation_warning_is_logged_without_rejecting_sphere(tmp_path: Path, monkeypatch):
    from spherepar.benchmark import dataset_generator

    mesh = trimesh.creation.icosphere(subdivisions=1)
    sphere_vertices = np.asarray(mesh.vertices).copy()
    sphere_vertices[1] = sphere_vertices[0]
    validation = {
        "is_valid": False,
        "errors": ["sphere has collapsed or near-duplicate vertices"],
        "min_vertex_separation": 0.0,
        "near_duplicate_vertex_count": 2,
    }
    monkeypatch.setattr(
        dataset_generator,
        "compute_spherical_parametrization",
        lambda **kwargs: (sphere_vertices, {"method": "cem", "sphere_validation": validation}),
    )
    log_path = tmp_path / "logs" / "errors.log"

    paths = dataset_generator.save_spherical_parametrization(
        root=str(tmp_path),
        name="tr_reg_000",
        vertices=mesh.vertices,
        faces=mesh.faces,
        method="cem",
        log_path=str(log_path),
        template_id="tr_reg_000",
        deformation_case="case1_no",
    )

    assert (tmp_path / paths["sphere"]).is_file()
    spherical_label = json.loads((tmp_path / paths["spherical_label"]).read_text())
    assert spherical_label["metadata"]["sphere_validation"] == validation
    log_text = log_path.read_text()
    assert "spherical parametrization validation warning" in log_text
    assert '"min_vertex_separation": 0.0' in log_text


def test_cem_diagnostic_warning_is_logged_without_rejecting_sphere(tmp_path: Path, monkeypatch):
    from spherepar.benchmark import dataset_generator

    mesh = trimesh.creation.icosphere(subdivisions=1)
    cem_diagnostics = {
        "radius": 1.2,
        "cotangent_weights": {
            "negative_weight_count": 1,
            "negative_edge_ids": [[0, 1]],
            "affected_triangle_count": 2,
            "affected_triangle_ids": [0, 1],
            "is_intrinsic_delaunay": False,
        },
        "first_iteration_validation": {"is_valid": True, "errors": []},
        "final_validation": {"is_valid": True, "errors": []},
    }

    def fake_compute_spherical_parametrization(**kwargs):
        kwargs["cem_input_diagnostics_callback"]({
            "summary": (
                "faces=80, min_angle=5.0deg, below_5deg=2, "
                "negative_cotangent_edges=1, affected_triangles=2; "
                "angle_thresholds=[below_5deg=2(2.50%)]"
            )
        })
        return (
            np.asarray(mesh.vertices).copy(),
            {
                "method": "cem",
                "cem_radius": kwargs["cem_radius"],
                "cem_diagnostics": cem_diagnostics,
                "sphere_validation": {"is_valid": True, "errors": []},
            },
        )

    monkeypatch.setattr(
        dataset_generator,
        "compute_spherical_parametrization",
        fake_compute_spherical_parametrization,
    )
    log_path = tmp_path / "logs" / "errors.log"

    paths = dataset_generator.save_spherical_parametrization(
        root=str(tmp_path),
        name="tr_reg_000",
        vertices=mesh.vertices,
        faces=mesh.faces,
        method="cem",
        cem_radius=1.2,
        log_path=str(log_path),
    )

    assert (tmp_path / paths["sphere"]).is_file()
    sidecar = json.loads((tmp_path / paths["spherical_label"]).read_text())
    assert sidecar["metadata"]["cem_radius"] == 1.2
    assert sidecar["metadata"]["cem_diagnostics"] == cem_diagnostics
    log_text = log_path.read_text()
    assert "CEM input diagnostics" in log_text
    assert "angle_thresholds=[below_5deg=2(2.50%)]" in log_text
    assert "CEM diagnostic warning" in log_text
    assert "affected_triangles=2" in log_text
    assert log_text.index("CEM input diagnostics") < log_text.index("CEM diagnostic warning")
