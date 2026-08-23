from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
