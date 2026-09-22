"""Tests for local ADNI manifest discovery and participant-stratified labels."""

import importlib.util
import json
import sys
from pathlib import Path

from spherepar.benchmark.splits import TASK_ADNI_CLS, build_task_splits, is_valid_for_adni_cls


def _import_file_importer():
    path = Path(__file__).resolve().parents[1] / "examples" / "script_to_generate_dataset_from_files.py"
    spec = importlib.util.spec_from_file_location("adni_file_importer", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_adni_manifest_filters_and_collapses_diagnosis(tmp_path: Path):
    importer = _import_file_importer()
    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    manifest = tmp_path / "participants.tsv"
    manifest.write_text(
        "participant_id\tdiagnosis_sc\n"
        "sub-ADNI0001\tSMC\n"
        "sub-ADNI0002\tLMCI\n"
        "sub-ADNI0003\tAD\n"
    )
    for participant, sessions in (("sub-ADNI0001", ("M000", "M012")), ("sub-ADNI0002", ("M000",))):
        for session in sessions:
            for hip in ("left", "right"):
                (mesh_root / f"{participant}-ses-{session}_hip_{hip}.obj").touch()
    (mesh_root / "unrelated.obj").touch()

    first_left, skipped = importer._adni_mesh_inputs(mesh_root, manifest, "first", "left")
    assert len(first_left) == 2
    assert {item.class_id for item in first_left} == {0, 1}
    assert {item.session for item in first_left} == {"ses-M000"}
    assert skipped == ["unrelated.obj: filename does not match sub-<id>-ses-Mxxx_hip_<left|right>"]

    all_right, _ = importer._adni_mesh_inputs(mesh_root, manifest, "none", "right")
    assert len(all_right) == 3
    assert all(item.hip == "right" for item in all_right)


def test_adni_splits_keep_participants_together_and_are_stratified(tmp_path: Path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for participant_idx in range(12):
        class_id = participant_idx % 3
        for side in ("left", "right"):
            sample_id = f"sub-{participant_idx:04d}_{side}"
            (labels_dir / f"{sample_id}.json").write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "tasks": {
                            TASK_ADNI_CLS: {
                                "valid": True,
                                "label": class_id,
                                "participant_id": f"sub-{participant_idx:04d}",
                            }
                        },
                    }
                )
            )
    assert is_valid_for_adni_cls(
        {"tasks": {TASK_ADNI_CLS: {"valid": True, "label": 0, "participant_id": "sub-0000"}}}
    )
    summary = build_task_splits(str(tmp_path), tasks=[TASK_ADNI_CLS], num_folds=2, seed=11)
    assert summary["tasks"][TASK_ADNI_CLS]["participant_counts"] == 12
    for fold_idx in (1, 2):
        base = tmp_path / "folds" / f"fold{fold_idx}" / TASK_ADNI_CLS
        partitions = [set((base / f"{name}.txt").read_text().splitlines()) for name in ("train", "val", "test")]
        assert not partitions[0] & partitions[1]
        assert not partitions[0] & partitions[2]
        assert not partitions[1] & partitions[2]
        owners = {}
        for partition_idx, samples in enumerate(partitions):
            for sample in samples:
                participant = sample.rsplit("_", 1)[0]
                assert participant not in owners or owners[participant] == partition_idx
                owners[participant] = partition_idx
