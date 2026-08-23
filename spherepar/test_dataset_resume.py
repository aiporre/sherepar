from __future__ import annotations

import json
from pathlib import Path

import pytest

from spherepar.benchmark.dataset_generator import (
    add_path_labels,
    build_arg_parser,
    generate_dataset,
    _list_completed_samples,
    _next_sample_index,
)


def _write_sample(
    root: Path,
    sample_id: str,
    *,
    parametrization: dict | None = None,
) -> None:
    for dirname in ("meshes", "signals", "labels", "spheres"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    (root / "meshes" / f"{sample_id}.obj").write_text("o mesh\n")
    (root / "signals" / f"{sample_id}_iso_000.npy").write_bytes(b"signal")
    label = {
        "sample_id": sample_id,
        "metadata": {
            "template_id": "template",
            "deformation_case": "case2_small",
        },
        "parametrization": parametrization or {
            "method": None,
            "success": False,
            "error": None,
        },
    }
    (root / "labels" / f"{sample_id}.json").write_text(json.dumps(label))


def test_resume_scan_only_counts_matching_complete_artifacts(tmp_path: Path):
    root = tmp_path / "generated"
    complete_id = "template_s000000"
    _write_sample(root, complete_id)

    # These unpaired artifacts must not move the resume point.
    (root / "meshes" / "template_s000001.obj").write_text("o mesh\n")
    (root / "signals" / "template_s000002_iso_000.npy").write_bytes(b"signal")
    (root / "labels" / "template_s000003.json").write_text("{}")

    completed, counts = _list_completed_samples(str(root))

    assert set(completed) == {complete_id}
    assert counts == {"meshes": 2, "spheres": 0, "signals": 2, "labels": 1}
    assert _next_sample_index(list(completed)) == 1


def test_resume_scan_requires_sphere_after_successful_parametrization(tmp_path: Path):
    root = tmp_path / "generated"
    sample_id = "template_s000004"
    _write_sample(
        root,
        sample_id,
        parametrization={"method": "flash", "success": True, "error": None},
    )

    completed, _ = _list_completed_samples(str(root))
    assert completed == {}

    (root / "spheres" / f"{sample_id}.obj").write_text("o sphere\n")
    completed, _ = _list_completed_samples(str(root))
    assert set(completed) == {sample_id}


def test_resume_scan_retries_label_written_before_parametrization(tmp_path: Path):
    root = tmp_path / "generated"
    _write_sample(
        root,
        "template_s000005",
        parametrization={"method": "cem", "success": False, "error": None},
    )

    completed, _ = _list_completed_samples(str(root))
    assert completed == {}


def test_no_resume_cli_flag_disables_resume():
    args = build_arg_parser().parse_args(["input", "--no-resume"])
    assert args.resume is False


def test_workers_cli_flag_defaults_to_one_and_accepts_parallelism():
    parser = build_arg_parser()
    assert parser.parse_args(["input"]).workers == 1
    assert parser.parse_args(["input", "--workers", "2"]).workers == 2


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match="workers must be at least 1"):
        generate_dataset("unused", workers=0)


def test_add_path_labels_normalizes_dual_signal_label(tmp_path: Path):
    root = tmp_path / "generated"
    labels = root / "labels"
    labels.mkdir(parents=True)
    label_path = labels / "template_s000000.json"
    label_path.write_text(json.dumps({
        "sample_id": "template_s000000",
        "signal_files": {
            "iso_002": "signals/template_s000000_iso_000.npy",
            "iso_001_reg": "signals/template_s000000_iso_001.npy",
            "aniso_001": "signals/template_s000000_aniso_000.npy",
        },
    }))

    normalized = add_path_labels(str(label_path), str(root))

    assert normalized["paths"] == {
        "mesh": "meshes/template_s000000.obj",
        "signal": "signals/template_s000000_iso_001.npy",
        "label": "labels/template_s000000.json",
        "sphere": "spheres/template_s000000.obj",
        "spherical_label": "labels/template_s000000_spherical.json",
    }
    assert normalized["mesh_path"] == normalized["paths"]["mesh"]
    assert normalized["signal_path"] == normalized["paths"]["signal"]
    assert normalized["label_path"] == normalized["paths"]["label"]
    assert normalized["sphere_path"] == normalized["paths"]["sphere"]


def test_add_path_labels_uses_mnist_signal_and_updates_legacy_aliases(tmp_path: Path):
    root = tmp_path / "generated"
    labels = root / "labels"
    labels.mkdir(parents=True)
    label_path = labels / "mnist_s000000.json"
    label_path.write_text(json.dumps({
        "sample_id": "mnist_s000000",
        "mesh_file": "obsolete.obj",
        "signal_file": "obsolete.npy",
        "signal": {"signal_file": "signals/mnist_s000000_mnist.npy"},
    }))

    normalized = add_path_labels(str(label_path), str(root))

    assert normalized["paths"]["signal"] == "signals/mnist_s000000_mnist.npy"
    assert normalized["mesh_file"] == "meshes/mnist_s000000.obj"
    assert normalized["signal_file"] == "signals/mnist_s000000_mnist.npy"
