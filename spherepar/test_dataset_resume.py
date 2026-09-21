from __future__ import annotations

import json
from pathlib import Path

import pytest

from spherepar.benchmark.dataset_generator import (
    add_path_labels,
    build_arg_parser,
    generate_dataset,
    parametrization_request_matches,
    _list_completed_samples,
    _next_sample_index,
)
from examples.script_to_generate_dataset_from_files import (
    build_arg_parser as build_files_arg_parser,
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


def test_mobius_center_cli_flag_is_opt_in():
    parser = build_arg_parser()
    assert parser.parse_args(["input"]).mobius_center is False
    args = parser.parse_args(["input", "--param-method", "cem", "--mobius-center"])
    assert args.mobius_center is True


def test_cem_radius_cli_defaults_to_paper_value_and_accepts_override():
    parser = build_arg_parser()
    assert parser.parse_args(["input"]).cem_radius == 1.2
    assert parser.parse_args(["input", "--cem-radius", "1.7"]).cem_radius == 1.7


def test_anchor_diagnostics_cli_flag_is_opt_in():
    parser = build_arg_parser()
    assert parser.parse_args(["input"]).anchor_diagnostics is False
    assert parser.parse_args(["input", "--anchor-diagnostics"]).anchor_diagnostics is True


@pytest.mark.parametrize("parser", [build_arg_parser(), build_files_arg_parser()])
def test_phase2_cli_defaults_and_overrides(parser):
    argv = ["input"]
    if any(action.dest == "output_root" and action.required for action in parser._actions):
        argv = ["--input-dir", "input", "--output-root", "output"]
    defaults = parser.parse_args(argv)
    assert defaults.use_idt_remesh is False
    assert defaults.adaptive_radius is False
    assert defaults.cem_radius_candidates == (1.1, 1.3, 1.4, 1.5)
    assert defaults.reject_retry is False
    assert defaults.cem_max_attempts == 5
    assert defaults.cem_max_collapsed_faces == 0
    configured = parser.parse_args(argv + [
        "--use-idt-remesh", "--adaptive-radius", "--cem-radius-candidates", "1.05,1.25",
        "--reject-retry", "--cem-max-attempts", "3", "--cem-max-collapsed-faces", "2",
    ])
    assert configured.use_idt_remesh and configured.adaptive_radius and configured.reject_retry
    assert configured.cem_radius_candidates == (1.05, 1.25)
    assert configured.cem_max_attempts == 3
    assert configured.cem_max_collapsed_faces == 2


@pytest.mark.parametrize("parser", [build_arg_parser(), build_files_arg_parser()])
def test_anchor_strategy_cli_defaults_and_overrides(parser):
    argv = ["input"] if parser.prog != "pytest" else ["input"]
    if parser is not None and any(action.dest == "output_root" and action.required for action in parser._actions):
        argv = ["--input-dir", "input", "--output-root", "output"]
    defaults = parser.parse_args(argv)
    assert defaults.anchor_strategy == "regular"
    assert defaults.anchor_regularity_percentile == 10.0
    overridden = parser.parse_args(
        argv + [
            "--anchor-strategy", "central_regular",
            "--anchor-regularity-percentile", "23.5",
        ]
    )
    assert overridden.anchor_strategy == "central_regular"
    assert overridden.anchor_regularity_percentile == 23.5


def test_resume_cem_radius_treats_missing_value_as_legacy_one():
    legacy = {"parametrization": {"method": "cem", "mobius_center": False}}
    current = {
        "parametrization": {
            "method": "cem",
            "mobius_center": False,
            "cem_radius": 1.2,
        }
    }

    assert parametrization_request_matches(legacy, "cem", False, 1.0)
    assert not parametrization_request_matches(legacy, "cem", False, 1.2)
    assert parametrization_request_matches(current, "cem", False, 1.2)
    assert not parametrization_request_matches(current, "cem", False, 1.0)


def test_resume_anchor_diagnostics_requires_a_diagnostic_artifact_only_when_requested():
    ordinary = {
        "parametrization": {
            "method": "cem",
            "mobius_center": False,
            "cem_radius": 1.2,
            "anchor_diagnostics": False,
        }
    }
    diagnostic = {
        "parametrization": {
            "method": "cem",
            "mobius_center": False,
            "cem_radius": 1.2,
            "anchor_diagnostics": True,
        }
    }
    assert not parametrization_request_matches(ordinary, "cem", False, 1.2, True)
    assert parametrization_request_matches(diagnostic, "cem", False, 1.2, True)
    assert parametrization_request_matches(diagnostic, "cem", False, 1.2, False)


def test_resume_anchor_strategy_uses_legacy_regular_and_compares_central_percentile():
    legacy = {
        "parametrization": {
            "method": "cem",
            "mobius_center": False,
            "cem_radius": 1.2,
        }
    }
    central = {
        "parametrization": {
            "method": "cem",
            "mobius_center": False,
            "cem_radius": 1.2,
            "anchor_strategy": "central_regular",
            "anchor_regularity_percentile": 12.5,
        }
    }
    regular_nondefault_percentile = {
        "parametrization": {
            "method": "cem",
            "mobius_center": False,
            "cem_radius": 1.2,
            "anchor_strategy": "regular",
            "anchor_regularity_percentile": 99.0,
        }
    }

    assert parametrization_request_matches(legacy, "cem", False, 1.2)
    assert not parametrization_request_matches(
        legacy, "cem", False, 1.2, False, "central_regular", 10.0
    )
    assert parametrization_request_matches(
        central, "cem", False, 1.2, False, "central_regular", 12.5
    )
    assert not parametrization_request_matches(
        central, "cem", False, 1.2, False, "central_regular", 10.0
    )
    assert parametrization_request_matches(
        regular_nondefault_percentile, "cem", False, 1.2, False, "regular", 10.0
    )


def test_resume_phase2_fields_are_compared_only_when_relevant():
    legacy = {"parametrization": {"method": "cem", "cem_radius": 1.2}}
    configured = {"parametrization": {
        "method": "cem", "cem_radius": 1.2, "adaptive_radius": True,
        "cem_radius_candidates": [1.1, 1.35], "cem_max_attempts": 3,
    }}
    assert parametrization_request_matches(legacy, "cem", False, 1.2)
    assert not parametrization_request_matches(legacy, "cem", False, 1.2, adaptive_radius=True)
    assert parametrization_request_matches(
        configured, "cem", False, 1.2,
        adaptive_radius=True, cem_radius_candidates=(1.1, 1.35), cem_max_attempts=3,
    )


def test_workers_must_be_positive():
    with pytest.raises(ValueError, match="workers must be at least 1"):
        generate_dataset("unused", workers=0)


def test_parallel_worker_config_propagates_anchor_options(tmp_path: Path, monkeypatch):
    from spherepar.benchmark import dataset_generator

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "template.obj").write_text("o template\n")
    captured = {}

    monkeypatch.setattr(dataset_generator, "_GRAPHOP_AVAILABLE", True)
    monkeypatch.setattr(
        dataset_generator,
        "load_meshes_from_directory",
        lambda _: [("template", object())],
    )

    def capture_parallel(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(dataset_generator, "_generate_dataset_parallel", capture_parallel)
    result = dataset_generator.generate_dataset(
        str(input_dir),
        output_root=str(tmp_path / "output"),
        n_samples_per_mesh=1,
        deformation_cases=["case2_small"],
        param_method="cem",
        anchor_strategy="central_regular",
        anchor_regularity_percentile=17.0,
        use_idt_remesh=True,
        adaptive_radius=True,
        reject_retry=True,
        cem_radius_candidates=(1.05, 1.25),
        cem_max_attempts=3,
        cem_max_collapsed_faces=1,
        resume=False,
        workers=2,
    )

    assert result == 1
    config = captured["tasks"][0]["config"]
    assert config["anchor_strategy"] == "central_regular"
    assert config["anchor_regularity_percentile"] == 17.0
    assert config["use_idt_remesh"] is True
    assert config["adaptive_radius"] is True
    assert config["reject_retry"] is True
    assert config["cem_radius_candidates"] == (1.05, 1.25)
    assert config["cem_max_attempts"] == 3
    assert config["cem_max_collapsed_faces"] == 1


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
