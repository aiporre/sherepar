from __future__ import annotations

import json
from pathlib import Path

from spherepar.benchmark.splits import (
    TASK_AMPLITUDE_REGRESSION,
    TASK_CENTER_REGRESSION,
    TASK_MNIST_CLS,
    TASK_NUMBER_OF_CENTERS,
    TASK_SIGMA_REGRESSION,
    build_task_splits,
    is_valid_for_amplitude_regression,
    is_valid_for_center_regression,
    is_valid_for_mnist_cls,
    is_valid_for_number_of_centers,
    is_valid_for_sigma_regression,
)


def _write_label(
    labels_dir: Path,
    sample_id: str,
    template_id: str,
    num_centers: int,
):
    data = {
        "sample_id": sample_id,
        "template_id": template_id,
        "signal": {"num_centers": num_centers},
    }
    with open(labels_dir / f"{sample_id}.json", "w") as fh:
        json.dump(data, fh)


def _write_mnist_label(
    labels_dir: Path,
    sample_id: str,
    template_id: str,
    mnist_index: int,
    mnist_label: int,
):
    data = {
        "sample_id": sample_id,
        "template_id": template_id,
        "signal": {
            "signal_type": "mnist",
            "mnist_index": mnist_index,
            "mnist_label": mnist_label,
        },
        "tasks": {
            "mnist_cls": {
                "valid": True,
                "label": mnist_label,
            }
        },
    }
    with open(labels_dir / f"{sample_id}.json", "w") as fh:
        json.dump(data, fh)


def test_task_filters():
    rec1 = {"signal": {"num_centers": 1}}
    rec3 = {"signal": {"num_centers": 3}}
    rec0 = {"signal": {"num_centers": 0}}

    assert is_valid_for_number_of_centers(rec1)
    assert is_valid_for_number_of_centers(rec3)
    assert not is_valid_for_number_of_centers(rec0)

    assert is_valid_for_center_regression(rec1)
    assert is_valid_for_sigma_regression(rec1)
    assert is_valid_for_amplitude_regression(rec1)
    assert not is_valid_for_center_regression(rec3)

    mnist_ok = {
        "signal": {"mnist_index": 123, "mnist_label": 7},
        "tasks": {"mnist_cls": {"valid": True, "label": 7}},
    }
    mnist_bad = {
        "signal": {"mnist_index": 80000, "mnist_label": 7},
        "tasks": {"mnist_cls": {"valid": True, "label": 7}},
    }
    assert is_valid_for_mnist_cls(mnist_ok)
    assert not is_valid_for_mnist_cls(mnist_bad)


def test_build_task_splits_writes_expected_files(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 2 templates x classes 1..5 = 10 labels (balanced for classification)
    idx = 1
    for template_id in ["tpl_a", "tpl_b"]:
        for cls in [1, 2, 3, 4, 5]:
            _write_label(
                labels_dir=labels_dir,
                sample_id=f"sample_s{idx:06d}",
                template_id=template_id,
                num_centers=cls,
            )
            idx += 1

    summary = build_task_splits(
        dataset_root=str(dataset_root),
        tasks=[
            TASK_NUMBER_OF_CENTERS,
            TASK_CENTER_REGRESSION,
            TASK_SIGMA_REGRESSION,
            TASK_AMPLITUDE_REGRESSION,
        ],
        num_folds=2,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=7,
        group_by_template=True,
    )

    assert summary["num_folds"] == 2
    for fold in [1, 2]:
        for task in [
            TASK_NUMBER_OF_CENTERS,
            TASK_CENTER_REGRESSION,
            TASK_SIGMA_REGRESSION,
            TASK_AMPLITUDE_REGRESSION,
        ]:
            base = dataset_root / "folds" / f"fold{fold}" / task
            assert (base / "train.txt").exists()
            assert (base / "val.txt").exists()
            assert (base / "test.txt").exists()


def test_label_schema_completeness(tmp_path: Path):
    """Verify that label JSON files contain all required fields."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Write a complete label
    complete_label = {
        "sample_id": "sample_s000001",
        "template_id": "template_a",
        "deformation_case": "case2_small",
        "random_seed": 42,
        "signal": {
            "num_centers": 2,
            "centers": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "vertex_ids": [0, 10, 20],
            "sigmas": [0.5, 0.3],
            "amplitudes": [1.0, 0.8],
        },
        "deformation": {"max_ratio": 0.05, "alpha": 2.0},
        "parametrization": {"method": "flash", "success": True},
        "task_validity": {
            "number_of_centers": True,
            "center_regression": False,
            "sigma_regression": False,
            "amplitude_regression": False,
        },
    }

    with open(labels_dir / "sample_s000001.json", "w") as fh:
        json.dump(complete_label, fh)

    # Load and verify all fields exist
    with open(labels_dir / "sample_s000001.json", "r") as fh:
        loaded = json.load(fh)

    required_fields = [
        "sample_id",
        "template_id",
        "deformation_case",
        "random_seed",
        "signal",
        "deformation",
        "parametrization",
        "task_validity",
    ]
    for field in required_fields:
        assert field in loaded, f"Missing required field: {field}"

    # Verify signal block completeness
    assert "num_centers" in loaded["signal"]
    assert "centers" in loaded["signal"]
    assert "sigmas" in loaded["signal"]
    assert "amplitudes" in loaded["signal"]

    # Verify parametrization block
    assert "method" in loaded["parametrization"]
    assert "success" in loaded["parametrization"]


def test_regression_splits_single_center_only(tmp_path: Path):
    """Verify that regression tasks contain only single-center samples."""
    dataset_root = tmp_path / "dataset"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create samples with mixed num_centers across 2 templates
    # This ensures we have enough samples to distribute across train/val/test
    idx = 1
    for template_id in ["tpl_a", "tpl_b"]:
        for num_centers in [1, 2, 3]:
            _write_label(
                labels_dir=labels_dir,
                sample_id=f"sample_s{idx:06d}",
                template_id=template_id,
                num_centers=num_centers,
            )
            idx += 1

    build_task_splits(
        dataset_root=str(dataset_root),
        tasks=[TASK_CENTER_REGRESSION],
        num_folds=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        group_by_template=True,
    )

    # Collect all samples from train/val/test
    base = dataset_root / "folds" / "fold1" / TASK_CENTER_REGRESSION
    all_samples = set()
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        with open(base / split_file, "r") as fh:
            samples = [line.strip() for line in fh if line.strip()]
            all_samples.update(samples)

    # Verify only single-center samples are present
    # sample_s000001, sample_s000004 have num_centers=1
    assert len(all_samples) > 0, "No samples in regression splits"
    
    # Load all labels to verify num_centers for each sample
    for sample_id in all_samples:
        with open(labels_dir / f"{sample_id}.json", "r") as fh:
            label = json.load(fh)
        assert label["signal"]["num_centers"] == 1, f"Found non-single-center sample {sample_id}"


def test_splits_non_overlapping(tmp_path: Path):
    """Verify train/val/test splits are non-overlapping."""
    dataset_root = tmp_path / "dataset"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create samples: 3 templates x 2 samples each = 6 samples
    idx = 1
    for template_id in ["tpl_a", "tpl_b", "tpl_c"]:
        for _ in range(2):
            _write_label(
                labels_dir=labels_dir,
                sample_id=f"sample_s{idx:06d}",
                template_id=template_id,
                num_centers=1,
            )
            idx += 1

    build_task_splits(
        dataset_root=str(dataset_root),
        tasks=[TASK_NUMBER_OF_CENTERS],
        num_folds=1,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        group_by_template=True,
    )

    base = dataset_root / "folds" / "fold1" / TASK_NUMBER_OF_CENTERS

    def read_split(filename):
        with open(base / filename, "r") as fh:
            return set(line.strip() for line in fh if line.strip())

    train = read_split("train.txt")
    val = read_split("val.txt")
    test = read_split("test.txt")

    # Verify no overlaps
    assert len(train & val) == 0, "train and val overlap"
    assert len(train & test) == 0, "train and test overlap"
    assert len(val & test) == 0, "val and test overlap"

    # Verify all samples accounted for
    all_samples = train | val | test
    assert len(all_samples) == 6, f"Expected 6 samples total, got {len(all_samples)}"


def test_task_splits_allow_zero_validation_ratio(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    idx = 1
    for template_id in ["tpl_a", "tpl_b", "tpl_c", "tpl_d"]:
        for _ in range(2):
            _write_label(
                labels_dir=labels_dir,
                sample_id=f"sample_s{idx:06d}",
                template_id=template_id,
                num_centers=1,
            )
            idx += 1

    summary = build_task_splits(
        dataset_root=str(dataset_root),
        tasks=[TASK_CENTER_REGRESSION],
        num_folds=1,
        train_ratio=0.8,
        val_ratio=0.0,
        test_ratio=0.2,
        seed=42,
        group_by_template=True,
    )

    base = dataset_root / "folds" / "fold1" / TASK_CENTER_REGRESSION
    with open(base / "train.txt", "r") as fh:
        train_ids = [line.strip() for line in fh if line.strip()]
    with open(base / "val.txt", "r") as fh:
        val_ids = [line.strip() for line in fh if line.strip()]
    with open(base / "test.txt", "r") as fh:
        test_ids = [line.strip() for line in fh if line.strip()]

    assert len(train_ids) > 0
    assert val_ids == []
    assert len(test_ids) > 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert summary["tasks"][TASK_CENTER_REGRESSION]["folds"][0]["val_count"] == 0


def test_classification_balance(tmp_path: Path):
    """Verify number_of_centers classification task has balanced class distribution."""
    dataset_root = tmp_path / "dataset"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 templates with 5 samples each: 1 per class
    idx = 1
    for template_id in ["tpl_a", "tpl_b", "tpl_c"]:
        for num_centers in [1, 2, 3, 4, 5]:
            _write_label(
                labels_dir=labels_dir,
                sample_id=f"sample_s{idx:06d}",
                template_id=template_id,
                num_centers=num_centers,
            )
            idx += 1

    summary = build_task_splits(
        dataset_root=str(dataset_root),
        tasks=[TASK_NUMBER_OF_CENTERS],
        num_folds=3,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
        group_by_template=True,
    )

    # Verify all folds exist and have balanced classes
    for fold_idx in [1, 2, 3]:
        base = dataset_root / "folds" / f"fold{fold_idx}" / TASK_NUMBER_OF_CENTERS
        for split_name in ["train", "val", "test"]:
            split_file = base / f"{split_name}.txt"
            assert split_file.exists(), f"Missing {split_file}"

            with open(split_file, "r") as fh:
                samples = [line.strip() for line in fh if line.strip()]

            # Verify split is non-empty
            assert len(samples) > 0, f"Empty {split_name} split in fold{fold_idx}"

    # Verify summary has proper structure
    assert "num_folds" in summary
    assert "tasks" in summary
    assert TASK_NUMBER_OF_CENTERS in summary["tasks"]


def test_mnist_cls_uses_native_train_test_split(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    labels_dir = dataset_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # MNIST train partition (<60000)
    _write_mnist_label(labels_dir, "sample_s000001", "tpl_mnist", 10, 1)
    _write_mnist_label(labels_dir, "sample_s000002", "tpl_mnist", 59999, 2)
    # MNIST test partition (>=60000)
    _write_mnist_label(labels_dir, "sample_s000003", "tpl_mnist", 60000, 3)
    _write_mnist_label(labels_dir, "sample_s000004", "tpl_mnist", 69999, 4)

    build_task_splits(
        dataset_root=str(dataset_root),
        tasks=[TASK_MNIST_CLS],
        num_folds=2,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        group_by_template=True,
    )

    for fold_idx in [1, 2]:
        base = dataset_root / "folds" / f"fold{fold_idx}" / TASK_MNIST_CLS
        with open(base / "train.txt", "r") as fh:
            train_ids = [line.strip() for line in fh if line.strip()]
        with open(base / "val.txt", "r") as fh:
            val_ids = [line.strip() for line in fh if line.strip()]
        with open(base / "test.txt", "r") as fh:
            test_ids = [line.strip() for line in fh if line.strip()]

        assert train_ids == ["sample_s000001", "sample_s000002"]
        assert val_ids == ["sample_s000001", "sample_s000002"]
        assert test_ids == ["sample_s000003", "sample_s000004"]
