"""Task-specific fold/split builder for unified datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

TASK_NUMBER_OF_CENTERS = "number_of_centers"
TASK_CENTER_REGRESSION = "center_regression"
TASK_SIGMA_REGRESSION = "sigma_regression"
TASK_AMPLITUDE_REGRESSION = "amplitude_regression"
TASK_MNIST_CLS = "mnist_cls"
TASK_MODELNET40_CLS = "modelnet40_cls"
TASK_ADNI_CLS = "adni_cls"

DEFAULT_TASKS = [
    TASK_NUMBER_OF_CENTERS,
    TASK_CENTER_REGRESSION,
    TASK_SIGMA_REGRESSION,
    TASK_AMPLITUDE_REGRESSION,
]


def is_valid_for_number_of_centers(label: Dict[str, Any]) -> bool:
    num_centers = int(label.get("signal", {}).get("num_centers", -1))
    return num_centers in [1, 2, 3, 4, 5]


def is_valid_for_center_regression(label: Dict[str, Any]) -> bool:
    return int(label.get("signal", {}).get("num_centers", -1)) == 1


def is_valid_for_sigma_regression(label: Dict[str, Any]) -> bool:
    return int(label.get("signal", {}).get("num_centers", -1)) == 1


def is_valid_for_amplitude_regression(label: Dict[str, Any]) -> bool:
    return int(label.get("signal", {}).get("num_centers", -1)) == 1


def is_valid_for_mnist_cls(label: Dict[str, Any]) -> bool:
    task_entry = label.get("tasks", {}).get(TASK_MNIST_CLS, {})
    if isinstance(task_entry, dict) and task_entry.get("valid") is True:
        mnist_index = label.get("signal", {}).get("mnist_index")
        return isinstance(mnist_index, int) and 0 <= mnist_index < 70000
    return False


def is_valid_for_modelnet40_cls(label: Dict[str, Any]) -> bool:
    task_entry = label.get("tasks", {}).get(TASK_MODELNET40_CLS, {})
    if not isinstance(task_entry, dict) or task_entry.get("valid") is not True:
        return False
    return isinstance(task_entry.get("label"), int)


def is_valid_for_adni_cls(label: Dict[str, Any]) -> bool:
    task_entry = label.get("tasks", {}).get(TASK_ADNI_CLS, {})
    if not isinstance(task_entry, dict) or task_entry.get("valid") is not True:
        return False
    participant_id = task_entry.get("participant_id")
    class_id = task_entry.get("label")
    return isinstance(participant_id, str) and bool(participant_id) and isinstance(class_id, int) and class_id in (0, 1, 2)


TASK_FILTERS = {
    TASK_NUMBER_OF_CENTERS: is_valid_for_number_of_centers,
    TASK_CENTER_REGRESSION: is_valid_for_center_regression,
    TASK_SIGMA_REGRESSION: is_valid_for_sigma_regression,
    TASK_AMPLITUDE_REGRESSION: is_valid_for_amplitude_regression,
    TASK_MNIST_CLS: is_valid_for_mnist_cls,
    TASK_MODELNET40_CLS: is_valid_for_modelnet40_cls,
    TASK_ADNI_CLS: is_valid_for_adni_cls,
}


def _load_labels(labels_dir: Path) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    for path in sorted(labels_dir.glob("*.json")):
        if path.name.endswith("_signal.json") or path.name.endswith("_spherical.json"):
            continue
        with open(path, "r") as fh:
            data = json.load(fh)
        data["_label_file"] = str(path)
        sample_id = data.get("sample_id", data.get("name"))
        if sample_id is not None:
            data["sample_id"] = str(sample_id)
        labels.append(data)
    return labels


def _balance_number_of_centers(records: List[Dict[str, Any]], rng: np.random.Generator) -> List[Dict[str, Any]]:
    by_class: Dict[int, List[Dict[str, Any]]] = {k: [] for k in [1, 2, 3, 4, 5]}
    for rec in records:
        cls = int(rec.get("signal", {}).get("num_centers", -1))
        if cls in by_class:
            by_class[cls].append(rec)
    non_empty = [v for v in by_class.values() if len(v) > 0]
    if not non_empty:
        return []
    min_count = min(len(v) for v in non_empty)
    balanced: List[Dict[str, Any]] = []
    for cls in [1, 2, 3, 4, 5]:
        cls_records = by_class[cls]
        if not cls_records:
            continue
        idx = rng.permutation(len(cls_records))[:min_count]
        balanced.extend([cls_records[int(i)] for i in idx])
    return balanced


def _grouped_split(
    records: Sequence[Dict[str, Any]],
    num_folds: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_by_template: bool = True,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """Return per-fold (train_ids, val_ids, test_ids)."""
    if len(records) == 0:
        return [([], [], []) for _ in range(num_folds)]

    if train_ratio <= 0 or test_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio and test_ratio must be positive; val_ratio must be non-negative")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if group_by_template:
        groups: Dict[str, List[str]] = {}
        for rec in records:
            gid = str(rec.get("template_id", "unknown_template"))
            groups.setdefault(gid, []).append(str(rec["sample_id"]))
        group_keys = sorted(groups.keys())
        ordered_groups = [groups[k] for k in group_keys]
    else:
        ordered_groups = [[str(rec["sample_id"])] for rec in records]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ordered_groups))
    ordered_groups = [ordered_groups[int(i)] for i in order]
    n_groups = len(ordered_groups)
    test_group_count = max(1, int(round(n_groups * test_ratio)))

    folds: List[Tuple[List[str], List[str], List[str]]] = []
    for fold_idx in range(num_folds):
        start = (fold_idx * test_group_count) % n_groups
        test_idx = set((start + i) % n_groups for i in range(test_group_count))
        test_groups = [ordered_groups[i] for i in range(n_groups) if i in test_idx]
        remain_groups = [ordered_groups[i] for i in range(n_groups) if i not in test_idx]

        remain_n = len(remain_groups)
        if remain_n == 0:
            folds.append(([], [], [sid for g in test_groups for sid in g]))
            continue

        # train vs val split on remaining groups
        val_share_in_remain = val_ratio / (train_ratio + val_ratio)
        val_count = (
            max(1, int(round(remain_n * val_share_in_remain)))
            if val_ratio > 0 and remain_n > 1
            else 0
        )
        val_count = min(val_count, max(remain_n - 1, 0))
        val_groups = remain_groups[:val_count]
        train_groups = remain_groups[val_count:]

        train_ids = sorted([sid for g in train_groups for sid in g])
        val_ids = sorted([sid for g in val_groups for sid in g])
        test_ids = sorted([sid for g in test_groups for sid in g])
        folds.append((train_ids, val_ids, test_ids))
    return folds


def _mnist_native_split(records: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    train_ids: List[str] = []
    test_ids: List[str] = []
    for rec in records:
        sample_id = str(rec["sample_id"])
        mnist_index = int(rec["signal"]["mnist_index"])
        if mnist_index < 60000:
            train_ids.append(sample_id)
        else:
            test_ids.append(sample_id)

    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)
    # Keep validation aligned with the MNIST train partition, as requested.
    val_ids = list(train_ids)
    return train_ids, val_ids, test_ids


def _modelnet40_native_split(records: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """Preserve ModelNet40's supplied train/test partition."""
    train_ids = sorted(
        str(rec["sample_id"])
        for rec in records
        if rec.get("metadata", {}).get("source_split") == "train"
    )
    test_ids = sorted(
        str(rec["sample_id"])
        for rec in records
        if rec.get("metadata", {}).get("source_split") == "test"
    )
    return train_ids, [], test_ids


def _adni_grouped_split(
    records: Sequence[Dict[str, Any]],
    num_folds: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """Participant-stratified splits for ADNI clinical classification.

    Every participant is assigned to exactly one partition in each fold; all
    sessions and hip sides therefore remain together. Classes are sampled
    independently so each partition is approximately class balanced.
    """
    if train_ratio <= 0 or test_ratio <= 0 or val_ratio < 0:
        raise ValueError("train_ratio and test_ratio must be positive; val_ratio must be non-negative")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    by_participant: Dict[str, List[Dict[str, Any]]] = {}
    class_by_participant: Dict[str, int] = {}
    for rec in records:
        task_entry = rec.get("tasks", {}).get(TASK_ADNI_CLS, {})
        participant = str(task_entry.get("participant_id", ""))
        if not participant:
            raise ValueError(f"ADNI record {rec.get('sample_id')} is missing participant_id")
        class_id = int(task_entry["label"])
        prior = class_by_participant.setdefault(participant, class_id)
        if prior != class_id:
            raise ValueError(f"Participant {participant} has inconsistent ADNI class labels")
        by_participant.setdefault(participant, []).append(rec)

    by_class: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for participant, class_id in class_by_participant.items():
        by_class.setdefault(class_id, []).append(participant)
    rng = np.random.default_rng(seed)
    shuffled: Dict[int, List[str]] = {
        class_id: [by_class[class_id][int(i)] for i in rng.permutation(len(by_class[class_id]))]
        for class_id in sorted(by_class)
    }

    folds: List[Tuple[List[str], List[str], List[str]]] = []
    for fold_idx in range(num_folds):
        train_participants: List[str] = []
        val_participants: List[str] = []
        test_participants: List[str] = []
        for class_id in sorted(shuffled):
            participants = shuffled[class_id]
            n = len(participants)
            if n == 0:
                continue
            n_test = max(1, int(round(n * test_ratio))) if n > 1 else 0
            n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 and n > 2 else 0
            while n_test + n_val >= n and n > 1:
                if n_val > 0:
                    n_val -= 1
                else:
                    n_test -= 1
            offset = (fold_idx * max(n_test, 1)) % n
            rotated = participants[offset:] + participants[:offset]
            test_participants.extend(rotated[:n_test])
            val_participants.extend(rotated[n_test : n_test + n_val])
            train_participants.extend(rotated[n_test + n_val :])

        def sample_ids(participants: Sequence[str]) -> List[str]:
            return sorted(str(rec["sample_id"]) for participant in participants for rec in by_participant[participant])

        folds.append((sample_ids(train_participants), sample_ids(val_participants), sample_ids(test_participants)))
    return folds


def build_task_splits(
    dataset_root: str,
    tasks: Iterable[str] | None = None,
    num_folds: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
    group_by_template: bool = True,
    modelnet40_native_split: bool = False,
) -> Dict[str, Any]:
    """Build and save per-task fold split files under dataset_root/folds."""
    if tasks is None:
        tasks = DEFAULT_TASKS
    task_list = list(tasks)
    for t in task_list:
        if t not in TASK_FILTERS:
            raise ValueError(f"Unsupported task: {t}")

    root = Path(dataset_root)
    labels_dir = root / "labels"
    folds_dir = root / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    labels = _load_labels(labels_dir)
    summary: Dict[str, Any] = {
        "dataset_root": str(root),
        "num_labels": len(labels),
        "num_folds": int(num_folds),
        "seed": int(seed),
        "group_by_template": bool(group_by_template),
        "tasks": {},
    }

    for task_name in task_list:
        valid = [rec for rec in labels if TASK_FILTERS[task_name](rec)]
        rng = np.random.default_rng(seed)
        if task_name == TASK_NUMBER_OF_CENTERS:
            valid = _balance_number_of_centers(valid, rng)

        if task_name == TASK_MNIST_CLS:
            train_ids, val_ids, test_ids = _mnist_native_split(valid)
            folds = [(train_ids, val_ids, test_ids) for _ in range(num_folds)]
        elif task_name == TASK_MODELNET40_CLS and modelnet40_native_split:
            train_ids, val_ids, test_ids = _modelnet40_native_split(valid)
            folds = [(train_ids, val_ids, test_ids) for _ in range(num_folds)]
        elif task_name == TASK_ADNI_CLS:
            folds = _adni_grouped_split(
                records=valid,
                num_folds=num_folds,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )
        else:
            folds = _grouped_split(
                records=valid,
                num_folds=num_folds,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
                group_by_template=group_by_template,
            )
        task_summary = {
            "num_valid_samples": len(valid),
            "folds": [],
        }
        if task_name == TASK_NUMBER_OF_CENTERS:
            counts: Dict[int, int] = {}
            for rec in valid:
                cls = int(rec.get("signal", {}).get("num_centers", -1))
                counts[cls] = counts.get(cls, 0) + 1
            task_summary["class_counts"] = counts
        if task_name == TASK_MNIST_CLS:
            task_summary["split_source"] = "mnist_default"
        if task_name == TASK_MODELNET40_CLS and modelnet40_native_split:
            task_summary["split_source"] = "modelnet40_native"
        if task_name == TASK_ADNI_CLS:
            task_summary["split_source"] = "participant_stratified"
            task_summary["class_counts"] = {
                str(class_id): sum(1 for rec in valid if rec["tasks"][TASK_ADNI_CLS]["label"] == class_id)
                for class_id in (0, 1, 2)
            }
            participant_classes = {
                rec["tasks"][TASK_ADNI_CLS]["participant_id"]: rec["tasks"][TASK_ADNI_CLS]["label"]
                for rec in valid
            }
            task_summary["participant_counts"] = len(participant_classes)
            task_summary["class_participant_counts"] = {
                str(class_id): sum(1 for value in participant_classes.values() if value == class_id)
                for class_id in (0, 1, 2)
            }

        for fold_idx, (train_ids, val_ids, test_ids) in enumerate(folds, start=1):
            task_dir = folds_dir / f"fold{fold_idx}" / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
                with open(task_dir / f"{split_name}.txt", "w") as fh:
                    fh.write("\n".join(ids))
                    fh.write("\n" if ids else "")
            task_summary["folds"].append(
                {
                    "fold": fold_idx,
                    "train_count": len(train_ids),
                    "val_count": len(val_ids),
                    "test_count": len(test_ids),
                }
            )
        summary["tasks"][task_name] = task_summary

    summary_path = folds_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
