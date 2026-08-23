#!/usr/bin/env python3
"""Print the class distribution of generated ModelNet40 labels.

Usage
-----
    python examples/count_modelnet40_classes.py data/imported_modelnet40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


TASK_NAME = "modelnet40_cls"
BAR_WIDTH = 40


def iter_primary_labels(labels_dir: Path) -> Iterable[Path]:
    """Yield primary sample labels, excluding generated auxiliary metadata."""
    for path in sorted(labels_dir.glob("*.json")):
        name = path.name
        if name.endswith("_signal.json") or name.endswith("_spherical.json"):
            continue
        if "_iso_" in name or "_aniso_" in name:
            continue
        yield path


def count_modelnet40_classes(labels_dir: Path) -> Tuple[Dict[Tuple[int, str], Dict[str, int]], int, int]:
    """Return per-class counts, number of valid labels, and malformed labels."""
    counts: Dict[Tuple[int, str], Dict[str, int]] = defaultdict(
        lambda: {"train": 0, "test": 0, "total": 0}
    )
    valid = 0
    malformed = 0

    for path in iter_primary_labels(labels_dir):
        try:
            with open(path, "r") as fh:
                label = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: could not read {path.name}: {exc}", file=sys.stderr)
            malformed += 1
            continue

        task = label.get("tasks", {}).get(TASK_NAME, {})
        metadata = label.get("metadata", {})
        if not isinstance(task, dict) or task.get("valid") is not True:
            continue
        class_id = task.get("label", metadata.get("class_id"))
        class_name = metadata.get("class_name", task.get("class_name"))
        if not isinstance(class_id, int) or not isinstance(class_name, str):
            print(f"WARNING: incomplete ModelNet40 class metadata in {path.name}", file=sys.stderr)
            malformed += 1
            continue

        entry = counts[(class_id, class_name)]
        entry["total"] += 1
        source_split = metadata.get("source_split")
        if source_split in {"train", "test"}:
            entry[source_split] += 1
        valid += 1

    return dict(counts), valid, malformed


def print_distribution(counts: Dict[Tuple[int, str], Dict[str, int]], total: int) -> None:
    """Print tabular and ASCII-bar ModelNet40 class distributions."""
    print("ModelNet40 class counts")
    print(f"{'ID':>3}  {'Class':<24} {'Train':>7} {'Test':>7} {'Total':>7}")
    print("-" * 58)
    for (class_id, class_name), values in sorted(counts.items()):
        print(
            f"{class_id:>3}  {class_name:<24} {values['train']:>7} "
            f"{values['test']:>7} {values['total']:>7}"
        )
    train_total = sum(values["train"] for values in counts.values())
    test_total = sum(values["test"] for values in counts.values())
    print("-" * 58)
    print(f"{'All classes':<29} {train_total:>7} {test_total:>7} {total:>7}")

    print("\nClass share of all samples")
    if total == 0:
        print("No valid ModelNet40 classification labels found.")
        return
    for (_, class_name), values in sorted(counts.items()):
        percentage = 100.0 * values["total"] / total
        filled = round(BAR_WIDTH * values["total"] / total)
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        print(f"{class_name:<24}: {bar} | {percentage:5.1f}%")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count generated ModelNet40 classes from label JSON files.")
    parser.add_argument("dataset_root", help="Generated dataset root containing labels/.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    labels_dir = Path(args.dataset_root).expanduser().resolve() / "labels"
    if not labels_dir.is_dir():
        print(f"ERROR: labels directory not found: {labels_dir}", file=sys.stderr)
        return 1

    counts, total, malformed = count_modelnet40_classes(labels_dir)
    print_distribution(counts, total)
    if malformed:
        print(f"\nWarnings: skipped {malformed} malformed label(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
