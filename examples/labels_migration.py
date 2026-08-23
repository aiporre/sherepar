#!/usr/bin/env python3
"""Normalize path fields in canonical dataset labels.

Usage
-----
    python examples/labels_migration.py data/generated

The script updates only primary ``labels/<sample_id>.json`` files.  It skips
per-signal and spherical-parametrization metadata labels, then delegates the
in-place rewrite to :func:`spherepar.benchmark.dataset_generator.add_path_labels`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherepar.benchmark.dataset_generator import add_path_labels  # noqa: E402


def iter_primary_labels(labels_dir: Path) -> Iterable[Path]:
    """Yield canonical labels while excluding generated auxiliary labels."""
    for label_path in sorted(labels_dir.glob("*.json")):
        name = label_path.name
        if name.endswith("_signal.json") or name.endswith("_spherical.json"):
            continue
        if "_iso_" in name or "_aniso_" in name:
            continue
        yield label_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize canonical paths and flat path aliases in dataset labels."
    )
    parser.add_argument(
        "dataset_root",
        help="Dataset root containing labels/, meshes/, signals/, and optionally spheres/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List primary labels that would be migrated without rewriting them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    labels_dir = dataset_root / "labels"
    if not labels_dir.is_dir():
        print(f"ERROR: labels directory not found: {labels_dir}")
        return 1

    label_paths = list(iter_primary_labels(labels_dir))
    if not label_paths:
        print(f"No primary label JSON files found in {labels_dir}")
        return 0

    migrated = 0
    failed = 0
    for label_path in label_paths:
        if args.dry_run:
            print(f"would migrate {label_path.name}")
            continue
        try:
            add_path_labels(str(label_path), str(dataset_root))
            print(f"migrated {label_path.name}")
            migrated += 1
        except Exception as exc:  # noqa: BLE001
            print(f"failed {label_path.name}: {exc}")
            failed += 1

    if args.dry_run:
        print(f"Dry run: {len(label_paths)} primary label(s) would be migrated.")
    else:
        print(f"Done. migrated={migrated}, failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
