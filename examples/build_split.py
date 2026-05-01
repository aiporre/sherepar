#!/usr/bin/env python3
"""
Build stratified train/test/val splits for dataset with cross-validation folds.

This script reads label JSON files, groups by template_id for stratification,
and creates multiple folds with rotating test sets. Each fold has:
  - train.txt: training set (main data)
  - test.txt: test set (rotated per fold)
  - val.txt: validation set (random subset of train)
  - split_config.json: metadata for this fold

Usage:
    python build_split.py data/generated \
        --num-folds 5 \
        --test-ratio 0.2 \
        --val-ratio 0.2 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def load_labels(labels_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON label files from directory.
    
    Parameters
    ----------
    labels_dir : Path
        Directory containing label JSON files.
    
    Returns
    -------
    list of dict
        Each dict has 'sample_id', 'template_id', '_label_file' (filename).
    """
    labels = []
    for json_path in sorted(labels_dir.glob("*.json")):
        # Skip auxiliary files
        if json_path.name.endswith("_signal.json") or json_path.name.endswith("_spherical.json") or "_iso_" in json_path.name or "_aniso_" in json_path.name:
            continue
        
        try:
            with open(json_path, "r") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Warning: Could not load {json_path.name}: {e}")
            continue
        
        # Extract sample_id and template_id
        sample_id = data.get("sample_id", data.get("name", json_path.stem))
        template_id = data.get("metadata", {}).get("template_id", data.get("template_id", "unknown"))
        
        labels.append({
            "sample_id": str(sample_id),
            "template_id": str(template_id),
            "_label_file": json_path.name,  # Filename only
        })
    
    return labels


def group_by_template(labels: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group samples by template_id.
    
    Parameters
    ----------
    labels : list of dict
        Label records with 'sample_id', 'template_id', '_label_file'.
    
    Returns
    -------
    dict
        Maps template_id -> list of label records.
    """
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for label in labels:
        template_id = label["template_id"]
        groups[template_id].append(label)
    return dict(groups)


def stratified_train_test_split(
    labels: List[Dict[str, Any]],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split labels into train, test, and val using stratification by template.
    
    Strategy:
    1. Group by template_id
    2. Split templates at template level (not individual samples)
    3. Split remaining into train+val
    4. Randomly select val_ratio of train for validation
    
    Parameters
    ----------
    labels : list of dict
        All label records.
    test_ratio : float
        Fraction of data to use for test (e.g., 0.2).
    val_ratio : float
        Fraction of train+val to use for validation (e.g., 0.2).
    seed : int
        Random seed.
    
    Returns
    -------
    (train_labels, test_labels, val_labels) : tuple of lists
    """
    if not labels:
        return [], [], []
    
    rng = np.random.default_rng(seed)
    
    # Group by template
    template_groups = group_by_template(labels)
    template_ids = sorted(template_groups.keys())
    num_templates = len(template_ids)
    
    # Determine how many templates go to test
    test_template_count = max(1, int(round(num_templates * test_ratio)))
    test_template_count = min(test_template_count, num_templates - 1)  # Leave at least 1 for train
    
    # Randomly select templates for test
    perm = rng.permutation(num_templates)
    test_template_indices = perm[:test_template_count]
    train_val_template_indices = perm[test_template_count:]
    
    test_templates = {template_ids[int(i)] for i in test_template_indices}
    train_val_templates = {template_ids[int(i)] for i in train_val_template_indices}
    
    # Separate labels by template set
    test_labels = [l for l in labels if l["template_id"] in test_templates]
    train_val_labels = [l for l in labels if l["template_id"] in train_val_templates]
    
    # Split train_val into train and val
    if train_val_labels:
        train_labels, val_labels = train_test_split(
            train_val_labels,
            test_size=val_ratio,
            random_state=seed,
        )
    else:
        train_labels = []
        val_labels = []
    
    return train_labels, test_labels, val_labels


def create_cross_validation_folds(
    labels: List[Dict[str, Any]],
    num_folds: int,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Create multiple folds with rotating test sets.
    
    For each fold, a different subset of templates is used as test,
    with the remainder for train+val.
    
    Parameters
    ----------
    labels : list of dict
        All label records.
    num_folds : int
        Number of folds to create.
    test_ratio : float
        Fraction for test per fold.
    val_ratio : float
        Fraction for validation within each fold's train set.
    seed : int
        Random seed.
    
    Returns
    -------
    list of (train_labels, test_labels, val_labels)
        One tuple per fold.
    """
    if not labels:
        return [([], [], []) for _ in range(num_folds)]
    
    # Group by template
    template_groups = group_by_template(labels)
    template_ids = sorted(template_groups.keys())
    num_templates = len(template_ids)
    
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_templates)
    template_ids = [template_ids[int(i)] for i in order]
    
    # Compute how many templates per fold for test
    test_template_count = max(1, int(round(num_templates * test_ratio)))
    
    folds = []
    for fold_idx in range(num_folds):
        # Rotate which templates are test
        start = (fold_idx * test_template_count) % num_templates
        test_template_set = set(
            template_ids[(start + i) % num_templates] for i in range(test_template_count)
        )
        train_val_template_set = set(template_ids) - test_template_set
        
        # Collect labels by template set
        test_labels = [l for l in labels if l["template_id"] in test_template_set]
        train_val_labels = [l for l in labels if l["template_id"] in train_val_template_set]
        
        # Split train_val into train and val
        if train_val_labels:
            train_labels, val_labels = train_test_split(
                train_val_labels,
                test_size=val_ratio,
                random_state=seed + fold_idx,  # Different random state per fold
            )
        else:
            train_labels = []
            val_labels = []
        
        folds.append((train_labels, test_labels, val_labels))
    
    return folds


def write_fold_files(
    fold_dir: Path,
    train_labels: List[Dict[str, Any]],
    test_labels: List[Dict[str, Any]],
    val_labels: List[Dict[str, Any]],
) -> None:
    """Write train.txt, test.txt, val.txt files for a fold.
    
    Parameters
    ----------
    fold_dir : Path
        Directory for this fold.
    train_labels : list of dict
        Training set labels.
    test_labels : list of dict
        Test set labels.
    val_labels : list of dict
        Validation set labels.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, labels in [("train", train_labels), ("test", test_labels), ("val", val_labels)]:
        split_path = fold_dir / f"{split_name}.txt"
        filenames = sorted([label["_label_file"] for label in labels])
        with open(split_path, "w") as fh:
            for fname in filenames:
                fh.write(f"{fname}\n")


def write_split_config(
    fold_dir: Path,
    fold_idx: int,
    train_labels: List[Dict[str, Any]],
    test_labels: List[Dict[str, Any]],
    val_labels: List[Dict[str, Any]],
    test_ratio: float,
    val_ratio: float,
    seed: int,
    num_folds: int,
) -> None:
    """Write split_config.json metadata for a fold.
    
    Parameters
    ----------
    fold_dir : Path
        Directory for this fold.
    fold_idx : int
        Fold index (1-indexed).
    train_labels : list of dict
        Training set labels.
    test_labels : list of dict
        Test set labels.
    val_labels : list of dict
        Validation set labels.
    test_ratio : float
        Test ratio parameter.
    val_ratio : float
        Validation ratio parameter.
    seed : int
        Random seed.
    num_folds : int
        Total number of folds.
    """
    # Count templates
    train_templates = len(set(l["template_id"] for l in train_labels))
    test_templates = len(set(l["template_id"] for l in test_labels))
    val_templates = len(set(l["template_id"] for l in val_labels)) if val_labels else 0
    
    config = {
        "fold": fold_idx,
        "num_folds": num_folds,
        "seed": seed,
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "train_count": len(train_labels),
        "test_count": len(test_labels),
        "val_count": len(val_labels),
        "train_templates": train_templates,
        "test_templates": test_templates,
        "val_templates": val_templates,
        "train_template_ids": sorted(set(l["template_id"] for l in train_labels)),
        "test_template_ids": sorted(set(l["template_id"] for l in test_labels)),
    }
    
    config_path = fold_dir / "split_config.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)


def write_split_summary(
    folds_dir: Path,
    num_folds: int,
    all_labels: List[Dict[str, Any]],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    """Write summary.json with overall split statistics.
    
    Parameters
    ----------
    folds_dir : Path
        Root folds directory.
    num_folds : int
        Number of folds.
    all_labels : list of dict
        All labels in dataset.
    test_ratio : float
        Test ratio.
    val_ratio : float
        Validation ratio.
    seed : int
        Random seed.
    """
    summary = {
        "dataset_root": str(folds_dir.parent),
        "num_labels": len(all_labels),
        "num_templates": len(set(l["template_id"] for l in all_labels)),
        "num_folds": num_folds,
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "template_ids": sorted(set(l["template_id"] for l in all_labels)),
    }
    
    summary_path = folds_dir / "split_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Build stratified train/test/val splits with cross-validation folds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_root",
        help="Root directory of dataset (contains labels/, signals/, etc.).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds to create.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of data to use for test set.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of train+val to use for validation (applied within each fold).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="folds",
        help="Name of output directory (created under dataset_root).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    
    dataset_root = Path(args.dataset_root)
    labels_dir = dataset_root / "labels"
    folds_dir = dataset_root / args.output_prefix
    
    # Validate input
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Load labels
    print(f"Loading labels from {labels_dir}...")
    labels = load_labels(labels_dir)
    print(f"  Loaded {len(labels)} labels from {len(set(l['template_id'] for l in labels))} templates")
    
    if not labels:
        print("No labels found. Exiting.")
        return
    
    # Create folds
    print(f"\nCreating {args.num_folds} cross-validation folds...")
    print(f"  Test ratio: {args.test_ratio}, Val ratio: {args.val_ratio}, Seed: {args.seed}")
    
    folds = create_cross_validation_folds(
        labels,
        num_folds=args.num_folds,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    
    # Write fold files
    folds_dir.mkdir(parents=True, exist_ok=True)
    for fold_idx, (train_labels, test_labels, val_labels) in enumerate(folds, start=1):
        fold_num_dir = folds_dir / f"fold{fold_idx}"
        
        write_fold_files(fold_num_dir, train_labels, test_labels, val_labels)
        write_split_config(
            fold_num_dir,
            fold_idx,
            train_labels,
            test_labels,
            val_labels,
            args.test_ratio,
            args.val_ratio,
            args.seed,
            args.num_folds,
        )
        
        print(f"  Fold {fold_idx}: train={len(train_labels)}, test={len(test_labels)}, val={len(val_labels)}")
    
    # Write summary
    write_split_summary(folds_dir, args.num_folds, labels, args.test_ratio, args.val_ratio, args.seed)
    
    print(f"\n✓ Splits written to {folds_dir}")
    print(f"  Summary: {folds_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()
