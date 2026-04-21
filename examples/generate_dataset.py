#!/usr/bin/env python3
"""
examples/generate_dataset.py
=============================

Driver script for the mesh-deformation dataset generator.

Usage
-----
    cd /path/to/spherepar
    python examples/generate_dataset.py \\
        --input  data/ \\
        --output data/generated \\
        --n      5 \\
        --seed   42

See --help for the full list of options.

The script reads every .obj from --input, runs the full pipeline
(repair → deform → smooth → validate → patch → stats → save), and writes
results under --output::

    data/generated/
        meshes/   — deformed OBJ meshes
        patches/  — ROI patch arrays (.npy)
        labels/   — metadata JSON files
        logs/     — errors.log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is on the path so both graphop and spherepar are found
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherepar.benchmark.dataset_generator import generate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mesh-deformation dataset generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        default=str(REPO_ROOT / "data"),
        help="Directory containing input .obj meshes.",
    )
    p.add_argument(
        "--output", "-o",
        default=str(REPO_ROOT / "data" / "generated"),
        help="Root output directory for generated dataset.",
    )
    p.add_argument(
        "--n", "-n",
        type=int,
        default=5,
        help="Number of deformation samples to attempt per input mesh.",
    )
    p.add_argument(
        "--patch-radius-ratio",
        type=float,
        default=0.15,
        help="ROI patch radius as a fraction of the bounding-box diagonal.",
    )
    p.add_argument(
        "--smoothing-iter",
        type=int,
        default=3,
        help="Number of Humphrey smoothing passes after deformation.",
    )
    p.add_argument(
        "--method",
        default="sre_arap",
        choices=["sre_arap", "original_arap", "spokes_and_rims"],
        help="Deformation algorithm.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.02,
        help="SRE-ARAP smoothness weight.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=50,
        help="Maximum ARAP iterations.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--no-repair",
        action="store_true",
        help="Skip hole-repair; meshes that are not watertight will be used as-is.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Mesh-deformation dataset generator")
    print("=" * 60)
    print(f"  Input     : {input_dir}")
    print(f"  Output    : {args.output}")
    print(f"  Samples/mesh: {args.n}")
    print(f"  Method    : {args.method}")
    print(f"  Seed      : {args.seed}")
    print("=" * 60)

    generate_dataset(
        input_dir=str(input_dir),
        output_root=args.output,
        n_samples_per_mesh=args.n,
        patch_radius_ratio=args.patch_radius_ratio,
        smoothing_iterations=args.smoothing_iter,
        deform_method=args.method,
        alpha=args.alpha,
        max_iter=args.max_iter,
        seed=args.seed,
        repair_holes=not args.no_repair,
    )


if __name__ == "__main__":
    main()