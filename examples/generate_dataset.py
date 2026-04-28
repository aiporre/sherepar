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
(repair → deform → smooth → validate → signal → stats → save), and writes
results under --output::

    data/generated/
        meshes/   — deformed OBJ meshes
        signals/  — per-vertex signal arrays (.npy)
        labels/   — metadata JSON files
        logs/     — errors.log
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Ensure the repo root is on the path so both graphop and spherepar are found
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherepar.benchmark.dataset_generator import generate_dataset  # noqa: E402
from spherepar.benchmark.dataset_generator import build_arg_parser


def main(argv: Optional[List[str]]=None) -> None:
    args = build_arg_parser().parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Mesh-deformation dataset generator")
    print("=" * 60)
    print(f"  Input             : {input_dir}")
    print(f"  Output            : {args.output_root}")
    print(f"  Samples/mesh      : {args.n_samples_per_mesh}")
    print(f"  Patch radius ratio: {args.patch_radius_ratio}")
    print(f"  Smoothing iters    : {args.smoothing_iterations}")
    print(f"  Group candidates   : {args.group_candidates}")
    print(f"  ROI vertex ratio   : {args.roi_vertex_ratio}")
    print(f"  Max ratio          : {args.max_ratio}")
    print(f"  Ring size          : {args.ring_size}")
    print(f"  Method             : {args.deform_method}")
    print(f"  Signal type        : {args.signal_type}")
    print(f"  Signal sigma       : {args.signal_sigma}")
    print(f"  Signal amplitude   : {args.signal_amplitude}")
    print(f"  Signal centers     : {args.signal_num_centers}")
    print(f"  Alpha              : {args.alpha}")
    print(f"  Max iter           : {args.max_iter}")
    print(f"  Seed               : {args.seed}")
    print(f"  Repair holes       : {not args.no_repair_holes}")
    print(f"  Drop non-watertight: {args.drop_non_watertight}")
    print("=" * 60)

    generate_dataset(
        input_dir=args.input_dir,
        output_root=args.output_root,
        n_samples_per_mesh=args.n_samples_per_mesh,
        patch_radius_ratio=args.patch_radius_ratio,
        smoothing_iterations=args.smoothing_iterations,
        group_candidates=args.group_candidates,
        roi_vertex_ratio=args.roi_vertex_ratio,
        max_ratio=args.max_ratio,
        ring_size=args.ring_size,
        deform_method=args.deform_method,
        signal_type=args.signal_type,
        alpha=args.alpha,
        max_iter=args.max_iter,
        signal_sigma=args.signal_sigma,
        signal_amplitude=args.signal_amplitude,
        signal_num_centers=args.signal_num_centers,
        seed=args.seed,
        repair_holes=not args.no_repair_holes,
        drop_non_watertight=args.drop_non_watertight,
    )

if __name__ == "__main__":
    main()
