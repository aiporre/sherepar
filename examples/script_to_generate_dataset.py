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
        spheres/  — spheres result from the sphereical parametrization OBJ meshes with sphere coordinates.


This script is used to generate the dataset cases:

- Deforms and generate signals for each template on input directory
    - Per-vertex signal arrays (.npy)
    - Obj mesh file
    - JSON file labels
- For each generated mesh creates a spherical parametrization with FLASH or CEM
- CASE 2 is small deformation controlled by small range of parameters:
    - Max Ratio: used for displacement % from the point distance to the Center of Mass (CoM)
    - Number of candidates: if larger more deformation points
    - Group candidates: make the deformations more global, if use one one candidate, then the ROI is smaller. Grouping
      will make the deformation more global, and thus the ROI larger.
    - alpha: controls the deformation strength, if larger more deformation
    - smooth-iterations: after each deformation the meshes are smooth out, this controls how many times the mesh
      is smoothed after each deformation, if larger more smoothing and thus less deformation.
- CASE 3 is large deformation controlled by a larger range of parameters, and thus more challenging for the
  parametrization and convolution to learn.

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherepar.benchmark.dataset_generator import build_arg_parser, generate_dataset  # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
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
    print(f"  Signal sigma ani   : {args.signal_sigma_ani}")
    print(f"  Signal sigma var % : {args.signal_sigma_variation}")
    print(f"  Signal amp var %   : {args.signal_amplitude_variation}")
    print(f"  Signal amplitude   : {args.signal_amplitude}")
    print(f"  Signal centers     : {args.signal_num_centers}")
    print(f"  Alpha              : {args.alpha}")
    print(f"  Max iter           : {args.max_iter}")
    print(f"  Seed               : {args.seed}")
    print(f"  Repair holes       : {not args.no_repair_holes}")
    print(f"  Drop non-watertight: {args.drop_non_watertight}")
    print(f"  Param method       : {args.param_method}")
    print(f"  Signal centers opt : {args.signal_centers_options}")
    print(f"  CEM eps            : {args.cem_eps}")
    print(f"  CEM max iters      : {args.cem_max_iters}")
    print(f"  CEM verbose        : {args.cem_verbose}")
    print(f"  Deformation cases  : {args.deformation_cases}")
    print(f"  Create splits      : {args.create_splits}")
    print(f"  Split tasks        : {args.split_tasks}")
    print(f"  Num folds          : {args.num_folds}")
    print(f"  Train ratio        : {args.train_ratio}")
    print(f"  Val ratio          : {args.val_ratio}")
    print(f"  Test ratio         : {args.test_ratio}")
    print(f"  Split seed         : {args.split_seed}")
    print(f"  Group by template  : {args.group_by_template}")
    print(f"  Offset sample ctr  : {0}")
    print("=" * 60)

    signal_type = args.signal_type
    param_method = None if args.param_method == "none" else args.param_method
    signal_centers_options = None
    if args.signal_centers_options:
        signal_centers_options = [int(x.strip()) for x in args.signal_centers_options.split(",") if x.strip()]
    deformation_cases = [x.strip() for x in args.deformation_cases.split(",") if x.strip()]
    split_tasks = [x.strip() for x in args.split_tasks.split(",") if x.strip()]
    # generate 200 samples for each num signal secnter from 1 to signal_num_centers
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
        signal_type=signal_type,
        alpha=args.alpha,
        max_iter=args.max_iter,
        signal_sigma=args.signal_sigma,
        signal_sigma_ani=args.signal_sigma_ani,
        signal_sigma_variation_percent=args.signal_sigma_variation,
        signal_amplitude_variation_percent=args.signal_amplitude_variation,
        signal_amplitude=args.signal_amplitude,
        signal_num_centers=args.signal_num_centers,
        signal_centers_options=signal_centers_options,
        param_method=param_method,
        cem_eps=args.cem_eps,
        cem_max_iters=args.cem_max_iters,
        cem_verbose=args.cem_verbose,
        deformation_cases=deformation_cases,
        create_splits=args.create_splits,
        split_tasks=split_tasks,
        num_folds=args.num_folds,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        group_by_template=args.group_by_template,
        seed=args.seed,
        repair_holes=not args.no_repair_holes,
        drop_non_watertight=args.drop_non_watertight,
        offset_sample_counter=0,
        mnist_percentage=args.mnist_percentage if hasattr(args, 'mnist_percentage') else 100.0,
        mnist_total_count=args.mnist_total_count if hasattr(args, 'mnist_total_count') else None,
    )


if __name__ == "__main__":
    main()
