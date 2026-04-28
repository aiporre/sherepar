#!/usr/bin/env python3
"""Driver script for dataset generation with optional spherical parametrization."""

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
    print(f"  Deform method     : {args.deform_method}")
    print(f"  Param method      : {args.param_method}")
    print(f"  Signal type       : {args.signal_type}")
    print("=" * 60)

    signal_type = None if args.signal_type == "none" else args.signal_type
    param_method = None if args.param_method == "none" else args.param_method

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
        signal_amplitude=args.signal_amplitude,
        signal_num_centers=args.signal_num_centers,
        param_method=param_method,
        cem_eps=args.cem_eps,
        cem_max_iters=args.cem_max_iters,
        cem_verbose=args.cem_verbose,
        seed=args.seed,
        repair_holes=not args.no_repair_holes,
        drop_non_watertight=args.drop_non_watertight,
        offset_sample_counter=0,
    )


if __name__ == "__main__":
    main()
