"""
spherepar.benchmark
===================

Stage 1 synthetic data generation pipeline for pmConv.

Sub-modules
-----------
surface          : SurfaceFactory and Surface classes
signals          : isotropic and anisotropic Gaussian signal generators
dataset_generator: end-to-end dataset generation from raw .obj meshes
"""

from spherepar.benchmark.surface import Surface, SurfaceFactory
from spherepar.benchmark.signals import (
    isotropic_gaussian,
    anisotropic_gaussian,
)
from spherepar.benchmark.dataset_generator import (
    load_meshes_from_directory,
    repair_mesh_if_needed,
    compute_sampling_cube_from_volume,
    sample_handle_centers,
    compute_valid_displacement,
    extract_roi_patch,
    deform_mesh_with_graphop,
    smooth_and_validate_mesh,
    compute_patch_to_mesh_stats,
    save_sample,
    append_error_log,
    generate_dataset,
)

__all__ = [
    "Surface",
    "SurfaceFactory",
    "isotropic_gaussian",
    "anisotropic_gaussian",
    # dataset generator
    "load_meshes_from_directory",
    "repair_mesh_if_needed",
    "compute_sampling_cube_from_volume",
    "sample_handle_centers",
    "compute_valid_displacement",
    "extract_roi_patch",
    "deform_mesh_with_graphop",
    "smooth_and_validate_mesh",
    "compute_patch_to_mesh_stats",
    "save_sample",
    "append_error_log",
    "generate_dataset",
]