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
    genus_zero_filter_reason,
    repair_mesh_if_needed,
    compute_sampling_cube_from_volume,
    sample_handle_centers,
    compute_valid_displacement,
    deform_mesh_with_graphop,
    smooth_and_validate_mesh,
    compute_patch_to_mesh_stats,
    save_sample_signal,
    save_spherical_parametrization,
    validate_saved_sample,
    append_error_log,
    generate_dataset,
)
from spherepar.benchmark.utils import extract_roi_patch
from spherepar.benchmark.splits import (
    build_task_splits,
    TASK_MODELNET40_CLS,
    is_valid_for_number_of_centers,
    is_valid_for_center_regression,
    is_valid_for_sigma_regression,
    is_valid_for_amplitude_regression,
    is_valid_for_mnist_cls,
)

__all__ = [
    "Surface",
    "SurfaceFactory",
    "isotropic_gaussian",
    "anisotropic_gaussian",
    # dataset generator
    "load_meshes_from_directory",
    "genus_zero_filter_reason",
    "repair_mesh_if_needed",
    "compute_sampling_cube_from_volume",
    "sample_handle_centers",
    "compute_valid_displacement",
    "extract_roi_patch",
    "deform_mesh_with_graphop",
    "smooth_and_validate_mesh",
    "compute_patch_to_mesh_stats",
    "save_sample_signal",
    "save_spherical_parametrization",
    "validate_saved_sample",
    "append_error_log",
    "generate_dataset",
    "build_task_splits",
    "TASK_MODELNET40_CLS",
    "is_valid_for_number_of_centers",
    "is_valid_for_center_regression",
    "is_valid_for_sigma_regression",
    "is_valid_for_amplitude_regression",
    "is_valid_for_mnist_cls",
]
