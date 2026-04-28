"""Compatibility wrapper for spherical parametrization utilities."""
from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
import trimesh

from spherepar.cem_parametrization import stretch_parametrization
from spherepar.flash_parametrization import (  # noqa: F401
    flash_map,
    load_mesh_with_trimesh,
)
from spherepar.mesh import MeshFactory


def compute_spherical_parametrization(
    vertices: np.ndarray,
    faces: np.ndarray,
    method: str = "flash",
    cem_eps: float = 1e-6,
    cem_max_iters: int = 100,
    cem_verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)

    if method == "flash":
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        sphere_vertices = flash_map(mesh)
        meta: Dict[str, Any] = {"method": "flash"}
    elif method == "cem":
        mesh_surf = MeshFactory.make_mesh("surf", vertices, faces)
        stretch = stretch_parametrization(mesh_surf, eps=cem_eps, max_iters=cem_max_iters, verbose=cem_verbose)
        sphere_vertices = stretch.convert_mesh().get_vertices_collection()
        meta = {
            "method": "cem",
            "eps": float(cem_eps),
            "max_iters": int(cem_max_iters),
            "verbose": bool(cem_verbose),
        }
    else:
        raise ValueError("method must be one of: 'flash', 'cem'")

    norms = np.linalg.norm(sphere_vertices, axis=1)
    meta.update(
        {
            "n_vertices": int(sphere_vertices.shape[0]),
            "n_faces": int(faces.shape[0]),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
        }
    )
    return sphere_vertices, meta
