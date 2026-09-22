"""SphereMap/MoebiusRegistration spherical-parametrization backend.

This module delegates the numerical work to the compiled SphereMap binary via
the Python wrapper shipped in the sibling ``MoebiusRegistration`` repository.
It intentionally does not reimplement CMCF or Möbius centering.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import trimesh


DEFAULT_SPHEREMAP_REPOSITORY = Path("/home/sauron/Documents/Phd/code/MoebiusRegistration")


class SphereMapBackendError(RuntimeError):
    """Raised when the external SphereMap backend cannot produce a map."""


def _load_wrapper(repository: str | Path | None = None):
    """Load the sibling repository's Python wrapper without hard dependency."""
    if repository is not None:
        root = Path(repository).expanduser().resolve()
        if not root.exists():
            raise SphereMapBackendError(
                "MoebiusRegistration Python wrapper is unavailable; install "
                f"{root} or pass a valid spheremap_repository"
            )
        sys.path.insert(0, str(root))
        try:
            from spheremap import SphereMap, SphereMapError
            from spheremap.core import _read_result_ply
            return SphereMap, SphereMapError, _read_result_ply
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise SphereMapBackendError(
                f"could not import spheremap from {root}"
            ) from exc
    try:
        from spheremap import SphereMap, SphereMapError
        from spheremap.core import _read_result_ply
        return SphereMap, SphereMapError, _read_result_ply
    except ImportError:
        root = Path(repository or DEFAULT_SPHEREMAP_REPOSITORY).expanduser().resolve()
        if not root.exists():
            raise SphereMapBackendError(
                "MoebiusRegistration Python wrapper is unavailable; install "
                f"{root} or pass spheremap_repository=..."
            )
        sys.path.insert(0, str(root))
        try:
            from spheremap import SphereMap, SphereMapError
            from spheremap.core import _read_result_ply
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise SphereMapBackendError(
                f"could not import spheremap from {root}"
            ) from exc
        return SphereMap, SphereMapError, _read_result_ply


def _json_command_result(result: Any) -> dict[str, Any]:
    return {
        "returncode": int(result.returncode),
        "output": str(result.output),
        "ply_output": str(result.ply_output) if result.ply_output is not None else None,
        "stdout": str(result.stdout)[-4000:],
        "stderr": str(result.stderr)[-4000:],
        "duration_sec": float(result.duration_sec),
        "command": list(result.command),
    }


def spheremap_map_with_diagnostics(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    binary: str | Path | None = None,
    repository: str | Path | None = None,
    auto_build: bool = False,
    iters: int = 25,
    step_size: float = 1.0,
    threads: int = 4,
    resolution: int | None = None,
    degree: int | None = 4,
    a_steps: int | None = 10,
    a_step_size: float | None = 0.05,
    mesh: int = 1,
    fill: int = 0,
    cut_off: float | None = None,
    smooth: float | None = None,
    poincare_max_norm: float | None = 2.0,
    c2i: int | None = 0,
    gss_tolerance: float | None = 1e-6,
    lump: bool = False,
    no_center: bool = False,
    verbose: bool = False,
    timeout: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run SphereMap and return spherical coordinates plus diagnostics.

    The returned vertex order and connectivity are required to match the
    input exactly, which preserves sherepar's vertex-aligned signal contract.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite array with shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if len(faces) == 0 or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces must be non-empty and reference valid vertices")
    if int(iters) < 1 or int(threads) < 1:
        raise ValueError("iters and threads must be positive")
    if not np.isfinite(step_size) or float(step_size) <= 0.0:
        raise ValueError("step_size must be finite and positive")

    SphereMap, SphereMapError, read_result_ply = _load_wrapper(repository)
    parameters: dict[str, Any] = {
        "iters": int(iters),
        "step_size": float(step_size),
        "threads": int(threads),
        "mesh": int(mesh),
        "fill": int(fill),
        "lump": bool(lump),
        "no_center": bool(no_center),
        "verbose": bool(verbose),
    }
    optional = {
        "resolution": resolution,
        "degree": degree,
        "a_steps": a_steps,
        "a_step_size": a_step_size,
        "cut_off": cut_off,
        "smooth": smooth,
        "poincare_max_norm": poincare_max_norm,
        "c2i": c2i,
        "gss_tolerance": gss_tolerance,
    }
    parameters.update({key: value for key, value in optional.items() if value is not None})

    repository_root = Path(repository or DEFAULT_SPHEREMAP_REPOSITORY).expanduser().resolve()
    resolved_binary = (
        Path(binary).expanduser().resolve()
        if binary is not None
        else repository_root / "Bin" / "Linux" / "SphereMap"
    )
    if not resolved_binary.exists():
        resolved_binary = None
    diagnostics: dict[str, Any] = {
        "backend": "MoebiusRegistration.SphereMap",
        "repository": str(repository_root),
        "binary": str(resolved_binary) if resolved_binary is not None else None,
        "parameters": parameters.copy(),
        "success": False,
    }
    with tempfile.TemporaryDirectory(prefix="sherepar-spheremap-") as directory:
        work = Path(directory)
        input_ply = work / "input.ply"
        output_ply = work / "spheremap.ply"
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(input_ply)
        try:
            job = SphereMap(
                binary=resolved_binary,
                auto_build=bool(auto_build),
                **parameters,
            ).run(input_ply, output_ply)
            result = job.wait(timeout=timeout)
        except Exception as exc:
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, SphereMapError):
                diagnostics["backend_error"] = str(exc)
            raise SphereMapBackendError(diagnostics["error"]) from exc

        diagnostics["process"] = _json_command_result(result)
        try:
            mapped_vertices_raw, mapped_faces_raw = read_result_ply(output_ply)
            mapped_vertices = np.asarray(mapped_vertices_raw, dtype=np.float64)
            mapped_faces = np.asarray(mapped_faces_raw, dtype=np.int32)
        except Exception as exc:
            diagnostics["error"] = f"SphereMap output extraction failed: {exc}"
            raise SphereMapBackendError(diagnostics["error"]) from exc

    if mapped_vertices.shape != vertices.shape:
        raise SphereMapBackendError(
            f"SphereMap changed vertex shape: {mapped_vertices.shape} != {vertices.shape}"
        )
    if not np.array_equal(mapped_faces, faces):
        raise SphereMapBackendError("SphereMap changed face connectivity or ordering")
    if not np.isfinite(mapped_vertices).all():
        raise SphereMapBackendError("SphereMap returned non-finite spherical coordinates")
    norms = np.linalg.norm(mapped_vertices, axis=1)
    if np.any(norms <= 0.0):
        raise SphereMapBackendError("SphereMap returned zero-length spherical coordinates")
    mapped_vertices = mapped_vertices / norms[:, None]
    diagnostics.update(
        {
            "success": True,
            "vertex_count": int(len(mapped_vertices)),
            "face_count": int(len(mapped_faces)),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
        }
    )
    return mapped_vertices, diagnostics


def spheremap_map(vertices: np.ndarray, faces: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Run SphereMap and return only the unit spherical coordinates."""
    mapped, _ = spheremap_map_with_diagnostics(vertices, faces, **kwargs)
    return mapped
