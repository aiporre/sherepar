"""
spherepar.benchmark.surface
============================

Stage 1 surface generation pipeline for pmConv.

Classes
-------
Surface
    A single generated example: vertices, faces, signal, and metadata.
    Knows how to save itself under a structured output tree.

SurfaceFactory
    Creates Surface objects by calling the CGAL deformation backend
    (graphop.deform_surface) and attaching synthetic signals.

Both classes are designed so that two usage patterns feel natural:

    # Workflow A: many different deformed surfaces
    factory = SurfaceFactory(root="/data/", template_mesh_path="data/ellipsoid.obj")
    for i in range(100):
        surface = factory.generate_surface(handle_ids=[...], target_positions=[...])
        surface.save()

    # Workflow B: one fixed deformation, many signals
    surface_template = factory.generate_surface(handle_ids=[...], target_positions=[...])
    for i in range(100):
        surface = surface_template.update_signal(signal_type="isotropic", ...)
        surface.update_fname(suffix=f"file_{i}")
        surface.save()
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from spherepar.benchmark.signals import isotropic_gaussian, anisotropic_gaussian

# ---------------------------------------------------------------------------
# Lazy import of the C++ extension so the module can be imported even if the
# extension has not been built yet (useful for documentation / IDE introspection).
# ---------------------------------------------------------------------------
def _load_graphop_extension():
    """Load the compiled graphop extension from the repository root."""
    repo_root = Path(__file__).resolve().parents[2]
    for pattern in ("graphop*.so", "graphop*.pyd"):
        for ext_path in sorted(repo_root.glob(pattern)):
            sys.modules.pop("graphop", None)
            spec = importlib.util.spec_from_file_location("graphop", ext_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "deform_surface"):
                return module

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import graphop as module  # type: ignore[import-not-found]

    if not hasattr(module, "deform_surface"):
        raise ImportError("Imported 'graphop' but did not find the compiled extension exports")
    return module


try:
    _graphop = _load_graphop_extension()
    _GRAPHOP_AVAILABLE = True
except ImportError:
    _graphop = None  # type: ignore[assignment]
    _GRAPHOP_AVAILABLE = False


# ── Surface ───────────────────────────────────────────────────────────────────

class Surface:
    """A single generated surface example.

    Stores geometry (vertices + faces), an optional per-vertex signal array,
    deformation metadata, and signal metadata.  Provides :meth:`save` to
    write everything to disk under a structured directory tree, and mutation
    methods :meth:`update_signal` and :meth:`update_fname` to support
    Workflow B (one geometry, many signals).

    Parameters
    ----------
    vertices:
        Vertex positions, shape (N, 3).
    faces:
        Face connectivity (0-based), shape (M, 3).
    deform_meta:
        Metadata dict from the deformation backend.
    root:
        Root output directory.
    fname:
        Base filename (without extension) for all saved files.
    signal:
        Optional per-vertex signal array, shape (N,).
    signal_meta:
        Optional dict describing the signal (family, parameters, …).
    """

    def __init__(
            self,
            vertices: np.ndarray,
            faces: np.ndarray,
            deform_meta: Dict[str, Any],
            root: str,
            fname: str,
            signal: Optional[np.ndarray] = None,
            signal_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vertices: np.ndarray = np.asarray(vertices, dtype=np.float64)
        self.faces: np.ndarray = np.asarray(faces, dtype=np.int32)
        self.deform_meta: Dict[str, Any] = dict(deform_meta)
        self.root: str = str(root)
        self.fname: str = str(fname)
        self.signal: Optional[np.ndarray] = (
            np.asarray(signal, dtype=np.float64) if signal is not None else None)
        self.signal_meta: Optional[Dict[str, Any]] = (
            dict(signal_meta) if signal_meta is not None else None)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def update_signal(
            self,
            signal: np.ndarray,
            signal_meta: Optional[Dict[str, Any]] = None,
    ) -> "Surface":
        """Return a copy of this surface with a new signal attached.

        Parameters
        ----------
        signal:
            Per-vertex signal array, shape (N,).
        signal_meta:
            Metadata describing the signal (family, parameters, etc.).

        Returns
        -------
        Surface
            A shallow-copy of self with the new signal; geometry is shared
            (not duplicated) since it is immutable after deformation.
        """
        new = copy.copy(self)
        new.signal = np.asarray(signal, dtype=np.float64)
        new.signal_meta = dict(signal_meta) if signal_meta is not None else {}
        return new

    def update_fname(self, suffix: str = "", prefix: str = "") -> "Surface":
        """Return a copy of this surface with a modified file name.

        Parameters
        ----------
        suffix:
            String appended to the current fname (before any extension).
        prefix:
            String prepended to the current fname.

        Returns
        -------
        Surface
            A shallow-copy of self with the updated fname.
        """
        new = copy.copy(self)
        new.fname = f"{prefix}{self.fname}{suffix}"
        return new

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> Dict[str, str]:
        """Save geometry, signal, and metadata to disk.

        Directory layout::

            <root>/
              surfaces/<fname>.obj      — mesh geometry
              signals/<fname>.npy       — signal array (if present)
              labels/<fname>.json       — metadata JSON

        Returns
        -------
        dict
            Paths of the files that were written, keyed by 'surface',
            'signal' (if present), and 'labels'.
        """
        root = Path(self.root)
        surfaces_dir = root / "surfaces"
        labels_dir = root / "labels"
        signals_dir = root / "signals"

        surfaces_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        signals_dir.mkdir(parents=True, exist_ok=True)

        surface_path = surfaces_dir / f"{self.fname}.obj"
        labels_path = labels_dir / f"{self.fname}.json"

        # ── Write OBJ ──────────────────────────────────────────────────────
        self._write_obj(surface_path)

        # ── Write signal ───────────────────────────────────────────────────
        signal_path: Optional[Path] = None
        if self.signal is not None:
            signal_path = signals_dir / f"{self.fname}.npy"
            np.save(signal_path, self.signal)

        # ── Write metadata JSON ────────────────────────────────────────────
        metadata = {
            "fname": self.fname,
            "surface_file": str(surface_path),
            "signal_file": str(signal_path) if signal_path else None,
            "deformation": self._json_safe(self.deform_meta),
            "signal": self._json_safe(self.signal_meta) if self.signal_meta else None,
            "n_vertices": int(self.vertices.shape[0]),
            "n_faces": int(self.faces.shape[0]),
        }
        with open(labels_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        result: Dict[str, str] = {
            "surface": str(surface_path),
            "labels": str(labels_path),
        }
        if signal_path is not None:
            result["signal"] = str(signal_path)
        return result

    def save_only_signal(self) -> Dict[str, str]:
        """Save only the signal array and its metadata.

        Directory layout::

            <root>/
              signals/<fname>.npy            — signal array
              labels/<fname>_signal.json     — signal metadata

        Returns
        -------
        dict
            Paths of the files that were written, keyed by 'signal' and
            'signal_label'.
        """
        if self.signal is None:
            raise ValueError("Cannot save_only_signal() because no signal is attached to this Surface")

        root = Path(self.root)
        labels_dir = root / "labels"
        signals_dir = root / "signals"

        labels_dir.mkdir(parents=True, exist_ok=True)
        signals_dir.mkdir(parents=True, exist_ok=True)

        signal_path = signals_dir / f"{self.fname}.npy"
        signal_label_path = labels_dir / f"{self.fname}_signal.json"

        np.save(signal_path, self.signal)

        signal_metadata = {
            "fname": self.fname,
            "signal_file": str(signal_path),
            "signal": self._json_safe(self.signal_meta) if self.signal_meta else None,
            "n_vertices": int(self.vertices.shape[0]),
        }
        with open(signal_label_path, "w") as fh:
            json.dump(signal_metadata, fh, indent=2)

        return {
            "signal": str(signal_path),
            "signal_label": str(signal_label_path),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _write_obj(self, path: Path) -> None:
        """Write geometry to a minimal OBJ file."""
        with open(path, "w") as fh:
            fh.write(f"# Generated by spherepar.benchmark\n")
            fh.write(f"# Vertices: {self.vertices.shape[0]}\n")
            fh.write(f"# Faces:    {self.faces.shape[0]}\n\n")
            for v in self.vertices:
                fh.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
            fh.write("\n")
            for f in self.faces:
                # OBJ is 1-based
                fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

    @staticmethod
    def _json_safe(obj: Any) -> Any:
        """Recursively convert numpy types to plain Python for JSON serialisation."""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: Surface._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [Surface._json_safe(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    def __repr__(self) -> str:
        sig_shape = self.signal.shape if self.signal is not None else None
        return (
            f"Surface(fname={self.fname!r}, "
            f"vertices={self.vertices.shape}, faces={self.faces.shape}, "
            f"signal={sig_shape})"
        )


# ── SurfaceFactory ────────────────────────────────────────────────────────────

class SurfaceFactory:
    """Factory for generating deformed surfaces with attached synthetic signals.

    Parameters
    ----------
    root:
        Root output directory.  Sub-directories ``surfaces/``, ``signals/``,
        and ``labels/`` will be created inside it as needed.
    template_mesh_path:
        Path to the template OBJ mesh (genus-0 closed surface).

    Raises
    ------
    ImportError
        If the ``graphop`` C++ extension is not available.
    FileNotFoundError
        If *template_mesh_path* does not exist.
    """

    def __init__(self, root: str, template_mesh_path: str) -> None:
        if not _GRAPHOP_AVAILABLE:
            raise ImportError(
                "The 'graphop' C++ extension is required but could not be imported. "
                "Build it with CMake (see BUILD.md) and ensure it is on PYTHONPATH."
            )
        if not os.path.isfile(template_mesh_path):
            raise FileNotFoundError(
                f"Template mesh not found: {template_mesh_path!r}"
            )
        self.root: str = str(root)
        self.template_mesh_path: str = str(template_mesh_path)
        self._counter: int = 0  # auto-incremented when no fname given

    # ── Surface generation ────────────────────────────────────────────────────

    def generate_surface(
            self,
            handle_ids: List[int],
            target_positions: Union[np.ndarray, List],
            ring_size: float = 0.0,
            roi_ids: Optional[List[int]] = None,
            method: str = "sre_arap",
            alpha: float = 0.02,
            max_iter: int = 50,
            signal_type: Optional[str] = None,
            signal_params: Optional[Dict[str, Any]] = None,
            fname: Optional[str] = None,
    ) -> Surface:
        """Generate a deformed surface and optionally attach a signal.

        Parameters
        ----------
        handle_ids:
            0-based vertex indices used as positional handles.
        target_positions:
            Target 3-D positions for each handle, shape (H, 3) or flat (3H,).
        ring_size:
            Euclidean radius around each handle. Every vertex within this
            radius is translated by the same displacement as the handle.
            ``0.0`` keeps the classic single-vertex handle behaviour.
        roi_ids:
            Optional region-of-interest vertex indices.  ``None`` = whole mesh.
        method:
            Deformation algorithm: ``'sre_arap'`` (default), ``'original_arap'``,
            or ``'spokes_and_rims'``.
        alpha:
            SRE-ARAP smoothness weight (default 0.02).
        max_iter:
            Maximum ARAP iterations (default 50).
        signal_type:
            Optional signal to attach immediately.  One of:
            ``'isotropic'``, ``'anisotropic'``.
        signal_params:
            Parameters forwarded to the signal generator.
            See :func:`~spherepar.benchmark.signals.isotropic_gaussian` and
            :func:`~spherepar.benchmark.signals.anisotropic_gaussian` for
            accepted keys.
        fname:
            Base file name (without extension) for saved files.
            Auto-generated if ``None``.

        Returns
        -------
        Surface
        """
        # ── Deformation ───────────────────────────────────────────────────
        target_positions = np.asarray(target_positions, dtype=np.float64)
        if target_positions.ndim == 2:
            target_positions = target_positions.ravel()

        roi = roi_ids if roi_ids is not None else []

        V_new, F, deform_meta = _graphop.deform_surface(
            mesh_path=self.template_mesh_path,
            handle_ids=list(handle_ids),
            target_positions=target_positions,
            ring_size=ring_size,
            roi_ids=list(roi),
            method=method,
            alpha=alpha,
            max_iter=max_iter,
        )

        # ── File name ─────────────────────────────────────────────────────
        if fname is None:
            fname = f"surface_{self._counter:05d}"
            self._counter += 1

        # ── Optional signal ───────────────────────────────────────────────
        signal: Optional[np.ndarray] = None
        signal_meta: Optional[Dict[str, Any]] = None

        if signal_type is not None:
            signal, signal_meta = self._compute_signal(
                signal_type, V_new, signal_params or {}
            )

        return Surface(
            vertices=V_new,
            faces=F,
            deform_meta=deform_meta,
            root=self.root,
            fname=fname,
            signal=signal,
            signal_meta=signal_meta,
        )

    def generate_surface_with_angles(
            self,
            handle_transforms: List[Dict[str, Any]],
            roi_ids: Optional[List[int]] = None,
            method: str = "sre_arap",
            alpha: float = 0.02,
            max_iter: int = 50,
            signal_type: Optional[str] = None,
            signal_params: Optional[Dict[str, Any]] = None,
            fname: Optional[str] = None,
    ) -> "Surface":
        """Generate a deformed surface using per-handle rotation specifications.

        Each handle is a dict with three required keys:

        * ``vertex_id`` (int)   — 0-based index of the center vertex.
        * ``angle``     (float) — rotation in radians around the surface normal
          at the center vertex.
        * ``ring_size`` (float) — Euclidean radius; all vertices within this
          distance of the center vertex become positional handles whose target
          positions are computed by rotating them by *angle* around the surface
          normal at *vertex_id*.

        Parameters
        ----------
        handle_transforms:
            List of per-handle dicts, each with keys ``vertex_id``, ``angle``,
            and ``ring_size``.
        roi_ids:
            Optional region-of-interest vertex indices.  ``None`` = whole mesh.
        method:
            Deformation algorithm: ``'sre_arap'`` (default), ``'original_arap'``,
            or ``'spokes_and_rims'``.
        alpha:
            SRE-ARAP smoothness weight (default 0.02).
        max_iter:
            Maximum ARAP iterations (default 50).
        signal_type:
            Optional signal to attach immediately.  ``'isotropic'`` or
            ``'anisotropic'``.
        signal_params:
            Parameters forwarded to the signal generator.
        fname:
            Base filename (without extension).  Auto-generated if ``None``.

        Returns
        -------
        Surface
        """
        # Validate required keys up-front for a clear error message
        for i, t in enumerate(handle_transforms):
            for key in ("vertex_id", "angle", "ring_size"):
                if key not in t:
                    raise ValueError(
                        f"handle_transforms[{i}] is missing required key '{key}'"
                    )

        roi = roi_ids if roi_ids is not None else []

        V_new, F, deform_meta = _graphop.deform_surface_with_angles(
            mesh_path=self.template_mesh_path,
            handle_transforms=list(handle_transforms),
            roi_ids=list(roi),
            method=method,
            alpha=alpha,
            max_iter=max_iter,
        )

        if fname is None:
            fname = f"surface_{self._counter:05d}"
            self._counter += 1

        signal: Optional[np.ndarray] = None
        signal_meta: Optional[Dict[str, Any]] = None
        if signal_type is not None:
            signal, signal_meta = self._compute_signal(
                signal_type, V_new, signal_params or {}
            )

        return Surface(
            vertices=V_new,
            faces=F,
            deform_meta=deform_meta,
            root=self.root,
            fname=fname,
            signal=signal,
            signal_meta=signal_meta,
        )

    # ── Signal helpers ────────────────────────────────────────────────────────

    def compute_signal(
            self,
            surface: Surface,
            signal_type: str,
            signal_params: Optional[Dict[str, Any]] = None,
    ) -> Surface:
        """Attach or replace the signal on an existing Surface.

        Parameters
        ----------
        surface:
            The surface to attach the signal to.
        signal_type:
            ``'isotropic'`` or ``'anisotropic'``.
        signal_params:
            Signal parameters (see signal generator docs).

        Returns
        -------
        Surface
            A new Surface with the signal attached.
        """
        sig, meta = self._compute_signal(signal_type, surface.vertices,
                                         signal_params or {})
        return surface.update_signal(sig, meta)

    def _compute_signal(
            self,
            signal_type: str,
            vertices: np.ndarray,
            params: Dict[str, Any],
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Internal signal dispatcher."""
        if signal_type == "isotropic":
            return self._isotropic_signal(vertices, params)
        if signal_type == "anisotropic":
            return self._anisotropic_signal(vertices, params)
        raise ValueError(
            f"Unknown signal_type {signal_type!r}. "
            "Choose 'isotropic' or 'anisotropic'."
        )

    def _isotropic_signal(
            self,
            vertices: np.ndarray,
            params: Dict[str, Any],
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Evaluate an isotropic Gaussian signal.

        Expected keys in *params*: center or centers (shape (3,), vertex index
        int, or a list of either), sigma/sigmas, amplitude/amplitudes (optional, default 1.0).
        Multiple centers are combined additively.
        Per-center sigma and amplitude can be provided as lists.
        """
        centers = _resolve_centers(vertices, params.get("centers", params.get("center")))
        num_centers = len(centers)
        
        # Support both single values and per-center lists
        sigma_list = params.get("sigmas", params.get("sigma"))
        amplitude_list = params.get("amplitudes", params.get("amplitude"))
        
        # Convert to lists if needed
        if not isinstance(sigma_list, (list, tuple)):
            sigma_list = [float(sigma_list)] * num_centers
        else:
            sigma_list = [float(s) for s in sigma_list]
        
        if not isinstance(amplitude_list, (list, tuple)):
            amplitude_list = [float(amplitude_list)] * num_centers
        else:
            amplitude_list = [float(a) for a in amplitude_list]
        
        # Ensure we have the right number of values
        if len(sigma_list) != num_centers:
            sigma_list = [sigma_list[0]] * num_centers if sigma_list else [0.1] * num_centers
        if len(amplitude_list) != num_centers:
            amplitude_list = [amplitude_list[0]] * num_centers if amplitude_list else [1.0] * num_centers

        sig = np.zeros(vertices.shape[0], dtype=float)
        for i, center in enumerate(centers):
            sig += isotropic_gaussian(vertices, center, sigma_list[i], amplitude_list[i])

        meta = {
            "family": "isotropic_gaussian",
            "center": centers[0].tolist(),
            "centers": [center.tolist() for center in centers],
            "num_centers": len(centers),
            "sigma": sigma_list[0],  # Keep first for backward compat
            "sigmas": sigma_list,
            "amplitude": amplitude_list[0],  # Keep first for backward compat
            "amplitudes": amplitude_list,
        }
        return sig, meta

    def _anisotropic_signal(
            self,
            vertices: np.ndarray,
            params: Dict[str, Any],
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Evaluate an anisotropic Gaussian signal.

        Expected keys in *params*: center, normal (optional, estimated from
        mesh if absent), sigma_u, sigma_v, amplitude, orientation_angle.
        """
        center = _resolve_center(vertices, params.get("center"))
        normal = params.get("normal")
        if normal is None:
            normal = _estimate_normal(vertices, center)
        normal = np.asarray(normal, dtype=float)

        sigma_u = float(params.get("sigma_u", 0.1))
        sigma_v = float(params.get("sigma_v", 0.05))
        amplitude = float(params.get("amplitude", 1.0))
        orientation_angle = float(params.get("orientation_angle", 0.0))

        sig = anisotropic_gaussian(
            vertices, center, normal, sigma_u, sigma_v,
            amplitude, orientation_angle
        )

        meta = {
            "family": "anisotropic_gaussian",
            "center": center.tolist(),
            "normal": normal.tolist(),
            "sigma_u": sigma_u,
            "sigma_v": sigma_v,
            "amplitude": amplitude,
            "orientation_angle": orientation_angle,
        }
        return sig, meta

    def __repr__(self) -> str:
        return (
            f"SurfaceFactory(root={self.root!r}, "
            f"template={self.template_mesh_path!r})"
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _resolve_center(vertices: np.ndarray,
                    center: Any) -> np.ndarray:
    """Return a (3,) center point from either a point or a vertex index."""
    if center is None:
        # Default: centroid of the mesh
        return vertices.mean(axis=0)
    c = np.asarray(center, dtype=float)
    if c.ndim == 0:
        # Integer-like: index into vertices
        idx = int(c)
        return vertices[idx].copy()
    if c.shape == (3,):
        return c
    raise ValueError(
        f"center must be a 3-D point or a scalar vertex index, got shape {c.shape}"
    )


def _resolve_centers(vertices: np.ndarray,
                     centers: Any) -> List[np.ndarray]:
    """Return one or more (3,) center points from points and/or vertex indices."""
    if centers is None:
        return [_resolve_center(vertices, None)]
    if isinstance(centers, (list, tuple)):
        if len(centers) == 0:
            raise ValueError("centers must not be empty")
        first = np.asarray(centers[0], dtype=float)
        if first.ndim == 0 or first.shape == (3,):
            return [_resolve_center(vertices, center) for center in centers]
    arr = np.asarray(centers, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return [center.copy() for center in arr]
    return [_resolve_center(vertices, centers)]


def _estimate_normal(vertices: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Cheap normal estimate: vector from mesh centroid to center point."""
    centroid = vertices.mean(axis=0)
    n = center - centroid
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return n / norm
