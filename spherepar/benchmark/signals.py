"""
spherepar.benchmark.signals
===========================

Synthetic signal families that can be attached to a triangulated surface.

Implemented signal families
----------------------------
A. Isotropic Gaussian (Euclidean distance in R^3)
B. Anisotropic Gaussian (local tangent-plane coordinates)

Both functions accept the same core parameters and share the Gaussian
evaluation logic; only the coordinate system they operate in differs.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ── Shared helper ─────────────────────────────────────────────────────────────

def _gaussian_from_sq_dist(sq_dist: np.ndarray,
                            amplitude: float,
                            sigma: float) -> np.ndarray:
    """Evaluate exp(-d^2 / (2*sigma^2)) * amplitude element-wise.

    Parameters
    ----------
    sq_dist:
        Squared distances from center, shape (N,).
    amplitude:
        Peak value A.
    sigma:
        Width parameter σ > 0.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return amplitude * np.exp(-sq_dist / (2.0 * sigma ** 2))


# ── Signal family A: Isotropic Gaussian ───────────────────────────────────────

def isotropic_gaussian(
    vertices: np.ndarray,
    center: np.ndarray,
    sigma: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Localized isotropic Gaussian signal on a mesh.

    Computes:
        f_i = A * exp(-||v_i - center||^2 / (2 * sigma^2))

    Uses Euclidean distance in R^3 (fast and sufficient for Stage 1).

    Parameters
    ----------
    vertices:
        Vertex positions, shape (N, 3).
    center:
        3-D center point of the Gaussian, shape (3,).
    sigma:
        Width parameter σ > 0; larger values spread the signal wider.
    amplitude:
        Peak value A (default 1.0).

    Returns
    -------
    np.ndarray, shape (N,)
        Per-vertex signal values.
    """
    vertices = np.asarray(vertices, dtype=float)
    center   = np.asarray(center,   dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")

    diff    = vertices - center[None, :]
    sq_dist = np.sum(diff ** 2, axis=1)
    return _gaussian_from_sq_dist(sq_dist, amplitude, sigma)


# ── Signal family B: Anisotropic Gaussian ─────────────────────────────────────

def _stable_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable orthonormal tangent basis {e1, e2} for a given normal.

    Uses the method of Hughes & Möller to avoid degeneracies.

    Parameters
    ----------
    normal:
        Unit normal vector, shape (3,).

    Returns
    -------
    e1, e2 : two orthonormal tangent vectors each of shape (3,).
    """
    n = normal / np.linalg.norm(normal)
    # Choose a reference vector not parallel to n
    if abs(n[0]) <= abs(n[1]) and abs(n[0]) <= abs(n[2]):
        ref = np.array([1.0, 0.0, 0.0])
    elif abs(n[1]) <= abs(n[2]):
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def anisotropic_gaussian(
    vertices: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    sigma_u: float,
    sigma_v: float,
    amplitude: float = 1.0,
    orientation_angle: float = 0.0,
) -> np.ndarray:
    """Localized anisotropic Gaussian signal defined in the tangent plane.

    Computes:
        f(x) = A * exp(-½ * ξ(x)^T Σ^{-1} ξ(x))

    where ξ(x) = (u, v) are local tangent-plane coordinates relative to
    *center*, Σ = diag(sigma_u^2, sigma_v^2) (possibly rotated by
    *orientation_angle*).

    Parameters
    ----------
    vertices:
        Vertex positions, shape (N, 3).
    center:
        3-D center point (Gaussian peak), shape (3,).
    normal:
        Surface normal at the center (need not be unit length), shape (3,).
    sigma_u:
        Width along the first tangent axis (before rotation).
    sigma_v:
        Width along the second tangent axis (before rotation).
    amplitude:
        Peak value A (default 1.0).
    orientation_angle:
        In-plane rotation of the anisotropy axes, in radians (default 0.0).

    Returns
    -------
    np.ndarray, shape (N,)
        Per-vertex signal values.
    """
    vertices = np.asarray(vertices, dtype=float)
    center   = np.asarray(center,   dtype=float)
    normal   = np.asarray(normal,   dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    if normal.shape != (3,):
        raise ValueError("normal must have shape (3,)")
    if sigma_u <= 0 or sigma_v <= 0:
        raise ValueError(f"sigma_u and sigma_v must be positive; got {sigma_u}, {sigma_v}")

    # Build tangent basis at center
    e1, e2 = _stable_tangent_basis(normal)

    # Project displacements onto tangent plane
    diff = vertices - center[None, :]
    u_raw = diff @ e1
    v_raw = diff @ e2

    # Rotate by orientation_angle clockwise in the (u, v) plane
    cos_a = np.cos(orientation_angle)
    sin_a = np.sin(orientation_angle)
    u = cos_a * u_raw + sin_a * v_raw
    v = -sin_a * u_raw + cos_a * v_raw

    # Evaluate anisotropic Gaussian
    exponent = (u ** 2) / (2.0 * sigma_u ** 2) + (v ** 2) / (2.0 * sigma_v ** 2)
    return amplitude * np.exp(-exponent)