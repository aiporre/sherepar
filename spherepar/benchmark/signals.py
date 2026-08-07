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
from typing import Any, Optional


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


def _normalize_vector(vec: np.ndarray, *, name: str, eps: float = 1e-8) -> np.ndarray:
    """Return a normalized 3D vector and fail fast on invalid input."""
    arr = np.asarray(vec, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values: {arr}")
    norm = float(np.linalg.norm(arr))
    if norm <= eps:
        raise ValueError(f"{name} has near-zero norm ({norm}); cannot normalize")
    return arr / norm


def _wrap_angle_pi(angle: float) -> float:
    """Wrap angle to [0, pi)."""
    wrapped = float(np.mod(angle, np.pi))
    if wrapped < 0.0:
        wrapped += float(np.pi)
    return wrapped


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
    n = _normalize_vector(normal, name="normal")
    # Choose a reference vector not parallel to n
    if abs(n[0]) <= abs(n[1]) and abs(n[0]) <= abs(n[2]):
        ref = np.array([1.0, 0.0, 0.0])
    elif abs(n[1]) <= abs(n[2]):
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(n, ref)
    e1 = _normalize_vector(e1, name="e1")
    e2 = np.cross(n, e1)
    e2 = _normalize_vector(e2, name="e2")
    if np.dot(np.cross(e1, e2), n) < 0.0:
        e2 = -e2
    return e1, e2


def fixed_gauge_tangent_basis(
    center: np.ndarray,
    *,
    gauge: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
    eps: float = 1e-8,
    min_projected_norm: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a fixed global gauge to center tangent plane and build {g1, g2}."""
    if min_projected_norm is None:
        min_projected_norm = eps
    if min_projected_norm < eps:
        raise ValueError("min_projected_norm must be >= eps")
    c = _normalize_vector(center, name="center", eps=eps)
    g = _normalize_vector(gauge, name="gauge", eps=eps)
    g1_tilde = g - float(np.dot(g, c)) * c
    g1_tilde_norm = float(np.linalg.norm(g1_tilde))
    if g1_tilde_norm <= float(min_projected_norm):
        raise ValueError(
            "Projected gauge has near-zero norm at center; center is too close to gauge pole"
        )
    g1 = g1_tilde / g1_tilde_norm
    g2 = _normalize_vector(np.cross(c, g1), name="g2", eps=eps)
    return g1, g2


def compute_gauge_relative_angle_and_target(
    center: np.ndarray,
    major_axis: np.ndarray,
    *,
    gauge: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
    eps: float = 1e-8,
    min_gauge_projection_norm: Optional[float] = None,
) -> dict[str, Any]:
    """Compute φ in [0,π) and doubled-angle target t from center and physical axis."""
    c = _normalize_vector(center, name="center", eps=eps)
    v = _normalize_vector(major_axis, name="major_axis", eps=eps)
    tangent_error = abs(float(np.dot(v, c)))
    if tangent_error > 5e-6:
        raise ValueError(
            f"major_axis must be tangent to center. |v·c|={tangent_error} exceeds tolerance"
        )
    g1, g2 = fixed_gauge_tangent_basis(
        c,
        gauge=gauge,
        eps=eps,
        min_projected_norm=min_gauge_projection_norm,
    )
    phi = _wrap_angle_pi(np.arctan2(float(np.dot(v, g2)), float(np.dot(v, g1))))
    target = np.array([np.cos(2.0 * phi), np.sin(2.0 * phi)], dtype=float)
    target_norm = float(np.linalg.norm(target))
    if not np.isfinite(target_norm) or abs(target_norm - 1.0) > 1e-6:
        raise ValueError("Invalid doubled-angle target norm")
    return {"phi": phi, "target": target, "g1": g1, "g2": g2}


def sample_hm_major_axis(
    center: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    delta: Optional[float] = None,
    gauge: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
    eps: float = 1e-8,
    min_gauge_projection_norm: Optional[float] = None,
) -> dict[str, Any]:
    """Sample physical major axis with Hughes–Möller frame and fixed-gauge target."""
    if rng is None:
        rng = np.random.default_rng()
    c = _normalize_vector(center, name="center", eps=eps)
    e1, e2 = _stable_tangent_basis(c)
    if delta is None:
        delta = float(rng.uniform(0.0, np.pi))
    else:
        delta = _wrap_angle_pi(float(delta))
    v = np.cos(delta) * e1 + np.sin(delta) * e2
    v = _normalize_vector(v, name="major_axis", eps=eps)
    if abs(float(np.dot(v, c))) > 5e-6:
        raise ValueError("Sampled major axis is not tangent to center")
    gauge_info = compute_gauge_relative_angle_and_target(
        c,
        v,
        gauge=gauge,
        eps=eps,
        min_gauge_projection_norm=min_gauge_projection_norm,
    )
    return {
        "center": c,
        "e1": e1,
        "e2": e2,
        "delta": delta,
        "major_axis": v,
        "phi": float(gauge_info["phi"]),
        "target": np.asarray(gauge_info["target"], dtype=float),
        "g1": np.asarray(gauge_info["g1"], dtype=float),
        "g2": np.asarray(gauge_info["g2"], dtype=float),
    }


def rotate_axis_and_target(
    rotation: np.ndarray,
    center: np.ndarray,
    major_axis: np.ndarray,
    *,
    gauge: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),
    eps: float = 1e-8,
    min_gauge_projection_norm: Optional[float] = None,
) -> dict[str, Any]:
    """Rotate center/axis with R and recompute gauge-relative angle and target."""
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {R.shape}")
    if not np.isfinite(R).all():
        raise ValueError("rotation contains non-finite values")
    c = _normalize_vector(center, name="center", eps=eps)
    v = _normalize_vector(major_axis, name="major_axis", eps=eps)
    c_rot = _normalize_vector(R @ c, name="center_rotated", eps=eps)
    v_rot = _normalize_vector(R @ v, name="major_axis_rotated", eps=eps)
    tangent_error = abs(float(np.dot(v_rot, c_rot)))
    if tangent_error > 5e-6:
        raise ValueError("Rotated major axis is not tangent to rotated center")
    gauge_info = compute_gauge_relative_angle_and_target(
        c_rot,
        v_rot,
        gauge=gauge,
        eps=eps,
        min_gauge_projection_norm=min_gauge_projection_norm,
    )
    return {
        "center_rotated": c_rot,
        "major_axis_rotated": v_rot,
        "phi_rotated": float(gauge_info["phi"]),
        "target_rotated": np.asarray(gauge_info["target"], dtype=float),
        "g1_rotated": np.asarray(gauge_info["g1"], dtype=float),
        "g2_rotated": np.asarray(gauge_info["g2"], dtype=float),
    }


def _sphere_log_map(center: np.ndarray, points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Log map on the unit sphere at center for batched unit points."""
    c = _normalize_vector(center, name="center", eps=eps)
    Y = np.asarray(points, dtype=float)
    if Y.ndim != 2 or Y.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {Y.shape}")
    if not np.isfinite(Y).all():
        raise ValueError("points contain non-finite values")
    norms = np.linalg.norm(Y, axis=1)
    if np.any(norms <= eps):
        raise ValueError("points contain near-zero vectors; cannot project to unit sphere")
    Y_unit = Y / norms[:, None]
    dots = np.clip(Y_unit @ c, -1.0, 1.0)
    theta = np.arccos(dots)
    sin_theta = np.sin(theta)
    tangent = Y_unit - dots[:, None] * c[None, :]
    scale = np.ones_like(theta)
    mask = theta > eps
    safe_sin = np.maximum(sin_theta[mask], eps)
    scale[mask] = theta[mask] / safe_sin
    return tangent * scale[:, None]


def anisotropic_gaussian_from_axis(
    vertices: np.ndarray,
    center: np.ndarray,
    major_axis: np.ndarray,
    sigma_parallel: float,
    sigma_perpendicular: float,
    amplitude: float = 1.0,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """Anisotropic Gaussian on S2 tangent space using a physical major axis."""
    if sigma_parallel <= 0 or sigma_perpendicular <= 0:
        raise ValueError(
            f"sigma_parallel and sigma_perpendicular must be positive; got "
            f"{sigma_parallel}, {sigma_perpendicular}"
        )
    c = _normalize_vector(center, name="center", eps=eps)
    v = _normalize_vector(major_axis, name="major_axis", eps=eps)
    tangent_error = abs(float(np.dot(v, c)))
    if tangent_error > 5e-6:
        raise ValueError(
            f"major_axis must be tangent to center. |v·c|={tangent_error} exceeds tolerance"
        )
    v_perp = _normalize_vector(np.cross(c, v), name="major_axis_perpendicular", eps=eps)
    z = _sphere_log_map(c, np.asarray(vertices, dtype=float), eps=eps)
    u_parallel = z @ v
    u_perp = z @ v_perp
    exponent = -0.5 * (
        (u_parallel ** 2) / (sigma_parallel ** 2)
        + (u_perp ** 2) / (sigma_perpendicular ** 2)
    )
    return (float(amplitude) * np.exp(exponent)).astype(float)


def anisotropic_gaussian(
    vertices: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    sigma_u: float,
    sigma_v: float,
    amplitude: float = 1.0,
    orientation_angle: float = 0.0,
    major_axis: Optional[np.ndarray] = None,
    eps: float = 1e-8,
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

    if major_axis is None:
        c_unit = _normalize_vector(center, name="center", eps=eps)
        n_unit = _normalize_vector(normal, name="normal", eps=eps)
        e1, e2 = _stable_tangent_basis(n_unit)
        delta = _wrap_angle_pi(float(orientation_angle))
        major_axis = _normalize_vector(
            np.cos(delta) * e1 + np.sin(delta) * e2,
            name="major_axis",
            eps=eps,
        )
        if abs(float(np.dot(major_axis, c_unit))) > 5e-6:
            # Keep compatibility: project to tangent plane then normalize.
            projected = major_axis - float(np.dot(major_axis, c_unit)) * c_unit
            major_axis = _normalize_vector(projected, name="major_axis_projected", eps=eps)

    return anisotropic_gaussian_from_axis(
        vertices=vertices,
        center=center,
        major_axis=np.asarray(major_axis, dtype=float),
        sigma_parallel=float(sigma_u),
        sigma_perpendicular=float(sigma_v),
        amplitude=float(amplitude),
        eps=eps,
    )