"""Reusable spherical parametrization methods (FLASH and CEM)."""

from __future__ import annotations

import warnings

import numpy as np
import trimesh
from numpy import cross
from scipy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix, find
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.spatial import cKDTree

_EPS_AREA = 1e-14
_EPS_DENOM = 1e-14
_EPS_PROJ = 1e-12


def load_mesh_with_trimesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            raise ValueError(f"No geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Could not load a valid mesh from: {mesh_path}")
    return mesh


def _beltrami_coefficient(v: np.ndarray, f: np.ndarray, mapped: np.ndarray) -> np.ndarray:
    nf = len(f)
    nv = mapped.shape[0]
    mi = np.repeat(np.arange(nf), 3)
    mj = f.flatten("C")

    e1 = v[f[:, 2], :2] - v[f[:, 1], :2]
    e2 = v[f[:, 0], :2] - v[f[:, 2], :2]
    e3 = v[f[:, 1], :2] - v[f[:, 0], :2]

    area = (-e2[:, 0] * e1[:, 1] + e1[:, 0] * e2[:, 1]) / 2
    area_safe = np.where(np.abs(area) < _EPS_AREA, _EPS_AREA, area)
    area_rep = np.repeat(area_safe, 3)

    mx = np.ravel([e1[:, 1], e2[:, 1], e3[:, 1]], order="F") / area_rep / 2
    my = -np.ravel([e1[:, 0], e2[:, 0], e3[:, 0]], order="F") / area_rep / 2

    dx = coo_matrix((mx, (mi, mj)), shape=(nf, nv))
    dy = coo_matrix((my, (mi, mj)), shape=(nf, nv))

    dxdu = dx.dot(mapped[:, 0])
    dxdv = dy.dot(mapped[:, 0])
    dydu = dx.dot(mapped[:, 1])
    dydv = dy.dot(mapped[:, 1])
    dzdu = dx.dot(mapped[:, 2])
    dzdv = dy.dot(mapped[:, 2])

    ecoef = dxdu ** 2 + dydu ** 2 + dzdu ** 2
    gcoef = dxdv ** 2 + dydv ** 2 + dzdv ** 2
    fcoef = dxdu * dxdv + dydu * dydv + dzdu * dzdv

    egf2 = np.maximum(ecoef * gcoef - fcoef ** 2, 0.0)
    denom = ecoef + gcoef + 2.0 * np.sqrt(egf2)
    denom = np.where(denom < _EPS_DENOM, _EPS_DENOM, denom)
    return (ecoef - gcoef + 2j * fcoef) / denom


def _cotangent_laplacian(v: np.ndarray, f: np.ndarray) -> coo_matrix:
    nv = len(v)
    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]

    l1 = np.sqrt(np.sum((v[f2, :] - v[f3, :]) ** 2, axis=1))
    l2 = np.sqrt(np.sum((v[f3, :] - v[f1, :]) ** 2, axis=1))
    l3 = np.sqrt(np.sum((v[f1, :] - v[f2, :]) ** 2, axis=1))

    s = (l1 + l2 + l3) * 0.5
    area = np.sqrt(np.maximum(s * (s - l1) * (s - l2) * (s - l3), 0.0))
    area = np.where(area < _EPS_AREA, _EPS_AREA, area)

    cot12 = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (area * 2)
    cot23 = (l2 ** 2 + l3 ** 2 - l1 ** 2) / (area * 2)
    cot31 = (l1 ** 2 + l3 ** 2 - l2 ** 2) / (area * 2)
    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23

    ii = np.concatenate([f1, f2, f2, f3, f3, f1, f1, f2, f3])
    jj = np.concatenate([f2, f1, f3, f2, f1, f3, f1, f2, f3])
    vv = np.concatenate([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])
    return coo_matrix((vv, (ii, jj)), shape=(nv, nv))


def _find_triangle(f: np.ndarray, v: np.ndarray) -> int:
    temp = v[f.flatten(), :3]
    e1 = np.sqrt(np.sum((temp[1::3, :3] - temp[2::3, :3]) ** 2, axis=1))
    e2 = np.sqrt(np.sum((temp[::3, :3] - temp[2::3, :3]) ** 2, axis=1))
    e3 = np.sqrt(np.sum((temp[::3, :3] - temp[1::3, :3]) ** 2, axis=1))
    regularity = (
        np.abs(e1 / (e1 + e2 + e3) - 1 / 3)
        + np.abs(e2 / (e1 + e2 + e3) - 1 / 3)
        + np.abs(e3 / (e1 + e2 + e3) - 1 / 3)
    )
    return int(np.argmin(regularity))


def _linear_beltrami_solver(
    v: np.ndarray,
    f: np.ndarray,
    mu: np.ndarray,
    landmark: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    nv = len(v)
    landmark = np.asarray(landmark)

    af = (1 - 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)
    bf = -2 * np.imag(mu) / (1.0 - np.abs(mu) ** 2)
    gf = (1 + 2 * np.real(mu) + np.abs(mu) ** 2) / (1.0 - np.abs(mu) ** 2)

    f0, f1, f2 = f[:, 0], f[:, 1], f[:, 2]
    uxv0 = v[f1, 1] - v[f2, 1]
    uyv0 = v[f2, 0] - v[f1, 0]
    uxv1 = v[f2, 1] - v[f0, 1]
    uyv1 = v[f0, 0] - v[f2, 0]
    uxv2 = v[f0, 1] - v[f1, 1]
    uyv2 = v[f1, 0] - v[f0, 0]

    l = np.sqrt(np.column_stack([uxv0 ** 2 + uyv0 ** 2, uxv1 ** 2 + uyv1 ** 2, uxv2 ** 2 + uyv2 ** 2]))
    s = np.sum(l, axis=1) * 0.5
    area = np.sqrt(np.maximum(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]), 0.0))
    area = np.where(area < _EPS_AREA, _EPS_AREA, area)

    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area
    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area

    i = np.concatenate([f0, f1, f2, f0, f1, f1, f2, f2, f0])
    j = np.concatenate([f0, f1, f2, f1, f0, f2, f1, f0, f2])
    vals = np.concatenate([v00, v11, v22, v01, v01, v12, v12, v20, v20]) / 2

    a = coo_matrix((-vals, (i, j)), shape=(nv, nv)).tolil()
    targetc = target[:, 0] + 1j * target[:, 1]
    b = -(a.tocsr()[:, landmark] @ targetc)
    b[landmark] = targetc

    a[landmark, :] = 0
    a[:, landmark] = 0
    a[landmark, landmark] = 1.0
    map_c = spsolve(csr_matrix(a), b)
    return np.column_stack([np.real(map_c), np.imag(map_c)])


def _spherical_tutte_map(f: np.ndarray, bigtri: int, vertex_count: int | None = None) -> np.ndarray:
    """Return the upstream FLASH spherical Tutte fallback map.

    The fallback uses the uniform Tutte Laplacian instead of cotangent weights,
    fixes the selected big-triangle vertices on the unit circle, and applies
    the same stereographic south/north rescaling as the harmonic path.
    """

    f = np.asarray(f, dtype=np.int32)
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("Tutte fallback requires a non-empty triangular connectivity array")
    nv = int(vertex_count if vertex_count is not None else f.max() + 1)
    if nv <= int(f.max()):
        raise ValueError("Tutte fallback vertex count is smaller than a face index")

    # W has weight 1/2 on both directions of every directed triangle edge.
    ii = f.reshape(-1)
    jj = np.roll(f, -1, axis=1).reshape(-1)
    weights = np.full(len(ii), 0.5, dtype=np.float64)
    w = coo_matrix((weights, (ii, jj)), shape=(nv, nv)).tocsr()
    w = w + w.T
    degree = np.asarray(w.sum(axis=1)).ravel()
    laplacian = w - coo_matrix((degree, (np.arange(nv), np.arange(nv))), shape=(nv, nv))

    boundary = np.asarray(f[int(bigtri)], dtype=np.int32)
    rows, cols, values = find(laplacian.tocsr()[boundary, :])
    global_rows = boundary[rows]
    laplacian = (
        laplacian
        - coo_matrix((values, (global_rows, cols)), shape=(nv, nv))
        + coo_matrix((np.ones(3), (boundary, boundary)), shape=(nv, nv))
    ).tocsr()
    rhs = np.zeros(nv, dtype=np.complex128)
    rhs[boundary] = np.exp(1j * (2.0 * np.pi * np.arange(3) / 3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        warnings.simplefilter("error", MatrixRankWarning)
        with np.errstate(all="raise"):
            z = spsolve(laplacian, rhs)
    if not np.isfinite(z).all():
        raise ValueError("spherical Tutte fallback produced non-finite harmonic coordinates")
    z = z - np.mean(z)

    s = np.column_stack(
        [
            2.0 * np.real(z) / (1.0 + np.abs(z) ** 2),
            2.0 * np.imag(z) / (1.0 + np.abs(z) ** 2),
            (-1.0 + np.abs(z) ** 2) / (1.0 + np.abs(z) ** 2),
        ]
    )
    index = np.argsort(np.abs(z[f[:, 0]]) + np.abs(z[f[:, 1]]) + np.abs(z[f[:, 2]]))
    inner = int(index[0])
    if inner == int(bigtri):
        inner = int(index[1])
    north_side = (
        np.abs(z[f[bigtri, 0]] - z[f[bigtri, 1]])
        + np.abs(z[f[bigtri, 1]] - z[f[bigtri, 2]])
        + np.abs(z[f[bigtri, 2]] - z[f[bigtri, 0]])
    ) / 3.0
    north_denom = 1.0 + s[:, 2]
    if np.any(north_denom <= _EPS_PROJ):
        raise ValueError("spherical Tutte fallback encountered a south-pole projection singularity")
    w_complex = s[:, 0] / north_denom + 1j * s[:, 1] / north_denom
    south_side = (
        np.abs(w_complex[f[inner, 0]] - w_complex[f[inner, 1]])
        + np.abs(w_complex[f[inner, 1]] - w_complex[f[inner, 2]])
        + np.abs(w_complex[f[inner, 2]] - w_complex[f[inner, 0]])
    ) / 3.0
    if not np.isfinite(north_side) or not np.isfinite(south_side) or north_side <= _EPS_PROJ or south_side <= _EPS_PROJ:
        raise ValueError("spherical Tutte fallback produced an invalid pole-triangle scale")
    z *= np.sqrt(north_side * south_side) / north_side
    result = np.column_stack(
        [
            2.0 * np.real(z) / (1.0 + np.abs(z) ** 2),
            2.0 * np.imag(z) / (1.0 + np.abs(z) ** 2),
            (-1.0 + np.abs(z) ** 2) / (1.0 + np.abs(z) ** 2),
        ]
    )
    if not np.isfinite(result).all():
        raise ValueError("spherical Tutte fallback produced non-finite sphere coordinates")
    return result


def _sphere_geometry_is_valid(vertices: np.ndarray, faces: np.ndarray) -> bool:
    """Cheap validity guard used before accepting a FLASH solver iterate."""

    if not np.isfinite(vertices).all():
        return False
    norms = np.linalg.norm(vertices, axis=1)
    if len(norms) == 0 or float(np.max(np.abs(norms - 1.0))) > 1e-5:
        return False
    triangles = vertices[faces]
    twice_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.isfinite(twice_areas).all() or len(twice_areas) == 0:
        return False
    threshold = max(1e-12, 1e-4 * float(np.median(twice_areas)))
    if float(np.median(twice_areas)) <= 0.0 or np.any(twice_areas <= threshold):
        return False
    unit = vertices / np.where(norms[:, None] > 0.0, norms[:, None], 1.0)
    if len(unit) > 1 and float(cKDTree(unit).query(unit, k=2)[0][:, 1].min()) <= 1e-5:
        return False
    signed = np.einsum("ij,ij->i", np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), triangles.mean(axis=1))
    return not (np.any(np.abs(signed) <= 1e-12) or (np.any(signed > 1e-12) and np.any(signed < -1e-12)))


def flash_map_with_diagnostics(mesh: trimesh.Trimesh) -> tuple[np.ndarray, dict]:
    """Compute FLASH while retaining a diagnostic record for failed solves."""

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError("FLASH requires a non-empty [V, 3] vertex array")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise ValueError("FLASH requires a non-empty [F, 3] face array")
    if not np.isfinite(v).all() or np.any(f < 0) or np.any(f >= len(v)):
        raise ValueError("FLASH input contains non-finite coordinates or invalid face indices")
    if np.any(f[:, 0] == f[:, 1]) or np.any(f[:, 1] == f[:, 2]) or np.any(f[:, 0] == f[:, 2]):
        raise ValueError("FLASH input contains a face with duplicate vertex indices")
    input_twice_areas = np.linalg.norm(np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0]), axis=1)
    if np.any(input_twice_areas <= _EPS_AREA):
        raise ValueError("FLASH input contains a zero-area face")
    if len(v) - 3 * len(f) / 2 + len(f) != 2:
        raise ValueError("The mesh is not a genus-0 closed surface.")

    diagnostics = {
        "success": False,
        "north_stage": "harmonic",
        "tutte_fallback_used": False,
        "solver_attempts": [],
        "retained_stage": None,
        "error": None,
    }
    bigtri = _find_triangle(f, v)
    nv = v.shape[0]
    m = _cotangent_laplacian(v, f)
    p1, p2, p3 = f[bigtri, :]
    fixed = [p1, p2, p3]
    m_sub = m.tocsr()[fixed, :]
    sub_rows, sub_cols, mval = find(m_sub)
    global_rows = np.array(fixed)[sub_rows]
    m = m - coo_matrix((mval, (global_rows, sub_cols)), shape=(nv, nv)) + coo_matrix((np.ones(3), (fixed, fixed)), shape=(nv, nv))

    x1, y1, x2, y2 = 0.0, 0.0, 1.0, 0.0
    a = v[p2] - v[p1]
    b = v[p3] - v[p1]
    sin1 = norm(cross(a, b)) / (norm(a) * norm(b))
    ori_h = norm(b) * sin1
    ratio = norm([x1 - x2, y1 - y2]) / norm(a)
    y3 = ori_h * ratio
    x3_square = norm(b) ** 2 * ratio ** 2 - y3 ** 2
    if x3_square < -_EPS_PROJ:
        raise ValueError("FLASH big-triangle boundary calculation produced a negative square")
    x3 = np.sqrt(max(0.0, x3_square))
    c = np.zeros(nv)
    c[p1], c[p2], c[p3] = x1, x2, x3
    d = np.zeros(nv)
    d[p1], d[p2], d[p3] = y1, y2, y3
    harmonic_error = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            warnings.simplefilter("error", MatrixRankWarning)
            z = spsolve(m, c + 1j * d)
        if not np.isfinite(z).all():
            raise ValueError("harmonic solve returned non-finite coordinates")
        z = z - np.mean(z)
    except Exception as exc:
        z = None
        harmonic_error = str(exc)

    def inverse_north(values: np.ndarray) -> np.ndarray:
        denominator = 1.0 + np.abs(values) ** 2
        return np.column_stack([2 * np.real(values) / denominator, 2 * np.imag(values) / denominator, (-1 + np.abs(values) ** 2) / denominator])

    def rescale(values: np.ndarray) -> np.ndarray:
        sphere = inverse_north(values)
        denominator = 1.0 + sphere[:, 2]
        if np.any(denominator <= _EPS_PROJ):
            raise ValueError("FLASH south-pole projection denominator is singular")
        projected = sphere[:, 0] / denominator + 1j * sphere[:, 1] / denominator
        index = np.argsort(np.abs(values[f[:, 0]]) + np.abs(values[f[:, 1]]) + np.abs(values[f[:, 2]]))
        inner = int(index[0]) if int(index[0]) != bigtri else int(index[1])
        north_side = (np.abs(values[f[bigtri, 0]] - values[f[bigtri, 1]]) + np.abs(values[f[bigtri, 1]] - values[f[bigtri, 2]]) + np.abs(values[f[bigtri, 2]] - values[f[bigtri, 0]])) / 3.0
        south_side = (np.abs(projected[f[inner, 0]] - projected[f[inner, 1]]) + np.abs(projected[f[inner, 1]] - projected[f[inner, 2]]) + np.abs(projected[f[inner, 2]] - projected[f[inner, 0]])) / 3.0
        if not np.isfinite(north_side) or not np.isfinite(south_side) or north_side <= _EPS_PROJ or south_side <= _EPS_PROJ:
            raise ValueError("FLASH pole-triangle rescaling is singular")
        return inverse_north(values * np.sqrt(north_side * south_side) / north_side)

    try:
        if z is None:
            raise ValueError(harmonic_error or "harmonic solve failed")
        harmonic_sphere = rescale(z)
    except Exception as exc:
        diagnostics["tutte_fallback_used"] = True
        diagnostics["north_stage"] = "tutte"
        harmonic_sphere = _spherical_tutte_map(f, bigtri, nv)
        diagnostics["north_error"] = str(exc)
    retained = harmonic_sphere
    diagnostics["retained_stage"] = "harmonic" if not diagnostics["tutte_fallback_used"] else "tutte"
    i = np.argsort(harmonic_sphere[:, 2])
    fixnum = max(round(len(v) / 10), 3)
    fixed = i[: min(len(v), fixnum)]
    denominator = 1.0 + harmonic_sphere[:, 2]
    if np.any(denominator <= _EPS_PROJ):
        diagnostics["error"] = "FLASH south-pole projection denominator is singular"
        return retained, diagnostics
    p = np.column_stack([harmonic_sphere[:, 0] / denominator, harmonic_sphere[:, 1] / denominator])
    mu = _beltrami_coefficient(p, f, v)
    if not np.isfinite(mu).all() or float(np.max(np.abs(mu))) >= 1.0 - 1e-10:
        diagnostics["error"] = "FLASH Beltrami coefficient is non-finite or singular (|mu| >= 1)"
        return retained, diagnostics

    mapped = None
    for attempt, landmark_count in enumerate((len(fixed), min(len(v), fixnum * 5)), start=1):
        if attempt > 1 and landmark_count == len(fixed):
            continue
        landmarks = i[:landmark_count]
        record = {"attempt": attempt, "landmark_count": int(landmark_count)}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                warnings.simplefilter("error", MatrixRankWarning)
                candidate = _linear_beltrami_solver(p, f, mu, landmarks, p[landmarks, :])
            record["finite"] = bool(np.isfinite(candidate).all())
            if not record["finite"]:
                raise ValueError("solver returned non-finite coordinates")
            candidate_z = candidate[:, 0] + 1j * candidate[:, 1]
            candidate_sphere = np.column_stack([2 * np.real(candidate_z) / (1 + np.abs(candidate_z) ** 2), 2 * np.imag(candidate_z) / (1 + np.abs(candidate_z) ** 2), -(np.abs(candidate_z) ** 2 - 1) / (1 + np.abs(candidate_z) ** 2)])
            record["geometry_valid"] = _sphere_geometry_is_valid(candidate_sphere, f)
            if record["geometry_valid"]:
                mapped = candidate_sphere
                diagnostics["solver_attempts"].append(record)
                diagnostics["success"] = True
                diagnostics["retained_stage"] = "beltrami"
                return mapped, diagnostics
            raise ValueError("solver result is geometrically collapsed or folded")
        except Exception as exc:
            record["error"] = str(exc)
            diagnostics["solver_attempts"].append(record)
    diagnostics["error"] = "FLASH Beltrami solver failed after all landmark retries"
    diagnostics["retained_stage"] = "harmonic" if not diagnostics["tutte_fallback_used"] else "tutte"
    return retained, diagnostics


def flash_map(mesh: trimesh.Trimesh) -> np.ndarray:
    return flash_map_with_diagnostics(mesh)[0]
