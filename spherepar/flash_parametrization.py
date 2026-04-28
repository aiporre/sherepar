"""Reusable spherical parametrization methods (FLASH and CEM)."""

from __future__ import annotations

import numpy as np
import trimesh
from numpy import cross
from scipy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix, find
from scipy.sparse.linalg import spsolve

_EPS_AREA = 1e-14
_EPS_DENOM = 1e-14
_EPS_PROJ = 1e-12


def load_mesh_with_trimesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh")
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


def _spherical_tutte_map(_f: np.ndarray, _bigtri: int) -> np.ndarray:
    raise NotImplementedError("Tutte map is not implemented yet.")


def flash_map(mesh: trimesh.Trimesh) -> np.ndarray:
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)
    if len(v) - 3 * len(f) / 2 + len(f) != 2:
        raise ValueError("The mesh is not a genus-0 closed surface.")

    bigtri = _find_triangle(f, v)
    nv = v.shape[0]
    m = _cotangent_laplacian(v, f)
    p1, p2, p3 = f[bigtri, :]
    fixed = [p1, p2, p3]

    m_sub = m.tocsr()[fixed, :]
    sub_rows, sub_cols, mval = find(m_sub)
    global_rows = np.array(fixed)[sub_rows]
    m = (
        m
        - coo_matrix((mval, (global_rows, sub_cols)), shape=(nv, nv))
        + coo_matrix((np.ones(len(fixed)), (fixed, fixed)), shape=(nv, nv))
    )

    x1, y1, x2, y2 = 0, 0, 1, 0
    a = v[p2, :3] - v[p1, :3]
    b = v[p3, :3] - v[p1, :3]
    sin1 = norm(cross(a, b)) / (norm(a) * norm(b))
    ori_h = norm(b) * sin1
    ratio = norm([x1 - x2, y1 - y2]) / norm(a)
    y3 = ori_h * ratio
    x3 = np.sqrt(norm(b) ** 2 * ratio ** 2 - y3 ** 2)

    c = np.zeros(nv)
    c[p1], c[p2], c[p3] = x1, x2, x3
    d = np.zeros(nv)
    d[p1], d[p2], d[p3] = y1, y2, y3
    z = spsolve(m, c + 1j * d)
    z = z - np.mean(z)

    s = np.column_stack([
        2 * np.real(z) / (1 + np.abs(z) ** 2),
        2 * np.imag(z) / (1 + np.abs(z) ** 2),
        (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2),
    ])

    w = np.array(s[:, 0] / (1 + s[:, 2]) + 1j * s[:, 1] / (1 + s[:, 2]))
    index = np.argsort(np.abs(z[f[:, 0]]) + np.abs(z[f[:, 1]]) + np.abs(z[f[:, 2]]))
    inner = int(index[0])
    if inner == bigtri:
        inner = int(index[1])

    north_tri_side = (
        np.abs(z[f[bigtri, 0]] - z[f[bigtri, 1]])
        + np.abs(z[f[bigtri, 1]] - z[f[bigtri, 2]])
        + np.abs(z[f[bigtri, 2]] - z[f[bigtri, 0]])
    ) / 3
    south_tri_side = (
        np.abs(w[f[inner, 0]] - w[f[inner, 1]])
        + np.abs(w[f[inner, 1]] - w[f[inner, 2]])
        + np.abs(w[f[inner, 2]] - w[f[inner, 0]])
    ) / 3
    z = z * (np.sqrt(north_tri_side * south_tri_side)) / north_tri_side

    s = np.column_stack([
        2 * np.real(z) / (1 + np.abs(z) ** 2),
        2 * np.imag(z) / (1 + np.abs(z) ** 2),
        (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2),
    ])
    if np.isnan(s).sum() != 0:
        s = _spherical_tutte_map(f, bigtri)

    i = np.argsort(s[:, 2])
    fixnum = max(round(len(v) / 10), 3)
    fixed = i[: min(len(v), fixnum)]

    sp_denom = np.maximum(1.0 + s[:, 2], _EPS_PROJ)
    p = np.column_stack([s[:, 0] / sp_denom, s[:, 1] / sp_denom])
    mu = _beltrami_coefficient(p, f, v)
    mapped = _linear_beltrami_solver(p, f, mu, fixed, p[fixed, :])

    if np.isnan(mapped).sum() != 0:
        fixnum = fixnum * 5
        fixed = i[: min(len(v), fixnum)]
        mapped = _linear_beltrami_solver(p, f, mu, fixed, p[fixed, :])
        if np.isnan(mapped).sum() != 0:
            mapped = p

    z = mapped[:, 0] + 1j * mapped[:, 1]
    return np.column_stack([
        2 * np.real(z) / (1 + np.abs(z) ** 2),
        2 * np.imag(z) / (1 + np.abs(z) ** 2),
        -(np.abs(z) ** 2 - 1) / (1 + np.abs(z) ** 2),
    ])


