from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from trimesh import Trimesh
import numpy as np
from scipy.sparse import find, coo_matrix, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
from numpy import cross

import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Numerical guard constants
_EPS_AREA  = 1e-14  # minimum triangle area: prevents division-by-zero in cotangent
                    # and Beltrami coefficient when triangles are nearly degenerate
_EPS_DENOM = 1e-14  # minimum |denominator| in Beltrami mu: guards against a
                    # collapsed/constant map where E+G+2√(EG-F²) → 0
_EPS_PROJ  = 1e-12  # minimum south-pole stereographic denominator (1+S[:,2]):
                    # larger than _EPS_AREA because S[:,2] ∈ [-1,1] and even a
                    # numerically-exact south-pole vertex gives 1+S[:,2] = O(ε_mach²)


def beltrami_coefficient(v, f, map):
    nf = len(f)
    nv = map.shape[0]

    # BUG 1 FIX: Mi must repeat each face index 3 times (face-grouped) and Mj
    # must list vertices in row-major (C) order so that Mi[k] and Mj[k] index
    # the same face.  The old code used np.tile (column-grouped Mi) with
    # f.flatten('F') (column-grouped Mj) but Mx/My are face-grouped, so every
    # triplet after k=0 was mismatched.
    Mi = np.repeat(np.arange(nf), 3)   # [0,0,0, 1,1,1, ..., nf-1,nf-1,nf-1]
    Mj = f.flatten('C')                 # [f[0,0],f[0,1],f[0,2], f[1,0],...]

    e1 = v[f[:, 2], :2] - v[f[:, 1], :2]
    e2 = v[f[:, 0], :2] - v[f[:, 2], :2]
    e3 = v[f[:, 1], :2] - v[f[:, 0], :2]

    area = (-e2[:, 0] * e1[:, 1] + e1[:, 0] * e2[:, 1]) / 2
    # Guard against degenerate (zero-area) triangles before dividing.
    area_safe = np.where(np.abs(area) < _EPS_AREA, _EPS_AREA, area)
    area_rep = np.repeat(area_safe, 3)  # face-grouped: [a0,a0,a0, a1,a1,a1, ...]

    Mx = np.ravel([e1[:, 1], e2[:, 1], e3[:, 1]], order='F') / area_rep / 2
    My = -np.ravel([e1[:, 0], e2[:, 0], e3[:, 0]], order='F') / area_rep / 2

    # Explicit shape (nf, nv) avoids undersized matrix when nv > max(Mj)+1.
    Dx = coo_matrix((Mx, (Mi, Mj)), shape=(nf, nv))
    Dy = coo_matrix((My, (Mi, Mj)), shape=(nf, nv))

    dXdu = Dx.dot(map[:, 0]); dXdv = Dy.dot(map[:, 0])
    dYdu = Dx.dot(map[:, 1]); dYdv = Dy.dot(map[:, 1])
    dZdu = Dx.dot(map[:, 2]); dZdv = Dy.dot(map[:, 2])

    E = dXdu ** 2 + dYdu ** 2 + dZdu ** 2
    G = dXdv ** 2 + dYdv ** 2 + dZdv ** 2
    F = dXdu * dXdv + dYdu * dYdv + dZdu * dZdv

    # Stability: EG - F² >= 0 by Cauchy-Schwarz but floating-point can give a
    # tiny negative value.  Clamp before sqrt to avoid NaN.
    EGF2 = np.maximum(E * G - F ** 2, 0.0)
    denom = E + G + 2.0 * np.sqrt(EGF2)
    # Guard denominator against zero (degenerate / collapsed mapping).
    # The denominator E+G+2√(EG-F²) is always ≥ 0 for real E,G,F, so
    # preserving a positive floor (rather than a signed one) is correct here.
    denom = np.where(denom < _EPS_DENOM, _EPS_DENOM, denom)
    mu = (E - G + 2j * F) / denom

    return mu


def cotangent_laplacian(v, f):
    nv = len(v)

    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]

    l1 = np.sqrt(np.sum((v[f2, :] - v[f3, :]) ** 2, axis=1))
    l2 = np.sqrt(np.sum((v[f3, :] - v[f1, :]) ** 2, axis=1))
    l3 = np.sqrt(np.sum((v[f1, :] - v[f2, :]) ** 2, axis=1))

    s = (l1 + l2 + l3) * 0.5
    # Stability: clamp Heron's product before sqrt to avoid NaN from tiny
    # negative values caused by floating-point rounding on near-degenerate
    # triangles.
    area = np.sqrt(np.maximum(s * (s - l1) * (s - l2) * (s - l3), 0.0))
    area = np.where(area < _EPS_AREA, _EPS_AREA, area)

    cot12 = (l1 ** 2 + l2 ** 2 - l3 ** 2) / (area * 2)
    cot23 = (l2 ** 2 + l3 ** 2 - l1 ** 2) / (area * 2)
    cot31 = (l1 ** 2 + l3 ** 2 - l2 ** 2) / (area * 2)
    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23

    II = np.concatenate([f1, f2, f2, f3, f3, f1, f1, f2, f3])
    JJ = np.concatenate([f2, f1, f3, f2, f1, f3, f1, f2, f3])
    V = np.concatenate([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])
    L = coo_matrix((V, (II, JJ)), shape=(nv, nv))

    return L


def find_triangle(f, v):
    # Find the most regular triangle as the "big triangle"
    temp = v[f.flatten(), :3]
    e1 = np.sqrt(np.sum((temp[1::3, :3] - temp[2::3, :3]) ** 2, axis=1))
    e2 = np.sqrt(np.sum((temp[::3, :3] - temp[2::3, :3]) ** 2, axis=1))
    e3 = np.sqrt(np.sum((temp[::3, :3] - temp[1::3, :3]) ** 2, axis=1))
    regularity = np.abs(e1 / (e1 + e2 + e3) - 1 / 3) + np.abs(e2 / (e1 + e2 + e3) - 1 / 3) + np.abs(
        e3 / (e1 + e2 + e3) - 1 / 3)
    bigtri = np.argmin(regularity)
    return bigtri


def linear_beltrami_solver(v, f, mu, landmark, target):
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

    # Stability: clamp Heron's product before sqrt (same as cotangent_laplacian).
    area = np.sqrt(np.maximum(s * (s - l[:, 0]) * (s - l[:, 1]) * (s - l[:, 2]), 0.0))
    area = np.where(area < _EPS_AREA, _EPS_AREA, area)

    v00 = (af * uxv0 * uxv0 + 2 * bf * uxv0 * uyv0 + gf * uyv0 * uyv0) / area
    v11 = (af * uxv1 * uxv1 + 2 * bf * uxv1 * uyv1 + gf * uyv1 * uyv1) / area
    v22 = (af * uxv2 * uxv2 + 2 * bf * uxv2 * uyv2 + gf * uyv2 * uyv2) / area
    v01 = (af * uxv1 * uxv0 + bf * uxv1 * uyv0 + bf * uxv0 * uyv1 + gf * uyv1 * uyv0) / area
    v12 = (af * uxv2 * uxv1 + bf * uxv2 * uyv1 + bf * uxv1 * uyv2 + gf * uyv2 * uyv1) / area
    v20 = (af * uxv0 * uxv2 + bf * uxv0 * uyv2 + bf * uxv2 * uyv0 + gf * uyv0 * uyv2) / area

    I = np.concatenate([f0, f1, f2, f0, f1, f1, f2, f2, f0])
    J = np.concatenate([f0, f1, f2, f1, f0, f2, f1, f0, f2])
    V = np.concatenate([v00, v11, v22, v01, v01, v12, v12, v20, v20]) / 2

    # BUG 3 FIX: keep A sparse throughout.  Converting to a dense array
    # (a) costs O(nv²) memory, (b) makes spsolve fail because it requires a
    # sparse matrix as its first argument.  Use lil_matrix for efficient
    # row/column zeroing, then convert to csr only for the solve.
    A = coo_matrix((-V, (I, J)), shape=(nv, nv)).tolil()

    targetc = target[:, 0] + 1j * target[:, 1]
    # Compute rhs BEFORE zeroing the landmark columns (matching MATLAB order).
    b = -(A.tocsr()[:, landmark] @ targetc)
    b[landmark] = targetc

    A[landmark, :] = 0
    A[:, landmark] = 0
    A[landmark, landmark] = 1.0   # enforce boundary condition: map[landmark] = target[landmark]

    map_c = spsolve(csr_matrix(A), b)
    map = np.column_stack([np.real(map_c), np.imag(map_c)])

    return map


def spherical_tutte_map(f, bigtri):
    raise NotImplementedError('Tutte map is not implemented yet.')


def flash_map(mesh: Trimesh):
    v, f = mesh.vertices, mesh.faces

    # Check whether the input mesh is genus-0
    if len(v) - 3 * len(f) / 2 + len(f) != 2:
        raise ValueError('The mesh is not a genus-0 closed surface.')

    bigtri = find_triangle(f, v)

    # North pole step: Compute spherical map by solving laplace equation on a big triangle
    nv = v.shape[0]
    M = cotangent_laplacian(v, f)

    p1, p2, p3 = f[bigtri, :]

    fixed = [p1, p2, p3]
    # BUG 2 FIX: find() on a submatrix returns LOCAL row indices (0,1,2).
    # The old code used those local indices directly as global vertex indices,
    # corrupting rows 0/1/2 instead of rows p1/p2/p3.
    # Fix: slice the CSR matrix (no full dense materialisation) and map
    # local sub-row indices back to global vertex indices via fixed[sub_rows].
    M_sub = M.tocsr()[fixed, :]
    sub_rows, sub_cols, mval = find(M_sub)
    global_rows = np.array(fixed)[sub_rows]   # local {0,1,2} -> global {p1,p2,p3}
    M = (M
         - coo_matrix((mval, (global_rows, sub_cols)), shape=(nv, nv))
         + coo_matrix((np.ones(len(fixed)), (fixed, fixed)), shape=(nv, nv)))

    # set the boundary condition for big triangle
    x1, y1, x2, y2 = 0, 0, 1, 0  # arbitrarily set the two points
    a = v[p2, :3] - v[p1, :3]
    b = v[p3, :3] - v[p1, :3]
    sin1 = norm(cross(a, b)) / (norm(a) * norm(b))
    ori_h = norm(b) * sin1
    ratio = norm([x1 - x2, y1 - y2]) / norm(a)
    y3 = ori_h * ratio  # compute the coordinates of the third vertex
    x3 = np.sqrt(norm(b) ** 2 * ratio ** 2 - y3 ** 2)

    # Solve the Laplace equation to obtain a harmonic map
    c = np.zeros(nv)
    c[p1] = x1
    c[p2] = x2
    c[p3] = x3
    d = np.zeros(nv)
    d[p1] = y1
    d[p2] = y2
    d[p3] = y3
    z = spsolve(M, c + 1j * d)
    z = z - np.mean(z)

    # inverse stereographic projection
    S = np.column_stack([2 * np.real(z) / (1 + np.abs(z) ** 2), 2 * np.imag(z) / (1 + np.abs(z) ** 2),
                         (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2)])

    w = np.array(S[:, 0] / (1 + S[:, 2]) + 1j * S[:, 1] / (1 + S[:, 2]))

    index = np.argsort(np.abs(z[f[:, 0]]) + np.abs(z[f[:, 1]]) + np.abs(z[f[:, 2]]))
    inner = index[0]
    if inner == bigtri:
        inner = index[1]
    # Compute the size of the northern most and the southern most triangles
    NorthTriSide = (np.abs(z[f[bigtri, 0]] - z[f[bigtri, 1]]) +
                    np.abs(z[f[bigtri, 1]] - z[f[bigtri, 2]]) +
                    np.abs(z[f[bigtri, 2]] - z[f[bigtri, 0]])) / 3

    SouthTriSide = (np.abs(w[f[inner, 0]] - w[f[inner, 1]]) +
                    np.abs(w[f[inner, 1]] - w[f[inner, 2]]) +
                    np.abs(w[f[inner, 2]] - w[f[inner, 0]])) / 3

    # rescale to get the best distribution
    z = z * (np.sqrt(NorthTriSide * SouthTriSide)) / NorthTriSide

    # inverse stereographic projection
    S = np.column_stack([2 * np.real(z) / (1 + np.abs(z) ** 2),
                         2 * np.imag(z) / (1 + np.abs(z) ** 2),
                         (-1 + np.abs(z) ** 2) / (1 + np.abs(z) ** 2)])

    if np.isnan(S).sum() != 0:
        # if harmonic map fails due to very bad triangulations, use tutte map
        S = spherical_tutte_map(f, bigtri)

    # south pole step: Compute the harmonic map from the spherical map
    I = np.argsort(S[:, 2])

    # number of points near the south pole to be fixed
    # simply set it to be 1/10 of the total number of vertices (can be changed)
    # In case the spherical parameterization is not good, change 10 to
    # something smaller (e.g. 2)
    fixnum = max(round(len(v) / 10), 3)
    fixed = I[:min(len(v), fixnum)]

    # south pole stereographic projection
    # Stability: clamp 1+S[:,2] away from zero to avoid division by zero when a
    # vertex sits at (or very close to) the south pole (S[:,2] = -1).
    sp_denom = np.maximum(1.0 + S[:, 2], _EPS_PROJ)
    P = np.column_stack([S[:, 0] / sp_denom, S[:, 1] / sp_denom])

    # compute the Beltrami coefficient
    mu = beltrami_coefficient(P, f, v)

    # compose the map with another quasi-conformal map to cancel the distortion
    map = linear_beltrami_solver(P, f, mu, fixed, P[fixed, :])

    if np.isnan(map).sum() != 0:
        # if the result has NaN entries, then most probably the number of
        # boundary constraints is not large enough

        # increase the number of boundary constrains and run again
        fixnum = fixnum * 5  # again, this number can be changed
        fixed = I[:min(len(v), fixnum)]
        map = linear_beltrami_solver(P, f, mu, fixed, P[fixed, :])

        if np.isnan(map).sum() != 0:
            map = P  # use the old result

    z = map[:, 0] + 1j * map[:, 1]

    # inverse south pole stereographic projection
    map = np.column_stack([2 * np.real(z) / (1 + np.abs(z) ** 2),
                           2 * np.imag(z) / (1 + np.abs(z) ** 2),
                           -(np.abs(z) ** 2 - 1) / (1 + np.abs(z) ** 2)])
    return map


def plot_mesh(mesh, ax=None):
    verts, faces, _, _ = mesh

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    ax_given = ax is not None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    mesh.set_facecolor('r')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")
    xx = verts[:, 0]
    yy = verts[:, 1]
    zz = verts[:, 2]
    ax.set_xlim(min(xx), max(xx))  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(min(yy), max(yy))  # b = 10
    ax.set_zlim(min(zz), max(zz))  # c = 16

    ax.set_aspect("equal")
    if not ax_given:
        plt.tight_layout()
        plt.show()


def main():
    # Load the mesh from the .ply file
    # mesh = trimesh.load_mesh("data/suzanne.obj")
    # mesh = trimesh.load_mesh("/home/sauron/Documents/Phd/code/sherepar/data/tr_reg_000.ply")
    mesh = trimesh.load_mesh("/home/sauron/Documents/Phd/code/sherepar/data/ellipsoid.obj")

    # Create a new figure for the plot
    fig = plt.figure()

    # Create a 3D subplot
    ax = fig.add_subplot(121, projection='3d')

    # Plot the vertices of the mesh
    # ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], s=0.1)
    plot_mesh((mesh.vertices, mesh.faces, None, None), ax=ax)

    # Set the labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Compute the flash map
    map = flash_map(mesh)
    map = map - map.mean(axis=0)
    # plot
    ax = fig.add_subplot(122, projection='3d')
    plot_mesh((map, mesh.faces, None, None), ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal', adjustable='box')
    # Display the plot
    plt.show()


if __name__ == "__main__":
    main()