from typing import Any

import numpy as np
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit

from spherepar.mesh import MeshSurf, StretchFunction, Vector, Vertex


def dirichlet_spherepar(mesh: MeshSurf) -> StretchFunction:
    face_reg = mesh.get_most_regular_face()
    # Laplacian matrix dirichlet energy, i.e. cotangent formula for weights
    Ld = mesh.get_laplacian_matrix(weight="cotangent").toarray()
    a, b, c = face_reg.u, face_reg.v, face_reg.w

    # solution of the Laplace-Beltrami equation
    h_b_real = -1 / Vector(b, a).norm()
    alpha = Vector(c, a).dot(Vector(b, a)) / (Vector(b, a).norm() ** 2)
    h_b_img = 1 / Vector(c, Vertex(a.pos + alpha * (b.pos - a.pos), _id=-1)).norm()
    j = np.array(1.j)
    h_b_zero = np.array([h_b_real, h_b_real, 0]) \
               + np.array([j * (1 - alpha) * h_b_img, j * alpha * h_b_img, -h_b_img])

    # calculating the matrices
    def get_indices_I_from_B(num_vertices, set_B):
        return [i for i in range(num_vertices) if i not in set_B]

    def get_indices_I_B_radius(h, radius=1.2):
        I = []
        for i in range(len(h)):
            if np.absolute(h[i]) < radius:
                I.append(i)
        B = [i for i in range(len(h)) if i not in I]
        return I, B

    B = [a.id, b.id, c.id]
    N = Ld.shape[0]
    I = get_indices_I_from_B(N, B)
    # solving the linear system of equation
    A_coeff = Ld[I][..., I]
    b_coeff = -Ld[I][..., B].dot(h_b_zero)
    h_i = np.linalg.solve(A_coeff, b_coeff)
    h = np.zeros((N,), dtype=complex)
    h[B] = h_b_zero
    h[I] = h_i
    count = 0
    max_iters = 1000
    while count < max_iters:
        count += 1
        # inversion:
        h = np.diag(1 / np.absolute(h) ** 2).dot(h)
        I, B = get_indices_I_B_radius(h)
        if len(B) == 0:
            print('Converged all vertices under the radius.')
            break
        # solving again new h_I
        A_coeff = Ld[I][..., I]
        h_b = h[B]
        b_coeff = -Ld[I][..., B].dot(h_b)
        h_i = np.linalg.solve(A_coeff, b_coeff)
        # update the new values of h_I
        h[I] = h_i
    return StretchFunction(mesh, h)


def stereo_projection(vertex: Vertex) -> ndarray[Any, dtype[floating[_64Bit]]]:
    """
    Stereographic projection of a vertex
    :param vertex:
    :return:
        ndarray[Any, type[complex[float64]]]
    """
    j = np.array(1.j)
    return (vertex.pos[0] + j * vertex.pos[1]) / (1 - vertex.pos[2])


def strech_paremetrization(mesh: MeshSurf) -> StretchFunction:
    # the initial mapping is a dirichlet energy minimization aka dirichlet_s
    dirichlet_stretch = dirichlet_spherepar(mesh)
    # strereo-graphic projection of the mesh
    vertices = mesh.get_vertices_collection()
    strech_vertices = [dirichlet_stretch(v) for v in vertices]
    # create a list of sterep-graphic projected vertices
    h = np.array([stereo_projection(v) for v in strech_vertices])
    # while max iterations reached
    count = 0
    max_iters = 1000

    def get_indices_I_B_radius(h, radius=1.2):
        I = []
        for i in range(len(h)):
            if np.absolute(h[i]) < radius:
                I.append(i)
        B = [i for i in range(len(h)) if i not in I]
        return I, B

    while count < max_iters:
        count += 1
        # Update linear equation matrices A with the laplacian stretch matrix
        Ls = mesh.get_laplacian_matrix(weight="stretch", stretch=h).toarray()
        # inversion:
        h = np.diag(1 / np.absolute(h) ** 2).dot(h)
        I, B = get_indices_I_B_radius(h)
        if len(B) == 0:
            print('Converged all vertices under the radius.')
            break
        # solving again new h_I
        A_coeff = Ls[I][..., I]
        h_b = h[B]
        b_coeff = -Ls[I][..., B].dot(h_b)
        h_i = np.linalg.solve(A_coeff, b_coeff)
        # update the new values of h_I
        h[I] = h_i
        # update the stretch function
        dirichlet_stretch.h = h
        # update h values
        strech_vertices = [dirichlet_stretch(v) for v in vertices]
        h = np.array([stereo_projection(v) for v in strech_vertices])

    return dirichlet_stretch
