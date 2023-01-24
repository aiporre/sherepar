import numpy as np

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
            if np.absolute(h) < radius:
                I.append(i)
        B = [i for i in range(len(h)) if i not in I]
        return I, B
    B = a.id, b.id, c.id
    N = Ld.shape[0]
    I = get_indices_I_from_B(N, B)
    # solving the linear system of equation
    A_coeff = Ld[I, I]
    b_coeff = Ld[I, B].dot(h_b_zero)
    h_i = np.linalg.solve(A_coeff, b_coeff)
    count = 0
    h = np.zeros((N,1))
    h[B] = h_b_zero
    h[I] = h_i
    while count < 100:
        count += 1
        # inversion:
        h = np.diag(1/np.absolute(h)**2).dot(h)
        I, B = get_indices_I_B_radius()
        # solving again new h_I
        A_coeff = Ld[I, I]
        h_b = h[B]
        b_coeff = Ld[I, B].dot(h_b)
        h_i = np.linalg.solve(A_coeff, b_coeff)
        # update the new values of h_I
        h[I] = h_i

