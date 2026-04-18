"""Spherical conformal parametrization.

Implements Algorithm 4.1 (initial spherical conformal map) and Algorithm 4.2
(CEM iteration) from:
  "A Novel Algorithm for Volume-Preserving Parameterizations of 3-Manifolds"

# =============================================================================
# DEBUG CHECKLIST
# =============================================================================
# After Algorithm 4.1 (dirichlet_parametrization):
#   [A4.1-1] h_B.shape == (3,) and h_B.dtype == complex128
#   [A4.1-2] L_D symmetric:  max|L_D - L_D^T| < 1e-10
#   [A4.1-3] L_D row-sums == 0:  max|L_D @ 1| < 1e-10  (nullspace = constants)
#   [A4.1-4] Reduced system: A_coeff.shape == (|I|, |I|), rhs.shape == (|I|,)
#   [A4.1-5] No NaN/Inf in h after solve
#   [A4.1-6] All sphere points ||f_i|| ~= 1  (atol < 1e-10)
#   [A4.1-7] h centered (mean ~= 0) and north/south poles balanced
#
# After each Algorithm 4.2 (stretch_parametrization) iteration:
#   [A4.2-1] |I| + |B| == N
#   [A4.2-2] min/max |h_i| printed each iteration
#   [A4.2-3] All sphere points ||f_i|| ~= 1
#   [A4.2-4] No NaN/Inf in h_I after solve
#   [A4.2-5] delta = E_D(g) - E_D(f) does not explode
#
# =============================================================================
# PAPER-TO-CODE MISMATCH TABLE
# =============================================================================
# Paper step        | Expected equation            | Old code (bug)
#                   |                              |   -> fix
# ------------------|------------------------------|----------------------------
# 4.1 Step 4 real   | h_B[0] = -1/||vb-va||^2     | h_b_real = -1/norm
#                   | h_B[1] =  1/||vb-va||^2      |   -> [-h_b,h_b,0]=[+1/norm,-1/norm,0]
#                   |                              |   (sign reversed + missing **2)
#                   |                              |   Fix: inv_sq_edge=1/norm**2
# ------------------|------------------------------|----------------------------
# 4.1 Step 4 imag   | coeff = 1/||vc-foot||^2      | h_b_img = 1/norm  (not squared)
#                   |                              |   Fix: inv_sq_foot=1/norm**2
# ------------------|------------------------------|----------------------------
# 4.1 scope         | Steps 1-7, no loop           | Contained a CEM loop at end
#                   |                              |   Fix: loop removed
# ------------------|------------------------------|----------------------------
# 4.1 Step 8        | (new) center h; rescale for  | Not done at all
#                   | balanced pole coverage       |   Fix: centering + FLASH-style
#                   | (FLASH north/south rescale)  |         rescaling added
# ------------------|------------------------------|----------------------------
# 4.2 Step 3b       | r = 1  (unit-circle boundary | r = median(|h|)  (drifts
#                   | of Möbius inversion)         |   arbitrarily away from 1)
#                   |                              |   Fix: r = 1.0
# ------------------|------------------------------|----------------------------
# 4.2 Step 3c       | [L_D]_{I,I} h_I = ...       | Used Ls (stretch Laplacian)
#                   |                              |   Fix: use Ld (cotangent)
# ------------------|------------------------------|----------------------------
# 4.2 Step 3e       | delta = E_D(g) - E_D(f)     | Not computed at all
#                   |                              |   Fix: added _dirichlet_energy
# ------------------|------------------------------|----------------------------
# 4.2 Step 3f       | stop only when 0<=delta<=eps | stop when delta <= eps
#                   | (small positive improvement) |   (also stops on divergence)
#                   |                              |   Fix: 0 <= delta <= eps
# ------------------|------------------------------|----------------------------
# 4.2 Step 3a       | h_i <- h_i / |h_i|^2        | No guard for |h_i|=0
#                   |                              |   Fix: clamp |h|^2 >= _EPS_INV
# ------------------|------------------------------|----------------------------
# Stereo proj       | g1/(1-g3) + i*g2/(1-g3)     | Blows up when g3 ~= 1
#                   |                              |   Fix: clamp denom >= _EPS_PROJ
# =============================================================================
"""

from typing import Any

import numpy as np

from spherepar.mesh import MeshSurf, StretchFunction, Vector, Vertex

# ---------------------------------------------------------------------------
# Numerical safeguards
# ---------------------------------------------------------------------------
_EPS_PROJ = 1e-12   # minimum |1 - z| in stereographic projection (north-pole guard)
_EPS_INV  = 1e-14   # minimum |h|^2 in Mobius inversion step (zero-division guard)


# ---------------------------------------------------------------------------
# Helper: stereographic projection  Pi: S^2 -> C
# ---------------------------------------------------------------------------
def stereo_projection(vertex: Vertex) -> np.complex128:
    """Stereographic projection of a sphere vertex to a complex number.

    Pi(g) = (g1 + i*g2) / (1 - g3)

    Numerical guard: if |1 - g3| < _EPS_PROJ (near north pole) the denominator
    is clamped to _EPS_PROJ to avoid division by zero.
    """
    denom = 1.0 - vertex.pos[2]
    if abs(denom) < _EPS_PROJ:
        denom = np.sign(denom) * _EPS_PROJ if denom != 0.0 else _EPS_PROJ
    return np.complex128((vertex.pos[0] + 1j * vertex.pos[1]) / denom)


# ---------------------------------------------------------------------------
# Helper: inverse stereographic projection  Pi^{-1}: C -> S^2  (vectorised)
# ---------------------------------------------------------------------------
def _inverse_stereo_projection(h: np.ndarray) -> np.ndarray:
    """Vectorised inverse stereographic projection.

    Pi^{-1}(z) = ( 2 Re(z)/(|z|^2+1),  2 Im(z)/(|z|^2+1),  (|z|^2-1)/(|z|^2+1) )

    The denominator |z|^2+1 >= 1 so no division-by-zero can occur.

    Parameters
    ----------
    h : complex ndarray of shape (N,)

    Returns
    -------
    ndarray of shape (N, 3) with each row on the unit sphere.
    """
    r2    = np.abs(h) ** 2
    denom = r2 + 1.0
    return np.column_stack([
        2.0 * np.real(h) / denom,
        2.0 * np.imag(h) / denom,
        (r2 - 1.0)        / denom,
    ])


# ---------------------------------------------------------------------------
# Helper: Dirichlet energy
# ---------------------------------------------------------------------------
def _dirichlet_energy(Ld: np.ndarray, h: np.ndarray) -> float:
    """Dirichlet energy of the map encoded by h.

    E_D(f) = sum_k  g_k^T L_D g_k,   g = Pi^{-1}(h) in R^{N x 3}

    Equivalently: (1/2) sum_{edges (i,j)} w_ij ||f_i - f_j||^2

    Parameters
    ----------
    Ld : cotangent Laplacian (N, N) real ndarray
    h  : complex map  (N,) complex ndarray

    Returns
    -------
    float
    """
    g   = _inverse_stereo_projection(h)   # (N, 3)
    Ldg = Ld @ g                          # (N, 3)
    return float(np.einsum('ij,ij->', g, Ldg))


# ---------------------------------------------------------------------------
# Helper: mesh validity assertions
# ---------------------------------------------------------------------------
def _assert_mesh_valid(mesh: MeshSurf) -> None:
    """Assert that *mesh* is a closed, genus-0, non-degenerate triangulation.

    Checks performed
    ----------------
    1. Every edge has exactly 2 adjacent faces  (watertight / closed surface).
    2. Euler characteristic V - E + F == 2      (genus-0 topology).
    3. No face has zero area                    (no degenerate triangles).
    """
    # --- (1) every edge must bound exactly 2 faces ---------------------------
    for e_id in mesh.edges:
        faces = mesh.get_edge_faces(e_id)
        n = len(faces) if faces is not None else 0
        assert n == 2, (
            f"Edge {e_id} bounds {n} face(s); "
            "mesh must be a closed (watertight) genus-0 surface."
        )

    # --- (2) Euler characteristic = 2 for genus-0 ----------------------------
    V, E, F = len(mesh.vertices), len(mesh.edges), len(mesh.faces)
    chi = V - E + F
    assert chi == 2, (
        f"Euler characteristic V-E+F = {V}-{E}+{F} = {chi}; "
        "expected 2 for a closed genus-0 surface."
    )

    # --- (3) no degenerate triangles -----------------------------------------
    for f_id, face in mesh.faces.items():
        a = face.area()
        assert a > 0.0, f"Degenerate (zero-area) triangle detected: face {f_id}."


# ---------------------------------------------------------------------------
# Algorithm 4.1 - initial spherical conformal parameterisation
# ---------------------------------------------------------------------------
def dirichlet_parametrization(mesh: MeshSurf) -> StretchFunction:
    """Algorithm 4.1: initial spherical conformal parameterisation.

    Follows eq. (4.6) of the paper exactly.  The returned StretchFunction
    stores the complex map h so that calling it on a Vertex applies the
    inverse stereographic projection Pi^{-1} and returns the sphere point.

    Assertions / diagnostics are embedded after each numbered step.
    """
    # ----- Mesh validity -----------------------------------------------------
    _assert_mesh_valid(mesh)

    # ----- Step 1: most-regular triangle [va, vb, vc] ------------------------
    face_reg = mesh.get_most_regular_face()
    a, b, c  = face_reg.u, face_reg.v, face_reg.w

    # ----- Step 2: B = {a, b, c},  I = {0,...,N-1} \ B ----------------------
    B = [a.id, b.id, c.id]
    N = len(mesh.vertices)
    I = [i for i in range(N) if i not in set(B)]

    # ----- Step 3: alpha = (vc-va)^T (vb-va) / ||vb-va||^2 ------------------
    vec_ba = Vector(b, a)   # vb - va
    vec_ca = Vector(c, a)   # vc - va
    alpha  = vec_ca.dot(vec_ba) / (vec_ba.norm() ** 2)

    # ----- Step 4: h_B per eq. (4.6) ----------------------------------------
    #
    # Real part  : [-1/||vb-va||^2,   1/||vb-va||^2,  0]
    # Imaginary  : [(1-alpha)/||vc-foot||^2, alpha/||vc-foot||^2, -1/||vc-foot||^2]
    # where foot = va + alpha (vb - va)
    #
    # BUG 1 (fixed): old code  h_b_real = -1 / norm  (not squared)
    #   -> [-h_b_real, h_b_real, 0] = [+1/norm, -1/norm, 0]  (signs AND power wrong)
    #   Fix: inv_sq_edge = 1/norm**2  so [-inv_sq_edge, inv_sq_edge, 0] is correct.
    #
    # BUG 2 (fixed): old code  h_b_img = 1 / norm  (not squared)
    #   Fix: inv_sq_foot = 1/norm**2
    #
    inv_sq_edge = 1.0 / (vec_ba.norm() ** 2)

    foot_pos    = a.pos + alpha * (b.pos - a.pos)   # va + alpha (vb - va)
    foot_vertex = Vertex(foot_pos, _id=-1)
    vec_cfoot   = Vector(c, foot_vertex)             # vc - foot
    inv_sq_foot = 1.0 / (vec_cfoot.norm() ** 2)

    h_B = (np.array([-inv_sq_edge,
                      inv_sq_edge,
                      0.0],          dtype=np.complex128)
           + 1j * np.array([(1.0 - alpha) * inv_sq_foot,
                              alpha        * inv_sq_foot,
                             -inv_sq_foot]))

    # [A4.1-1] h_B shape/dtype
    assert h_B.shape == (3,) and h_B.dtype == np.complex128, (
        f"h_B shape/dtype mismatch: shape={h_B.shape}, dtype={h_B.dtype}"
    )

    # ----- Step 5: cotangent Laplacian L_D -----------------------------------
    Ld = mesh.get_laplacian_matrix(weight='cotangent').toarray()

    # [A4.1-2] symmetry
    sym_err = float(np.max(np.abs(Ld - Ld.T)))
    assert sym_err < 1e-10, (
        f"L_D is not symmetric; max|L_D - L_D^T| = {sym_err:.3e}"
    )
    # [A4.1-3] row sums
    row_sum_err = float(np.max(np.abs(Ld.sum(axis=1))))
    assert row_sum_err < 1e-10, (
        f"L_D row sums are not zero; max = {row_sum_err:.3e}"
    )

    # ----- Step 6: solve [L_D]_{I,I} h_I = -[L_D]_{I,B} h_B ----------------
    A_coeff = Ld[np.ix_(I, I)]
    rhs     = -Ld[np.ix_(I, B)] @ h_B

    # [A4.1-4] dimensions
    assert A_coeff.shape == (len(I), len(I)), (
        f"A_coeff shape wrong: {A_coeff.shape}; expected ({len(I)}, {len(I)})"
    )
    assert rhs.shape == (len(I),), (
        f"rhs shape wrong: {rhs.shape}; expected ({len(I)},)"
    )

    h_I = np.linalg.solve(A_coeff, rhs)

    # [A4.1-5] no NaN/Inf
    assert np.all(np.isfinite(h_I)), "NaN/Inf detected in h_I after the linear solve"

    # ----- Step 7: assemble h; Pi^{-1} applied on-demand by StretchFunction -
    h    = np.zeros(N, dtype=np.complex128)
    h[B] = h_B
    h[I] = h_I

    # [A4.1-6] sphere norms
    sphere_pts   = _inverse_stereo_projection(h)
    sphere_norms = np.linalg.norm(sphere_pts, axis=1)
    assert np.allclose(sphere_norms, 1.0, atol=1e-10), (
        f"Sphere norms not ~1 after Algorithm 4.1: "
        f"min={sphere_norms.min():.8f}, max={sphere_norms.max():.8f}"
    )

    # ----- Step 8: center and rescale for balanced pole coverage -------------
    #
    # Without centering the stereographic image can sit far off-origin so that
    # nearly all vertices land on one hemisphere after Pi^{-1}, giving a very
    # uneven initial map that the CEM loop has to correct.
    #
    # After centering we rescale so that the average edge-length of the
    # boundary ("north") face in the complex plane equals the geometric mean
    # of the north and south face edge-lengths.  This mirrors the FLASH
    # north/south balancing step and dramatically improves the starting point.
    #
    h = h - np.mean(h)

    # Average edge length of the boundary face (north face) in the h-plane.
    # B always has exactly 3 elements (set at Step 2 from the 3 boundary vertices).
    h_a, h_b, h_c = h[B[0]], h[B[1]], h[B[2]]
    NorthTriSide  = (abs(h_a - h_b) + abs(h_b - h_c) + abs(h_c - h_a)) / 3.0

    if NorthTriSide > 0:
        # South-pole stereographic projection: w = (g1 + i*g2) / (1 + g3)
        # Points near the south pole map to small |w|, so the innermost
        # triangle in the w-plane is the antipode of the north face.
        sphere_pts_h = _inverse_stereo_projection(h)          # (N, 3)
        denom_s = np.maximum(1.0 + sphere_pts_h[:, 2], _EPS_PROJ)
        w = (sphere_pts_h[:, 0] + 1j * sphere_pts_h[:, 1]) / denom_s

        # Find the face with the smallest total |h| (i.e., the face whose
        # stereographic pre-image is closest to the south pole).
        faces_arr = mesh.get_faces_collection()   # (F, 3) int array
        abs_h_sum = (np.abs(h[faces_arr[:, 0]])
                     + np.abs(h[faces_arr[:, 1]])
                     + np.abs(h[faces_arr[:, 2]]))
        order  = np.argsort(abs_h_sum)
        B_set  = set(B)
        inner_face = None
        for idx in order:
            candidate = faces_arr[idx]
            # Skip the boundary face itself (all 3 vertices are in B)
            if set(candidate) != B_set:
                inner_face = candidate
                break

        if inner_face is not None:
            w0, w1, w2 = w[inner_face[0]], w[inner_face[1]], w[inner_face[2]]
            SouthTriSide = (abs(w0 - w1) + abs(w1 - w2) + abs(w2 - w0)) / 3.0
            if SouthTriSide > 0:
                h = h * np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide

    # [A4.1-7] diagnostics
    sphere_pts_final = _inverse_stereo_projection(h)
    z3 = sphere_pts_final[:, 2]
    print(f"[A4.1] After rescaling: g3 in [{z3.min():.4f}, {z3.max():.4f}] "
          f"(balanced hemispheres → close to [-1, 1])")

    return StretchFunction(mesh, h)


# ---------------------------------------------------------------------------
# Algorithm 4.2 - CEM (conformally-exact-map) iteration
# ---------------------------------------------------------------------------
def stretch_parametrization(mesh: MeshSurf,
                            eps: float = 1e-6,
                            max_iters: int = 1000,
                            verbose: bool = True) -> StretchFunction:
    """Algorithm 4.2: CEM iteration to minimise the Dirichlet energy on S^2.

    Starts from the Algorithm 4.1 result and iterates until the improvement
    in Dirichlet energy drops below *eps*.

    Parameters
    ----------
    mesh      : closed genus-0 triangular surface mesh
    eps       : convergence threshold  delta = E_D(g) - E_D(f) <= eps
    max_iters : maximum number of CEM iterations
    verbose   : print per-iteration diagnostics

    Returns
    -------
    StretchFunction  - the improved conformal map (h stored in C)
    """
    # ----- Run Algorithm 4.1 -------------------------------------------------
    dirichlet_stretch = dirichlet_parametrization(mesh)
    h = dirichlet_stretch.h.copy()   # complex map (stereo projection of sphere)

    # ----- Cotangent Laplacian L_D (fixed throughout Algorithm 4.2) ----------
    # BUG 4 (fixed): old code recomputed Ls (stretch Laplacian) every iteration.
    # Algorithm 4.2 uses L_D (cotangent Laplacian) in all iterations.
    Ld = mesh.get_laplacian_matrix(weight='cotangent').toarray()

    N   = len(h)
    E_g = _dirichlet_energy(Ld, h)

    if verbose:
        print(f"[A4.2] iter 0 (init from Algo 4.1): E_D = {E_g:.6e}")

    # ----- Step 2: h is already the stereo projection of the Algo 4.1 sphere -
    # (StretchFunction stores h in C directly)

    # ----- Step 3: iterate ---------------------------------------------------
    for count in range(1, max_iters + 1):

        # Step 3a: h_i <- h_i / |h_i|^2  (Mobius inversion)
        # BUG 6 (fixed): no guard for |h_i|=0 -> division by zero.
        abs_h_sq = np.abs(h) ** 2
        abs_h_sq = np.where(abs_h_sq < _EPS_INV, _EPS_INV, abs_h_sq)  # guard
        h = h / abs_h_sq

        # Step 3b: I = {i : |h_i| < 1},  B = complement
        # The Möbius inversion h -> h/|h|^2 maps the open unit disk to its
        # exterior and vice versa.  Therefore the natural partition boundary is
        # the unit circle (r = 1), not the empirical median, which drifts
        # arbitrarily and produces an inconsistent Dirichlet sub-problem.
        abs_h = np.abs(h)
        r     = 1.0
        I     = np.where(abs_h <  r)[0].tolist()
        B     = np.where(abs_h >= r)[0].tolist()

        if verbose:
            print(f"[A4.2] iter {count}: |I|={len(I)}, |B|={len(B)}, "
                  f"r=1.0, |h| in [{abs_h.min():.4e}, {abs_h.max():.4e}]")

        # [A4.2-1] partition sanity
        assert len(I) + len(B) == N, (
            f"Partition mismatch at iter {count}: |I|+|B|={len(I)+len(B)} != N={N}"
        )

        if len(B) == 0:
            if verbose:
                print("[A4.2] B is empty - all vertices interior. Stopping.")
            break

        # Step 3c: [L_D]_{I,I} h_I = -[L_D]_{I,B} h_B
        A_coeff = Ld[np.ix_(I, I)]
        h_b     = h[B]
        b_coeff = -Ld[np.ix_(I, B)] @ h_b
        h_I     = np.linalg.solve(A_coeff, b_coeff)

        # [A4.2-4] NaN/Inf check
        assert np.all(np.isfinite(h_I)), (
            f"NaN/Inf in h_I at iteration {count}"
        )

        h[I] = h_I

        # Step 3d: map back to sphere via Pi^{-1} (done by _inverse_stereo_projection)
        sphere_pts   = _inverse_stereo_projection(h)
        sphere_norms = np.linalg.norm(sphere_pts, axis=1)

        # [A4.2-3] sphere norms
        assert np.allclose(sphere_norms, 1.0, atol=1e-9), (
            f"Sphere norms not ~1 at iter {count}: "
            f"min={sphere_norms.min():.6f}, max={sphere_norms.max():.6f}"
        )

        # Step 3e: delta = E_D(g) - E_D(f)
        E_f   = _dirichlet_energy(Ld, h)
        delta = E_g - E_f

        if verbose:
            print(f"[A4.2] iter {count}: "
                  f"E_D(g)={E_g:.6e}, E_D(f)={E_f:.6e}, delta={delta:.6e}")

        # Step 3f: g <- f; stop only on small *positive* improvement
        # Stopping when delta < 0 (energy increased) would exit on divergence.
        # We only converge when the improvement 0 <= delta <= eps.
        E_g = E_f

        if 0 <= delta <= eps:
            if verbose:
                print(f"[A4.2] Converged at iteration {count}: "
                      f"delta={delta:.3e} <= eps={eps:.3e}")
            break

    dirichlet_stretch.h = h
    return dirichlet_stretch