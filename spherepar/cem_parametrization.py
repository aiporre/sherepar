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
# 4.2 Step 3b       | r is an algorithm input      | r was hardcoded to 1.0
#                   | (paper experiments: 1.2)     |   Fix: expose radius, default 1.2
# ------------------|------------------------------|----------------------------
# 4.2 Step 3c       | [L_D]_{I,I} h_I = ...       | Used Ls (stretch Laplacian)
#                   |                              |   Fix: use Ld (cotangent)
# ------------------|------------------------------|----------------------------
# 4.2 Step 3e       | delta = E_D(g) - E_D(f)     | Not computed at all
#                   |                              |   Fix: added _dirichlet_energy
# ------------------|------------------------------|----------------------------
# 4.2 Step 3f       | continue while delta > eps   | accepted energy-increasing steps
#                   |                              |   Fix: rollback when delta < 0
# ------------------|------------------------------|----------------------------
# 4.2 Step 3a       | h_i <- h_i / |h_i|^2        | No guard for |h_i|=0
#                   |                              |   Fix: clamp |h|^2 >= _EPS_INV
# ------------------|------------------------------|----------------------------
# Stereo proj       | g1/(1-g3) + i*g2/(1-g3)     | Blows up when g3 ~= 1
#                   |                              |   Fix: clamp denom >= _EPS_PROJ
# =============================================================================
"""

import copy
from typing import Any, Callable, Optional, Sequence
import warnings

import numpy as np

from spherepar.mesh import MeshSurf, StretchFunction, Vector, Vertex
from spherepar.cem_anchor_diagnostics import (
    collect_anchor_geometry,
    compute_anchor_collapse_diagnostics,
)
from spherepar.parametrization_validation import validate_sphere_parameterization

# ---------------------------------------------------------------------------
# Numerical safeguards
# ---------------------------------------------------------------------------
_EPS_PROJ = 1e-12   # minimum |1 - z| in stereographic projection (north-pole guard)
_EPS_INV  = 1e-14   # minimum |h|^2 in Mobius inversion step (zero-division guard)
_ANCHOR_STRATEGIES = ("regular", "central_regular")


def _validate_anchor_options(
    anchor_strategy: str,
    anchor_regularity_percentile: float,
) -> None:
    if anchor_strategy not in _ANCHOR_STRATEGIES:
        raise ValueError(
            "anchor_strategy must be one of " + ", ".join(repr(value) for value in _ANCHOR_STRATEGIES)
        )
    if (
        not np.isfinite(anchor_regularity_percentile)
        or not 0.0 <= anchor_regularity_percentile <= 100.0
    ):
        raise ValueError("anchor_regularity_percentile must be finite and in [0, 100]")


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

    E_D(f) = 1/2 trace(g^T L_D g),   g = Pi^{-1}(h) in R^{N x 3}

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
    return 0.5 * float(np.einsum('ij,ij->', g, Ldg))


def _cotangent_weight_diagnostics(
    mesh: MeshSurf,
    laplacian: np.ndarray,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Report the intrinsic-Delaunay cotangent condition on mesh edges."""
    edge_ids = np.asarray(mesh.get_edges_collection(), dtype=np.int64)
    weights = np.asarray([
        -0.5 * (laplacian[i, j] + laplacian[j, i])
        for i, j in edge_ids
    ], dtype=np.float64)
    negative = weights < -tolerance
    negative_edge_ids = edge_ids[negative]
    negative_edge_set = {tuple(edge) for edge in negative_edge_ids.tolist()}
    affected_face_ids = []
    for face_id, (a, b, c) in enumerate(mesh.get_faces_collection()):
        face_edges = (
            tuple(sorted((int(a), int(b)))),
            tuple(sorted((int(b), int(c)))),
            tuple(sorted((int(c), int(a)))),
        )
        if any(edge in negative_edge_set for edge in face_edges):
            affected_face_ids.append(face_id)
    return {
        "tolerance": float(tolerance),
        "edge_count": int(len(edge_ids)),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "negative_weight_count": int(np.count_nonzero(negative)),
        "negative_edge_ids": negative_edge_ids.tolist(),
        "affected_triangle_count": int(len(affected_face_ids)),
        "affected_triangle_ids": affected_face_ids,
        "is_intrinsic_delaunay": not bool(np.any(negative)),
    }


def _face_angle_diagnostics(
    mesh: MeshSurf,
    target_face_count: int = 0,
) -> dict[str, Any]:
    """Measure minimum angles and identify low-quality input triangles."""
    vertices = np.asarray(mesh.get_vertices_collection(), dtype=np.float64)
    faces = np.asarray(mesh.get_faces_collection(), dtype=np.int64)
    triangles = vertices[faces]

    def corner_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        if np.any(denominator <= np.finfo(np.float64).eps):
            raise ValueError("input mesh contains a zero-length triangle edge")
        cosine = np.einsum("ij,ij->i", first, second) / denominator
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    angles = np.column_stack((
        corner_angle(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        corner_angle(triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 1]),
        corner_angle(triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 2]),
    ))
    angles_deg = np.rad2deg(angles)
    minimum_angles_deg = angles_deg.min(axis=1)
    maximum_angles_deg = angles_deg.max(axis=1)
    target_percentage = 100.0 * target_face_count / len(faces) if len(faces) else 0.0
    threshold_comparison = []
    for threshold in range(5, 61, 5):
        count = int(np.count_nonzero(minimum_angles_deg < threshold))
        percentage = 100.0 * count / len(faces) if len(faces) else 0.0
        threshold_comparison.append({
            "threshold_degrees": threshold,
            "face_count": count,
            "percentage": float(percentage),
            "difference_from_affected_percentage": float(percentage - target_percentage),
        })
        if percentage >= target_percentage:
            break
    closest_threshold = min(
        threshold_comparison,
        key=lambda item: abs(item["difference_from_affected_percentage"]),
    )
    threshold_counts = {
        str(item["threshold_degrees"]): item["face_count"]
        for item in threshold_comparison
    }
    below_five_ids = np.flatnonzero(minimum_angles_deg < 5.0)
    worst_face_ids = np.argsort(minimum_angles_deg)[:min(10, len(faces))]
    return {
        "face_count": int(len(faces)),
        "minimum_angle_degrees": float(minimum_angles_deg.min()),
        "median_minimum_angle_degrees": float(np.median(minimum_angles_deg)),
        "faces_below_angle_degrees": threshold_counts,
        "angle_threshold_comparison": threshold_comparison,
        "negative_weight_affected_triangle_count": int(target_face_count),
        "negative_weight_affected_triangle_percentage": float(target_percentage),
        "closest_threshold_degrees": int(closest_threshold["threshold_degrees"]),
        "closest_threshold_face_count": int(closest_threshold["face_count"]),
        "closest_threshold_percentage": float(closest_threshold["percentage"]),
        "below_5_degree_face_count": int(len(below_five_ids)),
        "below_5_degree_face_ids": below_five_ids.tolist(),
        "obtuse_face_count": int(np.count_nonzero(maximum_angles_deg > 90.0)),
        "worst_faces": [
            {
                "face_id": int(face_id),
                "minimum_angle_degrees": float(minimum_angles_deg[face_id]),
                "angles_degrees": angles_deg[face_id].tolist(),
            }
            for face_id in worst_face_ids
        ],
    }


def _format_cem_input_summary(
    angle_diagnostics: dict[str, Any],
    cotangent_diagnostics: dict[str, Any],
) -> str:
    """Return the compact preflight line used by stdout and dataset logs."""
    return (
        f"faces={angle_diagnostics['face_count']}, "
        f"min_angle={angle_diagnostics['minimum_angle_degrees']:.6f}deg, "
        f"median_min_angle={angle_diagnostics['median_minimum_angle_degrees']:.6f}deg, "
        f"below_5deg={angle_diagnostics['below_5_degree_face_count']}, "
        f"obtuse_faces={angle_diagnostics['obtuse_face_count']}, "
        f"negative_cotangent_edges={cotangent_diagnostics['negative_weight_count']}, "
        f"affected_triangles={cotangent_diagnostics['affected_triangle_count']}"
    )


def _print_cem_input_diagnostics(
    angle_diagnostics: dict[str, Any],
    cotangent_diagnostics: dict[str, Any],
) -> None:
    """Print the input summary and cumulative 5-degree angle comparison."""
    print(
        "[CEM input] "
        + _format_cem_input_summary(angle_diagnostics, cotangent_diagnostics),
        flush=True,
    )
    target_percentage = angle_diagnostics["negative_weight_affected_triangle_percentage"]
    for row in angle_diagnostics["angle_threshold_comparison"]:
        print(
            f"[CEM input] faces below {row['threshold_degrees']:2d}deg: "
            f"{row['face_count']:5d} / {angle_diagnostics['face_count']} "
            f"({row['percentage']:.2f}%); "
            f"negative-weight affected target={target_percentage:.2f}%",
            flush=True,
        )
    print(
        "[CEM input] closest cumulative angle threshold: "
        f"below {angle_diagnostics['closest_threshold_degrees']}deg -> "
        f"{angle_diagnostics['closest_threshold_face_count']} faces "
        f"({angle_diagnostics['closest_threshold_percentage']:.2f}%)",
        flush=True,
    )


def _cem_input_log_message(
    angle_diagnostics: dict[str, Any],
    cotangent_diagnostics: dict[str, Any],
) -> str:
    """Return a one-line form of the complete input preflight table."""
    threshold_text = ", ".join(
        f"below_{row['threshold_degrees']}deg={row['face_count']}({row['percentage']:.2f}%)"
        for row in angle_diagnostics["angle_threshold_comparison"]
    )
    return (
        _format_cem_input_summary(angle_diagnostics, cotangent_diagnostics)
        + f"; angle_thresholds=[{threshold_text}]"
        + "; closest_threshold="
        + f"{angle_diagnostics['closest_threshold_degrees']}deg:"
        + f"{angle_diagnostics['closest_threshold_face_count']}"
        + f"({angle_diagnostics['closest_threshold_percentage']:.2f}%)"
    )


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
def dirichlet_parametrization(
    mesh: MeshSurf,
    anchor_diagnostics: bool = False,
    anchor_strategy: str = "regular",
    anchor_regularity_percentile: float = 10.0,
) -> StretchFunction:
    """Algorithm 4.1: initial spherical conformal parameterisation.

    Follows eq. (4.6) of the paper exactly.  The returned StretchFunction
    stores the complex map h so that calling it on a Vertex applies the
    inverse stereographic projection Pi^{-1} and returns the sphere point.

    Assertions / diagnostics are embedded after each numbered step.
    """
    _validate_anchor_options(anchor_strategy, anchor_regularity_percentile)

    # ----- Mesh validity -----------------------------------------------------
    _assert_mesh_valid(mesh)

    # ----- Step 1: most-regular triangle [va, vb, vc] ------------------------
    face_reg = (
        mesh.get_most_regular_face()
        if anchor_strategy == "regular"
        else mesh.get_central_regular_face(anchor_regularity_percentile)
    )
    a, b, c  = face_reg.u, face_reg.v, face_reg.w
    anchor_meta = (
        collect_anchor_geometry(
            mesh,
            face_reg,
            anchor_strategy=anchor_strategy,
            anchor_regularity_percentile=anchor_regularity_percentile,
        )
        if anchor_diagnostics else None
    )

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

    result = StretchFunction(mesh, h)
    result.anchor_diagnostics = anchor_meta
    return result


# ---------------------------------------------------------------------------
# Algorithm 4.2 - CEM (conformally-exact-map) iteration
# ---------------------------------------------------------------------------
def _prepare_cem(
    mesh: MeshSurf,
    *,
    input_diagnostics_callback: Optional[Callable[[dict[str, Any]], None]],
    anchor_diagnostics: bool,
    anchor_strategy: str,
    anchor_regularity_percentile: float,
) -> tuple[StretchFunction, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Run fixed CEM setup exactly once for one radius search."""
    mesh_faces = mesh.get_faces_collection()
    laplacian = mesh.get_laplacian_matrix(weight="cotangent").toarray()
    cotangent = _cotangent_weight_diagnostics(mesh, laplacian)
    angles = _face_angle_diagnostics(
        mesh, target_face_count=cotangent["affected_triangle_count"]
    )
    _print_cem_input_diagnostics(angles, cotangent)
    if input_diagnostics_callback is not None:
        input_diagnostics_callback({
            "summary": _cem_input_log_message(angles, cotangent),
            "input_mesh_quality": angles,
            "cotangent_weights": cotangent,
        })
    if not cotangent["is_intrinsic_delaunay"]:
        warnings.warn(
            "CEM input has "
            f"{cotangent['negative_weight_count']} negative cotangent edge weight(s), "
            f"affecting {cotangent['affected_triangle_count']} triangle(s); "
            "the intrinsic-Delaunay convex-combination guarantee does not apply.",
            RuntimeWarning,
            stacklevel=3,
        )
    initial = dirichlet_parametrization(
        mesh,
        anchor_diagnostics=anchor_diagnostics,
        anchor_strategy=anchor_strategy,
        anchor_regularity_percentile=anchor_regularity_percentile,
    )
    if initial.h.shape != (len(mesh.vertices),):
        raise AssertionError("Algorithm 4.1 returned the wrong harmonic-map shape")
    return initial, laplacian, angles, cotangent


def _stretch_parametrization_attempt(mesh: MeshSurf,
                            eps: float = 1e-6,
                            max_iters: int = 1000,
                            verbose: bool = True,
                            radius: float = 1.2,
                            input_diagnostics_callback: Optional[
                                Callable[[dict[str, Any]], None]
                            ] = None,
                            anchor_diagnostics: bool = False,
                            anchor_strategy: str = "regular",
                            anchor_regularity_percentile: float = 10.0,
                            _prepared: Optional[
                                tuple[StretchFunction, np.ndarray, dict[str, Any], dict[str, Any]]
                            ] = None,
                            validation_vertices: Optional[np.ndarray] = None,
                            validation_faces: Optional[np.ndarray] = None,
                            ) -> StretchFunction:
    """Algorithm 4.2: CEM iteration to minimise the Dirichlet energy on S^2.

    Starts from the Algorithm 4.1 result and iterates until the improvement
    in Dirichlet energy drops below *eps*.

    Parameters
    ----------
    mesh      : closed genus-0 triangular surface mesh
    eps       : convergence threshold  delta = E_D(g) - E_D(f) <= eps
    max_iters : maximum number of CEM iterations
    verbose   : print per-iteration diagnostics
    radius    : stereographic partition radius from Algorithm 4.2
    input_diagnostics_callback : optional callback invoked before Algorithm 4.1
    anchor_diagnostics : collect anchor-hop/collapse diagnostics when true
    anchor_strategy : deterministic Algorithm 4.1 anchor selector
    anchor_regularity_percentile : candidate percentile for ``central_regular``

    Returns
    -------
    StretchFunction  - the improved conformal map (h stored in C)
    """
    _validate_anchor_options(anchor_strategy, anchor_regularity_percentile)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be finite and non-negative")
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")

    mesh_vertices = (
        mesh.get_vertices_collection()
        if validation_vertices is None else np.asarray(validation_vertices, dtype=np.float64)
    )
    mesh_faces = (
        mesh.get_faces_collection()
        if validation_faces is None else np.asarray(validation_faces, dtype=np.int32)
    )
    if _prepared is None:
        _prepared = _prepare_cem(
            mesh,
            input_diagnostics_callback=input_diagnostics_callback,
            anchor_diagnostics=anchor_diagnostics,
            anchor_strategy=anchor_strategy,
            anchor_regularity_percentile=anchor_regularity_percentile,
        )
    initial_stretch, Ld, angle_diagnostics, cotangent_diagnostics = _prepared
    dirichlet_stretch = StretchFunction(mesh, initial_stretch.h.copy())
    dirichlet_stretch.anchor_diagnostics = copy.deepcopy(initial_stretch.anchor_diagnostics)
    h = dirichlet_stretch.h.copy()   # complex map (stereo projection of sphere)

    N   = len(h)
    E_g = _dirichlet_energy(Ld, h)
    initial_energy = E_g
    first_iteration_validation = None
    first_iteration_partition = None
    attempted_iterations = 0
    accepted_iterations = 0
    last_delta = None
    stop_reason = "max_iters"

    if verbose:
        print(f"[A4.2] iter 0 (init from Algo 4.1): E_D = {E_g:.6e}")

    # ----- Step 2: h is already the stereo projection of the Algo 4.1 sphere -
    # (StretchFunction stores h in C directly)

    # ----- Step 3: iterate ---------------------------------------------------
    for count in range(1, max_iters + 1):
        attempted_iterations = count
        h_previous = h.copy()
        E_previous = E_g

        # Step 3a: h_i <- h_i / |h_i|^2  (Mobius inversion)
        # BUG 6 (fixed): no guard for |h_i|=0 -> division by zero.
        abs_h_sq = np.abs(h_previous) ** 2
        abs_h_sq = np.where(abs_h_sq < _EPS_INV, _EPS_INV, abs_h_sq)  # guard
        h_candidate = h_previous / abs_h_sq

        # Step 3b: I = {i : |h_i| < radius}, B = complement.
        abs_h = np.abs(h_candidate)
        I = np.where(abs_h < radius)[0].tolist()
        B = np.where(abs_h >= radius)[0].tolist()
        if count == 1:
            first_iteration_partition = {
                "interior_count": int(len(I)),
                "boundary_count": int(len(B)),
            }

        if verbose:
            print(f"[A4.2] iter {count}: |I|={len(I)}, |B|={len(B)}, "
                  f"r={radius:g}, |h| in [{abs_h.min():.4e}, {abs_h.max():.4e}]")

        # [A4.2-1] partition sanity
        assert len(I) + len(B) == N, (
            f"Partition mismatch at iter {count}: |I|+|B|={len(I)+len(B)} != N={N}"
        )

        if len(B) == 0:
            stop_reason = "empty_boundary_rollback"
            if verbose:
                print("[A4.2] B is empty - retaining the previous iterate.")
            break

        # Step 3c: [L_D]_{I,I} h_I = -[L_D]_{I,B} h_B
        A_coeff = Ld[np.ix_(I, I)]
        h_b     = h_candidate[B]
        b_coeff = -Ld[np.ix_(I, B)] @ h_b
        h_I     = np.linalg.solve(A_coeff, b_coeff)

        # [A4.2-4] NaN/Inf check
        assert np.all(np.isfinite(h_I)), (
            f"NaN/Inf in h_I at iteration {count}"
        )

        h_candidate[I] = h_I

        # Step 3d: map back to sphere via Pi^{-1} (done by _inverse_stereo_projection)
        sphere_pts   = _inverse_stereo_projection(h_candidate)
        sphere_norms = np.linalg.norm(sphere_pts, axis=1)

        # [A4.2-3] sphere norms
        assert np.allclose(sphere_norms, 1.0, atol=1e-9), (
            f"Sphere norms not ~1 at iter {count}: "
            f"min={sphere_norms.min():.6f}, max={sphere_norms.max():.6f}"
        )

        if count == 1:
            first_iteration_validation = validate_sphere_parameterization(
                mesh_vertices, mesh_faces, sphere_pts, mesh_faces
            )
            if not first_iteration_validation["is_valid"]:
                warnings.warn(
                    "CEM first iteration produced invalid spherical geometry: "
                    + "; ".join(first_iteration_validation["errors"]),
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Step 3e: delta = E_D(g) - E_D(f)
        E_f = _dirichlet_energy(Ld, h_candidate)
        delta = E_previous - E_f
        last_delta = float(delta)

        if verbose:
            print(f"[A4.2] iter {count}: "
                  f"E_D(g)={E_previous:.6e}, E_D(f)={E_f:.6e}, delta={delta:.6e}")

        # Reject an energy-increasing candidate instead of returning a worse map.
        if delta < 0.0:
            stop_reason = "energy_increase_rollback"
            if verbose:
                print(f"[A4.2] Energy increased at iteration {count}; "
                      "retaining the previous iterate.")
            break

        h = h_candidate
        E_g = E_f
        accepted_iterations += 1

        if delta <= eps:
            stop_reason = "converged"
            if verbose:
                print(f"[A4.2] Converged at iteration {count}: "
                      f"delta={delta:.3e} <= eps={eps:.3e}")
            break

    final_sphere_pts = _inverse_stereo_projection(h)
    final_validation = validate_sphere_parameterization(
        mesh_vertices, mesh_faces, final_sphere_pts, mesh_faces
    )
    if not final_validation["is_valid"]:
        warnings.warn(
            "CEM final iterate has invalid spherical geometry: "
            + "; ".join(final_validation["errors"]),
            RuntimeWarning,
            stacklevel=2,
        )

    dirichlet_stretch.h = h
    cem_diagnostics: dict[str, Any] = {
        "radius": float(radius),
        "anchor_strategy": anchor_strategy,
        "anchor_regularity_percentile": float(anchor_regularity_percentile),
        "input_mesh_quality": angle_diagnostics,
        "cotangent_weights": cotangent_diagnostics,
        "first_iteration_partition": first_iteration_partition,
        "first_iteration_validation": first_iteration_validation,
        "final_validation": final_validation,
        "convergence": {
            "attempted_iterations": int(attempted_iterations),
            "accepted_iterations": int(accepted_iterations),
            "stop_reason": stop_reason,
            "initial_energy": float(initial_energy),
            "final_energy": float(E_g),
            "last_delta": last_delta,
            "eps": float(eps),
            "max_iters": int(max_iters),
        },
    }
    if anchor_diagnostics:
        anchor_meta = dirichlet_stretch.anchor_diagnostics
        if anchor_meta is None:
            raise RuntimeError("Algorithm 4.1 did not capture requested anchor diagnostics")
        anchor_meta.update(
            compute_anchor_collapse_diagnostics(
                mesh_vertices,
                mesh_faces,
                final_sphere_pts,
                anchor_meta["minimum_hop_distances"],
                sphere_stage="pre_mobius_cem",
            )
        )
        cem_diagnostics["anchor"] = anchor_meta
    dirichlet_stretch.cem_diagnostics = cem_diagnostics
    return dirichlet_stretch


def _ordered_radii(base: float, candidates: Sequence[float], maximum: int) -> list[float]:
    if maximum < 1:
        raise ValueError("cem_max_attempts must be at least 1")
    ordered: list[float] = []
    for value in (base, *candidates):
        radius = float(value)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("CEM radius candidates must be finite and positive")
        if not any(np.isclose(radius, old, rtol=0.0, atol=1e-12) for old in ordered):
            ordered.append(radius)
    return ordered[:min(maximum, 5)]


def stretch_parametrization(
    mesh: MeshSurf,
    eps: float = 1e-6,
    max_iters: int = 1000,
    verbose: bool = True,
    radius: float = 1.2,
    input_diagnostics_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    anchor_diagnostics: bool = False,
    anchor_strategy: str = "regular",
    anchor_regularity_percentile: float = 10.0,
    adaptive_radius: bool = False,
    radius_candidates: Sequence[float] = (1.1, 1.3, 1.4, 1.5),
    reject_retry: bool = False,
    max_attempts: int = 5,
    max_collapsed_faces: int = 0,
    validation_vertices: Optional[np.ndarray] = None,
    validation_faces: Optional[np.ndarray] = None,
) -> StretchFunction:
    """Run CEM, optionally searching radii from one shared harmonic map."""
    _validate_anchor_options(anchor_strategy, anchor_regularity_percentile)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if max_collapsed_faces < 0:
        raise ValueError("cem_max_collapsed_faces must be non-negative")
    radii = _ordered_radii(radius, radius_candidates, max_attempts)
    prepared = _prepare_cem(
        mesh,
        input_diagnostics_callback=input_diagnostics_callback,
        anchor_diagnostics=anchor_diagnostics,
        anchor_strategy=anchor_strategy,
        anchor_regularity_percentile=anchor_regularity_percentile,
    )
    attempts: list[tuple[int, StretchFunction, int]] = []
    records: list[dict[str, Any]] = []
    search_enabled = bool(adaptive_radius or reject_retry)
    for order, attempt_radius in enumerate(radii):
        if order > 0:
            if not search_enabled:
                break
            if attempts and attempts[0][2] <= max_collapsed_faces:
                break
        try:
            result = _stretch_parametrization_attempt(
                mesh,
                eps=eps,
                max_iters=max_iters,
                verbose=verbose,
                radius=attempt_radius,
                anchor_diagnostics=anchor_diagnostics,
                anchor_strategy=anchor_strategy,
                anchor_regularity_percentile=anchor_regularity_percentile,
                _prepared=prepared,
                validation_vertices=validation_vertices,
                validation_faces=validation_faces,
            )
            validation = result.cem_diagnostics["final_validation"]
            collapsed = int(validation.get("degenerate_face_count", 0))
            convergence = result.cem_diagnostics["convergence"]
            attempts.append((order, result, collapsed))
            record = {
                "attempt_order": int(order),
                "radius": float(attempt_radius),
                "numerical_success": True,
                "collapsed_face_count": collapsed,
                "validation_valid": bool(validation.get("is_valid", False)),
                "convergence": copy.deepcopy(convergence),
                "selected": False,
                "error": None,
            }
            records.append(record)
            print(
                f"[CEM radius] radius={attempt_radius:g} collapsed={collapsed} "
                f"convergence={convergence.get('stop_reason')} selected=pending"
            )
            if collapsed == 0:
                break
        except Exception as exc:  # numerical failures do not abort later radii
            records.append({
                "attempt_order": int(order),
                "radius": float(attempt_radius),
                "numerical_success": False,
                "collapsed_face_count": None,
                "validation_valid": False,
                "convergence": None,
                "selected": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[CEM radius] radius={attempt_radius:g} numerical_failure={type(exc).__name__}: {exc}")
            if not search_enabled:
                raise
    if not attempts:
        errors = "; ".join(record["error"] or "unknown failure" for record in records)
        raise RuntimeError(f"all CEM radius attempts failed numerically: {errors}")
    selected_order, selected, selected_collapsed = min(
        attempts,
        key=lambda item: (item[2], round(abs(radii[item[0]] - radius), 12), item[0]),
    )
    for record in records:
        record["selected"] = record["attempt_order"] == selected_order
        record["selection_status"] = "selected" if record["selected"] else "not_selected"
        record["threshold"] = int(max_collapsed_faces)
    accepted = selected_collapsed <= max_collapsed_faces
    reason = (
        f"collapsed face count {selected_collapsed} is within threshold {max_collapsed_faces}"
        if accepted else
        f"collapsed face count {selected_collapsed} exceeds threshold {max_collapsed_faces}"
    )
    for record in records:
        record["selection_reason"] = reason if record["selected"] else "higher collapsed-face count or tie-break rank"
    selected.cem_diagnostics["radius_attempts"] = records
    selected.cem_diagnostics["acceptance"] = {
        "policy_enabled": bool(reject_retry),
        "accepted": bool(accepted if reject_retry else True),
        "geometric_threshold_met": bool(accepted),
        "max_collapsed_faces": int(max_collapsed_faces),
        "selected_collapsed_face_count": int(selected_collapsed),
        "reason": reason,
    }
    selected.cem_diagnostics["selected_radius"] = float(radii[selected_order])
    print(
        f"[CEM radius] radius={radii[selected_order]:g} collapsed={selected_collapsed} "
        f"convergence={selected.cem_diagnostics['convergence'].get('stop_reason')} "
        f"selected=true threshold={max_collapsed_faces} reason={reason}"
    )
    if reject_retry and not accepted:
        print(
            f"[CEM reject] radius={radii[selected_order]:g} collapsed={selected_collapsed} "
            f"threshold={max_collapsed_faces} reason={reason}"
        )
    return selected
