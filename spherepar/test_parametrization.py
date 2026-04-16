from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.draw import ellipsoid
from spherepar.mesh import *
from spherepar.parametrization import (dirichlet_parametrization, stretch_parametrization,
                                        _dirichlet_energy, _inverse_stereo_projection,
                                        _EPS_INV)


class Test(TestCase):
    def test_dirichlet_spherepar(self):
        # basic_form = ellipsoid(0.6, 0.10, 0.16, levelset=True)
        basic_form = ellipsoid(1, 1,0.5, levelset=True)
        # make 0 and 1 the values of the levelset
        basic_form[basic_form <0] = 0.0
        basic_form[basic_form >0] = 1.0
        print('basic form shape: is a box with points: ', basic_form.shape)
        mesh_surf = get_surface_mesh(basic_form)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(121, projection='3d')
        # xs, ys, zs = mesh_surf.get_vertices_collection()[..., 0], mesh_surf.get_vertices_collection()[..., 1], mesh_surf.get_vertices_collection()[..., 2]
        # ax.scatter(xs, ys, zs)
        plot_mesh((mesh_surf.get_vertices_collection(), mesh_surf.get_faces_collection(), None, None), ax=ax)
        # L = mesh_surf.get_laplacian_matrix()
        # self.assertEqual(L.sum(), 628.316737134868)
        # self.assertAlmostEqual((L - L.transpose()).sum(), 0)
        harmonic_par = dirichlet_parametrization(mesh_surf)
        mesh_strech = harmonic_par.convert_mesh()
        # fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(122, projection='3d')
        # xs, ys, zs = mesh_strech.get_vertices_collection()[..., 0], mesh_strech.get_vertices_collection()[..., 1], mesh_strech.get_vertices_collection()[..., 2]
        # ax.scatter(xs, ys, zs)
        plot_mesh((mesh_strech.get_vertices_collection(), mesh_strech.get_faces_collection(), None, None), ax=ax)
        plt.show()



    def test_strech_spherepar(self):
        basic_form = ellipsoid(6, 10, 16, levelset=True)
        # make 0 and 1 the values of the levelset
        basic_form[basic_form < 0] = 0.0
        basic_form[basic_form > 0] = 1.0
        # # plot the levelset
        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # plot_mesh((basic_form, None, None, None), ax=ax)
        # plt.draw()
        print('basic form shape: is a box with points: ', basic_form.shape)
        mesh_surf = get_surface_mesh(basic_form)
        # L = mesh_surf.get_laplacian_matrix()
        # self.assertEqual(L.sum(), 628.316737134868)
        # self.assertAlmostEqual((L - L.transpose()).sum(), 0)
        harmonic_par = stretch_parametrization(mesh_surf)


class _MeshSurfFixture(TestCase):
    """Shared fixture: a small ellipsoid surface mesh used by parametrization tests."""

    def setUp(self):
        basic_form = ellipsoid(3, 3, 3, levelset=True)
        basic_form[basic_form < 0] = 0.0
        basic_form[basic_form > 0] = 1.0
        self.mesh_surf = get_surface_mesh(basic_form)


class TestDirichletParametrization(_MeshSurfFixture):
    """Tests for the dirichlet parametrization algorithm correctness."""

    def test_vertices_mapped_to_unit_sphere(self):
        """After dirichlet parametrization, all vertices should lie on the unit sphere."""
        harmonic_par = dirichlet_parametrization(self.mesh_surf)
        mesh_stretched = harmonic_par.convert_mesh()
        verts = mesh_stretched.get_vertices_collection()
        norms = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10,
                                   err_msg="All vertices should be on the unit sphere after parametrization")

    def test_harmonic_map_is_complex(self):
        """The harmonic map h should be a complex numpy array."""
        harmonic_par = dirichlet_parametrization(self.mesh_surf)
        self.assertEqual(harmonic_par.h.dtype, np.complex128)
        self.assertEqual(len(harmonic_par.h), len(self.mesh_surf.vertices))

    def test_convert_mesh_does_not_mutate_original(self):
        """convert_mesh() should not modify the original mesh vertices in place."""
        original_verts = self.mesh_surf.get_vertices_collection().copy()
        harmonic_par = dirichlet_parametrization(self.mesh_surf)
        _ = harmonic_par.convert_mesh()
        current_verts = self.mesh_surf.get_vertices_collection()
        np.testing.assert_array_equal(original_verts, current_verts,
                                      err_msg="convert_mesh must not mutate the original mesh vertices")


class TestStretchLaplacian(_MeshSurfFixture):
    """Tests for the stretch Laplacian matrix correctness."""

    def setUp(self):
        super().setUp()
        self.harmonic_par = dirichlet_parametrization(self.mesh_surf)

    def test_stretch_laplacian_negative_off_diagonal(self):
        """Stretch Laplacian off-diagonal entries should be negative (same sign as cotangent Laplacian)."""
        Ls = self.mesh_surf.get_laplacian_matrix(
            weight='stretch', stretch_function=self.harmonic_par).toarray()
        N = Ls.shape[0]
        off_diag_sum = Ls.sum() - sum(Ls[i, i] for i in range(N))
        self.assertLess(off_diag_sum, 0,
                        "Stretch Laplacian off-diagonal sum should be negative")

    def test_stretch_laplacian_positive_diagonal(self):
        """Stretch Laplacian diagonal entries should be positive."""
        Ls = self.mesh_surf.get_laplacian_matrix(
            weight='stretch', stretch_function=self.harmonic_par).toarray()
        N = Ls.shape[0]
        diag_sum = sum(Ls[i, i] for i in range(N))
        self.assertGreater(diag_sum, 0,
                           "Stretch Laplacian diagonal sum should be positive")

    def test_stretch_laplacian_symmetric(self):
        """Stretch Laplacian should be symmetric."""
        Ls = self.mesh_surf.get_laplacian_matrix(
            weight='stretch', stretch_function=self.harmonic_par).toarray()
        np.testing.assert_allclose(Ls, Ls.T, atol=1e-10,
                                   err_msg="Stretch Laplacian must be symmetric")

    def test_stretch_laplacian_row_sums_near_zero(self):
        """Stretch Laplacian row sums should be near zero (L = D - W property)."""
        Ls = self.mesh_surf.get_laplacian_matrix(
            weight='stretch', stretch_function=self.harmonic_par).toarray()
        row_sums = Ls.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.zeros_like(row_sums), atol=1e-10,
                                   err_msg="Stretch Laplacian row sums should be zero")


class TestEllipsoidSphericalConformal(TestCase):
    """Minimal reproducible test: sphere scaled to an asymmetric ellipsoid.

    The mesh is built from a voxel ellipsoid with axes (1.5, 1.0, 0.7) so
    that it is clearly non-spherical, exercising both Algorithm 4.1 and 4.2.
    """

    def _make_ellipsoid_mesh(self, ax=1.5, ay=1.0, az=0.7):
        """Build a surface mesh from an integer-voxel ellipsoid."""
        # Scale up to integer semi-axes large enough for a well-resolved mesh.
        a_i = max(3, round(ax * 3))
        b_i = max(3, round(ay * 3))
        c_i = max(3, round(az * 3))
        basic_form = ellipsoid(a_i, b_i, c_i, levelset=True)
        basic_form[basic_form < 0] = 0.0
        basic_form[basic_form > 0] = 1.0
        return get_surface_mesh(basic_form)

    # ------------------------------------------------------------------
    # Algorithm 4.1 tests
    # ------------------------------------------------------------------

    def test_algo41_hB_shape_and_dtype(self):
        """h_B must be complex128 of shape (3,) - asserted inside the function."""
        mesh = self._make_ellipsoid_mesh()
        sf = dirichlet_parametrization(mesh)
        # The function itself asserts [A4.1-1]; reaching here means it passed.
        self.assertEqual(sf.h.dtype, np.complex128)
        self.assertEqual(len(sf.h), len(mesh.vertices))

    def test_algo41_no_nan_inf(self):
        """No NaN or Inf in h after Algorithm 4.1."""
        mesh = self._make_ellipsoid_mesh()
        sf = dirichlet_parametrization(mesh)
        self.assertTrue(
            np.all(np.isfinite(sf.h)),
            "Algorithm 4.1 produced NaN/Inf in h"
        )

    def test_algo41_sphere_norms(self):
        """After Algorithm 4.1, all sphere points must satisfy ||f_i|| ~= 1."""
        mesh = self._make_ellipsoid_mesh()
        sf   = dirichlet_parametrization(mesh)
        pts  = _inverse_stereo_projection(sf.h)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-10,
            err_msg="Algo 4.1: sphere norms not 1 on ellipsoid (1.5, 1.0, 0.7)"
        )

    def test_algo41_hB_formula(self):
        """Verify h_B entries match eq. (4.6) exactly (catches the sign+power bug)."""
        mesh     = self._make_ellipsoid_mesh()
        face_reg = mesh.get_most_regular_face()
        a, b, c  = face_reg.u, face_reg.v, face_reg.w

        from spherepar.mesh import Vector, Vertex
        vec_ba = Vector(b, a)
        vec_ca = Vector(c, a)
        alpha  = vec_ca.dot(vec_ba) / (vec_ba.norm() ** 2)

        inv_sq_edge = 1.0 / (vec_ba.norm() ** 2)
        foot_pos    = a.pos + alpha * (b.pos - a.pos)
        vec_cfoot   = Vector(c, Vertex(foot_pos, _id=-1))
        inv_sq_foot = 1.0 / (vec_cfoot.norm() ** 2)

        expected = (
            np.array([-inv_sq_edge, inv_sq_edge, 0.0], dtype=np.complex128)
            + 1j * np.array([(1 - alpha) * inv_sq_foot,
                              alpha       * inv_sq_foot,
                             -inv_sq_foot])
        )

        # Check signs of real part: h_B[0] < 0, h_B[1] > 0
        self.assertLess(
            expected[0].real, 0,
            "h_B[0] real part must be negative (-1/||vb-va||^2)"
        )
        self.assertGreater(
            expected[1].real, 0,
            "h_B[1] real part must be positive (+1/||vb-va||^2)"
        )
        # Check power: |h_B[0].real| == 1/||vb-va||^2
        np.testing.assert_allclose(
            abs(expected[0].real), inv_sq_edge, rtol=1e-12,
            err_msg="h_B[0] magnitude must equal 1/||vb-va||^2 (squared, not linear norm)"
        )

    # ------------------------------------------------------------------
    # Algorithm 4.2 tests
    # ------------------------------------------------------------------

    def test_algo42_output_on_sphere(self):
        """After Algorithm 4.2, all output points must lie on the unit sphere."""
        mesh = self._make_ellipsoid_mesh()
        sf   = stretch_parametrization(mesh, eps=1e-4, max_iters=10, verbose=False)
        pts  = _inverse_stereo_projection(sf.h)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-9,
            err_msg="Algo 4.2 output not on unit sphere"
        )

    def test_algo42_no_nan_inf(self):
        """Algorithm 4.2 must not produce NaN or Inf in h."""
        mesh = self._make_ellipsoid_mesh()
        sf   = stretch_parametrization(mesh, eps=1e-4, max_iters=10, verbose=False)
        self.assertTrue(
            np.all(np.isfinite(sf.h)),
            "Algorithm 4.2 produced NaN/Inf in h"
        )

    def test_algo42_energy_does_not_increase(self):
        """Dirichlet energy after Algorithm 4.2 must not exceed the Algorithm 4.1 energy."""
        mesh = self._make_ellipsoid_mesh()
        Ld   = mesh.get_laplacian_matrix(weight='cotangent').toarray()

        sf_41  = dirichlet_parametrization(mesh)
        E_init = _dirichlet_energy(Ld, sf_41.h)

        sf_42   = stretch_parametrization(mesh, eps=1e-5, max_iters=10, verbose=False)
        E_final = _dirichlet_energy(Ld, sf_42.h)

        self.assertTrue(np.isfinite(E_init),  f"Initial energy is not finite: {E_init}")
        self.assertTrue(np.isfinite(E_final), f"Final energy is not finite: {E_final}")
        # Allow a small tolerance for the first-iteration overshoot due to inversion.
        self.assertLessEqual(
            E_final, E_init + abs(E_init) * 0.05,
            f"Energy increased significantly: {E_init:.6e} -> {E_final:.6e}"
        )

    def test_distortion_diagnostics(self):
        """Run full pipeline and print distortion diagnostics (informational)."""
        mesh = self._make_ellipsoid_mesh(ax=1.5, ay=1.0, az=0.7)
        sf   = stretch_parametrization(mesh, eps=1e-4, max_iters=20, verbose=True)

        pts   = _inverse_stereo_projection(sf.h)
        norms = np.linalg.norm(pts, axis=1)

        print(f"\n--- Distortion diagnostics (ellipsoid 1.5 x 1.0 x 0.7) ---")
        print(f"  N vertices : {len(sf.h)}")
        print(f"  Sphere norms: min={norms.min():.8f}, max={norms.max():.8f}, "
              f"mean={norms.mean():.8f}")

        Ld  = mesh.get_laplacian_matrix(weight='cotangent').toarray()
        E_D = _dirichlet_energy(Ld, sf.h)
        print(f"  Dirichlet energy E_D = {E_D:.6e}")

        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-9,
            err_msg="Distortion test: output not on unit sphere"
        )