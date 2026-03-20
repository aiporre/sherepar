from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from skimage.draw import ellipsoid
from spherepar.mesh import *
from spherepar.parametrization import dirichlet_parametrization, stretch_parametrization


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
