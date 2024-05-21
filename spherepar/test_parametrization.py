from unittest import TestCase

import matplotlib.pyplot as plt
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
