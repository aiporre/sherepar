from unittest import TestCase
from skimage import measure
from skimage.draw import ellipsoid
from spherepar.mesh import *
from spherepar.parametrization import dirichlet_spherepar, stretch_paremetrization
class Test(TestCase):
    def test_dirichlet_spherepar(self):
        basic_form = ellipsoid(6, 10, 16, levelset=True)
        # make 0 and 1 the values of the levelset
        basic_form[basic_form <0] = 0.0
        basic_form[basic_form >0] = 1.0
        print('basic form shape: is a box with points: ', basic_form.shape)
        mesh_surf = get_surface_mesh(basic_form)

        # fig = plt.figure(figsize=(10, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # plot_mesh((mesh_surf.get_vertices_collection(), mesh_surf.get_faces_collection(), None, None), ax=ax)
        # plt.draw()
        # L = mesh_surf.get_laplacian_matrix()
        # self.assertEqual(L.sum(), 628.316737134868)
        # self.assertAlmostEqual((L - L.transpose()).sum(), 0)
        harmonic_par = dirichlet_spherepar(mesh_surf)

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
        harmonic_par = stretch_paremetrization(mesh_surf)
