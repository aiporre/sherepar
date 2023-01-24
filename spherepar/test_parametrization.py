from unittest import TestCase
from skimage import measure
from skimage.draw import ellipsoid
from spherepar.mesh import *

class Test(TestCase):
    def test_dirichlet_spherepar(self):
        basic_form = ellipsoid(6, 10, 16, levelset=True)
        print('basic form shape: is a box with points: ', basic_form.shape)
        mesh_surf = get_surface_mesh(basic_form)
        L = mesh_surf.get_laplacian_matrix()
        self.assertEqual(L.sum(), 628.316737134868)
        self.assertAlmostEqual((L - L.transpose()).sum(), 0)
