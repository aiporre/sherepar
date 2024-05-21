import unittest
from skimage import measure
from skimage.draw import ellipsoid
from spherepar.mesh import *


class TestMeshSurf(unittest.TestCase):
    def test_create_a_mesh_surf(self):
        basic_form = ellipsoid(6, 10, 16, levelset=True)
        print('basic form shape: is a box with points: ', basic_form.shape)
        mesh_surf = get_surface_mesh(basic_form)
        print(mesh_surf)

    def test_get_xyz_collections(self):
        basic_form = ellipsoid(6, 10, 16, levelset=True)
        print('basic form shape: is a box with points: ', basic_form.shape)
        mesh_surf = get_surface_mesh(basic_form)
        N = 1942
        vertices_collection = mesh_surf.get_vertices_collection()
        faces_collection = mesh_surf.get_faces_collection()
        self.assertTrue(isinstance(vertices_collection, np.ndarray))
        self.assertTupleEqual(vertices_collection.shape, (N, 3))
        self.assertTrue(vertices_collection.dtype == np.dtype(np.float64))
        self.assertTrue(faces_collection.dtype == np.dtype(np.int32))
        plot_mesh((mesh_surf.get_vertices_collection(), mesh_surf.get_faces_collection(), None, None))

    def test_get_xyz_collections_v2(self):
        basic_form = ellipsoid(6, 10, 16, levelset=True)
        print('basic form shape: is a box with points: ', basic_form.shape)
        mesh_surf = get_surface_mesh(basic_form)
        L = mesh_surf.get_laplacian_matrix()
        self.assertEqual(L.sum(), 628.316737134868)
        self.assertAlmostEqual((L - L.transpose()).sum(), 0)

class TestVector(unittest.TestCase):
    def test_vector(self):
        a = Vertex(pos=(1, 1, 0), _id=0)
        b = Vertex(pos=(0, 1, 0), _id=1)
        c = Vertex(pos=(1, 0, 0), _id=2)
        u = Vector(a, b)
        v = Vector(b, c)
        # test dot product
        self.assertEqual(u.dot(v), -1)
        # test norm
        self.assertEqual(u.norm(), 1)
        # test cross product
        self.assertEqual(u.cross(v).norm(), 1)