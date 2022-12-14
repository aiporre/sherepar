import os
import tempfile
import numpy as np
import nibabel as nb
import meshio
import pygalmesh
from skimage import measure
from skimage import segmentation
from nibabel.testing import data_path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_surface_mesh(data):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(data, 0)
    return (verts, faces, normals, values)


def plot_mesh(mesh):
    verts, faces, _, _ = mesh

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")
    xx = verts[:,0]
    yy = verts[:,1]
    zz = verts[:,2]
    ax.set_xlim(min(xx), max(xx))  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(min(yy), max(yy))  # b = 10
    ax.set_zlim(min(zz), max(zz))  # c = 16

    plt.tight_layout()
    plt.show()

def get_segments(data, mask=None, num_segments=100):
    # use the slic algorithm to get the segments
    segments = segmentation.slic(data, mask=mask, compactness=10, n_segments=num_segments, channel_axis=None)
    return segments

class Vertex:
    def __init__(self, pos, _id):
        self.pos = pos
        self.id = _id
class Edge:
    def __init__(self, u, v, _id):
        self.u = u
        self.v = v
        self.id = _id
    def get_vertex(self, v_id):
        if v_id == self.u.id:
            return self.u
        elif v_id == self.v.id:
            return self.v
        else:
            return None

class Face:
    def __init__(self, u, v, w, _id):
        self.u = u
        self.v = v
        self.w = w
        self.id = _id
        self._edges = None

    def get_edge(self, v1_id, v2_id):
        # create a list of edges
        if self._edges is None:
            triplet_vrtx = [self.u, self.v, self.w]
            for a, b in zip(triplet_vrtx[:-1], triplet_vrtx[1:]):
                if a.id > b.id:
                    a, b = b, a
                self._edges = {(a.id, b.id): (a, b)}
        # look for the pair of ids
        if v1_id > v2_id:
            v1_id, v2_id = v2_id, v1_id
        edge = self._edges.get((v1_id, v2_id), None)
        return edge
class Tetrahedron:
    def __init__(self, u, v, w, m, _id):
        self.u = u
        self.v = v
        self.w = w
        self.m = m
        self.id = _id
class Mesh:

    def __init__(self, vertices, edges, faces, tetrahedra):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.tetrahedra = tetrahedra

    def get_vertex_neighbors(self, v_id):
        neighbors = []
        for a, b in self.edges.keys():
            if a == v_id:
                neighbors.append(self.vertices[v_id])
            if b == v_id:
                neighbors.append(self.vertices[v_id])
        return neighbors

    def get_edge_faces(self, v_id):
        edge = self.edges[v_id]
        faces_with_edge = []
        for face in self.faces:
            if face.get_edge(edge.u._id, edge.v._id) is not None:
                faces_with_edge.append(face)
        return faces_with_edge

    def get_laplacian_matrix(self):
        raise NotImplementedError

class MeshVolume(Mesh):
    def __init__(self, *args, **kwargs):
        if "meshio_obj":
            meshio_obj = kwargs['meshio_obj']
            _vertices = {}
            for _id, v in enumerate(meshio_obj.points):
                _vertices[_id] = Vertex(v, _id)

            faces = [x for x in meshio_obj.cells if x.type == 'triangle'][0]
            _faces = {}
            for _id, f in enumerate(faces.data):
                u, v, w = self.vertices[f[0]], _vertices[f[1]], _vertices[f[2]]
                _faces[_id] = Face(u,v,w,_id)

            tetras = [x for x in meshio_obj.cells if x.type == 'tetras'][0]
            _tetrahedra = {}
            for _id, t in enumerate(tetras.data):
                u, v, w, m = _vertices[f[0]], _vertices[f[1]], _vertices[f[2]], _vertices[f[3]]
                _faces[_id] = Face(u,v,w, m, _id)

            # extract edges from
            _edges = {}
            _id = 0
            for tetra in tetras:
                for a, b in zip(tetra[:-1], tetra[1:]):
                    if a>b:
                        a, b = b, a
                    if (a,b) not in _edges:
                        v1, v2 = _vertices[a], _vertices[b]
                        _edges[(a,b)] = (v1,v2)










def get_edge_faces(self, id):
    raise NotImplementedError


class MeshFactory:
    def makeMesh(self, mesh_type, *args, **kwargs):

        if mesh_type == 'vol':
            return 0
        elif mesh_type == 'surf':
            return 1
        else:
            valid_mesh_types = ['vol', 'surf']
            raise Exception(f"mesh type {mesh_type} not implemented. Valid options are {valid_mesh_types}.")





class Segmentation:
    def __init__(self, data, mask=None, num_segments=100):
        self.data = data
        self.segments = get_segments(data, mask=mask, num_segments=num_segments)
        self.meshes = len(np.unique(self.segments))* [None]
        self.num_segments = num_segments

    def __len__(self):
        return self.num_segments
    def __getitem__(self, idx):
        if self.meshes[idx] is None:
            _mesh_surf = get_surface_mesh(self.segments==idx)
            # save in vtk file
            with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
                fname = f.name
                print('Created temporary file: ', fname)
                # write mesh surf into a vtk file
                points = _mesh_surf[0]
                cells = {'triangle': _mesh_surf[1]}
                mesh_output = meshio.Mesh(points, cells)
                mesh_output.write(fname)
                # convert mesh surface into mesh volume with pygalmesh
                mesh_vol = pygalmesh.generate_volume_mesh_from_surface_mesh(
                    fname,
                    min_facet_angle=25.0,
                    max_radius_surface_delaunay_ball=0.15,
                    max_facet_distance=0.008,
                    max_circumradius_edge_ratio=3.0,
                    verbose=False)
            self.meshes[idx] = mesh_vol

        return self.meshes[idx]


