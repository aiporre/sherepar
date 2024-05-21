from __future__ import annotations

import math
import os
from collections import OrderedDict
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
from scipy.sparse import coo_matrix
import trimesh


def get_surface_mesh(data):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(data, 0.5, allow_degenerate=False)

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # repair broken faces
    n = trimesh.repair.broken_faces(mesh)
    print('Broken faces: ', n)
    mask = np.ones((mesh.faces.shape[0],), dtype=bool)
    for i in n:
        mask[i] = False
    mesh.update_faces(mask)
    # fix broken faces
    n = trimesh.repair.fill_holes(mesh)
    print('Holes filled: ', n)
    # Compute the Euler characteristic
    V = mesh.vertices.shape[0]
    E = mesh.edges.shape[0]
    F = mesh.faces.shape[0]
    chi = V - E + F
    print('Is watertight: ', mesh.is_watertight, 'Euler characteristic: ', chi)

    # extract new values
    verts, faces, normals, values = mesh.vertices, mesh.faces, mesh.face_normals, values

    return MeshFactory.make_mesh('surf', verts, faces)


def plot_mesh(mesh, ax=None):
    verts, faces, _, _ = mesh

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    ax_given = ax is not None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    mesh.set_facecolor('r')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")
    xx = verts[:, 0]
    yy = verts[:, 1]
    zz = verts[:, 2]
    ax.set_xlim(min(xx), max(xx))  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(min(yy), max(yy))  # b = 10
    ax.set_zlim(min(zz), max(zz))  # c = 16
    if not ax_given:
        plt.tight_layout()
        plt.show()


def get_segments(data, mask=None, num_segments=100):
    # use the slic algorithm to get the segments
    segments = segmentation.slic(data, mask=mask, compactness=10, n_segments=num_segments, channel_axis=None)
    return segments


class Vertex:
    def __init__(self, pos: tuple | list | np.ndarray, _id: int):
        self._pos = pos
        self.id = _id

    @property
    def pos(self) -> np.ndarray:
        if isinstance(self._pos, list) or isinstance(self._pos, tuple):
            self._pos = np.array(self._pos)
        return self._pos

    def __str__(self):
        return f"Vertex(pos={self.pos}, id={self.id})"

    def __repr__(self):
        return self.__str__()


class EdgeBase:
    pass


class Edge(EdgeBase):
    def __init__(self, u: Vertex, v: Vertex):
        self.u = u
        self.v = v
        self.id = (u.id, v.id)

    def get_vertex(self, v_id: int) -> Vertex | None:
        if v_id == self.u.id:
            return self.u
        elif v_id == self.v.id:
            return self.v
        else:
            return None

    def __eq__(self, other_edge) -> bool:
        if self.u.id == other_edge.u.id and self.v.id == other_edge.v.id \
                and self.u.pos == other_edge.u.pos and self.v.pos == other_edge.v.pos:
            return True
        elif self.u.id == other_edge.v.id and self.v.id == other_edge.u.id \
                and self.u.pos == other_edge.v.pos and self.v.pos == other_edge.u.pos:
            return True
        else:
            return False

    def __str__(self):
        return f"Edge(u = {self.u},  v = {self.v})"

    def __repr__(self):
        return self.__str__()


class Vector(Edge):

    def __init__(self, u: Vertex, v: Vertex):
        self.data = u.pos - v.pos
        super(Vector, self).__init__(u, v)

    def __eq__(self, other_vector):
        if self.u.id == other_vector.u.id and self.v.id == other_vector.v.id \
                and self.u.pos == other_vector.u.pos and self.v.pos == other_vector.v.pos:
            return True
        else:
            return False

    def dot(self, w: 'Vector') -> float:
        # u . w , where u is the vector on which w dot product is operated
        return float(self.data.dot(w.data))

    def cross(self, w: 'Vector') -> 'Vector':
        # u . w , where u is the vector on which w dot product is operated
        uw_pos = np.cross(self.data, w.data)
        new_vertex = Vertex(uw_pos, self.u.id + self.v.id)
        return Vector(self.u, new_vertex)

    def norm(self) -> float:
        return float(np.linalg.norm(self.data))

    def __str__(self):
        return f"Vector({self.data}, u={self.u.id}, v={self.v.id})"

    def __repr__(self):
        return f"Vector({self.u}, {self.v})"


class Face:
    def __init__(self, u: Vertex, v: Vertex, w: Vertex):
        self.u = u
        self.v = v
        self.w = w
        self.id = (u.id, v.id, w.id)
        self._edges = None

    def get_edge(self, v1_id: int, v2_id: int) -> Edge | None:
        # look for the pair of ids
        if v1_id > v2_id:
            v1_id, v2_id = v2_id, v1_id
        edge = self._edges_dict.get((v1_id, v2_id), None)
        if edge is not None:
            edge = Edge(edge[0], edge[1])
        return edge

    def get_opposite_vertex(self, v1_id: int, v2_id: int) -> Vertex | None:
        # check if vertex in the faces
        if v1_id > v2_id:
            v1_id, v2_id = v2_id, v1_id
        edge_vertices = self._edges_dict.get((v1_id, v2_id), None)
        if edge_vertices is None:
            return None

        # if edge is not None means that it is the other index.
        if self.u.id not in (v1_id, v2_id):
            return self.u
        elif self.v.id not in (v1_id, v2_id):
            return self.v
        elif self.w.id not in (v1_id, v2_id):
            return self.w
        else:
            return None

    @property
    def _edges_dict(self) -> dict[tuple[int, int], tuple[Vertex, Vertex]]:
        # create a list of edges
        if self._edges is None:
            triplet_vrtx = [self.u, self.v, self.w]
            self._edges = {}
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = triplet_vrtx[i], triplet_vrtx[j]
                    if a.id > b.id:
                        a, b = b, a
                    self._edges[(a.id, b.id)] = (a, b)

        return self._edges

    def area(self) -> float:
        '''
        Computes area using the Heron's formula
        :return:
        '''
        u, v, w = self.u, self.v, self.w
        v_a = Vector(u, v)
        v_b = Vector(u, w)
        v_c = Vector(v, w)
        a, b, c = v_a.norm(), v_b.norm(), v_c.norm()  # vector lengths
        s = (a + b + c) / 2
        return math.sqrt(s * (s - a) * (s - b) * (s - c))

    def regularity(self) -> float:
        '''
        Applies the most regular function eq 19

        :return: Scalar value of regularity of the triangular face
        '''

        u, v, w = self.u, self.v, self.w
        v_a = Vector(u, v)
        v_b = Vector(u, w)
        v_c = Vector(v, w)
        a, b, c = v_a.norm(), v_b.norm(), v_c.norm()  # vector lengths
        reg_1 = np.array([a, b, c])
        reg_2 = (a + b + c) * np.ones_like(reg_1) / 3
        return float(np.linalg.norm(reg_1 - reg_2))

    def __str__(self):
        return f"Face(u={self.u}, v={self.v}, w={self.w}, id={self.id})"

    def __repr__(self):
        return f"Face({self.u}, {self.v}, {self.w}) # with Vector.id={self.id})"


class Tetrahedron:
    def __init__(self, u: Vertex, v: Vertex, w: Vertex, m: Vertex):
        self.u = u
        self.v = v
        self.w = w
        self.m = m
        self.id = (u.id, v.id, w.id, m.id)

    def __str__(self):
        return f"Tetrahedron(u={self.u}, v={self.v}, w={self.w}, m={self.m} id={self.id})"

    def __repr__(self):
        return f"Face({self.u}, {self.v}, {self.w}, {self.m}) # with Tetra.id={self.id})"


class Mesh:

    def __init__(self,
                 vertices: dict[int, Vertex],
                 edges: dict[tuple[int, int], Edge],
                 faces: dict[tuple[int, int, int], Face]):
        self.vertices = vertices
        self.edges = edges
        self.faces = faces

    def get_vertex_neighbors(self, v_id: int) -> list[Vertex] | None:
        neighbors = []
        for a, b in self.edges.keys():
            if a == v_id:
                neighbors.append(self.vertices[b])
            elif b == v_id:
                neighbors.append(self.vertices[a])
        if len(neighbors) > 0:
            return neighbors
        else:
            return None

    def get_edge_faces(self, e_id: tuple[int, int]) -> list[Face] | None:
        # print('asdf')
        if e_id in self.edges:
            edge = self.edges[e_id]
        elif (e_id[1], e_id[0]) in self.edges:
            e_id = (e_id[1], e_id[0])
            edge = self.edges[e_id]
        else:
            return None
        faces_with_edge = []
        for _ids, face in self.faces.items():
            # print('lookin for ', (edge.u.id, edge.v.id), "---> ", face)
            if edge.u.id in _ids and edge.v.id in _ids:
                # print('found face: ', face)
                faces_with_edge.append(face)
        if len(faces_with_edge) > 0:
            return faces_with_edge
        else:
            return None

    def get_vertices_collection(self) -> np.ndarray:
        values = np.zeros((len(self.vertices), 3))
        for v in self.vertices.values():
            values[v.id] = v.pos
        return values

    def get_edges_collection(self, use_id: bool = True) -> np.ndarray | list[Edge]:
        if use_id:
            values = np.zeros((len(self.edges), 2))
        else:
            values = []
        for i, e in enumerate(self.edges.values()):
            if use_id:
                values[i] = np.array(e.id)
            else:
                values.append(e)
        return values

    def get_faces_collection(self, use_id: bool = True) -> np.ndarray | list[Edge]:
        if use_id:
            values = np.zeros((len(self.faces), 3), dtype=np.int32)
        else:
            values = []
        for i, f in enumerate(self.faces.values()):
            if use_id:
                values[i] = np.array(f.id, dtype=np.int32)
            else:
                values.append(f)
        return values

    def get_laplacian_matrix(self):
        raise NotImplementedError

    def __str__(self):
        return f"Mesh(vertices={len(self.vertices)}*Vertex, edges={len(self.edges)}*Edges, faces={len(self.faces)}*Faces)"

    def __repr__(self):
        return f"Mesh(vertices={len(self.vertices)}*Vertex, edges={len(self.edges)}*Edges, faces={len(self.faces)}*Faces)"


class MeshVolume(Mesh):
    def __init__(self, *args, **kwargs):
        if "meshio_obj":
            meshio_obj = kwargs['meshio_obj']
            _vertices = OrderedDict()
            for _id, v in enumerate(meshio_obj.points):
                _vertices[_id] = Vertex(v, _id)

            faces = [x for x in meshio_obj.cells if x.type == 'triangle'][0]
            _faces = {}
            for f in faces.data:
                u, v, w = _vertices[f[0]], _vertices[f[1]], _vertices[f[2]]
                f_obj = Face(u, v, w)
                _faces[f_obj.id] = f_obj

            tetras = [x for x in meshio_obj.cells if x.type == 'tetra'][0]
            _tetrahedra = {}
            for f in tetras.data:
                u, v, w, m = _vertices[f[0]], _vertices[f[1]], _vertices[f[2]], _vertices[f[3]]
                t_obj = Tetrahedron(u, v, w, m)
                _tetrahedra[t_obj.id] = t_obj

            # extract edges from
            _edges = {}
            for tetra in tetras.data:
                for i in range(4):
                    for j in range(i + 1, 4):
                        a, b = tetra[i], tetra[j]
                        if a > b:
                            a, b = b, a
                        if (a, b) not in _edges:
                            v1, v2 = _vertices[a], _vertices[b]
                            e_obj = Edge(v1, v2)
                            _edges[e_obj.id] = e_obj

        self.tetrahedra = _tetrahedra
        super(MeshVolume, self).__init__(_vertices, _edges, faces)

    def get_tetrahedra_collection(self, use_id: bool = True) -> np.ndarray | list[Edge]:
        if use_id:
            values = np.zeros((len(self.vertices), 4))
        else:
            values = []
        for i, t in enumerate(self.tetrahedra):
            if use_id:
                values[i] = np.array(t.id)
            else:
                values.append(t)
        return values

    def __str__(self):
        return f"Mesh(vertices={len(self.vertices)}*Vertex, edges={len(self.edges)}*Edges, faces={len(self.faces)}*Faces, tetras={len(self.tetrahedra)}*Tetras)"

    def __repr__(self):
        return self.__str__()


class MeshSurf(Mesh):
    def __init__(self, vertices: list[np.ndarray] | np.ndarray, faces: list[np.ndarray] | np.ndarray):
        _vertices = OrderedDict()
        for _id, v in enumerate(vertices):
            _vertices[_id] = Vertex(v, _id)
        # extract faces
        _faces = {}
        for f in faces:
            f.sort()
            u, v, w = _vertices[f[0]], _vertices[f[1]], _vertices[f[2]]
            f_obj = Face(u, v, w)
            if f_obj.id in _faces:
                print('Face already in the list', f_obj.id, 'skipping...')
                continue
            _faces[f_obj.id] = f_obj

        # extrac edges from the faces
        _edges = {}
        for f in faces:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = f[i], f[j]
                    if a > b:
                        a, b = b, a
                    if (a, b) not in _edges:
                        v1, v2 = _vertices[a], _vertices[b]
                        e_obj = Edge(v1, v2)
                        _edges[e_obj.id] = e_obj

                    # class properties
        super(MeshSurf, self).__init__(_vertices, _edges, _faces)

    def get_laplacian_matrix(self, weight: str = 'cotangent', *args, **kwargs):
        if weight == 'cotangent':
            return self._get_laplacian_cotangent()
        elif weight == 'stretch':
            return self._get_laplacian_stretch(*args, **kwargs)
        else:
            raise ValueError(f'Weight = {weight} not implemented valid is [cotangent, or ...')

    def _get_laplacian_cotangent(self):
        values = []
        index_row = []
        index_col = []
        for _id, v in self.vertices.items():
            neighbors = self.get_vertex_neighbors(_id)
            values_col = []
            for k in neighbors:
                face = self.get_edge_faces((v.id, k.id))
                if face is None:
                    print(f'Edge {(v.id, k.id)} has not two faces. Surface is not a genus-zero closed !')
                    continue
                assert len(face) == 2, 'Error calculation of laplacian matrix, wrong definition of faces. ' \
                                       f'Edge {(v.id, k.id)} must have two faces (now={face}). Number of faces={len(face)}'

                # v---k forms the central vector
                face_a, face_b = face[0], face[1]
                a = face_a.get_opposite_vertex(v.id, k.id)
                b = face_b.get_opposite_vertex(v.id, k.id)

                # a---->v and b----> are the U_vector's
                # a---->k and b---->k are the V_vector's
                # """
                #            V         k        V
                # a───────────────────► ◄──────────────────b
                #  \ ) α_ij            ▲           β_ij ( /
                #   \                  │                 /
                #    \                 │                /
                #     \                │               /
                #      \               │              /
                #       \              │             /
                #        \             │            /
                #         \            │           /
                #          \           │          /
                #        U  \          │         /U
                #            \         │        /
                #             \        │       /
                #              \       │      /
                #               \      │     /
                #                \     │    /
                #                 \    │   /
                #                  \   │  /
                #                   \  │ /
                #                    \ │/
                #                     \/
                #                     v
                # """
                # alpha_ij
                u_vec = Vector(a, v)
                v_vec = Vector(a, k)
                cotangent_alpha = u_vec.dot(v_vec) / u_vec.cross(v_vec).norm()
                # beta_ij
                u_vec = Vector(b, v)
                v_vec = Vector(b, k)
                cotangent_beta = u_vec.dot(v_vec) / u_vec.cross(v_vec).norm()
                values_col.append((cotangent_alpha + cotangent_beta) / 2)
                index_col.append(v.id)
                index_row.append(k.id)
            ### once completed the computation of w_ij for all j in neighbors to i we sum to compute the diagonal of L
            values.extend(values_col)
            values.append(sum(values_col))
            index_col.append(v.id)
            index_row.append(v.id)
        N = max((max(index_row), max(index_col))) + 1
        return coo_matrix((values, (index_row, index_col)), shape=(N, N))

    def _get_laplacian_stretch(self, stretch_function: "StretchFunction", *args, **kwargs) -> coo_matrix:
        values = []
        index_row = []
        index_col = []
        for _id, v in self.vertices.items():
            neighbors = self.get_vertex_neighbors(_id)
            values_col = []
            for k in neighbors:
                face = self.get_edge_faces((v.id, k.id))
                if face is None:
                    print(f'Edge {(v.id, k.id)} has not two faces. Surface is not a genus-zero closed !')
                    continue
                assert len(face) == 2, 'Error calculation of laplacian matrix, wrong definition of faces. ' \
                                       f'Edge {(v.id, k.id)} must have two faces (now={face}). N' \
                                       f'Number of faces={len(face)} '

                # v---k forms the central vector
                face_a, face_b = face[0], face[1]
                # stretch the faces:
                face_a_s, face_b_s = stretch_function(face_a), stretch_function(face_b)
                v_s, k_s = stretch_function(v), stretch_function(k)
                a_s = face_a_s.get_opposite_vertex(v_s.id, k_s.id)
                b_s = face_b_s.get_opposite_vertex(v_s.id, k_s.id)
                # a---->v and b----> are the U_vector's in the stretched triangle face
                # a---->k and b---->k are the V_vector's in the stretched triangle face
                # alpha_ij(f)
                u_vec = Vector(a_s, v)
                v_vec = Vector(a_s, k)
                cotangent_alpha = u_vec.dot(v_vec) / u_vec.cross(v_vec).norm()
                stretch_factor_alpha = stretch_function.stretch_factor(face_a)
                # beta_ij(f)
                u_vec = Vector(b_s, v)
                v_vec = Vector(b_s, k)
                cotangent_beta = u_vec.dot(v_vec) / u_vec.cross(v_vec).norm()
                stretch_factor_beta = stretch_function.stretch_factor(face_b)
                values_col.append((cotangent_alpha / stretch_factor_alpha + cotangent_beta / stretch_factor_beta) / 2)
                index_col.append(v.id)
                index_row.append(k.id)
            ### once completed the computation of w_ij for all j in neighbors to i we sum to compute the diagonal of L
            values.extend(values_col)
            values.append(sum(values_col))
            index_col.append(v.id)
            index_row.append(v.id)
        N = max((max(index_row), max(index_col))) + 1
        return coo_matrix((values, (index_row, index_col)), shape=(N, N))

    def get_most_regular_face(self) -> Face:
        f_best = next(iter(self.faces.values()))
        for f in self.faces.values():
            if f.regularity() < f_best.regularity():
                f_best = f
        return f_best


class StretchFunction:
    def __init__(self, mesh: Mesh, harmonic_map: np.ndarray):
        # TODO: change implementation to use a dictionary (id, harmonic_point or complex number)
        self.mesh = mesh
        self.h = harmonic_map

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, harmonic_map):
        # validate that the harmonic map is a complex number with the same number of elements as the number of vertices
        if not isinstance(harmonic_map, np.ndarray):
            raise ValueError(f"Harmonic map must be a numpy array, not {type(harmonic_map)}")
        if harmonic_map.dtype != np.complex128:
            raise ValueError(f"Harmonic map must be a complex number, not {harmonic_map.dtype}")
        if len(harmonic_map) != len(self.mesh.vertices):
            raise ValueError(
                f"Harmonic map must have the same number of elements as the number of vertices in the mesh.")
        self._h = harmonic_map

    def __call__(self, cell: Vertex | Face) -> Vertex | Face:
        def _stretch_vertex(v):
            # validate that the vertex is in the mesh. In other words that the v.id is in the mesh
            # TODO: the validation is not working properly since the id should be necessarity the same as an index.
            # if v.id not in self.mesh.vertices:
            #     raise ValueError(f"Vertex {v} is not in the mesh.")
            h_l = self.h[v.id]
            pos = [2 * np.real(h_l) / (np.absolute(h_l) ** 2 + 1),
                   2 * np.imag(h_l) / (np.absolute(h_l) ** 2 + 1),
                   (np.absolute(h_l) ** 2 - 1) / (np.absolute(h_l) ** 2 + 1)]
            return Vertex(pos, _id=v.id)

        if isinstance(cell, Vertex):
            return _stretch_vertex(cell)
        elif isinstance(cell, Face):
            u_s, v_s, w_s = _stretch_vertex(cell.u), _stretch_vertex(cell.v), _stretch_vertex(
                cell.w)  # stretch vertices
            return Face(u_s, v_s, w_s)  # stretched face
        else:
            raise ValueError(f" Cell instance {type(cell)} is not implemented in the StretchFunction .")

    def stretch_factor(self, face: Face) -> float:
        '''
        Computes the stretch factor of a face, as the given in the following equation:
            $$\sigma_{f^{-1}(\tau)} = \frac{\tau}{f(\tau)}$$
        It is a measurment of the local stretch of the face, by the map $f$.
        :param face:
        :return:
        '''
        face_s = self.__call__(face)
        return face.area() / face_s.area()

    def __str__(self) -> str:
        return f"StretchFunction(mesh={self.mesh}, harmonic_map={self.h})"

    def __repr__(self) -> str:
        return f"StretchFunction(mesh={self.mesh}, harmonic_map={self.h})"

    def convert_mesh(self) -> Mesh:
        vertices = self.mesh.vertices
        # print('old vertices: ', vertices)
        for id, v in self.mesh.vertices.items():
            print('id...?.', id)
            print('vertex: ', v)
            print('new vertex: ', self.__call__(v))
            vertices[id] = self.__call__(v)
        # print('new vertices: ', vertices)
        return Mesh(vertices, self.mesh.edges, self.mesh.faces)



class MeshFactory:
    @staticmethod
    def make_mesh(mesh_type, *args, **kwargs):
        if mesh_type == 'vol':
            return MeshVolume(*args, **kwargs)
        elif mesh_type == 'surf':
            return MeshSurf(*args, **kwargs)
        else:
            valid_mesh_types = ['vol', 'surf']
            raise Exception(f"mesh type {mesh_type} not implemented. Valid options are {valid_mesh_types}.")


class Segmentation:
    def __init__(self, data, mask=None, num_segments=100, only_surface=False):
        self.data = data
        self.segments = get_segments(data, mask=mask, num_segments=num_segments)
        self.meshes = (len(np.unique(self.segments)) - 1) * [None]
        self.num_segments = num_segments
        self.only_surface = only_surface

    def __len__(self):
        return self.num_segments

    def __getitem__(self, idx):
        # index start in 1 to len(self)
        if self.meshes[idx] is None:
            seg_value = idx + 1
            _mesh_surf = get_surface_mesh(self.segments == seg_value)
            if self.only_surface:
                print('Only surface mesh is required, returning the surface mesh')
                self.meshes[idx] = (None, _mesh_surf)
                return self.meshes[idx]

            # save in vtk file
            with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
                fname = f.name
                print('Created temporary file: ', fname)
                # write mesh surf into a vtk file
                points = _mesh_surf.get_vertices_collection()
                cells = {'triangle': _mesh_surf.get_faces_collection()}
                mesh_output = meshio.Mesh(points, cells)
                mesh_output.write(fname)
                # convert mesh surface into mesh volume with pygalmesh
                try:
                    mesh_vol = pygalmesh.generate_volume_mesh_from_surface_mesh(
                        fname,
                        min_facet_angle=25.0,
                        max_radius_surface_delaunay_ball=0.15,
                        max_facet_distance=0.008,
                        max_circumradius_edge_ratio=3.0,
                        verbose=False)
                except RuntimeError:
                    mesh_vol = pygalmesh.generate_volume_mesh_from_surface_mesh(
                        fname,
                        min_facet_angle=25.0,
                        max_radius_surface_delaunay_ball=0.15,
                        max_facet_distance=0.008,
                        max_circumradius_edge_ratio=3.0,
                        verbose=False,
                        reorient=True)
                mesh_vol = MeshFactory.make_mesh('vol', meshio_obj=mesh_vol)

            self.meshes[idx] = (mesh_vol, _mesh_surf)

        return self.meshes[idx]
