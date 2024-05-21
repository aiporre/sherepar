"""
script that applies the spherical parameterization to a single mesh

"""
import argparse
import numpy as np
import os
import sys
import trimesh
from matplotlib import pyplot as plt

from spherepar.mesh import MeshFactory, plot_mesh
from spherepar.parametrization import dirichlet_parametrization, stretch_parametrization
def main(mesh_path):
    # read mesh
    m = trimesh.load_mesh(mesh_path)
    # create a segmentation object
    vert, face = m.vertices, m.faces
    mesh = MeshFactory.make_mesh('surf', vert, face)
    sphere = mesh
    # compute the parameterization
    # stretch = stretch_parametrization(mesh)
    # stretch = dirichlet_parametrization(mesh)
    # sphere = stretch.convert_mesh()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh((sphere.get_vertices_collection(), sphere.get_faces_collection(), None, None), ax=ax)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply spherical parameterization to a single mesh')
    parser.add_argument('mesh_path', type=str, help='path to the mesh file')
    args = parser.parse_args()
    main(args.mesh_path)