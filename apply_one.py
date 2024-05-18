"""
script that applies the spherical parameterization to a single mesh

"""
import argparse
import numpy as np
import os
import sys
import trimesh

from spherepar.mesh import MeshFactory
from spherepar.parametrization import dirichlet_parametrization
def main(mesh_path):
    # read mesh
    m = trimesh.load_mesh(mesh_path)
    # create a segmentation object
    vert, face = m.vertices, m.faces
    mesh = MeshFactory.make_mesh('surf', vert, face)
    # compute the parameterization
    stretch = dirichlet_parametrization(mesh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply spherical parameterization to a single mesh')
    parser.add_argument('mesh_path', type=str, help='path to the mesh file')
    args = parser.parse_args()
    main(args.mesh_path)