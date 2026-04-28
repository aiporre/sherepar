"""Example script that runs FLASH via the reusable spherical module."""

from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from spherepar.flash_parametrization import flash_map, load_mesh_with_trimesh
import argparse


def plot_mesh(vertices, faces, ax=None):
    ax_given = ax is not None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(vertices[faces])
    poly.set_edgecolor("k")
    poly.set_facecolor("r")
    ax.add_collection3d(poly)
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_aspect("equal")
    if not ax_given:
        plt.tight_layout()
        plt.show()


def main(file_name):
    # mesh = load_mesh_with_trimesh("data/ellipsoid.obj")
    mesh = load_mesh_with_trimesh(file_name)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    plot_mesh(mesh.vertices, mesh.faces, ax=ax1)
    ax1.set_title("Original Mesh")

    sphere_vertices = flash_map(mesh)
    ax2 = fig.add_subplot(122, projection="3d")
    plot_mesh(sphere_vertices, mesh.faces, ax=ax2)
    ax2.set_title("FLASH Sphere Map")
    plt.show()


if __name__ == "__main__":
    # argments
    parser = argparse.ArgumentParser(
        description=" Example script that runs FLASH via the reusable spherical module."
    )

    parser.add_argument("--file_name", help="mesh file name", default="data/ellipsoid.obj")
    args = parser.parse_args()

    main(args.file_name)
