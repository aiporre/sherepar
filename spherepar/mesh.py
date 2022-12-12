import os
import numpy as np
import nibabel as nb
from skimage import measure
from skimage import segmentation
from nibabel.testing import data_path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def get_mesh(data):
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

class Segmentation:
    def __init__(self, data, mask=None, num_segments=100):
        self.data = data
        self.segments = get_segments(data, mask=mask, num_segments=num_segments)
        self.meshes = []
        for i in np.unique(self.segments):
            self.meshes.append(get_mesh(self.segments==i))


