import os
import numpy as np
from spherepar.read import read_nii
from spherepar.mesh import Segmentation, plot_mesh

datapath = 'data/T1_brain.nii.gz'
# datapath = os.path.join(data_path, 'example5d.nii.gz')
data = read_nii(datapath) 
img = data.copy()
mask = np.zeros_like(img)
mask[img>0] = 1
# make segmented image
segs = Segmentation(img, mask=mask, num_segments=100)

# Use marching cubes to obtain the surface mesh of these ellipsoids
a = segs[50]
print('Calculating the laplacian matrix')
mesh_vol, mesh_surf = a
L = mesh_surf.get_laplacian_matrix()
print(L)

# verts, faces, normals, values = segs[50]
# mesh = (verts, faces, normals, values)
# plot_mesh(mesh)

# import pygalmesh
#
# s = pygalmesh.Ball([0, 0, 0], 1.0)
# mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=0.2)
#
# # import pygalmesh
#
# mesh = pygalmesh.generate_volume_mesh_from_surface_mesh(
#     "elephant.vtu",
#     min_facet_angle=25.0,
#     max_radius_surface_delaunay_ball=0.15,
#     max_facet_distance=0.008,
#     max_circumradius_edge_ratio=3.0,
#     verbose=False,)



