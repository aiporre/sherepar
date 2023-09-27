import os
import numpy as np
from spherepar.read import read_nii
from spherepar.mesh import Segmentation, plot_mesh
import matplotlib.pyplot as plt

datapath = 'data/T1_brain.nii.gz'
# datapath = os.path.join(data_path, 'example5d.nii.gz')
data = read_nii(datapath) 
img = data.copy()
mask = np.zeros_like(img)
mask[img>0] = 1
only_surface = True
# make segmented image
segs = Segmentation(img, mask=mask, num_segments=10, only_surface=only_surface)




# Use marching cubes to obtain the surface mesh of these ellipsoids
# # create a plot 5x5 subplots
# fig, ax = plt.subplots(2, 5, subplot_kw=dict(projection='3d'))
# fig.set_size_inches(20, 20)
# ax = ax.flatten()
# plot the segmented mesh
for i in range(len(segs)):
    a = segs[i]
    if only_surface:
        _,  mesh_surf = a
    else:
        mesh_vol, mesh_surf = a
    a = segs[i]
    # create
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_mesh((mesh_surf.get_vertices_collection(), mesh_surf.get_faces_collection(), None, None), ax=ax)
    plt.draw()
    # print('Calculating the laplacian matrix')
    # L = mesh_surf.get_laplacian_matrix()
    # print(L)
plt.show()
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



