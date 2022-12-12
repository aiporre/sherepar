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
verts, faces, normals, values = segs.meshes[50]
mesh = (verts, faces, normals, values)
plot_mesh(mesh)




