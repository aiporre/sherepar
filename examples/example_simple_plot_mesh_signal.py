import numpy as np
import trimesh

# select backend
# import matplotlib
# matplotlib.use("Agg")  # interactive backend; if unavailable, try "TkAgg"

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mesh",
    default='data/wfA_000.obj',
    help="Path to the input mesh OBJ file.",
)

parser.add_argument(
    "--signal",
    default="data/wfA_000.npy",
    help="Path to the input signal NPY file.",
)

args = parser.parse_args()


# # mesh = trimesh.load('data/ellipsoid.obj')
# # signal = np.random.rand(len(mesh.vertices))
# mesh = trimesh.load_mesh('data/wfA_000.obj')
# signal = np.load('data/wfA_000.npy')
mesh_file = args.mesh
signal_file = args.signal
print("loading mesh:", mesh_file)
mesh = trimesh.load_mesh(mesh_file)

print("laoding signal: ", signal_file)
signal = np.load(signal_file)
print(">>> vertices shape ", mesh.vertices.shape)
print(">>> signal shape ", signal.shape)



fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

face_signal = signal[mesh.faces].mean(axis=1)

poly = Poly3DCollection(mesh.vertices[mesh.faces], alpha=0.8, edgecolor='black', linewidths=0.2)
poly.set_array(face_signal)
poly.set_cmap('viridis')

ax.add_collection3d(poly)



ax.set_xlim(mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max())
ax.set_ylim(mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
ax.set_zlim(mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max())

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mesh with Signal Values')

fig.colorbar(poly, ax=ax, label='Signal Value')
plt.show()