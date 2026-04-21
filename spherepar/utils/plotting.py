from __future__ import annotations

from typing import Optional

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
import numpy.typing as npt


def plot_mesh(
    verts: npt.ArrayLike,
    faces: npt.ArrayLike,
    ax: Optional[Axes3D] = None,
) -> Axes3D:
    """
    Plot a triangular surface mesh using Matplotlib.

    Parameters
    ----------
    verts : array-like of shape (V, 3)
        Vertex coordinates of the mesh, where `V` is the number of vertices.
        Each row stores the 3D coordinates `(x, y, z)` of one vertex.

    faces : array-like of shape (F, 3)
        Triangle connectivity array, where `F` is the number of faces.
        Each row contains three integer vertex indices defining one triangle.

    ax : matplotlib 3D axis, optional
        Existing 3D axis on which to draw the mesh. If not provided, a new
        figure and 3D axis are created.

    Returns
    -------
    Axes3D
        The 3D axis containing the plotted mesh.

    Raises
    ------
    ValueError
        If `verts` does not have shape `(V, 3)` or `faces` does not have
        shape `(F, 3)`.

    Notes
    -----
    This function creates a `Poly3DCollection` from the triangle list
    `verts[faces]`, adds it to the axis, and scales the axis limits to match
    the mesh extents.

    The function returns the axis so it can be reused for further drawing
    or customization.
    """
    # Convert inputs to NumPy arrays to ensure consistent indexing behavior.
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)

    # Basic shape validation for a triangular surface mesh.
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(
            f"`verts` must have shape (V, 3), but got {verts.shape}."
        )
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"`faces` must have shape (F, 3), but got {faces.shape}."
        )

    # Track whether the caller supplied an axis. If not, create a new figure.
    ax_given = ax is not None
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

    # Build the list of mesh triangles via fancy indexing:
    # verts[faces] has shape (F, 3, 3), i.e. one 3D triangle per face.
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor("k")
    mesh.set_facecolor("r")
    ax.add_collection3d(mesh)

    # Set generic axis labels. These are more reusable than dataset-specific
    # labels such as fixed ellipsoid semi-axes.
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Compute mesh bounds from the vertex coordinates.
    x_min, y_min, z_min = verts.min(axis=0)
    x_max, y_max, z_max = verts.max(axis=0)

    # Match axis limits to the mesh extents.
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_zlim(float(z_min), float(z_max))

    # Keep the displayed object visually undistorted.
    # For 3D axes, set_box_aspect is usually more reliable than set_aspect.
    ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))

    # Only finalize the figure if we created it in this function.
    if not ax_given:
        plt.tight_layout()
        plt.show()

    return ax