import argparse
import numpy as np
import matplotlib.pyplot as plt
from markdown_it.presets import default
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import SphericalVoronoi, geometric_slerp
from scipy.spatial import Delaunay
import trimesh


def generate_random_ellipsoid_points(a=2, b=1.5, c=1, num_points=100):
    """
    Generate random points on an ellipsoid surface using fibonacci sphere algorithm
    with random perturbation.

    Parameters:
    a, b, c: Semi-axes lengths
    num_points: Number of points to generate

    Returns:
    points: Array of (x, y, z) coordinates on the ellipsoid
    """
    # Use Fibonacci sphere for better distribution
    indices = np.arange(0, num_points, dtype=float) + 0.5

    # Golden angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    # Random perturbation for non-uniform distribution
    theta = golden_angle * indices + np.random.uniform(-0.3, 0.3, num_points)
    phi = np.arccos(1 - 2 * indices / num_points) + np.random.uniform(-0.1, 0.1, num_points)

    # Clamp phi to valid range
    phi = np.clip(phi, 0, np.pi)

    # Convert to Cartesian coordinates on ellipsoid
    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi)

    return np.column_stack((x, y, z))


def create_watertight_mesh(points):
    """
    Create a watertight mesh using Delaunay triangulation in 3D.

    Parameters:
    points: Array of (x, y, z) coordinates

    Returns:
    vertices: Vertex array
    faces: Face array (triangles on the convex hull)
    """
    # Use Delaunay triangulation
    tri = Delaunay(points)

    # Get convex hull faces (outer surface only - watertight)
    # Extract faces from the convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)

    return points, hull.simplices


def save_to_obj(filename, vertices, faces):
    """
    Save mesh data to OBJ file.

    Parameters:
    filename: Output OBJ file path
    vertices: Array of vertex coordinates (N x 3)
    faces: Array of face indices (M x 3)
    """
    # save with trimehs

    mesh_obj = trimesh.Trimesh(vertices, faces)
    mesh_obj.fix_normals()
    mesh_obj.export(filename)
    # with open(filename, 'w') as f:
    #     # Write header
    #     f.write("# Ellipsoid Mesh (Watertight)\n")
    #     f.write(f"# Vertices: {len(vertices)}\n")
    #     f.write(f"# Faces: {len(faces)}\n\n")
    #
    #     # Write vertices
    #     for vertex in vertices:
    #         f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
    #
    #     f.write("\n")
    #
    #     # Write faces (OBJ indices are 1-based, not 0-based)
    #     for face in faces:
    #         f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"✓ Saved to {filename}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")


def plot_ellipsoid(vertices, faces):
    """
    Plot the watertight ellipsoid mesh in 3D.

    Parameters:
    vertices: Array of vertex coordinates
    faces: Array of face indices
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Extract triangle vertices for plotting
    triangle_vertices = vertices[faces]

    # Plot as a collection of triangles
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(triangle_vertices, alpha=0.7, linewidths=0.3, edgecolors='black')
    poly.set_facecolor([0.2, 0.6, 0.9])
    poly.set_edgecolor('lightgray')
    ax.add_collection3d(poly)

    # Plot vertices as points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c='red', s=20, alpha=0.5, label='Vertices')

    # Set labels and title
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)
    ax.set_title(f'Watertight Ellipsoid Mesh ({len(vertices)} vertices, {len(faces)} faces)',
                 fontsize=12, fontweight='bold')

    # Set equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a watertight ellipsoid mesh."
    )
    parser.add_argument("--a", type=float, default=2, help="Semi-axis length along x.")
    parser.add_argument("--b", type=float, default=1.5, help="Semi-axis length along y.")
    parser.add_argument("--c", type=float, default=1, help="Semi-axis length along z.")
    parser.add_argument(
        "--num-point",
        "--num_points",
        dest="num_point",
        type=int,
        default=400,
        help="Number of sampled surface points.",
    )
    parser.add_argument(
        "--file-name",
        "--file_name",
        default="ellipsoid.obj",
        dest="file_name",
        type=str,
        help="File name of the ellipsoid file in ../data"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Generating random points on ellipsoid surface...")

    # Generate random points on ellipsoid
    vertices = generate_random_ellipsoid_points(
        a=args.a,
        b=args.b,
        c=args.c,
        num_points=args.num_point,
    )
    print(f"Generated {len(vertices)} random points")

    print("\nCreating watertight mesh using convex hull...")
    vertices, faces = create_watertight_mesh(vertices)
    print(f"Created watertight mesh with {len(faces)} triangular faces")

    # Save to OBJ file
    print("\nSaving to OBJ file...")
    save_to_obj(f"../data/{args.file_name}", vertices, faces)

    # Plot the mesh
    print("\nPlotting watertight ellipsoid...")
    plot_ellipsoid(vertices, faces)


if __name__ == "__main__":
    main()
