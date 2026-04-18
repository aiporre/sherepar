"""
Example: Read OBJ file and compute spherical conformal parameterization with LaPy

This script demonstrates how to:
1. Read an OBJ file using argparse
2. Convert it to a TriaMesh (as list of lists)
3. Compute spherical conformal parameterization
4. Plot both the original mesh and the parameterized result
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lapy.conformal import spherical_conformal_map
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import trimesh
import lapy

# try:
#     import lapy
# except ImportError:
#     raise ImportError(
#         "LaPy not installed. Install with: pip install lapy"
#     )


def read_obj_file(filepath):
    """
    Read OBJ file and extract vertices and faces as lists of lists.

    Args:
        filepath (str): Path to the OBJ file

    Returns:
        tuple: (vertices, faces) as lists of lists
    """
    vertices = []
    faces = []
    mesh = trimesh.load_mesh(str(filepath))
    for v in mesh.vertices:
        vertices.append(v.tolist())
    for f in mesh.faces:
        faces.append(f.tolist())

    # normalize vertices
    vertices = np.array(vertices)
    vertices_norm = (vertices - vertices.max()) / (vertices.max() - vertices.min())
    # convert to list of list
    vertices = []
    N = vertices_norm.shape[0]
    for i in range(N):
        row = []
        for j in range(3):
            row.append(vertices_norm[i, j])
        vertices.append(row)



    return vertices, faces


def create_lapy_mesh(vertices, faces):
    """
    Create a LaPy TriaMesh from vertices and faces (as lists of lists).

    Args:
        vertices (list of lists): List of [x, y, z] coordinates
        faces (list of lists): List of [v0, v1, v2] face indices

    Returns:
        lapy.TriaMesh: The triangular mesh
    """
    # Pass as lists directly - TriaMesh will convert to arrays and validate
    mesh = lapy.TriaMesh(vertices, faces)

    return mesh


def compute_spherical_parameterization(mesh):
    """
    Compute spherical conformal parameterization of the mesh.

    Args:
        mesh (lapy.TriaMesh): Input triangular mesh

    Returns:
        np.ndarray: Nx3 array of vertices on the unit sphere
    """
    try:
        print("Computing spherical conformal parameterization...")

        # Compute the spherical conformal map
        # This requires a genus-0 closed surface
        spherical_vertices = spherical_conformal_map(mesh)

        print(f"✓ Parameterization complete. Vertices shape: {spherical_vertices.shape}")

        return spherical_vertices

    except Exception as e:
        print(f"✗ Error computing parameterization: {e}")
        raise


def plot_meshes(mesh, parameterized_vertices):
    """
    Plot original mesh and parameterized mesh side by side.

    Args:
        mesh (lapy.TriaMesh): Original mesh
        parameterized_vertices (np.ndarray): Vertices on the sphere
    """
    fig = plt.figure(figsize=(16, 6))

    # Get original vertices and faces from mesh
    original_vertices = mesh.v
    faces = mesh.t

    # Plot 1: Original mesh
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(
        original_vertices[:, 0],
        original_vertices[:, 1],
        original_vertices[:, 2],
        triangles=faces,
        alpha=0.7,
        edgecolor='k',
        linewidth=0.3
    )
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Mesh')
    ax1.view_init(elev=20, azim=45)

    # Plot 2: Parameterized mesh (spherical)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(
        parameterized_vertices[:, 0],
        parameterized_vertices[:, 1],
        parameterized_vertices[:, 2],
        triangles=faces,
        alpha=0.7,
        edgecolor='k',
        linewidth=0.3,
        cmap='viridis'
    )
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Spherical Conformal Parameterization')
    ax2.view_init(elev=20, azim=45)

    # Plot 3: Sphere radius verification
    ax3 = fig.add_subplot(133)
    radii = np.linalg.norm(parameterized_vertices, axis=1)
    # ax3.hist(radii, bins=min(50, len(np.unique(radii))), edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(radii), color='r', linestyle='--', label=f'Mean: {np.mean(radii):.6f}')
    ax3.set_xlabel('Radius')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Sphere Radius Distribution\n(should be ≈ 1.0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compute spherical conformal parameterization of a mesh from OBJ file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mesh.obj
  %(prog)s --input data/brain.obj --verbose
  %(prog)s -i mesh.obj -v
        """
    )

    parser.add_argument(
        'input',
        nargs='?',
        help='Path to the OBJ file'
    )
    parser.add_argument(
        '-i', '--input',
        dest='input_file',
        help='Alternative way to specify input OBJ file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip visualization'
    )

    args = parser.parse_args()

    # Resolve input file
    input_file = args.input_file or args.input

    if not input_file:
        parser.print_help()
        print("\n✗ Error: Please specify an OBJ file")
        return 1

    # Check if file exists
    if not Path(input_file).exists():
        print(f"✗ Error: File not found: {input_file}")
        return 1

    try:
        if args.verbose:
            print(f"📂 Reading OBJ file: {input_file}")

        # Read OBJ file as lists of lists
        vertices, faces = read_obj_file(input_file)

        if args.verbose:
            print(f"   Vertices: {len(vertices)}")
            print(f"   Faces: {len(faces)}")

        # Create LaPy mesh
        if args.verbose:
            print("🔧 Creating LaPy mesh...")

        mesh = create_lapy_mesh(vertices, faces)

        if args.verbose:
            print(f"   Mesh vertices shape: {mesh.v.shape}")
            print(f"   Mesh faces shape: {mesh.t.shape}")
            print(f"   Mesh is closed: {mesh.is_closed()}")
            print(f"   Mesh is manifold: {mesh.is_manifold()}")
            print(f"   Mesh is oriented: {mesh.is_oriented()}")
            euler = mesh.euler()
            print(f"   Euler characteristic: {euler}")

        # Compute parameterization
        print("\n🧮 Computing spherical conformal parameterization...")
        parameterized = compute_spherical_parameterization(mesh)

        # Verify parameterization
        radii = np.linalg.norm(parameterized, axis=1)
        if args.verbose:
            print(f"   Mean radius: {np.mean(radii):.6f}")
            print(f"   Radius std: {np.std(radii):.6f}")
            print(f"   Radius range: [{np.min(radii):.6f}, {np.max(radii):.6f}]")

        # Plot results
        if not args.no_plot:
            print("\n📊 Plotting results...")
            plot_meshes(mesh, parameterized)

        print("\n✓ Done!")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())