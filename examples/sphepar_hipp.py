"""
Example script demonstrating spherical parametrization on the hippocampus mesh.

This script reads the hippocampus mesh, applies spherical conformal parametrization,
and visualizes both the original mesh and the parametrized mesh on a sphere.
"""

import numpy as np
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt

# Import spherepar modules
from spherepar.mesh import MeshFactory, plot_mesh
from spherepar.cem_parametrization import dirichlet_parametrization, stretch_parametrization


def load_mesh_from_file(mesh_path):
    """
    Load a mesh from a file (OBJ, STL, PLY, etc.) using trimesh.
    
    Parameters
    ----------
    mesh_path : str or Path
        Path to the mesh file
        
    Returns
    -------
    MeshSurf
        A surface mesh object compatible with spherepar
    """
    # Load mesh using trimesh
    mesh_data = trimesh.load_mesh(str(mesh_path))
    
    # Extract vertices and faces
    vertices = mesh_data.vertices
    faces = mesh_data.faces
    
    # Create a MeshSurf object
    mesh = MeshFactory.make_mesh('surf', vertices, faces)
    
    print(f"Loaded mesh from {mesh_path}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    print(f"  Edges: {len(mesh.edges)}")
    
    return mesh


def apply_dirichlet_parametrization(mesh):
    """
    Apply Dirichlet (conformal) parametrization to map the mesh to a sphere.
    
    This is Algorithm 4.1 from the paper: "A Novel Algorithm for 
    Volume-Preserving Parameterizations of 3-Manifolds"
    
    Parameters
    ----------
    mesh : MeshSurf
        Input surface mesh
        
    Returns
    -------
    StretchFunction
        A stretch function that maps vertices to sphere points
    """
    print("\nApplying Dirichlet parametrization...")
    stretch_func = dirichlet_parametrization(mesh)
    print("✓ Dirichlet parametrization complete")
    
    return stretch_func


def apply_stretch_parametrization(mesh, eps=1e-6, max_iters=1000, verbose=False):
    """
    Apply CEM (Conformally-Exact-Map) iteration to refine the spherical parametrization.
    
    This is Algorithm 4.2 from the paper and builds upon the Dirichlet parametrization
    to minimize the Dirichlet energy on the sphere.
    
    Parameters
    ----------
    mesh : MeshSurf
        Input surface mesh
    eps : float, optional
        Convergence threshold for Dirichlet energy improvement (default: 1e-6)
    max_iters : int, optional
        Maximum number of CEM iterations (default: 1000)
    verbose : bool, optional
        Print iteration details (default: False)
        
    Returns
    -------
    StretchFunction
        A stretch function that maps vertices to sphere points
    """
    print("\nApplying stretch parametrization (CEM iteration)...")
    stretch_func = stretch_parametrization(mesh, eps=eps, max_iters=max_iters, verbose=verbose)
    print("✓ Stretch parametrization complete")
    
    return stretch_func


def get_parametrized_mesh(stretch_func):
    """
    Convert the parametrization to a mesh on the sphere.
    
    Parameters
    ----------
    stretch_func : StretchFunction
        The parametrization function
        
    Returns
    -------
    MeshSurf
        The mesh mapped to the unit sphere
    """
    return stretch_func.convert_mesh()


def visualize_meshes(original_mesh, parametrized_mesh, title="Spherical Parametrization"):
    """
    Visualize both the original mesh and its parametrization on a sphere.
    
    Parameters
    ----------
    original_mesh : MeshSurf
        The original mesh
    parametrized_mesh : MeshSurf
        The mesh parametrized on the sphere
    title : str, optional
        Title for the figure
    """
    fig = plt.figure(figsize=(16, 7))
    
    # Plot original mesh
    ax1 = fig.add_subplot(121, projection='3d')
    original_data = (original_mesh.get_vertices_collection(), 
                     original_mesh.get_faces_collection(), None, None)
    plot_mesh(original_data, ax=ax1)
    ax1.set_title("Original Mesh")
    
    # Plot parametrized mesh
    ax2 = fig.add_subplot(122, projection='3d')
    parametrized_data = (parametrized_mesh.get_vertices_collection(), 
                         parametrized_mesh.get_faces_collection(), None, None)
    plot_mesh(parametrized_data, ax=ax2)
    ax2.set_title("Spherical Parametrization")
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def compute_mesh_statistics(original_mesh, parametrized_mesh):
    """
    Compute and print statistics about the parametrization.
    
    Parameters
    ----------
    original_mesh : MeshSurf
        The original mesh
    parametrized_mesh : MeshSurf
        The mesh parametrized on the sphere
    """
    # Original mesh statistics
    orig_verts = original_mesh.get_vertices_collection()
    orig_center = np.mean(orig_verts, axis=0)
    orig_scale = np.linalg.norm(orig_verts - orig_center, axis=1).max()
    
    # Parametrized mesh statistics (should be on unit sphere)
    param_verts = parametrized_mesh.get_vertices_collection()
    param_norms = np.linalg.norm(param_verts, axis=1)
    
    print("\n" + "="*60)
    print("MESH STATISTICS")
    print("="*60)
    print(f"Original Mesh:")
    print(f"  Center: {orig_center}")
    print(f"  Bounding sphere radius: {orig_scale:.6f}")
    print(f"  Vertices: {len(orig_verts)}")
    print(f"  Faces: {len(original_mesh.faces)}")
    
    print(f"\nParametrized Mesh (on sphere):")
    print(f"  Vertex norms (distance from origin):")
    print(f"    Min: {param_norms.min():.6f}")
    print(f"    Max: {param_norms.max():.6f}")
    print(f"    Mean: {param_norms.mean():.6f}")
    print(f"    Std: {param_norms.std():.6f}")
    print(f"  Expected: all vertices should have norm ≈ 1.0")
    print("="*60)


def main():
    """
    Main example: Load the hippocampus mesh and apply spherical parametrization.
    """
    # Define path to hippocampus mesh
    data_dir = Path(__file__).parent.parent / "data"
    mesh_path = data_dir / "hipp_left.obj"
    
    # Check if file exists
    if not mesh_path.exists():
        print(f"Error: Mesh file not found at {mesh_path}")
        print("Please ensure the data/hipp_left.obj file exists in the repository.")
        return
    
    print("="*60)
    print("SPHERICAL PARAMETRIZATION EXAMPLE")
    print("="*60)
    
    # Step 1: Load the mesh
    print("\nStep 1: Loading mesh...")
    original_mesh = load_mesh_from_file(mesh_path)
    
    # Step 2: Apply Dirichlet parametrization (Algorithm 4.1)
    print("\nStep 2: Applying Dirichlet parametrization...")
    dirichlet_stretch = apply_dirichlet_parametrization(original_mesh)
    dirichlet_mesh = get_parametrized_mesh(dirichlet_stretch)
    
    # Step 3 (Optional): Apply stretch parametrization (Algorithm 4.2) for refinement
    # Note: This takes longer but produces better results
    print("\nStep 3: Applying stretch parametrization for refinement...")
    print("  (This may take several minutes for large meshes)")
    stretch_func = apply_stretch_parametrization(original_mesh, eps=1e-6, max_iters=100, verbose=True)
    parametrized_mesh = get_parametrized_mesh(stretch_func)
    
    # Step 4: Compute statistics
    compute_mesh_statistics(original_mesh, parametrized_mesh)
    
    # Step 5: Visualize
    print("\nStep 5: Visualizing results...")
    fig1 = visualize_meshes(original_mesh, dirichlet_mesh, 
                            "Dirichlet Parametrization (Algorithm 4.1)")
    fig2 = visualize_meshes(original_mesh, parametrized_mesh, 
                            "CEM Parametrization (Algorithm 4.2)")
    
    plt.show()


if __name__ == "__main__":
    main()

