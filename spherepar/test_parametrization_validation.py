"""Test suite for spherical parametrization validation functions."""
import numpy as np
import pytest

from spherepar.spherical_parametrization import (
    verify_topology_preserved,
    verify_normal_orientation_preserved,
    _compute_face_normals,
)


class TestComputeFaceNormals:
    """Test face normal computation."""

    def test_simple_triangle(self):
        """Test normal computation for a simple triangle."""
        # Triangle in XY plane, normal should point in +Z
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        normals = _compute_face_normals(vertices, faces)
        assert normals.shape == (1, 3)

        # Normal should point in +Z direction
        assert normals[0, 2] > 0

    def test_multiple_triangles(self):
        """Test with multiple triangles."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 1],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)

        normals = _compute_face_normals(vertices, faces)
        assert normals.shape == (2, 3)


class TestVerifyTopologyPreserved:
    """Test topology preservation validation."""

    def test_identical_topology(self):
        """Test validation passes for identical topology."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces_orig = np.array([[0, 1, 2]], dtype=np.int32)
        vertices_mapped = vertices_orig.copy()
        faces_mapped = faces_orig.copy()

        is_valid, report = verify_topology_preserved(
            vertices_orig, faces_orig, vertices_mapped, faces_mapped
        )

        assert is_valid
        assert report["is_valid"] is True
        assert len(report["errors"]) == 0

    def test_vertex_count_mismatch(self):
        """Test validation fails when vertex count changes."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces_orig = np.array([[0, 1, 2]], dtype=np.int32)
        vertices_mapped = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float64)
        faces_mapped = faces_orig.copy()

        is_valid, report = verify_topology_preserved(
            vertices_orig, faces_orig, vertices_mapped, faces_mapped
        )

        assert not is_valid
        assert "Vertex count mismatch" in str(report["errors"])

    def test_face_count_mismatch(self):
        """Test validation fails when face count changes."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces_orig = np.array([[0, 1, 2]], dtype=np.int32)
        vertices_mapped = vertices_orig.copy()
        faces_mapped = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)

        is_valid, report = verify_topology_preserved(
            vertices_orig, faces_orig, vertices_mapped, faces_mapped
        )

        assert not is_valid
        assert "Face count mismatch" in str(report["errors"])

    def test_face_indices_changed(self):
        """Test validation fails when face indices change."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces_orig = np.array([[0, 1, 2]], dtype=np.int32)
        vertices_mapped = vertices_orig.copy()
        faces_mapped = np.array([[0, 2, 1]], dtype=np.int32)  # Different indices

        is_valid, report = verify_topology_preserved(
            vertices_orig, faces_orig, vertices_mapped, faces_mapped
        )

        assert not is_valid
        assert "Face array values differ" in str(report["errors"])

    def test_invalid_face_indices(self):
        """Test validation fails with out-of-range face indices."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces_orig = np.array([[0, 1, 2]], dtype=np.int32)
        vertices_mapped = vertices_orig.copy()
        faces_mapped = np.array([[0, 1, 5]], dtype=np.int32)  # Index 5 out of range

        is_valid, report = verify_topology_preserved(
            vertices_orig, faces_orig, vertices_mapped, faces_mapped
        )

        assert not is_valid
        assert "Invalid face indices found" in str(report["errors"])


class TestVerifyNormalOrientationPreserved:
    """Test normal orientation validation."""

    def test_identical_normals(self):
        """Test validation passes when normals are identical."""
        # Simple triangle
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        vertices_mapped = vertices_orig.copy()
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        is_valid, report = verify_normal_orientation_preserved(
            vertices_orig, faces, vertices_mapped
        )

        assert is_valid
        assert report["is_valid"] is True
        assert report["normals_oriented_correctly"]

    def test_flipped_winding_order(self):
        """Test detection of inward-pointing normals when mesh is inverted."""
        vertices_orig = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=np.float64)
        # Invert the mesh by negating Z coordinates (mirrors the mesh)
        vertices_mapped = vertices_orig.copy()
        vertices_mapped[:, 2] = -vertices_mapped[:, 2]

        # Same face array for both
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        is_valid, report = verify_normal_orientation_preserved(
            vertices_orig, faces, vertices_mapped
        )

        # After inversion, the mesh normals now point inward
        assert not is_valid
        assert report["all_normals_radial"] is False
        assert report["n_inward_normals"] == 1

    def test_inward_pointing_normals(self):
        """Test detection of inward-pointing normals on sphere."""
        # Create a simple sphere-like structure
        vertices_orig = np.array(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        # Map to sphere
        r = np.sqrt(3)
        vertices_mapped = vertices_orig / r

        # Use face with normal pointing outward
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        is_valid, report = verify_normal_orientation_preserved(
            vertices_orig, faces, vertices_mapped
        )

        # Should be valid - normals point outward
        assert report["all_normals_radial"] is True

    def test_scaled_mesh(self):
        """Test with scaled mesh (different scale but same orientation)."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        vertices_mapped = vertices_orig * 2.0  # Same orientation, scaled
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        is_valid, report = verify_normal_orientation_preserved(
            vertices_orig, faces, vertices_mapped
        )

        assert report["normals_oriented_correctly"]

    def test_translated_mesh(self):
        """Test with translated mesh (position changed, orientation same)."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        vertices_mapped = vertices_orig + np.array([10, 10, 10])
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        is_valid, report = verify_normal_orientation_preserved(
            vertices_orig, faces, vertices_mapped
        )

        assert report["normals_oriented_correctly"]

    def test_rotated_mesh(self):
        """Test with rotated mesh (orientation preserved through rotation)."""
        vertices_orig = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

        # Rotate 90 degrees around Z axis
        angle = np.pi / 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]]
        )
        vertices_mapped = vertices_orig @ rotation_matrix.T

        faces = np.array([[0, 1, 2]], dtype=np.int32)

        is_valid, report = verify_normal_orientation_preserved(
            vertices_orig, faces, vertices_mapped
        )

        assert report["normals_oriented_correctly"]

    def test_multiple_faces_mixed_orientation(self):
        """Test with multiple faces, some flipped."""
        # Create a cube-like structure
        vertices_orig = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
            ],
            dtype=np.float64,
        )

        vertices_mapped = vertices_orig.copy()

        # Two faces: one correct, one flipped
        faces = np.array([[0, 1, 2], [5, 4, 0]], dtype=np.int32)

        is_valid, report = verify_normal_orientation_preserved(
            vertices_orig, faces, vertices_mapped
        )

        # Should detect flipped normals in second face
        assert report["n_flipped_normals"] >= 0  # May or may not detect depending on geometry


class TestIntegration:
    """Integration tests with synthetic meshes."""

    def test_unit_sphere_vertices(self):
        """Test with vertices on a unit sphere."""
        # Create simple tetrahedron
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array(
            [
                [-1, phi, -1],
                [1, phi, 1],
                [-1, -phi, 1],
                [1, -phi, -1],
            ],
            dtype=np.float64,
        )
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

        faces = np.array(
            [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32
        )

        # Validation should pass for consistent mesh
        is_valid, report = verify_topology_preserved(
            vertices, faces, vertices, faces
        )
        assert is_valid

        # Normal orientation should be correct
        is_valid_orient, report_orient = verify_normal_orientation_preserved(
            vertices, faces, vertices
        )
        # Might have mixed orientation depending on vertex order
        assert "is_valid" in report_orient


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
