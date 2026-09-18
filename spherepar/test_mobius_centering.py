from __future__ import annotations

import numpy as np
import pytest
import trimesh

from spherepar.mobius_centering import (
    apply_inverse_mobius_sequence,
    apply_mobius_sequence,
    area_weighted_centroid,
    center_spherical_mesh,
    mobius_transform,
    original_face_weights,
    transform_to_json,
)
from spherepar.spherical_parametrization import compute_spherical_parametrization
from spherepar.cem_parametrization import stretch_parametrization
from spherepar.mesh import MeshFactory


def test_incremental_maps_preserve_sphere_and_composed_inverse_round_trip():
    rng = np.random.default_rng(7)
    points = rng.normal(size=(128, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    centers = np.asarray([[0.2, -0.1, 0.05], [-0.12, 0.16, 0.08], [0.03, 0.02, -0.2]])

    current = points
    for center in centers:
        current = mobius_transform(current, center)
        np.testing.assert_allclose(np.linalg.norm(current, axis=1), 1.0, atol=1e-12, rtol=0.0)

    transform = transform_to_json(centers)
    mapped = apply_mobius_sequence(points, transform)
    restored = apply_inverse_mobius_sequence(mapped, transform)
    np.testing.assert_allclose(restored, points, atol=1e-10, rtol=0.0)


def test_weighted_centering_uses_nonuniform_original_face_areas():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    unit_vertices = np.asarray(sphere.vertices, dtype=np.float64)
    faces = np.asarray(sphere.faces, dtype=np.int64)
    original_vertices = unit_vertices * np.asarray([3.0, 0.8, 1.4])[None, :]
    displaced = mobius_transform(unit_vertices, [0.35, -0.12, 0.08])
    weights = original_face_weights(original_vertices, faces)
    before = np.linalg.norm(area_weighted_centroid(displaced, faces, weights))

    centered, metadata = center_spherical_mesh(original_vertices, faces, displaced)
    after = np.linalg.norm(area_weighted_centroid(centered, faces, weights))

    assert np.ptp(weights) > 0.0
    assert after < 1e-10
    assert after < before
    reconstructed = apply_inverse_mobius_sequence(centered, metadata["transform"])
    np.testing.assert_allclose(reconstructed, displaced, atol=1e-12, rtol=0.0)


def test_centering_fails_explicitly_when_iteration_budget_is_exhausted():
    sphere = trimesh.creation.icosphere(subdivisions=1)
    displaced = mobius_transform(sphere.vertices, [0.3, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="did not converge"):
        center_spherical_mesh(sphere.vertices, sphere.faces, displaced, max_iterations=0)


def test_centering_is_cem_only():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    with pytest.raises(ValueError, match="only with method='cem'"):
        compute_spherical_parametrization(
            mesh.vertices, mesh.faces, method="flash", mobius_center=True, verify=False
        )


def test_cem_postprocessing_preserves_base_result_and_saved_sequence():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    direct = stretch_parametrization(
        MeshFactory.make_mesh("surf", mesh.vertices, mesh.faces),
        eps=1e-6,
        max_iters=30,
        verbose=False,
    ).convert_mesh().get_vertices_collection()
    uncentered, uncentered_meta = compute_spherical_parametrization(
        mesh.vertices, mesh.faces, method="cem", cem_max_iters=30, verify=False
    )
    centered, centered_meta = compute_spherical_parametrization(
        mesh.vertices, mesh.faces, method="cem", cem_max_iters=30,
        mobius_center=True, verify=False,
    )

    np.testing.assert_array_equal(uncentered, direct)
    assert uncentered_meta["mobius_center"] is False
    centering = centered_meta["mobius_centering"]
    assert centering["after"]["centroid_norm"] < centering["before"]["centroid_norm"]
    np.testing.assert_allclose(
        apply_inverse_mobius_sequence(centered, centering["transform"]),
        uncentered,
        atol=1e-12,
        rtol=0.0,
    )


def test_mobius_differential_is_conformal_on_tangent_plane():
    point = np.asarray([0.3, -0.4, np.sqrt(0.75)])
    point /= np.linalg.norm(point)
    center = np.asarray([0.21, -0.08, 0.12])
    tangent_1 = np.cross(point, [0.0, 0.0, 1.0])
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(point, tangent_1)
    eps = 1e-7

    def sphere_step(tangent):
        return np.cos(eps) * point + np.sin(eps) * tangent

    image = mobius_transform(point, center)
    differential = []
    for tangent in (tangent_1, tangent_2):
        image_step = mobius_transform(sphere_step(tangent), center)
        derivative = (image_step - image) / eps
        derivative -= derivative.dot(image) * image
        differential.append(derivative)
    gram = np.asarray([[a.dot(b) for b in differential] for a in differential])
    assert abs(gram[0, 1]) < 1e-7
    np.testing.assert_allclose(gram[0, 0], gram[1, 1], atol=1e-7, rtol=1e-7)
