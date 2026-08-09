from __future__ import annotations

import numpy as np

from spherepar.benchmark.signals import (
    _stable_tangent_basis,
    anisotropic_gaussian_from_axis,
    compute_gauge_relative_angle_and_target,
    fixed_gauge_tangent_basis,
    rotate_axis_and_target,
    sample_hm_major_axis,
)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-12:
        raise ValueError("zero vector")
    return v / n


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    a = _unit(np.asarray(axis, dtype=float))
    x, y, z = a
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


def _exp_map(center: np.ndarray, tangent_dir: np.ndarray, dist: float) -> np.ndarray:
    c = _unit(np.asarray(center, dtype=float))
    t = _unit(np.asarray(tangent_dir, dtype=float))
    return _unit(np.cos(dist) * c + np.sin(dist) * t)


def test_hm_basis_is_orthonormal_and_tangent():
    c = _unit(np.array([0.2, -0.7, 0.6], dtype=float))
    e1, e2 = _stable_tangent_basis(c)
    assert np.isclose(np.dot(e1, c), 0.0, atol=1e-8)
    assert np.isclose(np.dot(e2, c), 0.0, atol=1e-8)
    assert np.isclose(np.dot(e1, e2), 0.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(e1), 1.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(e2), 1.0, atol=1e-8)
    assert np.isclose(np.dot(np.cross(e1, e2), c), 1.0, atol=1e-8)


def test_sampled_major_axis_is_unit_and_tangent():
    c = _unit(np.array([0.4, 0.3, 0.85], dtype=float))
    sampled = sample_hm_major_axis(c, rng=np.random.default_rng(123), delta=0.9)
    v = sampled["major_axis"]
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-8)
    assert np.isclose(np.dot(v, sampled["center"]), 0.0, atol=1e-8)


def test_fixed_gauge_basis_is_orthonormal_and_tangent():
    c = _unit(np.array([-0.4, 0.7, 0.5], dtype=float))
    g1, g2 = fixed_gauge_tangent_basis(c, eps=1e-8)
    assert np.isclose(np.dot(g1, c), 0.0, atol=1e-8)
    assert np.isclose(np.dot(g2, c), 0.0, atol=1e-8)
    assert np.isclose(np.dot(g1, g2), 0.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(g1), 1.0, atol=1e-8)
    assert np.isclose(np.linalg.norm(g2), 1.0, atol=1e-8)


def test_doubled_angle_target_has_unit_norm_and_is_sign_invariant():
    c = _unit(np.array([0.1, 0.8, 0.55], dtype=float))
    sampled = sample_hm_major_axis(c, rng=np.random.default_rng(7), delta=1.1)
    v = sampled["major_axis"]
    t = sampled["target"]
    assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-8)
    flipped = compute_gauge_relative_angle_and_target(c, -v)
    assert np.allclose(t, flipped["target"], atol=1e-8)


def test_rotation_updates_physical_axis_and_target_consistently():
    c = _unit(np.array([0.33, -0.41, 0.85], dtype=float))
    sampled = sample_hm_major_axis(c, rng=np.random.default_rng(5), delta=0.42)
    v = sampled["major_axis"]

    R = _rotation_matrix(axis=np.array([0.3, 0.4, 0.2]), angle=0.73)
    rotated = rotate_axis_and_target(R, c, v)

    expected_c = _unit(R @ c)
    expected_v = _unit(R @ v)
    assert np.allclose(rotated["center_rotated"], expected_c, atol=1e-8)
    assert np.allclose(rotated["major_axis_rotated"], expected_v, atol=1e-8)

    recomputed = compute_gauge_relative_angle_and_target(expected_c, expected_v)
    assert np.allclose(rotated["target_rotated"], recomputed["target"], atol=1e-8)


def test_centers_near_gauge_poles_are_rejected():
    with np.testing.assert_raises(ValueError):
        fixed_gauge_tangent_basis(np.array([0.0, 0.0, 1.0]), eps=1e-8)
    with np.testing.assert_raises(ValueError):
        sample_hm_major_axis(np.array([0.0, 0.0, -1.0]), rng=np.random.default_rng(1), eps=1e-8)
    near_pole = _unit(np.array([1e-4, 1e-4, 1.0], dtype=float))
    with np.testing.assert_raises(ValueError):
        sample_hm_major_axis(
            near_pole,
            rng=np.random.default_rng(1),
            eps=1e-8,
            min_gauge_projection_norm=0.05,
        )


def test_anisotropic_gaussian_is_invariant_under_axis_sign():
    c = _unit(np.array([0.2, -0.3, 0.93], dtype=float))
    sampled = sample_hm_major_axis(c, rng=np.random.default_rng(3), delta=0.66)
    v = sampled["major_axis"]
    vertices = np.array([
        _exp_map(c, v, 0.0),
        _exp_map(c, v, 0.2),
        _exp_map(c, -v, 0.2),
        _exp_map(c, np.cross(c, v), 0.2),
    ])
    sig_pos = anisotropic_gaussian_from_axis(vertices, c, v, sigma_parallel=0.4, sigma_perpendicular=0.2)
    sig_neg = anisotropic_gaussian_from_axis(vertices, c, -v, sigma_parallel=0.4, sigma_perpendicular=0.2)
    assert np.allclose(sig_pos, sig_neg, atol=1e-10)


def test_anisotropic_gaussian_elongates_along_major_axis():
    c = _unit(np.array([0.5, 0.2, 0.84], dtype=float))
    sampled = sample_hm_major_axis(c, rng=np.random.default_rng(11), delta=0.35)
    v = sampled["major_axis"]
    v_perp = _unit(np.cross(c, v))
    d = 0.25
    y_parallel = _exp_map(c, v, d)
    y_perp = _exp_map(c, v_perp, d)
    sig = anisotropic_gaussian_from_axis(
        np.vstack([y_parallel, y_perp]),
        c,
        v,
        sigma_parallel=0.45,
        sigma_perpendicular=0.15,
    )
    assert sig[0] > sig[1]
