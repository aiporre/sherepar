"""Validation tests for the FLASH implementation and failure paths."""

from __future__ import annotations

import numpy as np
import trimesh

import spherepar.flash_parametrization as flash
from spherepar.spherical_parametrization import compute_spherical_parametrization


def test_flash_regular_icosphere_is_a_valid_unit_sphere():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mapped, diagnostics = flash.flash_map_with_diagnostics(mesh)

    assert diagnostics["success"] is True
    assert diagnostics["tutte_fallback_used"] is False
    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=1e-12, rtol=0.0)
    assert np.isfinite(mapped).all()


def test_spherical_tutte_fallback_is_finite_and_unit_length():
    mesh = trimesh.creation.icosphere(subdivisions=1)
    mapped = flash._spherical_tutte_map(mesh.faces, 0, len(mesh.vertices))

    assert mapped.shape == mesh.vertices.shape
    assert np.isfinite(mapped).all()
    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_flash_uses_tutte_fallback_after_harmonic_failure(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    original_spsolve = flash.spsolve
    calls = {"count": 0}

    def fail_first_solve(matrix, rhs):
        calls["count"] += 1
        if calls["count"] == 1:
            return np.full(np.asarray(rhs).shape, np.nan + 0.0j)
        return original_spsolve(matrix, rhs)

    monkeypatch.setattr(flash, "spsolve", fail_first_solve)
    mapped, diagnostics = flash.flash_map_with_diagnostics(mesh)

    assert calls["count"] >= 2
    assert diagnostics["tutte_fallback_used"] is True
    assert diagnostics["north_stage"] == "tutte"
    assert np.isfinite(mapped).all()
    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_flash_solver_failure_retains_harmonic_sphere_and_reports_failure(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=1)

    def failed_solver(*args, **kwargs):
        vertex_count = len(args[0])
        return np.full((vertex_count, 2), np.nan)

    monkeypatch.setattr(flash, "_linear_beltrami_solver", failed_solver)
    mapped, diagnostics = flash.flash_map_with_diagnostics(mesh)

    assert diagnostics["success"] is False
    assert diagnostics["retained_stage"] == "harmonic"
    assert diagnostics["solver_attempts"]
    assert "failed" in diagnostics["error"]
    assert np.isfinite(mapped).all()
    np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_flash_failure_is_structured_by_spherical_wrapper(monkeypatch):
    mesh = trimesh.creation.icosphere(subdivisions=1)

    def failed_solver(*args, **kwargs):
        return np.full((len(args[0]), 2), np.nan)

    monkeypatch.setattr(flash, "_linear_beltrami_solver", failed_solver)
    mapped, metadata = compute_spherical_parametrization(
        mesh.vertices,
        mesh.faces,
        method="flash",
        verify=True,
    )

    assert mapped.shape == mesh.vertices.shape
    assert metadata["success"] is False
    assert metadata["error"]
    assert metadata["flash_diagnostics"]["success"] is False
    assert metadata["sphere_validation"]["is_valid"] is True
