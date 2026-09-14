#!/usr/bin/env python3
"""Visualize Sherepar's spherical log-map plane at a mesh vertex.

Example
-------
python examples/plot_tangent_log_plane.py meshes/sample.obj 42 --output tangent.png

The left panel shows the generated per-vertex anisotropic signal on the mesh,
the origin-centred sphere through the selected vertex, and the tangent plane.
The right panel shows the same vertices in their spherical log-map coordinates.
The script follows Sherepar's signal construction: unit-sphere normalization,
log map at the center, Hughes--Möller basis, sampled angle ``delta``, then
anisotropic-Gaussian evaluation along ``v`` and its perpendicular direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Make ``python examples/plot_tangent_log_plane.py ...`` work from any cwd.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spherepar.benchmark.signals import (
    _sphere_log_map,
    _stable_tangent_basis,
    anisotropic_gaussian_from_axis,
)
from spherepar.flash_parametrization import load_mesh_with_trimesh


def _set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    """Set equal 3-D limits that contain all supplied points."""
    low = points.min(axis=0)
    high = points.max(axis=0)
    midpoint = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-8)
    ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    ax.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="Input OBJ mesh.")
    parser.add_argument("vertex_index", type=int, help="Zero-based selected vertex index.")
    parser.add_argument(
        "--plane-size",
        type=float,
        default=0.35,
        help="Tangent-plane half-width as a fraction of the reference-sphere radius (default: 0.35).",
    )
    parser.add_argument(
        "--log-radius",
        type=float,
        default=0.6,
        help="Maximum spherical distance, in radians, shown in the log-map panel (default: 0.6).",
    )
    parser.add_argument(
        "--sigma-parallel",
        type=float,
        default=0.30,
        help="Gaussian width along the orange major-axis arrow v, in radians (default: 0.30).",
    )
    parser.add_argument(
        "--sigma-perpendicular",
        type=float,
        default=0.15,
        help="Gaussian width perpendicular to v, in radians (default: 0.15).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="Angle from Hughes--Möller e1 to major axis v, in radians and modulo pi (default: 0).",
    )
    parser.add_argument("--amplitude", type=float, default=1.0, help="Gaussian peak amplitude (default: 1).")
    parser.add_argument("--output", type=Path, help="Write the figure to this file instead of opening it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.plane_size, args.log_radius, args.sigma_parallel, args.sigma_perpendicular, args.amplitude) <= 0.0:
        raise ValueError(
            "--plane-size, --log-radius, --sigma-parallel, --sigma-perpendicular, and --amplitude must be positive"
        )

    mesh = load_mesh_with_trimesh(str(args.mesh))
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if not 0 <= args.vertex_index < len(vertices):
        raise IndexError(f"vertex_index must be in [0, {len(vertices) - 1}], got {args.vertex_index}")

    center = vertices[args.vertex_index]
    sphere_radius = float(np.linalg.norm(center))
    if sphere_radius <= 1e-8:
        raise ValueError("Selected vertex is at the origin; it cannot define a sphere/log-map center")

    # These calls intentionally use Sherepar's anisotropic-Gaussian geometry.
    # The Hughes--Möller frame supplies coordinates for the tangent/log-map
    # plane without requiring a valid fixed-gauge orientation target.
    center_unit = center / sphere_radius
    marker_center = center + 0.01 * sphere_radius * center_unit
    e1, e2 = _stable_tangent_basis(center_unit)
    delta = float(np.mod(args.delta, np.pi))
    v = np.cos(delta) * e1 + np.sin(delta) * e2
    v_perp = np.cross(center_unit, v)
    log_vectors = _sphere_log_map(center, vertices)
    log_xy = np.column_stack((log_vectors @ e1, log_vectors @ e2))
    angular_distance = np.linalg.norm(log_vectors, axis=1)
    gaussian_at_vertices = anisotropic_gaussian_from_axis(
        vertices,
        center,
        major_axis=v,
        sigma_parallel=args.sigma_parallel,
        sigma_perpendicular=args.sigma_perpendicular,
        amplitude=args.amplitude,
    )

    # Draw the tangent plane in ambient coordinates.  Scaling log coordinates
    # by the sphere radius makes it tangent to the displayed sphere at `center`.
    half_width = args.plane_size * sphere_radius
    grid = np.linspace(-half_width, half_width, 80)
    uu, vv = np.meshgrid(grid, grid)
    plane = center[None, None, :] + uu[..., None] * e1 + vv[..., None] * e2
    # uu/vv are displayed in ambient XYZ units; divide by the reference radius
    # to recover the angular tangent/log-map coordinates used by Sherepar.
    plane_log_vectors = (uu[..., None] * e1 + vv[..., None] * e2) / sphere_radius
    plane_parallel = np.sum(plane_log_vectors * v, axis=-1)
    plane_perpendicular = np.sum(plane_log_vectors * v_perp, axis=-1)
    plane_gaussian = args.amplitude * np.exp(
        -0.5
        * (
            (plane_parallel / args.sigma_parallel) ** 2
            + (plane_perpendicular / args.sigma_perpendicular) ** 2
        )
    )
    plane_colors = plt.get_cmap("viridis")(plane_gaussian / args.amplitude)

    azimuth = np.linspace(0.0, 2.0 * np.pi, 48)
    polar = np.linspace(0.0, np.pi, 25)
    azimuth_grid, polar_grid = np.meshgrid(azimuth, polar)
    sphere_x = sphere_radius * np.sin(polar_grid) * np.cos(azimuth_grid)
    sphere_y = sphere_radius * np.sin(polar_grid) * np.sin(azimuth_grid)
    sphere_z = sphere_radius * np.cos(polar_grid)

    figure = plt.figure(figsize=(14, 7), constrained_layout=True)
    ax_3d = figure.add_subplot(1, 2, 1, projection="3d")
    ax_log = figure.add_subplot(1, 2, 2)

    face_signal = gaussian_at_vertices[faces].mean(axis=1)
    mesh_collection = Poly3DCollection(
        vertices[faces],
        facecolors=plt.get_cmap("viridis")(face_signal / args.amplitude),
        edgecolor="#707070",
        linewidth=0.15,
        alpha=0.55,
    )
    ax_3d.add_collection3d(mesh_collection)
    ax_3d.plot_surface(sphere_x, sphere_y, sphere_z, color="#f2c14e", alpha=0.18, linewidth=0)
    ax_3d.plot_surface(
        plane[..., 0],
        plane[..., 1],
        plane[..., 2],
        facecolors=plane_colors,
        shade=False,
        alpha=0.5,
    )
    ax_3d.scatter(*marker_center, color="crimson", s=55, depthshade=False, label=f"vertex {args.vertex_index}")
    ax_3d.plot([0.0, center[0]], [0.0, center[1]], [0.0, center[2]], color="crimson", linewidth=1.5)
    ax_3d.quiver(*center, *center_unit, length=0.2 * sphere_radius, color="black", arrow_length_ratio=0.12)
    ax_3d.quiver(*center, *v, length=0.35 * sphere_radius, color="#f97316", linewidth=2.5, arrow_length_ratio=0.14)
    _set_equal_axes(ax_3d, np.vstack((vertices, sphere_radius * np.eye(3), -sphere_radius * np.eye(3))))
    ax_3d.set(title="Anisotropic signal on mesh and tangent plane", xlabel="x", ylabel="y", zlabel="z")
    ax_3d.legend(loc="upper right")

    nearby = angular_distance <= args.log_radius
    grid = np.linspace(-args.log_radius, args.log_radius, 180)
    grid_u, grid_v = np.meshgrid(grid, grid)
    grid_log_vectors = grid_u[..., None] * e1 + grid_v[..., None] * e2
    grid_parallel = np.sum(grid_log_vectors * v, axis=-1)
    grid_perpendicular = np.sum(grid_log_vectors * v_perp, axis=-1)
    gaussian_grid = args.amplitude * np.exp(
        -0.5 * ((grid_parallel / args.sigma_parallel) ** 2 + (grid_perpendicular / args.sigma_perpendicular) ** 2)
    )
    ax_log.contourf(grid_u, grid_v, gaussian_grid, levels=30, cmap="viridis", alpha=0.85)
    points = ax_log.scatter(
        log_xy[nearby, 0],
        log_xy[nearby, 1],
        c=gaussian_at_vertices[nearby],
        cmap="viridis",
        vmin=0.0,
        vmax=args.amplitude,
        s=12,
        edgecolors="#d9d9d9",
        linewidths=0.25,
        label="mesh vertices",
    )
    ax_log.scatter(0.0, 0.0, color="crimson", s=55, label="selected vertex")
    ax_log.arrow(
        0.0,
        0.0,
        min(args.sigma_parallel, args.log_radius * 0.8) * float(np.dot(v, e1)),
        min(args.sigma_parallel, args.log_radius * 0.8) * float(np.dot(v, e2)),
        color="#f97316",
        width=0.007 * args.log_radius,
        head_width=0.06 * args.log_radius,
        length_includes_head=True,
        label="major axis v",
    )
    ax_log.axhline(0.0, color="0.6", linewidth=0.8)
    ax_log.axvline(0.0, color="0.6", linewidth=0.8)
    ax_log.set_aspect("equal", adjustable="box")
    ax_log.set(
        title=f"Log-map coordinates; HM angle delta = {delta:.3f} rad",
        xlabel="dot(log_c(x), e1)",
        ylabel="dot(log_c(x), e2)",
        xlim=(-args.log_radius, args.log_radius),
        ylim=(-args.log_radius, args.log_radius),
    )
    ax_log.legend()
    figure.colorbar(points, ax=ax_log, label="anisotropic Gaussian value")

    print("Anisotropic signal construction")
    print(f"  center vertex: {args.vertex_index}; center = {center.tolist()}")
    print(f"  unit-sphere center c_hat = {center_unit.tolist()}")
    print(f"  Hughes--Moller e1 = {e1.tolist()}")
    print(f"  Hughes--Moller e2 = {e2.tolist()}")
    print(f"  delta = {delta:.8f} rad; major axis v = {v.tolist()}")
    print(
        "  sigma_parallel = "
        f"{args.sigma_parallel}; sigma_perpendicular = {args.sigma_perpendicular}; amplitude = {args.amplitude}"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
