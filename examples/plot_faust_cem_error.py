#!/usr/bin/env python3
"""Plot collapsed triangles and input-mesh quality for one CEM sphere.

Collapsed or nearly collapsed triangles are drawn in red, with red dots
sampled along their edges. Four figures are produced: the CEM sphere, the
source mesh with the same collapsed face IDs, source faces incident to
negative cotangent-weight edges, and source faces whose minimum angle is less
than the requested threshold. By default the script loads ``tr_reg_000`` from
``data/faust_cem_one_v2``. A different generated dataset or direct sphere OBJ
can be selected from the command line.

Examples
--------
python examples/plot_faust_cem_error.py

python examples/plot_faust_cem_error.py \
    --dataset-root data/faust_cem_r12 \
    --sample-id tr_reg_000 \
    --output data/faust_cem_r12/collapsed_triangles.png \
    --no-show

python examples/plot_faust_cem_error.py \
    --sphere data/faust_cem_r12/spheres/tr_reg_000.obj
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spherepar.mobius_centering import apply_inverse_mobius_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "faust_cem_one_v2",
        help="Generated dataset root containing labels/ and spheres/.",
    )
    parser.add_argument(
        "--sample-id",
        default="tr_reg_000",
        help="Sample ID used to resolve the primary label and sphere.",
    )
    parser.add_argument(
        "--sphere",
        type=Path,
        default=None,
        help="Direct sphere OBJ path; bypasses dataset-label path resolution.",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="Direct original-mesh OBJ path; normally resolved from the primary label.",
    )
    parser.add_argument(
        "--pre-centering",
        action="store_true",
        help="Undo the stored Möbius transform and plot the original CEM sphere.",
    )
    parser.add_argument(
        "--relative-area-threshold",
        type=float,
        default=1e-4,
        help="Collapse threshold relative to the median twice-area.",
    )
    parser.add_argument(
        "--absolute-twice-area-threshold",
        type=float,
        default=1e-12,
        help="Absolute lower bound for the twice-area collapse threshold.",
    )
    parser.add_argument(
        "--edge-points",
        type=int,
        default=8,
        help="Number of red dots sampled along each collapsed edge.",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=0,
        help="Print vertex coordinates for this many collapsed faces (0 disables details).",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Minimum face-angle threshold in degrees for the fourth plot.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG/PDF output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    return parser.parse_args()


def _dataset_path(dataset_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else dataset_root / candidate


def _load_sidecar(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.is_file():
        return None
    with path.open() as file:
        return json.load(file)


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Resolve sphere, spherical sidecar, and original-mesh paths."""
    dataset_root = args.dataset_root.expanduser().resolve()
    explicit_mesh = args.mesh.expanduser().resolve() if args.mesh is not None else None
    if args.sphere is not None:
        sphere_path = args.sphere.expanduser().resolve()
        conventional_sidecar = sphere_path.parent.parent / "labels" / f"{sphere_path.stem}_spherical.json"
        conventional_mesh = sphere_path.parent.parent / "meshes" / sphere_path.name
        mesh_path = explicit_mesh or (conventional_mesh if conventional_mesh.is_file() else None)
        return sphere_path, conventional_sidecar if conventional_sidecar.is_file() else None, mesh_path

    primary_label_path = dataset_root / "labels" / f"{args.sample_id}.json"
    if primary_label_path.is_file():
        with primary_label_path.open() as file:
            label = json.load(file)
        paths = label.get("paths", {})
        sphere_value = paths.get("sphere", label.get("sphere_path"))
        mesh_value = paths.get("mesh", label.get("mesh_path"))
        sidecar_value = paths.get("spherical_label")
        if sphere_value is None:
            raise KeyError(f"No sphere path in {primary_label_path}")
        sphere_path = _dataset_path(dataset_root, sphere_value)
        sidecar_path = (
            _dataset_path(dataset_root, sidecar_value)
            if sidecar_value is not None
            else dataset_root / "labels" / f"{args.sample_id}_spherical.json"
        )
        mesh_path = explicit_mesh or (
            _dataset_path(dataset_root, mesh_value) if mesh_value is not None else None
        )
        return sphere_path, sidecar_path if sidecar_path.is_file() else None, mesh_path

    sphere_path = dataset_root / "spheres" / f"{args.sample_id}.obj"
    sidecar_path = dataset_root / "labels" / f"{args.sample_id}_spherical.json"
    conventional_mesh = dataset_root / "meshes" / f"{args.sample_id}.obj"
    mesh_path = explicit_mesh or (conventional_mesh if conventional_mesh.is_file() else None)
    return sphere_path, sidecar_path if sidecar_path.is_file() else None, mesh_path


def load_sphere(
    sphere_path: Path,
    sidecar_path: Optional[Path],
    pre_centering: bool,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if not sphere_path.is_file():
        raise FileNotFoundError(f"Sphere OBJ not found: {sphere_path}")

    sphere = trimesh.load_mesh(sphere_path, process=False)
    if not isinstance(sphere, trimesh.Trimesh):
        raise ValueError(f"Expected one triangular mesh in {sphere_path}")

    vertices = np.asarray(sphere.vertices, dtype=np.float64)
    faces = np.asarray(sphere.faces, dtype=np.int64)
    mode = "stored sphere"

    if pre_centering:
        sidecar = _load_sidecar(sidecar_path)
        centering = (sidecar or {}).get("metadata", {}).get("mobius_centering")
        if centering is None:
            raise ValueError("--pre-centering requested, but no Möbius transform was found in the sidecar")
        vertices = apply_inverse_mobius_sequence(vertices, centering["transform"])
        mode = "pre-centering CEM sphere"

    return vertices, faces, mode


def load_original_mesh(mesh_path: Optional[Path], sphere_faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mesh_path is None or not mesh_path.is_file():
        raise FileNotFoundError(
            "Original mesh was not found. Use a dataset primary label or provide --mesh PATH."
        )
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected one triangular mesh in {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.shape != sphere_faces.shape or not np.array_equal(faces, sphere_faces):
        raise ValueError("Original mesh and sphere do not have identical face ordering")
    return vertices, faces


def collapsed_geometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    relative_threshold: float,
    absolute_twice_area_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if relative_threshold < 0.0 or absolute_twice_area_threshold < 0.0:
        raise ValueError("area thresholds must be non-negative")

    triangles = vertices[faces]
    twice_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    threshold = max(
        absolute_twice_area_threshold,
        relative_threshold * float(np.median(twice_areas)),
    )
    collapsed_face_ids = np.flatnonzero(twice_areas <= threshold)
    return triangles, twice_areas, collapsed_face_ids, threshold


def collapsed_edge_points(
    vertices: np.ndarray,
    collapsed_faces: np.ndarray,
    samples_per_edge: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if samples_per_edge < 2:
        raise ValueError("--edge-points must be at least 2")
    if len(collapsed_faces) == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=np.float64)

    edges = np.concatenate(
        (
            collapsed_faces[:, [0, 1]],
            collapsed_faces[:, [1, 2]],
            collapsed_faces[:, [2, 0]],
        ),
        axis=0,
    )
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    edge_t = np.linspace(0.0, 1.0, samples_per_edge)
    start = vertices[edges[:, 0]]
    end = vertices[edges[:, 1]]
    points = (
        start[:, None, :] * (1.0 - edge_t[None, :, None])
        + end[:, None, :] * edge_t[None, :, None]
    ).reshape(-1, 3)
    return edges, points


def input_mesh_quality_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    angle_threshold_degrees: float,
    cotangent_tolerance: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find low-angle faces and faces touching negative cotangent weights.

    For an interior edge ``(i, j)``, the cotangent weight is
    ``w_ij = 0.5 * (cot(alpha) + cot(beta))``. CEM stores the corresponding
    Laplacian off-diagonal as ``L_ij = -w_ij``.
    """
    if not 0.0 < angle_threshold_degrees < 180.0:
        raise ValueError("--angle-threshold must be between 0 and 180 degrees")

    triangles = vertices[faces]

    def corner_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        if np.any(denominator <= np.finfo(np.float64).eps):
            raise ValueError("input mesh contains a zero-length triangle edge")
        cosine = np.einsum("ij,ij->i", first, second) / denominator
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    angles = np.column_stack((
        corner_angle(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        corner_angle(triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 1]),
        corner_angle(triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 2]),
    ))
    minimum_angles_degrees = np.rad2deg(angles).min(axis=1)
    low_angle_face_ids = np.flatnonzero(
        minimum_angles_degrees < angle_threshold_degrees
    )

    edge_weights: dict[tuple[int, int], float] = {}
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, ((a, b, c), face_angles) in enumerate(zip(faces, angles)):
        # Each corner angle contributes half its cotangent to the opposite edge.
        opposite_edges = ((b, c), (c, a), (a, b))
        for edge, angle in zip(opposite_edges, face_angles):
            edge_key = tuple(sorted((int(edge[0]), int(edge[1]))))
            edge_weights[edge_key] = edge_weights.get(edge_key, 0.0) + 0.5 / np.tan(angle)
            edge_faces.setdefault(edge_key, []).append(face_id)

    negative_edges = np.asarray(
        sorted(edge for edge, weight in edge_weights.items() if weight < -cotangent_tolerance),
        dtype=np.int64,
    ).reshape(-1, 2)
    negative_edge_set = {tuple(edge) for edge in negative_edges.tolist()}
    negative_weight_face_ids = np.asarray(
        sorted({face_id for edge in negative_edge_set for face_id in edge_faces[edge]}),
        dtype=np.int64,
    )
    return (
        minimum_angles_degrees,
        negative_edges,
        negative_weight_face_ids,
        low_angle_face_ids,
    )


def print_summary(
    sphere_path: Path,
    mode: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    twice_areas: np.ndarray,
    collapsed_face_ids: np.ndarray,
    threshold: float,
    max_details: int,
) -> None:
    print(f"Sphere:              {sphere_path}")
    print(f"Displayed geometry:  {mode}")
    print(f"Vertices:            {len(vertices)}")
    print(f"Total triangles:     {len(faces)}")
    print(f"Collapsed triangles: {len(collapsed_face_ids)}")
    print(f"Minimum area:        {0.5 * float(twice_areas.min()):.8e}")
    print(f"Median area:         {0.5 * float(np.median(twice_areas)):.8e}")
    print(f"Maximum area:        {0.5 * float(twice_areas.max()):.8e}")
    print(f"Area threshold:      {0.5 * threshold:.8e}")

    preview = collapsed_face_ids[:20].tolist()
    suffix = " ..." if len(collapsed_face_ids) > len(preview) else ""
    print(f"Collapsed face IDs:  {preview}{suffix}")

    for face_id in collapsed_face_ids[:max(0, max_details)]:
        vertex_ids = faces[face_id]
        triangle = vertices[vertex_ids]
        distances = (
            np.linalg.norm(triangle[0] - triangle[1]),
            np.linalg.norm(triangle[1] - triangle[2]),
            np.linalg.norm(triangle[2] - triangle[0]),
        )
        print(f"\nFace {face_id}; area={0.5 * twice_areas[face_id]:.8e}")
        for vertex_id, position in zip(vertex_ids, triangle):
            print(f"  vertex {vertex_id}: {position.tolist()}")
        print(f"  edge distances: {distances}")


def set_equal_axes(ax: plt.Axes, vertices: np.ndarray) -> None:
    low = vertices.min(axis=0)
    high = vertices.max(axis=0)
    center = (low + high) / 2.0
    radius = max(0.5 * float(np.max(high - low)), 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_collapsed_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    triangles: np.ndarray,
    collapsed_face_ids: np.ndarray,
    red_edge_points: np.ndarray,
    mode: str,
) -> plt.Figure:
    figure = plt.figure(figsize=(12, 12))
    ax = figure.add_subplot(111, projection="3d")

    mesh_collection = Poly3DCollection(
        triangles,
        facecolor=(0.72, 0.78, 0.85, 0.20),
        edgecolor=(0.25, 0.25, 0.25, 0.12),
        linewidth=0.15,
    )
    ax.add_collection3d(mesh_collection)

    if len(collapsed_face_ids):
        collapsed_collection = Poly3DCollection(
            triangles[collapsed_face_ids],
            facecolor=(1.0, 0.0, 0.0, 0.50),
            edgecolor=(0.6, 0.0, 0.0, 1.0),
            linewidth=0.8,
        )
        ax.add_collection3d(collapsed_collection)

    if len(red_edge_points):
        ax.scatter(
            red_edge_points[:, 0],
            red_edge_points[:, 1],
            red_edge_points[:, 2],
            color="red",
            s=5,
            depthshade=False,
            label="collapsed-triangle edges",
        )
        ax.legend(loc="upper right")

    set_equal_axes(ax, vertices)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_title(
        f"FAUST {mode}: {len(collapsed_face_ids)} / {len(faces)} collapsed triangles"
    )
    figure.tight_layout()
    return figure


def plot_original_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    collapsed_face_ids: np.ndarray,
) -> plt.Figure:
    """Plot the source mesh with sphere-collapsed faces highlighted."""
    triangles = vertices[faces]
    figure = plt.figure(figsize=(12, 12))
    ax = figure.add_subplot(111, projection="3d")

    mesh_collection = Poly3DCollection(
        triangles,
        facecolor=(0.72, 0.78, 0.85, 0.28),
        edgecolor=(0.25, 0.25, 0.25, 0.12),
        linewidth=0.15,
    )
    ax.add_collection3d(mesh_collection)

    if len(collapsed_face_ids):
        collapsed_collection = Poly3DCollection(
            triangles[collapsed_face_ids],
            facecolor=(1.0, 0.0, 0.0, 0.50),
            edgecolor=(0.65, 0.0, 0.0, 0.80),
            linewidth=0.5,
        )
        ax.add_collection3d(collapsed_collection)

    set_equal_axes(ax, vertices)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_title(
        "Original mesh: "
        f"{len(collapsed_face_ids)} sphere-collapsed triangles highlighted"
    )
    figure.tight_layout()
    return figure


def plot_original_mesh_highlights(
    vertices: np.ndarray,
    faces: np.ndarray,
    highlighted_face_ids: np.ndarray,
    facecolor: tuple[float, float, float, float],
    edgecolor: tuple[float, float, float, float],
    title: str,
) -> plt.Figure:
    """Plot selected source-mesh faces with the requested translucent color."""
    triangles = vertices[faces]
    figure = plt.figure(figsize=(12, 12))
    ax = figure.add_subplot(111, projection="3d")

    ax.add_collection3d(Poly3DCollection(
        triangles,
        facecolor=(0.72, 0.78, 0.85, 0.28),
        edgecolor=(0.25, 0.25, 0.25, 0.12),
        linewidth=0.15,
    ))
    if len(highlighted_face_ids):
        ax.add_collection3d(Poly3DCollection(
            triangles[highlighted_face_ids],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.5,
        ))

    set_equal_axes(ax, vertices)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_title(title)
    figure.tight_layout()
    return figure


def main() -> None:
    args = parse_args()
    sphere_path, sidecar_path, mesh_path = resolve_inputs(args)
    vertices, faces, mode = load_sphere(sphere_path, sidecar_path, args.pre_centering)
    mesh_vertices, mesh_faces = load_original_mesh(mesh_path, faces)
    (
        minimum_angles_degrees,
        negative_edges,
        negative_weight_face_ids,
        low_angle_face_ids,
    ) = input_mesh_quality_faces(
        mesh_vertices,
        mesh_faces,
        args.angle_threshold,
    )
    triangles, twice_areas, collapsed_face_ids, threshold = collapsed_geometry(
        vertices,
        faces,
        args.relative_area_threshold,
        args.absolute_twice_area_threshold,
    )
    collapsed_faces = faces[collapsed_face_ids]
    _, red_edge_points = collapsed_edge_points(vertices, collapsed_faces, args.edge_points)

    print_summary(
        sphere_path,
        mode,
        vertices,
        faces,
        twice_areas,
        collapsed_face_ids,
        threshold,
        args.max_details,
    )
    print(f"Negative cotangent edges: {len(negative_edges)}")
    print(f"Triangles touching them:  {len(negative_weight_face_ids)}")
    print(f"Minimum input angle:      {minimum_angles_degrees.min():.8f} degrees")
    print(
        f"Triangles below {args.angle_threshold:g} degrees: "
        f"{len(low_angle_face_ids)}"
    )
    print("Sign convention:          L_ij = -w_ij for off-diagonal entries")
    collapsed_negative_overlap = np.intersect1d(
        collapsed_face_ids,
        negative_weight_face_ids,
        assume_unique=True,
    )
    collapsed_recall = (
        len(collapsed_negative_overlap) / len(collapsed_face_ids)
        if len(collapsed_face_ids)
        else 1.0
    )
    negative_collapse_fraction = (
        len(collapsed_negative_overlap) / len(negative_weight_face_ids)
        if len(negative_weight_face_ids)
        else 0.0
    )
    collapsed_is_subset = len(collapsed_negative_overlap) == len(collapsed_face_ids)
    print("Collapsed/negative-weight overlap:")
    print(
        "  intersection:           "
        f"{len(collapsed_negative_overlap)} face(s)"
    )
    print(f"  collapsed-face recall:  {100.0 * collapsed_recall:.2f}%")
    print(
        "  negative-face collapse: "
        f"{100.0 * negative_collapse_fraction:.2f}%"
    )
    print(f"  collapsed is subset:    {collapsed_is_subset}")
    figure = plot_collapsed_faces(
        vertices,
        faces,
        triangles,
        collapsed_face_ids,
        red_edge_points,
        mode,
    )
    mesh_figure = plot_original_mesh(mesh_vertices, mesh_faces, collapsed_face_ids)
    negative_weight_figure = plot_original_mesh_highlights(
        mesh_vertices,
        mesh_faces,
        negative_weight_face_ids,
        facecolor=(1.0, 0.85, 0.0, 0.50),
        edgecolor=(0.75, 0.55, 0.0, 0.85),
        title=(
            "Original mesh: "
            f"{len(negative_weight_face_ids)} triangles touching negative cotangent weights"
        ),
    )
    low_angle_figure = plot_original_mesh_highlights(
        mesh_vertices,
        mesh_faces,
        low_angle_face_ids,
        facecolor=(1.0, 0.40, 0.0, 0.50),
        edgecolor=(0.75, 0.20, 0.0, 0.85),
        title=(
            "Original mesh: "
            f"{len(low_angle_face_ids)} triangles with minimum angle "
            f"< {args.angle_threshold:g} degrees"
        ),
    )

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Saved plot:          {output}")
        mesh_output = output.with_name(f"{output.stem}_original_mesh{output.suffix}")
        mesh_figure.savefig(mesh_output, dpi=180, bbox_inches="tight")
        print(f"Saved mesh plot:     {mesh_output}")
        negative_output = output.with_name(
            f"{output.stem}_negative_cotangent_weights{output.suffix}"
        )
        negative_weight_figure.savefig(negative_output, dpi=180, bbox_inches="tight")
        print(f"Saved weight plot:   {negative_output}")
        angle_label = f"{args.angle_threshold:g}".replace(".", "p")
        angle_output = output.with_name(
            f"{output.stem}_angles_below_{angle_label}deg{output.suffix}"
        )
        low_angle_figure.savefig(angle_output, dpi=180, bbox_inches="tight")
        print(f"Saved angle plot:    {angle_output}")

    if args.no_show:
        plt.close(figure)
        plt.close(mesh_figure)
        plt.close(negative_weight_figure)
        plt.close(low_angle_figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
