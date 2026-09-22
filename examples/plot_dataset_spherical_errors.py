#!/usr/bin/env python3
"""Analyze folded and collapsed spherical triangles for a dataset root.

The input root must contain ``meshes/`` and ``spheres/`` directories.  Files
are paired by stem, and the output contains per-sample metrics, a Markdown
summary, and colored plots.  The collapse threshold is the same rule used by
the sherepar validator and the MoebiusRegistration metrics script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from tqdm.auto import tqdm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spherepar.parametrization_validation import collapsed_face_geometry


MESH_EXTENSIONS = {".obj", ".ply", ".off", ".stl"}
COLORS = {
    "normal": (0.72, 0.78, 0.85, 0.25),
    "folded": (0.35, 0.20, 0.80, 0.85),
    "near_zero": (1.00, 0.55, 0.00, 0.90),
    "actual_zero": (0.90, 0.05, 0.05, 0.95),
}


def _load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) != 1:
            raise ValueError(f"expected one mesh, found {len(loaded.geometry)} geometries")
        loaded = next(iter(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"expected a triangular mesh, got {type(loaded).__name__}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh must contain vertices [N,3] and triangular faces [F,3]")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _resolve_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in MESH_EXTENSIONS
    }


def _orientation(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    signed = np.einsum("ij,ij->i", cross, triangles.mean(axis=1))
    tolerance = 1e-12
    outward = signed > tolerance
    inward = signed < -tolerance
    near_zero = ~(outward | inward)
    return signed, int(outward.sum()), int(inward.sum()), int(near_zero.sum())


def measure_sample(
    sample_id: str,
    mesh_path: Path | None,
    sphere_path: Path | None,
    relative_threshold: float = 1e-4,
    absolute_threshold: float = 1e-12,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "mesh_path": str(mesh_path) if mesh_path else None,
        "sphere_path": str(sphere_path) if sphere_path else None,
        "status": "error",
        "error": None,
    }
    if sphere_path is None:
        result["error"] = "missing sphere file"
        return result
    try:
        sphere = _load_triangle_mesh(sphere_path)
        sphere_vertices = np.asarray(sphere.vertices, dtype=np.float64)
        sphere_faces = np.asarray(sphere.faces, dtype=np.int64)
        if not np.isfinite(sphere_vertices).all():
            raise ValueError("sphere contains non-finite coordinates")
        norms = np.linalg.norm(sphere_vertices, axis=1)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ValueError("sphere contains zero-length or non-finite vertices")
        unit_vertices = sphere_vertices / norms[:, None]
        _, twice_areas, _, threshold = collapsed_face_geometry(
            unit_vertices,
            sphere_faces,
            relative_threshold=relative_threshold,
            absolute_twice_area_threshold=absolute_threshold,
        )
        actual_zero = twice_areas == 0.0
        near_zero = (twice_areas > 0.0) & (twice_areas <= threshold)
        signed, outward_count, inward_count, orientation_zero_count = _orientation(
            unit_vertices, sphere_faces
        )
        folded = signed < -1e-12
        topology_ok = None
        mesh_faces = None
        if mesh_path is not None:
            mesh = _load_triangle_mesh(mesh_path)
            mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
            topology_ok = (
                len(mesh.vertices) == len(unit_vertices)
                and mesh_faces.shape == sphere_faces.shape
                and np.array_equal(mesh_faces, sphere_faces)
            )
        result.update(
            {
                "status": "ok",
                "vertex_count": int(len(unit_vertices)),
                "face_count": int(len(sphere_faces)),
                "topology_ok": topology_ok,
                "collapsed_count": int(np.count_nonzero(actual_zero | near_zero)),
                "collapsed_fraction": float(np.mean(actual_zero | near_zero)),
                "actual_zero_count": int(np.count_nonzero(actual_zero)),
                "actual_zero_fraction": float(np.mean(actual_zero)),
                "near_zero_count": int(np.count_nonzero(near_zero)),
                "near_zero_fraction": float(np.mean(near_zero)),
                "folded_face_count": int(np.count_nonzero(folded)),
                "folded_fraction": float(np.mean(folded)),
                "folded_count": int(min(outward_count, inward_count)),
                "outward_count": outward_count,
                "inward_count": inward_count,
                "near_zero_orientation_count": orientation_zero_count,
                "min_twice_area": float(np.min(twice_areas)),
                "median_twice_area": float(np.median(twice_areas)),
                "min_area": float(np.min(twice_areas) / 2.0),
                "collapse_threshold": float(threshold),
                "actual_zero_face_ids": np.flatnonzero(actual_zero).tolist(),
                "near_zero_face_ids": np.flatnonzero(near_zero).tolist(),
                "folded_face_ids": np.flatnonzero(folded).tolist(),
            }
        )
        if topology_ok is False:
            result["error"] = "mesh and sphere face topology/order differ"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _category_ids(metrics: dict[str, Any]) -> dict[str, set[int]]:
    actual = set(metrics.get("actual_zero_face_ids", []))
    near = set(metrics.get("near_zero_face_ids", [])) - actual
    folded = set(metrics.get("folded_face_ids", [])) - actual - near
    return {"actual_zero": actual, "near_zero": near, "folded": folded}


def set_equal_axes(ax: Any, vertices: np.ndarray) -> None:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_sample(metrics: dict[str, Any], output_path: Path, show: bool = False) -> None:
    sphere = _load_triangle_mesh(Path(metrics["sphere_path"]))
    vertices = np.asarray(sphere.vertices, dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(sphere.faces, dtype=np.int64)
    triangles = vertices[faces]
    categories = _category_ids(metrics)
    colors = [COLORS["normal"] for _ in range(len(faces))]
    for category in ("folded", "near_zero", "actual_zero"):
        for face_id in categories[category]:
            colors[face_id] = COLORS[category]

    figure = plt.figure(figsize=(11, 10))
    ax = figure.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolor=colors, edgecolor=(0.2, 0.2, 0.2, 0.15), linewidth=0.15)
    )
    for category, label in (
        ("actual_zero", "actual zero"),
        ("near_zero", "near zero"),
        ("folded", "folded"),
        ("normal", "other"),
    ):
        ax.scatter([], [], [], color=COLORS[category], label=f"{label}: {len(categories.get(category, []))}")
    ax.legend(loc="upper right")
    set_equal_axes(ax, vertices)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_title(
        f"{metrics['sample_id']} ({metrics.get('backend', '')}) — "
        f"folded {metrics.get('folded_face_count', 0)}, "
        f"near-zero {metrics.get('near_zero_count', 0)}, "
        f"actual-zero {metrics.get('actual_zero_count', 0)}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def plot_original_mesh(metrics: dict[str, Any], output_path: Path, show: bool = False) -> None:
    """Plot the original mesh with the spherical error categories transferred by face ID."""
    mesh_path = metrics.get("mesh_path")
    if not mesh_path:
        return
    mesh = _load_triangle_mesh(Path(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    categories = _category_ids(metrics)
    colors = [COLORS["normal"] for _ in range(len(faces))]
    for category in ("folded", "near_zero", "actual_zero"):
        for face_id in categories[category]:
            if 0 <= face_id < len(colors):
                colors[face_id] = COLORS[category]

    figure = plt.figure(figsize=(11, 10))
    ax = figure.add_subplot(111, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(triangles, facecolor=colors, edgecolor=(0.2, 0.2, 0.2, 0.15), linewidth=0.15)
    )
    for category, label in (
        ("actual_zero", "actual zero"),
        ("near_zero", "near zero"),
        ("folded", "folded"),
        ("normal", "other"),
    ):
        ax.scatter([], [], [], color=COLORS[category], label=f"{label}: {len(categories.get(category, []))}")
    ax.legend(loc="upper right")
    set_equal_axes(ax, vertices)
    ax.set(xlabel="x", ylabel="y", zlabel="z")
    ax.set_title(f"{metrics['sample_id']} ({metrics.get('backend', '')}) — original mesh")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def write_report(metrics: list[dict[str, Any]], backend: str, root: Path, output_root: Path) -> None:
    valid = [item for item in metrics if item["status"] == "ok"]
    lines = [
        "# Spherical parametrization error report",
        "",
        f"Backend: **{backend}**  ",
        f"Input root: `{root}`  ",
        f"Samples analyzed: **{len(valid)} / {len(metrics)}**",
        "",
        "Triangles are normalized to the unit sphere. Actual-zero triangles have "
        "exactly zero twice-area; near-zero triangles have positive twice-area "
        "at or below `max(1e-12, 1e-4 * median_twice_area)`; folded triangles "
        "have negative radial orientation. Plot colors use actual-zero, near-zero, "
        "folded, then normal precedence.",
        "",
        "| Sample | Faces | Folded | Near-zero | Actual-zero | Threshold | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        if item["status"] != "ok":
            lines.append(f"| {item['sample_id']} | — | — | — | — | — | {item['error']} |")
            continue
        lines.append(
            f"| {item['sample_id']} | {item['face_count']} | {item['folded_face_count']} "
            f"({item['folded_fraction']:.6g}) | {item['near_zero_count']} "
            f"({item['near_zero_fraction']:.6g}) | {item['actual_zero_count']} "
            f"({item['actual_zero_fraction']:.6g}) | {item['collapse_threshold']:.6g} | ok |"
        )
    if valid:
        for key, label in (("folded_fraction", "folded"), ("near_zero_fraction", "near-zero"), ("actual_zero_fraction", "actual-zero")):
            values = np.asarray([item[key] for item in valid], dtype=float)
            lines.append("")
            lines.append(f"- {label} fraction median: **{np.median(values):.6g}**")
            lines.append(f"- {label} fraction maximum: **{np.max(values):.6g}**")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary_report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root containing meshes/ and spheres/.")
    parser.add_argument("--backend", choices=("flash", "cem", "cmcf"), required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--pattern", default=None, help="Optional glob applied to sphere filenames.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--relative-area-threshold", type=float, default=1e-4)
    parser.add_argument("--absolute-twice-area-threshold", type=float, default=1e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    mesh_files = _resolve_files(root / "meshes")
    sphere_files = _resolve_files(root / "spheres")
    sample_ids = sorted(set(mesh_files) | set(sphere_files))
    if args.sample_id is not None:
        sample_ids = [args.sample_id]
    if args.pattern is not None:
        import fnmatch
        sample_ids = [sample_id for sample_id in sample_ids if fnmatch.fnmatch(sphere_files.get(sample_id, Path(sample_id)).name, args.pattern)]
    output_root = (args.output_root or root / "error_analysis").expanduser().resolve()
    metrics = []
    for sample_id in tqdm(sample_ids, desc=f"Analyzing {args.backend}", unit="sample"):
        item = measure_sample(
            sample_id,
            mesh_files.get(sample_id),
            sphere_files.get(sample_id),
            relative_threshold=args.relative_area_threshold,
            absolute_threshold=args.absolute_twice_area_threshold,
        )
        item["backend"] = args.backend
        metrics.append(item)
        if item["status"] == "ok" and not args.no_plots:
            plot_sample(item, output_root / "plots" / f"{sample_id}_sphere.png", show=args.show)
            if item.get("mesh_path") and item.get("topology_ok") is not False:
                plot_original_mesh(item, output_root / "plots" / f"{sample_id}_mesh.png", show=args.show)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    fields = sorted({key for item in metrics for key, value in item.items() if not isinstance(value, list)})
    with (output_root / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
    write_report(metrics, args.backend, root, output_root)
    print(f"Analyzed {sum(item['status'] == 'ok' for item in metrics)} / {len(metrics)} samples")
    print(f"Wrote {output_root}")
    return 0 if metrics and any(item["status"] == "ok" for item in metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
