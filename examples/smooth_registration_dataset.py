#!/usr/bin/env python3
"""Smooth every registration mesh in a directory tree.

This is a batch wrapper around ``smooth_sliver_triangles.smooth_mesh``.  It
keeps each input mesh's vertex count and face connectivity unchanged, writes
the same relative filenames under a separate output directory, and records
per-file quality statistics.

Example::

    python examples/smooth_registration_dataset.py \
        --input-dir /path/to/registrations \
        --output-dir /path/to/registrations_smoothed \
        --iterations 10 --angle-threshold 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm.auto import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from smooth_sliver_triangles import laplacian_smooth_mesh, load_mesh, smooth_mesh


MESH_EXTENSIONS = {".obj", ".ply", ".off", ".stl"}


def _set_equal_axes(ax: Any, vertices: np.ndarray) -> None:
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def plot_comparison(
    original_vertices: np.ndarray,
    faces: np.ndarray,
    smoothed_vertices: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save a side-by-side original/smoothed mesh comparison."""
    figure = plt.figure(figsize=(14, 7))
    for index, (vertices, label, color) in enumerate(
        (
            (original_vertices, "original", (0.60, 0.70, 0.85, 0.55)),
            (smoothed_vertices, "smoothed", (0.35, 0.75, 0.45, 0.65)),
        )
    ):
        ax = figure.add_subplot(1, 2, index + 1, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor=color,
                edgecolor=(0.15, 0.15, 0.15, 0.12),
                linewidth=0.1,
            )
        )
        _set_equal_axes(ax, vertices)
        ax.set_title(label)
        ax.set(xlabel="x", ylabel="y", zlabel="z")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def discover_meshes(input_dir: Path, pattern: str | None, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    paths = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in MESH_EXTENSIONS:
            continue
        if pattern is not None and not path.match(pattern):
            continue
        paths.append(path)
    return sorted(paths)


def smooth_one(
    input_path: Path,
    output_path: Path,
    *,
    iterations: int,
    angle_threshold: float,
    step: float,
    min_step: float,
    preserve_boundary: bool,
    allow_normal_motion: bool,
    method: str,
    laplacian_rings: int,
    plot: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input": str(input_path),
        "output": str(output_path),
        "method": method,
        "status": "error",
        "error": None,
    }
    try:
        mesh = load_mesh(input_path)
        if method == "laplacian":
            smoothed, stats = laplacian_smooth_mesh(
                mesh.vertices,
                mesh.faces,
                iterations=iterations,
                step=step,
                preserve_boundary=preserve_boundary,
                angle_threshold=angle_threshold,
                ring_size=laplacian_rings,
            )
        else:
            smoothed, stats = smooth_mesh(
                mesh.vertices,
                mesh.faces,
                iterations=iterations,
                angle_threshold=angle_threshold,
                step=step,
                min_step=min_step,
                preserve_boundary=preserve_boundary,
                allow_normal_motion=allow_normal_motion,
            )
        if smoothed.shape != np.asarray(mesh.vertices).shape:
            raise ValueError("smoothing changed the vertex array shape")
        if not np.isfinite(smoothed).all():
            raise ValueError("smoothing produced non-finite coordinates")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(
            vertices=smoothed,
            faces=np.asarray(mesh.faces, dtype=np.int64),
            process=False,
        ).export(output_path)
        if plot:
            plot_comparison(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                smoothed,
                output_path.with_suffix(".png"),
                f"{input_path.name} — {method}",
            )
        result.update(
            {
                "status": "ok",
                "vertex_count": int(len(mesh.vertices)),
                "face_count": int(len(mesh.faces)),
                **{key: value for key, value in stats.items()},
            }
        )
    except (OSError, ValueError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def write_reports(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoothing_metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    fields = sorted({key for result in results for key, value in result.items() if not isinstance(value, (dict, list))})
    with (output_dir / "smoothing_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing registration meshes.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Separate directory for smoothed meshes and reports.")
    parser.add_argument("--pattern", default=None, help="Optional filename glob, e.g. '*.ply' or 'tr_reg_*.ply'.")
    parser.add_argument("--recursive", action="store_true", help="Search input-dir recursively.")
    parser.add_argument("--method", choices=("quality", "laplacian"), default="quality", help="smoothing algorithm")
    parser.add_argument("--iterations", "--laplacian-iterations", dest="iterations", type=int, default=10)
    parser.add_argument("--angle-threshold", type=float, default=10.0)
    parser.add_argument("--step", "--laplacian-step", dest="step", type=float, default=0.35)
    parser.add_argument("--laplacian-rings", type=int, default=1, help="one-ring, two-ring, ... Laplacian neighborhood")
    parser.add_argument("--min-step", type=float, default=1.0 / 128.0)
    parser.add_argument("--allow-normal-motion", action="store_true")
    parser.add_argument("--move-boundary", action="store_true", help="Smooth boundary vertices instead of preserving them.")
    parser.add_argument("--plot", action="store_true", help="Write a side-by-side original/smoothed PNG next to each output mesh.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}")
        return 2
    if input_dir == output_dir:
        print("ERROR: output directory must differ from input directory")
        return 2
    try:
        meshes = discover_meshes(input_dir, args.pattern, args.recursive)
    except OSError as exc:
        print(f"ERROR: could not scan {input_dir}: {exc}")
        return 2
    if not meshes:
        print(f"ERROR: no mesh files found in {input_dir}")
        return 2

    results = []
    for input_path in tqdm(meshes, desc="Smoothing registrations", unit="mesh"):
        relative = input_path.relative_to(input_dir)
        output_path = output_dir / relative
        results.append(
            smooth_one(
                input_path,
                output_path,
                iterations=args.iterations,
                angle_threshold=args.angle_threshold,
                step=args.step,
                min_step=args.min_step,
                preserve_boundary=not args.move_boundary,
                allow_normal_motion=args.allow_normal_motion,
                method=args.method,
                laplacian_rings=args.laplacian_rings,
                plot=args.plot,
            )
        )
    write_reports(results, output_dir)
    succeeded = sum(result["status"] == "ok" for result in results)
    print(f"Smoothed {succeeded} / {len(results)} registrations")
    print(f"Output directory: {output_dir}")
    print(f"Metrics: {output_dir / 'smoothing_metrics.json'}")
    return 0 if succeeded == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
