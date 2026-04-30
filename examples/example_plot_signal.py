#!/usr/bin/env python3
"""
examples/example_plot_signal.py
===============================

Batch plotting utility for mesh signals stored in a dataset directory.

Expected input layout::

    <input_dir>/
        labels/
        meshes/ (or spheres/ if --use-spheres)
        spheres/ (optional; contains spherical parametrization OBJ files)
        signals/

For each sample name found in ``labels/`` (excluding ``*_signal.json``),
the script loads the matching mesh from ``meshes/<name>.obj`` by default (or
from ``spheres/<name>.obj`` if ``--use-spheres`` is specified), and the matching
signal from ``signals/<name>.npy``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create signal plots for every mesh in a dataset directory."
    )
    parser.add_argument("input_dir", help="Dataset root containing labels/, meshes/, and signals/.")
    parser.add_argument(
        "--use-spheres",
        action="store_true",
        help="Use OBJ files from <input_dir>/spheres/ instead of meshes/ (spherical parametrization output).",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument("--linewidths", type=float, default=0.2, help="Triangle edge linewidth.")
    parser.add_argument("--figsize-x", type=float, default=10, help="Figure width in inches.")
    parser.add_argument("--figsize-y", type=float, default=8, help="Figure height in inches.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Face alpha for the mesh collection.")
    parser.add_argument("--edgecolor", default="black", help="Triangle edge color.")
    parser.add_argument("--title", default="Mesh with Signal Values", help="Base plot title.")
    parser.add_argument("--num-views", type=int, default=5, help="Number of azimuth views to render per sample.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots. Defaults to <input_dir>/signal_plots.",
    )
    return parser.parse_args()


def iter_sample_names(labels_dir: Path) -> Iterable[str]:
    for label_path in sorted(labels_dir.glob("*.json")):
        if label_path.stem.endswith("_signal"):
            continue
        yield label_path.stem


def load_signal(input_dir: Path, sample_name: str, mesh: trimesh.Trimesh) -> tuple[np.ndarray, str]:
    signal_path = input_dir / "signals" / f"{sample_name}.npy"
    if not signal_path.is_file():
        raise FileNotFoundError(f"signal file not found for {sample_name}: {signal_path}")

    signal = np.load(signal_path)
    if signal.shape[0] != len(mesh.vertices):
        raise ValueError(
            f"Signal length mismatch for {sample_name}: "
            f"{signal.shape[0]} values for {len(mesh.vertices)} vertices"
        )
    return np.asarray(signal, dtype=float), str(signal_path)


def plot_sample(
        mesh: trimesh.Trimesh,
        signal: np.ndarray,
        sample_name: str,
        signal_source: str,
        output_dir: Path,
        cmap: str,
        linewidths: float,
        figsize_x: float,
        figsize_y: float,
        alpha: float,
        edgecolor: str,
        title: str,
        num_views: int,
) -> None:
    face_signal = signal[mesh.faces].mean(axis=1)
    if num_views <= 0:
        raise ValueError(f"num_views must be positive, got {num_views}")

    for view_idx, azim in enumerate(np.linspace(0.0, 360.0, num_views, endpoint=False)):
        fig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            mesh.vertices[mesh.faces],
            alpha=alpha,
            edgecolor=edgecolor,
            linewidths=linewidths,
        )
        poly.set_array(face_signal)
        poly.set_cmap(cmap)
        ax.add_collection3d(poly)

        ax.set_xlim(mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max())
        ax.set_ylim(mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max())
        ax.set_zlim(mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max())

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title}: {sample_name} (view {view_idx})")
        ax.view_init(elev=30.0, azim=float(azim))

        cbar = fig.colorbar(poly, ax=ax, label="Signal Value")
        cbar.ax.set_title("src")
        cbar.ax.text(0.5, -0.08, signal_source, transform=cbar.ax.transAxes, ha="center", va="top", fontsize=8)

        plt.tight_layout()
        output_path = output_dir / f"{sample_name}_view{view_idx:02d}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    labels_dir = input_dir / "labels"
    spheres_dir = input_dir / "spheres"
    meshes_dir = input_dir / "meshes"
    signals_dir = input_dir / "signals"
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir / "signal_plots"

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")
    if args.use_spheres:
        if not spheres_dir.is_dir():
            raise FileNotFoundError(f"spheres directory not found: {spheres_dir}")
        meshes_dir = spheres_dir
    else:
        if not meshes_dir.is_dir():
            raise FileNotFoundError(f"meshes directory not found: {meshes_dir}")
    if not signals_dir.is_dir():
        raise FileNotFoundError(f"signals directory not found: {signals_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_names = list(iter_sample_names(labels_dir))
    if not sample_names:
        raise FileNotFoundError(f"No label files found in: {labels_dir}")

    for sample_name in sample_names:
        label_path = labels_dir / f"{sample_name}.json"
        mesh_path = meshes_dir / f"{sample_name}.obj"
        if not mesh_path.is_file():
            with open(label_path) as fh:
                label_data = json.load(fh)
            mesh_path = Path(label_data.get("mesh_file", mesh_path))

        if not mesh_path.is_file():
            print(f"Skipping {sample_name}: mesh file not found")
            continue

        mesh = trimesh.load_mesh(mesh_path)
        signal, signal_source = load_signal(input_dir, sample_name, mesh)
        plot_sample_names = f"par_{sample_name}" if args.use_spheres else sample_name
        plot_sample(
            mesh=mesh,
            signal=signal,
            sample_name=plot_sample_names,
            signal_source=signal_source,
            output_dir=output_dir,
            cmap=args.cmap,
            linewidths=args.linewidths,
            figsize_x=args.figsize_x,
            figsize_y=args.figsize_y,
            alpha=args.alpha,
            edgecolor=args.edgecolor,
            title=args.title,
            num_views=args.num_views,
        )
        print(f"Saved {args.num_views} views for {sample_name} in {output_dir}")


if __name__ == "__main__":
    main()
