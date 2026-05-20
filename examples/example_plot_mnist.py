#!/usr/bin/env python3
"""
examples/example_plot_mnist.py
==============================

Batch plotting utility for MNIST signals stored in a dataset directory.

Expected input layout::

    <input_dir>/
        labels/
        meshes/
        signals/

For each label file whose ``signal.signal_type`` is ``mnist``, the script loads
the matching mesh and MNIST signal, then saves a plot to the output directory.
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
        description="Create plots for every MNIST signal in a dataset directory."
    )
    parser.add_argument("input_dir", help="Dataset root containing labels/, meshes/, and signals/.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plots. Defaults to <input_dir>/mnist_plots.",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument("--linewidths", type=float, default=0.2, help="Triangle edge linewidth.")
    parser.add_argument("--figsize-x", type=float, default=10, help="Figure width in inches.")
    parser.add_argument("--figsize-y", type=float, default=8, help="Figure height in inches.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Face alpha for the mesh collection.")
    parser.add_argument("--edgecolor", default="black", help="Triangle edge color.")
    parser.add_argument("--title", default="MNIST Signal", help="Base plot title.")
    parser.add_argument("--num-views", type=int, default=1, help="Number of azimuth views per sample.")
    parser.add_argument(
        "--mesh-subdir",
        choices=("meshes", "spheres"),
        default="meshes",
        help="Which mesh directory to read from.",
    )
    return parser.parse_args()


def iter_sample_names(labels_dir: Path) -> Iterable[str]:
    for label_path in sorted(labels_dir.glob("*.json")):
        stem = label_path.stem
        if stem.endswith("_signal"):
            continue
        if "_iso_" in stem or "_aniso_" in stem:
            continue
        yield stem


def is_mnist_label(label_data: dict) -> bool:
    signal_meta = label_data.get("signal", {})
    if signal_meta.get("signal_type") == "mnist":
        return True
    if signal_meta.get("family") == "mnist":
        return True
    if isinstance(label_data.get("signal_file"), str) and label_data["signal_file"].endswith("_mnist.npy"):
        return True
    tasks = label_data.get("tasks", {})
    if isinstance(tasks, dict) and "mnist_cls" in tasks:
        return True
    return False


def load_signal(
        input_dir: Path,
        sample_name: str,
        mesh: trimesh.Trimesh,
        label_data: dict,
) -> tuple[np.ndarray, str]:
    signal_meta = label_data.get("signal", {})
    signal_path = signal_meta.get("signal_file") or label_data.get("signal_file")
    if signal_path:
        path = Path(signal_path)
        if not path.is_absolute():
            path = input_dir / path
        if path.is_file():
            signal = np.load(path)
            if signal.shape[0] != len(mesh.vertices):
                raise ValueError(
                    f"Signal length mismatch for {sample_name}: "
                    f"{signal.shape[0]} values for {len(mesh.vertices)} vertices"
                )
            return np.asarray(signal, dtype=float), str(path)

    path = input_dir / "signals" / f"{sample_name}_mnist.npy"
    if not path.is_file():
        raise FileNotFoundError(f"MNIST signal file not found for {sample_name}: {path}")

    signal = np.load(path)
    if signal.shape[0] != len(mesh.vertices):
        raise ValueError(
            f"Signal length mismatch for {sample_name}: "
            f"{signal.shape[0]} values for {len(mesh.vertices)} vertices"
        )
    return np.asarray(signal, dtype=float), str(path)


def extract_mnist_label(label_data: dict) -> int | None:
    tasks = label_data.get("tasks", {})
    if isinstance(tasks, dict):
        mnist_cls = tasks.get("mnist_cls", {})
        if isinstance(mnist_cls, dict) and "label" in mnist_cls:
            try:
                return int(mnist_cls["label"])
            except (TypeError, ValueError):
                return None
    task_groups = label_data.get("task_groups", {})
    if isinstance(task_groups, dict):
        mnist_group = task_groups.get("mnist_cls", {})
        if isinstance(mnist_group, dict):
            label = mnist_group.get("label")
            if label is not None:
                try:
                    return int(label)
                except (TypeError, ValueError):
                    return None
    return None


def plot_sample(
        mesh: trimesh.Trimesh,
        signal: np.ndarray,
        sample_name: str,
        mnist_label: int | None,
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
        ax.view_init(elev=30.0, azim=float(azim))
        label_txt = "?" if mnist_label is None else str(mnist_label)
        ax.set_title(f"{title}: {sample_name} | mnist_cls={label_txt} | view={view_idx}")

        cbar = fig.colorbar(poly, ax=ax, label="Signal Value")
        cbar.ax.set_title("src")
        cbar.ax.text(0.5, -0.08, signal_source, transform=cbar.ax.transAxes, ha="center", va="top", fontsize=8)

        plt.tight_layout()
        suffix = f"_view{view_idx:02d}" if num_views > 1 else ""
        output_path = output_dir / f"{sample_name}{suffix}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    labels_dir = input_dir / "labels"
    meshes_dir = input_dir / args.mesh_subdir
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir / "mnist_plots"

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")
    if not meshes_dir.is_dir():
        raise FileNotFoundError(f"mesh directory not found: {meshes_dir}")
    if not (input_dir / "signals").is_dir():
        raise FileNotFoundError(f"signals directory not found: {input_dir / 'signals'}")

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_names = list(iter_sample_names(labels_dir))
    if not sample_names:
        raise FileNotFoundError(f"No label files found in: {labels_dir}")

    for sample_name in sample_names:
        label_path = labels_dir / f"{sample_name}.json"
        with open(label_path) as fh:
            label_data = json.load(fh)

        if not is_mnist_label(label_data):
            continue

        mesh_path = meshes_dir / f"{sample_name}.obj"
        if not mesh_path.is_file():
            mesh_path = Path(label_data.get("mesh_file", mesh_path))
            if not mesh_path.is_absolute():
                mesh_path = input_dir / mesh_path

        if not mesh_path.is_file():
            print(f"Skipping {sample_name}: mesh file not found")
            continue

        mesh = trimesh.load(str(mesh_path), force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh_list = list(mesh.geometry.values())
            if not mesh_list:
                print(f"Skipping {sample_name}: empty scene mesh")
                continue
            mesh = trimesh.util.concatenate(mesh_list)
        signal, signal_source = load_signal(input_dir, sample_name, mesh, label_data)
        mnist_label = extract_mnist_label(label_data)
        plot_sample(
            mesh=mesh,
            signal=signal,
            sample_name=sample_name,
            mnist_label=mnist_label,
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
        print(f"Saved {args.num_views} view(s) for {sample_name} in {output_dir}")


if __name__ == "__main__":
    main()
