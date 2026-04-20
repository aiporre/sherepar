#!/usr/bin/env python3
"""
examples/demo_stage1.py
=======================

Stage 1 demo: synthetic benchmark pipeline for pmConv.

Demonstrates both usage workflows:

  Workflow A — many deformed surfaces, each saved independently.
  Workflow B — one fixed deformed surface, many different signals resampled
               on the same geometry.

Usage
-----
    # Build the graphop extension first (see BUILD.md), then:
    cd /path/to/sherepar
    python examples/demo_stage1.py --root /tmp/stage1_output --n 5

The script is intentionally kept short; production runs would increase --n.
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# ── Locate the graphop extension ─────────────────────────────────────────────
# The compiled extension (.so) is installed at the repo root by CMake.
# Add the repo root to sys.path so "import graphop" works when running from
# the examples/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherepar.benchmark import SurfaceFactory


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_simple_handles(vertices: np.ndarray,
                        n_handles: int = 4,
                        rng: np.random.Generator = None):
    """Pick evenly-spaced handles and perturb their target positions slightly.

    Returns (handle_ids, target_positions).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    nv = vertices.shape[0]
    step = max(1, nv // n_handles)
    handle_ids = list(range(0, min(nv, n_handles * step), step))[:n_handles]

    # Small random displacement per handle: up to 5% of bounding-box diagonal
    bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    disp = rng.uniform(-0.05 * bbox_diag, 0.05 * bbox_diag, size=(len(handle_ids), 3))
    target_positions = vertices[handle_ids] + disp

    return handle_ids, target_positions


def load_vertices(mesh_path: str) -> np.ndarray:
    """Read OBJ vertex positions (v lines) into a (N,3) array."""
    verts = []
    with open(mesh_path) as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts)


# ── Workflow A ────────────────────────────────────────────────────────────────

def workflow_a(factory: SurfaceFactory,
               template_vertices: np.ndarray,
               n: int,
               seed: int = 42) -> None:
    """Generate *n* differently-deformed surfaces, each with one isotropic signal."""
    print(f"\n=== Workflow A: {n} deformed surfaces ===")
    rng = np.random.default_rng(seed)

    for i in range(n):
        handle_ids, target_positions = make_simple_handles(template_vertices, n_handles=4, rng=rng)

        # Random center vertex for the signal
        center_idx = int(rng.integers(0, template_vertices.shape[0]))

        surface = factory.generate_surface(
            handle_ids       = handle_ids,
            target_positions = target_positions,
            method           = "sre_arap",
            alpha            = 0.02,
            max_iter         = 50,
            signal_type      = "isotropic",
            signal_params    = {
                "center":    center_idx,   # vertex index; resolved after deformation
                "sigma":     0.15,
                "amplitude": 1.0,
            },
            fname = f"wfA_{i:03d}",
        )
        paths = surface.save()
        print(f"  [{i+1}/{n}] saved → {paths['labels']}")


# ── Workflow B ────────────────────────────────────────────────────────────────

def workflow_b(factory: SurfaceFactory,
               template_vertices: np.ndarray,
               n: int,
               seed: int = 99) -> None:
    """Deform once; resample *n* different signals on the same geometry."""
    print(f"\n=== Workflow B: 1 deformed surface, {n} signals ===")
    rng = np.random.default_rng(seed)

    # Fixed deformation
    handle_ids, target_positions = make_simple_handles(template_vertices, n_handles=4, rng=rng)
    surface_template = factory.generate_surface(
        handle_ids       = handle_ids,
        target_positions = target_positions,
        method           = "sre_arap",
        alpha            = 0.02,
        max_iter         = 50,
        fname            = "wfB_template",
    )
    print(f"  Template deformed: {surface_template}")

    V = surface_template.vertices

    # Alternate between isotropic and anisotropic signals
    for i in range(n):
        if i % 2 == 0:
            # Isotropic Gaussian
            center_idx = int(rng.integers(0, V.shape[0]))
            sigma      = float(rng.uniform(0.05, 0.25))
            from spherepar.benchmark.signals import isotropic_gaussian
            sig = isotropic_gaussian(V, V[center_idx], sigma=sigma, amplitude=1.0)
            sig_meta = {
                "family":    "isotropic_gaussian",
                "center":    V[center_idx].tolist(),
                "sigma":     sigma,
                "amplitude": 1.0,
            }
        else:
            # Anisotropic Gaussian
            center_idx = int(rng.integers(0, V.shape[0]))
            sigma_u    = float(rng.uniform(0.05, 0.20))
            sigma_v    = float(rng.uniform(0.02, sigma_u))
            angle      = float(rng.uniform(0, np.pi))
            from spherepar.benchmark.signals import anisotropic_gaussian
            # Estimate normal as direction from centroid to center vertex
            centroid = V.mean(axis=0)
            normal   = V[center_idx] - centroid
            if np.linalg.norm(normal) < 1e-12:
                normal = np.array([0.0, 0.0, 1.0])
            sig = anisotropic_gaussian(
                V, V[center_idx], normal,
                sigma_u=sigma_u, sigma_v=sigma_v,
                amplitude=1.0, orientation_angle=angle
            )
            sig_meta = {
                "family":            "anisotropic_gaussian",
                "center":            V[center_idx].tolist(),
                "normal":            (normal / np.linalg.norm(normal)).tolist(),
                "sigma_u":           sigma_u,
                "sigma_v":           sigma_v,
                "amplitude":         1.0,
                "orientation_angle": angle,
            }

        new_surface = (
            surface_template
            .update_signal(sig, sig_meta)
            .update_fname(suffix=f"_signal_{i:03d}")
        )
        paths = new_surface.save()
        print(f"  [{i+1}/{n}] saved → {paths['labels']}")


# ── Workflow C ────────────────────────────────────────────────────────────────

def workflow_c(factory: SurfaceFactory,
               template_vertices: np.ndarray,
               n: int,
               seed: int = 7) -> None:
    """Rotation-based deformations: per-handle angle + ring_size.

    Each handle is described by:
      vertex_id  — center vertex of the local rotation
      angle      — rotation amount in radians (around the surface normal)
      ring_size  — Euclidean radius; all vertices within this distance of the
                   center vertex become handles and are rotated by *angle*
    """
    print(f"\n=== Workflow C: {n} angle+ring_size deformed surfaces ===")
    rng = np.random.default_rng(seed)
    nv  = template_vertices.shape[0]

    for i in range(n):
        # Pick 2 random center vertices; assign independent angles and rings
        center_ids  = rng.integers(0, nv, size=2).tolist()
        angles      = rng.uniform(-0.4, 0.4, size=2).tolist()
        ring_sizes  = rng.uniform(0.15, 0.45, size=2).tolist()

        handle_transforms = [
            {"vertex_id": int(vid), "angle": float(a), "ring_size": float(r)}
            for vid, a, r in zip(center_ids, angles, ring_sizes)
        ]

        # Random center vertex for the isotropic signal
        center_idx = int(rng.integers(0, nv))

        surface = factory.generate_surface_with_angles(
            handle_transforms = handle_transforms,
            method            = "sre_arap",
            alpha             = 0.02,
            max_iter          = 50,
            signal_type       = "isotropic",
            signal_params     = {"center": center_idx, "sigma": 0.2, "amplitude": 1.0},
            fname             = f"wfC_{i:03d}",
        )
        paths = surface.save()
        # Print the handles so it's clear what parameters were used
        for h in handle_transforms:
            print(f"    handle: vertex={h['vertex_id']:4d}  "
                  f"angle={h['angle']:+.3f} rad  ring={h['ring_size']:.3f}")
        print(f"  [{i+1}/{n}] saved → {paths['labels']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1 synthetic benchmark demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--root",
        default="/tmp/stage1_output",
        help="Root output directory for generated data.",
    )
    p.add_argument(
        "--template",
        default=str(REPO_ROOT / "data" / "ellipsoid.obj"),
        help="Path to the template OBJ mesh.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of samples per workflow.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.template).exists():
        print(f"ERROR: template mesh not found: {args.template}")
        print("Run examples/generate_mesh_elipsoid.py first, or specify --template.")  # existing script
        sys.exit(1)

    print(f"Root output : {args.root}")
    print(f"Template    : {args.template}")
    print(f"Samples (n) : {args.n}")

    factory = SurfaceFactory(root=args.root, template_mesh_path=args.template)
    template_vertices = load_vertices(args.template)
    print(f"Template mesh loaded: {template_vertices.shape[0]} vertices")

    workflow_a(factory, template_vertices, n=args.n)
    workflow_b(factory, template_vertices, n=args.n)
    workflow_c(factory, template_vertices, n=args.n)

    print(f"\nDone! Output written to: {args.root}")
    print("  surfaces/  — OBJ meshes")
    print("  signals/   — NumPy .npy signal arrays")
    print("  labels/    — JSON metadata files")


if __name__ == "__main__":
    main()