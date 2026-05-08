#!/usr/bin/env python3
"""
Create a portable dataset structure from meshes.

The script creates:
  meshes/, signals/, labels/, logs/, spheres/

For each input mesh:
  1) save/copy mesh to meshes/<name>.obj
  2) compute spherical parametrization to spheres/<name>.obj
  3) generate signal:
     - default: zeros
     - --dataname FAUST: vertex-index signal (0..N-1)
  4) write labels/<name>.json with relative paths
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MESH_EXTS = {".obj", ".ply", ".stl", ".off"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create meshes/signals/labels/logs/spheres from mesh files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input root for generic mode (non-FAUST).",
    )
    parser.add_argument(
        "--faust-dir",
        default=None,
        help="FAUST root directory. Script reads meshes from <faust-dir>/registrations.",
    )
    parser.add_argument("--output-root", required=True, help="Output dataset root.")
    parser.add_argument(
        "--dataname",
        default="generic",
        help="Dataset mode. Use 'FAUST' for FAUST vertex-index signals.",
    )
    parser.add_argument(
        "--param-method",
        choices=["flash", "cem"],
        default="flash",
        help="Spherical parametrization method.",
    )
    parser.add_argument("--cem-eps", type=float, default=1e-6, help="CEM convergence tolerance.")
    parser.add_argument("--cem-max-iters", type=int, default=100, help="Maximum CEM iterations.")
    parser.add_argument("--cem-verbose", action="store_true", help="Verbose CEM output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample artifacts.")
    return parser


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            raise ValueError("Scene has no geometries")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded object is not a triangle mesh")
    if len(mesh.faces) == 0:
        raise ValueError("Mesh has zero faces")
    return mesh


def _resolve_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_faust_id(stem: str) -> Optional[str]:
    patterns = [
        r"^tr_(?:reg|scan)_(\d+)$",
        r"^(?:reg|scan)_(\d+)$",
        r".*?(\d+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, stem)
        if match:
            return match.group(1)
    return None


def _build_signal(
    *,
    dataname: str,
    sample_name: str,
    mesh_vertices: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    warnings: List[str] = []
    dataname_norm = dataname.strip().upper()
    n_vertices = int(mesh_vertices.shape[0])

    if dataname_norm != "FAUST":
        signal = np.zeros(n_vertices, dtype=np.float32)
        meta = {
            "type": "zeros",
            "dtype": "float32",
            "shape": [int(n_vertices)],
            "masked_by_gt": False,
            "invalid_value": None,
        }
        return signal, meta, warnings

    faust_id = _extract_faust_id(sample_name)
    signal = np.arange(n_vertices, dtype=np.float32)

    meta = {
        "type": "faust_vertex_index",
        "dtype": "float32",
        "shape": [int(n_vertices)],
        "faust_id": faust_id,
        "masked_by_gt": False,
        "invalid_value": None,
    }
    return signal, meta, warnings


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        from spherepar.benchmark.dataset_generator import (
            append_error_log,
            save_sample_mesh,
            save_spherical_parametrization,
        )
    except ModuleNotFoundError as exc:
        print(
            "ERROR: missing Python dependency while importing spherepar modules. "
            f"Install project requirements first. Details: {exc}"
        )
        return 1

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    output_root = Path(args.output_root).expanduser().resolve()
    faust_dir = Path(args.faust_dir).expanduser().resolve() if args.faust_dir else None
    dataname_norm = "FAUST" if faust_dir is not None else args.dataname.strip().upper()

    if faust_dir is None and input_dir is None:
        print("ERROR: provide --input-dir (generic mode) or --faust-dir (FAUST mode).")
        return 1

    output_dirs = {
        "meshes": output_root / "meshes",
        "signals": output_root / "signals",
        "labels": output_root / "labels",
        "logs": output_root / "logs",
        "spheres": output_root / "spheres",
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    log_path = output_dirs["logs"] / "errors.log"
    if dataname_norm == "FAUST":
        if faust_dir is None:
            print("ERROR: FAUST mode requires --faust-dir.")
            return 1
        if not faust_dir.is_dir():
            print(f"ERROR: FAUST directory not found: {faust_dir}")
            return 1
        faust_registrations_dir = faust_dir / "registrations"
        if not faust_registrations_dir.is_dir():
            print(f"ERROR: FAUST registrations dir not found: {faust_registrations_dir}")
            return 1
        mesh_files = sorted(
            [p for p in faust_registrations_dir.iterdir() if p.is_file() and p.suffix.lower() in MESH_EXTS]
        )
    else:
        if input_dir is None or not input_dir.is_dir():
            print(f"ERROR: input directory not found: {input_dir}")
            return 1
        mesh_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in MESH_EXTS])

    if not mesh_files:
        location = (faust_dir / "registrations") if dataname_norm == "FAUST" else input_dir
        print(f"No mesh files ({', '.join(sorted(MESH_EXTS))}) found in {location}")
        return 0

    total = len(mesh_files)
    saved = 0
    failed = 0
    skipped = 0

    print("=" * 68)
    print("Create dataset structure from OBJ")
    print("=" * 68)
    if dataname_norm == "FAUST":
        print(f"FAUST dir       : {faust_dir}")
        print(f"Input meshes    : {faust_dir / 'registrations'}")
    else:
        print(f"Input dir       : {input_dir}")
    print(f"Output root     : {output_root}")
    print(f"Found meshes    : {total}")
    print(f"Dataset mode    : {dataname_norm.lower()}")
    print(f"Param method    : {args.param_method}")
    print(f"Overwrite       : {args.overwrite}")
    print("=" * 68)

    for idx, mesh_src_path in enumerate(mesh_files, start=1):
        sample_name = mesh_src_path.stem
        label_path = output_dirs["labels"] / f"{sample_name}.json"
        signal_path = output_dirs["signals"] / f"{sample_name}.npy"
        mesh_out_path = output_dirs["meshes"] / f"{sample_name}.obj"
        sphere_out_path = output_dirs["spheres"] / f"{sample_name}.obj"
        spherical_label_path = output_dirs["labels"] / f"{sample_name}_spherical.json"

        if label_path.exists() and not args.overwrite:
            print(f"[{idx}/{total}] skip {sample_name} (label exists)")
            skipped += 1
            continue

        try:
            mesh = _load_mesh(mesh_src_path)
            save_sample_mesh(root=str(output_root), name=sample_name, mesh=mesh)

            sphere_rel: Optional[str] = None
            spherical_label_rel: Optional[str] = None
            param_error: Optional[str] = None
            param_success = False
            try:
                sphere_paths = save_spherical_parametrization(
                    root=str(output_root),
                    name=sample_name,
                    vertices=np.asarray(mesh.vertices, dtype=np.float64),
                    faces=np.asarray(mesh.faces, dtype=np.int32),
                    method=args.param_method,
                    cem_eps=float(args.cem_eps),
                    cem_max_iters=int(args.cem_max_iters),
                    cem_verbose=bool(args.cem_verbose),
                )
                sphere_rel = sphere_paths.get("sphere")
                spherical_label_rel = sphere_paths.get("spherical_label")
                param_success = True
            except Exception as exc:  # noqa: BLE001
                param_error = str(exc)
                append_error_log(
                    str(log_path),
                    sample_name,
                    f"spherical parametrization failed: {exc}",
                    template_id=sample_name,
                    deformation_case="case1_no",
                    traceback_text=traceback.format_exc(),
                )

            signal, signal_meta, warnings = _build_signal(
                dataname=args.dataname,
                sample_name=sample_name,
                mesh_vertices=np.asarray(mesh.vertices, dtype=np.float32),
            )
            np.save(str(signal_path), signal)

            mesh_rel = _resolve_relative(mesh_out_path, output_root)
            signal_rel = _resolve_relative(signal_path, output_root)
            label_rel = _resolve_relative(label_path, output_root)
            sphere_rel_fallback = _resolve_relative(sphere_out_path, output_root)
            spherical_label_rel_fallback = _resolve_relative(spherical_label_path, output_root)

            label: Dict[str, Any] = {
                "schema_version": "0.3",
                "sample_id": sample_name,
                "name": sample_name,
                "template_id": sample_name,
                "deformation_case": "case1_no",
                "random_seed": 0,
                "metadata": {
                    "dataname": args.dataname,
                    "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source_mesh": str(mesh_src_path.name),
                },
                "paths": {
                    "mesh": mesh_rel,
                    "signal": signal_rel,
                    "label": label_rel,
                    "sphere": sphere_rel if sphere_rel is not None else sphere_rel_fallback,
                    "spherical_label": (
                        spherical_label_rel if spherical_label_rel is not None else spherical_label_rel_fallback
                    ),
                },
                "mesh_path": mesh_rel,
                "signal_path": signal_rel,
                "label_path": label_rel,
                "sphere_path": sphere_rel if sphere_rel is not None else sphere_rel_fallback,
                "mesh": {
                    "n_vertices": int(mesh.vertices.shape[0]),
                    "n_faces": int(mesh.faces.shape[0]),
                },
                "signal": {
                    "num_centers": 0,
                    "centers": [],
                    "center_vertex_ids": [],
                    "sigmas": [],
                    "amplitudes": [],
                    "family": signal_meta.get("type"),
                    "meta": signal_meta,
                },
                "signal_files": {"main": signal_rel},
                "signals": [
                    {
                        "signal_id": "main",
                        "family": signal_meta.get("type"),
                        "storage": {
                            "path_key": "main",
                            "dtype": signal_meta.get("dtype"),
                            "shape": signal_meta.get("shape"),
                        },
                    }
                ],
                "deformation": {"type": "none"},
                "parametrization": {
                    "method": args.param_method,
                    "success": bool(param_success),
                    "error": param_error,
                },
                "warnings": warnings,
            }

            with open(label_path, "w") as fh:
                json.dump(label, fh, indent=2)

            print(f"[{idx}/{total}] saved {sample_name} -> {label_rel}")
            saved += 1
        except Exception as exc:  # noqa: BLE001
            append_error_log(
                str(log_path),
                sample_name,
                f"sample generation failed: {exc}",
                template_id=sample_name,
                deformation_case="case1_no",
                traceback_text=traceback.format_exc(),
            )
            print(f"[{idx}/{total}] failed {sample_name}: {exc}")
            failed += 1

    print("=" * 68)
    print(
        f"Done. total={total}, saved={saved}, skipped={skipped}, failed={failed}, "
        f"log={_resolve_relative(log_path, output_root)}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
