#!/usr/bin/env python3
"""
Create a portable dataset structure from mesh files.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MESH_EXTS = {".obj", ".ply", ".stl", ".off"}


@dataclass(frozen=True)
class MeshInput:
    path: Path
    sample_name: str
    dataname: str
    source_root: Path
    source_split: Optional[str] = None
    class_name: Optional[str] = None
    class_id: Optional[int] = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create meshes/signals/labels/logs/spheres from mesh files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-dir",
        default=None,
        help="Input root for generic mode (non-FAUST).",
    )
    input_group.add_argument(
        "--faust-dir",
        default=None,
        help="FAUST root directory. Script reads meshes from <faust-dir>/registrations.",
    )
    input_group.add_argument(
        "--cylinders-dir",
        default=None,
        help="Flat directory containing Cylinders mesh files.",
    )
    input_group.add_argument(
        "--modelnet40-dir",
        default=None,
        help="ModelNet40 root containing <class>/{train,test}/*.off files.",
    )
    parser.add_argument("--output-root", required=True, help="Output dataset root.")
    parser.add_argument(
        "--dataname",
        default="generic",
        help="Dataset mode for --input-dir. Dedicated dataset-root options select their own mode.",
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
    parser.add_argument(
        "--percentage",
        type=float,
        default=100.0,
        help="Percentage of ModelNet40 meshes to retain (1-100); ignored by other modes.",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds for generated splits.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training ratio for generated splits.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio for generated splits.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio for generated splits.")
    parser.add_argument("--split-seed", type=int, default=0, help="Random seed for ModelNet40 sampling and splits.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Regenerate every selected input, even when complete artifacts already exist.",
    )
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


def _safe_sample_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def _flat_mesh_inputs(directory: Path, dataname: str) -> List[MeshInput]:
    return [
        MeshInput(path=path, sample_name=path.stem, dataname=dataname, source_root=directory)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in MESH_EXTS
    ]


def _modelnet40_mesh_inputs(root: Path, percentage: float, seed: int) -> List[MeshInput]:
    """Discover ModelNet40 meshes and optionally sample each class evenly."""
    candidates_by_class: Dict[str, List[MeshInput]] = {}
    class_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    for class_id, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        candidates: List[MeshInput] = []
        for source_split in ("train", "test"):
            split_dir = class_dir / source_split
            if not split_dir.is_dir():
                continue
            for path in sorted(split_dir.glob("*.off")):
                sample_name = "_".join(
                    (_safe_sample_component(class_name), source_split, _safe_sample_component(path.stem))
                )
                candidates.append(
                    MeshInput(
                        path=path,
                        sample_name=sample_name,
                        dataname="MODELNET40",
                        source_root=root,
                        source_split=source_split,
                        class_name=class_name,
                        class_id=class_id,
                    )
                )
        if candidates:
            candidates_by_class[class_name] = candidates

    if percentage == 100.0:
        return [item for class_name in sorted(candidates_by_class) for item in candidates_by_class[class_name]]

    rng = np.random.default_rng(seed)
    selected: List[MeshInput] = []
    for class_name in sorted(candidates_by_class):
        candidates = candidates_by_class[class_name]
        count = max(1, int(round(len(candidates) * percentage / 100.0)))
        selected.extend(candidates[int(i)] for i in rng.permutation(len(candidates))[:count])
    return sorted(selected, key=lambda item: str(item.path))


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
            _list_completed_samples,
            append_error_log,
            genus_zero_filter_reason,
            save_sample_mesh,
            save_spherical_parametrization,
        )
        from spherepar.benchmark.splits import TASK_MODELNET40_CLS, build_task_splits
    except ModuleNotFoundError as exc:
        print(
            "ERROR: missing Python dependency while importing spherepar modules. "
            f"Install project requirements first. Details: {exc}"
        )
        return 1

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    output_root = Path(args.output_root).expanduser().resolve()
    faust_dir = Path(args.faust_dir).expanduser().resolve() if args.faust_dir else None
    cylinders_dir = Path(args.cylinders_dir).expanduser().resolve() if args.cylinders_dir else None
    modelnet40_dir = Path(args.modelnet40_dir).expanduser().resolve() if args.modelnet40_dir else None

    if all(path is None for path in (input_dir, faust_dir, cylinders_dir, modelnet40_dir)):
        print("ERROR: provide one input root: --input-dir, --faust-dir, --cylinders-dir, or --modelnet40-dir.")
        return 1
    if not 1.0 <= args.percentage <= 100.0:
        print("ERROR: --percentage must be in the range [1, 100].")
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
    mode: str
    mesh_inputs: List[MeshInput]
    if faust_dir is not None:
        if not faust_dir.is_dir():
            print(f"ERROR: FAUST directory not found: {faust_dir}")
            return 1
        faust_registrations_dir = faust_dir / "registrations"
        if not faust_registrations_dir.is_dir():
            print(f"ERROR: FAUST registrations dir not found: {faust_registrations_dir}")
            return 1
        mode = "FAUST"
        mesh_inputs = _flat_mesh_inputs(faust_registrations_dir, mode)
    elif cylinders_dir is not None:
        if not cylinders_dir.is_dir():
            print(f"ERROR: Cylinders directory not found: {cylinders_dir}")
            return 1
        mode = "CYLINDERS"
        mesh_inputs = _flat_mesh_inputs(cylinders_dir, mode)
    elif modelnet40_dir is not None:
        if not modelnet40_dir.is_dir():
            print(f"ERROR: ModelNet40 directory not found: {modelnet40_dir}")
            return 1
        mode = "MODELNET40"
        mesh_inputs = _modelnet40_mesh_inputs(modelnet40_dir, float(args.percentage), int(args.split_seed))
    else:
        if input_dir is None or not input_dir.is_dir():
            print(f"ERROR: input directory not found: {input_dir}")
            return 1
        mode = args.dataname.strip().upper()
        mesh_inputs = _flat_mesh_inputs(input_dir, mode)

    if not mesh_inputs:
        location = (
            faust_dir / "registrations"
            if faust_dir is not None
            else cylinders_dir if cylinders_dir is not None else modelnet40_dir if modelnet40_dir is not None else input_dir
        )
        print(f"No mesh files ({', '.join(sorted(MESH_EXTS))}) found in {location}")
        return 0

    total = len(mesh_inputs)
    if args.resume:
        completed_samples, artifact_counts = _list_completed_samples(str(output_root))
        planned_sample_ids = {item.sample_name for item in mesh_inputs}
        completed_for_request = planned_sample_ids & set(completed_samples)
        print(
            "Resume scan: "
            f"meshes={artifact_counts['meshes']}, "
            f"spheres={artifact_counts['spheres']}, "
            f"signals={artifact_counts['signals']}, "
            f"labels={artifact_counts['labels']}, "
            f"complete={len(completed_samples)}"
        )
        print(
            "Resume decision: "
            f"matched_current_request={len(completed_for_request)}, "
            f"regenerate={total - len(completed_for_request)}."
        )
    else:
        completed_for_request = set()
        print("Resume disabled: regenerating the full selected input plan.")
    saved = 0
    failed = 0
    skipped = 0
    filtered = 0

    print("=" * 68)
    print("Create dataset structure from mesh files")
    print("=" * 68)
    if mode == "FAUST":
        print(f"FAUST dir       : {faust_dir}")
        print(f"Input meshes    : {faust_dir / 'registrations'}")
    elif mode == "CYLINDERS":
        print(f"Cylinders dir   : {cylinders_dir}")
    elif mode == "MODELNET40":
        print(f"ModelNet40 dir  : {modelnet40_dir}")
        print(f"Percentage      : {args.percentage}")
    else:
        print(f"Input dir       : {input_dir}")
    print(f"Output root     : {output_root}")
    print(f"Found meshes    : {total}")
    print(f"Dataset mode    : {mode.lower()}")
    print(f"Param method    : {args.param_method}")
    print("Filter non-g0   : enabled (required for spherical parametrization)")
    print(f"Resume          : {args.resume}")
    print(f"Overwrite       : {args.overwrite}")
    print("=" * 68)

    for idx, mesh_input in tqdm(enumerate(mesh_inputs, start=1), total=len(mesh_inputs), desc="Creating dataset", unit="meshes"):
        mesh_src_path = mesh_input.path
        sample_name = mesh_input.sample_name
        label_path = output_dirs["labels"] / f"{sample_name}.json"
        signal_path = output_dirs["signals"] / f"{sample_name}.npy"
        mesh_out_path = output_dirs["meshes"] / f"{sample_name}.obj"
        sphere_out_path = output_dirs["spheres"] / f"{sample_name}.obj"
        spherical_label_path = output_dirs["labels"] / f"{sample_name}_spherical.json"

        if args.resume and not args.overwrite and sample_name in completed_for_request:
            print(f"[{idx}/{total}] skip {sample_name} (complete artifacts found)")
            skipped += 1
            continue

        try:
            mesh = _load_mesh(mesh_src_path)
            topology_reason = genus_zero_filter_reason(mesh)
            if topology_reason is not None:
                print(f"[{idx}/{total}] filtered {sample_name}: {topology_reason}")
                filtered += 1
                continue

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
                dataname=mesh_input.dataname,
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
                    "dataname": mesh_input.dataname,
                    "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source_mesh": _resolve_relative(mesh_src_path, mesh_input.source_root),
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
                "tasks": (
                    {
                        TASK_MODELNET40_CLS: {
                            "valid": True,
                            "label": mesh_input.class_id,
                            "class_name": mesh_input.class_name,
                        }
                    }
                    if mesh_input.class_id is not None
                    else {}
                ),
                "deformation": {"type": "none"},
                "parametrization": {
                    "method": args.param_method,
                    "success": bool(param_success),
                    "error": param_error,
                },
                "warnings": warnings,
            }

            if mesh_input.class_name is not None:
                label["metadata"].update(
                    {
                        "class_name": mesh_input.class_name,
                        "class_id": mesh_input.class_id,
                        "source_split": mesh_input.source_split,
                    }
                )

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

    if mode == "MODELNET40":
        try:
            build_task_splits(
                dataset_root=str(output_root),
                tasks=[TASK_MODELNET40_CLS],
                num_folds=int(args.num_folds),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                test_ratio=float(args.test_ratio),
                seed=int(args.split_seed),
                group_by_template=False,
                modelnet40_native_split=float(args.percentage) == 100.0,
            )
        except Exception as exc:  # noqa: BLE001
            append_error_log(str(log_path), "split_builder", f"split generation failed: {exc}")
            print(f"ModelNet40 split generation failed: {exc}")
            failed += 1

    print("=" * 68)
    print(
        f"Done. total={total}, saved={saved}, skipped={skipped}, filtered={filtered}, failed={failed}, "
        f"log={_resolve_relative(log_path, output_root)}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
