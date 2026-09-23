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
import csv
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
    participant_id: Optional[str] = None
    diagnosis_sc: Optional[str] = None
    session: Optional[str] = None
    session_number: Optional[int] = None
    hip: Optional[str] = None


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
        "--adni-participants",
        default=None,
        help="ADNI participants.tsv manifest. Enables ADNI clinical classification labels and participant splits.",
    )
    parser.add_argument(
        "--adni-session",
        choices=("none", "first", "last"),
        default="none",
        help="ADNI session selection per participant (all, earliest, or latest numeric session).",
    )
    parser.add_argument(
        "--adni-hip",
        choices=("both", "left", "right"),
        default="both",
        help="ADNI hip side filter.",
    )
    parser.add_argument(
        "--param-method",
        choices=["flash", "cem", "spheremap"],
        default="flash",
        help="Spherical parametrization method (SphereMap uses MoebiusRegistration).",
    )
    parser.add_argument("--cem-eps", type=float, default=1e-6, help="CEM convergence tolerance.")
    parser.add_argument("--cem-max-iters", type=int, default=100, help="Maximum CEM iterations.")
    parser.add_argument("--cem-radius", type=float, default=1.2, help="CEM stereographic partition radius.")
    parser.add_argument("--use-idt-remesh", action="store_true", help="Use fixed-vertex IDT connectivity for CEM.")
    parser.add_argument("--adaptive-radius", action="store_true", help="Search fallback radii after a collapsed CEM result.")
    parser.add_argument(
        "--cem-radius-candidates",
        type=lambda value: tuple(float(item.strip()) for item in value.split(",") if item.strip()),
        default=(1.1, 1.3, 1.4, 1.5),
        help="Comma-separated fallback CEM radii.",
    )
    parser.add_argument("--reject-retry", action="store_true", help="Reject maps above the collapse limit after retrying.")
    parser.add_argument("--cem-max-attempts", type=int, default=5, help="Maximum total CEM radius attempts.")
    parser.add_argument("--cem-max-collapsed-faces", type=int, default=0, help="Accepted collapsed-face limit.")
    parser.add_argument("--cem-verbose", action="store_true", help="Verbose CEM output.")
    parser.add_argument(
        "--force-outward-winding",
        action="store_true",
        help="For CEM, reverse every inward spherical face (mesh-repair mode; hides fold diagnostics).",
    )
    parser.add_argument("--spheremap-binary", default=None, help="Path to the MoebiusRegistration SphereMap binary.")
    parser.add_argument("--spheremap-repository", default=None, help="MoebiusRegistration repository containing spheremap/ and Bin/Linux/SphereMap.")
    parser.add_argument("--spheremap-auto-build", action="store_true", help="Build SphereMap if the binary is missing.")
    parser.add_argument("--spheremap-iters", type=int, default=25, help="SphereMap CMCF iteration count.")
    parser.add_argument("--spheremap-step-size", type=float, default=1.0, help="SphereMap CMCF step size.")
    parser.add_argument("--spheremap-threads", type=int, default=4, help="SphereMap worker threads.")
    parser.add_argument("--spheremap-no-center", action="store_true", help="Disable SphereMap's built-in Möbius centering.")
    parser.add_argument("--spheremap-degree", type=int, default=4, help="Optional SphereMap spherical-harmonic centering degree.")
    parser.add_argument("--spheremap-a-steps", type=int, default=10, help="SphereMap Möbius-centering line-search steps.")
    parser.add_argument("--spheremap-a-step-size", type=float, default=0.05, help="SphereMap Möbius-centering step size.")
    parser.add_argument("--spheremap-poincare-max-norm", type=float, default=2.0, help="SphereMap Poincare-map maximum norm.")
    parser.add_argument("--spheremap-c2i", type=int, default=0, help="SphereMap C2I mode.")
    parser.add_argument("--spheremap-gss-tolerance", type=float, default=1e-6, help="SphereMap golden-section tolerance.")
    parser.add_argument("--spheremap-lump", action="store_true", help="Use lumped SphereMap mass matrix.")
    parser.add_argument("--spheremap-verbose", action="store_true", help="Enable SphereMap verbose output.")
    parser.add_argument(
        "--mobius-center",
        action="store_true",
        help="Apply area-weighted Möbius centering after CEM or SphereMap (requires --param-method cem or spheremap).",
    )
    parser.add_argument(
        "--anchor-diagnostics",
        action="store_true",
        help="Analyze CEM collapse by mesh-hop distance from Algorithm 4.1's anchor face.",
    )
    parser.add_argument(
        "--anchor-strategy",
        choices=("regular", "central_regular"),
        default="regular",
        help="Deterministic CEM Algorithm 4.1 anchor selection strategy.",
    )
    parser.add_argument(
        "--anchor-regularity-percentile",
        type=float,
        default=10.0,
        help="Inclusive normalized-regularity candidate percentile for central_regular.",
    )
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
        "--create-splits",
        action="store_true",
        help="Create task folds after generation (ADNI always creates clinical folds when a manifest is supplied).",
    )
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


ADNI_DIAGNOSIS_TO_CLASS = {
    "CN": ("CN", 0),
    "SMC": ("CN", 0),
    "EMCI": ("MCI", 1),
    "LMCI": ("MCI", 1),
    "MCI": ("MCI", 1),
    "AD": ("AD", 2),
}
_ADNI_FILENAME_RE = re.compile(
    r"^(?P<participant>sub-[A-Za-z0-9]+)-ses-(?P<session>M\d+)_hip_(?P<hip>left|right)$",
    re.IGNORECASE,
)


def _load_adni_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    """Load the local ADNI participant manifest without copying its contents anywhere."""
    if not path.is_file():
        raise FileNotFoundError(f"ADNI participants manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "participant_id" not in reader.fieldnames or "diagnosis_sc" not in reader.fieldnames:
            raise ValueError("ADNI manifest must contain participant_id and diagnosis_sc columns")
        manifest: Dict[str, Dict[str, str]] = {}
        for row in reader:
            participant = (row.get("participant_id") or "").strip()
            diagnosis = (row.get("diagnosis_sc") or "").strip().upper()
            if not participant or diagnosis not in ADNI_DIAGNOSIS_TO_CLASS:
                continue
            manifest[participant] = {"participant_id": participant, "diagnosis_sc": diagnosis}
    if not manifest:
        raise ValueError(f"ADNI manifest contains no supported diagnosis rows: {path}")
    return manifest


def _adni_mesh_inputs(
    root: Path,
    manifest_path: Path,
    session_filter: str,
    hip_filter: str,
) -> Tuple[List[MeshInput], List[str]]:
    manifest = _load_adni_manifest(manifest_path)
    candidates: List[MeshInput] = []
    skipped: List[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MESH_EXTS:
            continue
        match = _ADNI_FILENAME_RE.match(path.stem)
        if match is None:
            skipped.append(f"{path.name}: filename does not match sub-<id>-ses-Mxxx_hip_<left|right>")
            continue
        participant = match.group("participant")
        row = manifest.get(participant)
        if row is None:
            skipped.append(f"{path.name}: participant {participant} absent from manifest")
            continue
        hip = match.group("hip").lower()
        session = match.group("session").upper()
        session_number = int(session[1:])
        class_name, class_id = ADNI_DIAGNOSIS_TO_CLASS[row["diagnosis_sc"]]
        candidates.append(
            MeshInput(
                path=path,
                sample_name=path.stem,
                dataname="ADNI",
                source_root=root,
                class_name=class_name,
                class_id=class_id,
                participant_id=participant,
                diagnosis_sc=row["diagnosis_sc"],
                session=f"ses-{session}",
                session_number=session_number,
                hip=hip,
            )
        )
    if session_filter != "none":
        by_participant: Dict[str, List[MeshInput]] = {}
        for item in candidates:
            by_participant.setdefault(str(item.participant_id), []).append(item)
        candidates = [
            item
            for participant_items in by_participant.values()
            for item in participant_items
            if item.session_number == (
                min(x.session_number for x in participant_items)
                if session_filter == "first"
                else max(x.session_number for x in participant_items)
            )
        ]
    if hip_filter != "both":
        candidates = [item for item in candidates if item.hip == hip_filter]
    return sorted(candidates, key=lambda item: item.sample_name), skipped


def _adni_request_matches(
    label: Dict[str, Any],
    manifest: Optional[Path],
    output_root: Path,
    session_filter: str,
    hip_filter: str,
) -> bool:
    metadata = label.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    if metadata.get("adni_session_filter") != session_filter or metadata.get("adni_hip_filter") != hip_filter:
        return False
    if manifest is not None and metadata.get("adni_manifest") != _resolve_relative(manifest, output_root):
        return False
    return True


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
    if args.mobius_center and args.param_method not in ("cem", "spheremap"):
        print("ERROR: --mobius-center requires --param-method cem or spheremap.")
        return 1
    if args.anchor_diagnostics and args.param_method != "cem":
        print("ERROR: --anchor-diagnostics requires --param-method cem.")
        return 1
    if args.force_outward_winding and args.param_method != "cem":
        print("ERROR: --force-outward-winding requires --param-method cem.")
        return 1
    if (args.use_idt_remesh or args.adaptive_radius or args.reject_retry) and args.param_method != "cem":
        print("ERROR: CEM Phase 2 options require --param-method cem.")
        return 1
    if not np.isfinite(args.cem_radius) or args.cem_radius <= 0.0:
        print("ERROR: --cem-radius must be finite and positive.")
        return 1
    if args.cem_max_attempts < 1:
        print("ERROR: --cem-max-attempts must be at least 1.")
        return 1
    if args.cem_max_collapsed_faces < 0:
        print("ERROR: --cem-max-collapsed-faces must be non-negative.")
        return 1
    if not args.cem_radius_candidates or any(
        not np.isfinite(value) or value <= 0.0 for value in args.cem_radius_candidates
    ):
        print("ERROR: --cem-radius-candidates must contain finite positive values.")
        return 1
    if (
        not np.isfinite(args.anchor_regularity_percentile)
        or not 0.0 <= args.anchor_regularity_percentile <= 100.0
    ):
        print("ERROR: --anchor-regularity-percentile must be finite and in [0, 100].")
        return 1
    try:
        from spherepar.benchmark.dataset_generator import (
            _list_completed_samples,
            append_error_log,
            genus_zero_filter_reason,
            parametrization_request_matches,
            save_sample_mesh,
            save_spherical_parametrization,
        )
        from spherepar.benchmark.splits import TASK_ADNI_CLS, TASK_MODELNET40_CLS, build_task_splits
    except ModuleNotFoundError as exc:
        print(
            "ERROR: missing Python dependency while importing spherepar modules. "
            f"Install project requirements first. Details: {exc}"
        )
        return 1

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    adni_manifest = Path(args.adni_participants).expanduser().resolve() if args.adni_participants else None
    output_root = Path(args.output_root).expanduser().resolve()
    faust_dir = Path(args.faust_dir).expanduser().resolve() if args.faust_dir else None
    cylinders_dir = Path(args.cylinders_dir).expanduser().resolve() if args.cylinders_dir else None
    modelnet40_dir = Path(args.modelnet40_dir).expanduser().resolve() if args.modelnet40_dir else None

    if all(path is None for path in (input_dir, faust_dir, cylinders_dir, modelnet40_dir)):
        print("ERROR: provide one input root: --input-dir, --faust-dir, --cylinders-dir, or --modelnet40-dir.")
        return 1
    if adni_manifest is not None and input_dir is None:
        print("ERROR: --adni-participants requires --input-dir pointing to ADNI mesh files.")
        return 1
    if adni_manifest is not None and any(path is not None for path in (faust_dir, cylinders_dir, modelnet40_dir)):
        print("ERROR: --adni-participants is only supported with --input-dir.")
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
    adni_skipped: List[str] = []
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
        if adni_manifest is not None:
            try:
                mode = "ADNI"
                mesh_inputs, adni_skipped = _adni_mesh_inputs(
                    input_dir,
                    adni_manifest,
                    args.adni_session,
                    args.adni_hip,
                )
            except (OSError, ValueError) as exc:
                print(f"ERROR: could not load ADNI inputs: {exc}")
                return 1
        else:
            mode = args.dataname.strip().upper()
            mesh_inputs = _flat_mesh_inputs(input_dir, mode)

    if adni_skipped:
        print(f"ADNI discovery skipped {len(adni_skipped)} file(s) (see {log_path})")
        for message in adni_skipped:
            append_error_log(str(log_path), "adni_discovery", message)

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
        completed_for_request = {
            sample_id for sample_id in planned_sample_ids & set(completed_samples)
            if parametrization_request_matches(
                completed_samples[sample_id],
                args.param_method,
                bool(args.mobius_center),
                float(args.cem_radius),
                bool(args.anchor_diagnostics),
                args.anchor_strategy,
                float(args.anchor_regularity_percentile),
                bool(args.use_idt_remesh),
                bool(args.adaptive_radius),
                args.cem_radius_candidates,
                bool(args.reject_retry),
                int(args.cem_max_attempts),
                int(args.cem_max_collapsed_faces),
                args.spheremap_binary,
                args.spheremap_repository,
                bool(args.spheremap_auto_build),
                int(args.spheremap_iters),
                float(args.spheremap_step_size),
                int(args.spheremap_threads),
                bool(args.spheremap_no_center),
                args.spheremap_degree,
                args.spheremap_a_steps,
                args.spheremap_a_step_size,
                args.spheremap_poincare_max_norm,
                args.spheremap_c2i,
                args.spheremap_gss_tolerance,
                bool(args.spheremap_lump),
                bool(args.spheremap_verbose),
                bool(args.force_outward_winding),
            )
            and (
                mode != "ADNI"
                or _adni_request_matches(
                    completed_samples[sample_id],
                    adni_manifest,
                    output_root,
                    args.adni_session,
                    args.adni_hip,
                )
            )
        }
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
    if mode == "FAUST" and faust_dir is not None:
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
    if mode == "ADNI":
        print(f"ADNI manifest   : {adni_manifest}")
        print(f"ADNI session    : {args.adni_session}")
        print(f"ADNI hip        : {args.adni_hip}")
    print(f"Param method    : {args.param_method}")
    print(f"CEM radius      : {args.cem_radius}")
    print(f"Möbius center  : {args.mobius_center}")
    print(f"Anchor diagnose: {args.anchor_diagnostics}")
    print(f"Anchor strategy: {args.anchor_strategy}")
    print(f"Anchor pctile  : {args.anchor_regularity_percentile}")
    print(f"IDT remesh     : {args.use_idt_remesh}")
    print(f"Adaptive radius: {args.adaptive_radius}")
    print(f"Reject/retry   : {args.reject_retry}")
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
            sphere_paths: Dict[str, Any] = {}
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
                    cem_radius=float(args.cem_radius),
                    mobius_center=bool(args.mobius_center),
                    force_outward_winding=bool(args.force_outward_winding),
                    anchor_diagnostics=bool(args.anchor_diagnostics),
                    anchor_strategy=args.anchor_strategy,
                    anchor_regularity_percentile=float(args.anchor_regularity_percentile),
                    use_idt_remesh=bool(args.use_idt_remesh),
                    adaptive_radius=bool(args.adaptive_radius),
                    cem_radius_candidates=args.cem_radius_candidates,
                    reject_retry=bool(args.reject_retry),
                    cem_max_attempts=int(args.cem_max_attempts),
                    cem_max_collapsed_faces=int(args.cem_max_collapsed_faces),
                    spheremap_binary=args.spheremap_binary,
                    spheremap_repository=args.spheremap_repository,
                    spheremap_auto_build=bool(args.spheremap_auto_build),
                    spheremap_iters=int(args.spheremap_iters),
                    spheremap_step_size=float(args.spheremap_step_size),
                    spheremap_threads=int(args.spheremap_threads),
                    spheremap_no_center=bool(args.spheremap_no_center),
                    spheremap_degree=args.spheremap_degree,
                    spheremap_a_steps=args.spheremap_a_steps,
                    spheremap_a_step_size=args.spheremap_a_step_size,
                    spheremap_poincare_max_norm=args.spheremap_poincare_max_norm,
                    spheremap_c2i=args.spheremap_c2i,
                    spheremap_gss_tolerance=args.spheremap_gss_tolerance,
                    spheremap_lump=bool(args.spheremap_lump),
                    spheremap_verbose=bool(args.spheremap_verbose),
                    log_path=str(log_path),
                    template_id=sample_name,
                    deformation_case="case1_no",
                )
                sphere_rel = sphere_paths.get("sphere")
                spherical_label_rel = sphere_paths.get("spherical_label")
                param_success = bool(sphere_paths.get("parametrization_success", True))
                param_error = sphere_paths.get("parametrization_error")
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
                    "mobius_center": bool(args.mobius_center),
                    "force_outward_winding": bool(args.force_outward_winding),
                    "cem_radius": float(args.cem_radius) if args.param_method == "cem" else None,
                    "anchor_diagnostics": bool(args.anchor_diagnostics),
                    "anchor_strategy": args.anchor_strategy if args.param_method == "cem" else None,
                    "anchor_regularity_percentile": (
                        float(args.anchor_regularity_percentile)
                        if args.param_method == "cem" else None
                    ),
                    "use_idt_remesh": bool(args.use_idt_remesh),
                    "adaptive_radius": bool(args.adaptive_radius),
                    "cem_radius_candidates": list(args.cem_radius_candidates),
                    "reject_retry": bool(args.reject_retry),
                    "cem_max_attempts": int(args.cem_max_attempts),
                    "cem_max_collapsed_faces": int(args.cem_max_collapsed_faces),
                    "spheremap_binary": args.spheremap_binary,
                    "spheremap_repository": args.spheremap_repository,
                    "spheremap_auto_build": bool(args.spheremap_auto_build),
                    "spheremap_iters": int(args.spheremap_iters),
                    "spheremap_step_size": float(args.spheremap_step_size),
                    "spheremap_threads": int(args.spheremap_threads),
                    "spheremap_no_center": bool(args.spheremap_no_center),
                    "spheremap_degree": args.spheremap_degree,
                    "spheremap_a_steps": args.spheremap_a_steps,
                    "spheremap_a_step_size": args.spheremap_a_step_size,
                    "spheremap_poincare_max_norm": args.spheremap_poincare_max_norm,
                    "spheremap_c2i": args.spheremap_c2i,
                    "spheremap_gss_tolerance": args.spheremap_gss_tolerance,
                    "spheremap_lump": bool(args.spheremap_lump),
                    "spheremap_verbose": bool(args.spheremap_verbose),
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
                        (TASK_ADNI_CLS if mode == "ADNI" else TASK_MODELNET40_CLS): {
                            "valid": True,
                            "label": mesh_input.class_id,
                            "class_name": mesh_input.class_name,
                            **(
                                {
                                    "participant_id": mesh_input.participant_id,
                                    "diagnosis_sc": mesh_input.diagnosis_sc,
                                    "session": mesh_input.session,
                                    "hip": mesh_input.hip,
                                }
                                if mode == "ADNI"
                                else {}
                            ),
                        }
                    }
                    if mesh_input.class_id is not None
                    else {}
                ),
                "deformation": {"type": "none"},
                "parametrization": {
                    "method": args.param_method,
                    "cem_radius": float(args.cem_radius) if args.param_method == "cem" else None,
                    "cem_selected_radius": sphere_paths.get("cem_selected_radius") if sphere_rel else None,
                    "face_winding_correction": sphere_paths.get("face_winding_correction") if sphere_rel else None,
                    "mobius_center": bool(args.mobius_center),
                    "force_outward_winding": bool(args.force_outward_winding),
                    "anchor_diagnostics": bool(args.anchor_diagnostics),
                    "anchor_strategy": args.anchor_strategy if args.param_method == "cem" else None,
                    "anchor_regularity_percentile": (
                        float(args.anchor_regularity_percentile)
                        if args.param_method == "cem" else None
                    ),
                    "use_idt_remesh": bool(args.use_idt_remesh),
                    "adaptive_radius": bool(args.adaptive_radius),
                    "cem_radius_candidates": list(args.cem_radius_candidates),
                    "reject_retry": bool(args.reject_retry),
                    "cem_max_attempts": int(args.cem_max_attempts),
                    "cem_max_collapsed_faces": int(args.cem_max_collapsed_faces),
                    "spheremap_binary": args.spheremap_binary,
                    "spheremap_repository": args.spheremap_repository,
                    "spheremap_auto_build": bool(args.spheremap_auto_build),
                    "spheremap_iters": int(args.spheremap_iters),
                    "spheremap_step_size": float(args.spheremap_step_size),
                    "spheremap_threads": int(args.spheremap_threads),
                    "spheremap_no_center": bool(args.spheremap_no_center),
                    "spheremap_degree": args.spheremap_degree,
                    "spheremap_a_steps": args.spheremap_a_steps,
                    "spheremap_a_step_size": args.spheremap_a_step_size,
                    "spheremap_poincare_max_norm": args.spheremap_poincare_max_norm,
                    "spheremap_c2i": args.spheremap_c2i,
                    "spheremap_gss_tolerance": args.spheremap_gss_tolerance,
                    "spheremap_lump": bool(args.spheremap_lump),
                    "spheremap_verbose": bool(args.spheremap_verbose),
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
            if mode == "ADNI":
                label["metadata"].update(
                    {
                        "adni_manifest": _resolve_relative(adni_manifest, output_root)
                        if adni_manifest is not None
                        else None,
                        "adni_session_filter": args.adni_session,
                        "adni_hip_filter": args.adni_hip,
                        "participant_id": mesh_input.participant_id,
                        "diagnosis_sc": mesh_input.diagnosis_sc,
                        "session": mesh_input.session,
                        "hip": mesh_input.hip,
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

    if mode == "ADNI" and (args.create_splits or adni_manifest is not None):
        try:
            build_task_splits(
                dataset_root=str(output_root),
                tasks=[TASK_ADNI_CLS],
                num_folds=int(args.num_folds),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                test_ratio=float(args.test_ratio),
                seed=int(args.split_seed),
                group_by_template=False,
            )
            print(f"ADNI clinical splits written under {output_root / 'folds'}")
        except Exception as exc:  # noqa: BLE001
            append_error_log(str(log_path), "split_builder", f"ADNI split generation failed: {exc}")
            print(f"ADNI split generation failed: {exc}")
            failed += 1

    print("=" * 68)
    print(
        f"Done. total={total}, saved={saved}, skipped={skipped}, filtered={filtered}, failed={failed}, "
        f"log={_resolve_relative(log_path, output_root)}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
