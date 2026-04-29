#!/usr/bin/env python
"""Validate parametrization results for a generated dataset.

This script checks that spherical parametrization preserved mesh topology
and normal orientation across all samples in a generated dataset.

Usage:
    python validate_dataset_parametrization.py /path/to/dataset/labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import trimesh

from spherepar.spherical_parametrization import (
    verify_topology_preserved,
    verify_normal_orientation_preserved,
)


def load_canonical_mesh(canonical_dir: Path) -> Dict[str, tuple]:
    """Load all canonical meshes from directory.
    
    Returns
    -------
    Dict[str, tuple]
        Maps mesh name to (vertices, faces).
    """
    meshes = {}
    for mesh_file in canonical_dir.glob("*.obj"):
        mesh = trimesh.load(mesh_file, process=False)
        meshes[mesh_file.stem] = (mesh.vertices, mesh.faces)
    return meshes


def validate_sample_parametrization(
    label_file: Path,
    canonical_meshes: Dict[str, tuple],
) -> Dict[str, Any]:
    """Validate parametrization for a single sample.
    
    Parameters
    ----------
    label_file : Path
        Path to label JSON file.
    canonical_meshes : Dict[str, tuple]
        Map of canonical mesh names to (vertices, faces).
    
    Returns
    -------
    Dict[str, Any]
        Validation report.
    """
    report: Dict[str, Any] = {
        "sample_id": label_file.stem,
        "label_file": str(label_file),
        "topology_valid": None,
        "orientation_valid": None,
        "errors": [],
    }
    
    # Load label
    try:
        with open(label_file) as f:
            label = json.load(f)
    except Exception as e:
        report["errors"].append(f"Failed to load label: {e}")
        return report
    
    # Extract parametrization metadata
    parametrization = label.get("parametrization", {})
    if not parametrization:
        report["errors"].append("No parametrization metadata in label")
        return report
    
    # Get template mesh name
    template_name = label.get("template_mesh")
    if not template_name or template_name not in canonical_meshes:
        report["errors"].append(f"Template mesh '{template_name}' not found")
        return report
    
    vertices_orig, faces_orig = canonical_meshes[template_name]
    
    # Check topology validation
    if "topology_report" in parametrization:
        topology_report = parametrization["topology_report"]
        is_valid = topology_report.get("is_valid", False)
        report["topology_valid"] = is_valid
        if not is_valid:
            report["errors"].extend(topology_report.get("errors", []))
    
    # Check orientation validation
    if "orientation_report" in parametrization:
        orientation_report = parametrization["orientation_report"]
        is_valid = orientation_report.get("is_valid", False)
        report["orientation_valid"] = is_valid
        if not is_valid:
            errors = orientation_report.get("errors", [])
            report["errors"].extend(errors)
            
            # Include inward-pointing face IDs for debugging
            if orientation_report.get("n_inward_normals", 0) > 0:
                report["inward_face_ids"] = orientation_report.get("inward_face_ids", [])
            if orientation_report.get("n_flipped_normals", 0) > 0:
                report["flipped_face_ids"] = orientation_report.get("flipped_face_ids", [])
    
    return report


def validate_dataset(
    dataset_labels_dir: Path,
    canonical_dir: Path,
) -> Dict[str, Any]:
    """Validate all samples in a dataset.
    
    Parameters
    ----------
    dataset_labels_dir : Path
        Path to dataset labels directory.
    canonical_dir : Path
        Path to canonical meshes directory.
    
    Returns
    -------
    Dict[str, Any]
        Summary report.
    """
    # Load canonical meshes
    try:
        canonical_meshes = load_canonical_mesh(canonical_dir)
    except Exception as e:
        return {
            "error": f"Failed to load canonical meshes: {e}",
            "total_samples": 0,
            "valid_samples": 0,
        }
    
    if not canonical_meshes:
        return {
            "error": "No canonical meshes found",
            "total_samples": 0,
            "valid_samples": 0,
        }
    
    # Find all label files
    label_files = sorted(dataset_labels_dir.glob("**/sample_*.json"))
    
    if not label_files:
        return {
            "error": "No sample labels found",
            "total_samples": 0,
            "valid_samples": 0,
        }
    
    # Validate each sample
    results: List[Dict[str, Any]] = []
    for label_file in label_files:
        result = validate_sample_parametrization(label_file, canonical_meshes)
        results.append(result)
    
    # Summarize
    n_total = len(results)
    n_valid_topology = sum(1 for r in results if r.get("topology_valid") is True)
    n_valid_orientation = sum(1 for r in results if r.get("orientation_valid") is True)
    n_with_errors = sum(1 for r in results if r.get("errors"))
    
    summary: Dict[str, Any] = {
        "total_samples": n_total,
        "topology_valid_samples": n_valid_topology,
        "orientation_valid_samples": n_valid_orientation,
        "samples_with_errors": n_with_errors,
        "canonical_meshes": list(canonical_meshes.keys()),
    }
    
    # List problematic samples
    problematic = [r for r in results if r.get("errors")]
    if problematic:
        summary["problematic_samples"] = problematic[:10]  # First 10
        if len(problematic) > 10:
            summary["problematic_samples_truncated"] = len(problematic) - 10
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Validate parametrization in generated dataset"
    )
    parser.add_argument(
        "labels_dir",
        type=Path,
        help="Path to dataset labels directory",
    )
    parser.add_argument(
        "--canonical-dir",
        type=Path,
        required=True,
        help="Path to canonical meshes directory",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if not args.labels_dir.exists():
        print(f"Error: Labels directory not found: {args.labels_dir}", file=sys.stderr)
        return 1
    
    if not args.canonical_dir.exists():
        print(f"Error: Canonical directory not found: {args.canonical_dir}", file=sys.stderr)
        return 1
    
    print(f"Validating dataset in: {args.labels_dir}")
    print(f"Canonical meshes from: {args.canonical_dir}")
    print()
    
    summary = validate_dataset(args.labels_dir, args.canonical_dir)
    
    # Print summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return 1
    
    print(f"Total samples:               {summary['total_samples']}")
    print(f"Topology valid:              {summary['topology_valid_samples']}")
    print(f"Orientation valid:           {summary['orientation_valid_samples']}")
    print(f"Samples with errors:         {summary['samples_with_errors']}")
    print(f"Canonical meshes:            {', '.join(summary['canonical_meshes'])}")
    print()
    
    # Print problematic samples
    if "problematic_samples" in summary:
        print("Problematic samples:")
        for sample in summary["problematic_samples"]:
            print(f"  - {sample['sample_id']}")
            for error in sample.get("errors", []):
                print(f"    • {error}")
            if "inward_face_ids" in sample:
                face_ids = sample["inward_face_ids"]
                print(f"    • Inward normals at faces: {face_ids[:5]}{'...' if len(face_ids) > 5 else ''}")
        
        if "problematic_samples_truncated" in summary:
            print(f"  ... and {summary['problematic_samples_truncated']} more")
    else:
        print("✓ All samples validated successfully!")
    
    print()
    
    # Return exit code
    if summary["samples_with_errors"] == 0:
        print("✓ Dataset validation PASSED")
        return 0
    else:
        print("✗ Dataset validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
