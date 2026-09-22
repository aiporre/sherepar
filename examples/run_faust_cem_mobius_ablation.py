#!/usr/bin/env python3
"""Run and score the five-subject CEM Möbius-centering ablation.

The parametrizations are produced by the same ``save_spherical_parametrization``
path used by the dataset generator.  Spherical OBJ outputs are converted to
temporary PLY files only so that the existing MoebiusRegistration scorer can
be reused verbatim.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spherepar.benchmark.dataset_generator import save_spherical_parametrization


SUBJECTS = ("007", "028", "049", "066", "085")
SCORER = Path("/home/sauron/Documents/Phd/code/MoebiusRegistration/scripts/cmcf_metrics.py")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _score_ply(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    proc = subprocess.run(
        [sys.executable, str(SCORER), str(path), "--json"],
        cwd=str(SCORER.parent.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode not in (0, 1):
        return None, proc.stderr.strip() or proc.stdout.strip() or f"scorer exit {proc.returncode}"
    try:
        return json.loads(proc.stdout), proc.stderr.strip() or None
    except json.JSONDecodeError as exc:
        return None, f"could not parse scorer output: {exc}; stdout={proc.stdout!r}"


def _run_one(
    *,
    source: Path,
    output_root: Path,
    subject: str,
    mobius_center: bool,
    cem_max_iters: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "subject": subject,
        "mobius_center": bool(mobius_center),
        "source": str(source),
        "output_root": str(output_root),
        "status": "error",
        "warnings": [],
        "error": None,
    }
    try:
        mesh = trimesh.load(str(source), force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"expected trimesh.Trimesh, got {type(mesh).__name__}")
        log_path = output_root / "logs" / "errors.log"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            saved = save_spherical_parametrization(
                root=str(output_root),
                name=f"tr_reg_{subject}",
                vertices=np.asarray(mesh.vertices, dtype=np.float64),
                faces=np.asarray(mesh.faces, dtype=np.int32),
                method="cem",
                cem_radius=1.2,
                cem_max_iters=cem_max_iters,
                cem_verbose=False,
                mobius_center=mobius_center,
                anchor_strategy="central_regular",
                use_idt_remesh=False,
                adaptive_radius=False,
                reject_retry=False,
                log_path=str(log_path),
            )
        result["warnings"] = [str(item.message) for item in caught]
        result["saved"] = _json_safe(saved)

        sphere_obj = output_root / saved["sphere"]
        sidecar = output_root / saved["spherical_label"]
        metadata = json.loads(sidecar.read_text())["metadata"]
        result["parametrization_success"] = bool(saved.get("parametrization_success", True))
        result["metadata"] = metadata

        metrics_dir = output_root / "metrics_ply"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        sphere_mesh = trimesh.load(str(sphere_obj), force="mesh", process=False)
        metric_ply = metrics_dir / f"tr_reg_{subject}.ply"
        trimesh.Trimesh(
            vertices=np.asarray(sphere_mesh.vertices, dtype=np.float64),
            faces=np.asarray(sphere_mesh.faces, dtype=np.int64),
            process=False,
        ).export(str(metric_ply))
        score, score_error = _score_ply(metric_ply)
        result["metric_ply"] = str(metric_ply)
        result["score"] = score
        result["scorer_stderr"] = score_error
        if score is None:
            result["error"] = score_error or "scoring failed"
            return result
        result["status"] = "ok"
        return result
    except Exception as exc:  # record each requested run; never silently skip one
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _fmt_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.6f}%"


def _write_report(path: Path, results: list[dict[str, Any]]) -> None:
    valid = [item for item in results if item.get("score") is not None]
    lines = [
        "# FAUST CEM Möbius-centering ablation",
        "",
        "CEM was run at radius `1.2` with `anchor_strategy=central_regular` on "
        "subjects `007, 028, 049, 066, 085`, with and without area-weighted "
        "Möbius centering. Spherical outputs were scored through the existing "
        "`MoebiusRegistration/scripts/cmcf_metrics.py` implementation.",
        "",
        "## Per-subject results",
        "",
        "| Subject | Configuration | Collapsed faces | Total faces | Fraction | Status | Centering | Error |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for item in results:
        score = item.get("score") or {}
        metadata = item.get("metadata") or {}
        centering = metadata.get("mobius_centering") or {}
        centering_text = (
            f"{centering.get('iterations')} iterations"
            if item.get("mobius_center") and centering
            else ("not requested" if not item.get("mobius_center") else "failed/not recorded")
        )
        error = item.get("error") or ""
        escaped_error = error.replace("|", "\\|")
        lines.append(
            f"| {item['subject']} | {'with' if item['mobius_center'] else 'without'} Möbius center | "
            f"{score.get('collapsed_count', 'n/a')} | {score.get('face_count', 'n/a')} | "
            f"{_fmt_percent(score.get('collapsed_frac')) if score.get('collapsed_frac') is not None else 'n/a'} | "
            f"{item.get('status')} | {centering_text} | {escaped_error} |"
        )

    lines.extend(["", "## Aggregate summary", ""])
    for centered in (False, True):
        label = "with Möbius centering" if centered else "without Möbius centering"
        fractions = [
            float(item["score"]["collapsed_frac"])
            for item in valid
            if bool(item["mobius_center"]) == centered
        ]
        if fractions:
            lines.append(
                f"- **{label}:** median {_fmt_percent(statistics.median(fractions))}; "
                f"maximum {_fmt_percent(max(fractions))}; n={len(fractions)}."
            )
        else:
            lines.append(f"- **{label}:** no successful scores.")

    lines.extend([
        "",
        "## Comparison and verdict",
        "",
        "The earlier `central_regular` baseline was approximately 36.8–37.0% "
        "collapsed faces on subjects 000/001/002. The best CMCF result was "
        "approximately 32.96% median on a disjoint 20-subject cohort. Because "
        "the subject sets differ, these comparisons are directional rather than "
        "strict apples-to-apples rankings.",
        "",
    ])
    base = [float(i["score"]["collapsed_frac"]) for i in valid if not i["mobius_center"]]
    centered = [float(i["score"]["collapsed_frac"]) for i in valid if i["mobius_center"]]
    if base and centered:
        delta = statistics.median(centered) - statistics.median(base)
        verdict = (
            "Möbius centering measurably reduced the median collapsed fraction."
            if delta < -1e-12
            else "Möbius centering did not reduce the median collapsed fraction; the result is flat or worse."
        )
        lines.append(
            f"**Verdict:** {verdict} Median delta (centered − uncentered) = {_fmt_percent(delta)}."
        )
    else:
        lines.append("**Verdict:** Inconclusive because one configuration has no complete scores.")

    failures = [item for item in results if item.get("status") != "ok" or item.get("warnings")]
    lines.extend(["", "## Errors and warnings", ""])
    centered_results = [item for item in results if item.get("mobius_center")]
    centered_ok = [
        item for item in centered_results
        if item.get("status") == "ok" and (item.get("metadata") or {}).get("mobius_centering")
    ]
    if len(centered_ok) == len(centered_results) == 5:
        iterations = sorted(
            (item["metadata"]["mobius_centering"].get("iterations") for item in centered_ok)
        )
        lines.append(
            "All five centered runs converged successfully with no centering "
            f"exception or fallback (iterations: {iterations})."
        )
        lines.append("")
    if not failures:
        lines.append("No run errors, scorer failures, or runtime warnings were recorded.")
    else:
        for item in failures:
            lines.append(
                f"- `{item['subject']}` ({'with' if item['mobius_center'] else 'without'} center): "
                f"status={item.get('status')}, error={item.get('error') or 'none'}, "
                f"warnings={item.get('warnings') or []}."
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faust-root",
        type=Path,
        default=Path("/media/sauron/GG2/datasets/deformed_blobs/faust_6890"),
    )
    parser.add_argument("--output-base", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--cem-max-iters", type=int, default=100)
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "data/faust_cem_mobius_ablation_report.md",
    )
    args = parser.parse_args()

    source_dir = args.faust_root / "registrations"
    configs = (
        (False, args.output_base / "faust_cem_central_regular_r12"),
        (True, args.output_base / "faust_cem_central_regular_mobius_r12"),
    )
    for _, root in configs:
        root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for mobius_center, root in configs:
        for subject in SUBJECTS:
            source = source_dir / f"tr_reg_{subject}.ply"
            print(
                f"[CEM ablation] subject={subject} "
                f"mobius_center={mobius_center} output={root}",
                flush=True,
            )
            result = _run_one(
                source=source,
                output_root=root,
                subject=subject,
                mobius_center=mobius_center,
                cem_max_iters=args.cem_max_iters,
            )
            results.append(result)
            score = result.get("score") or {}
            print(
                f"  status={result.get('status')} collapsed={score.get('collapsed_count', 'n/a')} "
                f"fraction={score.get('collapsed_frac', 'n/a')} error={result.get('error')}",
                flush=True,
            )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("")
    _write_report(args.report, results)
    summary_path = args.report.with_suffix(".json")
    summary_path.write_text(json.dumps(_json_safe(results), indent=2) + "\n")
    print(f"Report: {args.report}")
    print(f"Results: {summary_path}")
    return 0 if len(results) == 10 and all(item.get("score") is not None for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
