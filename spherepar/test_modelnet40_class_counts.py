from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_counter_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "count_modelnet40_classes.py"
    spec = importlib.util.spec_from_file_location("modelnet40_class_counter", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_label(labels_dir: Path, name: str, class_id: int, class_name: str, source_split: str) -> None:
    with open(labels_dir / f"{name}.json", "w") as fh:
        json.dump(
            {
                "metadata": {"class_id": class_id, "class_name": class_name, "source_split": source_split},
                "tasks": {"modelnet40_cls": {"valid": True, "label": class_id}},
            },
            fh,
        )


def test_counts_generated_modelnet40_labels_and_prints_bars(tmp_path: Path, capsys):
    module = _load_counter_module()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    _write_label(labels_dir, "chair_train", 0, "chair", "train")
    _write_label(labels_dir, "chair_test", 0, "chair", "test")
    _write_label(labels_dir, "table_train", 1, "table", "train")
    (labels_dir / "chair_spherical.json").write_text("{}")

    counts, total, malformed = module.count_modelnet40_classes(labels_dir)
    assert total == 3
    assert malformed == 0
    assert counts[(0, "chair")] == {"train": 1, "test": 1, "total": 2}
    assert counts[(1, "table")] == {"train": 1, "test": 0, "total": 1}

    module.print_distribution(counts, total)
    output = capsys.readouterr().out
    assert "chair" in output and " 66.7%" in output
    assert "table" in output and " 33.3%" in output
