# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Smoke-test a Codabench submission folder before zipping or uploading."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path


DUMMY_INPUT = {
    "benchmark": "mmlupro",
    "condition": "zero-shot",
    "subject_content": "Name: DummyModel\nOrganization: Example",
    "item_content": "A triangle has angles 40 and 60 degrees. What is the third angle?",
}


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_submission(submission_dir: str | Path) -> None:
    submission_dir = Path(submission_dir)
    model_path = submission_dir / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.py: {model_path}")

    model = load_module(model_path, "codabench_model")
    pred = model.predict(DUMMY_INPUT, labeled=[])
    if not isinstance(pred, float) or not math.isfinite(pred) or not (0.0 <= pred <= 1.0):
        raise ValueError(f"predict() returned invalid value: {pred!r}")
    print(f"predict() -> {pred:.6f}")

    labeling_path = submission_dir / "labeling.py"
    if labeling_path.exists():
        labeling = load_module(labeling_path, "codabench_labeling")
        score = labeling.acquisition_function(DUMMY_INPUT)
        if not isinstance(score, float) or not math.isfinite(score):
            raise ValueError(f"acquisition_function() returned invalid value: {score!r}")
        print(f"acquisition_function() -> {score:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir")
    args = parser.parse_args()
    smoke_submission(args.submission_dir)


if __name__ == "__main__":
    main()
