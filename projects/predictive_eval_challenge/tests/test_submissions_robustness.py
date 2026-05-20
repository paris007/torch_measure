# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Robustness tests for the Codabench submission wrappers.

These tests load each submission folder's `model.py` and `labeling.py` as
isolated modules and verify the contract enforced by the Codabench infra:

  * `predict()` must return a finite float in [0, 1] for any reasonable input
  * `predict()` must never raise on adversarial / heterogeneous inputs
  * `acquisition_function()` must return a finite float

These mirror the NumPy 2.x `float(2d_array)` regression we hit when the first
embedding submission ran on Codabench.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "codabench_submissions"

SUBMISSION_NAMES = [
    "baseline",
    "embedding",
    "factor_pge",
    "factor_baseline_ensemble",
    "factor_pge_multiseed",
]

GOOD_INPUT = {
    "benchmark": "mmlupro",
    "condition": "zero-shot",
    "subject_content": "Name: DummyModel\nOrganization: Example",
    "item_content": "A triangle has angles 40 and 60 degrees. What is the third angle?",
}

ADVERSARIAL_INPUTS = [
    {  # missing optional fields
        "benchmark": "mmlupro",
        "condition": "none",
        "subject_content": "",
        "item_content": "",
    },
    {  # missing some keys entirely
        "benchmark": "ai2d_test",
        "item_content": "x",
    },
    {  # huge item content
        "benchmark": "mmlupro",
        "condition": "zero-shot",
        "subject_content": "Name: BigModel",
        "item_content": "A" * 50_000,
    },
    {  # non-string types
        "benchmark": 7,
        "condition": None,
        "subject_content": 42,
        "item_content": ["not", "a", "string"],
    },
]

ADVERSARIAL_LABELED = [
    None,
    [],
    [  # heterogeneous types that would break a naive sort
        {
            "benchmark": 7,
            "condition": None,
            "subject_content": "Name: A",
            "item_content": "foo",
            "label": "1",
        },
        {
            "benchmark": "mmlupro",
            "condition": "zero-shot",
            "subject_content": "Name: B",
            "item_content": "bar",
            "label": None,
        },
    ],
]


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_module(submission_name: str) -> ModuleType:
    path = SUBMISSIONS_DIR / submission_name / "model.py"
    return _load_module(path, f"_codabench_{submission_name}_model")


def _labeling_module(submission_name: str) -> ModuleType | None:
    path = SUBMISSIONS_DIR / submission_name / "labeling.py"
    if not path.exists():
        return None
    return _load_module(path, f"_codabench_{submission_name}_labeling")


@pytest.mark.parametrize("submission_name", SUBMISSION_NAMES)
def test_predict_returns_probability_for_good_input(submission_name: str) -> None:
    model = _model_module(submission_name)
    out = model.predict(GOOD_INPUT, labeled=[])
    assert isinstance(out, float)
    assert math.isfinite(out)
    assert 0.0 <= out <= 1.0


@pytest.mark.parametrize("submission_name", SUBMISSION_NAMES)
@pytest.mark.parametrize("bad_input", ADVERSARIAL_INPUTS)
def test_predict_does_not_raise_on_adversarial_input(
    submission_name: str, bad_input: dict
) -> None:
    model = _model_module(submission_name)
    out = model.predict(bad_input, labeled=[])
    assert isinstance(out, float)
    assert math.isfinite(out)
    assert 0.0 <= out <= 1.0


@pytest.mark.parametrize("submission_name", SUBMISSION_NAMES)
@pytest.mark.parametrize("labeled", ADVERSARIAL_LABELED)
def test_predict_handles_adversarial_labeled_list(
    submission_name: str, labeled
) -> None:
    model = _model_module(submission_name)
    out = model.predict(GOOD_INPUT, labeled=labeled)
    assert isinstance(out, float)
    assert math.isfinite(out)
    assert 0.0 <= out <= 1.0


@pytest.mark.parametrize("submission_name", SUBMISSION_NAMES)
def test_acquisition_function_returns_finite_float(submission_name: str) -> None:
    labeling = _labeling_module(submission_name)
    if labeling is None:
        pytest.skip(f"{submission_name} has no labeling.py")
    score = labeling.acquisition_function(GOOD_INPUT)
    assert isinstance(score, float)
    assert math.isfinite(score)


@pytest.mark.parametrize("submission_name", SUBMISSION_NAMES)
def test_clip_probability_handles_non_finite(submission_name: str) -> None:
    model = _model_module(submission_name)
    assert 0.0 <= model._clip_probability(float("nan")) <= 1.0
    assert 0.0 <= model._clip_probability(float("inf")) <= 1.0
    assert 0.0 <= model._clip_probability(-1.0) <= 1.0
    assert 0.0 <= model._clip_probability(2.0) <= 1.0


@pytest.mark.parametrize("submission_name", SUBMISSION_NAMES)
def test_format_item_text_truncates(submission_name: str) -> None:
    model = _model_module(submission_name)
    if not hasattr(model, "_format_item_text"):
        pytest.skip(f"{submission_name} does not expose _format_item_text")
    huge = "X" * 50_000
    text = model._format_item_text({"benchmark": "b", "condition": "c", "item_content": huge})
    assert text.count("X") < 2_000  # well under MAX_ITEM_CHARS even with prefix overhead


@pytest.mark.parametrize("submission_name", SUBMISSION_NAMES)
def test_parse_subject_name_extracts_name_line(submission_name: str) -> None:
    model = _model_module(submission_name)
    name = model._parse_subject_name("Name: GPT-4\nOrganization: OpenAI\n")
    assert name == "gpt-4"
    assert model._parse_subject_name("") == ""
    assert model._parse_subject_name(None) == ""
