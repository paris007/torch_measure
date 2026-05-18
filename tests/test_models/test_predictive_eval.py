# Copyright (c) 2026 AIMS Foundations. MIT License.

from torch_measure.models.predictive_eval import (
    SmoothedPriorPredictiveEvaluator,
    clip_probability,
    format_item_text,
    format_pair_text,
    parse_subject_name,
)


def test_parse_subject_name_uses_name_line():
    assert parse_subject_name("Name: StrongModel\nOrganization: Example") == "strongmodel"


def test_embedding_text_formatters_are_stable():
    row = {
        "benchmark": "mmlupro",
        "condition": "",
        "subject_content": "Name: Model",
        "item_content": "What is 2 + 2?",
    }

    assert format_item_text(row) == "Benchmark: mmlupro\nCondition: none\nItem: What is 2 + 2?"
    assert format_pair_text(row) == (
        "Benchmark: mmlupro\n"
        "Condition: none\n"
        "Subject: Name: Model\n"
        "Item: What is 2 + 2?"
    )


def test_clip_probability_returns_native_float_in_unit_interval():
    assert clip_probability(-10.0) == 1e-4
    assert clip_probability(10.0) == 1.0 - 1e-4
    assert isinstance(clip_probability(0.5), float)


def test_smoothed_prior_predicts_known_subject_signal():
    rows = [
        {
            "benchmark": "math",
            "condition": "zero-shot",
            "subject_content": "Name: strong",
            "item_content": "q1",
            "label": 1,
        },
        {
            "benchmark": "math",
            "condition": "zero-shot",
            "subject_content": "Name: strong",
            "item_content": "q2",
            "label": 1,
        },
        {
            "benchmark": "math",
            "condition": "zero-shot",
            "subject_content": "Name: weak",
            "item_content": "q1",
            "label": 0,
        },
    ]
    evaluator = SmoothedPriorPredictiveEvaluator.fit(rows, strength=1.0)

    strong = evaluator.predict(
        {
            "benchmark": "math",
            "condition": "zero-shot",
            "subject_content": "Name: strong",
            "item_content": "new",
        }
    )
    weak = evaluator.predict(
        {
            "benchmark": "math",
            "condition": "zero-shot",
            "subject_content": "Name: weak",
            "item_content": "new",
        }
    )

    assert strong > weak
    assert 0.0 <= weak <= 1.0
    assert 0.0 <= strong <= 1.0


def test_labeled_calibration_handles_empty_and_moves_prediction():
    evaluator = SmoothedPriorPredictiveEvaluator.fit(
        [
            {
                "benchmark": "math",
                "condition": "none",
                "subject_content": "Name: model",
                "item_content": "q",
                "label": 0,
            }
        ],
        strength=1.0,
    )
    input_row = {
        "benchmark": "math",
        "condition": "none",
        "subject_content": "Name: model",
        "item_content": "new",
    }

    base = evaluator.predict(input_row, labeled=[])
    calibrated = evaluator.predict(input_row, labeled=[{**input_row, "label": 1}])

    assert calibrated > base


def test_smoothed_prior_round_trip_serialization():
    rows = [
        {
            "benchmark": "math",
            "condition": "none",
            "subject_content": "Name: model",
            "item_content": "q",
            "label": 1,
        }
    ]
    evaluator = SmoothedPriorPredictiveEvaluator.fit(rows)
    loaded = SmoothedPriorPredictiveEvaluator.from_dict(evaluator.to_dict())
    input_row = {
        "benchmark": "math",
        "condition": "none",
        "subject_content": "Name: model",
        "item_content": "new",
    }

    assert loaded.predict(input_row) == evaluator.predict(input_row)
