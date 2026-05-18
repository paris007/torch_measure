# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Predictive evaluation helpers for cold-start benchmark items."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Mapping


InputRow = Mapping[str, object]


def clip_probability(value: float, eps: float = 1e-4) -> float:
    """Return a native Python probability clipped away from exact 0 and 1."""
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def parse_subject_name(subject_content: str) -> str:
    """Extract a stable subject name from the challenge subject text field."""
    match = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def format_item_text(input: InputRow) -> str:
    """Format benchmark, condition, and item content for item-side embeddings."""
    return (
        f"Benchmark: {input.get('benchmark', '')}\n"
        f"Condition: {input.get('condition', 'none') or 'none'}\n"
        f"Item: {input.get('item_content', '')}"
    )


def format_pair_text(input: InputRow) -> str:
    """Format all runtime input fields for pair-level embedding or acquisition."""
    return (
        f"Benchmark: {input.get('benchmark', '')}\n"
        f"Condition: {input.get('condition', 'none') or 'none'}\n"
        f"Subject: {input.get('subject_content', '')}\n"
        f"Item: {input.get('item_content', '')}"
    )


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


@dataclass
class SmoothedPriorPredictiveEvaluator:
    """Smoothed subject/benchmark priors for predictive evaluation.

    This is intentionally small and dependency-light so it can be reused both
    in offline validation and in Codabench submission wrappers.
    """

    global_mean: float = 0.5
    strength: float = 25.0
    subject: dict[str, float] = field(default_factory=dict)
    benchmark: dict[str, float] = field(default_factory=dict)
    benchmark_condition: dict[str, float] = field(default_factory=dict)
    subject_benchmark: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: list[InputRow], strength: float = 25.0) -> "SmoothedPriorPredictiveEvaluator":
        labels = [float(row["label"]) for row in rows if "label" in row]
        global_mean = sum(labels) / len(labels) if labels else 0.5

        evaluator = cls(global_mean=global_mean, strength=strength)
        evaluator.subject = evaluator._group_mean(rows, ("subject_name",))
        evaluator.benchmark = evaluator._group_mean(rows, ("benchmark",))
        evaluator.benchmark_condition = evaluator._group_mean(rows, ("benchmark", "condition"))
        evaluator.subject_benchmark = evaluator._group_mean(rows, ("subject_name", "benchmark"))
        return evaluator

    def predict(self, input: InputRow, labeled: list[InputRow] | None = None) -> float:
        """Predict the probability that the subject answers the item correctly."""
        base = self._base_predict(input)
        offset = self.calibration_offset(labeled)
        return clip_probability(base + offset)

    def to_dict(self) -> dict[str, object]:
        """Serialize fitted state to JSON-compatible primitives."""
        return asdict(self)

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "SmoothedPriorPredictiveEvaluator":
        """Load fitted state from JSON-compatible primitives."""
        return cls(
            global_mean=float(state.get("global_mean", 0.5)),
            strength=float(state.get("strength", 25.0)),
            subject=dict(state.get("subject", {})),
            benchmark=dict(state.get("benchmark", {})),
            benchmark_condition=dict(state.get("benchmark_condition", {})),
            subject_benchmark=dict(state.get("subject_benchmark", {})),
        )

    def calibration_offset(self, labeled: list[InputRow] | None) -> float:
        """Estimate a conservative per-round offset from revealed labels."""
        if not labeled:
            return 0.0
        residuals = [
            float(row["label"]) - self._base_predict(row)
            for row in labeled
            if "label" in row
        ]
        if not residuals:
            return 0.0
        return float(min(0.12, max(-0.12, sum(residuals) / len(residuals))))

    def _base_predict(self, input: InputRow) -> float:
        subject_name = parse_subject_name(str(input.get("subject_content", "")))
        benchmark = str(input.get("benchmark", ""))
        condition = str(input.get("condition", "none") or "none")

        weighted_values = [
            (self.subject.get(subject_name), 0.35),
            (self.benchmark.get(benchmark), 0.15),
            (self.benchmark_condition.get(_key(benchmark, condition)), 0.25),
            (self.subject_benchmark.get(_key(subject_name, benchmark)), 0.25),
        ]
        available = [(value, weight) for value, weight in weighted_values if value is not None]
        if not available:
            return clip_probability(self.global_mean)
        total_weight = sum(weight for _, weight in available)
        pred = sum(float(value) * weight for value, weight in available) / total_weight
        return clip_probability(0.85 * pred + 0.15 * self.global_mean)

    def _group_mean(self, rows: list[InputRow], fields: tuple[str, ...]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            if "label" not in row:
                continue
            group_key = self._row_group_key(row, fields)
            sums[group_key] = sums.get(group_key, 0.0) + float(row["label"])
            counts[group_key] = counts.get(group_key, 0) + 1

        return {
            key: (sums[key] + self.global_mean * self.strength) / (counts[key] + self.strength)
            for key in sums
        }

    @staticmethod
    def _row_group_key(row: InputRow, fields: tuple[str, ...]) -> str:
        values = []
        for field_name in fields:
            if field_name == "subject_name":
                values.append(parse_subject_name(str(row.get("subject_content", ""))))
            elif field_name == "condition":
                values.append(str(row.get("condition", "none") or "none"))
            else:
                values.append(str(row.get(field_name, "")))
        return _key(*values)
