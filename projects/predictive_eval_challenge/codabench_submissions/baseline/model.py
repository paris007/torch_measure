"""Codabench baseline submission for the Predictive Evaluation Challenge.

This file is intentionally self-contained: Codabench imports only the files in
the submission ZIP, so runtime code should not depend on the local
`torch_measure` package being installed.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping


EPS = 1e-4
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {"global_mean": 0.5}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()
_ROUND_CACHE_KEY = None
_ROUND_OFFSET = 0.0


def _base_predict(row: Mapping[str, object]) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    weighted_values = [
        (ARTIFACT.get("subject", {}).get(subject_name), 0.35),
        (ARTIFACT.get("benchmark", {}).get(benchmark), 0.15),
        (ARTIFACT.get("benchmark_condition", {}).get(_key(benchmark, condition)), 0.25),
        (ARTIFACT.get("subject_benchmark", {}).get(_key(subject_name, benchmark)), 0.25),
    ]
    available = [(value, weight) for value, weight in weighted_values if value is not None]
    if not available:
        return _clip_probability(global_mean)

    total_weight = sum(weight for _, weight in available)
    pred = sum(float(value) * weight for value, weight in available) / total_weight
    return _clip_probability(0.85 * pred + 0.15 * global_mean)


def _labeled_key(labeled: list[dict] | None):
    if not labeled:
        return ()
    return tuple(
        sorted(
            (
                row.get("benchmark", ""),
                row.get("condition", ""),
                row.get("subject_content", ""),
                row.get("item_content", ""),
                row.get("label", None),
            )
            for row in labeled
        )
    )


def _calibration_offset(labeled: list[dict] | None) -> float:
    if not labeled:
        return 0.0
    residuals = [
        float(row["label"]) - _base_predict(row)
        for row in labeled
        if "label" in row
    ]
    if not residuals:
        return 0.0
    return float(min(0.12, max(-0.12, sum(residuals) / len(residuals))))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair."""
    global _ROUND_CACHE_KEY, _ROUND_OFFSET
    key = _labeled_key(labeled)
    if key != _ROUND_CACHE_KEY:
        _ROUND_CACHE_KEY = key
        _ROUND_OFFSET = _calibration_offset(labeled)
    return _clip_probability(_base_predict(input) + _ROUND_OFFSET)
