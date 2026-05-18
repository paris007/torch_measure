"""Codabench embedding submission for the Predictive Evaluation Challenge.

This file is self-contained because Codabench imports only the files inside
the submission ZIP. The sentence-transformer encoder is declared in
`models.txt` and pre-fetched by the platform.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np


EPS = 1e-4
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "embedding_head.npz"


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _format_item_text(row: Mapping[str, object]) -> str:
    return (
        f"Benchmark: {row.get('benchmark', '')}\n"
        f"Condition: {row.get('condition', 'none') or 'none'}\n"
        f"Item: {row.get('item_content', '')}"
    )


ARTIFACT = None
ENCODER = None
SUBJECT_MEANS: dict[str, float] = {}
GLOBAL_MEAN = 0.5

if ARTIFACT_PATH.exists():
    try:
        ARTIFACT = np.load(ARTIFACT_PATH, allow_pickle=False)
        GLOBAL_MEAN = float(ARTIFACT["global_mean"])
        SUBJECT_MEANS = {
            str(name): float(value)
            for name, value in zip(
                ARTIFACT["subject_names"].tolist(),
                ARTIFACT["subject_values"].tolist(),
            )
        }
        from sentence_transformers import SentenceTransformer

        ENCODER = SentenceTransformer(str(ARTIFACT["encoder_id"]))
    except Exception as exc:  # noqa: BLE001 - keep submission robust at runtime
        print(f"[embedding] could not initialize: {exc}", flush=True)
        ARTIFACT = None
        ENCODER = None


_ROUND_CACHE_KEY = None
_ROUND_OFFSET = 0.0


def _base_predict(row: Mapping[str, object]) -> float:
    if ARTIFACT is None or ENCODER is None:
        return _clip_probability(GLOBAL_MEAN)

    embedding = ENCODER.encode(
        [_format_item_text(row)],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    subject_prior = SUBJECT_MEANS.get(
        _parse_subject_name(str(row.get("subject_content", ""))),
        GLOBAL_MEAN,
    )
    features = np.concatenate(
        [np.asarray(embedding, dtype=np.float32), np.array([[subject_prior]], dtype=np.float32)],
        axis=1,
    )
    scaled = (features - ARTIFACT["scaler_mean"]) / ARTIFACT["scaler_scale"]
    logit = float(scaled @ ARTIFACT["coef"].T + ARTIFACT["intercept"])
    return _clip_probability(1.0 / (1.0 + math.exp(-logit)))


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
    return float(min(0.10, max(-0.10, sum(residuals) / len(residuals))))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair."""
    global _ROUND_CACHE_KEY, _ROUND_OFFSET
    key = _labeled_key(labeled)
    if key != _ROUND_CACHE_KEY:
        _ROUND_CACHE_KEY = key
        _ROUND_OFFSET = _calibration_offset(labeled)
    return _clip_probability(_base_predict(input) + _ROUND_OFFSET)
