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
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128


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
    item_content = str(row.get("item_content", ""))[:MAX_ITEM_CHARS]
    return (
        f"Benchmark: {row.get('benchmark', '')}\n"
        f"Condition: {row.get('condition', 'none') or 'none'}\n"
        f"Item: {item_content}"
    )


ARTIFACT = None
ENCODER = None
SUBJECT_MEANS: dict[str, float] = {}
SUBJECT_EMBEDDINGS: dict[str, np.ndarray] = {}
GLOBAL_SUBJECT_EMBEDDING: np.ndarray | None = None
WITH_SUBJECT_EMBED = False
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
        if "with_subject_embed" in ARTIFACT.files:
            WITH_SUBJECT_EMBED = bool(int(ARTIFACT["with_subject_embed"]))
        if WITH_SUBJECT_EMBED and "subject_embed_names" in ARTIFACT.files:
            SUBJECT_EMBEDDINGS = {
                str(name): np.asarray(value, dtype=np.float32)
                for name, value in zip(
                    ARTIFACT["subject_embed_names"].tolist(),
                    ARTIFACT["subject_embed_values"],
                )
            }
            GLOBAL_SUBJECT_EMBEDDING = np.asarray(
                ARTIFACT["global_subject_embedding"], dtype=np.float32
            )
        from sentence_transformers import SentenceTransformer

        ENCODER = SentenceTransformer(str(ARTIFACT["encoder_id"]))
        ENCODER.max_seq_length = MAX_SEQ_LENGTH
    except Exception as exc:  # noqa: BLE001 - keep submission robust at runtime
        print(f"[embedding] could not initialize: {exc}", flush=True)
        ARTIFACT = None
        ENCODER = None


_ROUND_CACHE_KEY = None
_ROUND_OFFSET = 0.0


def _base_predict(row: Mapping[str, object]) -> float:
    if ARTIFACT is None or ENCODER is None:
        return _clip_probability(GLOBAL_MEAN)

    try:
        embedding = ENCODER.encode(
            [_format_item_text(row)],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embed_arr = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        subject_name = _parse_subject_name(str(row.get("subject_content", "")))
        subject_prior = SUBJECT_MEANS.get(subject_name, GLOBAL_MEAN)
        blocks = [embed_arr, np.array([[subject_prior]], dtype=np.float32)]
        if WITH_SUBJECT_EMBED and GLOBAL_SUBJECT_EMBEDDING is not None:
            subject_embed = SUBJECT_EMBEDDINGS.get(subject_name, GLOBAL_SUBJECT_EMBEDDING)
            blocks.append(np.asarray(subject_embed, dtype=np.float32).reshape(1, -1))
        features = np.concatenate(blocks, axis=1)
        scaled = (features - ARTIFACT["scaler_mean"]) / ARTIFACT["scaler_scale"]
        coef = np.asarray(ARTIFACT["coef"]).reshape(-1)
        intercept = float(np.asarray(ARTIFACT["intercept"]).reshape(-1)[0])
        logit = float(scaled.reshape(-1) @ coef) + intercept
        return _clip_probability(1.0 / (1.0 + math.exp(-logit)))
    except Exception as exc:  # noqa: BLE001 - never let predict() raise
        print(f"[embedding] _base_predict fallback: {exc}", flush=True)
        return _clip_probability(GLOBAL_MEAN)


def _labeled_key(labeled: list[dict] | None):
    if not labeled:
        return ()
    try:
        return tuple(
            sorted(
                (
                    str(row.get("benchmark", "")),
                    str(row.get("condition", "")),
                    str(row.get("subject_content", "")),
                    str(row.get("item_content", "")),
                    float(row.get("label", 0.0) or 0.0),
                )
                for row in labeled
            )
        )
    except Exception:  # noqa: BLE001
        return ("__unsortable__", len(labeled))


def _calibration_offset(labeled: list[dict] | None) -> float:
    if not labeled:
        return 0.0
    residuals = []
    for row in labeled:
        if "label" not in row:
            continue
        try:
            residuals.append(float(row["label"]) - _base_predict(row))
        except Exception:  # noqa: BLE001
            continue
    if not residuals:
        return 0.0
    return float(min(0.10, max(-0.10, sum(residuals) / len(residuals))))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair."""
    global _ROUND_CACHE_KEY, _ROUND_OFFSET
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSET = _calibration_offset(labeled)
        return _clip_probability(_base_predict(input) + _ROUND_OFFSET)
    except Exception as exc:  # noqa: BLE001 - infrastructure must never see a raise
        print(f"[embedding] predict fallback: {exc}", flush=True)
        return _clip_probability(GLOBAL_MEAN)
