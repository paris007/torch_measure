"""Codabench factor / PGE submission for the Predictive Evaluation Challenge.

Self-contained: at inference we encode the item text with sentence-transformers,
push it through a small MLP to recover (a, b) item parameters, look up the
subject's ability theta, and compute sigmoid(a*theta - b).
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Mapping

import numpy as np


EPS = 1e-4
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "factor_pge.npz"
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


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


ARTIFACT = None
ENCODER = None
SUBJECT_THETA: dict[str, float] = {}
GLOBAL_THETA = 0.0
GLOBAL_MEAN = 0.5
TEMPERATURE = 1.0
MLP_LAYERS: list[tuple[np.ndarray, np.ndarray]] = []

if ARTIFACT_PATH.exists():
    try:
        ARTIFACT = np.load(ARTIFACT_PATH, allow_pickle=False)
        GLOBAL_THETA = float(ARTIFACT["global_theta"])
        GLOBAL_MEAN = float(ARTIFACT["global_mean"])
        if "temperature" in ARTIFACT.files:
            t_val = float(ARTIFACT["temperature"])
            # The stored T is fit by `recalibrate_temperature.py` on a
            # whole-benchmark cold-start holdout, so we trust it directly.
            # Empirically v3 with T clamped to 0.7 scored worse than v2 with
            # the unclamped T=0.275, confirming the cold-start optimum is
            # well below 1.0. See report §5 for the ablation.
            TEMPERATURE = t_val if t_val > 0 and math.isfinite(t_val) else 1.0
        SUBJECT_THETA = {
            str(name): float(value)
            for name, value in zip(
                ARTIFACT["subject_names"].tolist(),
                ARTIFACT["subject_theta"].tolist(),
            )
        }
        layer_indices = sorted(
            int(k[len("mlp_w") :])
            for k in ARTIFACT.files
            if k.startswith("mlp_w")
        )
        MLP_LAYERS = [
            (
                np.asarray(ARTIFACT[f"mlp_w{i}"], dtype=np.float32),
                np.asarray(ARTIFACT[f"mlp_b{i}"], dtype=np.float32),
            )
            for i in layer_indices
        ]
        from sentence_transformers import SentenceTransformer

        ENCODER = SentenceTransformer(str(ARTIFACT["encoder_id"]))
        ENCODER.max_seq_length = MAX_SEQ_LENGTH
    except Exception as exc:  # noqa: BLE001 - never crash submission on init
        print(f"[factor_pge] could not initialize: {exc}", flush=True)
        ARTIFACT = None
        ENCODER = None


_ROUND_CACHE_KEY = None
_ROUND_GLOBAL_OFFSET = 0.0
_ROUND_CATEGORY_OFFSETS: dict[tuple[str, str], float] = {}
_CALIB_CLIP = 0.08
_CALIB_SHRINKAGE = 5.0


def _item_params_from_embedding(embed: np.ndarray) -> tuple[float, float]:
    """Forward pass of the small MLP returning (a, b)."""
    h = embed.reshape(1, -1)
    for idx, (w, bias) in enumerate(MLP_LAYERS):
        h = h @ w.T + bias
        if idx < len(MLP_LAYERS) - 1:
            h = _gelu(h)
    log_a = float(h.ravel()[0])
    b = float(h.ravel()[1])
    a = math.exp(log_a)
    return a, b


def _base_predict(row: Mapping[str, object]) -> float:
    if ARTIFACT is None or ENCODER is None or not MLP_LAYERS:
        return _clip_probability(GLOBAL_MEAN)

    try:
        embedding = ENCODER.encode(
            [_format_item_text(row)],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embed_arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        a, b = _item_params_from_embedding(embed_arr)
        subject_name = _parse_subject_name(str(row.get("subject_content", "")))
        theta = SUBJECT_THETA.get(subject_name, GLOBAL_THETA)
        logit = (a * theta - b) / max(TEMPERATURE, 1e-3)
        return _clip_probability(1.0 / (1.0 + math.exp(-logit)))
    except Exception as exc:  # noqa: BLE001
        print(f"[factor_pge] _base_predict fallback: {exc}", flush=True)
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


def _row_category(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("benchmark", "")), str(row.get("condition", "")))


def _calibration_offsets(
    labeled: list[dict] | None,
) -> tuple[float, dict[tuple[str, str], float]]:
    """Compute (global, per-category) residual offsets with EB shrinkage.

    v6: labeling.py now samples uniformly per category, so the K=5 revealed
    labels are an unbiased estimate of mean(label - base_predict) in that
    category. We shrink each per-category estimate toward the global mean
    using an empirical-Bayes weight of n / (n + _CALIB_SHRINKAGE), then clip
    the final offset to +-_CALIB_CLIP to bound damage if the model is well
    calibrated already.
    """
    if not labeled:
        return 0.0, {}
    residuals_global: list[float] = []
    bucket: dict[tuple[str, str], list[float]] = {}
    for row in labeled:
        if "label" not in row:
            continue
        try:
            r = float(row["label"]) - _base_predict(row)
        except Exception:  # noqa: BLE001
            continue
        residuals_global.append(r)
        bucket.setdefault(_row_category(row), []).append(r)
    if not residuals_global:
        return 0.0, {}
    global_mean = sum(residuals_global) / len(residuals_global)
    global_offset = max(-_CALIB_CLIP, min(_CALIB_CLIP, global_mean))
    per_cat: dict[tuple[str, str], float] = {}
    for cat, rs in bucket.items():
        n = len(rs)
        local_mean = sum(rs) / n
        w = n / (n + _CALIB_SHRINKAGE)
        shrunk = w * local_mean + (1.0 - w) * global_mean
        per_cat[cat] = max(-_CALIB_CLIP, min(_CALIB_CLIP, shrunk))
    return global_offset, per_cat


def _resolve_round(labeled: list[dict] | None) -> None:
    """Cache offsets for this evaluation round (Codabench calls predict() many
    times per round with the same `labeled` list)."""
    global _ROUND_CACHE_KEY, _ROUND_GLOBAL_OFFSET, _ROUND_CATEGORY_OFFSETS
    key = _labeled_key(labeled)
    if key == _ROUND_CACHE_KEY:
        return
    _ROUND_GLOBAL_OFFSET, _ROUND_CATEGORY_OFFSETS = _calibration_offsets(labeled)
    _ROUND_CACHE_KEY = key


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair.

    v6: uniform-label calibration. labeling.py picks K=5 items per category
    via a deterministic hash (no length bias). _calibration_offsets() then
    produces an unbiased per-category residual offset, shrunk toward the
    global mean with empirical-Bayes weighting and clipped to +-0.08. The
    offset is added in probability space, which is approximately a logit
    shift for moderate probabilities.
    """
    try:
        _resolve_round(labeled)
        base = _base_predict(input)
        cat = _row_category(input)
        offset = _ROUND_CATEGORY_OFFSETS.get(cat, _ROUND_GLOBAL_OFFSET)
        return _clip_probability(base + offset)
    except Exception as exc:  # noqa: BLE001
        print(f"[factor_pge] predict fallback: {exc}", flush=True)
        return _clip_probability(GLOBAL_MEAN)
