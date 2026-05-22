"""Codabench submission: smoothed prior with adaptive logit residuals."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Mapping

EPS = 1e-4
TARGET_ALPHA = 0.25
SHRINK_N = 6.0
LOGIT_CLIP = 0.22
W_GLOBAL = 0.35
W_CATEGORY = 0.35
W_BC = 0.05
W_SUBJECT_CATEGORY = 0.25
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"

CATEGORY_BY_BENCHMARK = {
    "swebench": "coding",
    "livecodebench": "coding",
    "bfcl": "tool_use",
    "agentdojo": "tool_use",
    "androidworld": "tool_use",
    "matharena": "math",
    "mathvista_mini": "math_vision",
    "ai2d_test": "vision",
    "mmbench_v11": "vision",
    "mmlupro": "knowledge",
    "hle": "knowledge",
    "rewardbench": "preference",
    "ultrafeedback": "preference",
    "mtbench": "chat",
    "afrimedqa": "medical",
    "cybench": "cyber",
}


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _logit(p: float) -> float:
    p = _clip_probability(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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
_ROUND_OFFSETS = {"global": 0.0, "category": {}, "benchmark_condition": {}, "subject_category": {}}


def _category(row: Mapping[str, object]) -> str:
    benchmark = str(row.get("benchmark", ""))
    return CATEGORY_BY_BENCHMARK.get(benchmark, benchmark)


def _row_group(row: Mapping[str, object]) -> tuple[str, str]:
    condition = str(row.get("condition", "none") or "none")
    return (_category(row), condition)


def _row_benchmark_condition(row: Mapping[str, object]) -> tuple[str, str]:
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")
    return (benchmark, condition)


def _row_subject_category(row: Mapping[str, object]) -> tuple[str, str, str]:
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    condition = str(row.get("condition", "none") or "none")
    return (subject_name, _category(row), condition)


def _base_predict(row: Mapping[str, object]) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {}).get(_key(subject_name, benchmark, condition))
    if sbc is not None:
        return _clip_probability(0.95 * float(sbc) + 0.05 * global_mean)
    sb = ARTIFACT.get("subject_benchmark", {}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _clip_probability(0.92 * float(sb) + 0.08 * global_mean)
    bc = ARTIFACT.get("benchmark_condition", {}).get(_key(benchmark, condition))
    s = ARTIFACT.get("subject", {}).get(subject_name)
    b = ARTIFACT.get("benchmark", {}).get(benchmark)
    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip_probability(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _clip_probability(0.85 * pred + 0.15 * global_mean)


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
    except Exception:
        return ("__unsortable__", len(labeled))


def _shrunk_clipped_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    n = float(len(values))
    raw = sum(values) / n
    shrunk = (n / (n + SHRINK_N)) * raw
    return float(min(LOGIT_CLIP, max(-LOGIT_CLIP, shrunk)))


def _target_logit(label: object) -> float:
    y = 1.0 if float(label) >= 0.5 else 0.0
    p = (y + TARGET_ALPHA) / (1.0 + 2.0 * TARGET_ALPHA)
    return _logit(p)


def _calibration_offsets(labeled: list[dict] | None) -> dict:
    if not labeled:
        return {"global": 0.0, "category": {}, "benchmark_condition": {}, "subject_category": {}}
    global_residuals = []
    category_residuals: dict[tuple[str, str], list[float]] = {}
    bc_residuals: dict[tuple[str, str], list[float]] = {}
    subject_category_residuals: dict[tuple[str, str, str], list[float]] = {}
    for row in labeled:
        if "label" not in row:
            continue
        try:
            residual = _target_logit(row["label"]) - _logit(_base_predict(row))
        except Exception:
            continue
        if not math.isfinite(residual):
            continue
        global_residuals.append(residual)
        category_residuals.setdefault(_row_group(row), []).append(residual)
        bc_residuals.setdefault(_row_benchmark_condition(row), []).append(residual)
        subject_category_residuals.setdefault(_row_subject_category(row), []).append(residual)
    return {
        "global": _shrunk_clipped_mean(global_residuals),
        "category": {key: _shrunk_clipped_mean(values) for key, values in category_residuals.items()},
        "benchmark_condition": {
            key: _shrunk_clipped_mean(values)
            for key, values in bc_residuals.items()
            if len(values) >= 2
        },
        "subject_category": {
            key: _shrunk_clipped_mean(values)
            for key, values in subject_category_residuals.items()
        },
    }


def _calibration_logit_offset(row: Mapping[str, object], offsets: dict) -> float:
    global_offset = float(offsets.get("global", 0.0))
    category_offset = float(offsets.get("category", {}).get(_row_group(row), 0.0))
    benchmark_offset = float(offsets.get("benchmark_condition", {}).get(_row_benchmark_condition(row), 0.0))
    subject_category_offset = float(offsets.get("subject_category", {}).get(_row_subject_category(row), 0.0))
    offset = (
        W_GLOBAL * global_offset
        + W_CATEGORY * category_offset
        + W_BC * benchmark_offset
        + W_SUBJECT_CATEGORY * subject_category_offset
    )
    return float(min(LOGIT_CLIP, max(-LOGIT_CLIP, offset)))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSETS = _calibration_offsets(labeled)
        base = _base_predict(input)
        return _clip_probability(_sigmoid(_logit(base) + _calibration_logit_offset(input, _ROUND_OFFSETS)))
    except Exception as exc:
        print(f"[logit_residual] predict fallback: {exc}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
