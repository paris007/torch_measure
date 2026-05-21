"""Codabench submission: category-offset prior with final log-loss calibration.

This keeps the winning category-offset strategy:
  base prior -> adaptive global/category/benchmark-condition offset
Then it applies a tiny final calibration transform:
  p_final = sigmoid(LOGIT_SCALE * logit(p) + LOGIT_BIAS) + PROB_SHIFT

The goal is to improve negative log-loss by making probabilities slightly more
or less confident without changing the core prior model.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

EPS = 1e-4
OFFSET_CLIP = 0.05
SHRINK_N = 5.0
W_GLOBAL = 0.25
W_CATEGORY = 0.50
W_BC = 0.25
LOGIT_SCALE = 1.0
LOGIT_BIAS = 0.0
PROB_SHIFT = 0.02
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"

CATEGORY_BY_BENCHMARK = {
    # Coding / software engineering
    "swebench": "coding",
    "livecodebench": "coding",
    "bigcodebench": "coding",
    "humaneval": "coding",
    "mbpp": "coding",
    # Tool / agent use
    "bfcl": "tool_use",
    "agentdojo": "tool_use",
    "androidworld": "tool_use",
    "tau2": "tool_use",
    # Math / STEM reasoning
    "matharena": "math",
    "mathvista_mini": "math_vision",
    "gsm8k": "math",
    "aime": "math",
    # Vision / multimodal
    "ai2d_test": "vision",
    "mmbench_v11": "vision",
    "mmmu": "vision",
    # Knowledge / exams
    "mmlupro": "knowledge",
    "hle": "knowledge",
    "mmlu": "knowledge",
    "gpqa": "knowledge",
    # Preference / chat quality
    "rewardbench": "preference",
    "ultrafeedback": "preference",
    "mtbench": "chat",
    # Specialty domains
    "afrimedqa": "medical",
    "medqa": "medical",
    "cybench": "cyber",
}


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


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


def _final_calibrate(p: float) -> float:
    # LOGIT_SCALE < 1 softens overconfident predictions.
    # LOGIT_SCALE > 1 sharpens underconfident predictions.
    # PROB_SHIFT tests a tiny global up/down probability shift.
    p2 = _sigmoid(LOGIT_SCALE * _logit(p) + LOGIT_BIAS)
    return _clip_probability(p2 + PROB_SHIFT)


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _group_key(row: Mapping[str, object]) -> str:
    benchmark_raw = str(row.get("benchmark", ""))
    benchmark = _norm(benchmark_raw)
    condition = str(row.get("condition", "none") or "none")
    category = CATEGORY_BY_BENCHMARK.get(benchmark, benchmark_raw)
    return _key(category, condition)


def _bc_key(row: Mapping[str, object]) -> str:
    return _key(str(row.get("benchmark", "")), str(row.get("condition", "none") or "none"))


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {"global_mean": 0.6528605818748474}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()
_ROUND_CACHE_KEY = None
_ROUND_OFFSETS = (0.0, {}, {})  # global_offset, category_offsets, benchmark_condition_offsets


def _base_predict(row: Mapping[str, object]) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {}).get(
        _key(subject_name, benchmark, condition)
    )
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
    except Exception:  # noqa: BLE001
        return ("__unsortable__", len(labeled))


def _shrunk_mean(values: list[float], shrink_n: float = SHRINK_N) -> float:
    if not values:
        return 0.0
    n = float(len(values))
    raw = sum(values) / n
    return float((n / (n + shrink_n)) * raw)


def _fit_offsets(labeled: list[dict] | None):
    if not labeled:
        return (0.0, {}, {})
    all_resid: list[float] = []
    by_group: dict[str, list[float]] = defaultdict(list)
    by_bc: dict[str, list[float]] = defaultdict(list)

    for row in labeled:
        if "label" not in row:
            continue
        try:
            resid = float(row["label"]) - _base_predict(row)
        except Exception:  # noqa: BLE001
            continue
        if not math.isfinite(resid):
            continue
        all_resid.append(resid)
        by_group[_group_key(row)].append(resid)
        by_bc[_bc_key(row)].append(resid)

    global_offset = _shrunk_mean(all_resid)
    group_offsets = {k: _shrunk_mean(v) for k, v in by_group.items()}
    bc_offsets = {k: _shrunk_mean(v) for k, v in by_bc.items()}
    return global_offset, group_offsets, bc_offsets


def _adaptive_offset(row: Mapping[str, object]) -> float:
    global_offset, group_offsets, bc_offsets = _ROUND_OFFSETS
    group_offset = float(group_offsets.get(_group_key(row), 0.0))
    bc_offset = float(bc_offsets.get(_bc_key(row), 0.0))
    offset = W_GLOBAL * float(global_offset) + W_CATEGORY * group_offset + W_BC * bc_offset
    return float(min(OFFSET_CLIP, max(-OFFSET_CLIP, offset)))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSETS = _fit_offsets(labeled)
        p = _clip_probability(_base_predict(input) + _adaptive_offset(input))
        return _final_calibrate(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[calibration_variant] predict fallback: {exc}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
