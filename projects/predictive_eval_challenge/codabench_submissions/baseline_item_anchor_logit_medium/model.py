"""Baseline v2 with inferred category-level adaptive calibration.

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
ITEM_MODE = 'logit'
ITEM_CLIP = 0.28
ITEM_SHRINK_N = 4.0
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"


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
_ROUND_OFFSETS = {"global": 0.0, "category": {}, "benchmark_condition": {}, "item": {}}

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


def _base_predict(row: Mapping[str, object]) -> float:
    """v2 baseline: prefer the deepest populated cell, fall back upward.

    Cell hierarchy (deepest to shallowest):
      subject_benchmark_condition (4-way)  ~6k cells, ~700 rows/cell
      subject_benchmark            (3-way)  ~1.5k cells
      benchmark_condition          (3-way)  ~223 cells
      subject                       (1-way) ~900 cells
      benchmark                     (1-way) 16 cells

    The deepest cell with >=1 observation is the lowest-variance unbiased
    estimator of P(correct) for new items in that cell (under the
    item-exchangeability assumption), so we prefer it whenever present.
    Only the final prediction is gently shrunk toward the global mean to
    bound damage in case the prior is mis-specified for new (subject,
    benchmark, condition) tuples we haven't seen.
    """
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
    except Exception:  # noqa: BLE001
        return ("__unsortable__", len(labeled))


def _item_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")
    item = str(row.get("item_content", ""))[:1600]
    return (benchmark, condition, item)


def _row_group(row: Mapping[str, object]) -> tuple[str, str]:
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")
    return (CATEGORY_BY_BENCHMARK.get(benchmark, benchmark), condition)


def _row_benchmark_condition(row: Mapping[str, object]) -> tuple[str, str]:
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")
    return (benchmark, condition)


def _clipped_mean(values: list[float], clip: float = 0.05) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return float(min(clip, max(-clip, mean)))


def _calibration_offsets(labeled: list[dict] | None) -> dict:
    if not labeled:
        return {"global": 0.0, "category": {}, "benchmark_condition": {}, "item": {}}
    global_residuals = []
    category_residuals: dict[tuple[str, str], list[float]] = {}
    bc_residuals: dict[tuple[str, str], list[float]] = {}
    item_prob_residuals: dict[tuple[str, str, str], list[float]] = {}
    item_logit_numerators: dict[tuple[str, str, str], float] = {}
    item_logit_denominators: dict[tuple[str, str, str], float] = {}
    for row in labeled:
        if "label" not in row:
            continue
        try:
            base_p = _base_predict(row)
            residual = float(row["label"]) - base_p
        except Exception:  # noqa: BLE001
            continue
        global_residuals.append(residual)
        category_residuals.setdefault(_row_group(row), []).append(residual)
        bc_residuals.setdefault(_row_benchmark_condition(row), []).append(residual)
        item = _item_key(row)
        item_prob_residuals.setdefault(item, []).append(residual)
        item_logit_numerators[item] = item_logit_numerators.get(item, 0.0) + residual
        item_logit_denominators[item] = item_logit_denominators.get(item, 0.0) + base_p * (1.0 - base_p)
    return {
        "global": _clipped_mean(global_residuals),
        "category": {key: _clipped_mean(values) for key, values in category_residuals.items()},
        "benchmark_condition": {
            key: _clipped_mean(values)
            for key, values in bc_residuals.items()
            if len(values) >= 2
        },
        "item": {
            key: _item_delta(key, item_prob_residuals, item_logit_numerators, item_logit_denominators)
            for key in item_prob_residuals
        },
    }


def _item_delta(
    key: tuple[str, str, str],
    prob_residuals: dict[tuple[str, str, str], list[float]],
    logit_numerators: dict[tuple[str, str, str], float],
    logit_denominators: dict[tuple[str, str, str], float],
) -> float:
    values = prob_residuals.get(key, [])
    if not values:
        return 0.0
    n = float(len(values))
    if ITEM_MODE == "logit":
        raw = logit_numerators.get(key, 0.0) / (ITEM_SHRINK_N + logit_denominators.get(key, 0.0))
    else:
        raw = (sum(values) / n) * (n / (n + ITEM_SHRINK_N))
    return float(min(ITEM_CLIP, max(-ITEM_CLIP, raw)))


def _apply_item_delta(base: float, input: Mapping[str, object], offsets: dict) -> float:
    delta = float(offsets.get("item", {}).get(_item_key(input), 0.0))
    if delta == 0.0:
        return _clip_probability(base)
    if ITEM_MODE == "logit":
        return _clip_probability(_sigmoid(_logit(base) + delta))
    return _clip_probability(base + delta)


def _calibration_offset(input: Mapping[str, object], offsets: dict) -> float:
    global_offset = float(offsets.get("global", 0.0))
    category_offset = float(offsets.get("category", {}).get(_row_group(input), 0.0))
    benchmark_offset = float(
        offsets.get("benchmark_condition", {}).get(_row_benchmark_condition(input), 0.0)
    )
    offset = 0.50 * category_offset + 0.25 * benchmark_offset + 0.25 * global_offset
    return float(min(0.05, max(-0.05, offset)))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair."""
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSETS = _calibration_offsets(labeled)
        base = _clip_probability(_base_predict(input) + _calibration_offset(input, _ROUND_OFFSETS))
        return _apply_item_delta(base, input, _ROUND_OFFSETS)
    except Exception as exc:  # noqa: BLE001
        print(f"[baseline] predict fallback: {exc}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
