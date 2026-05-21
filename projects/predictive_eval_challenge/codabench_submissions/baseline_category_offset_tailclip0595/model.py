"""Codabench submission: category-offset prior plus one tiny exploration change.

This keeps the current winning structure:
  smoothed prior -> global/category/benchmark-condition adaptive offset

Then it optionally applies one of:
  - tail clipping
  - tiny item-length residual
  - tiny item-complexity residual

All changes are intentionally small because the category-offset prior is already
the best observed family.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

EPS = 1e-4
TAIL_LOW = 0.05
TAIL_HIGH = 0.95
OFFSET_CLIP = 0.05
SHRINK_N = 5.0
W_GLOBAL = 0.25
W_CATEGORY = 0.50
W_BC = 0.25
LENGTH_WEIGHT = 0.0
COMPLEXITY_WEIGHT = 0.0
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
    lo = max(eps, float(TAIL_LOW))
    hi = min(1.0 - eps, float(TAIL_HIGH))
    if hi <= lo:
        lo, hi = eps, 1.0 - eps
    return float(min(hi, max(lo, value)))


def _safe_clip01(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _logit(p: float) -> float:
    p = _safe_clip01(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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
_ROUND_OFFSETS = (0.0, {}, {})


def _base_predict(row: Mapping[str, object]) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {}).get(
        _key(subject_name, benchmark, condition)
    )
    if sbc is not None:
        return _safe_clip01(0.95 * float(sbc) + 0.05 * global_mean)

    sb = ARTIFACT.get("subject_benchmark", {}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _safe_clip01(0.92 * float(sb) + 0.08 * global_mean)

    bc = ARTIFACT.get("benchmark_condition", {}).get(_key(benchmark, condition))
    s = ARTIFACT.get("subject", {}).get(subject_name)
    b = ARTIFACT.get("benchmark", {}).get(benchmark)

    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _safe_clip01(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _safe_clip01(0.85 * pred + 0.15 * global_mean)


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

    return (
        _shrunk_mean(all_resid),
        {k: _shrunk_mean(v) for k, v in by_group.items()},
        {k: _shrunk_mean(v) for k, v in by_bc.items()},
    )


def _adaptive_offset(row: Mapping[str, object]) -> float:
    global_offset, group_offsets, bc_offsets = _ROUND_OFFSETS
    group_offset = float(group_offsets.get(_group_key(row), 0.0))
    bc_offset = float(bc_offsets.get(_bc_key(row), 0.0))
    offset = W_GLOBAL * float(global_offset) + W_CATEGORY * group_offset + W_BC * bc_offset
    return float(min(OFFSET_CLIP, max(-OFFSET_CLIP, offset)))


def _item_length_logit_delta(row: Mapping[str, object]) -> float:
    if LENGTH_WEIGHT == 0.0:
        return 0.0
    text = str(row.get("item_content", "") or "")
    n = len(text)
    z = (min(max(n, 0), 5000) - 900.0) / 3000.0
    z = min(1.0, max(-0.30, z))
    return float(LENGTH_WEIGHT * z)


def _item_complexity_logit_delta(row: Mapping[str, object]) -> float:
    if COMPLEXITY_WEIGHT == 0.0:
        return 0.0
    text = str(row.get("item_content", "") or "")
    lower = text.lower()

    score = 0.0
    if "```" in text or "def " in lower or "class " in lower or "import " in lower:
        score += 1.0
    if any(sym in text for sym in ["∑", "√", "≤", "≥", "≈", "∫", "$", "\\frac", "^2"]):
        score += 0.8
    if any(word in lower for word in ["prove", "derive", "calculate", "compute", "solve", "estimate"]):
        score += 0.6
    if any(word in lower for word in ["vulnerability", "exploit", "cve", "payload", "xss", "sql injection"]):
        score += 0.5
    if any(word in lower for word in ["patient", "diagnosis", "symptom", "treatment", "clinical"]):
        score += 0.4
    option_count = len(re.findall(r"(?:^|\n)\s*\(?[A-Ja-j]\)?[\).:]", text))
    if option_count >= 5:
        score += 0.5

    score = min(3.0, max(0.0, score))
    return float(COMPLEXITY_WEIGHT * score)


def _apply_item_residual(row: Mapping[str, object], p: float) -> float:
    delta = _item_length_logit_delta(row) + _item_complexity_logit_delta(row)
    if delta == 0.0:
        return p
    return _safe_clip01(_sigmoid(_logit(p) + delta))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSETS = _fit_offsets(labeled)
        p = _safe_clip01(_base_predict(input) + _adaptive_offset(input))
        p = _apply_item_residual(input, p)
        return _clip_probability(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[wave1_variant] predict fallback: {exc}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
