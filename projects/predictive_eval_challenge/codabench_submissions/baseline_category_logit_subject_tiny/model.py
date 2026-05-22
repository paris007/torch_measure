from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

EPS = 1e-4
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"

LOGIT_CLIP = 0.3
GLOBAL_WEIGHT = 0.2
CATEGORY_WEIGHT = 0.45
BC_WEIGHT = 0.2
SUBJECT_CATEGORY_WEIGHT = 0.15
GLOBAL_PRIOR = 5.0
CATEGORY_PRIOR = 5.0
BC_PRIOR = 7.0
SUBJECT_CATEGORY_PRIOR = 12.0

CATEGORY_BY_BENCHMARK = {
    "swebench": "coding", "livecodebench": "coding", "bigcodebench": "coding",
    "humaneval": "coding", "mbpp": "coding",
    "bfcl": "tool_use", "agentdojo": "tool_use", "androidworld": "tool_use", "tau2": "tool_use",
    "matharena": "math", "mathvista_mini": "math_vision", "gsm8k": "math", "aime": "math",
    "ai2d_test": "vision", "mmbench_v11": "vision", "mmmu": "vision",
    "mmlupro": "knowledge", "hle": "knowledge", "mmlu": "knowledge", "gpqa": "knowledge",
    "rewardbench": "preference", "ultrafeedback": "preference", "mtbench": "chat",
    "afrimedqa": "medical", "medqa": "medical", "cybench": "cyber",
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


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _category(row: Mapping[str, object]) -> str:
    benchmark_raw = str(row.get("benchmark", ""))
    benchmark = _norm(benchmark_raw)
    return str(CATEGORY_BY_BENCHMARK.get(benchmark, benchmark_raw))


def _group_key(row: Mapping[str, object]) -> str:
    return _key(_category(row), str(row.get("condition", "none") or "none"))


def _bc_key(row: Mapping[str, object]) -> str:
    return _key(str(row.get("benchmark", "")), str(row.get("condition", "none") or "none"))


def _subject_category_key(row: Mapping[str, object]) -> str:
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    return _key(subject_name, _category(row), str(row.get("condition", "none") or "none"))


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {"global_mean": 0.6528605818748474}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()
_ROUND_CACHE_KEY = None
_ROUND_OFFSETS = (0.0, {}, {}, {})


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


def _newton_logit_offset(rows: list[dict], prior_precision: float) -> float:
    if not rows:
        return 0.0
    score = 0.0
    info = 0.0
    for row in rows:
        if "label" not in row:
            continue
        try:
            y = float(row["label"])
            p = _base_predict(row)
        except Exception:
            continue
        if not math.isfinite(y) or not math.isfinite(p):
            continue
        score += y - p
        info += p * (1.0 - p)
    if info <= 0.0:
        return 0.0
    delta = score / (float(prior_precision) + info)
    return float(min(LOGIT_CLIP, max(-LOGIT_CLIP, delta)))


def _fit_offsets(labeled: list[dict] | None):
    if not labeled:
        return (0.0, {}, {}, {})

    valid = [row for row in labeled if "label" in row]
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_bc: dict[str, list[dict]] = defaultdict(list)
    by_subject_category: dict[str, list[dict]] = defaultdict(list)

    for row in valid:
        by_group[_group_key(row)].append(row)
        by_bc[_bc_key(row)].append(row)
        by_subject_category[_subject_category_key(row)].append(row)

    global_offset = _newton_logit_offset(valid, GLOBAL_PRIOR)
    group_offsets = {k: _newton_logit_offset(v, CATEGORY_PRIOR) for k, v in by_group.items()}
    bc_offsets = {k: _newton_logit_offset(v, BC_PRIOR) for k, v in by_bc.items()}
    subject_category_offsets = {k: _newton_logit_offset(v, SUBJECT_CATEGORY_PRIOR) for k, v in by_subject_category.items()}
    return global_offset, group_offsets, bc_offsets, subject_category_offsets


def _adaptive_logit_offset(row: Mapping[str, object]) -> float:
    global_offset, group_offsets, bc_offsets, subject_category_offsets = _ROUND_OFFSETS
    offset = (
        GLOBAL_WEIGHT * float(global_offset)
        + CATEGORY_WEIGHT * float(group_offsets.get(_group_key(row), 0.0))
        + BC_WEIGHT * float(bc_offsets.get(_bc_key(row), 0.0))
        + SUBJECT_CATEGORY_WEIGHT * float(subject_category_offsets.get(_subject_category_key(row), 0.0))
    )
    return float(min(LOGIT_CLIP, max(-LOGIT_CLIP, offset)))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSETS = _fit_offsets(labeled)

        base = _base_predict(input)
        logit = _logit(base) + _adaptive_logit_offset(input)
        return _clip_probability(_sigmoid(logit))
    except Exception as exc:
        print(f"[logit_adaptive] predict fallback: {exc}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
