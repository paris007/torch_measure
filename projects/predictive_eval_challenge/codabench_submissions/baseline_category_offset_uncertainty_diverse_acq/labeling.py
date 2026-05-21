"""Adaptive-label acquisition: prefer uncertain and simple-diverse candidates."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"
EPS = 1e-4
_SEEN_BUCKETS = Counter()


def _clip(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - EPS, max(EPS, value)))


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {{"global_mean": 0.6528605818748474}}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()


def _hash_noise(input: dict, scale: float = 1e-6) -> float:
    text = "\n".join(
        [input.get("benchmark", ""), input.get("condition", ""), input.get("subject_content", ""), input.get("item_content", "")]
    )
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return scale * float(int(digest[:8], 16) / 16**8)


def _bucket(input: dict) -> str:
    text = str(input.get("item_content", "") or "")
    lower = text.lower()
    length_bin = "short" if len(text) < 500 else "med" if len(text) < 1600 else "long"
    has_code = "code" if ("```" in text or "def " in lower or "class " in lower) else "nocode"
    has_math = "math" if any(sym in text for sym in ["∑", "√", "≤", "≥", "≈", "∫", "$", "\\frac", "^2"]) else "nomath"
    style = "mcq" if re.search(r"(?:^|\n)\s*\(?[A-Ja-j]\)?[\).:]", text) else "open"
    return "|".join([str(input.get("benchmark", "")), str(input.get("condition", "")), length_bin, has_code, has_math, style])


def _base_predict(row: dict) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {{}}).get(_key(subject_name, benchmark, condition))
    if sbc is not None:
        return _clip(0.95 * float(sbc) + 0.05 * global_mean)

    sb = ARTIFACT.get("subject_benchmark", {{}}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _clip(0.92 * float(sb) + 0.08 * global_mean)

    bc = ARTIFACT.get("benchmark_condition", {{}}).get(_key(benchmark, condition))
    s = ARTIFACT.get("subject", {{}}).get(subject_name)
    b = ARTIFACT.get("benchmark", {{}}).get(benchmark)

    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _clip(0.85 * pred + 0.15 * global_mean)


def acquisition_function(input: dict) -> float:
    try:
        p = _base_predict(input)
        uncertainty = max(0.0, 1.0 - 2.0 * abs(p - 0.5))
        b = _bucket(input)
        _SEEN_BUCKETS[b] += 1
        diversity = 1.0 / float(_SEEN_BUCKETS[b])
        return float(0.82 * uncertainty + 0.18 * diversity + _hash_noise(input))
    except Exception:
        return float(_hash_noise(input, scale=1.0))
