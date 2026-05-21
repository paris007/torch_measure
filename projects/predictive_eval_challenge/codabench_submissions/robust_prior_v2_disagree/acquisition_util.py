"""Shared acquisition + baseline scoring for labeling.py and model.py.

Loaded once at import from artifacts in this submission folder. Keeps
train/serve logic aligned so acquisition scores match predict() baselines.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Mapping

EPS = 1e-4
ARTIFACTS = Path(__file__).parent / "artifacts"
BASELINE_PATH = ARTIFACTS / "smoothed_prior.json"

_BENCH_CAP_FRACTION = 0.30
_UNCERTAINTY_TIE_BREAK = 0.05


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _logit(p: float) -> float:
    p = _clip_probability(p)
    return math.log(p / (1.0 - p))


def _parse_subject_name(subject_content: object) -> str:
    text = str(subject_content) if subject_content is not None else ""
    match = re.search(r"^Name:\s*(.+)$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return text.strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _stable_score(parts: tuple[object, ...]) -> float:
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


_BASELINE: dict = {"global_mean": 0.5}
try:
    if BASELINE_PATH.exists():
        _BASELINE = json.loads(BASELINE_PATH.read_text())
except Exception:  # noqa: BLE001
    pass


def baseline_predict_v1(row: Mapping[str, object]) -> float:
    """v1 / v3-era 3-way weighted cells (no sbc-first path)."""
    global_mean = float(_BASELINE.get("global_mean", 0.5))
    subject_name = _parse_subject_name(row.get("subject_content"))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    pairs = [
        (_BASELINE.get("subject", {}).get(subject_name), 0.35),
        (_BASELINE.get("benchmark", {}).get(benchmark), 0.15),
        (_BASELINE.get("benchmark_condition", {}).get(_key(benchmark, condition)), 0.25),
        (_BASELINE.get("subject_benchmark", {}).get(_key(subject_name, benchmark)), 0.25),
    ]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip_probability(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _clip_probability(0.85 * pred + 0.15 * global_mean)


def baseline_predict_v2(row: Mapping[str, object]) -> float:
    """v2 prior used by the robust router: deep subject-benchmark-condition first."""
    global_mean = float(_BASELINE.get("global_mean", 0.5))
    subject_name = _parse_subject_name(row.get("subject_content"))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = _BASELINE.get("subject_benchmark_condition", {}).get(
        _key(subject_name, benchmark, condition)
    )
    if sbc is not None:
        return _clip_probability(0.95 * float(sbc) + 0.05 * global_mean)
    sb = _BASELINE.get("subject_benchmark", {}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _clip_probability(0.92 * float(sb) + 0.08 * global_mean)
    bc = _BASELINE.get("benchmark_condition", {}).get(_key(benchmark, condition))
    s = _BASELINE.get("subject", {}).get(subject_name)
    b = _BASELINE.get("benchmark", {}).get(benchmark)
    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip_probability(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _clip_probability(0.85 * pred + 0.15 * global_mean)


def category_prior(row: Mapping[str, object]) -> float:
    """Benchmark-condition cell mean, else global."""
    global_mean = float(_BASELINE.get("global_mean", 0.5))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")
    bc = _BASELINE.get("benchmark_condition", {}).get(_key(benchmark, condition))
    if bc is not None:
        return _clip_probability(float(bc))
    b = _BASELINE.get("benchmark", {}).get(benchmark)
    if b is not None:
        return _clip_probability(0.85 * float(b) + 0.15 * global_mean)
    return _clip_probability(global_mean)


def acquisition_hash(row: Mapping[str, object]) -> float:
    return _stable_score(
        (
            row.get("benchmark", ""),
            row.get("condition", ""),
            row.get("subject_content", ""),
            row.get("item_content", ""),
        )
    )


def acquisition_uncertainty(row: Mapping[str, object], *, spread_weight: float = 0.15) -> float:
    """Higher => labels that best identify whether v1 or v2 should win."""
    p1 = baseline_predict_v1(row)
    p2 = baseline_predict_v2(row)
    disagreement = abs(p2 - p1)
    confidence = max(abs(p1 - 0.5), abs(p2 - 0.5))
    spread = max(abs(p1 - category_prior(row)), abs(p2 - category_prior(row)))
    score = disagreement + 0.25 * confidence + spread_weight * spread
    score += _UNCERTAINTY_TIE_BREAK * acquisition_hash(row)
    return float(score) if math.isfinite(score) else acquisition_hash(row)


def apply_benchmark_cap(
    rows: list[Mapping[str, object]],
    scores: list[float],
    *,
    cap_fraction: float = _BENCH_CAP_FRACTION,
) -> list[float]:
    """Down-rank excess high scores from dominant benchmarks (offline / batch)."""
    if not rows or len(rows) != len(scores):
        return scores
    n = len(rows)
    max_per_bench = max(1, int(math.ceil(cap_fraction * n)))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    bench_counts: dict[str, int] = {}
    capped = list(scores)
    for idx in order:
        bench = str(rows[idx].get("benchmark", ""))
        if bench_counts.get(bench, 0) >= max_per_bench:
            capped[idx] = scores[idx] - 1.0
        else:
            bench_counts[bench] = bench_counts.get(bench, 0) + 1
    return capped


def acquisition_score(
    row: Mapping[str, object],
    *,
    mode: str = "uncertainty",
) -> float:
    if mode == "hash":
        return acquisition_hash(row)
    if mode == "capped_uncertainty":
        return acquisition_uncertainty(row)
    return acquisition_uncertainty(row)
