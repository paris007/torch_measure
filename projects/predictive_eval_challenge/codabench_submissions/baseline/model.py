"""Codabench baseline submission for the Predictive Evaluation Challenge.

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
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


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
_ROUND_OFFSET = 0.0


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
    return float(min(0.12, max(-0.12, sum(residuals) / len(residuals))))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) for one hidden subject-item pair."""
    global _ROUND_CACHE_KEY, _ROUND_OFFSET
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSET = _calibration_offset(labeled)
        return _clip_probability(_base_predict(input) + _ROUND_OFFSET)
    except Exception as exc:  # noqa: BLE001
        print(f"[baseline] predict fallback: {exc}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
