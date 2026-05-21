"""Codabench labeling acquisition for the factor / PGE submission.

v6: pure deterministic-hash selection (no length bias). The previous version
biased toward the longest items, which over-sampled hard benchmarks
(swebench, livecodebench, ...), pushed the residual offset systematically
negative, and dragged predictions down across all 5000 hidden items. Random
selection by hash gives a representative K=5 per category so the
per-category calibration in model.py gets unbiased residual estimates.
"""

from __future__ import annotations

import hashlib
from typing import Mapping


def _stable_score(parts: tuple[object, ...]) -> float:
    digest = hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def acquisition_function(input: Mapping[str, object]) -> float:
    """Higher score => more likely to be labeled by Codabench (top K per category).

    A uniform deterministic hash gives a representative sample within each
    category without overweighting any subset of items.
    """
    return _stable_score(
        (
            input.get("benchmark", ""),
            input.get("condition", ""),
            input.get("subject_content", ""),
            input.get("item_content", ""),
        )
    )
