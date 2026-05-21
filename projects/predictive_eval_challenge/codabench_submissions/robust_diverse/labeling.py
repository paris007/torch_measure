"""Diversity-aware acquisition for the robust factor + baseline ensemble.

The challenge handbook notes that `acquisition_function()` is called once per
candidate but may keep module-level state during the candidate pass. We use
that to avoid spending all K labels in a hidden data category on nearly
duplicate rows: candidates in coarse buckets seen fewer times receive higher
priority, with the existing baseline uncertainty used inside each tier.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

from acquisition_util import acquisition_hash, acquisition_uncertainty


_SEEN_BUCKETS: Counter[tuple[str, str, str, int, int]] = Counter()


def _subject_family(subject_content: object) -> str:
    text = str(subject_content or "").lower()
    match = re.search(r"^name:\s*([^\n]+)", text, flags=re.MULTILINE)
    name = match.group(1) if match else text[:80]
    for token in ("gpt", "claude", "gemini", "llama", "qwen", "mistral", "deepseek"):
        if token in name:
            return token
    return name.split("/")[0].split("-")[0].split()[0][:24] if name.strip() else ""


def _bucket(input: Mapping[str, object]) -> tuple[str, str, str, int, int]:
    item = str(input.get("item_content", ""))
    length_bin = min(4, len(item) // 500)
    # Stable lexical bucket: cheap stand-in for the PDF's embedding clusters.
    lexical_bin = int(acquisition_hash({"item_content": item}) * 16.0)
    return (
        str(input.get("benchmark", "")),
        str(input.get("condition", "")),
        _subject_family(input.get("subject_content", "")),
        length_bin,
        lexical_bin,
    )


def acquisition_function(input: Mapping[str, object]) -> float:
    """Higher score => more likely to be labeled (top K per category)."""
    try:
        bucket = _bucket(input)
        _SEEN_BUCKETS[bucket] += 1
        diversity_tier = 1.0 / float(_SEEN_BUCKETS[bucket])
        uncertainty = acquisition_uncertainty(input)
        # Diversity dominates; uncertainty and hash break ties safely.
        return float(diversity_tier + 0.1 * uncertainty + 1e-4 * acquisition_hash(input))
    except Exception:  # noqa: BLE001
        return acquisition_hash(input)
