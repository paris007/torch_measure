"""Optional adaptive-label acquisition function for the embedding submission."""

from __future__ import annotations

import hashlib


def acquisition_function(input: dict) -> float:
    """Prefer longer, more substantive items with deterministic tie-breaking.

    Only the within-category ranking is used by the platform, so the absolute
    value is unimportant as long as it is finite and non-NaN.
    """
    item = input.get("item_content", "") or ""
    digest = hashlib.sha256(
        (input.get("benchmark", "") + "\n" + item).encode("utf-8", errors="ignore")
    ).hexdigest()
    jitter = int(digest[:10], 16) / float(16**10)
    length_score = min(len(item), 8000) / 8000.0
    return float(length_score + 0.01 * jitter)
