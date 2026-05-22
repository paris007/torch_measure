"""Optional adaptive-label acquisition function for the baseline submission."""

from __future__ import annotations

import hashlib


def acquisition_function(input: dict) -> float:
    """Return a deterministic diversity score.

    The absolute value is ignored by Codabench; only the within-category ranking
    matters. This score is cheap, finite, and deterministic, so it will not
    trigger the random fallback.
    """
    text = "\n".join(
        [
            input.get("benchmark", ""),
            input.get("condition", ""),
            input.get("subject_content", ""),
            input.get("item_content", ""),
        ]
    )
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return float(int(digest[:12], 16) / 16**12)
