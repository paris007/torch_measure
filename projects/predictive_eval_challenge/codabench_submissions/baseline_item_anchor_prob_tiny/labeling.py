"""Item-anchor acquisition for adaptive labels."""

from __future__ import annotations

import hashlib

SALT = "item_anchor_prob_tiny_parism"


def acquisition_function(input: dict) -> float:
    text = "\n".join(
        [
            SALT,
            str(input.get("benchmark", "")),
            str(input.get("condition", "none") or "none"),
            str(input.get("item_content", ""))[:1600],
        ]
    )
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return float(int(digest[:12], 16) / 16**12)
