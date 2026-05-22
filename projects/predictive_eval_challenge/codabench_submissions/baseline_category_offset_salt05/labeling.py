"""Salted pseudo-random adaptive-label acquisition.

SALT = "category_offset_salt_05_parism"
"""
from __future__ import annotations

import hashlib

SALT = "category_offset_salt_05_parism"


def acquisition_function(input: dict) -> float:
    text = "\n".join(
        [
            SALT,
            input.get("benchmark", ""),
            input.get("condition", ""),
            input.get("subject_content", ""),
            input.get("item_content", ""),
        ]
    )
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return float(int(digest[:12], 16) / 16**12)
