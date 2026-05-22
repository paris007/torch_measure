from __future__ import annotations
import hashlib

SALT = "category_logit_subject_plus_global_parism"


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
