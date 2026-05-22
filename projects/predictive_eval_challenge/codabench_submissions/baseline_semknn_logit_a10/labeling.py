from __future__ import annotations
import hashlib
SALT="semknn_logit_a10_parism"
def acquisition_function(input: dict) -> float:
    text="\n".join([SALT,input.get("benchmark",""),input.get("condition",""),input.get("item_content",""),input.get("subject_content","")])
    return float(int(hashlib.sha256(text.encode("utf-8",errors="ignore")).hexdigest()[:12],16)/16**12)
