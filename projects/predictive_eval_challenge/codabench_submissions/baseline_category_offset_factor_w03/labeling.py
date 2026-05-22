from __future__ import annotations
import hashlib

def acquisition_function(input: dict) -> float:
    text="\n".join([input.get("benchmark",""),input.get("condition",""),input.get("subject_content",""),input.get("item_content","")])
    return float(int(hashlib.sha256(text.encode("utf-8",errors="ignore")).hexdigest()[:12],16)/16**12)
