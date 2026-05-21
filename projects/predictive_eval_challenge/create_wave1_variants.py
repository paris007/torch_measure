#!/usr/bin/env python3
"""Create Wave-1 Codabench variants around the current winning category-offset model.

How to use:
1. Save this file as:
   projects/predictive_eval_challenge/create_wave1_variants.py

2. From the repo root, run:
   python projects/predictive_eval_challenge/create_wave1_variants.py

3. Upload the ZIPs printed at the end from:
   projects/predictive_eval_challenge/dist/wave1_variants/

These variants do NOT replace the current winner. They are exploratory variants
for the next 6 submissions. Keep rerunning baseline_category_offset_submission.zip
as the control/winner.
"""
from __future__ import annotations

import shutil
import textwrap
import zipfile
from pathlib import Path


MODEL_TEMPLATE = r"""
"""Codabench submission: category-offset prior plus one tiny exploration change.

This keeps the current winning structure:
  smoothed prior -> global/category/benchmark-condition adaptive offset

Then it optionally applies one of:
  - tail clipping
  - tiny item-length residual
  - tiny item-complexity residual

All changes are intentionally small because the category-offset prior is already
the best observed family.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

EPS = 1e-4
TAIL_LOW = {tail_low}
TAIL_HIGH = {tail_high}
OFFSET_CLIP = 0.05
SHRINK_N = 5.0
W_GLOBAL = 0.25
W_CATEGORY = 0.50
W_BC = 0.25
LENGTH_WEIGHT = {length_weight}
COMPLEXITY_WEIGHT = {complexity_weight}
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"

CATEGORY_BY_BENCHMARK = {{
    # Coding / software engineering
    "swebench": "coding",
    "livecodebench": "coding",
    "bigcodebench": "coding",
    "humaneval": "coding",
    "mbpp": "coding",
    # Tool / agent use
    "bfcl": "tool_use",
    "agentdojo": "tool_use",
    "androidworld": "tool_use",
    "tau2": "tool_use",
    # Math / STEM reasoning
    "matharena": "math",
    "mathvista_mini": "math_vision",
    "gsm8k": "math",
    "aime": "math",
    # Vision / multimodal
    "ai2d_test": "vision",
    "mmbench_v11": "vision",
    "mmmu": "vision",
    # Knowledge / exams
    "mmlupro": "knowledge",
    "hle": "knowledge",
    "mmlu": "knowledge",
    "gpqa": "knowledge",
    # Preference / chat quality
    "rewardbench": "preference",
    "ultrafeedback": "preference",
    "mtbench": "chat",
    # Specialty domains
    "afrimedqa": "medical",
    "medqa": "medical",
    "cybench": "cyber",
}}


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    lo = max(eps, float(TAIL_LOW))
    hi = min(1.0 - eps, float(TAIL_HIGH))
    if hi <= lo:
        lo, hi = eps, 1.0 - eps
    return float(min(hi, max(lo, value)))


def _safe_clip01(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _logit(p: float) -> float:
    p = _safe_clip01(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _group_key(row: Mapping[str, object]) -> str:
    benchmark_raw = str(row.get("benchmark", ""))
    benchmark = _norm(benchmark_raw)
    condition = str(row.get("condition", "none") or "none")
    category = CATEGORY_BY_BENCHMARK.get(benchmark, benchmark_raw)
    return _key(category, condition)


def _bc_key(row: Mapping[str, object]) -> str:
    return _key(str(row.get("benchmark", "")), str(row.get("condition", "none") or "none"))


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {{"global_mean": 0.6528605818748474}}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()
_ROUND_CACHE_KEY = None
_ROUND_OFFSETS = (0.0, {{}}, {{}})


def _base_predict(row: Mapping[str, object]) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {{}}).get(
        _key(subject_name, benchmark, condition)
    )
    if sbc is not None:
        return _safe_clip01(0.95 * float(sbc) + 0.05 * global_mean)

    sb = ARTIFACT.get("subject_benchmark", {{}}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _safe_clip01(0.92 * float(sb) + 0.08 * global_mean)

    bc = ARTIFACT.get("benchmark_condition", {{}}).get(_key(benchmark, condition))
    s = ARTIFACT.get("subject", {{}}).get(subject_name)
    b = ARTIFACT.get("benchmark", {{}}).get(benchmark)

    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _safe_clip01(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _safe_clip01(0.85 * pred + 0.15 * global_mean)


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


def _shrunk_mean(values: list[float], shrink_n: float = SHRINK_N) -> float:
    if not values:
        return 0.0
    n = float(len(values))
    raw = sum(values) / n
    return float((n / (n + shrink_n)) * raw)


def _fit_offsets(labeled: list[dict] | None):
    if not labeled:
        return (0.0, {{}}, {{}})
    all_resid: list[float] = []
    by_group: dict[str, list[float]] = defaultdict(list)
    by_bc: dict[str, list[float]] = defaultdict(list)

    for row in labeled:
        if "label" not in row:
            continue
        try:
            resid = float(row["label"]) - _base_predict(row)
        except Exception:  # noqa: BLE001
            continue
        if not math.isfinite(resid):
            continue
        all_resid.append(resid)
        by_group[_group_key(row)].append(resid)
        by_bc[_bc_key(row)].append(resid)

    return (
        _shrunk_mean(all_resid),
        {{k: _shrunk_mean(v) for k, v in by_group.items()}},
        {{k: _shrunk_mean(v) for k, v in by_bc.items()}},
    )


def _adaptive_offset(row: Mapping[str, object]) -> float:
    global_offset, group_offsets, bc_offsets = _ROUND_OFFSETS
    group_offset = float(group_offsets.get(_group_key(row), 0.0))
    bc_offset = float(bc_offsets.get(_bc_key(row), 0.0))
    offset = W_GLOBAL * float(global_offset) + W_CATEGORY * group_offset + W_BC * bc_offset
    return float(min(OFFSET_CLIP, max(-OFFSET_CLIP, offset)))


def _item_length_logit_delta(row: Mapping[str, object]) -> float:
    if LENGTH_WEIGHT == 0.0:
        return 0.0
    text = str(row.get("item_content", "") or "")
    n = len(text)
    z = (min(max(n, 0), 5000) - 900.0) / 3000.0
    z = min(1.0, max(-0.30, z))
    return float(LENGTH_WEIGHT * z)


def _item_complexity_logit_delta(row: Mapping[str, object]) -> float:
    if COMPLEXITY_WEIGHT == 0.0:
        return 0.0
    text = str(row.get("item_content", "") or "")
    lower = text.lower()

    score = 0.0
    if "```" in text or "def " in lower or "class " in lower or "import " in lower:
        score += 1.0
    if any(sym in text for sym in ["∑", "√", "≤", "≥", "≈", "∫", "$", "\\frac", "^2"]):
        score += 0.8
    if any(word in lower for word in ["prove", "derive", "calculate", "compute", "solve", "estimate"]):
        score += 0.6
    if any(word in lower for word in ["vulnerability", "exploit", "cve", "payload", "xss", "sql injection"]):
        score += 0.5
    if any(word in lower for word in ["patient", "diagnosis", "symptom", "treatment", "clinical"]):
        score += 0.4
    option_count = len(re.findall(r"(?:^|\n)\s*\(?[A-Ja-j]\)?[\).:]", text))
    if option_count >= 5:
        score += 0.5

    score = min(3.0, max(0.0, score))
    return float(COMPLEXITY_WEIGHT * score)


def _apply_item_residual(row: Mapping[str, object], p: float) -> float:
    delta = _item_length_logit_delta(row) + _item_complexity_logit_delta(row)
    if delta == 0.0:
        return p
    return _safe_clip01(_sigmoid(_logit(p) + delta))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSETS = _fit_offsets(labeled)
        p = _safe_clip01(_base_predict(input) + _adaptive_offset(input))
        p = _apply_item_residual(input, p)
        return _clip_probability(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[wave1_variant] predict fallback: {{exc}}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
"""


LABELING_HASH = r"""
"""Cheap deterministic adaptive-label acquisition function."""
from __future__ import annotations

import hashlib


def acquisition_function(input: dict) -> float:
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
"""


LABELING_UNCERTAINTY = r"""
"""Adaptive-label acquisition: prefer examples where the current prior is uncertain."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"
EPS = 1e-4


def _clip(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - EPS, max(EPS, value)))


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {{"global_mean": 0.6528605818748474}}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()


def _hash_noise(input: dict, scale: float = 1e-6) -> float:
    text = "\n".join(
        [input.get("benchmark", ""), input.get("condition", ""), input.get("subject_content", ""), input.get("item_content", "")]
    )
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return scale * float(int(digest[:8], 16) / 16**8)


def _base_predict(row: dict) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {{}}).get(_key(subject_name, benchmark, condition))
    if sbc is not None:
        return _clip(0.95 * float(sbc) + 0.05 * global_mean)

    sb = ARTIFACT.get("subject_benchmark", {{}}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _clip(0.92 * float(sb) + 0.08 * global_mean)

    bc = ARTIFACT.get("benchmark_condition", {{}}).get(_key(benchmark, condition))
    s = ARTIFACT.get("subject", {{}}).get(subject_name)
    b = ARTIFACT.get("benchmark", {{}}).get(benchmark)

    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _clip(0.85 * pred + 0.15 * global_mean)


def acquisition_function(input: dict) -> float:
    try:
        p = _base_predict(input)
        uncertainty = 1.0 - 2.0 * abs(p - 0.5)
        return float(max(0.0, uncertainty) + _hash_noise(input))
    except Exception:
        return float(_hash_noise(input, scale=1.0))
"""


LABELING_UNCERTAINTY_DIVERSE = r"""
"""Adaptive-label acquisition: prefer uncertain and simple-diverse candidates."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"
EPS = 1e-4
_SEEN_BUCKETS = Counter()


def _clip(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - EPS, max(EPS, value)))


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {{"global_mean": 0.6528605818748474}}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()


def _hash_noise(input: dict, scale: float = 1e-6) -> float:
    text = "\n".join(
        [input.get("benchmark", ""), input.get("condition", ""), input.get("subject_content", ""), input.get("item_content", "")]
    )
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return scale * float(int(digest[:8], 16) / 16**8)


def _bucket(input: dict) -> str:
    text = str(input.get("item_content", "") or "")
    lower = text.lower()
    length_bin = "short" if len(text) < 500 else "med" if len(text) < 1600 else "long"
    has_code = "code" if ("```" in text or "def " in lower or "class " in lower) else "nocode"
    has_math = "math" if any(sym in text for sym in ["∑", "√", "≤", "≥", "≈", "∫", "$", "\\frac", "^2"]) else "nomath"
    style = "mcq" if re.search(r"(?:^|\n)\s*\(?[A-Ja-j]\)?[\).:]", text) else "open"
    return "|".join([str(input.get("benchmark", "")), str(input.get("condition", "")), length_bin, has_code, has_math, style])


def _base_predict(row: dict) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {{}}).get(_key(subject_name, benchmark, condition))
    if sbc is not None:
        return _clip(0.95 * float(sbc) + 0.05 * global_mean)

    sb = ARTIFACT.get("subject_benchmark", {{}}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _clip(0.92 * float(sb) + 0.08 * global_mean)

    bc = ARTIFACT.get("benchmark_condition", {{}}).get(_key(benchmark, condition))
    s = ARTIFACT.get("subject", {{}}).get(subject_name)
    b = ARTIFACT.get("benchmark", {{}}).get(benchmark)

    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _clip(0.85 * pred + 0.15 * global_mean)


def acquisition_function(input: dict) -> float:
    try:
        p = _base_predict(input)
        uncertainty = max(0.0, 1.0 - 2.0 * abs(p - 0.5))
        b = _bucket(input)
        _SEEN_BUCKETS[b] += 1
        diversity = 1.0 / float(_SEEN_BUCKETS[b])
        return float(0.82 * uncertainty + 0.18 * diversity + _hash_noise(input))
    except Exception:
        return float(_hash_noise(input, scale=1.0))
"""


VARIANTS = {
    "baseline_category_offset_tailclip0298": dict(
        model=dict(tail_low=0.02, tail_high=0.98, length_weight=0.0, complexity_weight=0.0),
        labeling=LABELING_HASH,
    ),
    "baseline_category_offset_tailclip0595": dict(
        model=dict(tail_low=0.05, tail_high=0.95, length_weight=0.0, complexity_weight=0.0),
        labeling=LABELING_HASH,
    ),
    "baseline_category_offset_item_length_tiny": dict(
        model=dict(tail_low=0.0001, tail_high=0.9999, length_weight=-0.060, complexity_weight=0.0),
        labeling=LABELING_HASH,
    ),
    "baseline_category_offset_item_complexity_tiny": dict(
        model=dict(tail_low=0.0001, tail_high=0.9999, length_weight=0.0, complexity_weight=-0.035),
        labeling=LABELING_HASH,
    ),
    "baseline_category_offset_uncertainty_acq": dict(
        model=dict(tail_low=0.0001, tail_high=0.9999, length_weight=0.0, complexity_weight=0.0),
        labeling=LABELING_UNCERTAINTY,
    ),
    "baseline_category_offset_uncertainty_diverse_acq": dict(
        model=dict(tail_low=0.0001, tail_high=0.9999, length_weight=0.0, complexity_weight=0.0),
        labeling=LABELING_UNCERTAINTY_DIVERSE,
    ),
}


def _find_project_dir() -> Path:
    here = Path.cwd().resolve()
    candidates = [
        here,
        here / "projects" / "predictive_eval_challenge",
        Path(__file__).resolve().parent,
    ]
    for p in candidates:
        if (p / "codabench_submissions" / "baseline" / "artifacts" / "smoothed_prior.json").exists():
            return p
    raise FileNotFoundError(
        "Could not find projects/predictive_eval_challenge with "
        "codabench_submissions/baseline/artifacts/smoothed_prior.json. "
        "Run this from the repo root or place it inside projects/predictive_eval_challenge."
    )


def _write_zip(src_dir: Path, out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir))


def main() -> None:
    project = _find_project_dir()
    sub_root = project / "codabench_submissions"
    dist = project / "dist" / "wave1_variants"
    base_artifact = sub_root / "baseline" / "artifacts" / "smoothed_prior.json"
    dist.mkdir(parents=True, exist_ok=True)

    print("Creating Wave-1 variants around baseline_category_offset...\n")
    for name, cfg in VARIANTS.items():
        dst = sub_root / name
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "artifacts").mkdir(parents=True, exist_ok=True)

        model_code = MODEL_TEMPLATE.format(**cfg["model"])
        (dst / "model.py").write_text(textwrap.dedent(model_code).lstrip(), encoding="utf-8")
        (dst / "labeling.py").write_text(textwrap.dedent(cfg["labeling"]).lstrip(), encoding="utf-8")
        (dst / "models.txt").write_text("", encoding="utf-8")
        shutil.copyfile(base_artifact, dst / "artifacts" / "smoothed_prior.json")

        zip_path = dist / f"{name}_submission.zip"
        _write_zip(dst, zip_path)
        print(f"READY: {zip_path}")

    print("\nUpload these 6 first, after 1–2 control reruns of baseline_category_offset_submission.zip:")
    order = [
        "baseline_category_offset_tailclip0298_submission.zip",
        "baseline_category_offset_tailclip0595_submission.zip",
        "baseline_category_offset_item_length_tiny_submission.zip",
        "baseline_category_offset_item_complexity_tiny_submission.zip",
        "baseline_category_offset_uncertainty_acq_submission.zip",
        "baseline_category_offset_uncertainty_diverse_acq_submission.zip",
    ]
    for i, filename in enumerate(order, start=1):
        print(f"{i}. {filename}")

    print("\nDecision rule after these finish:")
    print("- If any new variant gets -0.59 or -0.58, rerun that new winner.")
    print("- If one ties -0.60, split remaining submissions between that and baseline_category_offset.")
    print("- If all are <= -0.62, use all remaining submissions on baseline_category_offset_submission.zip.")


if __name__ == "__main__":
    main()
