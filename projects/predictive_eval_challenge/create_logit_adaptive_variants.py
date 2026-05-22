#!/usr/bin/env python3
"""Create adaptive logit-offset variants.

These variants keep the same smoothed-prior backbone as the winning
category-offset model, but estimate adaptive corrections on the logit/odds
scale using a one-step logistic intercept update:

    delta_group = sum(y - p) / (prior_precision + sum(p * (1 - p)))

Use:
  python projects/predictive_eval_challenge/create_logit_adaptive_variants.py
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


MODEL_TEMPLATE = r"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

EPS = 1e-4
ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "smoothed_prior.json"

LOGIT_CLIP = __LOGIT_CLIP__
GLOBAL_WEIGHT = __GLOBAL_WEIGHT__
CATEGORY_WEIGHT = __CATEGORY_WEIGHT__
BC_WEIGHT = __BC_WEIGHT__
SUBJECT_CATEGORY_WEIGHT = __SUBJECT_CATEGORY_WEIGHT__
GLOBAL_PRIOR = __GLOBAL_PRIOR__
CATEGORY_PRIOR = __CATEGORY_PRIOR__
BC_PRIOR = __BC_PRIOR__
SUBJECT_CATEGORY_PRIOR = __SUBJECT_CATEGORY_PRIOR__

CATEGORY_BY_BENCHMARK = {
    "swebench": "coding", "livecodebench": "coding", "bigcodebench": "coding",
    "humaneval": "coding", "mbpp": "coding",
    "bfcl": "tool_use", "agentdojo": "tool_use", "androidworld": "tool_use", "tau2": "tool_use",
    "matharena": "math", "mathvista_mini": "math_vision", "gsm8k": "math", "aime": "math",
    "ai2d_test": "vision", "mmbench_v11": "vision", "mmmu": "vision",
    "mmlupro": "knowledge", "hle": "knowledge", "mmlu": "knowledge", "gpqa": "knowledge",
    "rewardbench": "preference", "ultrafeedback": "preference", "mtbench": "chat",
    "afrimedqa": "medical", "medqa": "medical", "cybench": "cyber",
}


def _clip_probability(value: float, eps: float = EPS) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(1.0 - eps, max(eps, value)))


def _logit(p: float) -> float:
    p = _clip_probability(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _norm(s: object) -> str:
    return str(s or "").strip().lower()


def _parse_subject_name(subject_content: str) -> str:
    match = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower()
    return (subject_content or "").strip().lower()


def _key(*parts: object) -> str:
    return "||".join(str(part) for part in parts)


def _category(row: Mapping[str, object]) -> str:
    benchmark_raw = str(row.get("benchmark", ""))
    benchmark = _norm(benchmark_raw)
    return str(CATEGORY_BY_BENCHMARK.get(benchmark, benchmark_raw))


def _group_key(row: Mapping[str, object]) -> str:
    return _key(_category(row), str(row.get("condition", "none") or "none"))


def _bc_key(row: Mapping[str, object]) -> str:
    return _key(str(row.get("benchmark", "")), str(row.get("condition", "none") or "none"))


def _subject_category_key(row: Mapping[str, object]) -> str:
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    return _key(subject_name, _category(row), str(row.get("condition", "none") or "none"))


def _load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        return {"global_mean": 0.6528605818748474}
    return json.loads(ARTIFACT_PATH.read_text())


ARTIFACT = _load_artifact()
_ROUND_CACHE_KEY = None
_ROUND_OFFSETS = (0.0, {}, {}, {})


def _base_predict(row: Mapping[str, object]) -> float:
    global_mean = float(ARTIFACT.get("global_mean", 0.5))
    subject_name = _parse_subject_name(str(row.get("subject_content", "")))
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")

    sbc = ARTIFACT.get("subject_benchmark_condition", {}).get(_key(subject_name, benchmark, condition))
    if sbc is not None:
        return _clip_probability(0.95 * float(sbc) + 0.05 * global_mean)

    sb = ARTIFACT.get("subject_benchmark", {}).get(_key(subject_name, benchmark))
    if sb is not None:
        return _clip_probability(0.92 * float(sb) + 0.08 * global_mean)

    bc = ARTIFACT.get("benchmark_condition", {}).get(_key(benchmark, condition))
    s = ARTIFACT.get("subject", {}).get(subject_name)
    b = ARTIFACT.get("benchmark", {}).get(benchmark)

    pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
    avail = [(v, w) for v, w in pairs if v is not None]
    if not avail:
        return _clip_probability(global_mean)
    tw = sum(w for _, w in avail)
    pred = sum(float(v) * w for v, w in avail) / tw
    return _clip_probability(0.85 * pred + 0.15 * global_mean)


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
    except Exception:
        return ("__unsortable__", len(labeled))


def _newton_logit_offset(rows: list[dict], prior_precision: float) -> float:
    if not rows:
        return 0.0
    score = 0.0
    info = 0.0
    for row in rows:
        if "label" not in row:
            continue
        try:
            y = float(row["label"])
            p = _base_predict(row)
        except Exception:
            continue
        if not math.isfinite(y) or not math.isfinite(p):
            continue
        score += y - p
        info += p * (1.0 - p)
    if info <= 0.0:
        return 0.0
    delta = score / (float(prior_precision) + info)
    return float(min(LOGIT_CLIP, max(-LOGIT_CLIP, delta)))


def _fit_offsets(labeled: list[dict] | None):
    if not labeled:
        return (0.0, {}, {}, {})

    valid = [row for row in labeled if "label" in row]
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_bc: dict[str, list[dict]] = defaultdict(list)
    by_subject_category: dict[str, list[dict]] = defaultdict(list)

    for row in valid:
        by_group[_group_key(row)].append(row)
        by_bc[_bc_key(row)].append(row)
        by_subject_category[_subject_category_key(row)].append(row)

    global_offset = _newton_logit_offset(valid, GLOBAL_PRIOR)
    group_offsets = {k: _newton_logit_offset(v, CATEGORY_PRIOR) for k, v in by_group.items()}
    bc_offsets = {k: _newton_logit_offset(v, BC_PRIOR) for k, v in by_bc.items()}
    subject_category_offsets = {k: _newton_logit_offset(v, SUBJECT_CATEGORY_PRIOR) for k, v in by_subject_category.items()}
    return global_offset, group_offsets, bc_offsets, subject_category_offsets


def _adaptive_logit_offset(row: Mapping[str, object]) -> float:
    global_offset, group_offsets, bc_offsets, subject_category_offsets = _ROUND_OFFSETS
    offset = (
        GLOBAL_WEIGHT * float(global_offset)
        + CATEGORY_WEIGHT * float(group_offsets.get(_group_key(row), 0.0))
        + BC_WEIGHT * float(bc_offsets.get(_bc_key(row), 0.0))
        + SUBJECT_CATEGORY_WEIGHT * float(subject_category_offsets.get(_subject_category_key(row), 0.0))
    )
    return float(min(LOGIT_CLIP, max(-LOGIT_CLIP, offset)))


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    global _ROUND_CACHE_KEY, _ROUND_OFFSETS
    try:
        key = _labeled_key(labeled)
        if key != _ROUND_CACHE_KEY:
            _ROUND_CACHE_KEY = key
            _ROUND_OFFSETS = _fit_offsets(labeled)

        base = _base_predict(input)
        logit = _logit(base) + _adaptive_logit_offset(input)
        return _clip_probability(_sigmoid(logit))
    except Exception as exc:
        print(f"[logit_adaptive] predict fallback: {exc}", flush=True)
        return _clip_probability(float(ARTIFACT.get("global_mean", 0.5)))
"""


LABELING_TEMPLATE = r"""
from __future__ import annotations
import hashlib

SALT = "__SALT__"


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
"""


VARIANTS = {
    "baseline_category_logit_offset": {
        "LOGIT_CLIP": 0.30,
        "GLOBAL_WEIGHT": 0.25,
        "CATEGORY_WEIGHT": 0.55,
        "BC_WEIGHT": 0.20,
        "SUBJECT_CATEGORY_WEIGHT": 0.00,
        "GLOBAL_PRIOR": 5.0,
        "CATEGORY_PRIOR": 5.0,
        "BC_PRIOR": 7.0,
        "SUBJECT_CATEGORY_PRIOR": 20.0,
        "SALT": "category_logit_offset_parism",
    },
    "baseline_category_logit_subject_tiny": {
        "LOGIT_CLIP": 0.30,
        "GLOBAL_WEIGHT": 0.20,
        "CATEGORY_WEIGHT": 0.45,
        "BC_WEIGHT": 0.20,
        "SUBJECT_CATEGORY_WEIGHT": 0.15,
        "GLOBAL_PRIOR": 5.0,
        "CATEGORY_PRIOR": 5.0,
        "BC_PRIOR": 7.0,
        "SUBJECT_CATEGORY_PRIOR": 12.0,
        "SALT": "category_logit_subject_tiny_parism",
    },
    "baseline_category_logit_subject_medium": {
        "LOGIT_CLIP": 0.35,
        "GLOBAL_WEIGHT": 0.15,
        "CATEGORY_WEIGHT": 0.35,
        "BC_WEIGHT": 0.20,
        "SUBJECT_CATEGORY_WEIGHT": 0.30,
        "GLOBAL_PRIOR": 5.0,
        "CATEGORY_PRIOR": 5.0,
        "BC_PRIOR": 7.0,
        "SUBJECT_CATEGORY_PRIOR": 8.0,
        "SALT": "category_logit_subject_medium_parism",
    },
    "baseline_category_logit_subject_plus_global": {
        "LOGIT_CLIP": 0.30,
        "GLOBAL_WEIGHT": 0.35,
        "CATEGORY_WEIGHT": 0.35,
        "BC_WEIGHT": 0.15,
        "SUBJECT_CATEGORY_WEIGHT": 0.15,
        "GLOBAL_PRIOR": 4.0,
        "CATEGORY_PRIOR": 6.0,
        "BC_PRIOR": 8.0,
        "SUBJECT_CATEGORY_PRIOR": 12.0,
        "SALT": "category_logit_subject_plus_global_parism",
    },
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
        "codabench_submissions/baseline/artifacts/smoothed_prior.json."
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
    dist = project / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    artifact = sub_root / "baseline" / "artifacts" / "smoothed_prior.json"
    if not artifact.exists():
        raise FileNotFoundError(f"Missing {artifact}")

    print("Creating adaptive logit-offset variants...\n")
    for name, cfg in VARIANTS.items():
        dst = sub_root / name
        if dst.exists():
            shutil.rmtree(dst)
        (dst / "artifacts").mkdir(parents=True, exist_ok=True)

        model_code = MODEL_TEMPLATE
        for key, value in cfg.items():
            if key == "SALT":
                continue
            model_code = model_code.replace(f"__{key}__", repr(float(value)))
        label_code = LABELING_TEMPLATE.replace("__SALT__", str(cfg["SALT"]))

        (dst / "model.py").write_text(model_code.lstrip(), encoding="utf-8")
        (dst / "labeling.py").write_text(label_code.lstrip(), encoding="utf-8")
        (dst / "models.txt").write_text("", encoding="utf-8")
        shutil.copyfile(artifact, dst / "artifacts" / "smoothed_prior.json")

        zip_path = dist / f"{name}_submission.zip"
        _write_zip(dst, zip_path)
        print(f"READY: {zip_path}")

    print("\nUpload once each in this order:")
    for i, name in enumerate(VARIANTS, start=1):
        print(f"{i}. {name}_submission.zip")

    print("\nDecision rule:")
    print("- If any gets -0.59 or -0.58, rerun that exact zip.")
    print("- If all are worse than -0.60, stop and keep baseline_category_offset/salt10.")


if __name__ == "__main__":
    main()
