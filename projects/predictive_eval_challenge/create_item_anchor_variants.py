#!/usr/bin/env python3
"""Create item-anchor adaptive variants around baseline_category_offset.

The predictor keeps the current winning category-offset backbone, then applies
an exact item-variant correction only when the revealed labels include that
same benchmark/condition/item text. The acquisition function hashes mostly the
item variant, encouraging Codabench to reveal multiple subjects for the same
hidden item when possible.

All ZIPs are written directly to projects/predictive_eval_challenge/dist/.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


VARIANTS = [
    {
        "name": "baseline_item_anchor_prob_tiny",
        "mode": "prob",
        "item_clip": 0.035,
        "item_shrink_n": 4.0,
        "salt": "item_anchor_prob_tiny_parism",
    },
    {
        "name": "baseline_item_anchor_prob_medium",
        "mode": "prob",
        "item_clip": 0.060,
        "item_shrink_n": 3.0,
        "salt": "item_anchor_prob_medium_parism",
    },
    {
        "name": "baseline_item_anchor_logit_tiny",
        "mode": "logit",
        "item_clip": 0.18,
        "item_shrink_n": 5.0,
        "salt": "item_anchor_logit_tiny_parism",
    },
    {
        "name": "baseline_item_anchor_logit_medium",
        "mode": "logit",
        "item_clip": 0.28,
        "item_shrink_n": 4.0,
        "salt": "item_anchor_logit_medium_parism",
    },
]


LABELING_TEMPLATE = '''\
"""Item-anchor acquisition for adaptive labels."""

from __future__ import annotations

import hashlib

SALT = "{salt}"


def acquisition_function(input: dict) -> float:
    text = "\\n".join(
        [
            SALT,
            str(input.get("benchmark", "")),
            str(input.get("condition", "none") or "none"),
            str(input.get("item_content", ""))[:1600],
        ]
    )
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    return float(int(digest[:12], 16) / 16**12)
'''


def zipdir(src: Path, out: Path) -> None:
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src))


def patch_model(source: str, variant: dict) -> str:
    mode = variant["mode"]
    if mode not in {"prob", "logit"}:
        raise ValueError(f"Unknown mode {mode}")

    model = source
    model = model.replace(
        "EPS = 1e-4\n",
        (
            "EPS = 1e-4\n"
            f"ITEM_MODE = {mode!r}\n"
            f"ITEM_CLIP = {variant['item_clip']!r}\n"
            f"ITEM_SHRINK_N = {variant['item_shrink_n']!r}\n"
        ),
        1,
    )
    model = model.replace(
        '_ROUND_OFFSETS = {"global": 0.0, "category": {}, "benchmark_condition": {}}\n',
        '_ROUND_OFFSETS = {"global": 0.0, "category": {}, "benchmark_condition": {}, "item": {}}\n',
        1,
    )
    model = model.replace(
        "def _parse_subject_name(subject_content: str) -> str:\n",
        (
            "def _logit(p: float) -> float:\n"
            "    p = _clip_probability(p)\n"
            "    return math.log(p / (1.0 - p))\n\n\n"
            "def _sigmoid(x: float) -> float:\n"
            "    if x >= 0:\n"
            "        z = math.exp(-x)\n"
            "        return 1.0 / (1.0 + z)\n"
            "    z = math.exp(x)\n"
            "    return z / (1.0 + z)\n\n\n"
            "def _parse_subject_name(subject_content: str) -> str:\n"
        ),
        1,
    )
    model = model.replace(
        "def _row_group(row: Mapping[str, object]) -> tuple[str, str]:\n",
        (
            "def _item_key(row: Mapping[str, object]) -> tuple[str, str, str]:\n"
            "    benchmark = str(row.get(\"benchmark\", \"\"))\n"
            "    condition = str(row.get(\"condition\", \"none\") or \"none\")\n"
            "    item = str(row.get(\"item_content\", \"\"))[:1600]\n"
            "    return (benchmark, condition, item)\n\n\n"
            "def _row_group(row: Mapping[str, object]) -> tuple[str, str]:\n"
        ),
        1,
    )
    model = model.replace(
        '        return {"global": 0.0, "category": {}, "benchmark_condition": {}}\n',
        '        return {"global": 0.0, "category": {}, "benchmark_condition": {}, "item": {}}\n',
        1,
    )
    model = model.replace(
        "    bc_residuals: dict[tuple[str, str], list[float]] = {}\n",
        (
            "    bc_residuals: dict[tuple[str, str], list[float]] = {}\n"
            "    item_prob_residuals: dict[tuple[str, str, str], list[float]] = {}\n"
            "    item_logit_numerators: dict[tuple[str, str, str], float] = {}\n"
            "    item_logit_denominators: dict[tuple[str, str, str], float] = {}\n"
        ),
        1,
    )
    model = model.replace(
        "            residual = float(row[\"label\"]) - _base_predict(row)\n",
        (
            "            base_p = _base_predict(row)\n"
            "            residual = float(row[\"label\"]) - base_p\n"
        ),
        1,
    )
    model = model.replace(
        "        bc_residuals.setdefault(_row_benchmark_condition(row), []).append(residual)\n",
        (
            "        bc_residuals.setdefault(_row_benchmark_condition(row), []).append(residual)\n"
            "        item = _item_key(row)\n"
            "        item_prob_residuals.setdefault(item, []).append(residual)\n"
            "        item_logit_numerators[item] = item_logit_numerators.get(item, 0.0) + residual\n"
            "        item_logit_denominators[item] = item_logit_denominators.get(item, 0.0) + base_p * (1.0 - base_p)\n"
        ),
        1,
    )
    model = model.replace(
        "        \"benchmark_condition\": {\n"
        "            key: _clipped_mean(values)\n"
        "            for key, values in bc_residuals.items()\n"
        "            if len(values) >= 2\n"
        "        },\n"
        "    }\n",
        (
            "        \"benchmark_condition\": {\n"
            "            key: _clipped_mean(values)\n"
            "            for key, values in bc_residuals.items()\n"
            "            if len(values) >= 2\n"
            "        },\n"
            "        \"item\": {\n"
            "            key: _item_delta(key, item_prob_residuals, item_logit_numerators, item_logit_denominators)\n"
            "            for key in item_prob_residuals\n"
            "        },\n"
            "    }\n"
        ),
        1,
    )
    model = model.replace(
        "def _calibration_offset(input: Mapping[str, object], offsets: dict) -> float:\n",
        (
            "def _item_delta(\n"
            "    key: tuple[str, str, str],\n"
            "    prob_residuals: dict[tuple[str, str, str], list[float]],\n"
            "    logit_numerators: dict[tuple[str, str, str], float],\n"
            "    logit_denominators: dict[tuple[str, str, str], float],\n"
            ") -> float:\n"
            "    values = prob_residuals.get(key, [])\n"
            "    if not values:\n"
            "        return 0.0\n"
            "    n = float(len(values))\n"
            "    if ITEM_MODE == \"logit\":\n"
            "        raw = logit_numerators.get(key, 0.0) / (ITEM_SHRINK_N + logit_denominators.get(key, 0.0))\n"
            "    else:\n"
            "        raw = (sum(values) / n) * (n / (n + ITEM_SHRINK_N))\n"
            "    return float(min(ITEM_CLIP, max(-ITEM_CLIP, raw)))\n\n\n"
            "def _apply_item_delta(base: float, input: Mapping[str, object], offsets: dict) -> float:\n"
            "    delta = float(offsets.get(\"item\", {}).get(_item_key(input), 0.0))\n"
            "    if delta == 0.0:\n"
            "        return _clip_probability(base)\n"
            "    if ITEM_MODE == \"logit\":\n"
            "        return _clip_probability(_sigmoid(_logit(base) + delta))\n"
            "    return _clip_probability(base + delta)\n\n\n"
            "def _calibration_offset(input: Mapping[str, object], offsets: dict) -> float:\n"
        ),
        1,
    )
    old_predict = (
        "        return _clip_probability(_base_predict(input) + _calibration_offset(input, _ROUND_OFFSETS))\n"
    )
    new_predict = (
        "        base = _clip_probability(_base_predict(input) + _calibration_offset(input, _ROUND_OFFSETS))\n"
        "        return _apply_item_delta(base, input, _ROUND_OFFSETS)\n"
    )
    if old_predict not in model:
        raise RuntimeError("Could not patch predict")
    return model.replace(old_predict, new_predict, 1)


def main() -> None:
    project = Path(__file__).resolve().parent
    source = project / "codabench_submissions" / "baseline_category_offset"
    dist = project / "dist"
    dist.mkdir(parents=True, exist_ok=True)

    source_model = (source / "model.py").read_text()
    ready: list[Path] = []
    for variant in VARIANTS:
        name = variant["name"]
        target = project / "codabench_submissions" / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
        (target / "model.py").write_text(patch_model(source_model, variant))
        (target / "labeling.py").write_text(LABELING_TEMPLATE.format(salt=variant["salt"]))

        out = dist / f"{name}_submission.zip"
        zipdir(target, out)
        ready.append(out)

    print("READY:")
    for out in ready:
        print(out)
    print("\nUpload in this order:")
    for idx, out in enumerate(ready, start=1):
        print(f"{idx}. {out.name}")


if __name__ == "__main__":
    main()
