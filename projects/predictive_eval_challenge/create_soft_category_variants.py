#!/usr/bin/env python3
"""Create conservative variants around the best category-offset submission.

The current public winner for this repo is baseline_category_offset at -0.60.
These variants keep that model intact and only test two calibration ideas:

1. Soften final probabilities slightly toward 0.5 to reduce hidden log-loss
   damage from overconfidence.
2. Remove benchmark-condition adaptive offsets and rely on category/global
   offsets only, since category-level calibration was the first clear win.

All ZIPs are written directly to projects/predictive_eval_challenge/dist/.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


VARIANTS = [
    {
        "name": "baseline_category_offset_soften002",
        "final_shrink": 0.02,
        "offset_clip": None,
        "weights": None,
    },
    {
        "name": "baseline_category_offset_soften004",
        "final_shrink": 0.04,
        "offset_clip": None,
        "weights": None,
    },
    {
        "name": "baseline_category_offset_soften006",
        "final_shrink": 0.06,
        "offset_clip": None,
        "weights": None,
    },
    {
        "name": "baseline_category_only_clip003",
        "final_shrink": 0.00,
        "offset_clip": 0.03,
        "weights": (0.25, 0.75, 0.00),
    },
    {
        "name": "baseline_category_only_clip004",
        "final_shrink": 0.00,
        "offset_clip": 0.04,
        "weights": (0.25, 0.75, 0.00),
    },
]


def zipdir(src: Path, out: Path) -> None:
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src))


def patch_model(source: str, variant: dict) -> str:
    model = source
    model = model.replace(
        "EPS = 1e-4\n",
        f"EPS = 1e-4\nFINAL_SHRINK_TO_HALF = {variant['final_shrink']!r}\n",
        1,
    )
    model = model.replace(
        "def _load_artifact() -> dict:\n",
        "\n"
        "def _final_calibrate(p: float) -> float:\n"
        "    if FINAL_SHRINK_TO_HALF <= 0.0:\n"
        "        return _clip_probability(p)\n"
        "    return _clip_probability((1.0 - FINAL_SHRINK_TO_HALF) * p + FINAL_SHRINK_TO_HALF * 0.5)\n\n\n"
        "def _load_artifact() -> dict:\n",
        1,
    )

    if variant["weights"] is not None:
        w_global, w_category, w_benchmark = variant["weights"]
        old = (
            "    offset = 0.50 * category_offset + 0.25 * benchmark_offset + 0.25 * global_offset\n"
            "    return float(min(0.05, max(-0.05, offset)))\n"
        )
        new = (
            f"    offset = {w_category:.2f} * category_offset + "
            f"{w_benchmark:.2f} * benchmark_offset + {w_global:.2f} * global_offset\n"
            f"    return float(min({variant['offset_clip']!r}, max(-{variant['offset_clip']!r}, offset)))\n"
        )
        if old not in model:
            raise RuntimeError("Could not patch adaptive offset weights")
        model = model.replace(old, new, 1)

    old_predict = (
        "        return _clip_probability(_base_predict(input) + _calibration_offset(input, _ROUND_OFFSETS))\n"
    )
    new_predict = (
        "        p = _clip_probability(_base_predict(input) + _calibration_offset(input, _ROUND_OFFSETS))\n"
        "        return _final_calibrate(p)\n"
    )
    if old_predict not in model:
        raise RuntimeError("Could not patch final prediction calibration")
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
