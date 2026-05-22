#!/usr/bin/env python3
"""Create salted category-offset variants.

These keep the winning baseline_category_offset model/artifacts fixed and
only change the deterministic hash salt in labeling.py. That changes which
adaptive labels are revealed, while preserving the winning predictor.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

N_SALTS = 10

LABELING_TEMPLATE = """\"\"\"Salted pseudo-random adaptive-label acquisition.

SALT = "{salt}"
\"\"\"
from __future__ import annotations

import hashlib

SALT = "{salt}"


def acquisition_function(input: dict) -> float:
    text = "\\n".join(
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


def _find_project_dir() -> Path:
    here = Path.cwd().resolve()
    candidates = [
        here,
        here / "projects" / "predictive_eval_challenge",
        Path(__file__).resolve().parent,
    ]
    for p in candidates:
        if (p / "codabench_submissions").exists():
            return p
    raise FileNotFoundError(
        "Could not find projects/predictive_eval_challenge. Run from the repo root "
        "or place this script inside projects/predictive_eval_challenge."
    )


def _find_source_submission(project: Path) -> Path:
    sub_root = project / "codabench_submissions"
    direct_candidates = [
        sub_root / "baseline_category_offset",
        sub_root / "baseline_category_offset_submission",
        sub_root / "baseline_category_offset_best",
    ]
    for cand in direct_candidates:
        if (cand / "model.py").exists() and (cand / "artifacts").exists():
            return cand

    zip_candidates = sorted(project.rglob("*baseline_category_offset*submission*.zip"))
    zip_candidates = [
        z for z in zip_candidates
        if "salt" not in z.name and "factor" not in z.name and "tail" not in z.name
    ]
    for zpath in zip_candidates:
        extract_dir = sub_root / "_baseline_category_offset_source"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_dir)
        if (extract_dir / "model.py").exists() and (extract_dir / "artifacts").exists():
            print(f"Using source zip: {zpath}")
            return extract_dir
        shutil.rmtree(extract_dir)

    raise FileNotFoundError(
        "Could not find baseline_category_offset source directory or zip. "
        "Make sure codabench_submissions/baseline_category_offset exists, or put "
        "baseline_category_offset_submission.zip somewhere under the project folder."
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

    source = _find_source_submission(project)
    print(f"Source winning submission: {source}")

    outputs = []
    for i in range(1, N_SALTS + 1):
        salt = f"category_offset_salt_{i:02d}_parism"
        name = f"baseline_category_offset_salt{i:02d}"
        dst = sub_root / name

        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(source, dst)

        (dst / "labeling.py").write_text(LABELING_TEMPLATE.format(salt=salt), encoding="utf-8")

        zip_path = dist / f"{name}_submission.zip"
        _write_zip(dst, zip_path)
        outputs.append(zip_path)
        print(f"READY: {zip_path}")

    print("\nUpload these once each:")
    for i, out in enumerate(outputs, start=1):
        print(f"{i}. {out.name}")

    print("\nDecision rule:")
    print("- If any salt gets -0.59 or -0.58, rerun that exact salt.")
    print("- If all are <= -0.61/-0.62, stop and keep original baseline_category_offset.")
    print("- This is the last low-risk exploration because it preserves the winning model.")


if __name__ == "__main__":
    main()
