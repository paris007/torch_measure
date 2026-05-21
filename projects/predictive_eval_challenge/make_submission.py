# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Build flat Codabench submission ZIP files."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = PROJECT_DIR / "codabench_submissions"


def make_submission(
    name: str,
    out_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Zip a Codabench submission directory with `model.py` at the archive root."""
    submission_dir = SUBMISSIONS_DIR / name
    if not submission_dir.exists():
        raise FileNotFoundError(f"Unknown submission directory: {submission_dir}")
    if not (submission_dir / "model.py").exists():
        raise FileNotFoundError(f"Submission is missing model.py: {submission_dir}")

    out_dir = Path(out_dir) if out_dir is not None else PROJECT_DIR / "dist"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{name}_submission.zip"
    if archive_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing submission: {archive_path}. "
            "Use --force only if you intentionally want to replace it."
        )

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if path.is_dir():
                continue
            if (
                "__pycache__" in path.parts
                or path.name.endswith(".pyc")
                or path.name == ".gitkeep"
                or path.name == ".DS_Store"
            ):
                continue
            archive.write(path, path.relative_to(submission_dir))

    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name",
        choices=[
            "baseline",
            "embedding",
            "factor_pge",
            "factor_baseline_ensemble",
            "robust_ensemble",
            "robust_hash",
            "robust_offset",
            "robust_diverse",
            "robust_multiseed",
            "robust_mpnet",
            "robust_prior",
            "robust_prior_v1",
            "robust_prior_v2",
            "robust_prior_clip",
            "robust_direct_residual",
            "robust_direct_residual_w15",
            "robust_direct_residual_w55",
            "factor_pge_multiseed",
        ],
    )
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "dist"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing dist/<name>_submission.zip.",
    )
    args = parser.parse_args()

    archive_path = make_submission(args.name, args.out_dir, force=args.force)
    print(f"wrote {archive_path}")


if __name__ == "__main__":
    main()
