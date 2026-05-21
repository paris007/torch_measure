#!/usr/bin/env python
# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Offline simulator for Codabench adaptive labeling.

This script treats public rows as fake hidden categories:

1. score every candidate row with a submission's `acquisition_function`
2. reveal the top K labels within each `(benchmark, condition)` category
3. score the remaining rows through `predict(input, labeled=revealed)`

It is intentionally a fast diagnostic, not a perfect leaderboard replica. The
main use is comparing acquisition and adaptive-calibration behavior without
waiting on Codabench.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = PROJECT_DIR / "codabench_submissions"
DEFAULT_DATA = PROJECT_DIR / "data" / "runtime_examples.parquet"
EPS = 1e-4


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _clear_submission_imports() -> None:
    """Avoid cross-submission reuse of helper modules like acquisition_util."""
    for module_name in [
        "acquisition_util",
        "model",
        "labeling",
    ]:
        sys.modules.pop(module_name, None)


def _row_category(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("benchmark", "")), str(row.get("condition", "none") or "none"))


def _clip(p: float) -> float:
    if not math.isfinite(p):
        return 0.5
    return min(1.0 - EPS, max(EPS, float(p)))


def log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    y = y.astype(float)
    return float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def load_submission(name: str) -> tuple[ModuleType, ModuleType | None]:
    submission_dir = SUBMISSIONS_DIR / name
    if not submission_dir.exists():
        raise FileNotFoundError(f"Unknown submission: {submission_dir}")
    _clear_submission_imports()
    model = _load_module(submission_dir / "model.py", f"_sim_{name}_model")
    _clear_submission_imports()
    labeling_path = submission_dir / "labeling.py"
    labeling = (
        _load_module(labeling_path, f"_sim_{name}_labeling")
        if labeling_path.exists()
        else None
    )
    return model, labeling


def candidate_score(row: dict[str, Any], labeling: ModuleType | None) -> float:
    if labeling is None:
        return 0.0
    try:
        score = float(labeling.acquisition_function(row))
        return score if math.isfinite(score) else 0.0
    except Exception:
        return 0.0


def make_eval_frame(
    df: pd.DataFrame,
    *,
    max_categories: int | None,
    max_rows_per_category: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["condition"] = df["condition"].fillna("none").astype(str).replace("", "none")
    df["benchmark"] = df["benchmark"].astype(str)
    categories = sorted(df[["benchmark", "condition"]].drop_duplicates().itertuples(index=False, name=None))
    if max_categories is not None and len(categories) > max_categories:
        idx = rng.choice(len(categories), size=max_categories, replace=False)
        keep = {categories[i] for i in idx}
        df = df[df.apply(lambda r: (str(r["benchmark"]), str(r["condition"])) in keep, axis=1)]

    chunks: list[pd.DataFrame] = []
    for _, group in df.groupby(["benchmark", "condition"], sort=True):
        if len(group) > max_rows_per_category:
            group = group.sample(n=max_rows_per_category, random_state=seed)
        chunks.append(group)
    if not chunks:
        raise RuntimeError("No rows left after sampling.")
    return pd.concat(chunks, ignore_index=True)


def simulate_submission(
    df: pd.DataFrame,
    submission_name: str,
    *,
    k_labels: int,
    max_eval_per_category: int,
    seed: int,
) -> dict[str, Any]:
    model, labeling = load_submission(submission_name)
    rows = df.to_dict("records")
    by_cat: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cat.setdefault(_row_category(row), []).append(row)

    rng = np.random.default_rng(seed)
    per_category: list[dict[str, Any]] = []
    y_all: list[float] = []
    p_all: list[float] = []
    revealed_count = 0
    eval_count = 0

    for cat, cat_rows in sorted(by_cat.items()):
        scored = [(candidate_score(row, labeling), i, row) for i, row in enumerate(cat_rows)]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        labeled = [dict(row) for _, _, row in scored[:k_labels]]
        labeled_ids = {id(row) for _, _, row in scored[:k_labels]}
        eval_rows = [row for row in cat_rows if id(row) not in labeled_ids]
        if max_eval_per_category and len(eval_rows) > max_eval_per_category:
            eval_rows = [eval_rows[i] for i in rng.choice(len(eval_rows), size=max_eval_per_category, replace=False)]

        y_cat: list[float] = []
        p_cat: list[float] = []
        for row in eval_rows:
            try:
                p = _clip(float(model.predict(row, labeled=labeled)))
            except Exception:
                p = 0.5
            y = float(row.get("label", 0.0) or 0.0)
            y_cat.append(y)
            p_cat.append(p)

        if y_cat:
            y_arr = np.asarray(y_cat, dtype=float)
            p_arr = np.asarray(p_cat, dtype=float)
            per_category.append(
                {
                    "benchmark": cat[0],
                    "condition": cat[1],
                    "n_labeled": len(labeled),
                    "n_eval": len(y_cat),
                    "log_likelihood": log_likelihood(y_arr, p_arr),
                    "mean_label": float(y_arr.mean()),
                    "mean_pred": float(p_arr.mean()),
                }
            )
            y_all.extend(y_cat)
            p_all.extend(p_cat)
            revealed_count += len(labeled)
            eval_count += len(y_cat)

    y = np.asarray(y_all, dtype=float)
    p = np.asarray(p_all, dtype=float)
    return {
        "submission": submission_name,
        "categories": len(per_category),
        "n_labeled": revealed_count,
        "n_eval": eval_count,
        "log_likelihood": log_likelihood(y, p),
        "per_category": per_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to runtime_examples.parquet.",
    )
    parser.add_argument(
        "--submissions",
        nargs="+",
        default=["robust_prior", "robust_prior_v1", "robust_prior_v2", "robust_direct_residual_w15"],
    )
    parser.add_argument("--k-labels", type=int, default=5)
    parser.add_argument("--max-categories", type=int, default=40)
    parser.add_argument("--max-rows-per-category", type=int, default=2000)
    parser.add_argument("--max-eval-per-category", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Missing {args.data}. Run `python download_data.py --out {DEFAULT_DATA}` first."
        )

    df = pd.read_parquet(args.data)
    required = {"benchmark", "condition", "subject_content", "item_content", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Data file is missing columns: {sorted(missing)}")

    df = make_eval_frame(
        df,
        max_categories=args.max_categories,
        max_rows_per_category=args.max_rows_per_category,
        seed=args.seed,
    )
    print(
        f"[sim] rows={len(df):,} categories={df.groupby(['benchmark', 'condition']).ngroups:,} "
        f"k={args.k_labels}",
        flush=True,
    )

    results = [
        simulate_submission(
            df,
            name,
            k_labels=args.k_labels,
            max_eval_per_category=args.max_eval_per_category,
            seed=args.seed,
        )
        for name in args.submissions
    ]

    print()
    print(f"{'submission':32s} {'cats':>5s} {'labels':>7s} {'eval':>8s} {'loglik':>10s}")
    for result in results:
        print(
            f"{result['submission']:32s} {result['categories']:5d} "
            f"{result['n_labeled']:7d} {result['n_eval']:8d} "
            f"{result['log_likelihood']:10.5f}"
        )

    if args.details:
        for result in results:
            print(f"\n[{result['submission']}] weakest categories")
            rows = sorted(result["per_category"], key=lambda r: r["log_likelihood"])[:12]
            for row in rows:
                print(
                    f"  {row['benchmark']:18s} {row['condition']:18s} "
                    f"n={row['n_eval']:5d} loglik={row['log_likelihood']:8.5f} "
                    f"label={row['mean_label']:.3f} pred={row['mean_pred']:.3f}"
                )


if __name__ == "__main__":
    main()
