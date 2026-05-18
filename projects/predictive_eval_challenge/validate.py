# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Validation utilities for the predictive evaluation challenge."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from torch_measure.models import SmoothedPriorPredictiveEvaluator


EPS = 1e-4


def negative_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean log-likelihood of labels under predicted probabilities."""
    y_pred = np.clip(y_pred.astype(float), EPS, 1.0 - EPS)
    y_true = y_true.astype(float)
    return float(np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """AUC-ROC, returning NaN for single-class validation folds."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def benchmark_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 321):
    """Yield whole-benchmark train/validation splits."""
    rng = np.random.default_rng(seed)
    benchmarks = np.array(sorted(df["benchmark"].dropna().unique()))
    rng.shuffle(benchmarks)
    for idx, fold_benchmarks in enumerate(np.array_split(benchmarks, n_folds), start=1):
        valid_mask = df["benchmark"].isin(set(fold_benchmarks.tolist()))
        yield f"fold_{idx}", df.loc[~valid_mask].copy(), df.loc[valid_mask].copy()


def validate_smoothed_prior(df: pd.DataFrame, n_folds: int = 5, seed: int = 321) -> list[dict[str, float]]:
    """Validate the smoothed-prior evaluator with whole-benchmark holdout."""
    results = []
    for name, train_df, valid_df in benchmark_folds(df, n_folds=n_folds, seed=seed):
        evaluator = SmoothedPriorPredictiveEvaluator.fit(train_df.to_dict("records"))
        preds = np.array([evaluator.predict(row) for row in valid_df.to_dict("records")], dtype=float)
        labels = valid_df["label"].to_numpy(dtype=float)
        result = {
            "fold": name,
            "n_train": float(len(train_df)),
            "n_valid": float(len(valid_df)),
            "neg_log_loss": negative_log_loss(labels, preds),
            "auc": auc(labels, preds),
        }
        results.append(result)
        print(
            f"{name}: n_train={len(train_df)} n_valid={len(valid_df)} "
            f"neg_log_loss={result['neg_log_loss']:.5f} auc={result['auc']:.5f}",
            flush=True,
        )
    return results


def print_summary(results: list[dict[str, float]]) -> None:
    scores = [result["neg_log_loss"] for result in results if math.isfinite(result["neg_log_loss"])]
    aucs = [result["auc"] for result in results if math.isfinite(result["auc"])]
    if scores:
        print(f"mean neg_log_loss={np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    if aucs:
        print(f"mean auc={np.mean(aucs):.5f} +/- {np.std(aucs):.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Joined runtime examples parquet.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row subsample for quick smoke tests.")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    results = validate_smoothed_prior(df, n_folds=args.folds, seed=args.seed)
    print_summary(results)


if __name__ == "__main__":
    main()
