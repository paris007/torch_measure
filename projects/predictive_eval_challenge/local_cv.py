# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Whole-benchmark cold-start cross-validation across all three submissions.

Codabench grades each submission's `predict()` on hidden cold-start items, so
local in-sample numbers are misleading. This script:

  1. Holds out a configurable fraction of benchmarks per fold.
  2. Refits the smoothed-prior baseline, the embedding head, and the 2-PL
     factor model + item-param MLP from scratch on the fold's training rows.
  3. Scores each refitted model on the held-out rows (true cold-start items).
  4. Also evaluates the logit-mean ensemble of the three models.

Output: a tidy table of NLL / AUC per (model, fold), printed to stdout and
optionally written to JSON.

Encoder calls are the bottleneck; we encode every unique item exactly once
across folds and cache the result to a `.npz` file keyed by encoder_id + the
sha1 of the unique-item list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from torch_measure.models import (
    SmoothedPriorPredictiveEvaluator,
    format_item_text,
    parse_subject_name,
)

from train_factor import (
    ItemParamMLP,
    fit_factor_model,
    fit_item_param_mlp,
)
from train_embedding import build_features, compute_subject_embeddings
try:
    from data_loading import item_variant_key
except ImportError:  # pragma: no cover - allows package-style imports
    from projects.predictive_eval_challenge.data_loading import item_variant_key


ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128
EPS = 1e-4


def negative_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred.astype(float), EPS, 1.0 - EPS)
    y_true = y_true.astype(float)
    return float(np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


def auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def encode_all_unique_items(
    df: pd.DataFrame,
    encoder_id: str,
    batch_size: int,
    device: str | None,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode every unique truncated item once, with disk cache."""
    unique = (
        df[["item_content", "benchmark", "condition"]].copy()
    )
    unique["item_key"] = unique.apply(item_variant_key, axis=1)
    unique["item_content"] = unique["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    unique = unique.drop_duplicates("item_key").reset_index(drop=True)
    items_list = unique["item_key"].astype(str).tolist()
    fingerprint_input = encoder_id + "::" + str(len(items_list)) + "::" + "||".join(items_list[:20])
    fingerprint = hashlib.sha1(fingerprint_input.encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"item_embeddings_{fingerprint}.npz"

    if cache_path.exists():
        print(f"[cv] reusing cached embeddings: {cache_path}", flush=True)
        cached = np.load(cache_path, allow_pickle=False)
        embeddings = cached["embeddings"]
        keys = cached["item_keys"].tolist()
        item_to_row = {str(k): i for i, k in enumerate(keys)}
        return embeddings, item_to_row

    from sentence_transformers import SentenceTransformer

    texts = [format_item_text(row) for row in unique.to_dict("records")]
    encoder = (
        SentenceTransformer(encoder_id, device=device)
        if device
        else SentenceTransformer(encoder_id)
    )
    encoder.max_seq_length = MAX_SEQ_LENGTH
    t0 = time.time()
    embeddings = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    print(f"[cv] encoded {len(texts)} unique items in {time.time() - t0:.1f}s", flush=True)
    item_to_row = {str(k): i for i, k in enumerate(items_list)}
    np.savez(
        cache_path,
        embeddings=embeddings,
        item_keys=np.array(items_list),
    )
    return embeddings, item_to_row


def benchmark_folds(df: pd.DataFrame, n_folds: int, seed: int):
    """Yield whole-benchmark train/validation splits.

    n_folds=1 means a single 80/20 split. n_folds>=2 means n_folds disjoint
    holdouts (1/n_folds of benchmarks per fold).
    """
    rng = np.random.default_rng(seed)
    benchmarks = np.array(sorted(df["benchmark"].dropna().astype(str).unique()))
    rng.shuffle(benchmarks)
    if n_folds <= 1:
        n_valid = max(1, int(round(len(benchmarks) * 0.2)))
        valid = set(benchmarks[:n_valid].tolist())
        valid_mask = df["benchmark"].astype(str).isin(valid)
        yield "fold_1", df.loc[~valid_mask].copy(), df.loc[valid_mask].copy(), sorted(valid)
        return
    splits = np.array_split(benchmarks, n_folds)
    for idx, fold_benchmarks in enumerate(splits, start=1):
        valid = set(fold_benchmarks.tolist())
        valid_mask = df["benchmark"].astype(str).isin(valid)
        yield f"fold_{idx}", df.loc[~valid_mask].copy(), df.loc[valid_mask].copy(), sorted(valid)


def score_baseline(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> np.ndarray:
    """Smoothed-prior baseline: refit on train rows, predict on valid rows."""
    evaluator = SmoothedPriorPredictiveEvaluator.fit(train_df.to_dict("records"))
    return np.array(
        [evaluator.predict(row) for row in valid_df.to_dict("records")],
        dtype=float,
    )


def score_embedding(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    item_to_row: dict[str, int],
    with_subject_embed: bool,
) -> np.ndarray:
    """Embedding head: refit logistic regression on train rows, score valid rows."""
    baseline = SmoothedPriorPredictiveEvaluator.fit(train_df.to_dict("records"))
    subject_embeddings = None
    global_subject_embedding = None
    if with_subject_embed:
        subject_embeddings, global_subject_embedding = compute_subject_embeddings(
            train_df, item_embeddings, item_to_row
        )
    train_features = build_features(
        train_df,
        item_embeddings,
        item_to_row,
        subject_means=baseline.subject,
        global_mean=baseline.global_mean,
        subject_embeddings=subject_embeddings,
        global_subject_embedding=global_subject_embedding,
    )
    valid_features = build_features(
        valid_df,
        item_embeddings,
        item_to_row,
        subject_means=baseline.subject,
        global_mean=baseline.global_mean,
        subject_embeddings=subject_embeddings,
        global_subject_embedding=global_subject_embedding,
    )
    train_labels = train_df["label"].to_numpy(dtype=np.int64)

    scaler = StandardScaler().fit(train_features)
    scaled_train = scaler.transform(train_features)
    scaled_valid = scaler.transform(valid_features)
    clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1).fit(scaled_train, train_labels)
    return clf.predict_proba(scaled_valid)[:, 1]


def score_factor(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    item_to_row: dict[str, int],
    factor_epochs: int,
    mlp_hidden: int,
    mlp_epochs: int,
    seed: int,
) -> np.ndarray:
    """2-PL factor + item-param MLP fit on train rows; score valid rows."""
    train = train_df.copy()
    train["subject_name"] = train["subject_content"].astype(str).map(parse_subject_name)
    train["item_key"] = train.apply(item_variant_key, axis=1)

    subjects = sorted(train["subject_name"].unique().tolist())
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    items = sorted(train["item_key"].unique().tolist())
    item_idx = {k: i for i, k in enumerate(items)}

    subj_ids = train["subject_name"].map(subj_to_idx).to_numpy(np.int64)
    it_ids = train["item_key"].map(item_idx).to_numpy(np.int64)
    labels = train["label"].to_numpy(np.float32)

    theta, a_arr, b_arr = fit_factor_model(
        subject_ids=subj_ids,
        item_ids=it_ids,
        labels=labels,
        n_subjects=len(subjects),
        n_items=len(items),
        n_epochs=factor_epochs,
        seed=seed,
    )

    embed_dim = item_embeddings.shape[1]
    aligned_train_embeds = np.zeros((len(items), embed_dim), dtype=np.float32)
    for key, idx in item_idx.items():
        row = item_to_row.get(key)
        if row is not None:
            aligned_train_embeds[idx] = item_embeddings[row]

    mlp = fit_item_param_mlp(
        item_embeddings=aligned_train_embeds,
        a=a_arr,
        b=b_arr,
        hidden=mlp_hidden,
        n_epochs=mlp_epochs,
        seed=seed,
    )

    global_theta = float(theta.mean())
    subject_theta = {name: float(theta[i]) for name, i in subj_to_idx.items()}

    valid = valid_df.copy()
    valid["item_key"] = valid.apply(item_variant_key, axis=1)
    valid["subject_name"] = valid["subject_content"].astype(str).map(parse_subject_name)

    valid_rows = np.array(
        [item_to_row.get(k, -1) for k in valid["item_key"].astype(str).values],
        dtype=np.int64,
    )
    fallback_embed = item_embeddings.mean(axis=0)
    valid_embeds = np.where(
        (valid_rows >= 0)[:, None],
        item_embeddings[np.clip(valid_rows, 0, len(item_embeddings) - 1)],
        fallback_embed[None, :],
    )

    with torch.no_grad():
        x = torch.as_tensor(valid_embeds, dtype=torch.float32)
        out = mlp(x).cpu().numpy()
    log_a_pred = out[:, 0]
    b_pred = out[:, 1]
    a_pred = np.exp(np.clip(log_a_pred, -8, 8))

    theta_lookup = np.array(
        [subject_theta.get(name, global_theta) for name in valid["subject_name"].values],
        dtype=np.float32,
    )
    logits = a_pred * theta_lookup - b_pred
    return 1.0 / (1.0 + np.exp(-logits))


def run_cv(
    df: pd.DataFrame,
    encoder_id: str,
    batch_size: int,
    device: str | None,
    cache_dir: Path,
    n_folds: int,
    seed: int,
    factor_epochs: int,
    mlp_hidden: int,
    mlp_epochs: int,
    with_subject_embed: bool,
) -> list[dict]:
    item_embeddings, item_to_row = encode_all_unique_items(
        df, encoder_id, batch_size, device, cache_dir
    )
    results: list[dict] = []
    for fold_name, train_df, valid_df, valid_benchmarks in benchmark_folds(df, n_folds, seed):
        labels = valid_df["label"].to_numpy(dtype=float)
        t0 = time.time()
        print(
            f"\n=== {fold_name}: train={len(train_df):,}  valid={len(valid_df):,}  "
            f"held-out benchmarks={valid_benchmarks}",
            flush=True,
        )
        preds = {}

        t = time.time()
        preds["baseline"] = score_baseline(train_df, valid_df)
        print(f"  [baseline]  {time.time() - t:.1f}s", flush=True)

        t = time.time()
        preds["embedding"] = score_embedding(
            train_df, valid_df, item_embeddings, item_to_row, with_subject_embed
        )
        print(f"  [embedding] {time.time() - t:.1f}s", flush=True)

        t = time.time()
        preds["factor"] = score_factor(
            train_df,
            valid_df,
            item_embeddings,
            item_to_row,
            factor_epochs=factor_epochs,
            mlp_hidden=mlp_hidden,
            mlp_epochs=mlp_epochs,
            seed=seed,
        )
        print(f"  [factor]    {time.time() - t:.1f}s", flush=True)

        # Logit-mean ensemble of all three.
        logits = np.stack(
            [_logit(preds[m]) for m in ("baseline", "embedding", "factor")],
            axis=0,
        ).mean(axis=0)
        preds["ensemble"] = 1.0 / (1.0 + np.exp(-logits))

        for model_name, p in preds.items():
            nll = negative_log_loss(labels, p)
            roc = auc(labels, p)
            results.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "n_valid": int(len(valid_df)),
                    "neg_log_loss": nll,
                    "auc": roc,
                }
            )
            print(f"  -> {model_name:10s} NLL={nll:.5f}  AUC={roc:.5f}", flush=True)
        print(f"  [fold total] {time.time() - t0:.1f}s", flush=True)
    return results


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def summarize(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        return
    print("\n=== Summary (mean across folds) ===")
    summary = (
        df.groupby("model")
        .agg(neg_log_loss=("neg_log_loss", "mean"), auc=("auc", "mean"))
        .sort_values("neg_log_loss", ascending=False)
    )
    print(summary.to_string(float_format=lambda v: f"{v:.5f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--encoder-id", default=ENCODER_ID)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="projects/predictive_eval_challenge/data/cache")
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--max-rows", type=int, default=300_000)
    parser.add_argument("--factor-epochs", type=int, default=5)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--mlp-epochs", type=int, default=300)
    parser.add_argument(
        "--with-subject-embed",
        action="store_true",
        help="Include per-subject mean embedding in the embedding-head features.",
    )
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    print(f"[cv] loaded {len(df):,} rows", flush=True)

    results = run_cv(
        df,
        encoder_id=args.encoder_id,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=Path(args.cache_dir),
        n_folds=args.n_folds,
        seed=args.seed,
        factor_epochs=args.factor_epochs,
        mlp_hidden=args.mlp_hidden,
        mlp_epochs=args.mlp_epochs,
        with_subject_embed=args.with_subject_embed,
    )
    summarize(results)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
