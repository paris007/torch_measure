# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Train an embedding-based predictive evaluator.

The runtime path is small on purpose: at test time the Codabench wrapper only
needs to encode the hidden item text, append a smoothed subject-mean prior, and
apply a linear head. The encoder is declared in `models.txt` and pre-fetched by
the platform; everything else lives in a single `.npz` artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from torch_measure.models import (
    SmoothedPriorPredictiveEvaluator,
    format_item_text,
    parse_subject_name,
)


ENCODER_ID = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_ARTIFACT_PATH = (
    "projects/predictive_eval_challenge/"
    "codabench_submissions/embedding/artifacts/embedding_head.npz"
)


def encode_unique_items(
    df: pd.DataFrame,
    model_id: str,
    batch_size: int = 128,
    device: str | None = None,
    max_chars: int = 800,
    max_seq_length: int = 128,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode each unique item text exactly once and return embeddings plus index."""
    from sentence_transformers import SentenceTransformer

    unique = df[["item_content", "benchmark", "condition"]].drop_duplicates("item_content")
    # Truncate item content before formatting so a single long outlier does not
    # pad an entire batch to thousands of tokens.
    unique = unique.copy()
    unique["item_content"] = unique["item_content"].astype(str).str.slice(0, max_chars)
    texts = [format_item_text(row) for row in unique.to_dict("records")]
    encoder = SentenceTransformer(model_id, device=device) if device else SentenceTransformer(model_id)
    encoder.max_seq_length = max_seq_length
    embeddings = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    item_to_row = {
        str(item): idx for idx, item in enumerate(unique["item_content"].astype(str).values)
    }
    return embeddings, item_to_row


def build_features(
    df: pd.DataFrame,
    item_embeddings: np.ndarray,
    item_to_row: dict[str, int],
    subject_means: dict[str, float],
    global_mean: float,
    max_chars: int = 800,
    subject_embeddings: dict[str, np.ndarray] | None = None,
    global_subject_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Concatenate item embeddings with a subject-prior feature.

    When `subject_embeddings` is provided, also append a per-subject mean item
    embedding so the linear head can model item × subject interactions.
    """
    item_keys = df["item_content"].astype(str).str.slice(0, max_chars).values
    rows = np.array(
        [item_to_row[str(item)] for item in item_keys],
        dtype=np.int64,
    )
    embed_features = item_embeddings[rows]
    parsed_names = [parse_subject_name(str(value)) for value in df["subject_content"].astype(str).values]
    subject_prior = np.array(
        [subject_means.get(name, global_mean) for name in parsed_names],
        dtype=np.float32,
    )[:, None]
    blocks = [embed_features, subject_prior]
    if subject_embeddings is not None:
        fallback = (
            global_subject_embedding
            if global_subject_embedding is not None
            else np.zeros(item_embeddings.shape[1], dtype=np.float32)
        )
        subject_embed = np.stack(
            [subject_embeddings.get(name, fallback) for name in parsed_names],
            axis=0,
        ).astype(np.float32)
        blocks.append(subject_embed)
    return np.concatenate(blocks, axis=1)


def compute_subject_embeddings(
    df: pd.DataFrame,
    item_embeddings: np.ndarray,
    item_to_row: dict[str, int],
    max_chars: int = 800,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Mean item embedding per subject; also return the global mean embedding."""
    item_keys = df["item_content"].astype(str).str.slice(0, max_chars).values
    rows = np.array([item_to_row[str(item)] for item in item_keys], dtype=np.int64)
    subject_names = [
        parse_subject_name(str(value)) for value in df["subject_content"].astype(str).values
    ]
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    embed_dim = item_embeddings.shape[1]
    for name, row in zip(subject_names, rows, strict=False):
        if name not in sums:
            sums[name] = np.zeros(embed_dim, dtype=np.float64)
            counts[name] = 0
        sums[name] += item_embeddings[row]
        counts[name] += 1
    subject_embeddings = {
        name: (sums[name] / max(counts[name], 1)).astype(np.float32)
        for name in sums
    }
    global_subject_embedding = item_embeddings.mean(axis=0).astype(np.float32)
    return subject_embeddings, global_subject_embedding


def train_embedding(
    data_path: str | Path,
    output_path: str | Path,
    encoder_id: str = ENCODER_ID,
    max_rows: int = 1_000_000,
    seed: int = 321,
    batch_size: int = 128,
    C: float = 1.0,
    device: str | None = None,
    class_weight: str | None = None,
    with_subject_embed: bool = False,
) -> Path:
    """Fit the embedding head and save weights as native numpy arrays."""
    df = pd.read_parquet(data_path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    baseline = SmoothedPriorPredictiveEvaluator.fit(df.to_dict("records"))
    item_embeddings, item_to_row = encode_unique_items(
        df, encoder_id, batch_size=batch_size, device=device
    )

    subject_embeddings: dict[str, np.ndarray] | None = None
    global_subject_embedding: np.ndarray | None = None
    if with_subject_embed:
        subject_embeddings, global_subject_embedding = compute_subject_embeddings(
            df, item_embeddings, item_to_row
        )

    features = build_features(
        df,
        item_embeddings,
        item_to_row,
        subject_means=baseline.subject,
        global_mean=baseline.global_mean,
        subject_embeddings=subject_embeddings,
        global_subject_embedding=global_subject_embedding,
    )
    labels = df["label"].to_numpy(dtype=np.int64)

    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight=class_weight,
        n_jobs=-1,
    ).fit(scaled, labels)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savez_kwargs: dict[str, np.ndarray] = dict(
        encoder_id=np.array(encoder_id),
        coef=clf.coef_.astype(np.float32),
        intercept=clf.intercept_.astype(np.float32),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        global_mean=np.array(baseline.global_mean, dtype=np.float32),
        subject_names=np.array(list(baseline.subject.keys())),
        subject_values=np.array(list(baseline.subject.values()), dtype=np.float32),
        with_subject_embed=np.array(1 if with_subject_embed else 0, dtype=np.int32),
    )
    if with_subject_embed and subject_embeddings is not None:
        names = list(subject_embeddings.keys())
        savez_kwargs["subject_embed_names"] = np.array(names)
        savez_kwargs["subject_embed_values"] = np.stack(
            [subject_embeddings[name] for name in names], axis=0
        ).astype(np.float32)
        savez_kwargs["global_subject_embedding"] = global_subject_embedding.astype(np.float32)
    np.savez(output_path, **savez_kwargs)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Joined runtime examples parquet.")
    parser.add_argument("--out", default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--encoder-id", default=ENCODER_ID)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default=None,
        help="Encoder device: 'cpu', 'mps', 'cuda', or omit for auto.",
    )
    parser.add_argument(
        "--class-weight",
        default=None,
        choices=[None, "balanced"],
        help="Pass 'balanced' to match the prior embedding submission; default leaves the LR "
        "head free to calibrate to the actual ~65/35 base rate (better NLL).",
    )
    parser.add_argument(
        "--with-subject-embed",
        action="store_true",
        help="Also save a per-subject mean item embedding and concat it into features.",
    )
    args = parser.parse_args()

    output_path = train_embedding(
        args.data,
        args.out,
        encoder_id=args.encoder_id,
        max_rows=args.max_rows,
        seed=args.seed,
        batch_size=args.batch_size,
        C=args.C,
        device=args.device,
        class_weight=args.class_weight,
        with_subject_embed=args.with_subject_embed,
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
