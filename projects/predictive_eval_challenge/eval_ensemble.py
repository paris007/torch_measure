#!/usr/bin/env python
# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Evaluate single / multi-seed factor models on a benchmark holdout.

Loads multiple `.npz` artifacts and reports per-seed NLL/AUC and an ensemble
(probability-space mean) NLL/AUC on the cold-start holdout. Uses the local
runtime_examples.parquet so no network calls are needed.

Usage:
    python projects/predictive_eval_challenge/eval_ensemble.py \\
        --artifacts \\
            projects/predictive_eval_challenge/codabench_submissions/factor_pge/artifacts/factor_pge.npz \\
            projects/predictive_eval_challenge/ensemble_artifacts/minilm_multiseed_v2/factor_pge_seed*.npz \\
        --holdout matharena,mmlupro,rewardbench,swebench \\
        --device mps
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128


def parse_subject_name(subject_content: str) -> str:
    m = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    return m.group(1).strip().lower() if m else (subject_content or "").strip().lower()


def format_item_text(row: dict) -> str:
    return (
        f"Benchmark: {row.get('benchmark', '')}\n"
        f"Condition: {row.get('condition', 'none') or 'none'}\n"
        f"Item: {str(row.get('item_content', ''))[:MAX_ITEM_CHARS]}"
    )


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def mlp_forward(embed: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]]):
    h = embed
    for idx, (w, bias) in enumerate(layers):
        h = h @ w.T + bias
        if idx < len(layers) - 1:
            h = gelu(h)
    return h


def load_artifact(path: Path):
    npz = np.load(path, allow_pickle=False)
    layer_indices = sorted(int(k[len("mlp_w") :]) for k in npz.files if k.startswith("mlp_w"))
    return dict(
        path=path,
        encoder_id=str(npz["encoder_id"]),
        subject_theta={
            str(n): float(v) for n, v in zip(npz["subject_names"].tolist(), npz["subject_theta"].tolist())
        },
        global_theta=float(npz["global_theta"]),
        global_mean=float(npz["global_mean"]),
        temperature=(
            float(npz["temperature"]) if "temperature" in npz.files else 1.0
        ),
        mlp_layers=[
            (np.asarray(npz[f"mlp_w{i}"], dtype=np.float32), np.asarray(npz[f"mlp_b{i}"], dtype=np.float32))
            for i in layer_indices
        ],
    )


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def predict_one_artifact(artifact, embeds_by_row: np.ndarray, subject_names_by_row):
    log_ab = mlp_forward(embeds_by_row, artifact["mlp_layers"])
    log_a = log_ab[:, 0]
    b = log_ab[:, 1]
    a = np.exp(np.clip(log_a, -8, 8))
    theta = np.array(
        [artifact["subject_theta"].get(s, artifact["global_theta"]) for s in subject_names_by_row],
        dtype=np.float32,
    )
    T = max(artifact["temperature"], 1e-3)
    logits = (a * theta - b) / T
    return 1.0 / (1.0 + np.exp(-logits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", nargs="+", required=True)
    parser.add_argument("--data", default="projects/predictive_eval_challenge/data/runtime_examples.parquet")
    parser.add_argument("--holdout", default="matharena,mmlupro,rewardbench,swebench")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=200_000, help="Cap eval rows for speed")
    args = parser.parse_args()

    holdout = [b.strip() for b in args.holdout.split(",") if b.strip()]
    paths: list[Path] = []
    for pattern in args.artifacts:
        matched = sorted(Path().glob(pattern))
        if matched:
            paths.extend(matched)
        else:
            paths.append(Path(pattern))
    print(f"[load] artifacts: {[p.name for p in paths]}", flush=True)
    artifacts = [load_artifact(p) for p in paths]

    df = pd.read_parquet(args.data)
    df = df[df["benchmark"].isin(holdout)].copy()
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=0).reset_index(drop=True)
    print(f"[data] holdout rows: {len(df):,}  benchmarks: {sorted(df['benchmark'].unique())}", flush=True)

    df["subject_name"] = df["subject_content"].astype(str).map(parse_subject_name)
    df["item_text"] = df.apply(lambda r: format_item_text(r), axis=1)
    y = df["label"].to_numpy(dtype=np.float32)

    encoder_id = artifacts[0]["encoder_id"]
    for a in artifacts[1:]:
        if a["encoder_id"] != encoder_id:
            raise RuntimeError(f"encoder mismatch: {encoder_id} vs {a['encoder_id']}")
    print(f"[encode] loading encoder {encoder_id}", flush=True)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_id, device=args.device)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    unique_texts = sorted(df["item_text"].unique().tolist())
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    print(f"[encode] unique items: {len(unique_texts):,}", flush=True)
    t0 = time.time()
    embeds_unique = np.asarray(
        encoder.encode(
            unique_texts,
            batch_size=args.encode_batch,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    print(f"[encode] done in {time.time() - t0:.1f}s, shape={embeds_unique.shape}", flush=True)
    row_emb_idx = df["item_text"].map(text_to_idx).to_numpy(dtype=np.int64)
    embeds_by_row = embeds_unique[row_emb_idx]

    subject_names = df["subject_name"].tolist()

    per_seed_probs = []
    print("\n[per-seed]")
    print(f"{'artifact':50s} {'T':>6s} {'n_subj':>7s} {'NLL':>8s} {'AUC':>8s}")
    for art in artifacts:
        p = predict_one_artifact(art, embeds_by_row, subject_names)
        per_seed_probs.append(p)
        print(
            f"{art['path'].name:50s} {art['temperature']:6.3f} {len(art['subject_theta']):7d} "
            f"{nll(y, p):8.5f} {auc(y, p):8.5f}",
            flush=True,
        )

    if len(per_seed_probs) > 1:
        ens = np.mean(per_seed_probs, axis=0)
        print(f"\n[ensemble of {len(per_seed_probs)}]  NLL={nll(y, ens):.5f}  AUC={auc(y, ens):.5f}")

        for i in range(1, len(per_seed_probs)):
            sub = np.mean(per_seed_probs[1 : i + 1], axis=0)
            print(
                f"  ensemble seeds 1..{i} (no v3):  NLL={nll(y, sub):.5f}  AUC={auc(y, sub):.5f}"
            )


if __name__ == "__main__":
    main()
