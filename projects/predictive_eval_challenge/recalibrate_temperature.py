# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Refit the factor model's temperature on a whole-benchmark holdout.

The training script fits `T` on a random in-sample slice; items in that slice
are mostly seen during 2-PL fitting, so the optimum `T` is too aggressive for
true cold-start (we observed v2 going from −0.594 in-sample to −0.62 on
Codabench precisely because of this).

This script holds out N whole benchmarks, computes raw factor logits on those
rows using the already-trained artifact, and grid-searches `T` to minimise NLL
on the held-out rows. The artifact is rewritten in-place with the new `T`;
nothing else changes, so the encoder and MLP do not need retraining.

Usage:
    python projects/predictive_eval_challenge/recalibrate_temperature.py \
        --data projects/predictive_eval_challenge/data/runtime_examples.parquet \
        --artifact projects/predictive_eval_challenge/codabench_submissions/factor_pge/artifacts/factor_pge.npz \
        --holdout-benchmarks hle mathvista_mini mtbench \
        --device mps
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from torch_measure.models import format_item_text, parse_subject_name


EPS = 1e-4
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _grid_search_T(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, dict[float, float]]:
    if grid is None:
        grid = np.concatenate(
            [
                np.linspace(0.15, 1.0, 35),
                np.linspace(1.05, 3.0, 40),
            ]
        )
        if not np.any(np.isclose(grid, 1.0)):
            grid = np.append(grid, 1.0)
    scores: dict[float, float] = {}
    best_T = 1.0
    best_nll = float("inf")
    for T in grid:
        p = 1.0 / (1.0 + np.exp(-logits / max(T, 1e-3)))
        nll = _nll(labels, p)
        scores[float(T)] = nll
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T, scores


def compute_factor_logits(
    df: pd.DataFrame,
    artifact: np.lib.npyio.NpzFile,
    encoder_id: str,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Recompute raw factor logits (pre-temperature) for every row in df."""
    from sentence_transformers import SentenceTransformer

    subject_theta = {
        str(name): float(value)
        for name, value in zip(
            artifact["subject_names"].tolist(),
            artifact["subject_theta"].tolist(),
        )
    }
    global_theta = float(artifact["global_theta"])

    layer_indices = sorted(
        int(k[len("mlp_w") :]) for k in artifact.files if k.startswith("mlp_w")
    )
    layers = [
        (
            np.asarray(artifact[f"mlp_w{i}"], dtype=np.float32),
            np.asarray(artifact[f"mlp_b{i}"], dtype=np.float32),
        )
        for i in layer_indices
    ]

    df = df.copy()
    df["item_key"] = df["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    df["subject_name"] = df["subject_content"].astype(str).map(parse_subject_name)

    unique_keys = df["item_key"].drop_duplicates().reset_index(drop=True)
    unique_rows = df.drop_duplicates("item_key").reset_index(drop=True)
    texts = [format_item_text(r) for r in unique_rows.to_dict("records")]

    encoder = (
        SentenceTransformer(encoder_id, device=device)
        if device
        else SentenceTransformer(encoder_id)
    )
    encoder.max_seq_length = MAX_SEQ_LENGTH
    embeds = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    h = embeds
    for idx, (w, bias) in enumerate(layers):
        h = h @ w.T + bias
        if idx < len(layers) - 1:
            h = 0.5 * h * (
                1.0
                + np.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h**3))
            )
    log_a = h[:, 0]
    b_arr = h[:, 1]
    a_arr = np.exp(np.clip(log_a, -8, 8))

    key_to_idx = {k: i for i, k in enumerate(unique_rows["item_key"].astype(str).tolist())}
    rows = np.array(
        [key_to_idx[k] for k in df["item_key"].astype(str).tolist()],
        dtype=np.int64,
    )
    a_pred = a_arr[rows]
    b_pred = b_arr[rows]
    theta_lookup = np.array(
        [
            subject_theta.get(name, global_theta)
            for name in df["subject_name"].astype(str).tolist()
        ],
        dtype=np.float32,
    )
    return a_pred * theta_lookup - b_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Joined runtime examples parquet.")
    parser.add_argument(
        "--artifact",
        default=(
            "projects/predictive_eval_challenge/codabench_submissions/"
            "factor_pge/artifacts/factor_pge.npz"
        ),
    )
    parser.add_argument(
        "--holdout-benchmarks",
        nargs="*",
        default=None,
        help=(
            "Specific benchmark names to treat as cold-start holdout. "
            "Defaults to a deterministic sample of `--n-holdout` benchmarks."
        ),
    )
    parser.add_argument("--n-holdout", type=int, default=3)
    parser.add_argument("--seed", type=int, default=987)
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the new T back to the artifact (otherwise dry-run).",
    )
    parser.add_argument(
        "--also-update",
        nargs="*",
        default=[
            "projects/predictive_eval_challenge/codabench_submissions/"
            "factor_baseline_ensemble/artifacts/factor_pge.npz"
        ],
        help="Other artifact paths to update with the same T.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    benchmarks = sorted(df["benchmark"].dropna().astype(str).unique().tolist())
    print(f"[recal] available benchmarks ({len(benchmarks)}): {benchmarks}", flush=True)

    if args.holdout_benchmarks:
        holdout = sorted(set(args.holdout_benchmarks) & set(benchmarks))
        if not holdout:
            raise SystemExit(
                f"No matching benchmarks in --holdout-benchmarks={args.holdout_benchmarks}"
            )
    else:
        rng = np.random.default_rng(args.seed)
        bench_arr = np.array(benchmarks)
        rng.shuffle(bench_arr)
        holdout = sorted(bench_arr[: args.n_holdout].tolist())

    print(f"[recal] cold-start holdout: {holdout}", flush=True)
    hold_df = df[df["benchmark"].astype(str).isin(holdout)].copy()
    if args.max_rows and len(hold_df) > args.max_rows:
        hold_df = hold_df.sample(n=args.max_rows, random_state=args.seed).reset_index(
            drop=True
        )
    print(
        f"[recal] {len(hold_df):,} held-out rows over {hold_df['item_content'].nunique():,} items",
        flush=True,
    )

    artifact = np.load(args.artifact, allow_pickle=False)
    encoder_id = str(artifact["encoder_id"])
    stored_T = float(artifact["temperature"]) if "temperature" in artifact.files else 1.0
    print(f"[recal] artifact T (stored): {stored_T:.4f}", flush=True)

    logits = compute_factor_logits(
        hold_df,
        artifact,
        encoder_id,
        batch_size=args.batch_size,
        device=args.device,
    )
    y = hold_df["label"].to_numpy(dtype=np.float32)

    best_T, scores = _grid_search_T(logits, y)
    nll_stored = scores[
        min(scores, key=lambda t: abs(t - stored_T))
    ] if stored_T <= max(scores) else _nll(y, 1.0 / (1.0 + np.exp(-logits / stored_T)))
    nll_T1 = scores[min(scores, key=lambda t: abs(t - 1.0))]
    print()
    print(f"  NLL @ T=stored ({stored_T:.3f})  : {nll_stored:.5f}")
    print(f"  NLL @ T=1.0                      : {nll_T1:.5f}")
    print(f"  NLL @ T*=best ({best_T:.3f})    : {scores[best_T]:.5f}")
    print()
    print("  Grid (top 10 nearest to best):")
    near = sorted(scores.items(), key=lambda kv: abs(kv[0] - best_T))[:10]
    for T, nll in sorted(near):
        marker = "  <- T*" if math.isclose(T, best_T) else ""
        print(f"    T={T:.3f}  NLL={nll:.5f}{marker}")

    if not args.write:
        print(
            "\n[recal] dry-run; re-run with --write to update the artifact.",
            flush=True,
        )
        return

    targets = [Path(args.artifact)] + [Path(p) for p in (args.also_update or [])]
    for path in targets:
        if not path.exists():
            print(f"[recal] skipping (missing): {path}", flush=True)
            continue
        data = dict(np.load(path, allow_pickle=False))
        data["temperature"] = np.array(best_T, dtype=np.float32)
        np.savez(path, **data)
        print(f"[recal] wrote T={best_T:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
