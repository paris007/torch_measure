#!/usr/bin/env python
# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Per-benchmark diagnostic: factor vs baseline vs ensemble NLL/AUC.

Loads the v3 factor artifact and the smoothed-prior baseline, scores each
benchmark in the cold-start holdout separately, and grid-searches the
optimal `factor_weight` and `temperature` per benchmark. The output tells us
where we are leaking NLL and whether per-benchmark routing would help.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4
MAX_ITEM_CHARS = 800
MAX_SEQ_LENGTH = 128


def parse_subject_name(s: str) -> str:
    m = re.search(r"^Name:\s*(.+)$", s or "", flags=re.MULTILINE)
    return m.group(1).strip().lower() if m else (s or "").strip().lower()


def format_item_text(row: dict) -> str:
    return (
        f"Benchmark: {row.get('benchmark', '')}\n"
        f"Condition: {row.get('condition', 'none') or 'none'}\n"
        f"Item: {str(row.get('item_content', ''))[:MAX_ITEM_CHARS]}"
    )


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def mlp_forward(embed: np.ndarray, layers):
    h = embed
    for idx, (w, bias) in enumerate(layers):
        h = h @ w.T + bias
        if idx < len(layers) - 1:
            h = gelu(h)
    return h


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def baseline_predict(df: pd.DataFrame, smoothed: dict) -> np.ndarray:
    """Mirror the v2 baseline (sbc-first) used in the deployed model.py."""
    global_mean = float(smoothed.get("global_mean", 0.5))
    subj_d = smoothed.get("subject", {})
    bench_d = smoothed.get("benchmark", {})
    bench_cond_d = smoothed.get("benchmark_condition", {})
    subj_bench_d = smoothed.get("subject_benchmark", {})
    sbc_d = smoothed.get("subject_benchmark_condition", {})
    out = np.empty(len(df), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        subj = row.subject_name
        bench = str(row.benchmark)
        cond = str(getattr(row, "condition", "none") or "none")
        sbc = sbc_d.get(f"{subj}||{bench}||{cond}")
        if sbc is not None:
            out[i] = 0.95 * float(sbc) + 0.05 * global_mean
            continue
        sb = subj_bench_d.get(f"{subj}||{bench}")
        if sb is not None:
            out[i] = 0.92 * float(sb) + 0.08 * global_mean
            continue
        bc = bench_cond_d.get(f"{bench}||{cond}")
        s = subj_d.get(subj)
        b = bench_d.get(bench)
        pairs = [(bc, 0.45), (s, 0.40), (b, 0.15)]
        avail = [(v, w) for v, w in pairs if v is not None]
        if not avail:
            out[i] = global_mean
        else:
            tw = sum(w for _, w in avail)
            pred = sum(float(v) * w for v, w in avail) / tw
            out[i] = 0.85 * pred + 0.15 * global_mean
    return np.clip(out, EPS, 1 - EPS)


def factor_logits(df: pd.DataFrame, art: dict, embeds_by_row: np.ndarray) -> np.ndarray:
    h = mlp_forward(embeds_by_row, art["mlp_layers"])
    log_a = h[:, 0]
    b = h[:, 1]
    a = np.exp(np.clip(log_a, -8, 8))
    theta = np.array(
        [art["subject_theta"].get(s, art["global_theta"]) for s in df["subject_name"]],
        dtype=np.float32,
    )
    return a * theta - b  # raw logits, no T applied yet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--factor-artifact",
        default="projects/predictive_eval_challenge/codabench_submissions/factor_pge/artifacts/factor_pge.npz",
    )
    parser.add_argument(
        "--smoothed-json",
        default="projects/predictive_eval_challenge/codabench_submissions/factor_baseline_ensemble/artifacts/smoothed_prior.json",
    )
    parser.add_argument(
        "--data",
        default="projects/predictive_eval_challenge/data/runtime_examples.parquet",
    )
    parser.add_argument("--holdout", default="matharena,mmlupro,rewardbench,swebench")
    parser.add_argument("--all-benchmarks", action="store_true",
                        help="ignore --holdout, evaluate per-benchmark weights on ALL benchmarks present in --data")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--max-rows-per-bench", type=int, default=80_000)
    parser.add_argument("--out-config", default="projects/predictive_eval_challenge/dist/per_benchmark_weights.json")
    args = parser.parse_args()

    factor_npz = np.load(args.factor_artifact, allow_pickle=False)
    layer_indices = sorted(int(k[len("mlp_w") :]) for k in factor_npz.files if k.startswith("mlp_w"))
    art = dict(
        encoder_id=str(factor_npz["encoder_id"]),
        subject_theta={
            str(n): float(v)
            for n, v in zip(factor_npz["subject_names"].tolist(), factor_npz["subject_theta"].tolist())
        },
        global_theta=float(factor_npz["global_theta"]),
        global_mean=float(factor_npz["global_mean"]),
        temperature=float(factor_npz["temperature"]) if "temperature" in factor_npz.files else 1.0,
        mlp_layers=[
            (np.asarray(factor_npz[f"mlp_w{i}"], dtype=np.float32), np.asarray(factor_npz[f"mlp_b{i}"], dtype=np.float32))
            for i in layer_indices
        ],
    )

    with open(args.smoothed_json) as fh:
        smoothed = json.load(fh)

    df_full = pd.read_parquet(args.data)
    df_full["subject_name"] = df_full["subject_content"].astype(str).map(parse_subject_name)
    df_full["item_text"] = df_full.apply(lambda r: format_item_text(r), axis=1)
    if args.all_benchmarks:
        holdout = sorted(df_full["benchmark"].dropna().unique().tolist())
    else:
        holdout = [b.strip() for b in args.holdout.split(",") if b.strip()]

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(art["encoder_id"], device=args.device)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    per_bench: list[dict] = []
    for bench in holdout:
        d = df_full[df_full["benchmark"] == bench].copy()
        if len(d) == 0:
            continue
        if args.max_rows_per_bench and len(d) > args.max_rows_per_bench:
            d = d.sample(n=args.max_rows_per_bench, random_state=0).reset_index(drop=True)
        print(f"[bench:{bench}] rows={len(d):,}", flush=True)
        unique_texts = sorted(d["item_text"].unique().tolist())
        text_to_idx = {t: i for i, t in enumerate(unique_texts)}
        t0 = time.time()
        embeds_unique = np.asarray(
            encoder.encode(
                unique_texts,
                batch_size=args.encode_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )
        print(f"  encoded {len(unique_texts):,} unique items in {time.time() - t0:.1f}s", flush=True)
        row_emb_idx = d["item_text"].map(text_to_idx).to_numpy(dtype=np.int64)
        embeds_by_row = embeds_unique[row_emb_idx]

        y = d["label"].to_numpy(dtype=np.float32)
        raw_logits = factor_logits(d, art, embeds_by_row)
        base_p = baseline_predict(d, smoothed)

        # Per-benchmark optimal T
        T_grid = np.concatenate([np.linspace(0.1, 1.0, 19), np.linspace(1.1, 3.0, 20)])
        best_T, best_T_nll = 1.0, float("inf")
        for T in T_grid:
            p = 1.0 / (1.0 + np.exp(-raw_logits / T))
            v = nll(y, p)
            if v < best_T_nll:
                best_T_nll, best_T = v, float(T)

        # NLL at the global T (=0.275)
        T_global = max(art["temperature"], 1e-3)
        p_factor_global = 1.0 / (1.0 + np.exp(-raw_logits / T_global))
        nll_factor_global = nll(y, p_factor_global)
        auc_factor_global = auc(y, p_factor_global)
        nll_baseline = nll(y, base_p)
        auc_baseline = auc(y, base_p)

        # Per-benchmark optimal logit-mean ensemble weight
        def logit(q):
            q = np.clip(q, EPS, 1 - EPS)
            return np.log(q / (1.0 - q))

        f_logit = raw_logits / T_global  # use global T for ensemble grid (matches deployed model)
        b_logit = logit(base_p)
        best_w, best_w_nll = 1.0, float("inf")
        for w in np.linspace(0.0, 1.0, 21):
            mix_logit = w * f_logit + (1.0 - w) * b_logit
            p = 1.0 / (1.0 + np.exp(-mix_logit))
            v = nll(y, p)
            if v < best_w_nll:
                best_w_nll, best_w = v, float(w)

        # Best-of-both: per-bench T AND per-bench weight
        f_logit_best = raw_logits / max(best_T, 1e-3)
        best_w2, best_w2_nll = 1.0, float("inf")
        for w in np.linspace(0.0, 1.0, 21):
            mix_logit = w * f_logit_best + (1.0 - w) * b_logit
            p = 1.0 / (1.0 + np.exp(-mix_logit))
            v = nll(y, p)
            if v < best_w2_nll:
                best_w2_nll, best_w2 = v, float(w)

        per_bench.append(
            dict(
                benchmark=bench,
                n=len(d),
                nll_factor_globalT=nll_factor_global,
                auc_factor_globalT=auc_factor_global,
                nll_baseline=nll_baseline,
                auc_baseline=auc_baseline,
                best_T_factor=best_T,
                nll_factor_bestT=best_T_nll,
                best_factor_weight_globalT=best_w,
                nll_ensemble_bestw_globalT=best_w_nll,
                best_factor_weight_bestT=best_w2,
                nll_ensemble_bestw_bestT=best_w2_nll,
            )
        )

    print()
    print(f"{'benchmark':14s} {'n':>7s} {'fac@T*':>8s} {'baseline':>8s} {'T*':>6s} {'fac@T*':>9s} "
          f"{'mix*g':>9s} {'w*g':>6s} {'mix*b':>9s} {'w*b':>6s}")
    for r in per_bench:
        print(
            f"{r['benchmark']:14s} {r['n']:7,d} {r['nll_factor_globalT']:8.4f} {r['nll_baseline']:8.4f} "
            f"{r['best_T_factor']:6.3f} {r['nll_factor_bestT']:9.4f} "
            f"{r['nll_ensemble_bestw_globalT']:9.4f} {r['best_factor_weight_globalT']:6.2f} "
            f"{r['nll_ensemble_bestw_bestT']:9.4f} {r['best_factor_weight_bestT']:6.2f}"
        )

    # Aggregate (weighted by n)
    if per_bench:
        ns = np.array([r["n"] for r in per_bench], dtype=np.float64)
        w = ns / ns.sum()
        agg_factor_global = sum(w[i] * per_bench[i]["nll_factor_globalT"] for i in range(len(per_bench)))
        agg_baseline = sum(w[i] * per_bench[i]["nll_baseline"] for i in range(len(per_bench)))
        agg_ens_globalT = sum(w[i] * per_bench[i]["nll_ensemble_bestw_globalT"] for i in range(len(per_bench)))
        agg_ens_bestT = sum(w[i] * per_bench[i]["nll_ensemble_bestw_bestT"] for i in range(len(per_bench)))
        print()
        print("--- weighted-by-n averages ---")
        print(f"factor (global T={T_global:.3f}):       {agg_factor_global:.5f}")
        print(f"baseline:                               {agg_baseline:.5f}")
        print(f"per-bench ensemble (global T):          {agg_ens_globalT:.5f}")
        print(f"per-bench ensemble (per-bench T):       {agg_ens_bestT:.5f}")

    out_cfg = {
        "factor_temperature_global": float(T_global),
        "per_benchmark": {
            r["benchmark"]: {
                "factor_weight": r["best_factor_weight_bestT"],
                "factor_temperature": r["best_T_factor"],
            }
            for r in per_bench
        },
    }
    Path(args.out_config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_config).write_text(json.dumps(out_cfg, indent=2))
    print(f"\n[write] per-benchmark config -> {args.out_config}")


if __name__ == "__main__":
    main()
