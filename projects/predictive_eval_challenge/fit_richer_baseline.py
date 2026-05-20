#!/usr/bin/env python
# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Fit a richer smoothed-prior baseline with a 4-way subject_benchmark_condition cell.

The deployed baseline uses subject, benchmark, benchmark_condition, and
subject_benchmark cells. The natural missing dimension is the 4-way
(subject, benchmark, condition) cell, which is the finest cell that can
still be reliably populated from the public training data.

We compute smoothed empirical means per cell using the same Beta-prior
formula as `SmoothedPriorPredictiveEvaluator`:

    p_cell = (n_correct + strength * p_parent) / (n_total + strength)

The parent for the 4-way cell is the 3-way (subject, benchmark) cell, which
falls back transitively to subject -> benchmark -> global.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def parse_subject_name(s: str) -> str:
    m = re.search(r"^Name:\s*(.+)$", s or "", flags=re.MULTILINE)
    return m.group(1).strip().lower() if m else (s or "").strip().lower()


def _key(*parts) -> str:
    return "||".join(str(p) for p in parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="projects/predictive_eval_challenge/data/runtime_examples.parquet",
    )
    parser.add_argument(
        "--existing",
        default="projects/predictive_eval_challenge/codabench_submissions/factor_baseline_ensemble/artifacts/smoothed_prior.json",
    )
    parser.add_argument(
        "--out",
        default="projects/predictive_eval_challenge/codabench_submissions/factor_baseline_ensemble/artifacts/smoothed_prior_v2.json",
    )
    parser.add_argument("--strength", type=float, default=25.0)
    args = parser.parse_args()

    print(f"[load] {args.data}", flush=True)
    df = pd.read_parquet(args.data)
    df["subject_name"] = df["subject_content"].astype(str).map(parse_subject_name)
    df["benchmark"] = df["benchmark"].astype(str)
    df["condition"] = df["condition"].fillna("none").astype(str).replace("", "none")
    df["label"] = df["label"].astype(np.float32)
    print(f"[load] {len(df):,} rows, {df['subject_name'].nunique()} subjects, "
          f"{df['benchmark'].nunique()} benchmarks", flush=True)

    global_mean = float(df["label"].mean())
    print(f"[global] mean={global_mean:.4f}", flush=True)

    def smoothed_mean(group_keys: list[str], parent: dict[str, float] | float) -> dict[str, float]:
        """Smooth Bayesian shrinkage of group means toward a per-row parent."""
        t0 = time.time()
        g = df.groupby(group_keys)["label"].agg(["sum", "count"]).reset_index()
        keys = [_key(*row) for row in g[group_keys].astype(str).itertuples(index=False)]
        out: dict[str, float] = {}
        if isinstance(parent, dict):
            # Parent is a dict keyed by the prefix of group_keys (e.g. subject_benchmark
            # for parent of subject_benchmark_condition). For each child row, look up
            # the parent value.
            n_parent_keys = len(group_keys) - 1
            parent_keys = [
                _key(*row[:n_parent_keys])
                for row in g[group_keys].astype(str).itertuples(index=False)
            ]
            parent_vals = np.array(
                [parent.get(pk, global_mean) for pk in parent_keys], dtype=np.float32
            )
        else:
            parent_vals = np.full(len(g), float(parent), dtype=np.float32)
        smoothed = (g["sum"].to_numpy(dtype=np.float32) + args.strength * parent_vals) / (
            g["count"].to_numpy(dtype=np.float32) + args.strength
        )
        for k, p in zip(keys, smoothed):
            out[k] = float(p)
        print(f"[fit] {group_keys}: {len(out)} cells in {time.time() - t0:.1f}s", flush=True)
        return out

    subject_d = smoothed_mean(["subject_name"], global_mean)
    benchmark_d = smoothed_mean(["benchmark"], global_mean)
    benchmark_condition_d = smoothed_mean(["benchmark", "condition"], benchmark_d)
    subject_benchmark_d = smoothed_mean(["subject_name", "benchmark"], subject_d)
    subject_benchmark_condition_d = smoothed_mean(
        ["subject_name", "benchmark", "condition"], subject_benchmark_d
    )

    out = {
        "global_mean": global_mean,
        "strength": args.strength,
        "subject": subject_d,
        "benchmark": benchmark_d,
        "benchmark_condition": benchmark_condition_d,
        "subject_benchmark": subject_benchmark_d,
        "subject_benchmark_condition": subject_benchmark_condition_d,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"\n[write] {args.out}  ({Path(args.out).stat().st_size:,} bytes)", flush=True)
    print(
        f"[stats] subject_benchmark_condition cells: "
        f"{len(subject_benchmark_condition_d):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
