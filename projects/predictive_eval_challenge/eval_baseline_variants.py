#!/usr/bin/env python
# Copyright (c) 2026 AIMS Foundations. MIT License.
"""Compare baseline variants (with/without 4-way cell) on holdout rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-4


def parse_subject_name(s: str) -> str:
    m = re.search(r"^Name:\s*(.+)$", s or "", flags=re.MULTILINE)
    return m.group(1).strip().lower() if m else (s or "").strip().lower()


def _key(*parts) -> str:
    return "||".join(str(p) for p in parts)


def baseline_v1(df, B):
    g = float(B.get("global_mean", 0.5))
    sd, bd, bcd, sbd = (
        B.get("subject", {}),
        B.get("benchmark", {}),
        B.get("benchmark_condition", {}),
        B.get("subject_benchmark", {}),
    )
    out = np.empty(len(df), dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        cond = str(getattr(r, "condition", "none") or "none")
        bench = str(r.benchmark)
        subj = r.subject_name
        pairs = [
            (sd.get(subj), 0.35),
            (bd.get(bench), 0.15),
            (bcd.get(_key(bench, cond)), 0.25),
            (sbd.get(_key(subj, bench)), 0.25),
        ]
        avail = [(v, w) for v, w in pairs if v is not None]
        if not avail:
            out[i] = g
        else:
            tw = sum(w for _, w in avail)
            pred = sum(float(v) * w for v, w in avail) / tw
            out[i] = 0.85 * pred + 0.15 * g
    return np.clip(out, EPS, 1 - EPS)


def baseline_v2(df, B, w_sbc: float = 0.40, w_sb: float = 0.20, w_bc: float = 0.20, w_s: float = 0.15, w_b: float = 0.05):
    g = float(B.get("global_mean", 0.5))
    sd, bd, bcd, sbd, sbcd = (
        B.get("subject", {}),
        B.get("benchmark", {}),
        B.get("benchmark_condition", {}),
        B.get("subject_benchmark", {}),
        B.get("subject_benchmark_condition", {}),
    )
    out = np.empty(len(df), dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        cond = str(getattr(r, "condition", "none") or "none")
        bench = str(r.benchmark)
        subj = r.subject_name
        pairs = [
            (sbcd.get(_key(subj, bench, cond)), w_sbc),
            (sbd.get(_key(subj, bench)), w_sb),
            (bcd.get(_key(bench, cond)), w_bc),
            (sd.get(subj), w_s),
            (bd.get(bench), w_b),
        ]
        avail = [(v, w) for v, w in pairs if v is not None]
        if not avail:
            out[i] = g
        else:
            tw = sum(w for _, w in avail)
            pred = sum(float(v) * w for v, w in avail) / tw
            out[i] = 0.85 * pred + 0.15 * g
    return np.clip(out, EPS, 1 - EPS)


def baseline_sbc_first(df, B):
    """Use subject_benchmark_condition if available, else falls back."""
    g = float(B.get("global_mean", 0.5))
    sd = B.get("subject", {})
    bd = B.get("benchmark", {})
    bcd = B.get("benchmark_condition", {})
    sbd = B.get("subject_benchmark", {})
    sbcd = B.get("subject_benchmark_condition", {})
    out = np.empty(len(df), dtype=np.float32)
    for i, r in enumerate(df.itertuples(index=False)):
        cond = str(getattr(r, "condition", "none") or "none")
        bench = str(r.benchmark)
        subj = r.subject_name
        v = sbcd.get(_key(subj, bench, cond))
        if v is None:
            v = sbd.get(_key(subj, bench))
        if v is None:
            v = bcd.get(_key(bench, cond))
        if v is None:
            v = sd.get(subj)
        if v is None:
            v = bd.get(bench)
        if v is None:
            v = g
        # Smaller global shrinkage when we have the deepest cell.
        out[i] = 0.95 * v + 0.05 * g
    return np.clip(out, EPS, 1 - EPS)


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="projects/predictive_eval_challenge/data/runtime_examples.parquet")
    parser.add_argument(
        "--prior",
        default="projects/predictive_eval_challenge/codabench_submissions/factor_baseline_ensemble/artifacts/smoothed_prior_v2.json",
    )
    parser.add_argument("--max-rows-per-bench", type=int, default=30000)
    parser.add_argument("--all-benchmarks", action="store_true", default=True)
    args = parser.parse_args()

    B = json.loads(Path(args.prior).read_text())
    df = pd.read_parquet(args.data)
    df["subject_name"] = df["subject_content"].astype(str).map(parse_subject_name)
    df["condition"] = df["condition"].fillna("none").astype(str).replace("", "none")
    df["benchmark"] = df["benchmark"].astype(str)

    benchmarks = sorted(df["benchmark"].unique().tolist())

    rows_v1 = []
    rows_v2 = []
    rows_sbc = []
    for bench in benchmarks:
        d = df[df["benchmark"] == bench]
        if len(d) > args.max_rows_per_bench:
            d = d.sample(n=args.max_rows_per_bench, random_state=0).reset_index(drop=True)
        y = d["label"].to_numpy(dtype=np.float32)
        p1 = baseline_v1(d, B)
        p2 = baseline_v2(d, B)
        ps = baseline_sbc_first(d, B)
        rows_v1.append((bench, len(d), nll(y, p1)))
        rows_v2.append((bench, len(d), nll(y, p2)))
        rows_sbc.append((bench, len(d), nll(y, ps)))

    print(f"{'benchmark':14s} {'n':>7s} {'v1(3-way)':>10s} {'v2(4-way mix)':>13s} {'sbc-first':>10s}")
    for r1, r2, rs in zip(rows_v1, rows_v2, rows_sbc):
        print(f"{r1[0]:14s} {r1[1]:7,d} {r1[2]:10.5f} {r2[2]:13.5f} {rs[2]:10.5f}")
    # Weighted aggregate
    ns = np.array([r[1] for r in rows_v1], dtype=np.float64)
    w = ns / ns.sum()
    v1_avg = sum(w[i] * rows_v1[i][2] for i in range(len(rows_v1)))
    v2_avg = sum(w[i] * rows_v2[i][2] for i in range(len(rows_v2)))
    sbc_avg = sum(w[i] * rows_sbc[i][2] for i in range(len(rows_sbc)))
    print()
    print(f"weighted avg: v1={v1_avg:.5f}  v2={v2_avg:.5f}  sbc-first={sbc_avg:.5f}")


if __name__ == "__main__":
    main()
