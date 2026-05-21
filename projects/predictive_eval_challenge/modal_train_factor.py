# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Train the factor / PGE model on Modal with a bigger encoder + more data.

Usage from your laptop (after `modal token new`):

    /opt/anaconda3/envs/torch_measure/bin/modal run \
        projects/predictive_eval_challenge/modal_train_factor.py

The trained `factor_pge.npz` and `smoothed_prior.json` are written into both
codabench submission folders so the next `make_submission.py` run picks them
up. The Modal Volume caches the joined parquet and SentenceTransformer
weights between runs, so re-running with a different config is fast.

This file is intentionally self-contained: the Modal container has no access
to the `torch_measure` package or any local files outside this module.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import modal


APP_NAME = "predictive-eval-factor"
VOLUME_NAME = "predictive-eval"
LOCAL_PROJECT = Path(__file__).resolve().parent
LOCAL_FACTOR_DIR = LOCAL_PROJECT / "codabench_submissions" / "factor_pge" / "artifacts"
LOCAL_ENSEMBLE_DIR = (
    LOCAL_PROJECT / "codabench_submissions" / "factor_baseline_ensemble" / "artifacts"
)


# --- Modal infrastructure ----------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.45,<4.55",
        "numpy<2",
        "pandas>=2.0",
        "pyarrow>=15",
        "datasets>=2.20,<3.0",
        "huggingface-hub>=0.24,<0.30",
        "sentence-transformers>=3.0,<4.0",
        "tqdm>=4.66",
    )
)

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


# --- Self-contained helpers (mirrors torch_measure.models) -------------------

REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
REPO_ID = "aims-foundations/measurement-db"
MAX_ITEM_CHARS = 1500
MAX_SEQ_LENGTH = 256


def _key(*parts: object) -> str:
    return "||".join(str(p) for p in parts)


def parse_subject_name(subject_content: str) -> str:
    text = subject_content or ""
    m = re.search(r"^Name:\s*(.+)$", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip().lower()
    return text.strip().lower()


def format_item_text(row: dict) -> str:
    return (
        f"Benchmark: {row.get('benchmark', '')}\n"
        f"Condition: {row.get('condition', 'none') or 'none'}\n"
        f"Item: {row.get('item_content', '')}"
    )


def render_subject_content(subject: dict, fallback: str) -> str:
    display_name = subject.get("display_name") or fallback
    lines = [f"Name: {display_name}"]
    for key, label in (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    ):
        value = subject.get(key)
        if value not in (None, ""):
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def fit_smoothed_prior(rows: list[dict], strength: float = 25.0) -> dict:
    labels = [float(r["label"]) for r in rows if "label" in r]
    global_mean = sum(labels) / len(labels) if labels else 0.5

    def group_mean(fields: tuple[str, ...]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in rows:
            if "label" not in row:
                continue
            vals = []
            for f in fields:
                if f == "subject_name":
                    vals.append(parse_subject_name(str(row.get("subject_content", ""))))
                elif f == "condition":
                    vals.append(str(row.get("condition", "none") or "none"))
                else:
                    vals.append(str(row.get(f, "")))
            k = _key(*vals)
            sums[k] = sums.get(k, 0.0) + float(row["label"])
            counts[k] = counts.get(k, 0) + 1
        return {
            k: (sums[k] + global_mean * strength) / (counts[k] + strength) for k in sums
        }

    return {
        "global_mean": global_mean,
        "strength": strength,
        "subject": group_mean(("subject_name",)),
        "benchmark": group_mean(("benchmark",)),
        "benchmark_condition": group_mean(("benchmark", "condition")),
        "subject_benchmark": group_mean(("subject_name", "benchmark")),
    }


def fit_rich_smoothed_prior(rows: list[dict], strength: float = 25.0) -> dict:
    """Fit the v2 prior hierarchy including subject-benchmark-condition."""
    labels = [float(r["label"]) for r in rows if "label" in r]
    global_mean = sum(labels) / len(labels) if labels else 0.5

    def vals_for(row: dict, fields: tuple[str, ...]) -> tuple[str, ...]:
        vals = []
        for f in fields:
            if f == "subject_name":
                vals.append(parse_subject_name(str(row.get("subject_content", ""))))
            elif f == "condition":
                vals.append(str(row.get("condition", "none") or "none"))
            else:
                vals.append(str(row.get(f, "")))
        return tuple(vals)

    def raw_counts(fields: tuple[str, ...]) -> tuple[dict[str, float], dict[str, int], dict[str, tuple[str, ...]]]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        key_parts: dict[str, tuple[str, ...]] = {}
        for row in rows:
            if "label" not in row:
                continue
            parts = vals_for(row, fields)
            k = _key(*parts)
            sums[k] = sums.get(k, 0.0) + float(row["label"])
            counts[k] = counts.get(k, 0) + 1
            key_parts[k] = parts
        return sums, counts, key_parts

    def smoothed(fields: tuple[str, ...], parent: dict[str, float] | float) -> dict[str, float]:
        sums, counts, key_parts = raw_counts(fields)
        out: dict[str, float] = {}
        for k, total in sums.items():
            if isinstance(parent, dict):
                pk = _key(*key_parts[k][:-1])
                parent_value = parent.get(pk, global_mean)
            else:
                parent_value = float(parent)
            out[k] = (total + strength * parent_value) / (counts[k] + strength)
        return out

    subject = smoothed(("subject_name",), global_mean)
    benchmark = smoothed(("benchmark",), global_mean)
    benchmark_condition = smoothed(("benchmark", "condition"), benchmark)
    subject_benchmark = smoothed(("subject_name", "benchmark"), subject)
    subject_benchmark_condition = smoothed(
        ("subject_name", "benchmark", "condition"), subject_benchmark
    )
    return {
        "global_mean": global_mean,
        "strength": strength,
        "subject": subject,
        "benchmark": benchmark,
        "benchmark_condition": benchmark_condition,
        "subject_benchmark": subject_benchmark,
        "subject_benchmark_condition": subject_benchmark_condition,
    }


# --- Data loading (runs inside the container) --------------------------------

def build_runtime_examples(cache_dir: Path):
    """Download + 4-table join. Caches to `cache_dir/runtime_examples.parquet`."""
    import pandas as pd
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    cache_path = cache_dir / "runtime_examples.parquet"
    if cache_path.exists():
        print(f"[data] using cached runtime examples: {cache_path}", flush=True)
        return pd.read_parquet(cache_path)

    print("[data] listing parquet files on the Hub...", flush=True)
    files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        f
        for f in files
        if f.endswith(".parquet")
        and f not in REGISTRY_FILES
        and not f.endswith("_traces.parquet")
    )
    print(f"[data] found {len(response_files)} response parquet shards", flush=True)

    response_features = Features(
        {
            "subject_id": Value("string"),
            "item_id": Value("string"),
            "benchmark_id": Value("string"),
            "trial": Value("int64"),
            "test_condition": Value("string"),
            "response": Value("float64"),
            "correct_answer": Value("string"),
            "trace": Value("string"),
        }
    )
    responses = load_dataset(
        REPO_ID, data_files=response_files, features=response_features, split="train"
    ).to_pandas()
    items = load_dataset(REPO_ID, data_files="items.parquet", split="train").to_pandas()
    subjects = load_dataset(
        REPO_ID, data_files="subjects.parquet", split="train"
    ).to_pandas()
    benchmarks = load_dataset(
        REPO_ID, data_files="benchmarks.parquet", split="train"
    ).to_pandas()

    item_cols = [c for c in ["item_id", "content"] if c in items.columns]
    subject_cols = [
        c
        for c in ["subject_id", "display_name", "provider", "params", "release_date", "family"]
        if c in subjects.columns
    ]
    benchmark_cols = [c for c in ["benchmark_id", "name"] if c in benchmarks.columns]

    df = responses.merge(items[item_cols], on="item_id", how="left")
    df = df.merge(subjects[subject_cols], on="subject_id", how="left")
    df = df.merge(
        benchmarks[benchmark_cols], on="benchmark_id", how="left", suffixes=("", "_meta")
    )

    subj_records = df[subject_cols].to_dict("records")
    df["subject_content"] = [
        render_subject_content(rec, fall)
        for rec, fall in zip(subj_records, df["subject_id"].astype(str), strict=False)
    ]
    df["benchmark"] = df["benchmark_id"].astype(str)
    df["condition"] = df["test_condition"].fillna("none").replace("", "none").astype(str)
    df["item_content"] = df["content"].fillna("").astype(str)
    df["label"] = df["response"].astype(float)
    df = df[df["label"].isin([0.0, 1.0])].copy()
    keep = [
        "benchmark",
        "condition",
        "subject_content",
        "item_content",
        "label",
        "subject_id",
        "item_id",
        "benchmark_id",
    ]
    df = df[keep].reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[data] cached {len(df):,} joined rows to {cache_path}", flush=True)
    return df


# --- Modal function: full training pipeline ----------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": vol},
    timeout=3 * 60 * 60,
)
def train_factor_bge(
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    max_rows: int = 4_000_000,
    seed: int = 321,
    factor_epochs: int = 8,
    factor_batch: int = 65536,
    mlp_hidden: int = 256,
    mlp_hidden_layers: int = 2,
    mlp_dropout: float = 0.15,
    mlp_epochs: int = 400,
    mlp_lr: float = 1e-3,
    encode_batch: int = 256,
    n_holdout_benchmarks: int = 4,
    theta_weight_decay: float = 1e-3,
    holdout_benchmarks: list[str] | None = None,
    run_label: str | None = None,
    holdout_mode: str = "benchmark",  # "benchmark" or "random_rows"
    calibration_frac: float = 0.05,
) -> tuple[bytes, dict]:
    """Full PGE pipeline. Returns (factor_pge.npz bytes, smoothed_prior dict)."""
    import os

    import numpy as np
    import pandas as pd
    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}  cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[setup] gpu: {torch.cuda.get_device_name(0)}", flush=True)

    cache_dir = Path("/cache")
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "hf_cache").mkdir(exist_ok=True)
    (cache_dir / "st_cache").mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "hf_cache")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir / "st_cache")

    df = build_runtime_examples(cache_dir)
    print(
        f"[data] total rows: {len(df):,}, unique items: "
        f"{df['item_content'].nunique():,}",
        flush=True,
    )

    rng = np.random.default_rng(seed)
    benchmarks = sorted(df["benchmark"].dropna().unique().tolist())
    if holdout_mode == "random_rows":
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_cal = max(1, int(len(df) * float(calibration_frac)))
        cal_df = df.iloc[:n_cal].copy()
        train_df = df.iloc[n_cal:].copy()
        holdout = []
        print(
            f"[holdout] random-row mode: {n_cal:,} calibration rows "
            f"(frac={calibration_frac:.3f}); training keeps all "
            f"{df['subject_content'].nunique():,} subjects",
            flush=True,
        )
    else:
        if holdout_benchmarks:
            holdout = sorted([b for b in holdout_benchmarks if b in benchmarks])
            missing = [b for b in holdout_benchmarks if b not in benchmarks]
            if missing:
                print(f"[holdout] WARN unknown benchmarks dropped: {missing}", flush=True)
        else:
            bench_arr = np.array(benchmarks)
            rng.shuffle(bench_arr)
            holdout = sorted(bench_arr[:n_holdout_benchmarks].tolist())
        print(f"[holdout] cold-start benchmarks: {holdout}", flush=True)
        is_holdout = df["benchmark"].isin(holdout)
        cal_df = df[is_holdout].copy()
        train_df = df[~is_holdout].copy()
        if max_rows and len(train_df) > max_rows:
            train_df = train_df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    print(
        f"[split] train rows: {len(train_df):,}  cal rows: {len(cal_df):,}",
        flush=True,
    )

    print("[baseline] fitting smoothed-prior baseline...", flush=True)
    smoothed = fit_smoothed_prior(train_df.to_dict("records"))
    print(f"[baseline] global_mean={smoothed['global_mean']:.4f}", flush=True)

    train_df["subject_name"] = train_df["subject_content"].astype(str).map(parse_subject_name)
    train_df["item_key"] = train_df["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    subject_names = sorted(train_df["subject_name"].unique().tolist())
    subject_to_idx = {n: i for i, n in enumerate(subject_names)}
    item_keys = sorted(train_df["item_key"].unique().tolist())
    item_to_idx = {k: i for i, k in enumerate(item_keys)}

    subj_ids = train_df["subject_name"].map(subject_to_idx).to_numpy(dtype=np.int64)
    item_ids = train_df["item_key"].map(item_to_idx).to_numpy(dtype=np.int64)
    labels = train_df["label"].to_numpy(dtype=np.float32)

    print(
        f"[factor] fitting 2-PL: {len(train_df):,} rows, "
        f"{len(subject_names)} subjects, {len(item_keys)} items",
        flush=True,
    )

    torch.manual_seed(seed)
    dev = torch.device(device)
    theta = nn.Parameter(torch.zeros(len(subject_names), device=dev))
    log_a = nn.Parameter(torch.zeros(len(item_keys), device=dev))
    b = nn.Parameter(torch.zeros(len(item_keys), device=dev))
    subj_t = torch.as_tensor(subj_ids, dtype=torch.long, device=dev)
    item_t = torch.as_tensor(item_ids, dtype=torch.long, device=dev)
    label_t = torch.as_tensor(labels, dtype=torch.float32, device=dev)
    optimizer = torch.optim.Adam(
        [theta, log_a, b], lr=0.05, weight_decay=theta_weight_decay
    )
    bce = nn.BCEWithLogitsLoss()
    n = subj_t.shape[0]
    for epoch in range(factor_epochs):
        order = torch.randperm(n, device=dev)
        total = 0.0
        n_batches = 0
        for start in range(0, n, factor_batch):
            idx = order[start : start + factor_batch]
            logit = torch.exp(log_a[item_t[idx]]) * theta[subj_t[idx]] - b[item_t[idx]]
            loss = bce(logit, label_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            n_batches += 1
        print(
            f"  factor epoch {epoch + 1}/{factor_epochs}  "
            f"loss={total / max(n_batches, 1):.5f}",
            flush=True,
        )

    theta_np = theta.detach().cpu().numpy().astype(np.float32)
    a_np = torch.exp(log_a).detach().cpu().numpy().astype(np.float32)
    b_np = b.detach().cpu().numpy().astype(np.float32)

    print(f"[encode] loading encoder {encoder_id}...", flush=True)
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(encoder_id, device=device)
    encoder.max_seq_length = MAX_SEQ_LENGTH

    all_items = (
        pd.concat(
            [
                train_df[["item_content", "benchmark", "condition"]],
                cal_df[["item_content", "benchmark", "condition"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates("item_content")
        .reset_index(drop=True)
    )
    all_items["item_content"] = (
        all_items["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    )
    texts = [format_item_text(r) for r in all_items.to_dict("records")]
    print(f"[encode] encoding {len(texts):,} unique items...", flush=True)
    embeds = np.asarray(
        encoder.encode(
            texts,
            batch_size=encode_batch,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    print(f"[encode] embeddings shape: {embeds.shape}", flush=True)
    item_text_to_row = {
        str(it): i for i, it in enumerate(all_items["item_content"].astype(str).values)
    }

    in_dim = embeds.shape[1]
    embed_by_factor_idx = np.zeros((len(item_keys), in_dim), dtype=np.float32)
    for key, idx in item_to_idx.items():
        row = item_text_to_row.get(key)
        if row is not None:
            embed_by_factor_idx[idx] = embeds[row]

    print(
        f"[mlp] training MLP (hidden={mlp_hidden}, layers={mlp_hidden_layers}, "
        f"dropout={mlp_dropout}, epochs={mlp_epochs})",
        flush=True,
    )

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list[nn.Module] = []
            prev = in_dim
            for _ in range(max(1, mlp_hidden_layers)):
                layers.append(nn.Linear(prev, mlp_hidden))
                layers.append(nn.GELU())
                if mlp_dropout > 0:
                    layers.append(nn.Dropout(mlp_dropout))
                prev = mlp_hidden
            layers.append(nn.Linear(prev, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    mlp = MLP().to(dev)
    x = torch.as_tensor(embed_by_factor_idx, dtype=torch.float32, device=dev)
    targets = torch.stack(
        [
            torch.as_tensor(np.log(np.clip(a_np, 1e-3, None)), dtype=torch.float32, device=dev),
            torch.as_tensor(b_np, dtype=torch.float32, device=dev),
        ],
        dim=1,
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=mlp_lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(mlp_epochs):
        mlp.train()
        opt.zero_grad()
        preds = mlp(x)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt.step()
        if (epoch + 1) % 25 == 0:
            print(
                f"  mlp epoch {epoch + 1}/{mlp_epochs}  loss={float(loss):.5f}",
                flush=True,
            )
    mlp.eval()

    cal_df["subject_name"] = cal_df["subject_content"].astype(str).map(parse_subject_name)
    cal_df["item_key"] = cal_df["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    mean_embed = embeds.mean(axis=0)
    cal_embeds = np.stack(
        [
            embeds[item_text_to_row[k]] if k in item_text_to_row else mean_embed
            for k in cal_df["item_key"].astype(str).values
        ],
        axis=0,
    )
    with torch.no_grad():
        out = mlp(torch.as_tensor(cal_embeds, dtype=torch.float32, device=dev)).cpu().numpy()
    log_a_cal = out[:, 0]
    b_cal = out[:, 1]
    a_cal = np.exp(np.clip(log_a_cal, -8, 8))
    global_theta = float(theta_np.mean())
    theta_cal = np.array(
        [
            float(theta_np[subject_to_idx[n]]) if n in subject_to_idx else global_theta
            for n in cal_df["subject_name"].values
        ],
        dtype=np.float32,
    )
    logits_cal = a_cal * theta_cal - b_cal
    y_cal = cal_df["label"].to_numpy(dtype=np.float32)

    eps = 1e-4
    grid = np.concatenate([np.linspace(0.15, 1.0, 35), np.linspace(1.05, 3.0, 40)])
    if not np.any(np.isclose(grid, 1.0)):
        grid = np.append(grid, 1.0)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        p = np.clip(1.0 / (1.0 + np.exp(-logits_cal / T)), eps, 1.0 - eps)
        nll = float(-np.mean(y_cal * np.log(p) + (1.0 - y_cal) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)

    p1 = np.clip(1.0 / (1.0 + np.exp(-logits_cal)), eps, 1.0 - eps)
    nll_T1 = float(-np.mean(y_cal * np.log(p1) + (1.0 - y_cal) * np.log(1.0 - p1)))
    print(
        f"[T*] cold-start NLL(T=1)={nll_T1:.5f} -> NLL(T*={best_T:.3f})={best_nll:.5f}",
        flush=True,
    )

    savez_kwargs: dict = dict(
        encoder_id=np.array(encoder_id),
        subject_names=np.array(subject_names),
        subject_theta=theta_np,
        global_theta=np.array(global_theta, dtype=np.float32),
        global_mean=np.array(smoothed["global_mean"], dtype=np.float32),
        mlp_hidden=np.array(mlp_hidden, dtype=np.int32),
        temperature=np.array(best_T, dtype=np.float32),
    )
    for i, mod in enumerate(mlp.net):
        if isinstance(mod, nn.Linear):
            savez_kwargs[f"mlp_w{i}"] = mod.weight.detach().cpu().numpy().astype(np.float32)
            savez_kwargs[f"mlp_b{i}"] = mod.bias.detach().cpu().numpy().astype(np.float32)

    buf = io.BytesIO()
    np.savez(buf, **savez_kwargs)
    artifact_bytes = buf.getvalue()
    print(f"[artifact] factor_pge.npz size: {len(artifact_bytes):,} bytes", flush=True)

    label = run_label or encoder_id.replace("/", "__")
    out_dir = Path("/cache") / "runs" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "factor_pge.npz").write_bytes(artifact_bytes)
    (out_dir / "smoothed_prior.json").write_text(json.dumps(smoothed))
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "encoder_id": encoder_id,
                "max_rows": max_rows,
                "n_train_rows": int(len(train_df)),
                "n_holdout_rows": int(len(cal_df)),
                "holdout_benchmarks": holdout,
                "n_subjects": len(subject_names),
                "n_items": len(item_keys),
                "temperature": best_T,
                "nll_T1": nll_T1,
                "nll_best": best_nll,
                "mlp_hidden": mlp_hidden,
                "mlp_hidden_layers": mlp_hidden_layers,
                "mlp_dropout": mlp_dropout,
                "factor_epochs": factor_epochs,
                "mlp_epochs": mlp_epochs,
            },
            indent=2,
        )
    )
    vol.commit()
    print(f"[artifact] persisted to volume at {out_dir}", flush=True)

    return artifact_bytes, smoothed


# --- Modal function: eval-only (apples-to-apples comparison) ----------------

@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": vol},
    timeout=30 * 60,
)
def eval_factor_on_holdout(
    artifact_bytes: bytes,
    encoder_id: str,
    holdout_benchmarks: list[str],
    label: str,
    max_rows: int = 200_000,
    encode_batch: int = 256,
    seed: int = 321,
) -> dict:
    """Score a trained factor artifact on a fixed whole-benchmark holdout.

    Returns NLL at the artifact's stored T as well as the optimal T fit on
    *this* holdout (so we can see how much calibration is doing).
    """
    import os

    import numpy as np
    import pandas as pd
    import torch

    cache_dir = Path("/cache")
    (cache_dir / "hf_cache").mkdir(exist_ok=True)
    (cache_dir / "st_cache").mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "hf_cache")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir / "st_cache")

    df = build_runtime_examples(cache_dir)
    hold = df[df["benchmark"].isin(holdout_benchmarks)].copy()
    if max_rows and len(hold) > max_rows:
        hold = hold.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    print(
        f"[eval:{label}] holdout benchmarks={holdout_benchmarks}  "
        f"rows={len(hold):,}  items={hold['item_content'].nunique():,}",
        flush=True,
    )

    artifact = np.load(io.BytesIO(artifact_bytes), allow_pickle=False)
    stored_T = float(artifact["temperature"]) if "temperature" in artifact.files else 1.0
    subject_theta = {
        str(n): float(v)
        for n, v in zip(
            artifact["subject_names"].tolist(),
            artifact["subject_theta"].tolist(),
        )
    }
    global_theta = float(artifact["global_theta"])
    layer_idx = sorted(
        int(k[len("mlp_w"):]) for k in artifact.files if k.startswith("mlp_w")
    )
    layers = [
        (
            np.asarray(artifact[f"mlp_w{i}"], dtype=np.float32),
            np.asarray(artifact[f"mlp_b{i}"], dtype=np.float32),
        )
        for i in layer_idx
    ]

    hold["item_key"] = hold["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    hold["subject_name"] = hold["subject_content"].astype(str).map(parse_subject_name)
    unique = hold.drop_duplicates("item_key").reset_index(drop=True)
    texts = [format_item_text(r) for r in unique.to_dict("records")]

    from sentence_transformers import SentenceTransformer

    print(f"[eval:{label}] loading encoder {encoder_id}", flush=True)
    encoder = SentenceTransformer(encoder_id, device="cuda")
    encoder.max_seq_length = MAX_SEQ_LENGTH
    print(f"[eval:{label}] encoding {len(texts):,} unique items", flush=True)
    embeds = np.asarray(
        encoder.encode(
            texts,
            batch_size=encode_batch,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )

    h = embeds
    for i, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if i < len(layers) - 1:
            h = 0.5 * h * (
                1.0
                + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h**3))
            )
    log_a = h[:, 0]
    b_arr = h[:, 1]
    a_arr = np.exp(np.clip(log_a, -8, 8))

    key_to_idx = {k: i for i, k in enumerate(unique["item_key"].astype(str).tolist())}
    rows = np.array(
        [key_to_idx[k] for k in hold["item_key"].astype(str).tolist()], dtype=np.int64
    )
    a_pred = a_arr[rows]
    b_pred = b_arr[rows]
    theta_arr = np.array(
        [
            subject_theta.get(n, global_theta)
            for n in hold["subject_name"].astype(str).tolist()
        ],
        dtype=np.float32,
    )
    logits = a_pred * theta_arr - b_pred
    y = hold["label"].to_numpy(dtype=np.float32)

    eps = 1e-4
    grid = np.concatenate([np.linspace(0.15, 1.0, 35), np.linspace(1.05, 5.0, 80)])
    if not np.any(np.isclose(grid, 1.0)):
        grid = np.append(grid, 1.0)

    def nll_at(T: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-logits / max(T, 1e-3))), eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    nll_stored = nll_at(stored_T)
    nll_T1 = nll_at(1.0)
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        score = nll_at(float(T))
        if score < best_nll:
            best_nll = score
            best_T = float(T)
    print(
        f"[eval:{label}] T_stored={stored_T:.3f}  NLL(stored)={nll_stored:.5f}  "
        f"NLL(T=1)={nll_T1:.5f}  T*={best_T:.3f}  NLL(T*)={best_nll:.5f}",
        flush=True,
    )
    return {
        "label": label,
        "encoder_id": encoder_id,
        "holdout_benchmarks": holdout_benchmarks,
        "n_rows": int(len(hold)),
        "n_items": int(hold["item_content"].nunique()),
        "T_stored": stored_T,
        "nll_T_stored": nll_stored,
        "nll_T1": nll_T1,
        "T_best": best_T,
        "nll_T_best": best_nll,
    }


@app.local_entrypoint()
def compare(
    holdout: str = "matharena,mmlupro,rewardbench,swebench",
    v3_path: str = "projects/predictive_eval_challenge/archive/factor_pge_before_bge_large.npz",
    bge_path: str = "projects/predictive_eval_challenge/archive/factor_pge_bge_large_20260519T1953.npz",
):
    """Apples-to-apples compare two artifacts on the same cold-start holdout."""
    holdout_list = [b.strip() for b in holdout.split(",") if b.strip()]
    print(f"[compare] holdout: {holdout_list}", flush=True)

    v3_bytes = Path(v3_path).read_bytes()
    bge_bytes = Path(bge_path).read_bytes()
    print(f"[compare] v3 artifact:  {len(v3_bytes):,} bytes -- {v3_path}", flush=True)
    print(f"[compare] bge artifact: {len(bge_bytes):,} bytes -- {bge_path}", flush=True)

    v3 = eval_factor_on_holdout.remote(
        v3_bytes, "sentence-transformers/all-MiniLM-L6-v2", holdout_list, "v3-minilm"
    )
    bge = eval_factor_on_holdout.remote(
        bge_bytes, "BAAI/bge-large-en-v1.5", holdout_list, "bge-large"
    )

    print("\n========== APPLES-TO-APPLES COMPARISON ==========")
    print(json.dumps({"v3": v3, "bge": bge}, indent=2))
    winner = "v3-minilm" if v3["nll_T_best"] < bge["nll_T_best"] else "bge-large"
    print(f"\nWinner on NLL(T*): {winner}")
    print(
        f"  v3:  {v3['nll_T_best']:.5f}  (T*={v3['T_best']:.3f})\n"
        f"  bge: {bge['nll_T_best']:.5f}  (T*={bge['T_best']:.3f})"
    )


# --- Local entrypoint --------------------------------------------------------

@app.local_entrypoint()
def main(
    encoder_id: str = "BAAI/bge-large-en-v1.5",
    max_rows: int = 4_000_000,
    seed: int = 321,
    factor_epochs: int = 8,
    mlp_hidden: int = 256,
    mlp_hidden_layers: int = 2,
    mlp_dropout: float = 0.15,
    mlp_epochs: int = 400,
    n_holdout_benchmarks: int = 4,
    out_label: str = "bge_large",
):
    """Train on Modal, then drop the artifact into both submission folders."""
    print(
        f"[modal] kicking off training: encoder={encoder_id}, max_rows={max_rows:,}",
        flush=True,
    )
    artifact_bytes, smoothed = train_factor_bge.remote(
        encoder_id=encoder_id,
        max_rows=max_rows,
        seed=seed,
        factor_epochs=factor_epochs,
        mlp_hidden=mlp_hidden,
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_dropout=mlp_dropout,
        mlp_epochs=mlp_epochs,
        n_holdout_benchmarks=n_holdout_benchmarks,
    )

    archive_dir = LOCAL_PROJECT / "archive"
    archive_dir.mkdir(exist_ok=True)
    old = LOCAL_FACTOR_DIR / "factor_pge.npz"
    if old.exists():
        backup = archive_dir / f"factor_pge_before_{out_label}.npz"
        backup.write_bytes(old.read_bytes())
        print(f"[archive] {old} -> {backup}", flush=True)

    for d in (LOCAL_FACTOR_DIR, LOCAL_ENSEMBLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
        (d / "factor_pge.npz").write_bytes(artifact_bytes)
        print(f"[write] {d / 'factor_pge.npz'}  ({len(artifact_bytes):,} bytes)", flush=True)

    (LOCAL_ENSEMBLE_DIR / "smoothed_prior.json").write_text(json.dumps(smoothed))
    print(f"[write] {LOCAL_ENSEMBLE_DIR / 'smoothed_prior.json'}", flush=True)

    print(
        "\n[done] new artifacts in place. Next, rebuild the ZIPs:\n"
        "  /opt/anaconda3/envs/torch_measure/bin/python "
        "projects/predictive_eval_challenge/make_submission.py factor_pge --out-dir dist\n"
        "  /opt/anaconda3/envs/torch_measure/bin/python "
        "projects/predictive_eval_challenge/make_submission.py "
        "factor_baseline_ensemble --out-dir dist",
        flush=True,
    )


# --- Modal: multi-seed MiniLM training for logit-space ensembling -----------


@app.local_entrypoint()
def train_direct_residual(
    encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_rows: int = 2_500_000,
    seed: int = 41,
    hidden: int = 256,
    hidden_layers: int = 2,
    dropout: float = 0.10,
    epochs: int = 8,
    batch_size: int = 8192,
    lr: float = 2e-4,
    encode_batch: int = 256,
    out_label: str = "direct_residual_minilm_seed41",
):
    """Train a direct probability residual model on Modal.

    The model predicts:
        logit(p) = logit(prior_v1_v2_blend) + residual_MLP(item_embed, prior_features)

    This keeps the empirical prior as the anchor and only asks text embeddings
    to learn residual difficulty/ranking signal.
    """
    artifact_bytes, prior = train_direct_residual_remote.remote(
        encoder_id=encoder_id,
        max_rows=max_rows,
        seed=seed,
        hidden=hidden,
        hidden_layers=hidden_layers,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        encode_batch=encode_batch,
        run_label=out_label,
    )
    out_dir = LOCAL_PROJECT / "ensemble_artifacts" / out_label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "direct_residual.npz").write_bytes(artifact_bytes)
    (out_dir / "smoothed_prior.json").write_text(json.dumps(prior))
    (out_dir / "smoothed_prior_v2.json").write_text(json.dumps(prior))
    print(f"[write] {out_dir / 'direct_residual.npz'}  ({len(artifact_bytes):,} bytes)")
    print(f"[write] {out_dir / 'smoothed_prior_v2.json'}")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/cache": vol},
    timeout=3 * 60 * 60,
)
def train_direct_residual_remote(
    encoder_id: str,
    max_rows: int,
    seed: int,
    hidden: int,
    hidden_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    encode_batch: int,
    run_label: str,
) -> tuple[bytes, dict]:
    import io
    import os

    import numpy as np
    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = Path("/cache")
    (cache_dir / "hf_cache").mkdir(exist_ok=True)
    (cache_dir / "st_cache").mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "hf_cache")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir / "st_cache")

    df = build_runtime_examples(cache_dir)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_cal = max(1, int(len(df) * 0.05))
    cal_df = df.iloc[:n_cal].copy()
    train_df = df.iloc[n_cal:].copy()
    print(
        f"[direct] rows train={len(train_df):,} cal={len(cal_df):,} "
        f"subjects={df['subject_content'].nunique():,}",
        flush=True,
    )

    prior = fit_rich_smoothed_prior(train_df.to_dict("records"))
    g = float(prior["global_mean"])
    print(f"[direct] prior global_mean={g:.4f}", flush=True)

    def clip(p):
        return np.clip(p, 1e-4, 1.0 - 1e-4)

    def logit(p):
        p = clip(p)
        return np.log(p / (1.0 - p))

    def prior_features(frame):
        out = np.empty((len(frame), 10), dtype=np.float32)
        subj = frame["subject_content"].astype(str).map(parse_subject_name).values
        bench = frame["benchmark"].astype(str).values
        cond = frame["condition"].fillna("none").astype(str).replace("", "none").values
        for i, (s, bmk, cnd) in enumerate(zip(subj, bench, cond, strict=False)):
            s_val = prior.get("subject", {}).get(s, g)
            b_val = prior.get("benchmark", {}).get(bmk, g)
            bc_val = prior.get("benchmark_condition", {}).get(_key(bmk, cnd), g)
            sb_val = prior.get("subject_benchmark", {}).get(_key(s, bmk), g)
            sbc_raw = prior.get("subject_benchmark_condition", {}).get(_key(s, bmk, cnd))
            sbc_val = sbc_raw if sbc_raw is not None else sb_val
            p1 = 0.85 * (
                0.35 * s_val + 0.15 * b_val + 0.25 * bc_val + 0.25 * sb_val
            ) + 0.15 * g
            p2 = 0.15 * p1 + 0.85 * (0.95 * sbc_val + 0.05 * g)
            base = 0.8 * p1 + 0.2 * p2
            vals = [p1, p2, base, s_val, b_val, bc_val, sb_val, sbc_val]
            out[i, :8] = clip(np.array(vals, dtype=np.float32))
            out[i, 8] = 1.0 if sbc_raw is not None else 0.0
            out[i, 9] = float(len(str(cnd)) > 0 and str(cnd) != "none")
        out[:, :8] = logit(out[:, :8])
        return out

    all_items = (
        df[["item_content", "benchmark", "condition"]]
        .drop_duplicates("item_content")
        .reset_index(drop=True)
    )
    all_items["item_content"] = all_items["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS)
    texts = [format_item_text(r) for r in all_items.to_dict("records")]
    from sentence_transformers import SentenceTransformer

    print(f"[direct] encoding {len(texts):,} unique items with {encoder_id}", flush=True)
    encoder = SentenceTransformer(encoder_id, device=device)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    embeds = np.asarray(
        encoder.encode(
            texts,
            batch_size=encode_batch,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    item_to_row = {str(k): i for i, k in enumerate(all_items["item_content"].astype(str).values)}

    def make_features(frame):
        keys = frame["item_content"].astype(str).str.slice(0, MAX_ITEM_CHARS).values
        rows = np.array([item_to_row[str(k)] for k in keys], dtype=np.int64)
        return np.concatenate([embeds[rows], prior_features(frame)], axis=1).astype(np.float32)

    x_train = make_features(train_df)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    x_cal = make_features(cal_df)
    y_cal = cal_df["label"].to_numpy(dtype=np.float32)
    base_logit_train = x_train[:, embeds.shape[1] + 2]
    base_logit_cal = x_cal[:, embeds.shape[1] + 2]

    mean = x_train.mean(axis=0).astype(np.float32)
    scale = x_train.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    x_train = (x_train - mean) / scale
    x_cal = (x_cal - mean) / scale

    class ResidualMLP(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            layers: list[nn.Module] = []
            prev = in_dim
            for _ in range(max(1, hidden_layers)):
                layers.append(nn.Linear(prev, hidden))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = hidden
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    torch.manual_seed(seed)
    model = ResidualMLP(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    base_t = torch.as_tensor(base_logit_train, dtype=torch.float32, device=device)
    n = x_t.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(n, device=device)
        total = 0.0
        nb = 0
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            residual = 0.5 * model(x_t[idx])
            logits = base_t[idx] + residual
            loss = loss_fn(logits, y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        print(f"  direct epoch {epoch + 1}/{epochs} loss={total / max(nb, 1):.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        residual_cal = 0.5 * model(torch.as_tensor(x_cal, dtype=torch.float32, device=device)).cpu().numpy()
    logits_cal = base_logit_cal + residual_cal
    eps = 1e-4
    grid = np.concatenate([np.linspace(0.2, 1.0, 33), np.linspace(1.05, 3.0, 40)])
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        p = np.clip(1.0 / (1.0 + np.exp(-logits_cal / T)), eps, 1.0 - eps)
        nll = float(-np.mean(y_cal * np.log(p) + (1.0 - y_cal) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    p_base = np.clip(1.0 / (1.0 + np.exp(-base_logit_cal)), eps, 1.0 - eps)
    nll_base = float(-np.mean(y_cal * np.log(p_base) + (1.0 - y_cal) * np.log(1.0 - p_base)))
    print(f"[direct] cal NLL base={nll_base:.5f} residual T*={best_T:.3f} nll={best_nll:.5f}", flush=True)

    save = {
        "encoder_id": np.array(encoder_id),
        "feature_mean": mean,
        "feature_scale": scale,
        "temperature": np.array(best_T, dtype=np.float32),
        "residual_scale": np.array(0.5, dtype=np.float32),
        "global_mean": np.array(g, dtype=np.float32),
    }
    for i, mod in enumerate(model.net):
        if isinstance(mod, nn.Linear):
            save[f"mlp_w{i}"] = mod.weight.detach().cpu().numpy().astype(np.float32)
            save[f"mlp_b{i}"] = mod.bias.detach().cpu().numpy().astype(np.float32)
    buf = io.BytesIO()
    np.savez(buf, **save)
    artifact_bytes = buf.getvalue()
    out_dir = Path("/cache") / "runs" / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "direct_residual.npz").write_bytes(artifact_bytes)
    (out_dir / "smoothed_prior_v2.json").write_text(json.dumps(prior))
    vol.commit()
    return artifact_bytes, prior


@app.local_entrypoint()
def train_single_keep(
    encoder_id: str = "sentence-transformers/all-mpnet-base-v2",
    max_rows: int = 2_500_000,
    seed: int = 11,
    factor_epochs: int = 8,
    mlp_hidden: int = 384,
    mlp_hidden_layers: int = 2,
    mlp_dropout: float = 0.10,
    mlp_epochs: int = 450,
    holdout_mode: str = "random_rows",
    calibration_frac: float = 0.05,
    out_label: str = "mpnet_factor_seed11",
):
    """Train one factor artifact on Modal and save it without overwriting dist.

    This is the safer experiment entrypoint: it writes to
    `ensemble_artifacts/<out_label>/` and leaves all existing Codabench
    submission artifacts untouched until we explicitly choose to package it.
    """
    print(
        f"[single-keep] encoder={encoder_id}  seed={seed}  "
        f"max_rows={max_rows:,}  out_label={out_label}",
        flush=True,
    )
    artifact_bytes, smoothed = train_factor_bge.remote(
        encoder_id=encoder_id,
        max_rows=max_rows,
        seed=seed,
        factor_epochs=factor_epochs,
        mlp_hidden=mlp_hidden,
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_dropout=mlp_dropout,
        mlp_epochs=mlp_epochs,
        holdout_mode=holdout_mode,
        calibration_frac=calibration_frac,
        run_label=out_label,
    )

    out_dir = LOCAL_PROJECT / "ensemble_artifacts" / out_label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "factor_pge.npz").write_bytes(artifact_bytes)
    (out_dir / "smoothed_prior.json").write_text(json.dumps(smoothed))
    print(f"[write] {out_dir / 'factor_pge.npz'}  ({len(artifact_bytes):,} bytes)")
    print(f"[write] {out_dir / 'smoothed_prior.json'}")


@app.local_entrypoint()
def train_multi_seed(
    encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    seeds: str = "0,1,2,3,4",
    max_rows: int = 2_500_000,
    factor_epochs: int = 8,
    mlp_hidden: int = 256,
    mlp_hidden_layers: int = 2,
    mlp_dropout: float = 0.15,
    mlp_epochs: int = 400,
    holdout_mode: str = "random_rows",
    calibration_frac: float = 0.05,
    out_label: str = "minilm_multiseed_v2",
    out_subdir: str | None = None,
):
    """Train N seeds of the MiniLM factor model with random-row T calibration.

    This matches the v3 local training recipe: fit the factor model on ~95%
    of all rows so every subject contributes thetas, then calibrate T on a
    random 5% in-sample holdout. The previous benchmark-holdout mode dropped
    ~40% of subjects from training, which is why those seeds underperformed
    the local v3 artifact (NLL 0.676 vs 0.625 on the same cold-start eval).
    Per-seed diversity comes from different (seed) factor-init and random
    in-sample holdouts.
    """
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    print(
        f"[multi-seed] encoder={encoder_id}  seeds={seed_list}  "
        f"max_rows={max_rows:,}  holdout_mode={holdout_mode}  "
        f"calibration_frac={calibration_frac}",
        flush=True,
    )

    kwargs_list = [
        dict(
            encoder_id=encoder_id,
            max_rows=max_rows,
            seed=s,
            factor_epochs=factor_epochs,
            mlp_hidden=mlp_hidden,
            mlp_hidden_layers=mlp_hidden_layers,
            mlp_dropout=mlp_dropout,
            mlp_epochs=mlp_epochs,
            holdout_mode=holdout_mode,
            calibration_frac=calibration_frac,
            run_label=f"{out_label}_seed{s}",
        )
        for s in seed_list
    ]
    # Spawn all seeds in parallel; each call lands in its own A100 container
    # (Modal will queue if we exceed our concurrency quota). Then gather.
    calls = [train_factor_bge.spawn(**kw) for kw in kwargs_list]
    print(
        f"[multi-seed] spawned {len(calls)} parallel training jobs",
        flush=True,
    )
    results = [c.get() for c in calls]

    out_dir = LOCAL_PROJECT / "ensemble_artifacts" / (out_subdir or out_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoothed_to_save = None
    for seed, (artifact_bytes, smoothed) in zip(seed_list, results):
        path = out_dir / f"factor_pge_seed{seed}.npz"
        path.write_bytes(artifact_bytes)
        print(f"[write] {path}  ({len(artifact_bytes):,} bytes)", flush=True)
        if smoothed_to_save is None:
            smoothed_to_save = smoothed
    if smoothed_to_save is not None:
        (out_dir / "smoothed_prior.json").write_text(json.dumps(smoothed_to_save))
        print(f"[write] {out_dir / 'smoothed_prior.json'}", flush=True)

    print(
        f"\n[done] {len(seed_list)} seeds saved to {out_dir}\n"
        f"Next: build the multi-seed submission with the new model.py.",
        flush=True,
    )
