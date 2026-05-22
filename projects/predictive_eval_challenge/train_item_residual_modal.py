#!/usr/bin/env python3
"""Modal trainer for a residualized item-difficulty model.

Run from repo root:
  modal run projects/predictive_eval_challenge/train_item_residual_modal.py

Writes:
  projects/predictive_eval_challenge/codabench_submissions/item_residual_modal/artifacts/
    smoothed_prior.json
    item_residual_model.npz
"""
from __future__ import annotations

import io, json, math, re, zipfile
from pathlib import Path
import modal

APP_NAME = "cs321m-item-residual-train"
DATASET_ID = "aims-foundations/measurement-db"
DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"
PROJECT_REL = Path("projects/predictive_eval_challenge")
OUT_REL = PROJECT_REL / "codabench_submissions/item_residual_modal/artifacts"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.19.0",
        "sentence-transformers>=3.0.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "pyarrow>=15.0.0",
        "torch>=2.2.0",
    )
)
app = modal.App(APP_NAME, image=image)


def _safe_str(x) -> str:
    return "" if x is None else str(x)


def _parse_subject_name(subject_content: str) -> str:
    m = re.search(r"^Name:\s*(.+)$", subject_content or "", flags=re.MULTILINE)
    return (m.group(1) if m else (subject_content or "")).strip().lower()


def _key(*parts) -> str:
    return "||".join(str(p) for p in parts)


def _clip(p: float, eps: float = 1e-4) -> float:
    if not math.isfinite(float(p)):
        return 0.5
    return float(min(1.0 - eps, max(eps, float(p))))


def _choose_column(cols, options):
    for c in options:
        if c in cols:
            return c
    lower = {str(c).lower(): c for c in cols}
    for opt in options:
        if opt.lower() in lower:
            return lower[opt.lower()]
    raise KeyError(f"None of {options} found in columns: {list(cols)[:40]}")


def _normalize_dataframe(df):
    import pandas as pd
    cols = set(df.columns)
    label_col = _choose_column(cols, ["label", "response", "correct", "score", "passed"])
    item_col = _choose_column(cols, ["item_content", "item_description", "item", "prompt", "question"])
    subj_col = _choose_column(cols, ["subject_content", "subject_description", "model_content", "model_id", "subject"])
    bench_col = _choose_column(cols, ["benchmark", "benchmark_name", "dataset", "eval_name"])
    cond_col = None
    for c in ["condition", "test_condition", "setting", "prompting"]:
        if c in cols:
            cond_col = c
            break

    out = pd.DataFrame()
    out["label"] = df[label_col].astype(float)
    out["item_content"] = df[item_col].map(_safe_str)
    out["subject_content"] = df[subj_col].map(_safe_str)
    out["benchmark"] = df[bench_col].map(_safe_str)
    out["condition"] = df[cond_col].map(_safe_str) if cond_col else "none"
    out.loc[out["condition"].isin(["", "nan", "None", "NaN"]), "condition"] = "none"
    out["subject_name"] = out["subject_content"].map(_parse_subject_name)
    out = out[out["label"].isin([0.0, 1.0])]
    return out.reset_index(drop=True)


def _smooth(sum_y, n, parent, strength):
    return float((sum_y + strength * parent) / (n + strength))


def _fit_prior(df, strength: float = 25.0) -> dict:
    g = float(df["label"].mean())
    prior = {"global_mean": g, "strength": float(strength)}

    subject = {}
    for s, part in df.groupby("subject_name"):
        subject[s] = _smooth(float(part["label"].sum()), len(part), g, strength)

    benchmark = {}
    for b, part in df.groupby("benchmark"):
        benchmark[b] = _smooth(float(part["label"].sum()), len(part), g, strength)

    bc = {}
    for (b, c), part in df.groupby(["benchmark", "condition"]):
        bc[_key(b, c)] = _smooth(float(part["label"].sum()), len(part), benchmark.get(b, g), strength)

    sb = {}
    for (s, b), part in df.groupby(["subject_name", "benchmark"]):
        parent = 0.65 * subject.get(s, g) + 0.35 * benchmark.get(b, g)
        sb[_key(s, b)] = _smooth(float(part["label"].sum()), len(part), parent, strength)

    sbc = {}
    for (s, b, c), part in df.groupby(["subject_name", "benchmark", "condition"]):
        parent = sb.get(_key(s, b), bc.get(_key(b, c), g))
        sbc[_key(s, b, c)] = _smooth(float(part["label"].sum()), len(part), parent, strength)

    prior.update(
        subject=subject,
        benchmark=benchmark,
        benchmark_condition=bc,
        subject_benchmark=sb,
        subject_benchmark_condition=sbc,
    )
    return prior


def _base_predict(row, prior: dict) -> float:
    gm = float(prior.get("global_mean", 0.5))
    s = row["subject_name"] if "subject_name" in row else _parse_subject_name(_safe_str(row.get("subject_content", "")))
    b = _safe_str(row.get("benchmark", ""))
    c = _safe_str(row.get("condition", "none") or "none")

    sbc = prior.get("subject_benchmark_condition", {}).get(_key(s, b, c))
    if sbc is not None:
        return _clip(0.95 * float(sbc) + 0.05 * gm)
    sb = prior.get("subject_benchmark", {}).get(_key(s, b))
    if sb is not None:
        return _clip(0.92 * float(sb) + 0.08 * gm)

    bc = prior.get("benchmark_condition", {}).get(_key(b, c))
    ps = prior.get("subject", {}).get(s)
    pb = prior.get("benchmark", {}).get(b)
    vals = [(bc, 0.45), (ps, 0.40), (pb, 0.15)]
    avail = [(v, w) for v, w in vals if v is not None]
    if not avail:
        return _clip(gm)
    tw = sum(w for _, w in avail)
    p = sum(float(v) * w for v, w in avail) / tw
    return _clip(0.85 * p + 0.15 * gm)


def _format_item_text(row, max_chars=1200) -> str:
    return f"Benchmark: {row['benchmark']}\nCondition: {row['condition'] or 'none'}\nItem: {str(row['item_content'])[:max_chars]}"


def _render_subject_content(subject: dict, fallback: str) -> str:
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


def _list_response_files() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo_id=DATASET_ID, repo_type="dataset")
    return sorted(
        name
        for name in files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )


def _load_joined_runtime_df():
    import pandas as pd
    from datasets import Features, Value, load_dataset

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
        DATASET_ID,
        data_files=_list_response_files(),
        features=response_features,
        split="train",
    ).to_pandas()
    items = load_dataset(DATASET_ID, data_files="items.parquet", split="train").to_pandas()
    subjects = load_dataset(DATASET_ID, data_files="subjects.parquet", split="train").to_pandas()

    item_cols = [col for col in ["item_id", "content"] if col in items.columns]
    subject_cols = [
        col
        for col in ["subject_id", "display_name", "provider", "params", "release_date", "family"]
        if col in subjects.columns
    ]
    df = responses.merge(items[item_cols], on="item_id", how="left")
    df = df.merge(subjects[subject_cols], on="subject_id", how="left")
    df["subject_content"] = [
        _render_subject_content(record, fallback)
        for record, fallback in zip(
            df[subject_cols].to_dict("records"),
            df["subject_id"].astype(str),
            strict=False,
        )
    ]
    df["benchmark"] = df["benchmark_id"].astype(str)
    df["condition"] = df["test_condition"].fillna("none").replace("", "none").astype(str)
    df["item_content"] = df["content"].fillna("").astype(str)
    df["label"] = df["response"].astype(float)
    df["subject_name"] = df["subject_content"].map(_parse_subject_name)
    df = df[df["label"].isin([0.0, 1.0])].copy()
    return df[["label", "item_content", "subject_content", "subject_name", "benchmark", "condition"]].reset_index(drop=True)


@app.function(gpu="A10G", timeout=6 * 60 * 60, memory=32768)
def train_remote(encoder_id: str = DEFAULT_ENCODER, ridge_alpha: float = 50.0, max_rows: int | None = None) -> bytes:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import Ridge

    df = _load_joined_runtime_df()
    if max_rows:
        df = df.sample(min(max_rows, len(df)), random_state=321).reset_index(drop=True)

    print(f"Loaded {len(df):,} public rows", flush=True)
    prior = _fit_prior(df)

    df["p0"] = [_base_predict(row, prior) for row in df.to_dict("records")]
    df["info"] = df["p0"] * (1.0 - df["p0"])
    df["resid"] = df["label"] - df["p0"]
    df["item_key"] = df["benchmark"].astype(str) + "||" + df["condition"].astype(str) + "||" + df["item_content"].astype(str).str.slice(0, 1200)

    rows = []
    lam = 6.0
    for item_key, part in df.groupby("item_key"):
        info = float(part["info"].sum())
        if len(part) < 2 or info <= 0.0:
            continue
        delta = float(part["resid"].sum() / (lam + info))
        delta = float(np.clip(delta, -1.5, 1.5))
        first = part.iloc[0]
        rows.append({
            "benchmark": first["benchmark"],
            "condition": first["condition"],
            "item_content": first["item_content"],
            "delta": delta,
            "weight": float(lam + info),
        })

    item_df = pd.DataFrame(rows)
    print(f"Training residual model on {len(item_df):,} item variants", flush=True)

    encoder = SentenceTransformer(encoder_id)
    encoder.max_seq_length = 256
    texts = [_format_item_text(r) for r in item_df.to_dict("records")]
    X = encoder.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    y = item_df["delta"].to_numpy(dtype=np.float32)
    w = item_df["weight"].to_numpy(dtype=np.float32)

    x_mean = X.mean(axis=0).astype(np.float32)
    x_std = (X.std(axis=0) + 1e-6).astype(np.float32)
    Xz = (X - x_mean) / x_std

    model = Ridge(alpha=float(ridge_alpha), fit_intercept=True, random_state=321)
    model.fit(Xz, y, sample_weight=w)
    coef = model.coef_.astype(np.float32)
    intercept = np.array([float(model.intercept_)], dtype=np.float32)
    pred = model.predict(Xz)
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 else 0.0
    print(f"Train item-delta corr={corr:.4f}; coef_norm={float(np.linalg.norm(coef)):.4f}", flush=True)

    buf_npz = io.BytesIO()
    np.savez_compressed(
        buf_npz,
        coef=coef,
        intercept=intercept,
        x_mean=x_mean,
        x_std=x_std,
        encoder_id=np.array(encoder_id),
        ridge_alpha=np.array([ridge_alpha], dtype=np.float32),
        train_item_corr=np.array([corr], dtype=np.float32),
        delta_clip=np.array([1.5], dtype=np.float32),
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("smoothed_prior.json", json.dumps(prior))
        zf.writestr("item_residual_model.npz", buf_npz.getvalue())
        zf.writestr("README_item_residual.txt", f"encoder={encoder_id}\nridge_alpha={ridge_alpha}\nrows={len(df)}\nitems={len(item_df)}\ntrain_corr={corr}\n")
    return zip_buf.getvalue()


@app.local_entrypoint()
def main(encoder_id: str = DEFAULT_ENCODER, ridge_alpha: float = 50.0, max_rows: int = 0):
    data = train_remote.remote(encoder_id=encoder_id, ridge_alpha=ridge_alpha, max_rows=None if max_rows <= 0 else max_rows)
    out_dir = Path.cwd() / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(out_dir)
    print(f"Wrote artifacts to {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(" -", p)
