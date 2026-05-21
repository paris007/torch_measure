# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Data loading for the CS321M Predictive Evaluation Challenge.

The public dataset is a Hugging Face parquet collection, not one homogeneous
`datasets` split. This module follows the starter-kit pattern: response parquet
files are listed explicitly, registry tables are loaded separately, and trace
tables are excluded.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
DEFAULT_MAX_ITEM_CHARS = 800


def item_variant_key(row: dict | pd.Series, max_chars: int = DEFAULT_MAX_ITEM_CHARS) -> str:
    """Return the condition-specific cold-start item key used by the challenge.

    The public response tables expose `item_variant_id`, but Codabench runtime
    rows only provide benchmark, condition, and item text. This key mirrors the
    hidden-item semantics closely enough for local factor/CV training: the same
    text under different benchmark-condition contexts is treated as a different
    item variant.
    """
    benchmark = str(row.get("benchmark", ""))
    condition = str(row.get("condition", "none") or "none")
    item = str(row.get("item_content", ""))[:max_chars]
    return f"{benchmark}||{condition}||{item}"


def list_response_files(repo_id: str = REPO_ID) -> list[str]:
    """List response parquet files while excluding registry and trace tables."""
    from huggingface_hub import HfApi

    repo_files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(
        name
        for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )


def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    """Render subject metadata in the same text style as Codabench inputs."""
    display_name = subject.get("display_name") or fallback_subject_id
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


def load_public_tables(
    response_files: Iterable[str] | None = None,
    repo_id: str = REPO_ID,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load response and registry tables from the public dataset."""
    from datasets import Features, Value, load_dataset

    if response_files is None:
        response_files = list_response_files(repo_id)

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
        repo_id,
        data_files=list(response_files),
        features=response_features,
        split="train",
    ).to_pandas()
    items = load_dataset(repo_id, data_files="items.parquet", split="train").to_pandas()
    subjects = load_dataset(repo_id, data_files="subjects.parquet", split="train").to_pandas()
    benchmarks = load_dataset(repo_id, data_files="benchmarks.parquet", split="train").to_pandas()
    return responses, items, subjects, benchmarks


def build_runtime_examples(
    response_files: Iterable[str] | None = None,
    repo_id: str = REPO_ID,
) -> pd.DataFrame:
    """Join public tables into the four-field runtime input shape.

    The returned frame includes `item_variant_id` because the competition treats
    the same upstream item under different normalized conditions as different
    hidden item variants.
    """
    responses, items, subjects, benchmarks = load_public_tables(response_files, repo_id)

    item_cols = [col for col in ["item_id", "content"] if col in items.columns]
    subject_cols = [
        col
        for col in ["subject_id", "display_name", "provider", "params", "release_date", "family"]
        if col in subjects.columns
    ]
    benchmark_cols = [col for col in ["benchmark_id", "name"] if col in benchmarks.columns]

    df = responses.merge(items[item_cols], on="item_id", how="left")
    df = df.merge(subjects[subject_cols], on="subject_id", how="left")
    df = df.merge(benchmarks[benchmark_cols], on="benchmark_id", how="left", suffixes=("", "_meta"))

    subject_records = df[subject_cols].to_dict("records")
    df["subject_content"] = [
        render_subject_content(record, fallback)
        for record, fallback in zip(subject_records, df["subject_id"].astype(str), strict=False)
    ]
    df["benchmark"] = df["benchmark_id"].astype(str)
    df["condition"] = df["test_condition"].fillna("none").replace("", "none").astype(str)
    df["item_content"] = df["content"].fillna("").astype(str)
    df["label"] = df["response"].astype(float)
    df["item_variant_id"] = (
        df["benchmark_id"].astype(str) + "||" + df["item_id"].astype(str) + "||" + df["condition"]
    )

    # The competition objective is binary correctness.
    df = df[df["label"].isin([0.0, 1.0])].copy()
    keep = [
        "benchmark",
        "condition",
        "subject_content",
        "item_content",
        "label",
        "subject_id",
        "item_id",
        "item_variant_id",
        "benchmark_id",
    ]
    return df[keep].reset_index(drop=True)


def save_runtime_examples(path: str | Path, response_files: Iterable[str] | None = None) -> Path:
    """Save joined runtime examples to parquet for offline training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = build_runtime_examples(response_files=response_files)
    examples.to_parquet(path, index=False)
    return path
