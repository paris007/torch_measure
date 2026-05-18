# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Train the smoothed-prior baseline for the predictive evaluation challenge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from torch_measure.models import SmoothedPriorPredictiveEvaluator


DEFAULT_ARTIFACT_PATH = (
    "projects/predictive_eval_challenge/"
    "codabench_submissions/baseline/artifacts/smoothed_prior.json"
)


def train_baseline(data_path: str | Path, output_path: str | Path, strength: float = 25.0) -> Path:
    """Fit and save the smoothed-prior evaluator."""
    df = pd.read_parquet(data_path)
    evaluator = SmoothedPriorPredictiveEvaluator.fit(df.to_dict("records"), strength=strength)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(evaluator.to_dict(), indent=2, sort_keys=True))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Joined runtime examples parquet.")
    parser.add_argument("--out", default=DEFAULT_ARTIFACT_PATH, help="Where to write the JSON artifact.")
    parser.add_argument("--strength", type=float, default=25.0, help="Prior strength for smoothed group means.")
    args = parser.parse_args()

    output_path = train_baseline(args.data, args.out, strength=args.strength)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
