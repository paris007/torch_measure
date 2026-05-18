# Copyright (c) 2026 AIMS Foundations. MIT License.

"""Download and join the public measurement-db training data."""

from __future__ import annotations

import argparse

from data_loading import save_runtime_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="projects/predictive_eval_challenge/data/runtime_examples.parquet",
        help="Output parquet path for joined runtime-shaped examples.",
    )
    args = parser.parse_args()

    output_path = save_runtime_examples(args.out)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
