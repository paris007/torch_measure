#!/bin/bash
#SBATCH --job-name=predictive-eval-baseline
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=projects/predictive_eval_challenge/slurm/baseline-%j.out
#SBATCH --error=projects/predictive_eval_challenge/slurm/baseline-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/envs/torch_measure/bin/python}"
DATA_PATH="projects/predictive_eval_challenge/data/runtime_examples.parquet"
BASELINE_DIR="projects/predictive_eval_challenge/codabench_submissions/baseline"

"$PYTHON" -m pip install -e .

"$PYTHON" projects/predictive_eval_challenge/download_data.py --out "$DATA_PATH"
"$PYTHON" projects/predictive_eval_challenge/train_baseline.py --data "$DATA_PATH"
"$PYTHON" projects/predictive_eval_challenge/smoke_submission.py "$BASELINE_DIR"
"$PYTHON" projects/predictive_eval_challenge/make_submission.py baseline
