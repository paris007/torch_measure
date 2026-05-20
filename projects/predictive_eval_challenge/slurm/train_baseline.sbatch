#!/bin/bash
#SBATCH --job-name=predictive-eval-baseline
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/baseline-%j.out
#SBATCH --error=slurm/baseline-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/envs/torch_measure/bin/python}"
DATA_PATH="data/runtime_examples.parquet"
BASELINE_DIR="codabench_submissions/baseline"

"$PYTHON" -m pip install -e .

"$PYTHON" download_data.py --out "$DATA_PATH"
"$PYTHON" fit_richer_baseline.py --data "$DATA_PATH" \
    --out "$BASELINE_DIR/artifacts/smoothed_prior.json"
"$PYTHON" smoke_submission.py "$BASELINE_DIR"
"$PYTHON" make_submission.py baseline
