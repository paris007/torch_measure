#!/bin/bash
#SBATCH --job-name=predictive-eval-factor
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/factor-%j.out
#SBATCH --error=slurm/factor-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/envs/torch_measure/bin/python}"
DATA_PATH="data/runtime_examples.parquet"
SUBMISSION_DIR="codabench_submissions/factor_pge"

"$PYTHON" -m pip install -e .

if [ ! -f "$DATA_PATH" ]; then
    "$PYTHON" download_data.py --out "$DATA_PATH"
fi

"$PYTHON" train_factor.py \
    --data "$DATA_PATH" \
    --batch-size 256 \
    --max-rows 1000000 \
    --factor-epochs 8 \
    --mlp-hidden 256 \
    --mlp-epochs 400

"$PYTHON" recalibrate_temperature.py \
    --artifact "$SUBMISSION_DIR/artifacts/factor_pge.npz" \
    --data "$DATA_PATH"

"$PYTHON" smoke_submission.py "$SUBMISSION_DIR"
"$PYTHON" make_submission.py factor_pge
