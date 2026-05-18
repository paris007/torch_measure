#!/bin/bash
#SBATCH --job-name=predictive-eval-embedding
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=projects/predictive_eval_challenge/slurm/embedding-%j.out
#SBATCH --error=projects/predictive_eval_challenge/slurm/embedding-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/envs/torch_measure/bin/python}"
DATA_PATH="projects/predictive_eval_challenge/data/runtime_examples.parquet"
EMBEDDING_DIR="projects/predictive_eval_challenge/codabench_submissions/embedding"

"$PYTHON" -m pip install -e .

if [ ! -f "$DATA_PATH" ]; then
    "$PYTHON" projects/predictive_eval_challenge/download_data.py --out "$DATA_PATH"
fi

"$PYTHON" projects/predictive_eval_challenge/train_embedding.py \
    --data "$DATA_PATH" \
    --batch-size 256 \
    --max-rows 1000000

"$PYTHON" projects/predictive_eval_challenge/smoke_submission.py "$EMBEDDING_DIR"
"$PYTHON" projects/predictive_eval_challenge/make_submission.py embedding
