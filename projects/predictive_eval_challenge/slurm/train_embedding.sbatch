#!/bin/bash
#SBATCH --job-name=predictive-eval-embedding
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/embedding-%j.out
#SBATCH --error=slurm/embedding-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="${PYTHON:-/opt/anaconda3/envs/torch_measure/bin/python}"
DATA_PATH="data/runtime_examples.parquet"
EMBEDDING_DIR="codabench_submissions/embedding"

"$PYTHON" -m pip install -e .

if [ ! -f "$DATA_PATH" ]; then
    "$PYTHON" download_data.py --out "$DATA_PATH"
fi

"$PYTHON" train_embedding.py \
    --data "$DATA_PATH" \
    --batch-size 256 \
    --max-rows 1000000

"$PYTHON" smoke_submission.py "$EMBEDDING_DIR"
"$PYTHON" make_submission.py embedding
